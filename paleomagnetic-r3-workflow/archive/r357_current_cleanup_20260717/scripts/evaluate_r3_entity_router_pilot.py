#!/usr/bin/env python3
"""Evaluate frozen R3 entity-router features on the unchanged V4 folds.

Every entity-router artifact, candidate, selection, file reference, and
feature-column declaration is validated before the R3 label source is opened.
The saved V4 OOF predictions are opened only after all RF/HGB fits and are used
solely to audit reproduction of the unchanged V4 random-forest baseline.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_trace_page_graph_features as freeze  # noqa: E402
import evaluate_r3_layout_pilot_grouped_oof as layout_eval  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
ENTITY_DIR = ROOT / "tmp/r3_entity_router_pilot_v1"
DEFAULT_ENTITY_FEATURES = ENTITY_DIR / "r3_entity_router_features_v1.npz"
DEFAULT_ENTITY_MANIFEST = ENTITY_DIR / "r3_entity_router_features_v1.manifest.json"
DEFAULT_ENTITY_CANDIDATE = ENTITY_DIR / "r3_entity_router_candidate_freeze_v1.json"
DEFAULT_ENTITY_AUDIT = ENTITY_DIR / "r3_entity_router_audit_v1.json"
DEFAULT_EVIDENCE_GRAPH = ENTITY_DIR / "r3_entity_router_evidence_graph_v1.jsonl"
DEFAULT_PILOT_SELECTION = ROOT / "tmp/r3_layout_pilot_v1/r3_layout_pilot_manifest_v1.json"
DEFAULT_V4_OOF = V4_DIR / "trace_clean_nested_oof_v4.npz"
DEFAULT_V4_REPORT = V4_DIR / "trace_clean_nested_oof_report_v4.json"
DEFAULT_OUTPUT_DIR = ENTITY_DIR

OOF_NAME = "r3_entity_router_grouped_oof_v1.npz"
REPORT_NAME = "r3_entity_router_grouped_oof_report_v1.json"
RECEIPT_NAME = "r3_entity_router_grouped_oof_receipt_v1.json"
EXPECTED_PILOT_ROWS = 222
THRESHOLD = 0.5
IDENTITY_FIELDS = ("REFNO", "RESULTNO", "ROCKNAME")
ABLATION_GROUPS = ("flags_only", "layout_only", "entity_only", "full")
ATOMIC_ABLATION_GROUPS = ABLATION_GROUPS[:3]
FAMILIES = ("random_forest", "hgb")
STRATEGIES = (
    "base",
    "flags_only",
    "layout_only",
    "entity_only",
    "full",
    "pilot_routed_full",
)
MODEL_NAMES = tuple(
    f"{strategy}_{family}" for family in FAMILIES for strategy in STRATEGIES
)
BOUNDARY_FIELDS = (
    "labels_read",
    "pmagresult_or_rockmag_read",
    "prior_predictions_read",
    "evaluation_results_read",
    "outer_test_metrics_used_for_selection",
)
FORBIDDEN_FEATURE_RE = re.compile(
    r"(?:^|_)(?:R3|R5|R7|PMAGRESULT|ROCKMAG|PREDICTIONS?|OOF|"
    r"EVALUATION|METRICS?)(?:$|_)",
    re.I,
)


class EntityRouterEvaluationError(ValueError):
    """Raised when the frozen entity-router evaluation boundary is invalid."""


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _nested(payload: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = payload
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _first(payload: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = _nested(payload, path)
        if value is not None:
            return value
    return None


def _require_sha256(value: Any, description: str) -> str:
    digest = clean(value)
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise EntityRouterEvaluationError(f"{description} is not a SHA-256 digest")
    return digest


def _validate_label_blind_boundary(
    payload: Mapping[str, Any], description: str
) -> None:
    boundary = payload.get("runtime_boundary")
    if not isinstance(boundary, Mapping):
        raise EntityRouterEvaluationError(f"{description} lacks runtime_boundary")
    for field in BOUNDARY_FIELDS:
        if boundary.get(field) is not False:
            raise EntityRouterEvaluationError(
                f"{description} boundary must declare {field}=false"
            )


def _resolve_declared_path(value: Any, manifest_path: Path) -> Path:
    declared = Path(clean(value)).expanduser()
    if not declared.is_absolute():
        declared = manifest_path.parent / declared
    return declared.resolve()


def _iter_path_references(value: Any, prefix: str = "") -> Iterable[tuple[str, Mapping[str, Any]]]:
    if isinstance(value, Mapping):
        if "path" in value:
            yield prefix or "reference", value
        for key, nested in value.items():
            if key == "path":
                continue
            child = f"{prefix}.{key}" if prefix else clean(key)
            yield from _iter_path_references(nested, child)
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _iter_path_references(nested, f"{prefix}[{index}]")


def _validate_file_reference(
    reference: Mapping[str, Any],
    *,
    manifest_path: Path,
    description: str,
    expected_path: Path | None = None,
) -> Path:
    if not clean(reference.get("path")):
        raise EntityRouterEvaluationError(f"{description} lacks path")
    path = _resolve_declared_path(reference["path"], manifest_path)
    if expected_path is not None and path != expected_path.resolve():
        raise EntityRouterEvaluationError(
            f"{description} path mismatch: declared {path}, expected {expected_path.resolve()}"
        )
    if not path.is_file():
        raise EntityRouterEvaluationError(f"{description} file does not exist: {path}")
    expected_sha = _require_sha256(reference.get("sha256"), f"{description} hash")
    if freeze.file_sha256(path) != expected_sha:
        raise EntityRouterEvaluationError(f"{description} file hash mismatch")
    return path


def _find_reference(
    manifest: Mapping[str, Any],
    *,
    section: str,
    aliases: Sequence[str],
    manifest_path: Path,
    expected_path: Path,
    description: str,
) -> Mapping[str, Any]:
    container = manifest.get(section)
    if not isinstance(container, Mapping):
        raise EntityRouterEvaluationError(f"entity manifest lacks {section}")
    matches = [container[name] for name in aliases if name in container]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise EntityRouterEvaluationError(
            f"entity manifest must declare exactly one {description} reference"
        )
    _validate_file_reference(
        matches[0],
        manifest_path=manifest_path,
        expected_path=expected_path,
        description=description,
    )
    return matches[0]


def _validate_all_manifest_file_references(
    manifest: Mapping[str, Any], manifest_path: Path
) -> None:
    for section in ("inputs", "outputs"):
        payload = manifest.get(section)
        if not isinstance(payload, Mapping):
            raise EntityRouterEvaluationError(f"entity manifest lacks {section}")
        for description, reference in _iter_path_references(payload, section):
            _validate_file_reference(
                reference,
                manifest_path=manifest_path,
                description=f"entity manifest {description}",
            )


def _validate_selection_manifest(
    path: Path,
    records: Sequence[Mapping[str, str]],
    expected_rows: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        return layout_eval._validate_selection_manifest(  # noqa: SLF001
            path, records, expected_rows=expected_rows
        )
    except Exception as exc:
        if isinstance(exc, EntityRouterEvaluationError):
            raise
        raise EntityRouterEvaluationError(
            f"invalid frozen pilot selection: {exc}"
        ) from exc


def _integer(value: Any, description: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise EntityRouterEvaluationError(f"{description} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise EntityRouterEvaluationError(f"{description} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise EntityRouterEvaluationError(f"{description} must be an integer")
    if positive and parsed <= 0:
        raise EntityRouterEvaluationError(f"{description} must be positive")
    return parsed


def _finite_float(value: Any, description: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise EntityRouterEvaluationError(f"{description} must be numeric") from exc
    if not np.isfinite(parsed):
        raise EntityRouterEvaluationError(f"{description} must be finite")
    return parsed


def _model_config(
    candidate: Mapping[str, Any], name: str, required: Sequence[str]
) -> dict[str, Any]:
    models = candidate.get("model_candidates")
    if not isinstance(models, Mapping) or not isinstance(models.get(name), Mapping):
        raise EntityRouterEvaluationError(
            f"entity candidate lacks fixed {name} model config"
        )
    config = dict(models[name])
    missing = [field for field in required if field not in config]
    if missing:
        raise EntityRouterEvaluationError(
            f"entity candidate {name} config lacks: {', '.join(missing)}"
        )
    return config


def _validate_fixed_model_configs(
    random_forest: dict[str, Any], hgb: dict[str, Any]
) -> None:
    rf_allowed = {
        "n_estimators",
        "max_features",
        "min_samples_leaf",
        "n_jobs",
        "random_state",
    }
    hgb_allowed = {
        "max_iter",
        "learning_rate",
        "max_leaf_nodes",
        "min_samples_leaf",
        "l2_regularization",
        "random_state",
    }
    if set(random_forest) - rf_allowed:
        raise EntityRouterEvaluationError(
            "entity random_forest config contains unsupported fields: "
            + ", ".join(sorted(set(random_forest) - rf_allowed))
        )
    if set(hgb) - hgb_allowed:
        raise EntityRouterEvaluationError(
            "entity hgb config contains unsupported fields: "
            + ", ".join(sorted(set(hgb) - hgb_allowed))
        )
    for field in ("n_estimators", "min_samples_leaf"):
        random_forest[field] = _integer(
            random_forest[field], f"random_forest.{field}", positive=True
        )
    random_forest["n_jobs"] = _integer(
        random_forest.get("n_jobs", -1), "random_forest.n_jobs"
    )
    if random_forest["n_jobs"] == 0:
        raise EntityRouterEvaluationError("random_forest.n_jobs cannot be zero")
    random_forest["random_state"] = _integer(
        random_forest["random_state"], "random_forest.random_state"
    )
    max_features = random_forest["max_features"]
    if isinstance(max_features, str):
        if max_features not in {"sqrt", "log2"}:
            raise EntityRouterEvaluationError(
                "random_forest.max_features string must be sqrt or log2"
            )
    elif isinstance(max_features, int) and not isinstance(max_features, bool):
        random_forest["max_features"] = _integer(
            max_features, "random_forest.max_features", positive=True
        )
    elif not isinstance(max_features, bool):
        parsed = _finite_float(max_features, "random_forest.max_features")
        if not 0.0 < parsed <= 1.0:
            raise EntityRouterEvaluationError(
                "numeric random_forest.max_features must be in (0, 1]"
            )
        random_forest["max_features"] = parsed
    else:
        raise EntityRouterEvaluationError(
            "random_forest.max_features cannot be boolean"
        )

    for field in ("max_iter", "max_leaf_nodes", "min_samples_leaf"):
        hgb[field] = _integer(hgb[field], f"hgb.{field}", positive=True)
    if hgb["max_leaf_nodes"] < 2:
        raise EntityRouterEvaluationError("hgb.max_leaf_nodes must be at least two")
    hgb["learning_rate"] = _finite_float(hgb["learning_rate"], "hgb.learning_rate")
    if hgb["learning_rate"] <= 0.0:
        raise EntityRouterEvaluationError("hgb.learning_rate must be positive")
    hgb["l2_regularization"] = _finite_float(
        hgb["l2_regularization"], "hgb.l2_regularization"
    )
    if hgb["l2_regularization"] < 0.0:
        raise EntityRouterEvaluationError("hgb.l2_regularization cannot be negative")
    hgb["random_state"] = _integer(hgb["random_state"], "hgb.random_state")


def _column_list(value: Any, group: str, width: int | None) -> list[int]:
    if not isinstance(value, list):
        raise EntityRouterEvaluationError(
            f"entity ablation {group} must declare columns as a list"
        )
    columns = [
        _integer(item, f"entity ablation {group} column") for item in value
    ]
    if not columns:
        raise EntityRouterEvaluationError(
            f"entity ablation {group} columns cannot be empty"
        )
    if any(column < 0 for column in columns):
        raise EntityRouterEvaluationError(
            f"entity ablation {group} columns must be non-negative"
        )
    if len(columns) != len(set(columns)):
        raise EntityRouterEvaluationError(
            f"entity ablation {group} columns contain duplicates"
        )
    if width is not None and any(column >= width for column in columns):
        raise EntityRouterEvaluationError(
            f"entity ablation {group} column is out of range for width {width}"
        )
    return columns


def _normalize_ablation_groups(
    payload: Mapping[str, Any], *, width: int | None, description: str
) -> dict[str, list[int]]:
    raw = payload.get("feature_ablation_candidates")
    if raw is None:
        raw = payload.get("feature_groups")
    if not isinstance(raw, Mapping) or set(raw) != set(ABLATION_GROUPS):
        raise EntityRouterEvaluationError(
            f"{description} must declare exactly {', '.join(ABLATION_GROUPS)}"
        )
    output: dict[str, list[int]] = {}
    for group in ABLATION_GROUPS:
        declaration = raw[group]
        if not isinstance(declaration, Mapping) or "columns" not in declaration:
            raise EntityRouterEvaluationError(
                f"{description} ablation {group} lacks explicit columns"
            )
        output[group] = _column_list(declaration["columns"], group, width)
    used: set[int] = set()
    for group in ATOMIC_ABLATION_GROUPS:
        overlap = used.intersection(output[group])
        if overlap:
            raise EntityRouterEvaluationError(
                f"entity atomic ablation groups overlap at columns {sorted(overlap)}"
            )
        used.update(output[group])
    concatenated = [
        column for group in ATOMIC_ABLATION_GROUPS for column in output[group]
    ]
    if output["full"] != concatenated:
        raise EntityRouterEvaluationError(
            "entity full columns must equal flags_only + layout_only + entity_only in order"
        )
    if width is not None and set(output["full"]) != set(range(width)):
        raise EntityRouterEvaluationError(
            "entity full ablation must cover every frozen matrix column exactly once"
        )
    return output


def normalize_entity_candidate(
    candidate: Mapping[str, Any], *, width: int | None = None
) -> dict[str, Any]:
    if clean(candidate.get("schema_version")) != "r3_entity_router_candidate_freeze_v1":
        raise EntityRouterEvaluationError("entity candidate has an invalid schema_version")
    if candidate.get("status") != "FROZEN_BEFORE_LABEL_ACCESS":
        raise EntityRouterEvaluationError("entity candidate was not frozen before labels")
    _validate_label_blind_boundary(candidate, "entity candidate")
    threshold = candidate.get("fixed_decision_threshold")
    if threshold is None or _finite_float(threshold, "entity threshold") != THRESHOLD:
        raise EntityRouterEvaluationError("entity candidate threshold must be fixed at 0.5")
    random_forest = _model_config(
        candidate,
        "random_forest",
        ("n_estimators", "max_features", "min_samples_leaf", "random_state"),
    )
    hgb = _model_config(
        candidate,
        "hgb",
        (
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
            "random_state",
        ),
    )
    _validate_fixed_model_configs(random_forest, hgb)
    groups = _normalize_ablation_groups(
        candidate, width=width, description="entity candidate"
    )
    route = candidate.get("deployment_route")
    if not isinstance(route, Mapping) or (
        clean(route.get("entity_scope")) != "frozen_pilot_rows_only"
        or clean(route.get("nonpilot_rows")) != "base_v4_unchanged"
    ):
        raise EntityRouterEvaluationError(
            "entity candidate lacks the frozen pilot-only deployment route"
        )
    return {
        "schema_version": clean(candidate.get("schema_version")),
        "fixed_decision_threshold": THRESHOLD,
        "random_forest": random_forest,
        "hgb": hgb,
        "feature_ablation_candidates": groups,
        "deployment_route": {
            "entity_scope": "frozen_pilot_rows_only",
            "nonpilot_rows": "base_v4_unchanged",
        },
    }


def load_entity_inputs(
    *,
    records: Sequence[Mapping[str, str]],
    entity_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    audit_path: Path,
    evidence_graph_path: Path,
    selection_path: Path,
    expected_pilot_rows: int | None,
) -> tuple[
    np.ndarray,
    list[str],
    np.ndarray,
    dict[str, Any],
    dict[str, Any],
]:
    """Validate the complete label-blind entity freeze and return its matrix."""

    paths = {
        "features": entity_path.resolve(),
        "manifest": manifest_path.resolve(),
        "candidate": candidate_path.resolve(),
        "audit": audit_path.resolve(),
        "evidence_graph": evidence_graph_path.resolve(),
        "selection": selection_path.resolve(),
    }
    for description, path in paths.items():
        if not path.is_file():
            raise EntityRouterEvaluationError(
                f"entity {description} input does not exist: {path}"
            )
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    candidate = json.loads(paths["candidate"].read_text(encoding="utf-8"))
    if clean(manifest.get("schema_version")) != "r3_entity_router_features_v1":
        raise EntityRouterEvaluationError("entity manifest has an invalid schema_version")
    if manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS":
        raise EntityRouterEvaluationError("entity features were not frozen without labels")
    _validate_label_blind_boundary(manifest, "entity manifest")
    _validate_label_blind_boundary(candidate, "entity candidate")

    _find_reference(
        manifest,
        section="outputs",
        aliases=("features", "entity_features", "feature_matrix"),
        manifest_path=paths["manifest"],
        expected_path=paths["features"],
        description="features",
    )
    _find_reference(
        manifest,
        section="outputs",
        aliases=("candidate", "candidate_freeze"),
        manifest_path=paths["manifest"],
        expected_path=paths["candidate"],
        description="candidate",
    )
    _find_reference(
        manifest,
        section="outputs",
        aliases=("audit", "build_audit"),
        manifest_path=paths["manifest"],
        expected_path=paths["audit"],
        description="audit",
    )
    _find_reference(
        manifest,
        section="outputs",
        aliases=("evidence_graph", "graph", "assignments"),
        manifest_path=paths["manifest"],
        expected_path=paths["evidence_graph"],
        description="evidence_graph",
    )
    _find_reference(
        manifest,
        section="inputs",
        aliases=("selection", "selection_manifest", "pilot_selection"),
        manifest_path=paths["manifest"],
        expected_path=paths["selection"],
        description="selection",
    )
    _validate_all_manifest_file_references(manifest, paths["manifest"])

    alignment = freeze._record_alignment_sha256(records)
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        if clean(payload.get("alignment_sha256")) != alignment:
            raise EntityRouterEvaluationError(
                f"entity {description} does not align with the V4 identity"
            )

    pilot_mask, selection = _validate_selection_manifest(
        paths["selection"], records, expected_pilot_rows
    )
    selection_sha = clean(selection.get("selection_sha256"))
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        declared = _first(
            payload,
            (
                ("pilot_selection_sha256",),
                ("selection_sha256",),
                ("inputs", "selection", "selection_sha256"),
                ("inputs", "selection_manifest", "selection_sha256"),
                ("inputs", "pilot_selection", "selection_sha256"),
            ),
        )
        if clean(declared) != selection_sha:
            raise EntityRouterEvaluationError(
                f"entity {description} pilot selection hash mismatch"
            )

    with np.load(paths["features"]) as archive:
        if archive.files != ["entity"]:
            raise EntityRouterEvaluationError(
                "entity NPZ must contain exactly one array named entity"
            )
        frozen = np.asarray(archive["entity"])
    feature_names = list(manifest.get("feature_names") or [])
    if (
        frozen.ndim != 2
        or frozen.shape != (len(records), len(feature_names))
        or not feature_names
        or len(feature_names) != len(set(map(clean, feature_names)))
        or any(not clean(name) for name in feature_names)
        or not np.issubdtype(frozen.dtype, np.number)
        or not np.isfinite(frozen).all()
    ):
        raise EntityRouterEvaluationError("entity feature matrix or names are invalid")
    forbidden = [name for name in feature_names if FORBIDDEN_FEATURE_RE.search(clean(name))]
    if forbidden:
        raise EntityRouterEvaluationError(
            f"entity feature names contain forbidden historical fields: {forbidden[:5]}"
        )
    matrix_sha = freeze.array_sha256(frozen)
    names_sha = freeze.json_sha256(feature_names)
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        if clean(payload.get("matrix_sha256")) != matrix_sha:
            raise EntityRouterEvaluationError(
                f"entity {description} matrix hash mismatch"
            )
        if clean(payload.get("feature_names_sha256")) != names_sha:
            raise EntityRouterEvaluationError(
                f"entity {description} feature-name hash mismatch"
            )
    normalized_candidate = normalize_entity_candidate(
        candidate, width=frozen.shape[1]
    )
    manifest_groups = _normalize_ablation_groups(
        manifest, width=frozen.shape[1], description="entity manifest"
    )
    if manifest_groups != normalized_candidate["feature_ablation_candidates"]:
        raise EntityRouterEvaluationError(
            "entity candidate ablation columns differ from builder manifest"
        )
    if np.any(np.asarray(frozen[~pilot_mask], dtype=np.float64) != 0.0):
        raise EntityRouterEvaluationError(
            "entity features outside the frozen pilot selection must be zero"
        )
    validated_file_sha256 = {
        name: freeze.file_sha256(path) for name, path in paths.items()
    }
    return (
        frozen.astype(np.float64),
        list(map(clean, feature_names)),
        pilot_mask,
        normalized_candidate,
        {
            "manifest": manifest,
            "candidate": candidate,
            "selection": selection,
            "audit": json.loads(paths["audit"].read_text(encoding="utf-8")),
            "validated_paths": {name: str(path) for name, path in paths.items()},
            "validated_file_sha256": validated_file_sha256,
        },
    )


def assert_entity_freeze_unchanged(payloads: Mapping[str, Any]) -> None:
    """Fail closed if a frozen entity artifact changed after initial loading."""

    paths = payloads.get("validated_paths")
    hashes = payloads.get("validated_file_sha256")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise EntityRouterEvaluationError("entity freeze lacks the validated hash snapshot")
    if set(paths) != set(hashes):
        raise EntityRouterEvaluationError("entity freeze hash snapshot keys changed")
    for name, value in paths.items():
        path = Path(clean(value)).resolve()
        expected = _require_sha256(hashes[name], f"entity {name} snapshot hash")
        if not path.is_file() or freeze.file_sha256(path) != expected:
            raise EntityRouterEvaluationError(
                f"entity {name} changed after validation during grouped OOF"
            )


def _fit_random_forest(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    max_features: Any = config["max_features"]
    if not isinstance(max_features, str):
        max_features = float(max_features)
    model = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_features=max_features,
        min_samples_leaf=int(config["min_samples_leaf"]),
        n_jobs=int(config.get("n_jobs", -1)),
        random_state=int(config["random_state"]),
    )
    model.fit(train, labels, sample_weight=weights)
    return np.asarray(model.predict_proba(test)[:, 1], dtype=np.float64)


def _fit_hgb(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    model = HistGradientBoostingClassifier(
        max_iter=int(config["max_iter"]),
        learning_rate=float(config["learning_rate"]),
        max_leaf_nodes=int(config["max_leaf_nodes"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        l2_regularization=float(config["l2_regularization"]),
        random_state=int(config["random_state"]),
    )
    model.fit(train, labels, sample_weight=weights)
    return np.asarray(model.predict_proba(test)[:, 1], dtype=np.float64)


def run_grouped_oof(
    *,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    base_matrix: np.ndarray,
    entity_matrix: np.ndarray,
    pilot_mask: np.ndarray,
    fold_ids: np.ndarray,
    model_configs: Mapping[str, Any],
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=np.int8)
    base_matrix = np.asarray(base_matrix, dtype=np.float64)
    entity_matrix = np.asarray(entity_matrix, dtype=np.float64)
    pilot_mask = np.asarray(pilot_mask, dtype=bool)
    fold_ids = np.asarray(fold_ids, dtype=np.int8)
    if (
        labels.shape != (len(records),)
        or pilot_mask.shape != (len(records),)
        or fold_ids.shape != (len(records),)
        or base_matrix.ndim != 2
        or entity_matrix.ndim != 2
        or base_matrix.shape[0] != len(records)
        or entity_matrix.shape[0] != len(records)
        or not set(map(int, labels)).issubset({0, 1})
        or not np.isfinite(base_matrix).all()
        or not np.isfinite(entity_matrix).all()
    ):
        raise EntityRouterEvaluationError("grouped OOF inputs do not align")
    for refno in {clean(record.get("REFNO")) for record in records}:
        assigned = {
            int(fold_ids[index])
            for index, record in enumerate(records)
            if clean(record.get("REFNO")) == refno
        }
        if len(assigned) != 1:
            raise EntityRouterEvaluationError("grouped OOF split a REFNO")
    groups = model_configs.get("feature_ablation_candidates")
    if not isinstance(groups, Mapping):
        raise EntityRouterEvaluationError("grouped OOF lacks frozen ablation groups")
    matrices = {"base": base_matrix}
    for strategy in ABLATION_GROUPS:
        columns = groups[strategy]
        matrices[strategy] = np.concatenate(
            (base_matrix, entity_matrix[:, columns]), axis=1
        )

    probabilities = {
        name: np.full(len(records), np.nan, dtype=np.float64) for name in MODEL_NAMES
    }
    folds = sorted(set(map(int, fold_ids)))
    if not folds or folds != list(range(1, len(folds) + 1)):
        raise EntityRouterEvaluationError("grouped OOF fold ids are incomplete")
    fold_audit: list[dict[str, Any]] = []
    for fold in folds:
        train_indices = np.flatnonzero(fold_ids != fold)
        test_indices = np.flatnonzero(fold_ids == fold)
        train_labels = labels[train_indices]
        if set(map(int, train_labels)) != {0, 1}:
            raise EntityRouterEvaluationError(
                f"grouped OOF training fold {fold} lacks both R3 classes"
            )
        weights = trace.inverse_identity_weights(train_indices, records)
        for family in FAMILIES:
            fit = _fit_random_forest if family == "random_forest" else _fit_hgb
            for strategy in ("base",) + ABLATION_GROUPS:
                probabilities[f"{strategy}_{family}"][test_indices] = fit(
                    matrices[strategy][train_indices],
                    matrices[strategy][test_indices],
                    train_labels,
                    weights,
                    model_configs[family],
                )
            probabilities[f"pilot_routed_full_{family}"][test_indices] = np.where(
                pilot_mask[test_indices],
                probabilities[f"full_{family}"][test_indices],
                probabilities[f"base_{family}"][test_indices],
            )
        current = {
            "outer_fold": fold,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "test_papers": len(
                {clean(records[int(index)].get("REFNO")) for index in test_indices}
            ),
            "model_selection": "none_fixed_entity_router_candidates",
            "outer_test_metrics_used_for_selection": False,
            "metrics": {
                name: trace.metric(labels[test_indices], values[test_indices])
                for name, values in probabilities.items()
            },
        }
        fold_audit.append(current)
        if verbose:
            print(
                f"R3 entity router OOF: fold={fold}/{len(folds)} "
                f"base_rf={current['metrics']['base_random_forest']['accuracy']:.4f} "
                f"entity_rf={current['metrics']['entity_only_random_forest']['accuracy']:.4f} "
                f"full_rf={current['metrics']['full_random_forest']['accuracy']:.4f} "
                f"routed_rf={current['metrics']['pilot_routed_full_random_forest']['accuracy']:.4f}",
                flush=True,
            )
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("entity-router grouped OOF probabilities are incomplete")
    for family in FAMILIES:
        routed = probabilities[f"pilot_routed_full_{family}"]
        base = probabilities[f"base_{family}"]
        full = probabilities[f"full_{family}"]
        if not np.array_equal(routed[~pilot_mask], base[~pilot_mask]):
            raise AssertionError("pilot route changed a nonpilot probability")
        if not np.array_equal(routed[pilot_mask], full[pilot_mask]):
            raise AssertionError("pilot route differs from full on pilot rows")
    return (
        {name: values.astype(np.float32) for name, values in probabilities.items()},
        fold_audit,
    )


def evaluation_masks(
    records: Sequence[Mapping[str, str]], labels: np.ndarray, pilot_mask: np.ndarray
) -> dict[str, np.ndarray]:
    by_refno: dict[str, list[int]] = defaultdict(list)
    identities = [
        (
            clean(record.get("REFNO")),
            canonical_target_identity(clean(record.get("ROCKNAME"))),
        )
        for record in records
    ]
    identity_counts = Counter(identities)
    for index, record in enumerate(records):
        by_refno[clean(record.get("REFNO"))].append(index)
    mixed = {
        refno: len({int(labels[index]) for index in indices}) > 1
        for refno, indices in by_refno.items()
    }
    pilot = np.asarray(pilot_mask, dtype=bool)
    duplicate = np.asarray(
        [identity_counts[identity] > 1 for identity in identities], dtype=bool
    )
    mixed_paper = np.asarray(
        [mixed[clean(record.get("REFNO"))] for record in records], dtype=bool
    )
    return {
        "overall": np.ones(len(records), dtype=bool),
        "pilot_222": pilot,
        "pilot_identifiable_target": pilot & ~duplicate,
        "pilot_duplicate_identity": pilot & duplicate,
        "pilot_mixed_label_posthoc_only": pilot & mixed_paper,
    }


def paired_changes(
    labels: np.ndarray,
    base: np.ndarray,
    augmented: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected_labels = np.asarray(labels)[mask]
    base_prediction = np.asarray(base)[mask] >= THRESHOLD
    augmented_prediction = np.asarray(augmented)[mask] >= THRESHOLD
    base_correct = base_prediction == selected_labels
    augmented_correct = augmented_prediction == selected_labels
    repairs = int(np.sum(~base_correct & augmented_correct))
    degradations = int(np.sum(base_correct & ~augmented_correct))
    return {
        "rows": int(mask.sum()),
        "base_correct": int(base_correct.sum()),
        "augmented_correct": int(augmented_correct.sum()),
        "repairs_base_wrong_augmented_right": repairs,
        "degradations_base_right_augmented_wrong": degradations,
        "net_repairs": repairs - degradations,
        "changed_predictions": int(np.sum(base_prediction != augmented_prediction)),
    }


def ablation_paired_report(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for family in FAMILIES:
        comparisons = {
            f"base_vs_{strategy}": (
                f"base_{family}",
                f"{strategy}_{family}",
            )
            for strategy in ABLATION_GROUPS
        }
        comparisons["base_vs_pilot_routed_full"] = (
            f"base_{family}",
            f"pilot_routed_full_{family}",
        )
        comparisons.update(
            {
                "flags_only_vs_full": (
                    f"flags_only_{family}",
                    f"full_{family}",
                ),
                "layout_only_vs_full": (
                    f"layout_only_{family}",
                    f"full_{family}",
                ),
                "entity_only_vs_full": (
                    f"entity_only_{family}",
                    f"full_{family}",
                ),
            }
        )
        output[family] = {
            comparison: {
                subset: paired_changes(
                    labels,
                    probabilities[base_name],
                    probabilities[augmented_name],
                    mask,
                )
                for subset, mask in masks.items()
                if mask.any()
            }
            for comparison, (base_name, augmented_name) in comparisons.items()
        }
    return output


def routing_invariants(
    probabilities: Mapping[str, np.ndarray], pilot_mask: np.ndarray
) -> dict[str, Any]:
    pilot = np.asarray(pilot_mask, dtype=bool)
    output: dict[str, Any] = {}
    for family in FAMILIES:
        base = np.asarray(probabilities[f"base_{family}"], dtype=np.float32)
        full = np.asarray(probabilities[f"full_{family}"], dtype=np.float32)
        routed = np.asarray(
            probabilities[f"pilot_routed_full_{family}"], dtype=np.float32
        )
        if not np.array_equal(routed[~pilot], base[~pilot]):
            raise AssertionError("entity route changed nonpilot probabilities")
        if not np.array_equal(routed[pilot], full[pilot]):
            raise AssertionError("entity route differs from full on pilot rows")
        output[family] = {
            "pilot_rows": int(pilot.sum()),
            "nonpilot_rows": int((~pilot).sum()),
            "pilot_probabilities_equal_full": True,
            "nonpilot_probabilities_equal_base_bitwise": True,
            "maximum_nonpilot_absolute_difference_from_base": 0.0,
        }
    return output


def audit_saved_v4_random_forest(
    *,
    oof_path: Path,
    report_path: Path,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    reproduced: np.ndarray,
    require_match: bool = True,
) -> dict[str, Any]:
    """Open V4 OOF only after fitting and audit unchanged RF reproduction."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_oof_sha = freeze.file_sha256(oof_path)
    if clean((report.get("output") or {}).get("sha256")) != actual_oof_sha:
        raise EntityRouterEvaluationError("saved V4 OOF hash does not match its report")
    with np.load(oof_path) as archive:
        required = {"labels", "fold_ids", "random_forest"}
        if not required.issubset(archive.files):
            raise EntityRouterEvaluationError(
                "saved V4 OOF lacks random-forest audit arrays"
            )
        saved_labels = np.asarray(archive["labels"])
        saved_folds = np.asarray(archive["fold_ids"])
        saved_rf = np.asarray(archive["random_forest"], dtype=np.float32)
    reproduced = np.asarray(reproduced, dtype=np.float32)
    if not np.array_equal(saved_labels, labels) or not np.array_equal(
        saved_folds, fold_ids
    ):
        raise EntityRouterEvaluationError("saved V4 OOF identity arrays changed")
    if saved_rf.shape != reproduced.shape:
        raise EntityRouterEvaluationError("saved V4 RF shape changed")
    maximum_difference = float(
        np.max(np.abs(saved_rf.astype(float) - reproduced.astype(float)))
    )
    prediction_disagreements = int(
        np.sum((saved_rf >= THRESHOLD) != (reproduced >= THRESHOLD))
    )
    passed = maximum_difference <= 1e-7 and prediction_disagreements == 0
    audit = {
        "purpose": "post_fit_reproduction_audit_only",
        "loaded_after_all_entity_models_fitted": True,
        "used_as_model_feature": False,
        "used_for_candidate_selection": False,
        "oof": {"path": str(oof_path.resolve()), "sha256": actual_oof_sha},
        "report": {
            "path": str(report_path.resolve()),
            "sha256": freeze.file_sha256(report_path),
        },
        "array_key": "random_forest",
        "exact_probability_array_match": bool(np.array_equal(saved_rf, reproduced)),
        "maximum_absolute_probability_difference": maximum_difference,
        "prediction_disagreements_at_0_5": prediction_disagreements,
        "passed": passed,
    }
    if require_match and not passed:
        raise EntityRouterEvaluationError(
            "base random forest does not reproduce saved V4 OOF within tolerance"
        )
    return audit


def write_outputs(
    *,
    output_dir: Path,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    fold_ids: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    report: dict[str, Any],
) -> dict[str, Path]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / OOF_NAME
    report_path = output_dir / REPORT_NAME
    receipt_path = output_dir / RECEIPT_NAME
    stored = {
        name: np.asarray(probabilities[name], dtype=np.float32) for name in MODEL_NAMES
    }
    freeze.atomic_npz(
        oof_path,
        labels=np.asarray(labels, dtype=np.int8),
        fold_ids=np.asarray(fold_ids, dtype=np.int8),
        **stored,
    )
    report["output"] = {
        "path": str(oof_path),
        "sha256": freeze.file_sha256(oof_path),
    }
    freeze.atomic_json(report_path, report)
    receipt = {
        "schema_version": "r3_entity_router_grouped_oof_receipt_v1",
        "status": "POSTHOC_EVALUATION",
        "rows": [
            {
                "row_index": index,
                "REFNO": clean(record.get("REFNO")),
                "RESULTNO": clean(record.get("RESULTNO")),
                "fold": int(fold_ids[index]),
                "label": int(labels[index]),
                "probabilities": {
                    name: round(float(stored[name][index]), 8) for name in MODEL_NAMES
                },
                "predictions": {
                    name: int(stored[name][index] >= THRESHOLD)
                    for name in MODEL_NAMES
                },
            }
            for index, record in enumerate(records)
        ],
        "report": {
            "path": str(report_path),
            "sha256": freeze.file_sha256(report_path),
        },
        "label_source": report["label_source"],
    }
    freeze.atomic_json(receipt_path, receipt)
    return {"oof": oof_path, "report": report_path, "receipt": receipt_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace.DEFAULT_TARGETS)
    parser.add_argument("--v4-features", type=Path, default=trace.FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=trace.MANIFEST)
    parser.add_argument("--v4-target-context", type=Path, default=trace.CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=trace.CANDIDATE_FREEZE)
    parser.add_argument("--v4-fold-manifest", type=Path, default=trace.FOLD_MANIFEST)
    parser.add_argument("--entity-features", type=Path, default=DEFAULT_ENTITY_FEATURES)
    parser.add_argument("--entity-manifest", type=Path, default=DEFAULT_ENTITY_MANIFEST)
    parser.add_argument("--entity-candidate", type=Path, default=DEFAULT_ENTITY_CANDIDATE)
    parser.add_argument("--entity-audit", type=Path, default=DEFAULT_ENTITY_AUDIT)
    parser.add_argument("--evidence-graph", type=Path, default=DEFAULT_EVIDENCE_GRAPH)
    parser.add_argument("--pilot-selection", type=Path, default=DEFAULT_PILOT_SELECTION)
    parser.add_argument("--v4-oof", type=Path, default=DEFAULT_V4_OOF)
    parser.add_argument("--v4-report", type=Path, default=DEFAULT_V4_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    (
        records,
        base_matrix,
        _base_feature_names,
        _documents,
        fold_ids,
        document_groups,
        _v4_candidate,
    ) = trace.load_frozen_inputs(
        identity_path=args.identity.resolve(),
        pdf_dir=args.pdf_dir.resolve(),
        feature_path=args.v4_features.resolve(),
        manifest_path=args.v4_manifest.resolve(),
        context_path=args.v4_target_context.resolve(),
        candidate_path=args.v4_candidate.resolve(),
        fold_path=args.v4_fold_manifest.resolve(),
    )
    (
        entity_matrix,
        entity_feature_names,
        pilot_mask,
        model_configs,
        entity_payloads,
    ) = load_entity_inputs(
        records=records,
        entity_path=args.entity_features,
        manifest_path=args.entity_manifest,
        candidate_path=args.entity_candidate,
        audit_path=args.entity_audit,
        evidence_graph_path=args.evidence_graph,
        selection_path=args.pilot_selection,
        expected_pilot_rows=EXPECTED_PILOT_ROWS,
    )

    # First R3 access: every frozen feature, file, candidate, and group is valid.
    label_map = trace.load_labels(
        args.targets.resolve(), {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [label_map[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    probabilities, fold_audit = run_grouped_oof(
        records=records,
        labels=labels,
        base_matrix=base_matrix,
        entity_matrix=entity_matrix,
        pilot_mask=pilot_mask,
        fold_ids=fold_ids,
        model_configs=model_configs,
        verbose=True,
    )

    # Detect any concurrent builder rewrite before opening saved evaluation data.
    assert_entity_freeze_unchanged(entity_payloads)

    # Saved V4 predictions are deliberately opened only after every model fit.
    reproduction_audit = audit_saved_v4_random_forest(
        oof_path=args.v4_oof.resolve(),
        report_path=args.v4_report.resolve(),
        labels=labels,
        fold_ids=fold_ids,
        reproduced=probabilities["base_random_forest"],
        require_match=True,
    )
    masks = evaluation_masks(records, labels, pilot_mask)
    metrics = {
        name: {
            subset: trace.metric(labels[mask], values[mask])
            for subset, mask in masks.items()
            if mask.any()
        }
        for name, values in probabilities.items()
    }
    paired = ablation_paired_report(labels, probabilities, masks)
    route_audit = routing_invariants(probabilities, pilot_mask)
    report = {
        "schema_version": "r3_entity_router_grouped_oof_report_v1",
        "status": "POSTHOC_EVALUATION_OF_FROZEN_ENTITY_ROUTER",
        "rows": len(records),
        "papers": len({clean(record.get("REFNO")) for record in records}),
        "pilot_rows": int(pilot_mask.sum()),
        "threshold": THRESHOLD,
        "model_configs_from_entity_candidate_freeze": model_configs,
        "entity_feature_names": entity_feature_names,
        "metrics": metrics,
        "paired_changes": paired,
        "routing_invariants": route_audit,
        "fold_audit": fold_audit,
        "subset_definitions": {
            "overall": "all V4 PDF-backed rows",
            "pilot_222": "the frozen label-blind layout/entity pilot selection",
            "pilot_identifiable_target": (
                "pilot rows with a unique canonical ROCKNAME identity within REFNO"
            ),
            "pilot_duplicate_identity": (
                "pilot rows sharing canonical ROCKNAME with another RESULTNO in REFNO"
            ),
            "pilot_mixed_label_posthoc_only": (
                "posthoc only: pilot REFNO contains both R3 classes; never used for selection"
            ),
        },
        "subset_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "v4_random_forest_reproduction_audit": reproduction_audit,
        "boundary": {
            "entity_feature_generation_used_labels": False,
            "entity_candidate_freeze_used_labels": False,
            "base_features_are_frozen_v4_structured_matrix": True,
            "each_ablation_is_v4_plus_declared_entity_matrix_columns": True,
            "deployment_route_preserves_nonpilot_probabilities_bitwise": True,
            "v4_oof_or_predictions_used_as_features": False,
            "v4_saved_rf_loaded_only_after_all_model_fits": True,
            "r3_used_in_outer_training_folds_only": True,
            "outer_test_metrics_used_for_model_selection": False,
            "pmagresult_or_rockmag_used": False,
            "mixed_subset_used_for_selection": False,
            "outer_test_refno_seen_in_training": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": freeze.file_sha256(path.resolve()),
            }
            for name, path in (
                ("v4_features", args.v4_features),
                ("v4_manifest", args.v4_manifest),
                ("v4_target_context", args.v4_target_context),
                ("v4_candidate", args.v4_candidate),
                ("v4_fold_manifest", args.v4_fold_manifest),
                ("entity_features", args.entity_features),
                ("entity_manifest", args.entity_manifest),
                ("entity_candidate", args.entity_candidate),
                ("entity_audit", args.entity_audit),
                ("evidence_graph", args.evidence_graph),
                ("pilot_selection", args.pilot_selection),
            )
        }
        | {
            "evaluator": {
                "path": str(Path(__file__).resolve()),
                "sha256": freeze.file_sha256(Path(__file__).resolve()),
            }
        },
        "label_source": {
            "path": str(args.targets.resolve()),
            "sha256": freeze.file_sha256(args.targets.resolve()),
            "columns": ["RESULTNO", "R3"],
            "labels_array_sha256": freeze.array_sha256(labels),
            "used_only_after_all_feature_and_candidate_validation": True,
        },
        "entity_freeze": {
            "feature_schema": clean(
                entity_payloads["manifest"].get("schema_version")
            ),
            "candidate_schema": clean(
                entity_payloads["candidate"].get("schema_version")
            ),
            "selection_sha256": clean(
                entity_payloads["selection"].get("selection_sha256")
            ),
        },
        "fold_isolation": {
            "document_content_groups": len(set(document_groups)),
            "fold_id_sha256": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
        },
    }
    snapshot_name_map = {
        "entity_features": "features",
        "entity_manifest": "manifest",
        "entity_candidate": "candidate",
        "entity_audit": "audit",
        "evidence_graph": "evidence_graph",
        "pilot_selection": "selection",
    }
    for report_name, snapshot_name in snapshot_name_map.items():
        report["inputs"][report_name]["sha256"] = entity_payloads[
            "validated_file_sha256"
        ][snapshot_name]
    paths = write_outputs(
        output_dir=args.output_dir,
        records=records,
        labels=labels,
        fold_ids=fold_ids,
        probabilities=probabilities,
        report=report,
    )
    # A rewrite during report serialization also invalidates this evaluation run.
    assert_entity_freeze_unchanged(entity_payloads)
    print(
        "R3 entity-router grouped OOF complete: "
        f"base_rf={metrics['base_random_forest']['overall']['accuracy']:.4f} "
        f"entity_rf={metrics['entity_only_random_forest']['overall']['accuracy']:.4f} "
        f"full_rf={metrics['full_random_forest']['overall']['accuracy']:.4f} "
        f"routed_rf={metrics['pilot_routed_full_random_forest']['overall']['accuracy']:.4f} "
        f"report={paths['report']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABLATION_GROUPS",
    "EntityRouterEvaluationError",
    "MODEL_NAMES",
    "ablation_paired_report",
    "assert_entity_freeze_unchanged",
    "audit_saved_v4_random_forest",
    "evaluation_masks",
    "load_entity_inputs",
    "normalize_entity_candidate",
    "paired_changes",
    "routing_invariants",
    "run_grouped_oof",
    "write_outputs",
]
