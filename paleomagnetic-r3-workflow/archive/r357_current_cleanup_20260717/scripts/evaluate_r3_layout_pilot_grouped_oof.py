#!/usr/bin/env python3
"""Evaluate frozen R3 layout features with the unchanged V4 grouped folds.

The layout matrix and all model configurations are validated before R3 is
opened.  Each learner evaluates base V4, the two frozen availability flags,
and the full layout matrix; a pilot-only deployment route is then derived from
the base and full probabilities.  Saved V4 OOF probabilities are loaded only
after fitting, solely to audit reproduction of the base random forest; they
are never model inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_trace_page_graph_features as freeze  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
LAYOUT_DIR = ROOT / "tmp/r3_layout_pilot_v1"
DEFAULT_LAYOUT_FEATURES = LAYOUT_DIR / "r3_layout_features_v1.npz"
DEFAULT_LAYOUT_MANIFEST = LAYOUT_DIR / "r3_layout_features_v1.manifest.json"
DEFAULT_LAYOUT_CANDIDATE = LAYOUT_DIR / "r3_layout_candidate_freeze_v1.json"
DEFAULT_PILOT_SELECTION = LAYOUT_DIR / "r3_layout_pilot_manifest_v1.json"
DEFAULT_V4_OOF = V4_DIR / "trace_clean_nested_oof_v4.npz"
DEFAULT_V4_REPORT = V4_DIR / "trace_clean_nested_oof_report_v4.json"
DEFAULT_OUTPUT_DIR = LAYOUT_DIR

OOF_NAME = "r3_layout_pilot_grouped_oof_v1.npz"
REPORT_NAME = "r3_layout_pilot_grouped_oof_report_v1.json"
RECEIPT_NAME = "r3_layout_pilot_grouped_oof_receipt_v1.json"
EXPECTED_PILOT_ROWS = 222
THRESHOLD = 0.5
IDENTITY_FIELDS = ("REFNO", "RESULTNO", "ROCKNAME")
MODEL_NAMES = (
    "base_random_forest",
    "flags_only_random_forest",
    "augmented_random_forest",
    "pilot_routed_full_random_forest",
    "base_hgb",
    "flags_only_hgb",
    "augmented_hgb",
    "pilot_routed_full_hgb",
)
FORBIDDEN_FEATURE_RE = re.compile(
    r"(?:^|_)(?:R3|R5|R7|PMAGRESULT|ROCKMAG|PREDICTIONS?|OOF|"
    r"EVALUATION|METRICS?)(?:$|_)",
    re.I,
)


class LayoutPilotEvaluationError(ValueError):
    """Raised when the frozen layout evaluation boundary is invalid."""


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
        raise LayoutPilotEvaluationError(f"{description} is not a SHA-256 digest")
    return digest


def _schema_is_layout(payload: Mapping[str, Any], *, candidate: bool) -> bool:
    schema = clean(payload.get("schema_version")).casefold()
    return schema.startswith("r3_layout_") and (
        ("candidate" in schema) if candidate else ("feature" in schema)
    )


def _validate_label_blind_boundary(
    payload: Mapping[str, Any], description: str, *, require_no_outer_selection: bool
) -> None:
    boundary = payload.get("runtime_boundary") or payload.get("boundary") or {}
    if not isinstance(boundary, Mapping):
        raise LayoutPilotEvaluationError(f"{description} lacks a runtime boundary")
    for name in (
        "labels_read",
        "pmagresult_or_rockmag_read",
        "prior_predictions_read",
        "evaluation_results_read",
    ):
        if boundary.get(name) is not False:
            raise LayoutPilotEvaluationError(
                f"{description} boundary must declare {name}=false"
            )
    if require_no_outer_selection and boundary.get(
        "outer_test_metrics_used_for_selection"
    ) is not False:
        raise LayoutPilotEvaluationError(
            f"{description} must declare outer_test_metrics_used_for_selection=false"
        )


def _validate_selection_manifest(
    path: Path,
    records: Sequence[Mapping[str, str]],
    *,
    expected_rows: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        clean(payload.get("schema_version")) != "r3_layout_pilot_manifest_v1"
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise LayoutPilotEvaluationError("pilot selection manifest has an invalid schema")
    if tuple(payload.get("identity_fields") or ()) != IDENTITY_FIELDS:
        raise LayoutPilotEvaluationError("pilot selection identity fields changed")
    _validate_label_blind_boundary(
        payload, "pilot selection", require_no_outer_selection=False
    )
    rows = payload.get("rows") or []
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping) or tuple(row.keys()) != IDENTITY_FIELDS for row in rows
    ):
        raise LayoutPilotEvaluationError(
            "pilot selection rows must contain only REFNO, RESULTNO, ROCKNAME"
        )
    if expected_rows is not None and len(rows) != expected_rows:
        raise LayoutPilotEvaluationError(
            f"pilot selection has {len(rows)} rows; expected {expected_rows}"
        )
    if len({clean(row["RESULTNO"]) for row in rows}) != len(rows):
        raise LayoutPilotEvaluationError("pilot selection contains duplicate RESULTNO values")
    if freeze.json_sha256(rows) != clean(payload.get("selected_rows_sha256")):
        raise LayoutPilotEvaluationError("pilot selected-row hash mismatch")
    rule = payload.get("selection_rule") or {}
    if freeze.json_sha256(rule) != clean(payload.get("selection_rule_sha256")):
        raise LayoutPilotEvaluationError("pilot selection-rule hash mismatch")
    pdfs = [
        {
            "REFNO": clean(paper.get("REFNO")),
            "name": clean((paper.get("pdf") or {}).get("name")),
            "sha256": clean((paper.get("pdf") or {}).get("sha256")),
        }
        for paper in payload.get("papers") or []
    ]
    if freeze.json_sha256(pdfs) != clean(payload.get("selected_pdf_set_sha256")):
        raise LayoutPilotEvaluationError("pilot selected-PDF hash mismatch")
    selection_payload = {"selection_rule": rule, "rows": rows, "pdfs": pdfs}
    if freeze.json_sha256(selection_payload) != clean(payload.get("selection_sha256")):
        raise LayoutPilotEvaluationError("pilot combined selection hash mismatch")

    record_index = {
        tuple(clean(record[field]) for field in IDENTITY_FIELDS): index
        for index, record in enumerate(records)
    }
    if len(record_index) != len(records):
        raise LayoutPilotEvaluationError("V4 identity contains duplicate complete rows")
    mask = np.zeros(len(records), dtype=bool)
    for row in rows:
        key = tuple(clean(row[field]) for field in IDENTITY_FIELDS)
        if key not in record_index:
            raise LayoutPilotEvaluationError(
                f"pilot selection row is absent from V4 PDF identity: {key}"
            )
        mask[record_index[key]] = True
    if int(mask.sum()) != len(rows):
        raise LayoutPilotEvaluationError("pilot selection does not map one-to-one to V4 rows")
    return mask, payload


def _declared_output_sha(payload: Mapping[str, Any], kind: str) -> str:
    aliases = {
        "features": ("features", "layout_features", "feature_matrix", "npz"),
        "candidate": ("candidate_freeze", "candidate", "model_candidate_freeze"),
    }[kind]
    paths: list[tuple[str, ...]] = []
    for alias in aliases:
        paths.extend(
            [
                ("outputs", alias, "sha256"),
                ("output", alias, "sha256"),
                (f"{alias}_sha256",),
            ]
        )
    value = _first(payload, paths)
    return _require_sha256(value, f"layout manifest {kind} output hash")


def _model_config(
    candidate: Mapping[str, Any], aliases: Sequence[str], required: Sequence[str]
) -> dict[str, Any]:
    models = (
        candidate.get("model_candidates")
        or candidate.get("models")
        or candidate.get("fixed_models")
        or {}
    )
    if not isinstance(models, Mapping):
        raise LayoutPilotEvaluationError("layout candidate model_candidates is invalid")
    declared = [models[name] for name in aliases if name in models]
    if not declared or any(not isinstance(config, Mapping) for config in declared):
        raise LayoutPilotEvaluationError(
            f"layout candidate lacks fixed model config for {aliases[0]}"
        )
    output = dict(declared[0])
    if any(dict(config) != output for config in declared[1:]):
        raise LayoutPilotEvaluationError(
            f"layout candidate declares inconsistent base/augmented {aliases[0]} configs"
        )
    missing = [name for name in required if name not in output]
    if missing:
        raise LayoutPilotEvaluationError(
            f"layout candidate {aliases[0]} config lacks: {', '.join(missing)}"
        )
    if "random_state" not in output:
        if "random_state" not in candidate:
            raise LayoutPilotEvaluationError(
                f"layout candidate {aliases[0]} config lacks random_state"
            )
        output["random_state"] = candidate["random_state"]
    return output


def _positive_int(value: Any, description: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LayoutPilotEvaluationError(f"{description} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise LayoutPilotEvaluationError(f"{description} must be an integer")
    if parsed <= 0:
        raise LayoutPilotEvaluationError(f"{description} must be positive")
    return parsed


def _integer(value: Any, description: str) -> int:
    if isinstance(value, bool):
        raise LayoutPilotEvaluationError(f"{description} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LayoutPilotEvaluationError(f"{description} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise LayoutPilotEvaluationError(f"{description} must be an integer")
    return parsed


def _finite_float(value: Any, description: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LayoutPilotEvaluationError(f"{description} must be numeric") from exc
    if not np.isfinite(parsed):
        raise LayoutPilotEvaluationError(f"{description} must be finite")
    return parsed


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
        raise LayoutPilotEvaluationError(
            "layout random_forest config contains unsupported fields: "
            + ", ".join(sorted(set(random_forest) - rf_allowed))
        )
    if set(hgb) - hgb_allowed:
        raise LayoutPilotEvaluationError(
            "layout hgb config contains unsupported fields: "
            + ", ".join(sorted(set(hgb) - hgb_allowed))
        )

    random_forest["n_estimators"] = _positive_int(
        random_forest["n_estimators"], "random_forest.n_estimators"
    )
    random_forest["min_samples_leaf"] = _positive_int(
        random_forest["min_samples_leaf"], "random_forest.min_samples_leaf"
    )
    random_forest["random_state"] = _integer(
        random_forest["random_state"], "random_forest.random_state"
    )
    random_forest["n_jobs"] = _integer(
        random_forest.get("n_jobs", -1), "random_forest.n_jobs"
    )
    if random_forest["n_jobs"] == 0:
        raise LayoutPilotEvaluationError("random_forest.n_jobs cannot be zero")
    max_features = random_forest["max_features"]
    if isinstance(max_features, str):
        if max_features not in {"sqrt", "log2"}:
            raise LayoutPilotEvaluationError(
                "random_forest.max_features string must be sqrt or log2"
            )
    elif isinstance(max_features, int) and not isinstance(max_features, bool):
        random_forest["max_features"] = _positive_int(
            max_features, "random_forest.max_features"
        )
    elif not isinstance(max_features, bool):
        parsed_max_features = _finite_float(
            max_features, "random_forest.max_features"
        )
        if not 0.0 < parsed_max_features <= 1.0:
            raise LayoutPilotEvaluationError(
                "numeric random_forest.max_features must be in (0, 1] or a positive integer"
            )
        random_forest["max_features"] = parsed_max_features
    else:
        raise LayoutPilotEvaluationError(
            "random_forest.max_features cannot be boolean"
        )

    hgb["max_iter"] = _positive_int(hgb["max_iter"], "hgb.max_iter")
    hgb["max_leaf_nodes"] = _positive_int(
        hgb["max_leaf_nodes"], "hgb.max_leaf_nodes"
    )
    if hgb["max_leaf_nodes"] < 2:
        raise LayoutPilotEvaluationError("hgb.max_leaf_nodes must be at least two")
    hgb["min_samples_leaf"] = _positive_int(
        hgb["min_samples_leaf"], "hgb.min_samples_leaf"
    )
    hgb["learning_rate"] = _finite_float(hgb["learning_rate"], "hgb.learning_rate")
    if hgb["learning_rate"] <= 0.0:
        raise LayoutPilotEvaluationError("hgb.learning_rate must be positive")
    hgb["l2_regularization"] = _finite_float(
        hgb["l2_regularization"], "hgb.l2_regularization"
    )
    if hgb["l2_regularization"] < 0.0:
        raise LayoutPilotEvaluationError("hgb.l2_regularization cannot be negative")
    hgb["random_state"] = _integer(hgb["random_state"], "hgb.random_state")


def _normalize_feature_ablation_candidates(
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    raw = candidate.get("feature_ablation_candidates")
    if not isinstance(raw, Mapping):
        raise LayoutPilotEvaluationError(
            "layout candidate lacks feature_ablation_candidates"
        )
    base = raw.get("base_v4")
    flags = raw.get("pilot_flags_only")
    full = raw.get("full_layout")
    if not all(isinstance(value, Mapping) for value in (base, flags, full)):
        raise LayoutPilotEvaluationError(
            "layout feature ablations must declare base_v4, pilot_flags_only, and full_layout"
        )
    if list(base.get("layout_columns") or []) != []:
        raise LayoutPilotEvaluationError("base_v4 ablation must use no layout columns")
    if list(flags.get("layout_columns") or []) != [0, 1]:
        raise LayoutPilotEvaluationError(
            "pilot_flags_only ablation must use layout columns [0, 1]"
        )
    flag_names = list(flags.get("feature_names") or [])
    if len(flag_names) != 2 or len(set(map(clean, flag_names))) != 2:
        raise LayoutPilotEvaluationError(
            "pilot_flags_only must declare two distinct feature names"
        )
    if full.get("layout_columns") != "all":
        raise LayoutPilotEvaluationError("full_layout ablation must use all layout columns")
    route = candidate.get("deployment_route")
    if not isinstance(route, Mapping) or (
        clean(route.get("layout_scope")) != "frozen_pilot_rows_only"
        or clean(route.get("nonpilot_rows")) != "base_v4_unchanged"
        or route.get("membership_feature_is_not_evidence") is not True
    ):
        raise LayoutPilotEvaluationError(
            "layout candidate lacks the frozen pilot-only deployment route"
        )
    effect_gates = candidate.get("effect_acceptance_gates") or {}
    if not isinstance(effect_gates, Mapping) or effect_gates.get(
        "require_nonpilot_rows_unchanged"
    ) is not True:
        raise LayoutPilotEvaluationError(
            "layout candidate must require nonpilot rows to remain unchanged"
        )
    return {
        "base_v4_layout_columns": [],
        "pilot_flags_only_layout_columns": [0, 1],
        "pilot_flags_only_feature_names": list(map(clean, flag_names)),
        "full_layout_columns": "all",
        "deployment_route": {
            "layout_scope": "frozen_pilot_rows_only",
            "nonpilot_rows": "base_v4_unchanged",
            "membership_feature_is_not_evidence": True,
        },
    }


def normalize_layout_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not _schema_is_layout(candidate, candidate=True):
        raise LayoutPilotEvaluationError("layout candidate has an invalid schema_version")
    if candidate.get("status") != "FROZEN_BEFORE_LABEL_ACCESS":
        raise LayoutPilotEvaluationError("layout candidate was not frozen before labels")
    _validate_label_blind_boundary(
        candidate, "layout candidate", require_no_outer_selection=True
    )
    threshold = _first(
        candidate,
        (("fixed_decision_threshold",), ("decision_threshold",), ("threshold",)),
    )
    if threshold is None or _finite_float(threshold, "layout candidate threshold") != THRESHOLD:
        raise LayoutPilotEvaluationError("layout candidate threshold must be fixed at 0.5")
    random_forest = _model_config(
        candidate,
        (
            "random_forest",
            "structured_random_forest",
            "rf",
            "base_random_forest",
            "augmented_random_forest",
        ),
        ("n_estimators", "max_features", "min_samples_leaf"),
    )
    hgb = _model_config(
        candidate,
        (
            "hgb",
            "structured_hgb",
            "hist_gradient_boosting",
            "base_hgb",
            "augmented_hgb",
        ),
        (
            "max_iter",
            "learning_rate",
            "max_leaf_nodes",
            "min_samples_leaf",
            "l2_regularization",
        ),
    )
    _validate_fixed_model_configs(random_forest, hgb)
    feature_ablations = _normalize_feature_ablation_candidates(candidate)
    return {
        "random_forest": random_forest,
        "hgb": hgb,
        "feature_ablation_candidates": feature_ablations,
        "fixed_decision_threshold": THRESHOLD,
        "schema_version": clean(candidate.get("schema_version")),
    }


def load_layout_inputs(
    *,
    records: Sequence[Mapping[str, str]],
    layout_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    selection_path: Path,
    expected_pilot_rows: int | None,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any], dict[str, Any]]:
    """Validate all label-blind layout inputs and return their frozen matrix."""

    layout_path = layout_path.resolve()
    manifest_path = manifest_path.resolve()
    candidate_path = candidate_path.resolve()
    selection_path = selection_path.resolve()
    for path in (layout_path, manifest_path, candidate_path, selection_path):
        if not path.is_file():
            raise LayoutPilotEvaluationError(f"layout input does not exist: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if not _schema_is_layout(manifest, candidate=False):
        raise LayoutPilotEvaluationError("layout feature manifest has an invalid schema")
    if manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS":
        raise LayoutPilotEvaluationError("layout features were not frozen without labels")
    _validate_label_blind_boundary(
        manifest, "layout feature manifest", require_no_outer_selection=False
    )
    normalized_candidate = normalize_layout_candidate(candidate)

    alignment = freeze._record_alignment_sha256(records)
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        if clean(payload.get("alignment_sha256")) != alignment:
            raise LayoutPilotEvaluationError(
                f"layout {description} does not align with the V4 identity"
            )
    if freeze.file_sha256(layout_path) != _declared_output_sha(manifest, "features"):
        raise LayoutPilotEvaluationError("layout NPZ file hash mismatch")
    if freeze.file_sha256(candidate_path) != _declared_output_sha(manifest, "candidate"):
        raise LayoutPilotEvaluationError("layout candidate file hash mismatch")

    pilot_mask, selection = _validate_selection_manifest(
        selection_path, records, expected_rows=expected_pilot_rows
    )
    selection_file_sha = freeze.file_sha256(selection_path)
    declared_selection_file_sha = _first(
        manifest,
        (
            ("inputs", "selection_manifest", "sha256"),
            ("inputs", "pilot_selection", "sha256"),
            ("selection_manifest_sha256",),
        ),
    )
    if _require_sha256(
        declared_selection_file_sha, "layout selection-manifest input hash"
    ) != selection_file_sha:
        raise LayoutPilotEvaluationError("layout selection-manifest file hash mismatch")
    selection_sha = clean(selection.get("selection_sha256"))
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        declared = _first(
            payload,
            (
                ("pilot_selection_sha256",),
                ("selection_sha256",),
                ("inputs", "selection_manifest", "selection_sha256"),
                ("inputs", "pilot_selection", "selection_sha256"),
            ),
        )
        if clean(declared) != selection_sha:
            raise LayoutPilotEvaluationError(
                f"layout {description} pilot selection hash mismatch"
            )

    with np.load(layout_path) as archive:
        if archive.files != ["layout"]:
            raise LayoutPilotEvaluationError(
                "layout NPZ must contain exactly one array named layout"
            )
        frozen = np.asarray(archive["layout"])
    feature_names = list(
        _first(manifest, (("feature_names",), ("features", "names"))) or []
    )
    if (
        frozen.ndim != 2
        or frozen.shape != (len(records), len(feature_names))
        or not feature_names
        or len(feature_names) != len(set(feature_names))
        or not np.issubdtype(frozen.dtype, np.number)
        or not np.isfinite(frozen).all()
    ):
        raise LayoutPilotEvaluationError("layout feature matrix or names are invalid")
    if any(not clean(name).startswith("layout_") for name in feature_names):
        raise LayoutPilotEvaluationError("every layout feature name must start with layout_")
    forbidden = [name for name in feature_names if FORBIDDEN_FEATURE_RE.search(clean(name))]
    if forbidden:
        raise LayoutPilotEvaluationError(
            f"layout feature names contain forbidden historical fields: {forbidden[:5]}"
        )
    matrix_hash = freeze.array_sha256(frozen)
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        declared = _first(
            payload, (("matrix_sha256",), ("layout_matrix_sha256",))
        )
        if clean(declared) != matrix_hash:
            raise LayoutPilotEvaluationError(f"layout {description} matrix hash mismatch")
    declared_names_hash = _first(
        manifest, (("feature_names_sha256",), ("features", "names_sha256"))
    )
    if declared_names_hash is not None and clean(declared_names_hash) != freeze.json_sha256(
        feature_names
    ):
        raise LayoutPilotEvaluationError("layout feature-name hash mismatch")
    candidate_names_hash = _first(
        candidate, (("feature_names_sha256",), ("features", "names_sha256"))
    )
    if candidate_names_hash is not None and clean(candidate_names_hash) != freeze.json_sha256(
        feature_names
    ):
        raise LayoutPilotEvaluationError("layout candidate feature-name hash mismatch")
    declared_flags = normalized_candidate["feature_ablation_candidates"][
        "pilot_flags_only_feature_names"
    ]
    if feature_names[:2] != declared_flags:
        raise LayoutPilotEvaluationError(
            "layout flags-only feature names do not match frozen columns [0, 1]"
        )
    if np.any(np.asarray(frozen[~pilot_mask], dtype=np.float64) != 0.0):
        raise LayoutPilotEvaluationError(
            "layout features outside the frozen pilot selection must be zero"
        )
    for item in (manifest.get("inputs") or {}).get("code") or []:
        code_path = Path(clean(item.get("path"))).resolve()
        expected = _require_sha256(item.get("sha256"), "layout code input hash")
        if not code_path.is_file() or freeze.file_sha256(code_path) != expected:
            raise LayoutPilotEvaluationError(
                f"layout frozen code hash mismatch: {code_path}"
            )
    return (
        frozen.astype(np.float64),
        feature_names,
        pilot_mask,
        normalized_candidate,
        {"manifest": manifest, "candidate": candidate, "selection": selection},
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
    layout_matrix: np.ndarray,
    pilot_mask: np.ndarray,
    fold_ids: np.ndarray,
    model_configs: Mapping[str, Mapping[str, Any]],
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=np.int8)
    base_matrix = np.asarray(base_matrix, dtype=np.float64)
    layout_matrix = np.asarray(layout_matrix, dtype=np.float64)
    pilot_mask = np.asarray(pilot_mask, dtype=bool)
    fold_ids = np.asarray(fold_ids, dtype=np.int8)
    if (
        labels.shape != (len(records),)
        or pilot_mask.shape != (len(records),)
        or fold_ids.shape != (len(records),)
        or base_matrix.ndim != 2
        or layout_matrix.ndim != 2
        or base_matrix.shape[0] != len(records)
        or layout_matrix.shape[0] != len(records)
        or not set(map(int, labels)).issubset({0, 1})
        or not np.isfinite(base_matrix).all()
        or not np.isfinite(layout_matrix).all()
    ):
        raise LayoutPilotEvaluationError("grouped OOF inputs do not align")
    for refno in {clean(record.get("REFNO")) for record in records}:
        assigned = {
            int(fold_ids[index])
            for index, record in enumerate(records)
            if clean(record.get("REFNO")) == refno
        }
        if len(assigned) != 1:
            raise LayoutPilotEvaluationError("grouped OOF split a REFNO")

    flags_only = np.concatenate((base_matrix, layout_matrix[:, :2]), axis=1)
    augmented = np.concatenate((base_matrix, layout_matrix), axis=1)
    probabilities = {
        name: np.full(len(records), np.nan, dtype=np.float64) for name in MODEL_NAMES
    }
    fold_audit: list[dict[str, Any]] = []
    folds = sorted(set(map(int, fold_ids)))
    if not folds or folds != list(range(1, len(folds) + 1)):
        raise LayoutPilotEvaluationError("grouped OOF fold ids are incomplete")
    for fold in folds:
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        train_labels = labels[train]
        if set(map(int, train_labels)) != {0, 1}:
            raise LayoutPilotEvaluationError(
                f"grouped OOF training fold {fold} lacks both R3 classes"
            )
        weights = trace.inverse_identity_weights(train, records)
        probabilities["base_random_forest"][test] = _fit_random_forest(
            base_matrix[train],
            base_matrix[test],
            train_labels,
            weights,
            model_configs["random_forest"],
        )
        probabilities["flags_only_random_forest"][test] = _fit_random_forest(
            flags_only[train],
            flags_only[test],
            train_labels,
            weights,
            model_configs["random_forest"],
        )
        probabilities["augmented_random_forest"][test] = _fit_random_forest(
            augmented[train],
            augmented[test],
            train_labels,
            weights,
            model_configs["random_forest"],
        )
        probabilities["pilot_routed_full_random_forest"][test] = np.where(
            pilot_mask[test],
            probabilities["augmented_random_forest"][test],
            probabilities["base_random_forest"][test],
        )
        probabilities["base_hgb"][test] = _fit_hgb(
            base_matrix[train],
            base_matrix[test],
            train_labels,
            weights,
            model_configs["hgb"],
        )
        probabilities["flags_only_hgb"][test] = _fit_hgb(
            flags_only[train],
            flags_only[test],
            train_labels,
            weights,
            model_configs["hgb"],
        )
        probabilities["augmented_hgb"][test] = _fit_hgb(
            augmented[train],
            augmented[test],
            train_labels,
            weights,
            model_configs["hgb"],
        )
        probabilities["pilot_routed_full_hgb"][test] = np.where(
            pilot_mask[test],
            probabilities["augmented_hgb"][test],
            probabilities["base_hgb"][test],
        )
        current = {
            "outer_fold": fold,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "test_papers": len({clean(records[int(i)].get("REFNO")) for i in test}),
            "model_selection": "none_fixed_candidates_from_layout_freeze",
            "outer_test_metrics_used_for_selection": False,
            "metrics": {
                name: trace.metric(labels[test], values[test])
                for name, values in probabilities.items()
            },
        }
        fold_audit.append(current)
        if verbose:
            print(
                f"R3 layout grouped OOF: fold={fold}/{len(folds)} "
                f"base_rf={current['metrics']['base_random_forest']['accuracy']:.4f} "
                f"flags_rf={current['metrics']['flags_only_random_forest']['accuracy']:.4f} "
                f"full_rf={current['metrics']['augmented_random_forest']['accuracy']:.4f} "
                f"routed_rf={current['metrics']['pilot_routed_full_random_forest']['accuracy']:.4f}",
                flush=True,
            )
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("layout grouped OOF probabilities are incomplete")
    for family in ("random_forest", "hgb"):
        base = probabilities[f"base_{family}"]
        full = probabilities[
            "augmented_random_forest" if family == "random_forest" else "augmented_hgb"
        ]
        routed = probabilities[f"pilot_routed_full_{family}"]
        if not np.array_equal(routed[~pilot_mask], base[~pilot_mask]):
            raise AssertionError("pilot route changed a nonpilot probability")
        if not np.array_equal(routed[pilot_mask], full[pilot_mask]):
            raise AssertionError("pilot route differs from full layout on pilot rows")
    return (
        {name: values.astype(np.float32) for name, values in probabilities.items()},
        fold_audit,
    )


def evaluation_masks(
    records: Sequence[Mapping[str, str]], labels: np.ndarray, pilot_mask: np.ndarray
) -> dict[str, np.ndarray]:
    by_refno: dict[str, list[int]] = defaultdict(list)
    identity_counts = Counter(
        (clean(record.get("REFNO")), canonical_target_identity(clean(record.get("ROCKNAME"))))
        for record in records
    )
    for index, record in enumerate(records):
        by_refno[clean(record.get("REFNO"))].append(index)
    distinct_counts = {
        refno: len(
            {
                canonical_target_identity(clean(records[index].get("ROCKNAME")))
                for index in indices
            }
        )
        for refno, indices in by_refno.items()
    }
    mixed = {
        refno: len({int(labels[index]) for index in indices}) > 1
        for refno, indices in by_refno.items()
    }
    pilot = np.asarray(pilot_mask, dtype=bool)
    distinct_gte_8 = np.asarray(
        [distinct_counts[clean(record.get("REFNO"))] >= 8 for record in records],
        dtype=bool,
    )
    duplicate_identity = np.asarray(
        [
            identity_counts[
                (
                    clean(record.get("REFNO")),
                    canonical_target_identity(clean(record.get("ROCKNAME"))),
                )
            ]
            > 1
            for record in records
        ],
        dtype=bool,
    )
    mixed_paper = np.asarray(
        [mixed[clean(record.get("REFNO"))] for record in records], dtype=bool
    )
    return {
        "overall": np.ones(len(records), dtype=bool),
        "pilot_222": pilot,
        "pilot_distinct_canonical_gte_8": pilot & distinct_gte_8,
        "pilot_duplicate_identity_ambiguity": pilot & duplicate_identity,
        "pilot_mixed_label_paper_diagnostic": pilot & mixed_paper,
    }


def paired_changes(
    labels: np.ndarray,
    base: np.ndarray,
    augmented: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    labels = np.asarray(labels)[mask]
    base_prediction = np.asarray(base)[mask] >= THRESHOLD
    augmented_prediction = np.asarray(augmented)[mask] >= THRESHOLD
    base_correct = base_prediction == labels
    augmented_correct = augmented_prediction == labels
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
    for family in ("random_forest", "hgb"):
        full_name = (
            "augmented_random_forest" if family == "random_forest" else "augmented_hgb"
        )
        names = {
            "base_vs_flags_only_availability": (
                f"base_{family}",
                f"flags_only_{family}",
            ),
            "flags_only_vs_full_layout": (f"flags_only_{family}", full_name),
            "base_vs_full_layout": (f"base_{family}", full_name),
            "base_vs_pilot_routed_full": (
                f"base_{family}",
                f"pilot_routed_full_{family}",
            ),
        }
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
            for comparison, (base_name, augmented_name) in names.items()
        }
    return output


def routing_invariants(
    probabilities: Mapping[str, np.ndarray], pilot_mask: np.ndarray
) -> dict[str, Any]:
    pilot_mask = np.asarray(pilot_mask, dtype=bool)
    output: dict[str, Any] = {}
    for family in ("random_forest", "hgb"):
        base = np.asarray(probabilities[f"base_{family}"], dtype=np.float32)
        full = np.asarray(
            probabilities[
                "augmented_random_forest"
                if family == "random_forest"
                else "augmented_hgb"
            ],
            dtype=np.float32,
        )
        routed = np.asarray(
            probabilities[f"pilot_routed_full_{family}"], dtype=np.float32
        )
        nonpilot_equal = np.array_equal(routed[~pilot_mask], base[~pilot_mask])
        pilot_equal = np.array_equal(routed[pilot_mask], full[pilot_mask])
        if not nonpilot_equal or not pilot_equal:
            raise AssertionError("frozen pilot routing invariant failed")
        output[family] = {
            "pilot_rows": int(pilot_mask.sum()),
            "nonpilot_rows": int((~pilot_mask).sum()),
            "pilot_probabilities_equal_full_layout": True,
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
    """Load saved V4 predictions only after fitting and audit base RF parity."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_oof_sha = clean((report.get("output") or {}).get("sha256"))
    actual_oof_sha = freeze.file_sha256(oof_path)
    if expected_oof_sha != actual_oof_sha:
        raise LayoutPilotEvaluationError("saved V4 OOF hash does not match its report")
    with np.load(oof_path) as archive:
        required = {"labels", "fold_ids", "random_forest"}
        if not required.issubset(archive.files):
            raise LayoutPilotEvaluationError("saved V4 OOF lacks random-forest audit arrays")
        saved_labels = np.asarray(archive["labels"])
        saved_folds = np.asarray(archive["fold_ids"])
        saved_rf = np.asarray(archive["random_forest"], dtype=np.float32)
    reproduced = np.asarray(reproduced, dtype=np.float32)
    if not np.array_equal(saved_labels, labels) or not np.array_equal(saved_folds, fold_ids):
        raise LayoutPilotEvaluationError("saved V4 OOF identity arrays changed")
    if saved_rf.shape != reproduced.shape:
        raise LayoutPilotEvaluationError("saved V4 RF shape changed")
    maximum_difference = float(np.max(np.abs(saved_rf.astype(float) - reproduced.astype(float))))
    prediction_disagreements = int(
        np.sum((saved_rf >= THRESHOLD) != (reproduced >= THRESHOLD))
    )
    passed = maximum_difference <= 1e-7 and prediction_disagreements == 0
    audit = {
        "purpose": "post_fit_reproduction_audit_only",
        "loaded_after_all_layout_models_fitted": True,
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
        raise LayoutPilotEvaluationError(
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
    stored = {name: np.asarray(probabilities[name], dtype=np.float32) for name in MODEL_NAMES}
    freeze.atomic_npz(
        oof_path,
        labels=np.asarray(labels, dtype=np.int8),
        fold_ids=np.asarray(fold_ids, dtype=np.int8),
        **stored,
    )
    report["output"] = {"path": str(oof_path), "sha256": freeze.file_sha256(oof_path)}
    freeze.atomic_json(report_path, report)
    receipt = {
        "schema_version": "r3_layout_pilot_grouped_oof_receipt_v1",
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
                    name: int(stored[name][index] >= THRESHOLD) for name in MODEL_NAMES
                },
            }
            for index, record in enumerate(records)
        ],
        "report": {"path": str(report_path), "sha256": freeze.file_sha256(report_path)},
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
    parser.add_argument("--layout-features", type=Path, default=DEFAULT_LAYOUT_FEATURES)
    parser.add_argument("--layout-manifest", type=Path, default=DEFAULT_LAYOUT_MANIFEST)
    parser.add_argument("--layout-candidate", type=Path, default=DEFAULT_LAYOUT_CANDIDATE)
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
        layout_matrix,
        layout_feature_names,
        pilot_mask,
        model_configs,
        layout_payloads,
    ) = load_layout_inputs(
        records=records,
        layout_path=args.layout_features,
        manifest_path=args.layout_manifest,
        candidate_path=args.layout_candidate,
        selection_path=args.pilot_selection,
        expected_pilot_rows=EXPECTED_PILOT_ROWS,
    )

    # This is the first R3 access in the process, after every feature/candidate check.
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
        layout_matrix=layout_matrix,
        pilot_mask=pilot_mask,
        fold_ids=fold_ids,
        model_configs=model_configs,
        verbose=True,
    )

    # The saved V4 prediction is deliberately opened only after every OOF fit.
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
    for family in ("random_forest", "hgb"):
        if (
            paired[family]["base_vs_full_layout"]["pilot_222"]
            != paired[family]["base_vs_pilot_routed_full"]["pilot_222"]
        ):
            raise AssertionError(
                "pilot paired result differs between full and routed layout"
            )
    report = {
        "schema_version": "r3_layout_pilot_grouped_oof_report_v1",
        "status": "POSTHOC_EVALUATION_OF_FROZEN_LAYOUT_FEATURES_AND_CANDIDATES",
        "rows": len(records),
        "papers": len({record["REFNO"] for record in records}),
        "pilot_rows": int(pilot_mask.sum()),
        "threshold": THRESHOLD,
        "comparison_pairs": {
            "random_forest": {
                "availability_only": [
                    "base_random_forest",
                    "flags_only_random_forest",
                ],
                "layout_beyond_availability": [
                    "flags_only_random_forest",
                    "augmented_random_forest",
                ],
                "full_raw_augmented": [
                    "base_random_forest",
                    "augmented_random_forest",
                ],
                "pilot_routed_deployment": [
                    "base_random_forest",
                    "pilot_routed_full_random_forest",
                ],
            },
            "hgb": {
                "availability_only": ["base_hgb", "flags_only_hgb"],
                "layout_beyond_availability": [
                    "flags_only_hgb",
                    "augmented_hgb",
                ],
                "full_raw_augmented": ["base_hgb", "augmented_hgb"],
                "pilot_routed_deployment": [
                    "base_hgb",
                    "pilot_routed_full_hgb",
                ],
            },
        },
        "model_configs_from_layout_candidate_freeze": model_configs,
        "layout_feature_names": layout_feature_names,
        "metrics": metrics,
        "paired_changes": paired,
        "routing_invariants": route_audit,
        "ablation_interpretation": {
            "base_vs_flags_only_availability": (
                "effect of frozen pilot-membership and document-availability flags; not layout evidence"
            ),
            "flags_only_vs_full_layout": (
                "increment attributable to raw layout-graph evidence beyond availability flags"
            ),
            "base_vs_full_layout": "combined flag and full-layout effect on pilot rows",
            "base_vs_pilot_routed_full": (
                "deployment comparison: full layout on pilot rows, unchanged base elsewhere"
            ),
        },
        "fold_audit": fold_audit,
        "subset_definitions": {
            "overall": "all V4 PDF-backed rows",
            "pilot_222": "the frozen label-blind layout pilot selection",
            "pilot_distinct_canonical_gte_8": (
                "pilot rows whose REFNO has at least eight distinct canonical ROCKNAME identities"
            ),
            "pilot_duplicate_identity_ambiguity": (
                "pilot rows whose REFNO and canonical ROCKNAME identify more than one RESULTNO"
            ),
            "pilot_mixed_label_paper_diagnostic": (
                "post-label pilot diagnostic only: REFNO contains both R3 classes"
            ),
        },
        "subset_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "v4_random_forest_reproduction_audit": reproduction_audit,
        "boundary": {
            "layout_feature_generation_used_labels": False,
            "layout_candidate_freeze_used_labels": False,
            "base_features_are_frozen_v4_structured_matrix": True,
            "augmented_features_are_v4_plus_layout_only": True,
            "flags_only_features_are_v4_plus_frozen_layout_columns_0_1": True,
            "pilot_membership_flag_treated_as_availability_not_evidence": True,
            "deployment_route_preserves_nonpilot_probabilities_bitwise": True,
            "v4_oof_or_predictions_used_as_features": False,
            "v4_saved_rf_loaded_only_after_all_model_fits": True,
            "r3_used_in_outer_training_folds_only": True,
            "outer_test_metrics_used_for_model_selection": False,
            "pmagresult_or_rockmag_used": False,
            "prior_predictions_or_evaluation_reports_used_for_selection": False,
            "outer_test_refno_seen_in_training": False,
            "outer_test_pdf_content_seen_in_training": False,
            "mixed_subset_used_for_selection": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": freeze.file_sha256(path.resolve())}
            for name, path in (
                ("v4_features", args.v4_features),
                ("v4_manifest", args.v4_manifest),
                ("v4_target_context", args.v4_target_context),
                ("v4_candidate", args.v4_candidate),
                ("v4_fold_manifest", args.v4_fold_manifest),
                ("layout_features", args.layout_features),
                ("layout_manifest", args.layout_manifest),
                ("layout_candidate", args.layout_candidate),
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
        "layout_freeze": {
            "feature_schema": clean(
                layout_payloads["manifest"].get("schema_version")
            ),
            "candidate_schema": clean(
                layout_payloads["candidate"].get("schema_version")
            ),
            "selection_sha256": clean(
                layout_payloads["selection"].get("selection_sha256")
            ),
        },
        "fold_isolation": {
            "document_content_groups": len(set(document_groups)),
            "fold_id_sha256": hashlib.sha256(fold_ids.tobytes()).hexdigest(),
        },
    }
    paths = write_outputs(
        output_dir=args.output_dir,
        records=records,
        labels=labels,
        fold_ids=fold_ids,
        probabilities=probabilities,
        report=report,
    )
    print(
        "R3 layout grouped OOF complete: "
        f"base_rf={metrics['base_random_forest']['overall']['accuracy']:.4f} "
        f"flags_rf={metrics['flags_only_random_forest']['overall']['accuracy']:.4f} "
        f"full_rf={metrics['augmented_random_forest']['overall']['accuracy']:.4f} "
        f"routed_rf={metrics['pilot_routed_full_random_forest']['overall']['accuracy']:.4f} "
        f"report={paths['report']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "LayoutPilotEvaluationError",
    "MODEL_NAMES",
    "ablation_paired_report",
    "audit_saved_v4_random_forest",
    "evaluation_masks",
    "load_layout_inputs",
    "normalize_layout_candidate",
    "paired_changes",
    "routing_invariants",
    "run_grouped_oof",
    "write_outputs",
]
