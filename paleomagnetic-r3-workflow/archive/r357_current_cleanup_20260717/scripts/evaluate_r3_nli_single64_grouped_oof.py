#!/usr/bin/env python3
"""Evaluate frozen single-target R3 NLI features with the V4 grouped folds.

All NLI artifacts and model candidates are hash-validated before R3 is read.
Saved V4 predictions are opened only after every model has been fitted, solely
to audit exact reproduction of the unchanged base random forest.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_trace_page_graph_features as freeze  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace  # noqa: E402


V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
NLI_DIR = ROOT / "tmp/r3_nli_single64_v1"
DEFAULT_NLI_FEATURES = NLI_DIR / "r3_nli_criterion_features_v1.npz"
DEFAULT_NLI_MANIFEST = NLI_DIR / "r3_nli_criterion_features_v1.manifest.json"
DEFAULT_NLI_CANDIDATE = NLI_DIR / "r3_nli_candidate_freeze_v1.json"
DEFAULT_SELECTION = NLI_DIR / "r3_nli_single64_selection_v1.json"
DEFAULT_V4_OOF = V4_DIR / "trace_clean_nested_oof_v4.npz"
DEFAULT_V4_REPORT = V4_DIR / "trace_clean_nested_oof_report_v4.json"
DEFAULT_OUTPUT_DIR = NLI_DIR

OOF_NAME = "r3_nli_single64_grouped_oof_v1.npz"
REPORT_NAME = "r3_nli_single64_grouped_oof_report_v1.json"
RECEIPT_NAME = "r3_nli_single64_grouped_oof_receipt_v1.json"
FEATURE_SCHEMA = "r3_nli_criterion_features_v1"
CANDIDATE_SCHEMA = "r3_nli_candidate_freeze_v1"
SELECTION_SCHEMA = "r3_nli_single64_selection_v1"
PREINFERENCE_SCHEMA = "r3_nli_preinference_freeze_v1"
EXPECTED_SELECTED_ROWS = 64
THRESHOLD = 0.5
FLAG_FEATURE_NAMES = (
    "nli_pilot_selected",
    "nli_source_available",
    "nli_retrieval_available",
)
ZERO_SHOT_FEATURE = "nli_criterion_chain_min_support"
IDENTITY_FIELDS = ("REFNO", "RESULTNO", "ROCKNAME")
RF_MODEL_NAMES = (
    "base_random_forest",
    "flags_only_random_forest",
    "full_nli_random_forest",
    "pilot_routed_full_nli_random_forest",
)
AUXILIARY_MODEL_NAMES = (
    "selected_only_standardized_logistic",
    "selected_zero_shot_criterion",
)
MODEL_NAMES = (*RF_MODEL_NAMES, *AUXILIARY_MODEL_NAMES)
FORBIDDEN_FEATURE_RE = re.compile(
    r"(?:^|_)(?:R3|R5|R7|PMAGRESULT|ROCKMAG|PREDICTIONS?|OOF|"
    r"EVALUATION|METRICS?|LABELS?|GOLD)(?:$|_)",
    re.I,
)


class NliSingle64EvaluationError(ValueError):
    """Raised when the frozen NLI evaluation boundary is invalid."""


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
    digest = clean(value).casefold()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise NliSingle64EvaluationError(f"{description} is not a SHA-256 digest")
    return digest


def _finite_float(value: Any, description: str) -> float:
    if isinstance(value, bool):
        raise NliSingle64EvaluationError(f"{description} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise NliSingle64EvaluationError(f"{description} must be numeric") from exc
    if not np.isfinite(parsed):
        raise NliSingle64EvaluationError(f"{description} must be finite")
    return parsed


def _integer(value: Any, description: str, *, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise NliSingle64EvaluationError(f"{description} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise NliSingle64EvaluationError(f"{description} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise NliSingle64EvaluationError(f"{description} must be an integer")
    if positive and parsed <= 0:
        raise NliSingle64EvaluationError(f"{description} must be positive")
    return parsed


def _validate_label_blind_boundary(
    payload: Mapping[str, Any], description: str, *, require_outer_guard: bool
) -> None:
    boundary = payload.get("runtime_boundary") or payload.get("boundary") or {}
    if not isinstance(boundary, Mapping):
        raise NliSingle64EvaluationError(f"{description} lacks a runtime boundary")
    for name in (
        "labels_read",
        "pmagresult_or_rockmag_read",
        "prior_predictions_read",
        "evaluation_results_read",
    ):
        if boundary.get(name) is not False:
            raise NliSingle64EvaluationError(
                f"{description} boundary must declare {name}=false"
            )
    if require_outer_guard and boundary.get("outer_test_metrics_used_for_selection") is not False:
        raise NliSingle64EvaluationError(
            f"{description} must declare outer_test_metrics_used_for_selection=false"
        )


def _selection_payload_hash(payload: Mapping[str, Any]) -> str:
    return freeze.json_sha256(
        {
            "salt": payload.get("salt"),
            "fold_quotas": payload.get("fold_quotas"),
            "rows": payload.get("rows"),
        }
    )


def _validate_selection_manifest(
    path: Path,
    records: Sequence[Mapping[str, str]],
    *,
    fold_ids: np.ndarray | None,
    expected_rows: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        clean(payload.get("schema_version")) != SELECTION_SCHEMA
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise NliSingle64EvaluationError("NLI selection manifest has an invalid schema")
    _validate_label_blind_boundary(
        payload, "NLI selection", require_outer_guard=True
    )
    identity_fields = payload.get("identity_fields")
    if identity_fields is not None and tuple(identity_fields) != IDENTITY_FIELDS:
        raise NliSingle64EvaluationError("NLI selection identity fields changed")
    if not isinstance(payload.get("selection_rule"), str) or not clean(
        payload.get("selection_rule")
    ):
        raise NliSingle64EvaluationError("NLI selection rule is missing")
    salt = clean(payload.get("salt"))
    raw_quotas = payload.get("fold_quotas")
    if not salt or not isinstance(raw_quotas, Mapping) or not raw_quotas:
        raise NliSingle64EvaluationError("NLI selection salt or fold quotas are invalid")
    quotas: dict[int, int] = {}
    for raw_fold, raw_quota in raw_quotas.items():
        fold = _integer(raw_fold, "NLI selection quota fold", positive=True)
        quota = _integer(raw_quota, "NLI selection fold quota", positive=True)
        if fold in quotas:
            raise NliSingle64EvaluationError("NLI selection has duplicate fold quotas")
        quotas[fold] = quota
    rows = payload.get("rows")
    required_row_fields = {
        "row_index",
        *IDENTITY_FIELDS,
        "fold",
        "selection_rank_within_fold",
        "selection_digest",
    }
    if not isinstance(rows, list) or any(
        not isinstance(row, Mapping)
        or not required_row_fields.issubset(row)
        for row in rows
    ):
        raise NliSingle64EvaluationError(
            "NLI selection rows lack builder identity, fold, rank, or digest fields"
        )
    for row in rows:
        forbidden = [key for key in row if FORBIDDEN_FEATURE_RE.search(clean(key))]
        if forbidden:
            raise NliSingle64EvaluationError(
                f"NLI selection row contains forbidden fields: {forbidden}"
            )
        row_index = _integer(row.get("row_index"), "NLI selection row_index")
        if row_index < 0:
            raise NliSingle64EvaluationError("NLI selection row_index cannot be negative")
        _integer(row.get("fold"), "NLI selection row fold", positive=True)
        _integer(
            row.get("selection_rank_within_fold"),
            "NLI selection rank",
            positive=True,
        )
        _require_sha256(row.get("selection_digest"), "NLI selection digest")
    if expected_rows is not None and len(rows) != expected_rows:
        raise NliSingle64EvaluationError(
            f"NLI selection has {len(rows)} rows; expected {expected_rows}"
        )
    if sum(quotas.values()) != len(rows):
        raise NliSingle64EvaluationError("NLI selection fold quotas do not sum to rows")
    if len({clean(row["RESULTNO"]) for row in rows}) != len(rows):
        raise NliSingle64EvaluationError("NLI selection contains duplicate RESULTNO")
    if "selected_rows" in payload and _integer(
        payload.get("selected_rows"), "NLI selected_rows"
    ) != len(rows):
        raise NliSingle64EvaluationError("NLI selected_rows count mismatch")
    if "selected_rows_sha256" in payload and freeze.json_sha256(rows) != _require_sha256(
        payload.get("selected_rows_sha256"), "NLI selected_rows_sha256"
    ):
        raise NliSingle64EvaluationError("NLI selected-row hash mismatch")
    if "selection_rule_sha256" in payload and freeze.json_sha256(
        payload.get("selection_rule")
    ) != _require_sha256(
        payload.get("selection_rule_sha256"), "NLI selection_rule_sha256"
    ):
        raise NliSingle64EvaluationError("NLI selection-rule hash mismatch")
    if _selection_payload_hash(payload) != _require_sha256(
        payload.get("selection_sha256"), "NLI selection_sha256"
    ):
        raise NliSingle64EvaluationError("NLI combined selection hash mismatch")

    record_index = {
        tuple(clean(record[field]) for field in IDENTITY_FIELDS): index
        for index, record in enumerate(records)
    }
    if len(record_index) != len(records):
        raise NliSingle64EvaluationError("V4 identity contains duplicate complete rows")
    alignment = freeze._record_alignment_sha256(records)
    if clean(payload.get("alignment_sha256")) != alignment:
        raise NliSingle64EvaluationError("NLI selection alignment hash mismatch")
    refno_counts = Counter(clean(record["REFNO"]) for record in records)
    if "eligible_single_target_rows" in payload and _integer(
        payload.get("eligible_single_target_rows"), "NLI eligible rows"
    ) != sum(count == 1 for count in refno_counts.values()):
        raise NliSingle64EvaluationError("NLI eligible single-target count mismatch")
    if fold_ids is not None:
        fold_ids = np.asarray(fold_ids, dtype=np.int8)
        if fold_ids.shape != (len(records),):
            raise NliSingle64EvaluationError("V4 fold ids do not align with selection")
        expected_fold_sha = hashlib.sha256(fold_ids.tobytes()).hexdigest()
        if clean(payload.get("fold_id_sha256")) != expected_fold_sha:
            raise NliSingle64EvaluationError("NLI selection fold-id hash mismatch")
    mask = np.zeros(len(records), dtype=bool)
    fold_ranks: dict[int, list[int]] = {fold: [] for fold in quotas}
    fold_digests: dict[int, list[str]] = {fold: [] for fold in quotas}
    for row in rows:
        key = tuple(clean(row[field]) for field in IDENTITY_FIELDS)
        if key not in record_index:
            raise NliSingle64EvaluationError(
                f"NLI selection row is absent from V4 identity: {key}"
            )
        index = record_index[key]
        if _integer(row.get("row_index"), "NLI selection row_index") != index:
            raise NliSingle64EvaluationError(
                "NLI selection row_index does not match V4 identity order"
            )
        fold = _integer(row.get("fold"), "NLI selection row fold", positive=True)
        if fold not in quotas:
            raise NliSingle64EvaluationError("NLI selection row uses an undeclared fold")
        if fold_ids is not None and int(fold_ids[index]) != fold:
            raise NliSingle64EvaluationError(
                "NLI selection fold does not match the frozen V4 fold"
            )
        rank = _integer(
            row.get("selection_rank_within_fold"),
            "NLI selection rank",
            positive=True,
        )
        fold_ranks[fold].append(rank)
        expected_digest = hashlib.sha256(
            f"{salt}|{clean(row['REFNO'])}|{clean(row['RESULTNO'])}".encode("utf-8")
        ).hexdigest()
        if expected_digest != _require_sha256(
            row.get("selection_digest"), "NLI selection digest"
        ):
            raise NliSingle64EvaluationError("NLI selection digest is not reproducible")
        fold_digests[fold].append(expected_digest)
        if refno_counts[clean(records[index]["REFNO"])] != 1:
            raise NliSingle64EvaluationError(
                "NLI single64 selection contains a non-single-target REFNO"
            )
        mask[index] = True
    for fold, quota in quotas.items():
        if sorted(fold_ranks[fold]) != list(range(1, quota + 1)):
            raise NliSingle64EvaluationError(
                "NLI selection fold ranks do not match the declared quota"
            )
        ranked_digests = [
            digest
            for _, digest in sorted(zip(fold_ranks[fold], fold_digests[fold]))
        ]
        if ranked_digests != sorted(ranked_digests):
            raise NliSingle64EvaluationError(
                "NLI selection rank order does not follow its digest"
            )
    if int(mask.sum()) != len(rows) or len({clean(row["REFNO"]) for row in rows}) != len(rows):
        raise NliSingle64EvaluationError(
            "NLI selection does not map one-to-one to single-target papers"
        )
    return mask, payload


def _random_forest_config(candidate: Mapping[str, Any]) -> dict[str, Any]:
    models = candidate.get("model_candidates") or {}
    raw = models.get("random_forest") or models.get("base_random_forest")
    if not isinstance(raw, Mapping):
        raise NliSingle64EvaluationError("NLI candidate lacks random_forest")
    config = dict(raw)
    allowed = {
        "n_estimators",
        "max_features",
        "min_samples_leaf",
        "n_jobs",
        "random_state",
    }
    if set(config) - allowed:
        raise NliSingle64EvaluationError(
            "NLI random_forest config contains unsupported fields: "
            + ", ".join(sorted(set(config) - allowed))
        )
    for name in ("n_estimators", "min_samples_leaf"):
        config[name] = _integer(config.get(name), f"random_forest.{name}", positive=True)
    config["n_jobs"] = _integer(
        config.get("n_jobs", -1), "random_forest.n_jobs"
    )
    if config["n_jobs"] == 0:
        raise NliSingle64EvaluationError("random_forest.n_jobs cannot be zero")
    config["random_state"] = _integer(
        config.get("random_state"), "random_forest.random_state"
    )
    if config["random_state"] != 35_720_260:
        raise NliSingle64EvaluationError(
            "random_forest.random_state must reproduce frozen V4"
        )
    max_features = config.get("max_features")
    if isinstance(max_features, str):
        if max_features not in {"sqrt", "log2"}:
            raise NliSingle64EvaluationError("unsupported random_forest.max_features")
    elif isinstance(max_features, int) and not isinstance(max_features, bool):
        config["max_features"] = _integer(
            max_features, "random_forest.max_features", positive=True
        )
    else:
        value = _finite_float(max_features, "random_forest.max_features")
        if not 0.0 < value <= 1.0:
            raise NliSingle64EvaluationError(
                "numeric random_forest.max_features must be in (0, 1]"
            )
        config["max_features"] = value
    return config


def _logistic_config(candidate: Mapping[str, Any]) -> dict[str, Any]:
    models = candidate.get("model_candidates") or {}
    raw = (
        models.get("selected_only_logistic")
        or models.get("selected_logistic")
        or models.get("standardized_logistic")
        or models.get("logistic")
    )
    if not isinstance(raw, Mapping):
        raise NliSingle64EvaluationError("NLI candidate lacks selected-only logistic")
    config = dict(raw)
    if _finite_float(config.get("C"), "logistic.C") != 1.0:
        raise NliSingle64EvaluationError("selected-only logistic must use C=1")
    if config.get("standardize") is not True:
        raise NliSingle64EvaluationError(
            "selected-only logistic must declare standardize=true"
        )
    if config.get("class_weight") is not None:
        raise NliSingle64EvaluationError(
            "selected-only logistic must declare class_weight=null"
        )
    solver = clean(config.get("solver") or "liblinear")
    if solver not in {"liblinear", "lbfgs"}:
        raise NliSingle64EvaluationError("unsupported selected-only logistic solver")
    return {
        "C": 1.0,
        "standardize": True,
        "class_weight": None,
        "solver": solver,
        "max_iter": _integer(
            config.get("max_iter", 2000), "logistic.max_iter", positive=True
        ),
        "random_state": _integer(
            config.get("random_state", 35_720_260), "logistic.random_state"
        ),
    }


def _ablation_config(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw = candidate.get("feature_ablation_candidates") or {}
    base = raw.get("base_v4")
    flags = raw.get("pilot_flags_only") or raw.get("flags_only")
    full = raw.get("full_nli") or raw.get("full")
    if not all(isinstance(value, Mapping) for value in (base, flags, full)):
        raise NliSingle64EvaluationError(
            "NLI candidate lacks base_v4, pilot_flags_only, or full_nli ablations"
        )
    column_key = "nli_columns"
    if list(base.get(column_key) or []) != []:
        raise NliSingle64EvaluationError("base_v4 must use no NLI columns")
    if list(flags.get(column_key) or []) != [0, 1, 2]:
        raise NliSingle64EvaluationError(
            "pilot_flags_only must use NLI columns [0, 1, 2]"
        )
    names = tuple(map(clean, flags.get("feature_names") or FLAG_FEATURE_NAMES))
    if names != FLAG_FEATURE_NAMES:
        raise NliSingle64EvaluationError("NLI availability flag names changed")
    if full.get(column_key) != "all":
        raise NliSingle64EvaluationError("full_nli must use all NLI columns")
    return {
        "flags_only_columns": [0, 1, 2],
        "flags_only_feature_names": list(FLAG_FEATURE_NAMES),
        "full_nli_columns": "all",
    }


def _zero_shot_configs(candidate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    plural = candidate.get("zero_shot_candidates")
    if plural is not None:
        if not isinstance(plural, Mapping) or not plural:
            raise NliSingle64EvaluationError("zero_shot_candidates must be nonempty")
        declared = dict(plural)
    else:
        raw = candidate.get("zero_shot_candidate") or candidate.get(
            "zero_shot_criterion"
        )
        if raw is False:
            return {}
        if isinstance(raw, Mapping):
            if raw.get("enabled", True) is False:
                return {}
            declared = {"criterion": raw}
        elif candidate.get("zero_shot_feature") is not None:
            declared = {
                "criterion": {
                    "feature_name": candidate.get("zero_shot_feature"),
                    "threshold": candidate.get("zero_shot_threshold", 0.5),
                }
            }
        else:
            return {}
    output: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in declared.items():
        name = re.sub(r"[^a-z0-9]+", "_", clean(raw_name).casefold()).strip("_")
        if not name or not isinstance(raw_spec, Mapping):
            raise NliSingle64EvaluationError("zero-shot candidate name/spec is invalid")
        feature = clean(raw_spec.get("feature_name") or raw_spec.get("feature"))
        if not feature.startswith("nli_") or FORBIDDEN_FEATURE_RE.search(feature):
            raise NliSingle64EvaluationError(
                f"zero-shot candidate {name} has an invalid feature"
            )
        if _finite_float(raw_spec.get("threshold", 0.5), "zero-shot threshold") != 0.5:
            raise NliSingle64EvaluationError(
                f"zero-shot candidate {name} threshold must be 0.5"
            )
        strategy = f"selected_zero_shot_{name}"
        if strategy in output:
            raise NliSingle64EvaluationError("duplicate normalized zero-shot candidate")
        output[strategy] = {"feature_name": feature, "threshold": 0.5}
    return output


def normalize_nli_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if (
        clean(candidate.get("schema_version")) != CANDIDATE_SCHEMA
        or candidate.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
    ):
        raise NliSingle64EvaluationError("NLI candidate has an invalid schema")
    _validate_label_blind_boundary(
        candidate, "NLI candidate", require_outer_guard=True
    )
    if _finite_float(candidate.get("fixed_decision_threshold"), "threshold") != 0.5:
        raise NliSingle64EvaluationError("NLI fixed decision threshold must be 0.5")
    return {
        "random_forest": _random_forest_config(candidate),
        "selected_only_logistic": _logistic_config(candidate),
        "feature_ablation_candidates": _ablation_config(candidate),
        "zero_shot_candidates": _zero_shot_configs(candidate),
        "fixed_decision_threshold": 0.5,
        "schema_version": CANDIDATE_SCHEMA,
    }


def _declared_output_sha(manifest: Mapping[str, Any], kind: str) -> str:
    aliases = {
        "features": ("features", "nli_features", "feature_matrix", "npz"),
        "candidate": ("candidate_freeze", "candidate", "model_candidate_freeze"),
    }[kind]
    paths: list[tuple[str, ...]] = []
    for alias in aliases:
        paths.extend(
            (("outputs", alias, "sha256"), ("output", alias, "sha256"), (f"{alias}_sha256",))
        )
    return _require_sha256(
        _first(manifest, paths), f"NLI manifest {kind} output hash"
    )


def _validate_file_reference(
    reference: Any,
    description: str,
    *,
    base_dir: Path,
    expected_path: Path | None = None,
) -> Path:
    if not isinstance(reference, Mapping):
        raise NliSingle64EvaluationError(f"{description} is not a file reference")
    raw_path = clean(reference.get("path"))
    if not raw_path:
        raise NliSingle64EvaluationError(f"{description} lacks a path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    if expected_path is not None and path != expected_path.resolve():
        raise NliSingle64EvaluationError(f"{description} path mismatch")
    expected_sha = _require_sha256(reference.get("sha256"), f"{description} hash")
    if not path.is_file() or freeze.file_sha256(path) != expected_sha:
        raise NliSingle64EvaluationError(f"{description} file hash mismatch")
    return path


def _validate_manifest_files(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    *,
    expected_outputs: Mapping[str, Path],
) -> tuple[dict[str, Path], dict[str, Path]]:
    outputs = manifest.get("outputs")
    required_outputs = {
        "selection",
        "hypotheses",
        "preinference",
        "features",
        "candidate",
        "evidence",
        "audit",
    }
    if not isinstance(outputs, Mapping) or not required_outputs.issubset(outputs):
        raise NliSingle64EvaluationError(
            "NLI manifest lacks the complete frozen output set"
        )
    output_paths: dict[str, Path] = {}
    for name, reference in outputs.items():
        expected = expected_outputs.get(clean(name))
        output_paths[clean(name)] = _validate_file_reference(
            reference,
            f"NLI manifest output {name}",
            base_dir=manifest_path.parent,
            expected_path=expected,
        )
    if len(set(output_paths.values())) != len(output_paths):
        raise NliSingle64EvaluationError("NLI manifest output paths are not unique")

    inputs = manifest.get("inputs")
    required_inputs = {"identity", "page_cache", "fold_manifest", "concept_snapshot"}
    if not isinstance(inputs, Mapping) or not required_inputs.issubset(inputs):
        raise NliSingle64EvaluationError(
            "NLI manifest lacks the frozen file inputs"
        )
    input_paths: dict[str, Path] = {}
    for name in required_inputs:
        input_paths[name] = _validate_file_reference(
            inputs[name],
            f"NLI manifest input {name}",
            base_dir=manifest_path.parent,
        )
    return output_paths, input_paths


def _validate_preinference_freeze(
    *,
    candidate: Mapping[str, Any],
    candidate_path: Path,
    manifest: Mapping[str, Any],
    output_paths: Mapping[str, Path],
    selection: Mapping[str, Any],
    selection_path: Path,
) -> dict[str, Any]:
    preinference_path = _validate_file_reference(
        candidate.get("preinference_freeze"),
        "NLI candidate preinference freeze",
        base_dir=candidate_path.parent,
        expected_path=output_paths["preinference"],
    )
    preinference = json.loads(preinference_path.read_text(encoding="utf-8"))
    if (
        not isinstance(preinference, Mapping)
        or clean(preinference.get("schema_version")) != PREINFERENCE_SCHEMA
        or preinference.get("status")
        != "FROZEN_BEFORE_INFERENCE_AND_LABEL_ACCESS"
    ):
        raise NliSingle64EvaluationError("NLI preinference freeze has an invalid schema")
    _validate_label_blind_boundary(
        preinference, "NLI preinference freeze", require_outer_guard=True
    )
    spec = preinference.get("candidate_spec")
    if not isinstance(spec, Mapping):
        raise NliSingle64EvaluationError("NLI preinference candidate spec is missing")
    _validate_label_blind_boundary(
        spec, "NLI preinference candidate spec", require_outer_guard=True
    )
    spec_sha = freeze.json_sha256(spec)
    if spec_sha != _require_sha256(
        preinference.get("candidate_spec_sha256"),
        "NLI preinference candidate-spec hash",
    ) or spec_sha != _require_sha256(
        candidate.get("preinference_candidate_spec_sha256"),
        "NLI candidate preinference-spec hash",
    ):
        raise NliSingle64EvaluationError("NLI preinference candidate-spec hash mismatch")

    comparable_candidate = dict(candidate)
    for name in (
        "schema_version",
        "status",
        "preinference_candidate_spec_sha256",
        "preinference_freeze",
        "matrix_sha256",
        "evidence_sha256",
        "finalized_after_label_blind_inference",
    ):
        comparable_candidate.pop(name, None)
    comparable_spec = dict(spec)
    comparable_spec.pop("schema_version", None)
    comparable_spec.pop("status", None)
    if comparable_candidate != comparable_spec:
        raise NliSingle64EvaluationError(
            "NLI final candidate differs from its preinference freeze"
        )
    if candidate.get("finalized_after_label_blind_inference") is not True:
        raise NliSingle64EvaluationError(
            "NLI candidate was not finalized after label-blind inference"
        )

    preselection_path = _validate_file_reference(
        preinference.get("selection"),
        "NLI preinference selection",
        base_dir=preinference_path.parent,
        expected_path=selection_path,
    )
    if preselection_path != output_paths["selection"] or clean(
        (preinference.get("selection") or {}).get("selection_sha256")
    ) != clean(selection.get("selection_sha256")):
        raise NliSingle64EvaluationError("NLI preinference selection mismatch")

    hypotheses_path = _validate_file_reference(
        preinference.get("hypotheses"),
        "NLI preinference hypotheses",
        base_dir=preinference_path.parent,
        expected_path=output_paths["hypotheses"],
    )
    if hypotheses_path != output_paths["hypotheses"]:
        raise NliSingle64EvaluationError("NLI preinference hypotheses path mismatch")
    hypotheses_sha = clean((preinference.get("hypotheses") or {}).get("hypotheses_sha256"))
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        if clean(payload.get("hypotheses_sha256")) != hypotheses_sha:
            raise NliSingle64EvaluationError(
                f"NLI {description} hypotheses hash mismatch"
            )

    _validate_file_reference(
        preinference.get("builder"),
        "NLI preinference builder",
        base_dir=preinference_path.parent,
    )
    selector = spec.get("selector")
    if not isinstance(selector, Mapping):
        raise NliSingle64EvaluationError("NLI preinference selector is missing")
    _validate_file_reference(
        selector,
        "NLI preinference selector",
        base_dir=preinference_path.parent,
    )
    return dict(preinference)


def load_nli_inputs(
    *,
    records: Sequence[Mapping[str, str]],
    nli_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    selection_path: Path,
    fold_ids: np.ndarray,
    expected_selected_rows: int | None,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any], dict[str, Any]]:
    paths = [
        path.resolve()
        for path in (nli_path, manifest_path, candidate_path, selection_path)
    ]
    if any(not path.is_file() for path in paths):
        raise NliSingle64EvaluationError("one or more frozen NLI inputs are missing")
    nli_path, manifest_path, candidate_path, selection_path = paths
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        clean(manifest.get("schema_version")) != FEATURE_SCHEMA
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise NliSingle64EvaluationError("NLI feature manifest has an invalid schema")
    _validate_label_blind_boundary(
        manifest, "NLI feature manifest", require_outer_guard=True
    )
    if manifest.get("contains_labels") is not False:
        raise NliSingle64EvaluationError("NLI feature manifest must contain no labels")
    if _integer(manifest.get("rows"), "NLI manifest rows", positive=True) != len(records):
        raise NliSingle64EvaluationError("NLI manifest row count mismatch")
    normalized_candidate = normalize_nli_candidate(candidate)
    alignment = freeze._record_alignment_sha256(records)
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        if clean(payload.get("alignment_sha256")) != alignment:
            raise NliSingle64EvaluationError(
                f"NLI {description} does not align with V4 identity"
            )
    output_paths, input_paths = _validate_manifest_files(
        manifest,
        manifest_path,
        expected_outputs={
            "features": nli_path,
            "candidate": candidate_path,
            "selection": selection_path,
        },
    )

    selected_mask, selection = _validate_selection_manifest(
        selection_path,
        records,
        fold_ids=fold_ids,
        expected_rows=expected_selected_rows,
    )
    selection_sha = clean(selection.get("selection_sha256"))
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        if clean(payload.get("selection_sha256")) != selection_sha:
            raise NliSingle64EvaluationError(
                f"NLI {description} selection hash mismatch"
            )
    selection_inputs = selection.get("inputs")
    if not isinstance(selection_inputs, Mapping):
        raise NliSingle64EvaluationError("NLI selection frozen inputs are missing")
    for name in ("identity", "fold_manifest"):
        _validate_file_reference(
            selection_inputs.get(name),
            f"NLI selection input {name}",
            base_dir=selection_path.parent,
            expected_path=input_paths[name],
        )

    candidate_input_sha = candidate.get("input_sha256")
    if not isinstance(candidate_input_sha, Mapping):
        raise NliSingle64EvaluationError("NLI candidate input hashes are missing")
    for name in ("identity", "page_cache", "fold_manifest", "concept_snapshot"):
        manifest_sha = _require_sha256(
            (manifest.get("inputs") or {}).get(name, {}).get("sha256"),
            f"NLI manifest input {name} hash",
        )
        if _require_sha256(
            candidate_input_sha.get(name), f"NLI candidate input {name} hash"
        ) != manifest_sha:
            raise NliSingle64EvaluationError(
                f"NLI candidate input {name} hash mismatch"
            )
    if _require_sha256(candidate.get("evidence_sha256"), "NLI candidate evidence hash") != (
        _require_sha256(
            (manifest.get("outputs") or {}).get("evidence", {}).get("sha256"),
            "NLI manifest evidence output hash",
        )
    ):
        raise NliSingle64EvaluationError("NLI candidate evidence hash mismatch")
    preinference = _validate_preinference_freeze(
        candidate=candidate,
        candidate_path=candidate_path,
        manifest=manifest,
        output_paths=output_paths,
        selection=selection,
        selection_path=selection_path,
    )

    with np.load(nli_path) as archive:
        if archive.files != ["nli"]:
            raise NliSingle64EvaluationError(
                "NLI NPZ must contain exactly one array named nli"
            )
        matrix = np.asarray(archive["nli"])
    feature_names = list(manifest.get("feature_names") or [])
    if (
        matrix.ndim != 2
        or matrix.shape != (len(records), len(feature_names))
        or not feature_names
        or len(feature_names) != len(set(feature_names))
        or not np.issubdtype(matrix.dtype, np.number)
        or not np.isfinite(matrix).all()
    ):
        raise NliSingle64EvaluationError("NLI feature matrix or names are invalid")
    if _integer(manifest.get("features"), "NLI manifest features", positive=True) != len(
        feature_names
    ):
        raise NliSingle64EvaluationError("NLI manifest feature count mismatch")
    if list(candidate.get("feature_names") or []) != feature_names:
        raise NliSingle64EvaluationError("NLI candidate feature names changed")
    if list(candidate.get("expected_matrix_shape") or []) != list(matrix.shape):
        raise NliSingle64EvaluationError("NLI candidate expected matrix shape mismatch")
    if tuple(feature_names[:3]) != FLAG_FEATURE_NAMES:
        raise NliSingle64EvaluationError("NLI matrix availability flags changed")
    forbidden = [name for name in feature_names if FORBIDDEN_FEATURE_RE.search(clean(name))]
    if forbidden:
        raise NliSingle64EvaluationError(
            f"NLI feature names contain forbidden fields: {forbidden[:5]}"
        )
    for strategy, spec in normalized_candidate["zero_shot_candidates"].items():
        if spec["feature_name"] not in feature_names:
            raise NliSingle64EvaluationError(
                f"NLI matrix lacks zero-shot feature for {strategy}: "
                f"{spec['feature_name']}"
            )
    matrix_sha = freeze.array_sha256(matrix)
    names_sha = freeze.json_sha256(feature_names)
    for description, payload in (("manifest", manifest), ("candidate", candidate)):
        if clean(payload.get("matrix_sha256")) != matrix_sha:
            raise NliSingle64EvaluationError(f"NLI {description} matrix hash mismatch")
        if clean(payload.get("feature_names_sha256")) != names_sha:
            raise NliSingle64EvaluationError(
                f"NLI {description} feature-name hash mismatch"
            )
    if np.any(matrix[~selected_mask] != 0.0):
        raise NliSingle64EvaluationError(
            "NLI features outside the frozen single64 selection must be zero"
        )
    if not np.all(matrix[selected_mask, 0] == 1.0):
        raise NliSingle64EvaluationError("NLI pilot-selected flag is incomplete")
    return (
        matrix.astype(np.float64),
        feature_names,
        selected_mask,
        normalized_candidate,
        {
            "manifest": manifest,
            "candidate": candidate,
            "selection": selection,
            "preinference": preinference,
        },
    )


def _fit_random_forest(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    model = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_features=config["max_features"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        n_jobs=int(config["n_jobs"]),
        random_state=int(config["random_state"]),
    )
    model.fit(train, labels, sample_weight=weights)
    return np.asarray(model.predict_proba(test)[:, 1], dtype=np.float64)


def _fit_standardized_logistic(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)
    model = LogisticRegression(
        C=1.0,
        class_weight=None,
        solver=clean(config["solver"]),
        max_iter=int(config["max_iter"]),
        random_state=int(config["random_state"]),
    )
    model.fit(train_scaled, labels)
    return np.asarray(model.predict_proba(test_scaled)[:, 1], dtype=np.float64)


def run_grouped_oof(
    *,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    base_matrix: np.ndarray,
    nli_matrix: np.ndarray,
    selected_mask: np.ndarray,
    fold_ids: np.ndarray,
    feature_names: Sequence[str],
    model_configs: Mapping[str, Any],
    verbose: bool = False,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    labels = np.asarray(labels, dtype=np.int8)
    base_matrix = np.asarray(base_matrix, dtype=np.float64)
    nli_matrix = np.asarray(nli_matrix, dtype=np.float64)
    selected_mask = np.asarray(selected_mask, dtype=bool)
    fold_ids = np.asarray(fold_ids, dtype=np.int8)
    if (
        labels.shape != (len(records),)
        or selected_mask.shape != labels.shape
        or fold_ids.shape != labels.shape
        or base_matrix.shape[0] != len(records)
        or nli_matrix.shape != (len(records), len(feature_names))
        or not np.isfinite(base_matrix).all()
        or not np.isfinite(nli_matrix).all()
        or set(map(int, labels)) != {0, 1}
    ):
        raise NliSingle64EvaluationError("NLI grouped OOF inputs do not align")
    for refno in {clean(record["REFNO"]) for record in records}:
        assigned = {
            int(fold_ids[index])
            for index, record in enumerate(records)
            if clean(record["REFNO"]) == refno
        }
        if len(assigned) != 1:
            raise NliSingle64EvaluationError("NLI grouped OOF split a REFNO")

    flags_only = np.concatenate((base_matrix, nli_matrix[:, :3]), axis=1)
    full = np.concatenate((base_matrix, nli_matrix), axis=1)
    zero_specs = model_configs.get("zero_shot_candidates") or {}
    probability_names = (
        *RF_MODEL_NAMES,
        "selected_only_standardized_logistic",
        *zero_specs.keys(),
    )
    probabilities = {
        name: np.full(len(records), np.nan, dtype=np.float64)
        for name in probability_names
    }
    for strategy, zero_spec in zero_specs.items():
        zero_values = nli_matrix[:, feature_names.index(zero_spec["feature_name"])]
        if np.any((zero_values[selected_mask] < 0.0) | (zero_values[selected_mask] > 1.0)):
            raise NliSingle64EvaluationError("zero-shot criterion score must be in [0, 1]")
        probabilities[strategy][selected_mask] = zero_values[selected_mask]

    folds = sorted(set(map(int, fold_ids)))
    if folds != list(range(1, len(folds) + 1)):
        raise NliSingle64EvaluationError("NLI grouped fold ids are incomplete")
    fold_audit: list[dict[str, Any]] = []
    for fold in folds:
        train = np.flatnonzero(fold_ids != fold)
        test = np.flatnonzero(fold_ids == fold)
        train_labels = labels[train]
        if set(map(int, train_labels)) != {0, 1}:
            raise NliSingle64EvaluationError(f"outer training fold {fold} lacks a class")
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
        probabilities["full_nli_random_forest"][test] = _fit_random_forest(
            full[train],
            full[test],
            train_labels,
            weights,
            model_configs["random_forest"],
        )
        probabilities["pilot_routed_full_nli_random_forest"][test] = np.where(
            selected_mask[test],
            probabilities["full_nli_random_forest"][test],
            probabilities["base_random_forest"][test],
        )

        selected_train = train[selected_mask[train]]
        selected_test = test[selected_mask[test]]
        selected_train_labels = labels[selected_train]
        if len(selected_test) == 0 or set(map(int, selected_train_labels)) != {0, 1}:
            raise NliSingle64EvaluationError(
                f"selected-only logistic fold {fold} lacks selected test rows or a training class"
            )
        probabilities["selected_only_standardized_logistic"][selected_test] = (
            _fit_standardized_logistic(
                nli_matrix[selected_train],
                nli_matrix[selected_test],
                selected_train_labels,
                model_configs["selected_only_logistic"],
            )
        )
        fold_audit.append(
            {
                "outer_fold": fold,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "selected_train_rows": int(len(selected_train)),
                "selected_test_rows": int(len(selected_test)),
                "selected_train_positive_labels": int(selected_train_labels.sum()),
                "standardizer_fitted_on_selected_training_rows_only": True,
                "logistic_class_weight": None,
                "outer_test_metrics_used_for_selection": False,
                "metrics": {
                    **{
                        name: trace.metric(labels[test], probabilities[name][test])
                        for name in RF_MODEL_NAMES
                    },
                    "selected_only_standardized_logistic": trace.metric(
                        labels[selected_test],
                        probabilities["selected_only_standardized_logistic"][selected_test],
                    ),
                    **{
                        strategy: trace.metric(
                            labels[selected_test], probabilities[strategy][selected_test]
                        )
                        for strategy in zero_specs
                    },
                },
            }
        )
        if verbose:
            print(
                f"R3 NLI grouped OOF: fold={fold}/{len(folds)} "
                f"selected_train={len(selected_train)} selected_test={len(selected_test)}",
                flush=True,
            )

    for name in RF_MODEL_NAMES:
        if not np.isfinite(probabilities[name]).all():
            raise AssertionError(f"NLI RF OOF probabilities incomplete: {name}")
    if not np.isfinite(
        probabilities["selected_only_standardized_logistic"][selected_mask]
    ).all() or np.isfinite(
        probabilities["selected_only_standardized_logistic"][~selected_mask]
    ).any():
        raise AssertionError("selected-only logistic scope is invalid")
    for strategy in zero_specs:
        if (
            not np.isfinite(probabilities[strategy][selected_mask]).all()
            or np.isfinite(probabilities[strategy][~selected_mask]).any()
        ):
            raise AssertionError(f"selected zero-shot scope is invalid: {strategy}")
    if not np.array_equal(
        probabilities["pilot_routed_full_nli_random_forest"][~selected_mask],
        probabilities["base_random_forest"][~selected_mask],
    ):
        raise AssertionError("NLI route changed a nonpilot probability")
    return (
        {name: values.astype(np.float32) for name, values in probabilities.items()},
        fold_audit,
    )


def evaluation_masks(
    records: Sequence[Mapping[str, str]], selected_mask: np.ndarray
) -> dict[str, np.ndarray]:
    counts = Counter(clean(record["REFNO"]) for record in records)
    single = np.asarray(
        [counts[clean(record["REFNO"])] == 1 for record in records], dtype=bool
    )
    selected = np.asarray(selected_mask, dtype=bool)
    if np.any(selected & ~single):
        raise NliSingle64EvaluationError("selected NLI rows are not globally single-target")
    return {
        "overall": np.ones(len(records), dtype=bool),
        "pilot64": selected,
        "single_target_global": single,
    }


def paired_changes(
    labels: np.ndarray, before: np.ndarray, after: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    y = np.asarray(labels)[mask]
    old = np.asarray(before)[mask] >= THRESHOLD
    new = np.asarray(after)[mask] >= THRESHOLD
    old_correct = old == y
    new_correct = new == y
    repairs = int(np.sum(~old_correct & new_correct))
    degradations = int(np.sum(old_correct & ~new_correct))
    return {
        "rows": int(mask.sum()),
        "base_correct": int(old_correct.sum()),
        "augmented_correct": int(new_correct.sum()),
        "repairs_base_wrong_augmented_right": repairs,
        "degradations_base_right_augmented_wrong": degradations,
        "net_repairs": repairs - degradations,
        "changed_predictions": int(np.sum(old != new)),
    }


def routing_invariants(
    probabilities: Mapping[str, np.ndarray], selected_mask: np.ndarray
) -> dict[str, Any]:
    selected = np.asarray(selected_mask, dtype=bool)
    base = np.asarray(probabilities["base_random_forest"], dtype=np.float32)
    full = np.asarray(probabilities["full_nli_random_forest"], dtype=np.float32)
    routed = np.asarray(
        probabilities["pilot_routed_full_nli_random_forest"], dtype=np.float32
    )
    if not np.array_equal(routed[~selected], base[~selected]):
        raise AssertionError("NLI route changed nonpilot rows")
    if not np.array_equal(routed[selected], full[selected]):
        raise AssertionError("NLI route differs from full NLI on selected rows")
    return {
        "pilot_rows": int(selected.sum()),
        "nonpilot_rows": int((~selected).sum()),
        "pilot_probabilities_equal_full_nli": True,
        "nonpilot_probabilities_equal_base_bitwise": True,
        "maximum_nonpilot_absolute_difference_from_base": 0.0,
    }


def audit_saved_v4_random_forest(
    *,
    oof_path: Path,
    report_path: Path,
    labels: np.ndarray,
    fold_ids: np.ndarray,
    reproduced: np.ndarray,
    require_match: bool = True,
) -> dict[str, Any]:
    """Open the saved V4 RF only after fitting and audit reproduction."""

    report = json.loads(report_path.read_text(encoding="utf-8"))
    actual_sha = freeze.file_sha256(oof_path)
    if clean((report.get("output") or {}).get("sha256")) != actual_sha:
        raise NliSingle64EvaluationError("saved V4 OOF hash differs from its report")
    with np.load(oof_path) as archive:
        if not {"labels", "fold_ids", "random_forest"}.issubset(archive.files):
            raise NliSingle64EvaluationError("saved V4 OOF lacks RF audit arrays")
        saved_labels = np.asarray(archive["labels"])
        saved_folds = np.asarray(archive["fold_ids"])
        saved_rf = np.asarray(archive["random_forest"], dtype=np.float32)
    reproduced = np.asarray(reproduced, dtype=np.float32)
    if not np.array_equal(saved_labels, labels) or not np.array_equal(saved_folds, fold_ids):
        raise NliSingle64EvaluationError("saved V4 OOF identity arrays changed")
    maximum = float(
        np.max(np.abs(saved_rf.astype(np.float64) - reproduced.astype(np.float64)))
    )
    disagreements = int(np.sum((saved_rf >= 0.5) != (reproduced >= 0.5)))
    exact = bool(np.array_equal(saved_rf, reproduced))
    passed = maximum <= 1e-7 and disagreements == 0
    audit = {
        "purpose": "post_fit_reproduction_audit_only",
        "loaded_after_all_nli_models_fitted": True,
        "used_as_model_feature": False,
        "used_for_candidate_selection": False,
        "oof": {"path": str(oof_path.resolve()), "sha256": actual_sha},
        "report": {
            "path": str(report_path.resolve()),
            "sha256": freeze.file_sha256(report_path),
        },
        "array_key": "random_forest",
        "exact_probability_array_match": exact,
        "maximum_absolute_probability_difference": maximum,
        "prediction_disagreements_at_0_5": disagreements,
        "passed": passed,
    }
    if require_match and not passed:
        raise NliSingle64EvaluationError(
            "base random forest does not reproduce saved V4 OOF"
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
    ordered_names = [
        *RF_MODEL_NAMES,
        "selected_only_standardized_logistic",
        *sorted(
            set(probabilities)
            - set(RF_MODEL_NAMES)
            - {"selected_only_standardized_logistic"}
        ),
    ]
    stored = {
        name: np.asarray(probabilities[name], dtype=np.float32)
        for name in ordered_names
    }
    freeze.atomic_npz(
        oof_path,
        labels=np.asarray(labels, dtype=np.int8),
        fold_ids=np.asarray(fold_ids, dtype=np.int8),
        **stored,
    )
    report["output"] = {"path": str(oof_path), "sha256": freeze.file_sha256(oof_path)}
    freeze.atomic_json(report_path, report)

    def probability(value: float) -> float | None:
        return round(float(value), 8) if np.isfinite(value) else None

    receipt = {
        "schema_version": "r3_nli_single64_grouped_oof_receipt_v1",
        "status": "POSTHOC_EVALUATION",
        "rows": [
            {
                "row_index": index,
                "REFNO": clean(record["REFNO"]),
                "RESULTNO": clean(record["RESULTNO"]),
                "fold": int(fold_ids[index]),
                "label": int(labels[index]),
                "probabilities": {
                    name: probability(stored[name][index]) for name in ordered_names
                },
                "predictions": {
                    name: (
                        int(stored[name][index] >= THRESHOLD)
                        if np.isfinite(stored[name][index])
                        else None
                    )
                    for name in ordered_names
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
    parser.add_argument("--nli-features", type=Path, default=DEFAULT_NLI_FEATURES)
    parser.add_argument("--nli-manifest", type=Path, default=DEFAULT_NLI_MANIFEST)
    parser.add_argument("--nli-candidate", type=Path, default=DEFAULT_NLI_CANDIDATE)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--v4-oof", type=Path, default=DEFAULT_V4_OOF)
    parser.add_argument("--v4-report", type=Path, default=DEFAULT_V4_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    records, base_matrix, _, _, fold_ids, _, _ = trace.load_frozen_inputs(
        identity_path=args.identity.resolve(),
        pdf_dir=args.pdf_dir.resolve(),
        feature_path=args.v4_features.resolve(),
        manifest_path=args.v4_manifest.resolve(),
        context_path=args.v4_target_context.resolve(),
        candidate_path=args.v4_candidate.resolve(),
        fold_path=args.v4_fold_manifest.resolve(),
    )
    nli_matrix, feature_names, selected_mask, configs, payloads = load_nli_inputs(
        records=records,
        nli_path=args.nli_features,
        manifest_path=args.nli_manifest,
        candidate_path=args.nli_candidate,
        selection_path=args.selection,
        fold_ids=fold_ids,
        expected_selected_rows=EXPECTED_SELECTED_ROWS,
    )

    # First R3 access occurs only after every frozen NLI artifact is validated.
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
        nli_matrix=nli_matrix,
        selected_mask=selected_mask,
        fold_ids=fold_ids,
        feature_names=feature_names,
        model_configs=configs,
        verbose=True,
    )

    # Saved predictions are opened only now, after all fitting is complete.
    reproduction = audit_saved_v4_random_forest(
        oof_path=args.v4_oof.resolve(),
        report_path=args.v4_report.resolve(),
        labels=labels,
        fold_ids=fold_ids,
        reproduced=probabilities["base_random_forest"],
        require_match=True,
    )
    masks = evaluation_masks(records, selected_mask)
    metrics = {
        name: {
            subset: trace.metric(labels[mask], probabilities[name][mask])
            for subset, mask in masks.items()
        }
        for name in RF_MODEL_NAMES
    }
    for name in probabilities:
        if name in RF_MODEL_NAMES:
            continue
        if np.isfinite(probabilities[name][selected_mask]).all():
            metrics[name] = {
                "pilot64": trace.metric(
                    labels[selected_mask], probabilities[name][selected_mask]
                )
            }
    paired = {
        "base_vs_flags_only": {
            subset: paired_changes(
                labels,
                probabilities["base_random_forest"],
                probabilities["flags_only_random_forest"],
                mask,
            )
            for subset, mask in masks.items()
        },
        "flags_only_vs_full_nli": {
            subset: paired_changes(
                labels,
                probabilities["flags_only_random_forest"],
                probabilities["full_nli_random_forest"],
                mask,
            )
            for subset, mask in masks.items()
        },
        "base_vs_full_nli": {
            subset: paired_changes(
                labels,
                probabilities["base_random_forest"],
                probabilities["full_nli_random_forest"],
                mask,
            )
            for subset, mask in masks.items()
        },
        "base_vs_pilot_routed": {
            subset: paired_changes(
                labels,
                probabilities["base_random_forest"],
                probabilities["pilot_routed_full_nli_random_forest"],
                mask,
            )
            for subset, mask in masks.items()
        },
    }
    route_audit = routing_invariants(probabilities, selected_mask)
    report = {
        "schema_version": "r3_nli_single64_grouped_oof_report_v1",
        "status": "POSTHOC_EVALUATION_OF_FROZEN_NLI_FEATURES_AND_CANDIDATES",
        "rows": len(records),
        "papers": len({record["REFNO"] for record in records}),
        "selected_rows": int(selected_mask.sum()),
        "threshold": THRESHOLD,
        "metrics": metrics,
        "paired_changes": paired,
        "fold_audit": fold_audit,
        "routing_invariants": route_audit,
        "v4_random_forest_reproduction_audit": reproduction,
        "subset_definitions": {
            "overall": "all V4 PDF-backed rows",
            "pilot64": "the frozen label-blind single-target NLI selection",
            "single_target_global": "all rows whose REFNO occurs once in the V4 PDF-backed identity",
        },
        "subset_rows": {name: int(mask.sum()) for name, mask in masks.items()},
        "strategy_scope": {
            "random_forest_strategies": "complete 2751-row grouped OOF",
            "selected_only_standardized_logistic": "trained and evaluated only on selected rows inside each unchanged V4 fold",
            "selected_zero_shot_candidates": "all fixed candidate-declared scores evaluated only on selected rows without selection",
        },
        "model_configs_from_nli_candidate_freeze": configs,
        "nli_feature_names": feature_names,
        "boundary": {
            "nli_feature_generation_used_labels": False,
            "nli_candidate_freeze_used_labels": False,
            "r3_read_only_after_feature_candidate_and_selection_validation": True,
            "r3_used_in_outer_training_folds_only": True,
            "selected_logistic_scaler_fit_on_selected_outer_training_rows_only": True,
            "selected_logistic_class_weight": None,
            "v4_saved_rf_loaded_only_after_all_model_fits": True,
            "v4_oof_or_predictions_used_as_features": False,
            "pmagresult_or_rockmag_used": False,
            "prior_predictions_or_evaluation_reports_used_for_selection": False,
            "outer_test_refno_seen_in_training": False,
            "outer_test_metrics_used_for_model_selection": False,
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
                ("nli_features", args.nli_features),
                ("nli_manifest", args.nli_manifest),
                ("nli_candidate", args.nli_candidate),
                ("selection", args.selection),
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
            "used_only_after_all_feature_candidate_and_selection_validation": True,
        },
        "nli_freeze": {
            "feature_schema": clean(payloads["manifest"].get("schema_version")),
            "candidate_schema": clean(payloads["candidate"].get("schema_version")),
            "selection_schema": clean(payloads["selection"].get("schema_version")),
            "selection_sha256": clean(payloads["selection"].get("selection_sha256")),
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
        f"R3 NLI single64 OOF complete: pilot RF="
        f"{metrics['pilot_routed_full_nli_random_forest']['pilot64']['accuracy']:.4f}, "
        f"logistic={metrics['selected_only_standardized_logistic']['pilot64']['accuracy']:.4f}, "
        f"report={paths['report']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUXILIARY_MODEL_NAMES",
    "FLAG_FEATURE_NAMES",
    "MODEL_NAMES",
    "NliSingle64EvaluationError",
    "RF_MODEL_NAMES",
    "ZERO_SHOT_FEATURE",
    "audit_saved_v4_random_forest",
    "evaluation_masks",
    "load_nli_inputs",
    "normalize_nli_candidate",
    "paired_changes",
    "routing_invariants",
    "run_grouped_oof",
    "write_outputs",
]
