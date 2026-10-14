#!/usr/bin/env python3
"""Evaluate a low-dimensional, residual-gated R3 evidence ensemble."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_query_adaptive_retrieval_features as retrieval  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import evaluate_r3_query_adaptive_single872_oof as query_eval  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_pdf_only_strategies as pdf_text  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402


V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
QUERY_DIR = ROOT / "tmp/r3_query_adaptive_single872_v1"
TRUSTED_DIR = ROOT / "tmp/r3_strict_trusted_summary_v1"
OUTPUT_DIR = ROOT / "tmp/r3_residual_gate_single872_v1"

DEFAULT_CONFIG = ROOT / "data/knowledge/r3_residual_gate_evaluation_v1.json"
DEFAULT_DOMAIN_FEATURES = V4_DIR / "combined_features.npz"
DEFAULT_DOMAIN_MANIFEST = V4_DIR / "combined_features.manifest.json"
DEFAULT_PAGE_CACHE = ROOT / "tmp/r3_all_labelled_pdf_page_cache_v1/page_cache.jsonl"
DEFAULT_TRUSTED_FEATURES = TRUSTED_DIR / "r3_strict_trusted_summary_v1.npz"
DEFAULT_TRUSTED_MANIFEST = TRUSTED_DIR / "r3_strict_trusted_summary_v1.manifest.json"
DEFAULT_V4_FEATURES = V4_DIR / trace_freeze.FEATURE_NAME
DEFAULT_V4_MANIFEST = V4_DIR / trace_freeze.MANIFEST_NAME
DEFAULT_V4_CONTEXT = V4_DIR / trace_freeze.CONTEXT_NAME
DEFAULT_V4_CANDIDATE = V4_DIR / trace_freeze.CANDIDATE_FREEZE_NAME
DEFAULT_V4_FOLDS = V4_DIR / trace_freeze.FOLD_MANIFEST_NAME
DEFAULT_V4_OOF = V4_DIR / "trace_clean_nested_oof_v4.npz"
DEFAULT_V4_REPORT = V4_DIR / "trace_clean_nested_oof_report_v4.json"

FREEZE_NAME = "r3_residual_gate_evaluation_freeze_v1.json"
OOF_NAME = "r3_residual_gate_single872_grouped_oof_v1.npz"
RECEIPT_NAME = "r3_residual_gate_single872_grouped_oof_receipt_v1.json"
REPORT_NAME = "r3_residual_gate_single872_grouped_oof_report_v1.json"
RUN_STATE_NAME = "r3_residual_gate_evaluation_run_state_v1.json"

COMPONENT_ORDER = (
    "v4_random_forest",
    "domain_hgb",
    "domain_rf",
    "domain_extra",
    "domain_logreg",
    "domain_mean4",
    "query_pca64_logreg",
    "query_compact_hgb",
    "front_word_logreg",
    "trusted_hgb",
    "trusted_extra",
)
META_COMPONENTS = (
    "v4_random_forest",
    "domain_mean4",
    "query_pca64_logreg",
    "query_compact_hgb",
    "front_word_logreg",
    "trusted_hgb",
    "trusted_extra",
)
CANDIDATE_ORDER = (
    "domain_mean4",
    "fixed_fusion",
    "low_dimensional_stack",
    "bounded_residual_gate",
)


class ResidualGateError(ValueError):
    pass


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def file_sha256(path: Path) -> str:
    return trace_freeze.file_sha256(path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def acquire_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".r3_residual_gate_evaluation.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise ResidualGateError("another residual-gate evaluation is running") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


class RunState:
    def __init__(self, path: Path, freeze_sha256: str) -> None:
        self.path = path
        self.freeze_sha256 = freeze_sha256
        self.finished = False
        atomic_json(
            path,
            {
                "schema_version": "r3_residual_gate_evaluation_run_state_v1",
                "status": "RUNNING",
                "pid": os.getpid(),
                "evaluation_freeze_sha256": freeze_sha256,
            },
        )

    def complete(self, report: Path, oof: Path, receipt: Path) -> None:
        atomic_json(
            self.path,
            {
                "schema_version": "r3_residual_gate_evaluation_run_state_v1",
                "status": "COMPLETE",
                "evaluation_freeze_sha256": self.freeze_sha256,
                "report": {"path": str(report), "sha256": file_sha256(report)},
                "oof": {"path": str(oof), "sha256": file_sha256(oof)},
                "receipt": {"path": str(receipt), "sha256": file_sha256(receipt)},
            },
        )
        self.finished = True

    def fail(self, error: BaseException) -> None:
        if self.finished:
            return
        atomic_json(
            self.path,
            {
                "schema_version": "r3_residual_gate_evaluation_run_state_v1",
                "status": "FAILED_OR_INTERRUPTED",
                "evaluation_freeze_sha256": self.freeze_sha256,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ResidualGateError(f"{label} must be a JSON object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != "r3_residual_gate_evaluation_v1"
        or config.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
    ):
        raise ResidualGateError("residual-gate config schema changed")
    protocol = config.get("protocol") or {}
    if (
        int(protocol.get("outer_folds", -1)) != 5
        or int(protocol.get("inner_folds", -1)) != 3
        or float(protocol.get("threshold", math.nan)) != 0.5
        or protocol.get("outer_test_selection") is not False
        or tuple(config.get("candidate_order") or ()) != CANDIDATE_ORDER
    ):
        raise ResidualGateError("residual-gate protocol changed")
    domain = tuple((item.get("name"), item.get("kind"), item.get("standardized")) for item in config.get("domain_experts") or ())
    if domain != (
        ("domain_hgb", "hgb10", False),
        ("domain_rf", "rf8", False),
        ("domain_extra", "extra8", False),
        ("domain_logreg", "logreg001", True),
    ):
        raise ResidualGateError("domain expert set changed")
    weights = (config.get("fixed_fusion") or {}).get("weights") or {}
    if set(weights) != {
        "domain_mean4",
        "query_pca64_logreg",
        "front_word_logreg",
        "trusted_expert_mean",
    } or not math.isclose(sum(map(float, weights.values())), 1.0, abs_tol=1e-12):
        raise ResidualGateError("fixed fusion weights changed")
    boundary = config.get("runtime_boundary") or {}
    for name in (
        "pmagresult_or_rockmag_read",
        "prior_predictions_read_before_preaudit",
        "evaluation_results_read_before_preaudit",
        "outer_test_labels_used_for_training_or_selection",
        "selective_abstention_used",
    ):
        if boundary.get(name) is not False:
            raise ResidualGateError(f"invalid config boundary: {name}")


def load_domain_features(
    path: Path, manifest_path: Path, records: Sequence[Mapping[str, str]]
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = load_json(manifest_path, "domain feature manifest")
    if (
        manifest.get("schema_version") != "r3_all_labelled_pdf_combined_features_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("contains_labels") is not False
        or manifest.get("contains_identifiers") is not False
        or int(manifest.get("rows", -1)) != len(records)
        or auxiliary.sha256_records(records) != clean(manifest.get("alignment_sha256"))
        or file_sha256(path) != clean((manifest.get("output") or {}).get("sha256"))
    ):
        raise ResidualGateError("domain feature manifest is invalid")
    semantic_inputs = {"identity", "page_cache", "taxonomy", "positive_kb"}
    if not semantic_inputs.issubset((manifest.get("inputs") or {})):
        raise ResidualGateError("domain manifest lacks a semantic input")
    for name in sorted(semantic_inputs):
        item = manifest["inputs"][name]
        current = Path(clean(item.get("path"))).resolve()
        if not current.is_file() or file_sha256(current) != clean(item.get("sha256")):
            raise ResidualGateError(f"domain input changed: {name}")
    with np.load(path, allow_pickle=False) as archive:
        matrix = archive["combined"].astype(np.float64)
    if (
        matrix.shape != (len(records), int(manifest.get("features", -1)))
        or not np.isfinite(matrix).all()
        or auxiliary.sha256_array(matrix.astype(np.float32))
        != clean(manifest.get("matrix_sha256"))
    ):
        raise ResidualGateError("domain feature matrix changed")
    return matrix, manifest


@dataclass(frozen=True)
class TrustedSummary:
    matrix: np.ndarray
    available: np.ndarray
    counts: np.ndarray
    fraction: np.ndarray
    category_counts: np.ndarray
    source_row_indices: np.ndarray
    manifest: dict[str, Any]


def load_trusted_summary(
    path: Path,
    manifest_path: Path,
    records: Sequence[Mapping[str, str]],
    fold_ids: np.ndarray,
    single_indices: np.ndarray,
) -> TrustedSummary:
    manifest = load_json(manifest_path, "strict-trusted manifest")
    if (
        manifest.get("schema_version") != "r3_strict_trusted_summary_manifest_v1"
        or manifest.get("feature_schema_version") != "r3_strict_trusted_summary_v1"
        or manifest.get("status") != "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS"
        or int(manifest.get("bags", -1)) != len(single_indices)
        or int(manifest.get("summary_feature_count", -1)) > 64
    ):
        raise ResidualGateError("strict-trusted manifest is invalid")
    outputs = manifest.get("outputs") or {}
    expected = clean((outputs.get("features") or outputs.get("summary") or {}).get("sha256"))
    if not expected or file_sha256(path) != expected:
        raise ResidualGateError("strict-trusted summary hash changed")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "summary",
            "available",
            "source_row_indices",
            "fold_ids",
            "refnos",
            "resultnos",
            "strict_trusted_counts",
            "strict_trusted_fraction",
            "strict_trusted_category_counts",
        }
        if not required.issubset(archive.files):
            raise ResidualGateError("strict-trusted archive is incomplete")
        matrix = archive["summary"].astype(np.float64)
        available = archive["available"].astype(bool)
        source = archive["source_row_indices"].astype(np.int32)
        saved_folds = archive["fold_ids"].astype(np.int8)
        refnos = archive["refnos"].astype(str)
        resultnos = archive["resultnos"].astype(str)
        counts = archive["strict_trusted_counts"].astype(np.float64)
        fraction = archive["strict_trusted_fraction"].astype(np.float64)
        categories = archive["strict_trusted_category_counts"].astype(np.float64)
        for name, expected_hash in (manifest.get("array_sha256") or {}).items():
            if name not in archive.files or trace_freeze.array_sha256(archive[name]) != clean(
                expected_hash
            ):
                raise ResidualGateError(f"strict-trusted array changed: {name}")
    expected_refnos = np.asarray([records[int(index)]["REFNO"] for index in single_indices])
    expected_resultnos = np.asarray([records[int(index)]["RESULTNO"] for index in single_indices])
    if (
        matrix.ndim != 2
        or matrix.shape[0] != len(single_indices)
        or available.shape != (len(single_indices),)
        or counts.shape != (len(single_indices),)
        or fraction.shape != (len(single_indices),)
        or categories.shape != (len(single_indices), 4)
        or not np.array_equal(source, single_indices.astype(np.int32))
        or not np.array_equal(saved_folds, fold_ids[single_indices])
        or not np.array_equal(refnos, expected_refnos)
        or not np.array_equal(resultnos, expected_resultnos)
        or not np.isfinite(matrix).all()
        or not np.isfinite(counts).all()
        or not np.isfinite(fraction).all()
        or not np.isfinite(categories).all()
    ):
        raise ResidualGateError("strict-trusted summary does not align")
    return TrustedSummary(matrix, available, counts, fraction, categories, source, manifest)


def front_documents(
    records: Sequence[Mapping[str, str]], page_cache: Path, characters: int
) -> list[str]:
    refnos = {record["REFNO"] for record in records}
    papers = pdf_text.load_page_text(page_cache, refnos)
    return [
        f"{record['ROCKNAME']}\n{papers[record['REFNO']][:characters]}"
        for record in records
    ]


def _predict_probability(model: Any, matrix: Any) -> np.ndarray:
    with warnings.catch_warnings(), np.errstate(
        divide="ignore", over="ignore", invalid="ignore"
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        probability = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if probability.ndim != 1 or not np.isfinite(probability).all():
        raise AssertionError("model produced invalid probabilities")
    return probability


def fit_front_text(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, str]],
    documents: Sequence[str],
    spec: Mapping[str, Any],
) -> np.ndarray:
    vectorizer = TfidfVectorizer(
        analyzer=clean(spec["analyzer"]),
        ngram_range=tuple(map(int, spec["ngram_range"])),
        min_df=int(spec["min_df"]),
        max_df=float(spec["max_df"]),
        max_features=int(spec["max_features"]),
        sublinear_tf=bool(spec["sublinear_tf"]),
        strip_accents=clean(spec["strip_accents"]),
        dtype=np.float32,
    )
    fit = vectorizer.fit_transform([documents[int(index)] for index in train])
    predict = vectorizer.transform([documents[int(index)] for index in test])
    model = LogisticRegression(
        C=float(spec["C"]),
        solver=clean(spec["solver"]),
        max_iter=int(spec["max_iter"]),
        random_state=35_720_260,
    )
    model.fit(
        fit,
        labels[train],
        sample_weight=trace_eval.inverse_identity_weights(train, records),
    )
    return _predict_probability(model, predict)


def fit_trusted_experts(
    train_global: np.ndarray,
    test_global: np.ndarray,
    labels: np.ndarray,
    global_to_single: np.ndarray,
    trusted: TrustedSummary,
    spec: Mapping[str, Any],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    train = global_to_single[train_global]
    test = global_to_single[test_global]
    if (train < 0).any() or (test < 0).any():
        raise AssertionError("trusted expert received a non-single-target row")
    hgb_spec = spec["hist_gradient_boosting"]
    hgb = HistGradientBoostingClassifier(
        max_iter=int(hgb_spec["max_iter"]),
        learning_rate=float(hgb_spec["learning_rate"]),
        max_leaf_nodes=int(hgb_spec["max_leaf_nodes"]),
        min_samples_leaf=int(hgb_spec["min_samples_leaf"]),
        l2_regularization=float(hgb_spec["l2_regularization"]),
        random_state=random_state,
    )
    extra_spec = spec["extra_trees"]
    extra = ExtraTreesClassifier(
        n_estimators=int(extra_spec["n_estimators"]),
        max_features=float(extra_spec["max_features"]),
        min_samples_leaf=int(extra_spec["min_samples_leaf"]),
        n_jobs=int(extra_spec["n_jobs"]),
        random_state=random_state,
    )
    hgb.fit(trusted.matrix[train], labels[train_global])
    extra.fit(trusted.matrix[train], labels[train_global])
    hgb_probability = _predict_probability(hgb, trusted.matrix[test])
    extra_probability = _predict_probability(extra, trusted.matrix[test])
    unavailable = ~trusted.available[test]
    fallback = float(spec["unavailable_probability"])
    hgb_probability[unavailable] = fallback
    extra_probability[unavailable] = fallback
    return hgb_probability, extra_probability


def domain_specs(config: Mapping[str, Any]) -> tuple[nested.ExpertSpec, ...]:
    return tuple(
        nested.ExpertSpec(
            clean(item["name"]),
            "combined",
            clean(item["kind"]),
            standardized=bool(item["standardized"]),
        )
        for item in config["domain_experts"]
    )


def fit_components(
    *,
    all_train: np.ndarray,
    single_train: np.ndarray,
    test: np.ndarray,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    v4: np.ndarray,
    domain: np.ndarray,
    query_matrices: Mapping[str, np.ndarray],
    query_config: Mapping[str, Any],
    documents: Sequence[str],
    trusted: TrustedSummary,
    global_to_single: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    random_state = int(config["protocol"]["random_state"])
    all_weights = trace_eval.inverse_identity_weights(all_train, records)
    v4_spec = query_eval.RawSpec("v4_random_forest", "random_forest", "v4")
    q_pca_config = config["query_experts"]["pca"]
    q_pca_spec = query_eval.RawSpec(
        clean(q_pca_config["name"]),
        clean(q_pca_config["family"]),
        clean(q_pca_config["matrix"]),
        c_value=float(q_pca_config["C"]),
        pca_dimension=int(q_pca_config["pca_dimension"]),
    )
    q_hgb_config = config["query_experts"]["compact_hgb"]
    q_hgb_spec = query_eval.RawSpec(
        clean(q_hgb_config["name"]),
        clean(q_hgb_config["family"]),
        clean(q_hgb_config["matrix"]),
    )
    output = {
        "v4_random_forest": query_eval.fit_predict(
            v4_spec, v4, all_train, test, labels, all_weights, query_config
        ),
        "query_pca64_logreg": query_eval.fit_predict(
            q_pca_spec,
            query_matrices[q_pca_spec.matrix],
            all_train,
            test,
            labels,
            all_weights,
            query_config,
        ),
        "query_compact_hgb": query_eval.fit_predict(
            q_hgb_spec,
            query_matrices[q_hgb_spec.matrix],
            all_train,
            test,
            labels,
            all_weights,
            query_config,
        ),
        "front_word_logreg": fit_front_text(
            all_train,
            test,
            labels,
            records,
            documents,
            config["front_text_expert"],
        ),
    }
    for expert in domain_specs(config):
        fitted = nested.fit_expert(expert, domain, all_train, labels, all_weights)
        output[expert.name] = fitted.predict(domain[test])
    output["domain_mean4"] = np.column_stack(
        [output[name] for name in ("domain_hgb", "domain_rf", "domain_extra", "domain_logreg")]
    ).mean(axis=1)
    output["trusted_hgb"], output["trusted_extra"] = fit_trusted_experts(
        single_train,
        test,
        labels,
        global_to_single,
        trusted,
        config["trusted_summary_experts"],
        random_state,
    )
    if set(output) != set(COMPONENT_ORDER) or any(
        values.shape != (len(test),) or not np.isfinite(values).all()
        for values in output.values()
    ):
        raise AssertionError("component prediction contract changed")
    return output


def probability_logit(value: np.ndarray, clip: float) -> np.ndarray:
    probability = np.clip(np.asarray(value, dtype=np.float64), clip, 1.0 - clip)
    return np.log(probability / (1.0 - probability))


def trusted_meta_features(
    trusted: TrustedSummary, global_indices: np.ndarray, global_to_single: np.ndarray
) -> np.ndarray:
    local = global_to_single[global_indices]
    if (local < 0).any():
        raise AssertionError("meta features received a non-single-target row")
    category_log = np.log1p(trusted.category_counts[local]) / math.log(65.0)
    return np.column_stack(
        (
            trusted.available[local].astype(np.float64),
            np.log1p(trusted.counts[local]) / math.log(65.0),
            trusted.fraction[local],
            category_log,
            (trusted.category_counts[local] > 0).mean(axis=1),
        )
    )


def component_meta_features(
    components: Mapping[str, np.ndarray], trusted_meta: np.ndarray, clip: float
) -> np.ndarray:
    probabilities = np.column_stack([components[name] for name in META_COMPONENTS])
    logits = probability_logit(probabilities, clip)
    decisions = probabilities >= 0.5
    summary = np.column_stack(
        (
            probabilities.mean(axis=1),
            probabilities.std(axis=1),
            probabilities.min(axis=1),
            probabilities.max(axis=1),
            decisions.mean(axis=1),
            np.abs(probabilities - 0.5).mean(axis=1),
            probabilities[:, 1] - probabilities[:, 0],
            probabilities[:, 2] - probabilities[:, 1],
            probabilities[:, 4:7].std(axis=1),
        )
    )
    matrix = np.column_stack((logits, summary, trusted_meta))
    if matrix.ndim != 2 or matrix.shape[1] != 24 or not np.isfinite(matrix).all():
        raise AssertionError("meta feature contract changed")
    return matrix


@dataclass
class LogisticFit:
    model: LogisticRegression | None
    scaler: StandardScaler | None
    constant: float | None = None


def fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    spec: Mapping[str, Any],
) -> LogisticFit:
    if len(matrix) != len(labels) or not len(matrix):
        return LogisticFit(None, None, constant=0.0)
    unique = np.unique(labels)
    if len(unique) < 2:
        return LogisticFit(None, None, constant=float(unique[0]))
    scaler = StandardScaler() if bool(spec.get("standardize")) else None
    fit = np.asarray(matrix, dtype=np.float64)
    if scaler is not None:
        fit = np.clip(scaler.fit_transform(fit), -12.0, 12.0)
    model = LogisticRegression(
        C=float(spec["C"]),
        solver=clean(spec["solver"]),
        max_iter=int(spec["max_iter"]),
        random_state=35_720_260,
    )
    model.fit(fit, labels)
    return LogisticFit(model, scaler)


def predict_logistic(fitted: LogisticFit, matrix: np.ndarray) -> np.ndarray:
    if fitted.model is None:
        return np.full(len(matrix), float(fitted.constant or 0.0), dtype=np.float64)
    predict = np.asarray(matrix, dtype=np.float64)
    if fitted.scaler is not None:
        predict = np.clip(fitted.scaler.transform(predict), -12.0, 12.0)
    return _predict_probability(fitted.model, predict)


def fixed_fusion(
    components: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> np.ndarray:
    weights = config["fixed_fusion"]["weights"]
    trusted_mean = (components["trusted_hgb"] + components["trusted_extra"]) / 2.0
    output = (
        float(weights["domain_mean4"]) * components["domain_mean4"]
        + float(weights["query_pca64_logreg"]) * components["query_pca64_logreg"]
        + float(weights["front_word_logreg"]) * components["front_word_logreg"]
        + float(weights["trusted_expert_mean"]) * trusted_mean
    )
    return np.asarray(output, dtype=np.float64)


def evidence_probability(
    components: Mapping[str, np.ndarray], gate_spec: Mapping[str, Any]
) -> np.ndarray:
    values = np.column_stack(
        [components[clean(name)] for name in gate_spec["evidence_components"]]
    )
    if gate_spec.get("evidence_aggregation") != "median":
        raise ResidualGateError("unsupported evidence aggregation")
    return np.median(values, axis=1)


@dataclass
class GateFit:
    fitted: LogisticFit | None
    enabled: bool
    disagreement_rows: int


def fit_gate(
    meta: np.ndarray,
    base: np.ndarray,
    evidence: np.ndarray,
    labels: np.ndarray,
    spec: Mapping[str, Any],
) -> GateFit:
    disagreement = (base >= 0.5) != (evidence >= 0.5)
    indices = np.flatnonzero(disagreement)
    target = ((evidence[indices] >= 0.5) == labels[indices]).astype(np.int8)
    minimum = int(spec["minimum_disagreement_rows"])
    if len(indices) < minimum or len(np.unique(target)) < 2:
        return GateFit(None, False, len(indices))
    return GateFit(fit_logistic(meta[indices], target, spec), True, len(indices))


def apply_gate(
    gate: GateFit,
    meta: np.ndarray,
    base: np.ndarray,
    evidence: np.ndarray,
    trusted_available: np.ndarray,
    spec: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output = np.asarray(base, dtype=np.float64).copy()
    quality = (
        predict_logistic(gate.fitted, meta)
        if gate.enabled and gate.fitted is not None
        else np.zeros(len(base), dtype=np.float64)
    )
    disagreement = (base >= 0.5) != (evidence >= 0.5)
    allowed = disagreement & (
        np.abs(base - 0.5) <= float(spec["maximum_base_margin_for_override"])
    )
    if bool(spec.get("require_trusted_summary")):
        allowed &= trusted_available
    switched = allowed & (quality >= float(spec["gate_probability_threshold"]))
    step = np.clip(
        evidence - base,
        -float(spec["maximum_probability_step"]),
        float(spec["maximum_probability_step"]),
    )
    output[switched] = base[switched] + step[switched]
    return np.clip(output, 0.0, 1.0), switched, quality


def metric(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    return trace_eval.metric(np.asarray(labels), np.asarray(probabilities))


def better_metric(
    labels: np.ndarray, probabilities: np.ndarray, stable_rank: int
) -> tuple[float, float, float, int]:
    value = metric(labels, probabilities)
    return (
        float(value["accuracy"]),
        float(value["macro_f1"]),
        float(value["balanced_accuracy"]),
        -stable_rank,
    )


def choose_candidate(
    labels: np.ndarray,
    candidates: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    order = tuple(config["candidate_order"])
    if order != CANDIDATE_ORDER or set(candidates) != set(order):
        raise AssertionError("candidate set changed")
    ranked = sorted(
        order,
        key=lambda name: better_metric(labels, candidates[name], order.index(name)),
        reverse=True,
    )
    selected = ranked[0]
    baseline = metric(labels, candidates["domain_mean4"])
    selected_metric = metric(labels, candidates[selected])
    gain = float(selected_metric["accuracy"] - baseline["accuracy"])
    minimum = float(config["protocol"]["minimum_accuracy_gain_over_domain_baseline"])
    accepted = selected != "domain_mean4" and gain >= minimum
    if not accepted:
        selected = "domain_mean4"
    audit = {
        "unconstrained_best": ranked[0],
        "selected": selected,
        "accepted_non_baseline": accepted,
        "accuracy_gain_over_domain_baseline": round(gain, 8),
        "minimum_accuracy_gain": minimum,
    }
    ranked_metrics = [
        {"name": name, "metrics": metric(labels, candidates[name])} for name in ranked
    ]
    return selected, ranked_metrics, audit


def assemble_candidates(
    components: Mapping[str, np.ndarray],
    meta: np.ndarray,
    trusted_available: np.ndarray,
    stack: LogisticFit,
    gate: GateFit,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    evidence = evidence_probability(components, config["residual_gate"])
    gated, switched, gate_quality = apply_gate(
        gate,
        meta,
        components["domain_mean4"],
        evidence,
        trusted_available,
        config["residual_gate"],
    )
    candidates = {
        "domain_mean4": components["domain_mean4"],
        "fixed_fusion": fixed_fusion(components, config),
        "low_dimensional_stack": predict_logistic(stack, meta),
        "bounded_residual_gate": gated,
    }
    return candidates, {
        "evidence_probability": evidence,
        "gate_switched": switched,
        "gate_quality": gate_quality,
    }


def crossfit_meta_candidates(
    outer_train: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    components: Mapping[str, np.ndarray],
    meta: np.ndarray,
    trusted_available: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], LogisticFit, GateFit]:
    local = {int(index): position for position, index in enumerate(outer_train)}
    stack_oof = np.full(len(outer_train), np.nan, dtype=np.float64)
    gate_oof = np.full(len(outer_train), np.nan, dtype=np.float64)
    for fit_global, validation_global in inner_splits:
        fit = np.asarray([local[int(index)] for index in fit_global], dtype=np.int64)
        validation = np.asarray(
            [local[int(index)] for index in validation_global], dtype=np.int64
        )
        stack_fit = fit_logistic(meta[fit], labels[fit_global], config["stack"])
        stack_oof[validation] = predict_logistic(stack_fit, meta[validation])
        base = components["domain_mean4"]
        evidence = evidence_probability(components, config["residual_gate"])
        gate_fit = fit_gate(
            meta[fit], base[fit], evidence[fit], labels[fit_global], config["residual_gate"]
        )
        gated, _, _ = apply_gate(
            gate_fit,
            meta[validation],
            base[validation],
            evidence[validation],
            trusted_available[validation],
            config["residual_gate"],
        )
        gate_oof[validation] = gated
    if not np.isfinite(stack_oof).all() or not np.isfinite(gate_oof).all():
        raise AssertionError("cross-fitted meta predictions are incomplete")
    fixed = fixed_fusion(components, config)
    candidates = {
        "domain_mean4": components["domain_mean4"],
        "fixed_fusion": fixed,
        "low_dimensional_stack": stack_oof,
        "bounded_residual_gate": gate_oof,
    }
    final_stack = fit_logistic(meta, labels[outer_train], config["stack"])
    final_gate = fit_gate(
        meta,
        components["domain_mean4"],
        evidence_probability(components, config["residual_gate"]),
        labels[outer_train],
        config["residual_gate"],
    )
    return candidates, final_stack, final_gate


def compare_to_v4(
    provisional: Path,
    v4_oof: Path,
    v4_report: Path,
    single_indices: np.ndarray,
) -> dict[str, Any]:
    with np.load(provisional, allow_pickle=False) as new, np.load(
        v4_oof, allow_pickle=False
    ) as saved:
        expected_labels = saved["labels"][single_indices]
        expected_folds = saved["fold_ids"][single_indices]
        expected_probability = saved["random_forest"][single_indices]
        if (
            not np.array_equal(new["labels"], expected_labels)
            or not np.array_equal(new["fold_ids"], expected_folds)
            or not np.array_equal(new["source_row_indices"], single_indices)
            or not np.array_equal(
                new["component__v4_random_forest"].astype(np.float32),
                expected_probability,
            )
        ):
            raise ResidualGateError("strict V4 reproduction failed")
    report = load_json(v4_report, "V4 report")
    expected_metric = report["metrics"]["random_forest"]["single_target_paper"]
    observed_metric = metric(expected_labels, expected_probability)
    if observed_metric != expected_metric:
        raise ResidualGateError("strict V4 metric reproduction failed")
    return {
        "saved_oof": {"path": str(v4_oof), "sha256": file_sha256(v4_oof)},
        "saved_report": {"path": str(v4_report), "sha256": file_sha256(v4_report)},
        "float32_probability_exact": True,
        "labels_exact": True,
        "folds_exact": True,
        "metrics_exact": True,
    }


def delta_record(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    base_correct = (baseline >= 0.5) == labels
    new_correct = (candidate >= 0.5) == labels
    repairs = int((~base_correct & new_correct).sum())
    degradations = int((base_correct & ~new_correct).sum())
    return {
        "repairs": repairs,
        "degradations": degradations,
        "net_repairs": repairs - degradations,
        "accuracy_delta": round(float(new_correct.mean() - base_correct.mean()), 8),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--domain-features", type=Path, default=DEFAULT_DOMAIN_FEATURES)
    parser.add_argument("--domain-manifest", type=Path, default=DEFAULT_DOMAIN_MANIFEST)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--trusted-features", type=Path, default=DEFAULT_TRUSTED_FEATURES)
    parser.add_argument("--trusted-manifest", type=Path, default=DEFAULT_TRUSTED_MANIFEST)
    parser.add_argument("--v4-features", type=Path, default=DEFAULT_V4_FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4-context", type=Path, default=DEFAULT_V4_CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=DEFAULT_V4_CANDIDATE)
    parser.add_argument("--v4-folds", type=Path, default=DEFAULT_V4_FOLDS)
    parser.add_argument("--v4-oof", type=Path, default=DEFAULT_V4_OOF)
    parser.add_argument("--v4-report", type=Path, default=DEFAULT_V4_REPORT)
    parser.add_argument("--query-dir", type=Path, default=QUERY_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    lock = acquire_lock(output_dir)
    run_state: RunState | None = None
    try:
        config = load_json(args.config.resolve(), "residual-gate config")
        validate_config(config)
        (
            records,
            v4,
            _v4_names,
            _context_documents,
            fold_ids,
            document_groups,
            _v4_candidate,
        ) = trace_eval.load_frozen_inputs(
            identity_path=args.identity.resolve(),
            pdf_dir=args.pdf_dir.resolve(),
            feature_path=args.v4_features.resolve(),
            manifest_path=args.v4_manifest.resolve(),
            context_path=args.v4_context.resolve(),
            candidate_path=args.v4_candidate.resolve(),
            fold_path=args.v4_folds.resolve(),
        )
        complete_identity = trace_freeze.load_three_column_identity(args.identity.resolve())
        multiplicity = Counter(record["REFNO"] for record in complete_identity)
        single_mask = np.asarray(
            [multiplicity[record["REFNO"]] == 1 for record in records], dtype=bool
        )
        single_indices = np.flatnonzero(single_mask)
        if len(single_indices) != 872:
            raise ResidualGateError("single-target scope changed")
        compact, pooled, query_single, query_manifest, query_candidate = query_eval.load_query_features(
            query_dir=args.query_dir.resolve(),
            records=records,
            v4_fold_ids=fold_ids,
            expected_single_mask=single_mask,
        )
        if not np.array_equal(query_single, single_mask):
            raise ResidualGateError("query single-target scope changed")
        query_matrices = query_eval.build_matrices(v4, compact, pooled)
        domain, domain_manifest = load_domain_features(
            args.domain_features.resolve(), args.domain_manifest.resolve(), records
        )
        trusted = load_trusted_summary(
            args.trusted_features.resolve(),
            args.trusted_manifest.resolve(),
            records,
            fold_ids,
            single_indices,
        )
        page_cache = args.page_cache.resolve()
        if (
            not page_cache.is_file()
            or file_sha256(page_cache)
            != clean((domain_manifest.get("inputs") or {}).get("page_cache", {}).get("sha256"))
        ):
            raise ResidualGateError("front-text page cache changed")
        documents = front_documents(
            records, page_cache, int(config["front_text_expert"]["characters"])
        )
        global_to_single = np.full(len(records), -1, dtype=np.int32)
        global_to_single[single_indices] = np.arange(len(single_indices), dtype=np.int32)
        query_config = query_candidate["evaluation_candidates"]

        freeze_path = output_dir / FREEZE_NAME
        freeze = {
            "schema_version": "r3_residual_gate_evaluation_freeze_v1",
            "status": "FROZEN_BEFORE_LABEL_ACCESS",
            "scope": "872 PDF-backed single-target papers at full coverage",
            "config": config,
            "component_order": list(COMPONENT_ORDER),
            "meta_components": list(META_COMPONENTS),
            "candidate_order": list(CANDIDATE_ORDER),
            "inputs": {
                name: {"path": str(path), "sha256": file_sha256(path)}
                for name, path in (
                    ("config", args.config.resolve()),
                    ("domain_features", args.domain_features.resolve()),
                    ("domain_manifest", args.domain_manifest.resolve()),
                    ("page_cache", page_cache),
                    ("trusted_features", args.trusted_features.resolve()),
                    ("trusted_manifest", args.trusted_manifest.resolve()),
                    ("v4_features", args.v4_features.resolve()),
                    ("v4_manifest", args.v4_manifest.resolve()),
                    ("v4_context", args.v4_context.resolve()),
                    ("v4_candidate", args.v4_candidate.resolve()),
                    ("v4_folds", args.v4_folds.resolve()),
                    ("query_features", args.query_dir.resolve() / retrieval.FEATURE_NAME),
                    ("query_manifest", args.query_dir.resolve() / retrieval.MANIFEST_NAME),
                    ("query_candidate", args.query_dir.resolve() / retrieval.CANDIDATE_NAME),
                    ("evaluator_code", Path(__file__).resolve()),
                )
            },
            "runtime_boundary": {
                "labels_read": False,
                "pmagresult_or_rockmag_read": False,
                "prior_predictions_read": False,
                "evaluation_results_read": False,
                "outer_test_labels_used_for_training_or_selection": False,
                "trusted_summary_generated_without_labels": True,
            },
        }
        atomic_json(freeze_path, freeze)
        run_state = RunState(output_dir / RUN_STATE_NAME, file_sha256(freeze_path))

        # First label access in this process.
        label_map = trace_eval.load_labels(
            args.targets.resolve(), {record["RESULTNO"] for record in records}
        )
        labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
        component_oof = {
            name: np.full(len(single_indices), np.nan, dtype=np.float64)
            for name in COMPONENT_ORDER
        }
        candidate_oof = {
            name: np.full(len(single_indices), np.nan, dtype=np.float64)
            for name in CANDIDATE_ORDER
        }
        primary = np.full(len(single_indices), np.nan, dtype=np.float64)
        gate_switched = np.zeros(len(single_indices), dtype=bool)
        gate_quality = np.zeros(len(single_indices), dtype=np.float64)
        selected_names = np.empty(len(single_indices), dtype="<U64")
        fold_audit: list[dict[str, Any]] = []

        for outer_fold in range(1, 6):
            outer_test = np.flatnonzero(single_mask & (fold_ids == outer_fold))
            outer_all_train = np.flatnonzero(fold_ids != outer_fold)
            outer_single_train = np.flatnonzero(single_mask & (fold_ids != outer_fold))
            test_groups = set(map(clean, document_groups[outer_test]))
            if test_groups & set(map(clean, document_groups[outer_all_train])):
                raise AssertionError("outer fold leaks a normalized document")
            inner_splits = nested.deterministic_group_folds(
                outer_single_train,
                [clean(document_groups[index]) for index in outer_single_train],
                n_splits=3,
                seed=int(config["protocol"]["inner_seed"]) + outer_fold,
            )
            inner_components = {
                name: np.full(len(outer_single_train), np.nan, dtype=np.float64)
                for name in COMPONENT_ORDER
            }
            local = {int(index): position for position, index in enumerate(outer_single_train)}
            for inner_train_single, inner_validation in inner_splits:
                validation_groups = set(map(clean, document_groups[inner_validation]))
                inner_train_all = np.asarray(
                    [
                        index
                        for index in outer_all_train
                        if clean(document_groups[index]) not in validation_groups
                    ],
                    dtype=np.int64,
                )
                predicted = fit_components(
                    all_train=inner_train_all,
                    single_train=inner_train_single,
                    test=inner_validation,
                    records=records,
                    labels=labels,
                    v4=v4,
                    domain=domain,
                    query_matrices=query_matrices,
                    query_config=query_config,
                    documents=documents,
                    trusted=trusted,
                    global_to_single=global_to_single,
                    config=config,
                )
                validation_local = np.asarray(
                    [local[int(index)] for index in inner_validation], dtype=np.int64
                )
                for name, values in predicted.items():
                    inner_components[name][validation_local] = values
            if any(not np.isfinite(value).all() for value in inner_components.values()):
                raise AssertionError("inner component OOF is incomplete")
            inner_trusted_meta = trusted_meta_features(
                trusted, outer_single_train, global_to_single
            )
            inner_meta = component_meta_features(
                inner_components, inner_trusted_meta, float(config["stack"]["probability_clip"])
            )
            inner_candidates, final_stack, final_gate = crossfit_meta_candidates(
                outer_single_train,
                inner_splits,
                inner_components,
                inner_meta,
                trusted.available[global_to_single[outer_single_train]],
                labels,
                config,
            )
            selected, ranked, selection_audit = choose_candidate(
                labels[outer_single_train], inner_candidates, config
            )
            outer_components = fit_components(
                all_train=outer_all_train,
                single_train=outer_single_train,
                test=outer_test,
                records=records,
                labels=labels,
                v4=v4,
                domain=domain,
                query_matrices=query_matrices,
                query_config=query_config,
                documents=documents,
                trusted=trusted,
                global_to_single=global_to_single,
                config=config,
            )
            outer_trusted_meta = trusted_meta_features(trusted, outer_test, global_to_single)
            outer_meta = component_meta_features(
                outer_components, outer_trusted_meta, float(config["stack"]["probability_clip"])
            )
            outer_candidates, gate_details = assemble_candidates(
                outer_components,
                outer_meta,
                trusted.available[global_to_single[outer_test]],
                final_stack,
                final_gate,
                config,
            )
            test_local = global_to_single[outer_test]
            for name, values in outer_components.items():
                component_oof[name][test_local] = values
            for name, values in outer_candidates.items():
                candidate_oof[name][test_local] = values
            primary[test_local] = outer_candidates[selected]
            selected_names[test_local] = selected
            gate_switched[test_local] = gate_details["gate_switched"]
            gate_quality[test_local] = gate_details["gate_quality"]
            fold_metric = metric(labels[outer_test], primary[test_local])
            fold_audit.append(
                {
                    "outer_fold": outer_fold,
                    "test_rows": len(outer_test),
                    "all_pdf_training_rows": len(outer_all_train),
                    "single_training_rows": len(outer_single_train),
                    "selected": selected,
                    "inner_ranked_candidates": ranked,
                    "selection_guard": selection_audit,
                    "gate_training_enabled": final_gate.enabled,
                    "gate_training_disagreement_rows": final_gate.disagreement_rows,
                    "gate_test_switches": int(gate_details["gate_switched"].sum()),
                    "outer_test_metrics": fold_metric,
                }
            )
            print(
                f"residual gate OOF: fold={outer_fold}/5 selected={selected} "
                f"accuracy={fold_metric['accuracy']:.4f}",
                flush=True,
            )

        if (
            not np.isfinite(primary).all()
            or any(not np.isfinite(value).all() for value in component_oof.values())
            or any(not np.isfinite(value).all() for value in candidate_oof.values())
            or (selected_names == "").any()
        ):
            raise AssertionError("outer OOF predictions are incomplete")
        arrays: dict[str, np.ndarray] = {
            "source_row_indices": single_indices.astype(np.int32),
            "labels": labels[single_indices].astype(np.int8),
            "fold_ids": fold_ids[single_indices].astype(np.int8),
            "nested_primary": primary.astype(np.float64),
            "selected_candidate": selected_names,
            "gate_switched": gate_switched,
            "gate_quality": gate_quality.astype(np.float64),
        }
        arrays.update(
            {f"component__{name}": value.astype(np.float64) for name, value in component_oof.items()}
        )
        arrays.update(
            {f"candidate__{name}": value.astype(np.float64) for name, value in candidate_oof.items()}
        )
        provisional = output_dir / (OOF_NAME + ".preaudit")
        provisional.unlink(missing_ok=True)
        atomic_npz(provisional, **arrays)

        v4_audit = compare_to_v4(
            provisional,
            args.v4_oof.resolve(),
            args.v4_report.resolve(),
            single_indices,
        )
        oof_path = output_dir / OOF_NAME
        provisional.replace(oof_path)
        with np.load(oof_path, allow_pickle=False) as frozen:
            frozen_labels = frozen["labels"].copy()
            frozen_primary = frozen["nested_primary"].copy()
            frozen_v4 = frozen["component__v4_random_forest"].copy()
            frozen_domain = frozen["component__domain_mean4"].copy()
            frozen_candidates = {
                name: frozen[f"candidate__{name}"].copy() for name in CANDIDATE_ORDER
            }
        primary_metric = metric(frozen_labels, frozen_primary)
        v4_metric = metric(frozen_labels, frozen_v4)
        domain_metric = metric(frozen_labels, frozen_domain)

        receipt_rows = []
        for local_index, global_index in enumerate(single_indices):
            receipt_rows.append(
                {
                    "row": local_index,
                    "source_row_index": int(global_index),
                    "REFNO": records[int(global_index)]["REFNO"],
                    "RESULTNO": records[int(global_index)]["RESULTNO"],
                    "ROCKNAME": records[int(global_index)]["ROCKNAME"],
                    "fold": int(fold_ids[int(global_index)]),
                    "label": int(labels[int(global_index)]),
                    "selected_candidate": clean(selected_names[local_index]),
                    "probability": float(frozen_primary[local_index]),
                    "prediction": int(frozen_primary[local_index] >= 0.5),
                    "v4_probability": float(frozen_v4[local_index]),
                    "domain_probability": float(frozen_domain[local_index]),
                    "gate_switched": bool(gate_switched[local_index]),
                    "gate_quality": float(gate_quality[local_index]),
                    "trusted_summary_available": bool(trusted.available[local_index]),
                    "strict_trusted_count": int(trusted.counts[local_index]),
                }
            )
        receipt_path = output_dir / RECEIPT_NAME
        atomic_json(
            receipt_path,
            {
                "schema_version": "r3_residual_gate_single872_grouped_oof_receipt_v1",
                "rows": receipt_rows,
            },
        )
        report_path = output_dir / REPORT_NAME
        report = {
            "schema_version": "r3_residual_gate_single872_grouped_oof_report_v1",
            "status": "POSTHOC_EVALUATION_OF_FROZEN_FEATURES_AND_NESTED_GATE",
            "scope": {
                "rows": len(single_indices),
                "coverage": 1.0,
                "skipped_without_pdf_not_in_scope": True,
            },
            "primary_strategy": "outer-train-selected low-dimensional residual candidate",
            "threshold": 0.5,
            "metrics": {
                "strict_v4_random_forest": v4_metric,
                "domain_mean4": domain_metric,
                "nested_primary": primary_metric,
                "fixed_candidates": {
                    name: metric(frozen_labels, probability)
                    for name, probability in frozen_candidates.items()
                },
            },
            "delta_vs_strict_v4": delta_record(frozen_labels, frozen_v4, frozen_primary),
            "delta_vs_domain_mean4": delta_record(
                frozen_labels, frozen_domain, frozen_primary
            ),
            "fold_audit": fold_audit,
            "selected_candidate_counts": dict(Counter(map(clean, selected_names))),
            "gate_switches": int(gate_switched.sum()),
            "v4_reproduction_audit": v4_audit,
            "boundary": {
                "feature_generation_used_labels": False,
                "pmagresult_or_rockmag_read_by_evaluator": False,
                "existing_knowledge_features_disclosed": True,
                "all_experts_fit_inside_outer_or_inner_training_partitions": True,
                "meta_models_selected_with_cross_fitted_outer_train_predictions": True,
                "outer_test_labels_used_for_model_or_selection": False,
                "prior_oof_opened_only_after_provisional_predictions_written": True,
                "selective_abstention_used": False,
                "development_internal_oof_not_external_validation": True,
            },
            "promotion_decision": {
                "reaches_90_percent_full_coverage_internal_oof": primary_metric["accuracy"] >= 0.90,
                "improves_strict_v4": primary_metric["accuracy"] > v4_metric["accuracy"],
                "promote_without_external_validation": False,
                "reason": "The 872-paper corpus has already informed method development; a locked external set remains required.",
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            "outputs": {
                "oof": {"path": str(oof_path), "sha256": file_sha256(oof_path)},
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": file_sha256(receipt_path),
                },
            },
        }
        atomic_json(report_path, report)
        run_state.complete(report_path, oof_path, receipt_path)
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "strict_v4": v4_metric,
                    "domain_mean4": domain_metric,
                    "nested_primary": primary_metric,
                    "delta_vs_v4": report["delta_vs_strict_v4"],
                    "promotion": report["promotion_decision"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        if run_state is not None:
            run_state.fail(error)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CANDIDATE_ORDER",
    "COMPONENT_ORDER",
    "GateFit",
    "LogisticFit",
    "ResidualGateError",
    "apply_gate",
    "choose_candidate",
    "component_meta_features",
    "evidence_probability",
    "fit_gate",
    "fixed_fusion",
    "probability_logit",
    "validate_config",
]
