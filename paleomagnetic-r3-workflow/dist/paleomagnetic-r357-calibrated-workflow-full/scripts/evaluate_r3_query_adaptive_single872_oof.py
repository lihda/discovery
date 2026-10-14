#!/usr/bin/env python3
"""Nested grouped OOF evaluation of frozen query-adaptive R3 retrieval."""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import sklearn
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_query_adaptive_retrieval_features as retrieval  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402


V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
DEFAULT_QUERY_DIR = ROOT / "tmp/r3_query_adaptive_single872_v1"
DEFAULT_V4_FEATURES = V4_DIR / trace_freeze.FEATURE_NAME
DEFAULT_V4_MANIFEST = V4_DIR / trace_freeze.MANIFEST_NAME
DEFAULT_V4_CONTEXT = V4_DIR / trace_freeze.CONTEXT_NAME
DEFAULT_V4_CANDIDATE = V4_DIR / trace_freeze.CANDIDATE_FREEZE_NAME
DEFAULT_V4_FOLDS = V4_DIR / trace_freeze.FOLD_MANIFEST_NAME
DEFAULT_V4_OOF = V4_DIR / "trace_clean_nested_oof_v4.npz"
DEFAULT_V4_REPORT = V4_DIR / "trace_clean_nested_oof_report_v4.json"

EVALUATION_FREEZE_NAME = "r3_query_adaptive_evaluation_freeze_v1.json"
OOF_NAME = "r3_query_adaptive_single872_grouped_oof_v1.npz"
RECEIPT_NAME = "r3_query_adaptive_single872_grouped_oof_receipt_v1.json"
REPORT_NAME = "r3_query_adaptive_single872_grouped_oof_report_v1.json"
RUN_STATE_NAME = "r3_query_adaptive_evaluation_run_state_v1.json"
EVALUATION_FREEZE_SCHEMA = "r3_query_adaptive_evaluation_freeze_v1"
REPORT_SCHEMA = "r3_query_adaptive_single872_grouped_oof_report_v1"
RANDOM_STATE = 35_720_260
INNER_FOLDS = 3
INNER_SEED = 35_720_260_761
THRESHOLD = 0.5
TRAINING_SCOPES = ("all_pdf_training", "single_target_training")


class RetrievalEvaluationError(ValueError):
    pass


class EvaluationRunState:
    def __init__(self, path: Path, evaluation_freeze_sha256: str) -> None:
        self.path = path
        self.evaluation_freeze_sha256 = evaluation_freeze_sha256
        self.completed = False
        retrieval.atomic_json(
            self.path,
            {
                "schema_version": "r3_query_adaptive_evaluation_run_state_v1",
                "status": "RUNNING",
                "evaluation_freeze_sha256": evaluation_freeze_sha256,
            },
        )
        atexit.register(self.finalize)

    def complete(self, report_path: Path, oof_path: Path, receipt_path: Path) -> None:
        retrieval.atomic_json(
            self.path,
            {
                "schema_version": "r3_query_adaptive_evaluation_run_state_v1",
                "status": "COMPLETE",
                "evaluation_freeze_sha256": self.evaluation_freeze_sha256,
                "report": {
                    "path": str(report_path),
                    "sha256": trace_freeze.file_sha256(report_path),
                },
                "oof": {
                    "path": str(oof_path),
                    "sha256": trace_freeze.file_sha256(oof_path),
                },
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": trace_freeze.file_sha256(receipt_path),
                },
            },
        )
        self.completed = True

    def finalize(self) -> None:
        if self.completed:
            return
        retrieval.atomic_json(
            self.path,
            {
                "schema_version": "r3_query_adaptive_evaluation_run_state_v1",
                "status": "FAILED_OR_INTERRUPTED",
                "evaluation_freeze_sha256": self.evaluation_freeze_sha256,
            },
        )


@dataclass(frozen=True)
class RawSpec:
    name: str
    family: str
    matrix: str
    c_value: float | None = None
    pca_dimension: int | None = None
    eligible_for_nested_selection: bool = True


@dataclass(frozen=True)
class CandidateRef:
    name: str
    scope: str
    kind: str
    raw_name: str
    alpha: float | None = None


def raw_specs() -> tuple[RawSpec, ...]:
    return (
        RawSpec("v4_rf", "random_forest", "v4"),
        RawSpec("retrieval_compact_rf", "random_forest", "retrieval_compact"),
        RawSpec(
            "v4_retrieval_compact_rf",
            "random_forest",
            "v4_plus_retrieval_compact",
        ),
        RawSpec("retrieval_compact_extra", "extra_trees", "retrieval_compact"),
        RawSpec(
            "v4_retrieval_compact_extra",
            "extra_trees",
            "v4_plus_retrieval_compact",
        ),
        RawSpec(
            "retrieval_compact_hgb",
            "hist_gradient_boosting",
            "retrieval_compact",
        ),
        RawSpec(
            "v4_retrieval_compact_hgb",
            "hist_gradient_boosting",
            "v4_plus_retrieval_compact",
        ),
        RawSpec(
            "retrieval_pooled_logreg_c001",
            "logistic",
            "retrieval_pooled",
            c_value=0.01,
        ),
        RawSpec(
            "retrieval_pooled_logreg_c01",
            "logistic",
            "retrieval_pooled",
            c_value=0.1,
        ),
        RawSpec(
            "retrieval_pooled_logreg_c1",
            "logistic",
            "retrieval_pooled",
            c_value=1.0,
        ),
        RawSpec(
            "retrieval_compact_pooled_logreg_c01",
            "logistic",
            "retrieval_compact_plus_pooled",
            c_value=0.1,
        ),
        RawSpec(
            "v4_retrieval_all_logreg_c01",
            "logistic",
            "v4_plus_retrieval_compact_plus_pooled",
            c_value=0.1,
        ),
        RawSpec(
            "retrieval_pooled_pca64_logreg_c1",
            "pca_logistic",
            "retrieval_pooled",
            c_value=1.0,
            pca_dimension=64,
            eligible_for_nested_selection=False,
        ),
        RawSpec(
            "v4_retrieval_all_pca64_logreg_c1",
            "pca_logistic",
            "v4_plus_retrieval_compact_plus_pooled",
            c_value=1.0,
            pca_dimension=64,
            eligible_for_nested_selection=False,
        ),
    )


def validate_evaluation_contract(
    config: Mapping[str, Any], specs: Sequence[RawSpec], matrix_names: Sequence[str]
) -> None:
    if tuple(config.get("training_scopes") or ()) != TRAINING_SCOPES:
        raise RetrievalEvaluationError("candidate training scopes changed")
    if set(config.get("feature_families") or ()) != set(matrix_names):
        raise RetrievalEvaluationError("candidate feature families changed")
    if int(config.get("inner_folds", -1)) != INNER_FOLDS:
        raise RetrievalEvaluationError("candidate inner-fold count changed")
    if int(config.get("inner_seed", -1)) != INNER_SEED:
        raise RetrievalEvaluationError("candidate inner seed changed")
    if float(config.get("threshold", math.nan)) != THRESHOLD:
        raise RetrievalEvaluationError("candidate threshold changed")
    c_values = {float(value) for value in config.get("logistic_c_values") or ()}
    pca_dimensions = {int(value) for value in config.get("pca_dimensions") or ()}
    if any(
        spec.c_value is not None and float(spec.c_value) not in c_values for spec in specs
    ):
        raise RetrievalEvaluationError("raw model uses an undeclared logistic C")
    if any(
        spec.pca_dimension is not None
        and int(spec.pca_dimension) not in pca_dimensions
        for spec in specs
    ):
        raise RetrievalEvaluationError("raw model uses an undeclared PCA dimension")
    required_configs = {
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "fusion_alphas",
    }
    if required_configs - set(config):
        raise RetrievalEvaluationError("candidate model configuration is incomplete")


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _verify_file(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or not clean(expected):
        raise RetrievalEvaluationError(f"missing frozen {label}: {path}")
    observed = trace_freeze.file_sha256(path)
    if observed != clean(expected):
        raise RetrievalEvaluationError(f"frozen {label} hash changed: {path}")


def load_query_features(
    *,
    query_dir: Path,
    records: Sequence[Mapping[str, str]],
    v4_fold_ids: np.ndarray,
    expected_single_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    manifest_path = query_dir / retrieval.MANIFEST_NAME
    feature_path = query_dir / retrieval.FEATURE_NAME
    candidate_path = query_dir / retrieval.CANDIDATE_NAME
    preinference_path = query_dir / retrieval.PREINFERENCE_NAME
    manifest = retrieval.load_json(manifest_path, "retrieval manifest")
    candidate = retrieval.load_json(candidate_path, "retrieval candidate freeze")
    preinference = retrieval.load_json(preinference_path, "retrieval preinference freeze")
    if (
        manifest.get("schema_version") != retrieval.FEATURE_SCHEMA
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("alignment_sha256")
        != trace_freeze._record_alignment_sha256(records)
    ):
        raise RetrievalEvaluationError("retrieval manifest does not align")
    if (
        candidate.get("schema_version") != retrieval.CANDIDATE_SCHEMA
        or candidate.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
        or candidate.get("alignment_sha256") != manifest.get("alignment_sha256")
    ):
        raise RetrievalEvaluationError("retrieval candidate freeze is invalid")
    if (
        preinference.get("schema_version") != retrieval.PREINFERENCE_SCHEMA
        or preinference.get("status")
        != "FROZEN_BEFORE_MODEL_INFERENCE_AND_LABEL_ACCESS"
        or preinference.get("alignment_sha256") != manifest.get("alignment_sha256")
    ):
        raise RetrievalEvaluationError("retrieval preinference freeze is invalid")
    for payload, label in ((manifest, "manifest"), (candidate, "candidate"), (preinference, "preinference")):
        boundary = payload.get("runtime_boundary") or {}
        if any(
            boundary.get(name) is not False
            for name in (
                "labels_read",
                "pmagresult_or_rockmag_read",
                "prior_predictions_read",
                "evaluation_results_read",
            )
        ):
            raise RetrievalEvaluationError(f"retrieval {label} boundary is invalid")
    for name, item in (preinference.get("inputs") or {}).items():
        if name in {"code", "pdf_membership"}:
            continue
        if not isinstance(item, Mapping):
            raise RetrievalEvaluationError(f"invalid preinference input entry: {name}")
        _verify_file(
            Path(clean(item.get("path"))).resolve(),
            item.get("sha256"),
            f"preinference input {name}",
        )
    for item in (preinference.get("inputs") or {}).get("code") or []:
        _verify_file(Path(clean(item.get("path"))).resolve(), item.get("sha256"), "builder code")
    outputs = manifest.get("outputs") or {}
    for name, path in (
        ("features", feature_path),
        ("candidate_freeze", candidate_path),
        ("preinference_freeze", preinference_path),
        ("distractor_stats", query_dir / retrieval.DISTRACTOR_NAME),
        ("evidence", query_dir / retrieval.EVIDENCE_NAME),
        ("audit", query_dir / retrieval.AUDIT_NAME),
    ):
        _verify_file(path, (outputs.get(name) or {}).get("sha256"), name)
    model_snapshot = manifest.get("model_snapshot") or {}
    if (
        model_snapshot.get("model_id") != retrieval.MODEL_ID
        or model_snapshot.get("revision") != retrieval.MODEL_REVISION
    ):
        raise RetrievalEvaluationError("retrieval model snapshot identity changed")
    snapshot_path = Path(clean(model_snapshot.get("path"))).resolve()
    for item in model_snapshot.get("files") or []:
        _verify_file(snapshot_path / clean(item.get("name")), item.get("sha256"), "model file")
    arrays = np.load(feature_path)
    required_arrays = {"compact", "pooled", "single_mask", "fold_ids"}
    if not required_arrays.issubset(arrays.files):
        raise RetrievalEvaluationError("retrieval feature archive is incomplete")
    compact = arrays["compact"].astype(np.float64)
    pooled = arrays["pooled"].astype(np.float64)
    single_mask = arrays["single_mask"].astype(bool)
    fold_ids = arrays["fold_ids"].astype(np.int8)
    expected_hashes = manifest.get("matrix_hashes") or {}
    for name, value in (
        ("compact", arrays["compact"]),
        ("pooled", arrays["pooled"]),
        ("single_mask", arrays["single_mask"]),
        ("fold_ids", arrays["fold_ids"]),
    ):
        if retrieval.array_sha256(value) != clean(expected_hashes.get(name)):
            raise RetrievalEvaluationError(f"retrieval {name} matrix hash changed")
    feature_names = list(manifest.get("feature_names") or [])
    expected_pooled = int(manifest.get("embedding_dimension", 0)) * len(
        manifest.get("pool_groups") or []
    )
    if (
        compact.shape != (len(records), len(feature_names))
        or pooled.shape != (len(records), expected_pooled)
        or single_mask.shape != (len(records),)
        or int(single_mask.sum()) != 872
        or expected_single_mask.shape != single_mask.shape
        or not np.array_equal(single_mask, expected_single_mask)
        or fold_ids.shape != (len(records),)
        or not np.array_equal(fold_ids, v4_fold_ids)
        or not np.isfinite(compact).all()
        or not np.isfinite(pooled).all()
    ):
        raise RetrievalEvaluationError("retrieval matrices do not align with V4")
    return compact, pooled, single_mask, manifest, candidate


def build_matrices(
    v4: np.ndarray, compact: np.ndarray, pooled: np.ndarray
) -> dict[str, np.ndarray]:
    return {
        "v4": np.asarray(v4, dtype=np.float64),
        "retrieval_compact": np.asarray(compact, dtype=np.float64),
        "v4_plus_retrieval_compact": np.concatenate((v4, compact), axis=1),
        "retrieval_pooled": np.asarray(pooled, dtype=np.float64),
        "retrieval_compact_plus_pooled": np.concatenate((compact, pooled), axis=1),
        "v4_plus_retrieval_compact_plus_pooled": np.concatenate(
            (v4, compact, pooled), axis=1
        ),
    }


def _predict_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings(), np.errstate(
        divide="ignore", over="ignore", invalid="ignore"
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        values = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise AssertionError("retrieval model produced invalid probabilities")
    return values


def fit_predict(
    spec: RawSpec,
    matrix: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    train_matrix = matrix[train]
    test_matrix = matrix[test]
    if spec.family == "random_forest":
        parameters = config["random_forest"]
        model: Any = RandomForestClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_features=float(parameters["max_features"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            n_jobs=int(parameters["n_jobs"]),
            random_state=int(parameters["random_state"]),
        )
        model.fit(train_matrix, labels[train], sample_weight=weights)
        return _predict_probability(model, test_matrix)
    if spec.family == "extra_trees":
        parameters = config["extra_trees"]
        model = ExtraTreesClassifier(
            n_estimators=int(parameters["n_estimators"]),
            max_features=float(parameters["max_features"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            n_jobs=int(parameters["n_jobs"]),
            random_state=int(parameters["random_state"]),
        )
        model.fit(train_matrix, labels[train], sample_weight=weights)
        return _predict_probability(model, test_matrix)
    if spec.family == "hist_gradient_boosting":
        parameters = config["hist_gradient_boosting"]
        model = HistGradientBoostingClassifier(
            max_iter=int(parameters["max_iter"]),
            learning_rate=float(parameters["learning_rate"]),
            max_leaf_nodes=int(parameters["max_leaf_nodes"]),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            l2_regularization=float(parameters["l2_regularization"]),
            random_state=int(parameters["random_state"]),
        )
        model.fit(train_matrix, labels[train], sample_weight=weights)
        return _predict_probability(model, test_matrix)
    scaler = StandardScaler()
    train_scaled = np.clip(scaler.fit_transform(train_matrix), -12.0, 12.0)
    test_scaled = np.clip(scaler.transform(test_matrix), -12.0, 12.0)
    if spec.family == "pca_logistic":
        if spec.pca_dimension is None:
            raise AssertionError("PCA candidate lacks a dimension")
        pca = PCA(
            n_components=int(spec.pca_dimension),
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        )
        train_scaled = pca.fit_transform(train_scaled)
        test_scaled = pca.transform(test_scaled)
    elif spec.family != "logistic":
        raise RetrievalEvaluationError(f"unknown model family: {spec.family}")
    model = LogisticRegression(
        C=float(spec.c_value),
        solver="liblinear",
        max_iter=3_000,
        random_state=RANDOM_STATE,
    )
    model.fit(train_scaled, labels[train], sample_weight=weights)
    return _predict_probability(model, test_scaled)


def _weights(
    train: np.ndarray, records: Sequence[Mapping[str, str]]
) -> np.ndarray:
    return trace_eval.inverse_identity_weights(train, records)


def fit_raw_candidates(
    specs: Sequence[RawSpec],
    matrices: Mapping[str, np.ndarray],
    train: np.ndarray,
    test: np.ndarray,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    weights = _weights(train, records)
    return {
        spec.name: fit_predict(
            spec,
            matrices[spec.matrix],
            train,
            test,
            labels,
            weights,
            config,
        )
        for spec in specs
    }


def candidate_refs(
    specs: Sequence[RawSpec], alphas: Sequence[float]
) -> tuple[CandidateRef, ...]:
    output = []
    for scope in TRAINING_SCOPES:
        eligible = [spec for spec in specs if spec.eligible_for_nested_selection]
        for spec in eligible:
            output.append(CandidateRef(f"{scope}::{spec.name}", scope, "raw", spec.name))
        for spec in eligible:
            if spec.name == "v4_rf":
                continue
            for alpha in alphas:
                output.append(
                    CandidateRef(
                        f"{scope}::v4_rf__mix__{spec.name}__a{alpha:.2f}",
                        scope,
                        "fusion",
                        spec.name,
                        float(alpha),
                    )
                )
    return tuple(output)


def apply_candidate(
    candidate: CandidateRef,
    raw: Mapping[str, Mapping[str, np.ndarray]],
) -> np.ndarray:
    values = np.asarray(raw[candidate.scope][candidate.raw_name], dtype=np.float64)
    if candidate.kind == "raw":
        return values
    alpha = float(candidate.alpha)
    baseline = np.asarray(raw[candidate.scope]["v4_rf"], dtype=np.float64)
    return (1.0 - alpha) * baseline + alpha * values


def _metric_key(labels: np.ndarray, probability: np.ndarray, order: int) -> tuple[float, ...]:
    observed = trace_eval.metric(labels, probability)
    return (
        float(observed["accuracy"]),
        float(observed["macro_f1"]),
        float(observed["balanced_accuracy"]),
        -float(order),
    )


def select_candidate(
    candidates: Sequence[CandidateRef],
    raw: Mapping[str, Mapping[str, np.ndarray]],
    labels: np.ndarray,
    indices: np.ndarray,
) -> tuple[CandidateRef, list[dict[str, Any]]]:
    scored = []
    for order, candidate in enumerate(candidates):
        probability = apply_candidate(candidate, raw)[indices]
        metric = trace_eval.metric(labels[indices], probability)
        scored.append((candidate, metric, _metric_key(labels[indices], probability, order)))
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored[0][0], [
        {"candidate": asdict(candidate), "metrics": metric}
        for candidate, metric, _ in scored[:10]
    ]


def fixed_metrics(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    single_mask: np.ndarray,
) -> dict[str, Any]:
    return {
        name: trace_eval.metric(labels[single_mask], values[single_mask])
        for name, values in probabilities.items()
    }


def paired_change(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
    baseline_correct = (baseline >= THRESHOLD) == labels
    candidate_correct = (candidate >= THRESHOLD) == labels
    repairs = int((~baseline_correct & candidate_correct).sum())
    degradations = int((baseline_correct & ~candidate_correct).sum())
    discordant = repairs + degradations
    p_value = (
        float(binomtest(min(repairs, degradations), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    delta = candidate_correct.astype(np.float64) - baseline_correct.astype(np.float64)
    rng = np.random.default_rng(35_720_260_797)
    samples = rng.choice(len(delta), size=(10_000, len(delta)), replace=True)
    bootstrap = delta[samples].mean(axis=1)
    return {
        "repairs": repairs,
        "degradations": degradations,
        "net_repairs": repairs - degradations,
        "mcnemar_exact_two_sided_p": round(p_value, 8),
        "accuracy_delta": round(float(delta.mean()), 8),
        "bootstrap_95_percentile_interval": [
            round(float(np.quantile(bootstrap, 0.025)), 8),
            round(float(np.quantile(bootstrap, 0.975)), 8),
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--v4-features", type=Path, default=DEFAULT_V4_FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4-context", type=Path, default=DEFAULT_V4_CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=DEFAULT_V4_CANDIDATE)
    parser.add_argument("--v4-folds", type=Path, default=DEFAULT_V4_FOLDS)
    parser.add_argument("--v4-oof", type=Path, default=DEFAULT_V4_OOF)
    parser.add_argument("--v4-report", type=Path, default=DEFAULT_V4_REPORT)
    parser.add_argument("--query-dir", type=Path, default=DEFAULT_QUERY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_QUERY_DIR)
    args = parser.parse_args(argv)

    (
        records,
        v4,
        _v4_names,
        _documents,
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
    query_dir = args.query_dir.resolve()
    complete_identity = trace_freeze.load_three_column_identity(args.identity.resolve())
    complete_multiplicity = Counter(record["REFNO"] for record in complete_identity)
    expected_single_mask = np.asarray(
        [complete_multiplicity[record["REFNO"]] == 1 for record in records], dtype=bool
    )
    compact, pooled, single_mask, query_manifest, query_candidate = load_query_features(
        query_dir=query_dir,
        records=records,
        v4_fold_ids=fold_ids,
        expected_single_mask=expected_single_mask,
    )
    matrices = build_matrices(v4, compact, pooled)
    specs = raw_specs()
    config = query_candidate["evaluation_candidates"]
    validate_evaluation_contract(config, specs, tuple(matrices))
    alphas = tuple(float(value) for value in config["fusion_alphas"])
    candidates = candidate_refs(specs, alphas)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluation_freeze_path = output_dir / EVALUATION_FREEZE_NAME
    evaluation_freeze = {
        "schema_version": EVALUATION_FREEZE_SCHEMA,
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "primary_scope": "872 PDF-backed single-target rows at full coverage",
        "raw_specs": [asdict(spec) for spec in specs],
        "nested_candidates": [asdict(candidate) for candidate in candidates],
        "selection_objective": "inner single-target accuracy, then macro-F1, then balanced accuracy, then stable candidate order",
        "outer_folds": 5,
        "inner_folds": INNER_FOLDS,
        "inner_seed": INNER_SEED,
        "threshold": THRESHOLD,
        "inputs": {
            "query_manifest": {
                "path": str(query_dir / retrieval.MANIFEST_NAME),
                "sha256": trace_freeze.file_sha256(query_dir / retrieval.MANIFEST_NAME),
            },
            "query_features": {
                "path": str(query_dir / retrieval.FEATURE_NAME),
                "sha256": trace_freeze.file_sha256(query_dir / retrieval.FEATURE_NAME),
            },
            "query_candidate": {
                "path": str(query_dir / retrieval.CANDIDATE_NAME),
                "sha256": trace_freeze.file_sha256(query_dir / retrieval.CANDIDATE_NAME),
            },
            "v4_manifest": {
                "path": str(args.v4_manifest.resolve()),
                "sha256": trace_freeze.file_sha256(args.v4_manifest.resolve()),
            },
            "v4_features": {
                "path": str(args.v4_features.resolve()),
                "sha256": trace_freeze.file_sha256(args.v4_features.resolve()),
            },
            "v4_folds": {
                "path": str(args.v4_folds.resolve()),
                "sha256": trace_freeze.file_sha256(args.v4_folds.resolve()),
            },
            "evaluator_code": {
                "path": str(Path(__file__).resolve()),
                "sha256": trace_freeze.file_sha256(Path(__file__).resolve()),
            },
        },
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
            "outer_test_labels_used_for_training_or_selection": False,
        },
    }
    retrieval.atomic_json(evaluation_freeze_path, evaluation_freeze)
    run_state = EvaluationRunState(
        output_dir / RUN_STATE_NAME,
        trace_freeze.file_sha256(evaluation_freeze_path),
    )

    # This is the first label access in this process.
    label_map = trace_eval.load_labels(
        args.targets.resolve(), {record["RESULTNO"] for record in records}
    )
    labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
    if set(map(int, np.unique(labels[single_mask]))) != {0, 1}:
        raise RetrievalEvaluationError("single-target evaluation requires both R3 classes")

    fixed = {
        scope: {
            spec.name: np.full(len(records), np.nan, dtype=np.float64) for spec in specs
        }
        for scope in TRAINING_SCOPES
    }
    primary = np.full(len(records), np.nan, dtype=np.float64)
    fold_audit: list[dict[str, Any]] = []
    for outer_fold in range(1, 6):
        outer_test = np.flatnonzero(single_mask & (fold_ids == outer_fold))
        outer_all_train = np.flatnonzero(fold_ids != outer_fold)
        outer_single_train = np.flatnonzero(single_mask & (fold_ids != outer_fold))
        test_groups = set(document_groups[outer_test])
        if test_groups & set(document_groups[outer_all_train]):
            raise AssertionError("outer fold leaks a normalized document")
        inner_splits = nested.deterministic_group_folds(
            outer_single_train,
            [clean(document_groups[index]) for index in outer_single_train],
            n_splits=INNER_FOLDS,
            seed=INNER_SEED + outer_fold,
        )
        inner_raw = {
            scope: {
                spec.name: np.full(len(records), np.nan, dtype=np.float64)
                for spec in specs
                if spec.eligible_for_nested_selection
            }
            for scope in TRAINING_SCOPES
        }
        selection_specs = [spec for spec in specs if spec.eligible_for_nested_selection]
        for inner_train_single, inner_validation in inner_splits:
            validation_groups = set(document_groups[inner_validation])
            inner_train_all = np.asarray(
                [
                    index
                    for index in outer_all_train
                    if document_groups[index] not in validation_groups
                ],
                dtype=np.int64,
            )
            for scope, train in (
                ("all_pdf_training", inner_train_all),
                ("single_target_training", inner_train_single),
            ):
                predicted = fit_raw_candidates(
                    selection_specs,
                    matrices,
                    train,
                    inner_validation,
                    records,
                    labels,
                    config,
                )
                for name, values in predicted.items():
                    inner_raw[scope][name][inner_validation] = values
        for scope in TRAINING_SCOPES:
            for values in inner_raw[scope].values():
                if not np.isfinite(values[outer_single_train]).all():
                    raise AssertionError("inner retrieval OOF is incomplete")
        selected, inner_top = select_candidate(
            candidates, inner_raw, labels, outer_single_train
        )
        outer_raw: dict[str, dict[str, np.ndarray]] = {}
        for scope, train in (
            ("all_pdf_training", outer_all_train),
            ("single_target_training", outer_single_train),
        ):
            predicted = fit_raw_candidates(
                specs,
                matrices,
                train,
                outer_test,
                records,
                labels,
                config,
            )
            outer_raw[scope] = predicted
            for name, values in predicted.items():
                fixed[scope][name][outer_test] = values
        primary[outer_test] = apply_candidate(selected, outer_raw)
        fold_metric = trace_eval.metric(labels[outer_test], primary[outer_test])
        fold_audit.append(
            {
                "outer_fold": outer_fold,
                "test_rows": len(outer_test),
                "all_training_rows": len(outer_all_train),
                "single_training_rows": len(outer_single_train),
                "selected": asdict(selected),
                "inner_top_candidates": inner_top,
                "outer_test_metrics": fold_metric,
            }
        )
        print(
            f"query-adaptive OOF: fold={outer_fold}/5 selected={selected.name} "
            f"accuracy={fold_metric['accuracy']:.4f}",
            flush=True,
        )

    if not np.isfinite(primary[single_mask]).all():
        raise AssertionError("nested query-adaptive OOF is incomplete")
    for scope in TRAINING_SCOPES:
        if any(not np.isfinite(values[single_mask]).all() for values in fixed[scope].values()):
            raise AssertionError(f"fixed retrieval OOF is incomplete for {scope}")

    single_indices = np.flatnonzero(single_mask)
    oof_path = output_dir / OOF_NAME
    oof_arrays: dict[str, np.ndarray] = {
        "single_row_indices": single_indices.astype(np.int32),
        "labels": labels[single_mask],
        "fold_ids": fold_ids[single_mask],
        "nested_primary": primary[single_mask].astype(np.float64),
    }
    for scope in TRAINING_SCOPES:
        for name, values in fixed[scope].items():
            oof_arrays[f"fixed__{scope}__{name}"] = values[single_mask].astype(np.float64)

    provisional_oof_path = oof_path.with_name(oof_path.name + ".preaudit")
    provisional_oof_path.unlink(missing_ok=True)
    retrieval.atomic_npz(provisional_oof_path, **oof_arrays)
    base_probability = fixed["all_pdf_training"]["v4_rf"]
    try:
        # The saved V4 result is opened only after all new predictions are fixed in
        # a provisional archive. It can audit, but cannot alter, those predictions.
        v4_report_path = args.v4_report.resolve()
        v4_oof_path = args.v4_oof.resolve()
        v4_report = retrieval.load_json(v4_report_path, "V4 report")
        _verify_file(
            v4_oof_path,
            (v4_report.get("output") or {}).get("sha256"),
            "saved V4 OOF",
        )
        with np.load(v4_oof_path) as saved:
            for name in ("random_forest", "labels", "fold_ids"):
                if name not in saved.files:
                    raise RetrievalEvaluationError(f"saved V4 OOF lacks {name}")
            raw_saved_v4 = saved["random_forest"]
            raw_saved_labels = saved["labels"]
            raw_saved_folds = saved["fold_ids"]
            if (
                raw_saved_v4.dtype != np.dtype("float32")
                or raw_saved_labels.dtype != np.dtype("int8")
                or raw_saved_folds.dtype != np.dtype("int8")
            ):
                raise RetrievalEvaluationError("saved V4 OOF dtypes changed")
            saved_v4 = raw_saved_v4.copy()
            saved_labels = raw_saved_labels.copy()
            saved_folds = raw_saved_folds.copy()
        if (
            saved_v4.shape != (len(records),)
            or not np.array_equal(saved_labels, labels)
            or not np.array_equal(saved_folds, fold_ids)
        ):
            raise RetrievalEvaluationError("saved V4 labels or folds changed")
        float32_reproduction = np.array_equal(
            saved_v4[single_mask],
            base_probability[single_mask].astype(np.float32),
        )
        max_v4_difference = float(
            np.max(
                np.abs(
                    saved_v4[single_mask].astype(np.float64)
                    - base_probability[single_mask]
                )
            )
        )
        if not float32_reproduction:
            raise RetrievalEvaluationError(
                "fixed V4 RF does not exactly reproduce saved float32 OOF"
            )
    except BaseException:
        provisional_oof_path.unlink(missing_ok=True)
        raise
    provisional_oof_path.replace(oof_path)

    with np.load(oof_path) as frozen:
        frozen_labels = frozen["labels"].astype(np.int8)
        frozen_folds = frozen["fold_ids"].astype(np.int8)
        frozen_primary = frozen["nested_primary"].astype(np.float64)
        frozen_fixed = {
            scope: {
                spec.name: frozen[f"fixed__{scope}__{spec.name}"].astype(np.float64)
                for spec in specs
            }
            for scope in TRAINING_SCOPES
        }
    if (
        not np.array_equal(frozen_labels, labels[single_mask])
        or not np.array_equal(frozen_folds, fold_ids[single_mask])
    ):
        raise AssertionError("published retrieval OOF identity arrays changed")
    frozen_base = frozen_fixed["all_pdf_training"]["v4_rf"]
    receipt_path = output_dir / RECEIPT_NAME
    receipt = {
        "schema_version": "r3_query_adaptive_single872_grouped_oof_receipt_v1",
        "status": "OOF_PREDICTIONS_ATOMICALLY_PUBLISHED_AFTER_V4_REPRODUCTION_AUDIT",
        "evaluation_freeze_sha256": trace_freeze.file_sha256(evaluation_freeze_path),
        "oof": {"path": str(oof_path), "sha256": trace_freeze.file_sha256(oof_path)},
        "rows": [
            {
                "single_row_index": offset,
                "v4_row_index": int(index),
                "REFNO": records[index]["REFNO"],
                "RESULTNO": records[index]["RESULTNO"],
                "fold": int(frozen_folds[offset]),
                "label": int(frozen_labels[offset]),
                "v4_rf_probability": float(frozen_base[offset]),
                "probability": float(frozen_primary[offset]),
                "prediction": int(frozen_primary[offset] >= THRESHOLD),
            }
            for offset, index in enumerate(single_indices)
        ],
    }
    retrieval.atomic_json(receipt_path, receipt)

    fixed_report = {
        scope: {
            name: trace_eval.metric(frozen_labels, values)
            for name, values in scope_values.items()
        }
        for scope, scope_values in frozen_fixed.items()
    }
    ranked_fixed = sorted(
        (
            {
                "strategy": f"{scope}::{name}",
                **metric,
            }
            for scope, values in fixed_report.items()
            for name, metric in values.items()
        ),
        key=lambda item: (
            item["accuracy"], item["macro_f1"], item["balanced_accuracy"]
        ),
        reverse=True,
    )
    primary_metric = trace_eval.metric(frozen_labels, frozen_primary)
    base_metric = trace_eval.metric(frozen_labels, frozen_base)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": "POSTHOC_EVALUATION_OF_FROZEN_FEATURES_AND_NESTED_CANDIDATES",
        "scope": {
            "rows": int(single_mask.sum()),
            "papers": int(single_mask.sum()),
            "coverage": 1.0,
            "positive": int(frozen_labels.sum()),
            "negative": int((frozen_labels == 0).sum()),
        },
        "primary_strategy": "query_adaptive_nested_candidate",
        "threshold": THRESHOLD,
        "metrics": {
            "strict_v4_random_forest": base_metric,
            "query_adaptive_nested_candidate": primary_metric,
        },
        "delta_vs_strict_v4_random_forest": paired_change(
            frozen_labels,
            frozen_base,
            frozen_primary,
        ),
        "fold_audit": fold_audit,
        "fixed_candidate_development_results": {
            "warning": "All candidates were frozen before labels, but this ranking compares multiple complete OOF development results and is not an independent confirmation set.",
            "ranked": ranked_fixed,
            "by_scope": fixed_report,
        },
        "v4_random_forest_reproduction_audit": {
            "saved_oof_opened_only_after_new_predictions_written_to_provisional_archive": True,
            "saved_report": {
                "path": str(v4_report_path),
                "sha256": trace_freeze.file_sha256(v4_report_path),
            },
            "saved_oof": {
                "path": str(v4_oof_path),
                "sha256": trace_freeze.file_sha256(v4_oof_path),
            },
            "saved_labels_exact": True,
            "saved_fold_ids_exact": True,
            "maximum_absolute_probability_difference": max_v4_difference,
            "float32_probability_array_exact": float32_reproduction,
        },
        "promotion_decision": {
            "reaches_90_percent_full_coverage_internal_oof": primary_metric["accuracy"] >= 0.90,
            "improves_strict_v4": primary_metric["accuracy"] > base_metric["accuracy"],
            "promote_to_strict_production_scoring": False,
            "reason": "Supervised grouped OOF can validate a deployable frozen model package, but production promotion requires packaging and an independent boundary audit.",
        },
        "claim_boundary": {
            "internal_grouped_nested_oof": True,
            "independent_external_validation": False,
            "feature_generation_used_labels": False,
            "pmagresult_or_rockmag_used": False,
            "prior_predictions_or_evaluations_used_as_model_input": False,
            "outer_test_labels_used_for_training_or_selection": False,
            "missing_pdf_targets_in_denominator": False,
            "selective_abstention_used": False,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
        },
        "inputs": {
            "query_manifest": {
                "path": str(query_dir / retrieval.MANIFEST_NAME),
                "sha256": trace_freeze.file_sha256(query_dir / retrieval.MANIFEST_NAME),
            },
            "evaluation_freeze": {
                "path": str(evaluation_freeze_path),
                "sha256": trace_freeze.file_sha256(evaluation_freeze_path),
            },
            "targets_postfreeze_only": {
                "path": str(args.targets.resolve()),
                "sha256": trace_freeze.file_sha256(args.targets.resolve()),
                "columns": ["RESULTNO", "R3"],
            },
        },
        "outputs": {
            "oof": {"path": str(oof_path), "sha256": trace_freeze.file_sha256(oof_path)},
            "receipt": {
                "path": str(receipt_path),
                "sha256": trace_freeze.file_sha256(receipt_path),
            },
        },
    }
    report_path = output_dir / REPORT_NAME
    retrieval.atomic_json(report_path, report)
    run_state.complete(report_path, oof_path, receipt_path)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "strict_v4": base_metric,
                "nested_primary": primary_metric,
                "delta": report["delta_vs_strict_v4_random_forest"],
                "top_fixed": ranked_fixed[:8],
                "promotion": report["promotion_decision"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateRef",
    "RawSpec",
    "RetrievalEvaluationError",
    "EvaluationRunState",
    "apply_candidate",
    "candidate_refs",
    "fit_predict",
    "load_query_features",
    "paired_change",
    "raw_specs",
    "select_candidate",
    "validate_evaluation_contract",
]
