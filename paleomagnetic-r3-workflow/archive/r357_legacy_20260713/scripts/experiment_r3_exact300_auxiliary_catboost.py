#!/usr/bin/env python3
"""Leakage-safe CatBoost experiment on the canonical exact300 R3 screen.

The 300-row identity and its five REFNO folds remain unchanged.  In each
outer fold, CatBoost may fit on the PDF-backed 2,172-row auxiliary pool only
after every row sharing an outer-test REFNO has been removed.  Numeric
features are built and hashed before R3 is opened; they use only ROCKNAME,
the frozen page cache, local PDF metadata, and the frozen expert knowledge
artifacts.

Model family, inner split, early stopping, candidate selection, and threshold
selection are all predeclared.  Outer-test labels are used only for reporting
after predictions have been produced.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import log_loss


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402


EXACT_IDENTITY = auxiliary.EXACT_IDENTITY
FULL_IDENTITY = auxiliary.FULL_IDENTITY
SCREENING_PROTOCOL = auxiliary.SCREENING_PROTOCOL
FOLD_MANIFEST = auxiliary.FOLD_MANIFEST
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/catboost_auxiliary_v1"

FEATURE_FREEZE_NAME = "feature_freeze.json"
REPORT_NAME = "report.json"
OOF_NAME = "oof_probabilities.npz"

EXPECTED_EXACT_ROWS = auxiliary.EXPECTED_EXACT_ROWS
EXPECTED_EXACT_PAPERS = auxiliary.EXPECTED_EXACT_PAPERS
EXPECTED_FULL_ROWS = auxiliary.EXPECTED_FULL_PDF_ROWS
EXPECTED_FULL_PAPERS = auxiliary.EXPECTED_FULL_PDF_PAPERS
N_SPLITS = auxiliary.N_SPLITS

CANONICAL_FOLD_MANIFEST_SHA256 = auxiliary.FOLD_MANIFEST_SHA256
CANONICAL_REFERENCE_ACCURACY = 0.87333333
TARGET_ACCURACY = 0.90
SMOKE_MINIMUM_ACCURACY = 0.85

RANDOM_STATE = 7321
INNER_SPLIT_SALT = "r3-catboost-auxiliary-inner-v1"
INNER_VALIDATION_FRACTION = 0.20
MAX_ITERATIONS = 700
LEARNING_RATE = 0.035
L2_LEAF_REG = 15.0
EARLY_STOPPING_ROUNDS = 60
THREAD_COUNT = 2
DEPTHS = (4, 6, 8)
MATRIX_NAMES = ("curation_all", "combined")
WEIGHT_MODES = ("inverse_paper", "inverse_paper_class_balanced")
THRESHOLDS = tuple(round(value, 3) for value in np.arange(0.30, 0.701, 0.025))


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    matrix: str
    depth: int
    weight_mode: str


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the compact family frozen before any label access."""

    return tuple(
        CandidateSpec(
            name=f"catboost_{matrix}_d{depth}_{weight_mode}",
            matrix=matrix,
            depth=depth,
            weight_mode=weight_mode,
        )
        for matrix in MATRIX_NAMES
        for depth in DEPTHS
        for weight_mode in WEIGHT_MODES
    )


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_folds(raw: str) -> tuple[int, ...]:
    if raw.strip().lower() == "all":
        return tuple(range(1, N_SPLITS + 1))
    try:
        folds = tuple(sorted({int(value.strip()) for value in raw.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds must be 'all' or comma-separated integers") from exc
    if not folds or any(fold < 1 or fold > N_SPLITS for fold in folds):
        raise argparse.ArgumentTypeError("folds must be between 1 and 5")
    return folds


def deterministic_inner_split(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Create a label-free, REFNO-isolated validation partition."""

    counts = Counter(records[int(index)]["REFNO"] for index in indices)
    ranked_refnos = sorted(
        counts,
        key=lambda refno: (
            hashlib.sha256(f"{INNER_SPLIT_SALT}\0{refno}".encode("utf-8")).digest(),
            refno,
        ),
    )
    target_rows = max(1, round(len(indices) * INNER_VALIDATION_FRACTION))
    validation_refnos: set[str] = set()
    validation_rows = 0
    for refno in ranked_refnos:
        if validation_rows >= target_rows:
            break
        validation_refnos.add(refno)
        validation_rows += counts[refno]
    fit = np.asarray(
        [
            int(index)
            for index in indices
            if records[int(index)]["REFNO"] not in validation_refnos
        ],
        dtype=np.int64,
    )
    validation = np.asarray(
        [
            int(index)
            for index in indices
            if records[int(index)]["REFNO"] in validation_refnos
        ],
        dtype=np.int64,
    )
    if len(fit) == 0 or len(validation) == 0:
        raise ValueError("deterministic CatBoost inner split is empty")
    fit_refnos = {records[int(index)]["REFNO"] for index in fit}
    validation_refnos_observed = {
        records[int(index)]["REFNO"] for index in validation
    }
    if fit_refnos & validation_refnos_observed:
        raise AssertionError("REFNO leaked across CatBoost inner split")
    return fit, validation


def paper_weights(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Give each paper unit mass, optionally balancing the two class masses."""

    if mode not in WEIGHT_MODES:
        raise ValueError(f"unknown CatBoost weight mode: {mode}")
    selected_labels = labels[indices].astype(np.int8)
    if len(indices) == 0 or set(int(value) for value in selected_labels) != {0, 1}:
        raise ValueError("CatBoost fitting weights require both classes")
    paper_counts = Counter(records[int(index)]["REFNO"] for index in indices)
    weights = np.asarray(
        [1.0 / paper_counts[records[int(index)]["REFNO"]] for index in indices],
        dtype=np.float64,
    )
    raw_class_mass = {
        value: float(weights[selected_labels == value].sum()) for value in (0, 1)
    }
    if mode == "inverse_paper_class_balanced":
        class_factors = {
            value: 0.5 / raw_class_mass[value] for value in (0, 1)
        }
        weights *= np.asarray(
            [class_factors[int(value)] for value in selected_labels],
            dtype=np.float64,
        )
    weights *= len(weights) / weights.sum()
    final_class_mass = {
        value: float(weights[selected_labels == value].sum()) for value in (0, 1)
    }
    paper_mass = {
        refno: float(
            weights[
                np.asarray(
                    [records[int(index)]["REFNO"] == refno for index in indices],
                    dtype=bool,
                )
            ].sum()
        )
        for refno in paper_counts
    }
    return weights.astype(np.float32), {
        "mode": mode,
        "rows": int(len(indices)),
        "papers": len(paper_counts),
        "label_rows": {
            str(key): value
            for key, value in sorted(Counter(int(value) for value in selected_labels).items())
        },
        "raw_inverse_paper_class_mass": {
            str(key): round(value, 8) for key, value in raw_class_mass.items()
        },
        "final_class_mass": {
            str(key): round(value, 8) for key, value in final_class_mass.items()
        },
        "weight_sum": round(float(weights.sum()), 8),
        "paper_mass_min": round(min(paper_mass.values()), 8),
        "paper_mass_max": round(max(paper_mass.values()), 8),
    }


def specificity(metric: Mapping[str, Any]) -> float:
    confusion = metric["confusion"]
    denominator = int(confusion["tn"]) + int(confusion["fp"])
    return float(confusion["tn"]) / denominator if denominator else 0.0


def select_inner_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Select a noncollapsed threshold using inner validation labels only."""

    candidates = [
        base.metric_record(labels, probabilities, threshold=threshold)
        for threshold in THRESHOLDS
    ]
    eligible = [
        metric
        for metric in candidates
        if 0 < int(metric["predicted_positive"]) < len(labels)
    ]
    if not eligible:
        metric = base.metric_record(labels, probabilities, threshold=0.5)
        return 0.5, metric
    selected = max(
        eligible,
        key=lambda metric: (
            metric["accuracy"],
            metric["balanced_accuracy"],
            metric["macro_f1"],
            specificity(metric),
            -abs(float(metric["threshold"]) - 0.5),
        ),
    )
    return float(selected["threshold"]), selected


def model_parameters(depth: int, iterations: int, seed: int) -> dict[str, Any]:
    return {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": LEARNING_RATE,
        "l2_leaf_reg": L2_LEAF_REG,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": int(seed),
        "random_strength": 0.5,
        "bootstrap_type": "Bayesian",
        "bagging_temperature": 0.5,
        "border_count": 128,
        "thread_count": THREAD_COUNT,
        "task_type": "CPU",
        "allow_writing_files": False,
        "verbose": False,
    }


def candidate_selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    metric = record["inner_validation"]
    return (
        metric["accuracy"],
        metric["balanced_accuracy"],
        metric["macro_f1"],
        specificity(metric),
        -float(record["inner_logloss"]),
        -abs(float(record["selected_threshold"]) - 0.5),
        -int(record["candidate_order"]),
    )


def fit_inner_candidate(
    spec: CandidateSpec,
    *,
    candidate_order: int,
    outer_fold: int,
    fit_index: np.ndarray,
    validation_index: np.ndarray,
    matrices: Mapping[str, np.ndarray],
    full_records: Sequence[Mapping[str, str]],
    full_labels: np.ndarray,
) -> dict[str, Any]:
    from catboost import CatBoostClassifier, Pool

    weights, weight_audit = paper_weights(
        fit_index, full_records, full_labels, spec.weight_mode
    )
    matrix = matrices[spec.matrix]
    seed = RANDOM_STATE + outer_fold * 100 + candidate_order
    model = CatBoostClassifier(
        **model_parameters(spec.depth, MAX_ITERATIONS, seed)
    )
    model.fit(
        Pool(matrix[fit_index], label=full_labels[fit_index], weight=weights),
        eval_set=Pool(matrix[validation_index], label=full_labels[validation_index]),
        use_best_model=True,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
    )
    probabilities = np.asarray(
        model.predict_proba(matrix[validation_index])[:, 1], dtype=np.float64
    )
    if not np.isfinite(probabilities).all():
        raise AssertionError("CatBoost inner probabilities are not finite")
    threshold, metric = select_inner_threshold(
        full_labels[validation_index], probabilities
    )
    iterations = max(1, int(model.get_best_iteration()) + 1)
    validation_logloss = float(
        log_loss(full_labels[validation_index], probabilities, labels=[0, 1])
    )
    return {
        "candidate": spec.name,
        "candidate_order": candidate_order,
        "matrix": spec.matrix,
        "depth": spec.depth,
        "weight_mode": spec.weight_mode,
        "seed": seed,
        "selected_iterations": iterations,
        "selected_threshold": round(threshold, 8),
        "inner_logloss": round(validation_logloss, 8),
        "inner_validation": metric,
        "fit_weight_audit": weight_audit,
    }


def run_fold(
    fold: int,
    *,
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    fold_ids: np.ndarray,
    matrices: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Predict one outer fold without accepting or reading its labels."""

    from catboost import CatBoostClassifier, Pool

    exact_test = np.flatnonzero(fold_ids == fold).astype(np.int64)
    full_lookup = {
        auxiliary.row_key(record): index for index, record in enumerate(full_records)
    }
    outer_train, test_full, test_refnos = auxiliary.auxiliary_split(
        full_records, exact_records, exact_test, full_lookup
    )
    fit_index, validation_index = deterministic_inner_split(
        outer_train, full_records
    )
    outer_train_refnos = {full_records[int(index)]["REFNO"] for index in outer_train}
    fit_refnos = {full_records[int(index)]["REFNO"] for index in fit_index}
    validation_refnos = {
        full_records[int(index)]["REFNO"] for index in validation_index
    }
    if (
        outer_train_refnos & test_refnos
        or fit_refnos & validation_refnos
        or fit_refnos & test_refnos
        or validation_refnos & test_refnos
    ):
        raise AssertionError("REFNO leaked across CatBoost outer or inner partitions")

    candidate_records: list[dict[str, Any]] = []
    for order, spec in enumerate(candidate_specs()):
        result = fit_inner_candidate(
            spec,
            candidate_order=order,
            outer_fold=fold,
            fit_index=fit_index,
            validation_index=validation_index,
            matrices=matrices,
            full_records=full_records,
            full_labels=full_labels,
        )
        candidate_records.append(result)
        print(
            f"fold {fold} inner {order + 1}/{len(candidate_specs())}: "
            f"{spec.name} acc={result['inner_validation']['accuracy']:.4f} "
            f"iterations={result['selected_iterations']} "
            f"threshold={result['selected_threshold']:.3f}",
            flush=True,
        )

    selected = max(candidate_records, key=candidate_selection_key)
    selected_spec = next(
        spec for spec in candidate_specs() if spec.name == selected["candidate"]
    )
    refit_weights, refit_weight_audit = paper_weights(
        outer_train, full_records, full_labels, selected_spec.weight_mode
    )
    matrix = matrices[selected_spec.matrix]
    refit_seed = RANDOM_STATE + fold * 100 + int(selected["candidate_order"])
    model = CatBoostClassifier(
        **model_parameters(
            selected_spec.depth, int(selected["selected_iterations"]), refit_seed
        )
    )
    model.fit(
        Pool(
            matrix[outer_train],
            label=full_labels[outer_train],
            weight=refit_weights,
        )
    )
    probabilities = np.asarray(
        model.predict_proba(matrix[test_full])[:, 1], dtype=np.float64
    )
    if probabilities.shape != (len(exact_test),) or not np.isfinite(probabilities).all():
        raise AssertionError("CatBoost outer predictions are incomplete")
    threshold = float(selected["selected_threshold"])
    decisions = (probabilities >= threshold).astype(np.int8)

    excluded_full_rows = sum(
        record["REFNO"] in test_refnos for record in full_records
    )
    expected_excluded = {
        index
        for index, record in enumerate(full_records)
        if record["REFNO"] in test_refnos
    }
    audit = {
        "fold": fold,
        "exact_test_rows": int(len(exact_test)),
        "exact_test_papers": len(test_refnos),
        "auxiliary_candidate_rows": len(full_records),
        "outer_train_rows": int(len(outer_train)),
        "outer_train_papers": len(outer_train_refnos),
        "full_rows_excluded_for_test_refnos": int(excluded_full_rows),
        "expected_rows_excluded_for_test_refnos": len(expected_excluded),
        "outer_train_test_refno_overlap": len(outer_train_refnos & test_refnos),
        "outer_train_exact_test_row_overlap": len(
            set(int(index) for index in outer_train) & set(int(index) for index in test_full)
        ),
        "excluded_test_refno_rows_remaining_in_outer_train": sum(
            full_records[int(index)]["REFNO"] in test_refnos for index in outer_train
        ),
        "inner_fit_rows": int(len(fit_index)),
        "inner_fit_papers": len(fit_refnos),
        "inner_validation_rows": int(len(validation_index)),
        "inner_validation_papers": len(validation_refnos),
        "inner_fit_validation_refno_overlap": len(fit_refnos & validation_refnos),
        "inner_fit_outer_test_refno_overlap": len(fit_refnos & test_refnos),
        "inner_validation_outer_test_refno_overlap": len(
            validation_refnos & test_refnos
        ),
        "candidate_selection_source": "inner REFNO-isolated validation only",
        "candidate_selection_rule": (
            "accuracy, balanced accuracy, macro-F1, specificity, negative logloss, "
            "threshold proximity to 0.5, declaration order"
        ),
        "selected_candidate": selected["candidate"],
        "selected_iterations": int(selected["selected_iterations"]),
        "selected_threshold": threshold,
        "selected_inner_validation": selected["inner_validation"],
        "selected_inner_logloss": selected["inner_logloss"],
        "refit_weight_audit": refit_weight_audit,
        "inner_candidates": candidate_records,
        "outer_test_labels_received_by_run_fold": False,
    }
    return exact_test, probabilities, decisions, audit


def write_prediction_artifact(
    path: Path,
    probability: np.ndarray,
    decision: np.ndarray,
    *,
    expected_rows: int,
) -> None:
    if probability.shape != (expected_rows,) or decision.shape != (expected_rows,):
        raise ValueError("CatBoost prediction artifact row count mismatch")
    if not np.isfinite(probability).all() or not np.isin(decision, (0, 1)).all():
        raise ValueError("CatBoost prediction artifact contains invalid values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            catboost_probability=probability.astype(np.float32),
            catboost_inner_selected_decision=decision.astype(np.int8),
        )
    temporary.replace(path)


def freeze_protocol() -> dict[str, Any]:
    return {
        "candidate_family": [asdict(spec) for spec in candidate_specs()],
        "candidate_count": len(candidate_specs()),
        "matrices": list(MATRIX_NAMES),
        "depths": list(DEPTHS),
        "weight_modes": list(WEIGHT_MODES),
        "maximum_iterations": MAX_ITERATIONS,
        "learning_rate": LEARNING_RATE,
        "l2_leaf_reg": L2_LEAF_REG,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "threshold_grid": list(THRESHOLDS),
        "thread_count": THREAD_COUNT,
        "random_state": RANDOM_STATE,
        "inner_split": {
            "unit": "REFNO",
            "labels_used_to_assign_groups": False,
            "method": "SHA256-ranked REFNO prefix by row target",
            "salt": INNER_SPLIT_SALT,
            "validation_fraction": INNER_VALIDATION_FRACTION,
        },
        "selection": {
            "source": "inner validation only",
            "candidate_order": [
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "specificity",
                "negative_logloss",
                "threshold_proximity_to_0.5",
                "declaration_order",
            ],
            "early_stopping_metric": "inner validation Logloss",
            "threshold_metric": "inner validation row accuracy",
            "primary_outer_decision": "probability >= per-fold inner-selected threshold",
        },
        "smoke_gate": {
            "fold": 1,
            "minimum_accuracy": SMOKE_MINIMUM_ACCURACY,
            "requires_finite_noncollapsed_predictions": True,
        },
        "promotion_gate": {"minimum_exact300_accuracy": TARGET_ACCURACY},
    }


def build_feature_freeze(
    *,
    exact_records: Sequence[Mapping[str, str]],
    full_records: Sequence[Mapping[str, str]],
    fold_ids: np.ndarray,
    matrices: Mapping[str, np.ndarray],
    base_features: Mapping[str, Any],
    input_audit: Mapping[str, Any],
    knowledge_audit: Mapping[str, Any],
    fold_manifest_path: Path,
    catboost_version: str,
) -> dict[str, Any]:
    protocol = freeze_protocol()
    return {
        "schema_version": "r3_exact300_auxiliary_catboost_feature_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": {
            "evaluation_rows": len(exact_records),
            "evaluation_papers": len({record["REFNO"] for record in exact_records}),
            "auxiliary_candidate_rows": len(full_records),
            "auxiliary_candidate_papers": len(
                {record["REFNO"] for record in full_records}
            ),
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": auxiliary.sha256_records(full_records),
        },
        "canonical_outer_fold_manifest": {
            "path": str(fold_manifest_path),
            "sha256": base.file_sha256(fold_manifest_path),
            "required_sha256": CANONICAL_FOLD_MANIFEST_SHA256,
            "fold_id_sha256": hashlib.sha256(
                bytes(int(value) for value in fold_ids)
            ).hexdigest(),
        },
        "candidate_protocol": protocol,
        "candidate_protocol_sha256": sha256_json(protocol),
        "feature_matrices": {
            name: {
                "shape": list(matrix.shape),
                "dtype": str(matrix.dtype),
                "sha256": auxiliary.sha256_array(matrix),
                "finite": bool(np.isfinite(matrix).all()),
            }
            for name, matrix in matrices.items()
        },
        "document_alignment": {
            "local_documents_sha256": auxiliary.sha256_strings(
                base_features["local_documents"]
            ),
            "rockname_documents_sha256": auxiliary.sha256_strings(
                base_features["rockname_documents"]
            ),
        },
        "allowed_predictive_sources": [
            "ROCKNAME from frozen identities",
            "original-paper text/layout from frozen page cache",
            "local PDF byte size",
            "frozen aggregate R3 taxonomy",
            "frozen positive ROCKMAG-derived expert vocabulary",
        ],
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_source_cells_read": False,
            "prior_predictions_or_evaluations_read": False,
            "refno_or_resultno_used_as_predictive_features": False,
            "identifiers_used_for_alignment_and_group_isolation_only": True,
        },
        "software": {
            "catboost": catboost_version,
            "thread_count": THREAD_COUNT,
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
        "inputs": dict(input_audit),
        "knowledge": dict(knowledge_audit),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=FULL_IDENTITY)
    parser.add_argument("--screening-protocol", type=Path, default=SCREENING_PROTOCOL)
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--folds", type=parse_folds, default=tuple(range(1, 6)))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    base.require_experiment_dependencies()
    try:
        import catboost
    except ImportError as exc:
        raise SystemExit(
            "CatBoost 1.2.8 is required; run with tmp/r3_embedding_venv/bin/python"
        ) from exc

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    protocol_path = args.screening_protocol.resolve()
    fold_manifest_path = args.fold_manifest.resolve()
    targets_path = args.targets.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy_path = args.taxonomy.resolve()
    positive_kb_path = args.positive_kb.resolve()
    output_dir = args.output_dir.resolve()
    folds_requested = tuple(args.folds)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_audit = auxiliary.validate_inputs(
        exact_identity,
        full_identity,
        protocol_path,
        fold_manifest_path,
        targets_path,
        page_cache_path,
        taxonomy_path,
        positive_kb_path,
    )
    if base.file_sha256(fold_manifest_path) != CANONICAL_FOLD_MANIFEST_SHA256:
        raise ValueError("CatBoost experiment requires the canonical exact300 fold manifest")
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(fold_manifest_path, exact_records)

    refnos = {record["REFNO"] for record in full_records}
    paper_text = base.load_page_text(page_cache_path, refnos)
    pages = curation.load_pages(page_cache_path, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )
    base_features = base.build_features(full_records, paper_text, concepts)
    curation_features = curation.build_features(
        full_records, pages, concepts, pdf_dir
    )
    curation_all = np.asarray(curation_features["all"], dtype=np.float32)
    combined = np.concatenate(
        (
            np.asarray(base_features["engineered_rows"], dtype=np.float32),
            curation_all,
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    matrices = {"curation_all": curation_all, "combined": combined}
    if any(matrix.shape[0] != EXPECTED_FULL_ROWS for matrix in matrices.values()):
        raise ValueError("CatBoost feature matrices are not aligned to the full pool")

    # This file is durably written before targets.xlsx is opened for R3 values.
    freeze = build_feature_freeze(
        exact_records=exact_records,
        full_records=full_records,
        fold_ids=fold_ids,
        matrices=matrices,
        base_features=base_features,
        input_audit=input_audit,
        knowledge_audit=knowledge_audit,
        fold_manifest_path=fold_manifest_path,
        catboost_version=catboost.__version__,
    )
    freeze_path = output_dir / FEATURE_FREEZE_NAME
    base.atomic_json(freeze_path, freeze)
    freeze_sha256 = base.file_sha256(freeze_path)

    # R3 is dereferenced only after membership, folds, candidates, and features freeze.
    full_labels_by_resultno = base.load_training_labels(
        targets_path, {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [full_labels_by_resultno[record["RESULTNO"]] for record in full_records],
        dtype=np.int8,
    )
    exact_labels = np.asarray(
        [full_labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )

    full_run = folds_requested == tuple(range(1, N_SPLITS + 1))
    if full_run:
        all_probability = np.full(EXPECTED_EXACT_ROWS, np.nan, dtype=np.float64)
        all_decision = np.full(EXPECTED_EXACT_ROWS, -1, dtype=np.int8)
    else:
        partial_probability: list[np.ndarray] = []
        partial_decision: list[np.ndarray] = []
    fold_audits: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in folds_requested:
        exact_test, probabilities, decisions, audit = run_fold(
            fold,
            full_records=full_records,
            exact_records=exact_records,
            full_labels=full_labels,
            fold_ids=fold_ids,
            matrices=matrices,
        )
        test_labels = exact_labels[exact_test]
        decision_metric = base.metric_record(
            test_labels, decisions.astype(np.float64), threshold=0.5
        )
        default_metric = base.metric_record(test_labels, probabilities, threshold=0.5)
        fold_metrics.append(
            {
                "fold": fold,
                "inner_selected_threshold_decision": decision_metric,
                "default_0.5_probability_diagnostic": default_metric,
            }
        )
        fold_audits.append(audit)
        if full_run:
            all_probability[exact_test] = probabilities
            all_decision[exact_test] = decisions
        else:
            partial_probability.append(probabilities)
            partial_decision.append(decisions)
        print(
            f"fold {fold}/{N_SPLITS}: selected={audit['selected_candidate']} "
            f"outer_acc={decision_metric['accuracy']:.4f} "
            f"confusion={decision_metric['confusion']}",
            flush=True,
        )

    if full_run:
        if not np.isfinite(all_probability).all() or not np.isin(all_decision, (0, 1)).all():
            raise AssertionError("full exact300 CatBoost OOF coverage is incomplete")
        artifact_probability = all_probability
        artifact_decision = all_decision
        evaluation_labels = exact_labels
        artifact_name = OOF_NAME
        report_name = REPORT_NAME
    else:
        artifact_probability = np.concatenate(partial_probability)
        artifact_decision = np.concatenate(partial_decision)
        evaluation_labels = np.concatenate(
            [exact_labels[np.flatnonzero(fold_ids == fold)] for fold in folds_requested]
        )
        fold_token = "_".join(str(fold) for fold in folds_requested)
        artifact_name = f"folds_{fold_token}_probabilities.npz"
        report_name = f"folds_{fold_token}_report.json"

    probability_metric = base.metric_record(
        evaluation_labels, artifact_probability, threshold=0.5
    )
    primary_metric = base.metric_record(
        evaluation_labels, artifact_decision.astype(np.float64), threshold=0.5
    )
    artifact_path = output_dir / artifact_name
    write_prediction_artifact(
        artifact_path,
        artifact_probability,
        artifact_decision,
        expected_rows=len(evaluation_labels),
    )
    stable_noncollapsed = bool(
        np.isfinite(artifact_probability).all()
        and 0 < int(artifact_decision.sum()) < len(artifact_decision)
    )
    zero_overlap_fields = (
        "outer_train_test_refno_overlap",
        "outer_train_exact_test_row_overlap",
        "excluded_test_refno_rows_remaining_in_outer_train",
        "inner_fit_validation_refno_overlap",
        "inner_fit_outer_test_refno_overlap",
        "inner_validation_outer_test_refno_overlap",
    )
    overlap_audit_passed = all(
        int(audit[field]) == 0
        for audit in fold_audits
        for field in zero_overlap_fields
    )
    report = {
        "schema_version": "r3_exact300_auxiliary_catboost_grouped_oof_v1",
        "status": "complete",
        "scope": {
            "evaluation_identity": "canonical frozen exact300",
            "evaluated_folds": list(folds_requested),
            "evaluated_rows": len(evaluation_labels),
            "evaluated_papers": len(
                {
                    exact_records[int(index)]["REFNO"]
                    for fold in folds_requested
                    for index in np.flatnonzero(fold_ids == fold)
                }
            ),
            "full_exact300_coverage": full_run,
            "auxiliary_candidate_rows": len(full_records),
            "auxiliary_candidate_papers": len(refnos),
            "full_2172_rows_evaluated": False,
            "full_pool_coverage": full_coverage,
            "exact_coverage": exact_coverage,
        },
        "guardrails": {
            "canonical_fold_manifest_sha256": base.file_sha256(fold_manifest_path),
            "canonical_fold_manifest_required_sha256": CANONICAL_FOLD_MANIFEST_SHA256,
            "outer_split": "frozen five-fold exact300 REFNO manifest",
            "training_rule": (
                "all PDF-backed fixed-training rows except every row sharing an "
                "outer-test REFNO"
            ),
            "candidate_protocol_frozen_before_label_access": True,
            "feature_freeze_before_label_access": True,
            "feature_freeze_sha256": freeze_sha256,
            "candidate_and_threshold_selection_source": (
                "deterministic REFNO-isolated inner validation only"
            ),
            "outer_test_labels_used_for_candidate_iteration_or_threshold_selection": False,
            "pmagresult_or_rockmag_source_cells_used_as_predictive_input": False,
            "prior_predictions_or_evaluations_used_as_predictive_input": False,
            "identifiers_used_as_predictive_features": False,
            "outer_and_inner_zero_overlap_audit_passed": overlap_audit_passed,
            "zero_overlap_fields": list(zero_overlap_fields),
            "posthoc_outer_label_thresholds_used": False,
        },
        "data": {
            "full_label_counts": {
                str(key): value
                for key, value in sorted(Counter(int(value) for value in full_labels).items())
            },
            "evaluated_label_counts": {
                str(key): value
                for key, value in sorted(
                    Counter(int(value) for value in evaluation_labels).items()
                )
            },
            "feature_shapes": {
                name: list(matrix.shape) for name, matrix in matrices.items()
            },
        },
        "predeclared_candidate_protocol": freeze["candidate_protocol"],
        "outer_fold_audit": fold_audits,
        "fold_metrics": fold_metrics,
        "primary_full_coverage_result": {
            "strategy": "per-fold inner-selected CatBoost candidate and threshold",
            **primary_metric,
            "stable_noncollapsed": stable_noncollapsed,
        },
        "default_threshold_diagnostic": probability_metric,
        "smoke_gate": (
            {
                "applicable": folds_requested == (1,),
                "minimum_accuracy": SMOKE_MINIMUM_ACCURACY,
                "observed_accuracy": primary_metric["accuracy"],
                "stable_noncollapsed": stable_noncollapsed,
                "continue_to_all_folds": bool(
                    folds_requested == (1,)
                    and primary_metric["accuracy"] >= SMOKE_MINIMUM_ACCURACY
                    and stable_noncollapsed
                    and overlap_audit_passed
                ),
            }
            if folds_requested == (1,)
            else {"applicable": False}
        ),
        "promotion_gate": {
            "applicable": full_run,
            "target_accuracy": TARGET_ACCURACY,
            "canonical_auxiliary_reference_accuracy": CANONICAL_REFERENCE_ACCURACY,
            "observed_accuracy": primary_metric["accuracy"],
            "absolute_delta_from_reference": round(
                primary_metric["accuracy"] - CANONICAL_REFERENCE_ACCURACY, 8
            ),
            "passed": bool(full_run and primary_metric["accuracy"] >= TARGET_ACCURACY),
            "decision": (
                "EXACT300_TARGET_REACHED"
                if full_run and primary_metric["accuracy"] >= TARGET_ACCURACY
                else "EXACT300_TARGET_NOT_REACHED"
                if full_run
                else "PARTIAL_SMOKE_ONLY"
            ),
        },
        "artifacts": {
            "feature_freeze": {
                "path": str(freeze_path),
                "sha256": freeze_sha256,
            },
            "predictions": {
                "path": str(artifact_path),
                "sha256": base.file_sha256(artifact_path),
                "rows_per_array": len(evaluation_labels),
                "contains_labels": False,
                "contains_identifiers": False,
                "arrays": [
                    "catboost_probability",
                    "catboost_inner_selected_decision",
                ],
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
    }
    report_path = output_dir / report_name
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "primary": report["primary_full_coverage_result"],
                "smoke_gate": report["smoke_gate"],
                "promotion_gate": report["promotion_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
