#!/usr/bin/env python3
"""Explore leakage-controlled thresholds for the all-labelled R3 OOF run."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_all_labelled_pdf_grouped_oof as all_oof  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OOF_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
OUTPUT = OOF_DIR / "nested_threshold_report.json"
INNER_SPLITS = 5
INNER_SEED = 20_260_713
GRID_THRESHOLDS = np.round(np.arange(0.20, 0.801, 0.01), 2)


def metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.int8)
    tn = int(((labels == 0) & (prediction == 0)).sum())
    fp = int(((labels == 0) & (prediction == 1)).sum())
    fn = int(((labels == 1) & (prediction == 0)).sum())
    tp = int(((labels == 1) & (prediction == 1)).sum())
    return {
        "rows": len(labels),
        "correct": int((labels == prediction).sum()),
        "accuracy": round(float((labels == prediction).mean()), 8),
        "macro_f1": round(float(f1_score(labels, prediction, average="macro")), 8),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(labels, prediction)), 8
        ),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def select_threshold(
    labels: np.ndarray, probabilities: np.ndarray, candidates: Iterable[float]
) -> tuple[float, float]:
    scored = [
        (float((labels == (probabilities >= threshold)).mean()), float(threshold))
        for threshold in candidates
    ]
    best_accuracy = max(accuracy for accuracy, _ in scored)
    tied = [
        threshold
        for accuracy, threshold in scored
        if np.isclose(accuracy, best_accuracy, rtol=0.0, atol=1e-15)
    ]
    threshold = min(tied, key=lambda value: (abs(value - 0.5), value))
    return threshold, best_accuracy


def exact_candidates(probabilities: np.ndarray) -> np.ndarray:
    values = np.unique(np.asarray(probabilities, dtype=np.float64))
    return np.concatenate(
        (
            [np.nextafter(values[0], -np.inf)],
            values,
            [np.nextafter(values[-1], np.inf)],
        )
    )


def main() -> int:
    feature_path = OOF_DIR / "combined_features.npz"
    fold_path = OOF_DIR / "refno_fold_manifest.json"
    receipt_path = OOF_DIR / "prediction_receipt.json"
    oof_path = OOF_DIR / "oof_probabilities.npz"
    matrix = np.load(feature_path)["combined"].astype(np.float32, copy=False)
    fold_manifest = json.loads(fold_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    records = [
        {key: str(row[key]) for key in ("REFNO", "RESULTNO", "ROCKNAME")}
        for row in receipt["rows"]
    ]
    fold_ids = np.asarray([int(row["fold"]) for row in receipt["rows"]], dtype=np.int8)
    outer_probability = np.load(oof_path)[all_oof.PRIMARY_STRATEGY].astype(np.float64)
    if matrix.shape != (len(records), 351) or len(records) != all_oof.EXPECTED_PDF_ROWS:
        raise ValueError("all-labelled threshold inputs do not align")
    if fold_manifest["alignment_sha256"] != all_oof.record_sha256(records):
        raise ValueError("threshold records do not match the frozen fold manifest")

    labels_by_resultno = all_oof.load_labels(
        base.DEFAULT_TARGETS, {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    canonical_targets = [
        base.canonical_target_identity(record["ROCKNAME"]) for record in records
    ]
    experts = (
        nested.ExpertSpec("expert_hgb_combined", "combined", "hgb10"),
        nested.ExpertSpec("expert_rf_combined", "combined", "rf8"),
        nested.ExpertSpec("expert_extra_combined", "combined", "extra8"),
        nested.ExpertSpec(
            "expert_logreg_combined", "combined", "logreg001", standardized=True
        ),
    )

    grid_prediction = np.zeros(len(records), dtype=np.int8)
    exact_prediction = np.zeros(len(records), dtype=np.int8)
    fold_reports: list[dict[str, Any]] = []
    for outer_fold in range(1, all_oof.N_SPLITS + 1):
        outer_train = np.flatnonzero(fold_ids != outer_fold)
        outer_test = np.flatnonzero(fold_ids == outer_fold)
        inner_groups = [records[int(index)]["REFNO"] for index in outer_train]
        inner_folds = nested.deterministic_group_folds(
            outer_train,
            inner_groups,
            n_splits=INNER_SPLITS,
            seed=INNER_SEED + outer_fold,
        )
        inner_probability = np.full(len(records), np.nan, dtype=np.float64)
        inner_audit: list[dict[str, int]] = []
        for inner_fold, (inner_train, inner_validation) in enumerate(
            inner_folds, start=1
        ):
            weights = base.inverse_target_multiplicity_weights(
                inner_train, records, canonical_targets
            )
            expert_probability = []
            for expert in experts:
                fitted = nested.fit_expert(
                    expert, matrix, inner_train, labels, weights
                )
                expert_probability.append(fitted.predict(matrix[inner_validation]))
            inner_probability[inner_validation] = np.column_stack(
                expert_probability
            ).mean(axis=1)
            inner_audit.append(
                {
                    "inner_fold": inner_fold,
                    "train_rows": len(inner_train),
                    "validation_rows": len(inner_validation),
                }
            )
        if not np.isfinite(inner_probability[outer_train]).all():
            raise AssertionError("inner OOF threshold probabilities are incomplete")
        grid_threshold, grid_inner_accuracy = select_threshold(
            labels[outer_train], inner_probability[outer_train], GRID_THRESHOLDS
        )
        exact_threshold, exact_inner_accuracy = select_threshold(
            labels[outer_train],
            inner_probability[outer_train],
            exact_candidates(inner_probability[outer_train]),
        )
        grid_prediction[outer_test] = (
            outer_probability[outer_test] >= grid_threshold
        ).astype(np.int8)
        exact_prediction[outer_test] = (
            outer_probability[outer_test] >= exact_threshold
        ).astype(np.int8)
        fold_reports.append(
            {
                "outer_fold": outer_fold,
                "train_rows": len(outer_train),
                "test_rows": len(outer_test),
                "grid_threshold": grid_threshold,
                "grid_inner_oof_accuracy": round(grid_inner_accuracy, 8),
                "grid_outer_result": metrics(
                    labels[outer_test], grid_prediction[outer_test]
                ),
                "exact_threshold": exact_threshold,
                "exact_inner_oof_accuracy": round(exact_inner_accuracy, 8),
                "exact_outer_result": metrics(
                    labels[outer_test], exact_prediction[outer_test]
                ),
                "inner_fold_audit": inner_audit,
            }
        )
        print(
            f"outer fold {outer_fold}: grid={grid_threshold:.2f}, "
            f"exact={exact_threshold:.6f}",
            flush=True,
        )

    global_exact_threshold, global_exact_accuracy = select_threshold(
        labels, outer_probability, exact_candidates(outer_probability)
    )
    fixed_thresholds = {
        f"{threshold:.2f}": metrics(labels, outer_probability >= threshold)
        for threshold in (0.50, 0.52, 0.54, 0.56, 0.57, 0.58, 0.60)
    }
    report = {
        "schema_version": "r3_all_labelled_nested_threshold_v1",
        "status": "complete_exploratory",
        "rows": len(records),
        "paper_groups": len({record["REFNO"] for record in records}),
        "method": {
            "outer_folds": "existing five REFNO-grouped OOF folds",
            "inner_folds": f"{INNER_SPLITS} REFNO-grouped OOF folds inside each outer training set",
            "grid_candidates": [float(value) for value in GRID_THRESHOLDS],
            "selection_objective": "inner OOF row accuracy",
            "tie_break": "threshold closest to 0.5, then lower threshold",
            "outer_test_labels_used_for_threshold_selection": False,
        },
        "experts": [asdict(expert) for expert in experts],
        "baseline_0_5": fixed_thresholds["0.50"],
        "nested_grid_result": metrics(labels, grid_prediction),
        "nested_exact_result": metrics(labels, exact_prediction),
        "folds": fold_reports,
        "posthoc_global_oracle": {
            "warning": "selected after reading all OOF labels; not a valid generalization estimate",
            "threshold": global_exact_threshold,
            "accuracy": round(global_exact_accuracy, 8),
            "result": metrics(labels, outer_probability >= global_exact_threshold),
        },
        "fixed_threshold_posthoc_comparison": fixed_thresholds,
        "inputs": {
            "combined_features": {
                "path": str(feature_path),
                "sha256": base.file_sha256(feature_path),
            },
            "fold_manifest": {
                "path": str(fold_path),
                "sha256": base.file_sha256(fold_path),
            },
            "prediction_receipt": {
                "path": str(receipt_path),
                "sha256": base.file_sha256(receipt_path),
            },
            "outer_oof": {"path": str(oof_path), "sha256": base.file_sha256(oof_path)},
            "targets": {
                "path": str(base.DEFAULT_TARGETS),
                "sha256": base.file_sha256(base.DEFAULT_TARGETS),
            },
        },
    }
    base.atomic_json(OUTPUT, report)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "baseline_0_5": report["baseline_0_5"],
                "nested_grid_result": report["nested_grid_result"],
                "nested_exact_result": report["nested_exact_result"],
                "posthoc_global_oracle": report["posthoc_global_oracle"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
