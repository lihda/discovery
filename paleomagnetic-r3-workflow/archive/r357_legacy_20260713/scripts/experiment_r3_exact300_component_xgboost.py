#!/usr/bin/env python3
"""Canonical exact300 screen of compact XGBoost models on component features."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_exact300_auxiliary_numeric_cache as numeric_builder  # noqa: E402
import build_r3_exact300_component_bridge_cache as component_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_component_bridge_auxiliary as component_eval  # noqa: E402
import experiment_r3_exact300_own_study_auxiliary as own_eval  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/component_xgboost_v1"
TARGET_ACCURACY = 0.90
RANDOM_STATE = 9357


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    matrix: str
    model: str


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("control_component_rf8", "combined_plus_component_all", "rf8"),
        CandidateSpec("xgb_control_combined_d2", "combined", "xgb_d2"),
        CandidateSpec(
            "xgb_component_all_d1", "combined_plus_component_all", "xgb_d1"
        ),
        CandidateSpec(
            "xgb_component_all_d2", "combined_plus_component_all", "xgb_d2"
        ),
        CandidateSpec(
            "xgb_component_all_d3", "combined_plus_component_all", "xgb_d3"
        ),
        CandidateSpec(
            "xgb_component_bridge_d2", "combined_plus_component_bridge", "xgb_d2"
        ),
        CandidateSpec(
            "xgb_own_target_component_d2",
            "combined_plus_own_target_component_all",
            "xgb_d2",
        ),
    )


def make_model(kind: str) -> Any:
    if kind == "rf8":
        return own_eval.make_model(kind)
    try:
        from xgboost import XGBClassifier
    except Exception as exc:  # pragma: no cover - depends on local OpenMP linkage
        raise RuntimeError(
            "XGBoost is optional; on macOS expose libomp.dylib in "
            "DYLD_LIBRARY_PATH before running this experiment"
        ) from exc
    common = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "subsample": 0.85,
        "colsample_bytree": 0.70,
        "reg_lambda": 8.0,
        "reg_alpha": 0.1,
        "min_child_weight": 8.0,
        "n_jobs": 4,
        "random_state": RANDOM_STATE,
    }
    if kind == "xgb_d1":
        return XGBClassifier(
            n_estimators=450, learning_rate=0.03, max_depth=1, **common
        )
    if kind == "xgb_d2":
        return XGBClassifier(
            n_estimators=350, learning_rate=0.025, max_depth=2, **common
        )
    if kind == "xgb_d3":
        return XGBClassifier(
            n_estimators=300,
            learning_rate=0.025,
            max_depth=3,
            **{**common, "min_child_weight": 12.0},
        )
    raise ValueError(f"unknown model kind: {kind}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-cache", type=Path, default=component_builder.DEFAULT_CACHE)
    parser.add_argument(
        "--component-manifest", type=Path, default=component_builder.DEFAULT_MANIFEST
    )
    parser.add_argument("--numeric-cache", type=Path, default=numeric_builder.DEFAULT_CACHE)
    parser.add_argument("--numeric-manifest", type=Path, default=numeric_builder.DEFAULT_MANIFEST)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--folds", type=own_eval.parse_folds, default=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    component_cache_path = args.component_cache.resolve()
    component_manifest_path = args.component_manifest.resolve()
    numeric_cache_path = args.numeric_cache.resolve()
    numeric_manifest_path = args.numeric_manifest.resolve()
    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    fold_manifest_path = args.fold_manifest.resolve()
    targets_path = args.targets.resolve()
    pdf_dir = args.pdf_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if base.file_sha256(exact_identity) != auxiliary.EXACT_IDENTITY_SHA256:
        raise ValueError("exact300 identity hash mismatch")
    if base.file_sha256(full_identity) != auxiliary.FULL_IDENTITY_SHA256:
        raise ValueError("full identity hash mismatch")
    if base.file_sha256(fold_manifest_path) != auxiliary.FOLD_MANIFEST_SHA256:
        raise ValueError("canonical fold manifest hash mismatch")
    if base.file_sha256(targets_path) != auxiliary.TARGETS_SHA256:
        raise ValueError("targets workbook hash mismatch")
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    full_lookup = auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(fold_manifest_path, exact_records)
    full_alignment = auxiliary.sha256_records(full_records)
    numeric_arrays, numeric_manifest = numeric_sweep.validate_cache(
        numeric_cache_path, numeric_manifest_path
    )
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from auxiliary identity")
    component_arrays, component_manifest = component_eval.validate_component_cache(
        component_cache_path,
        component_manifest_path,
        len(full_records),
        full_alignment,
    )
    matrices = {**numeric_arrays, **component_arrays}
    specs = candidate_specs()
    if len({spec.name for spec in specs}) != len(specs) or any(
        spec.matrix not in matrices for spec in specs
    ):
        raise ValueError("invalid predeclared candidate set")

    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
    freeze = {
        "schema_version": "r3_exact300_component_xgboost_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "evaluated_folds": list(args.folds),
        "candidates": [asdict(spec) for spec in specs],
        "model_parameters": {
            kind: make_model(kind).get_params()
            for kind in sorted({spec.model for spec in specs})
        },
        "threshold": 0.5,
        "inputs": {
            "component_cache": {
                "path": str(component_cache_path),
                "sha256": base.file_sha256(component_cache_path),
            },
            "component_manifest": {
                "path": str(component_manifest_path),
                "sha256": base.file_sha256(component_manifest_path),
            },
            "fold_manifest": auxiliary.FOLD_MANIFEST_SHA256,
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "pmagresult_or_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_feature": False,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    freeze_path = output_dir / (
        "candidate_freeze.json"
        if full_run
        else "candidate_freeze_folds_" + "_".join(map(str, args.folds)) + ".json"
    )
    base.atomic_json(freeze_path, freeze)

    labels_by_resultno = base.load_training_labels(
        targets_path, {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in full_records], dtype=np.int8
    )
    exact_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )
    probabilities = {
        spec.name: np.full(auxiliary.EXPECTED_EXACT_ROWS, np.nan, dtype=np.float64)
        for spec in specs
    }
    fold_audit: list[dict[str, Any]] = []
    canonical_targets = [
        base.canonical_target_identity(record["ROCKNAME"]) for record in full_records
    ]
    for fold in args.folds:
        exact_test = np.flatnonzero(fold_ids == fold)
        train_full, test_full, test_refnos = auxiliary.auxiliary_split(
            full_records, exact_records, exact_test, full_lookup
        )
        weights = base.inverse_target_multiplicity_weights(
            train_full, full_records, canonical_targets
        )
        for spec in specs:
            model = make_model(spec.model)
            model.fit(
                matrices[spec.matrix][train_full],
                full_labels[train_full],
                sample_weight=weights,
            )
            probabilities[spec.name][exact_test] = model.predict_proba(
                matrices[spec.matrix][test_full]
            )[:, 1]
        train_refnos = {full_records[int(index)]["REFNO"] for index in train_full}
        fold_audit.append(
            {
                "fold": fold,
                "test_rows": int(len(exact_test)),
                "test_papers": len(test_refnos),
                "train_rows": int(len(train_full)),
                "train_papers": len(train_refnos),
                "train_test_refno_overlap": len(train_refnos & test_refnos),
            }
        )
        print(f"fold {fold}: candidates={len(specs)}, test_rows={len(exact_test)}", flush=True)

    covered = np.isin(fold_ids, np.asarray(args.folds, dtype=np.int8))
    control_name = "control_component_rf8"
    control_probability = probabilities[control_name][covered]
    results: dict[str, Any] = {}
    for spec in specs:
        values = probabilities[spec.name]
        if not np.isfinite(values[covered]).all():
            raise AssertionError(f"candidate missing requested fold: {spec.name}")
        results[spec.name] = {
            "spec": asdict(spec),
            "aggregate": base.metric_record(exact_labels[covered], values[covered]),
            "folds": [
                {
                    "fold": fold,
                    **base.metric_record(
                        exact_labels[fold_ids == fold], values[fold_ids == fold]
                    ),
                }
                for fold in args.folds
            ],
            "transition_vs_component_rf_control": own_eval.transition_record(
                exact_labels[covered], control_probability, values[covered]
            ),
        }
    ranking = sorted(
        results,
        key=lambda name: (
            results[name]["aggregate"]["accuracy"],
            results[name]["aggregate"]["macro_f1"],
            results[name]["aggregate"]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best = ranking[0]
    control_accuracy = float(results[control_name]["aggregate"]["accuracy"])
    best_accuracy = float(results[best]["aggregate"]["accuracy"])
    oof_record = None
    if full_run:
        oof_path = output_dir / "oof_probabilities.npz"
        own_eval.write_probabilities(oof_path, probabilities)
        oof_record = {"path": str(oof_path), "sha256": base.file_sha256(oof_path)}
    report = {
        "schema_version": "r3_exact300_component_xgboost_v1",
        "status": "complete",
        "mode": "full_five_fold" if full_run else "staged_screen",
        "scope": {
            "evaluated_folds": list(args.folds),
            "evaluated_rows": int(covered.sum()),
            "full_coverage": full_run,
            "exact_membership": exact_coverage,
            "auxiliary_candidate_pool": full_coverage,
        },
        "guardrails": {
            "canonical_fold_manifest_sha256": auxiliary.FOLD_MANIFEST_SHA256,
            "feature_cache_frozen_without_labels": True,
            "candidate_receipt_written_before_label_access": True,
            "all_outer_test_refno_rows_excluded_from_training": True,
            "pmagresult_or_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_feature": False,
            "identifiers_used_as_predictive_features": False,
            "model_threshold": 0.5,
        },
        "feature_summary": component_manifest.get("feature_audit"),
        "candidate_freeze": {
            "path": str(freeze_path),
            "sha256": base.file_sha256(freeze_path),
        },
        "outer_fold_audit": fold_audit,
        "ranking": ranking,
        "best_observed": {"strategy": best, **results[best]},
        "models": results,
        "continuation_gate": {
            "control_strategy": control_name,
            "control_accuracy_on_same_folds": control_accuracy,
            "best_accuracy_on_same_folds": best_accuracy,
            "strict_improvement": best_accuracy > control_accuracy,
            "continue_to_full_five_folds": not full_run and best_accuracy > control_accuracy,
        },
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "full_coverage_required": True,
            "passed": bool(full_run and best_accuracy >= TARGET_ACCURACY),
            "decision": (
                "EXACT300_TARGET_REACHED"
                if full_run and best_accuracy >= TARGET_ACCURACY
                else "STAGED_SCREEN_ONLY"
                if not full_run
                else "EXACT300_TARGET_NOT_REACHED"
            ),
        },
        "artifacts": {"oof_probabilities": oof_record},
    }
    report_path = output_dir / (
        "report.json"
        if full_run
        else "folds_" + "_".join(map(str, args.folds)) + "_report.json"
    )
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "best_observed": report["best_observed"],
                "continuation_gate": report["continuation_gate"],
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
