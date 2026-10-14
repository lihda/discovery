#!/usr/bin/env python3
"""Evaluate the frozen random300 extension cohort as a paper-isolated holdout."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_exact300_auxiliary_numeric_cache as numeric_builder  # noqa: E402
import build_r3_exact300_component_bridge_cache as component_builder  # noqa: E402
import build_r3_exact300_own_study_cache as own_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_component_bridge_auxiliary as component_eval  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_exact300_own_study_auxiliary as own_eval  # noqa: E402
import experiment_r3_exact300_paper_consistency as paper_v1  # noqa: E402
import experiment_r3_exact300_paper_consistency_v2 as paper_v2  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import freeze_r3_random300_extension_cohort as cohort  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_random300_extension_v1/holdout"
TARGET_ACCURACY = 0.90


def write_probability_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    payload = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in sorted(arrays.items())
    }
    if any(
        values.shape != (cohort.SAMPLE_ROWS,) or not np.isfinite(values).all()
        for values in payload.values()
    ):
        raise ValueError("random300 holdout requires 300 finite probabilities")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def validate_frozen_inputs(paths: Mapping[str, Path]) -> dict[str, Any]:
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    if runtime != paper_v1.CANONICAL_RUNTIME:
        raise ValueError(
            f"canonical paper-consistency runtime required: "
            f"expected={paper_v1.CANONICAL_RUNTIME}, observed={runtime}"
        )
    expected = {
        "full_identity": auxiliary.FULL_IDENTITY_SHA256,
        "exact_identity": auxiliary.EXACT_IDENTITY_SHA256,
        "targets": auxiliary.TARGETS_SHA256,
    }
    for name, digest in expected.items():
        if base.file_sha256(paths[name]) != digest:
            raise ValueError(f"frozen input hash mismatch: {name}")
    return runtime


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=cohort.OUTPUT)
    parser.add_argument("--numeric-cache", type=Path, default=numeric_builder.DEFAULT_CACHE)
    parser.add_argument(
        "--numeric-manifest", type=Path, default=numeric_builder.DEFAULT_MANIFEST
    )
    parser.add_argument("--own-cache", type=Path, default=own_builder.DEFAULT_CACHE)
    parser.add_argument("--own-manifest", type=Path, default=own_builder.DEFAULT_MANIFEST)
    parser.add_argument(
        "--component-cache", type=Path, default=component_builder.DEFAULT_CACHE
    )
    parser.add_argument(
        "--component-manifest", type=Path, default=component_builder.DEFAULT_MANIFEST
    )
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = {
        "cohort": args.cohort.resolve(),
        "numeric_cache": args.numeric_cache.resolve(),
        "numeric_manifest": args.numeric_manifest.resolve(),
        "own_cache": args.own_cache.resolve(),
        "own_manifest": args.own_manifest.resolve(),
        "component_cache": args.component_cache.resolve(),
        "component_manifest": args.component_manifest.resolve(),
        "full_identity": args.full_identity.resolve(),
        "exact_identity": args.exact_identity.resolve(),
        "targets": args.targets.resolve(),
        "pdf_dir": args.pdf_dir.resolve(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = validate_frozen_inputs(paths)

    full_records, full_coverage = base.load_pdf_only_identity(
        paths["full_identity"], paths["pdf_dir"]
    )
    exact_records, _ = base.load_pdf_only_identity(
        paths["exact_identity"], paths["pdf_dir"]
    )
    manifest = json.loads(paths["cohort"].read_text(encoding="utf-8"))
    test_records, test_indices = cohort.validate_manifest(
        manifest, full_records, exact_records
    )
    test_refnos = {record["REFNO"] for record in test_records}
    train_indices = np.asarray(
        [
            index
            for index, record in enumerate(full_records)
            if record["REFNO"] not in test_refnos
        ],
        dtype=np.int64,
    )
    excluded_indices = np.asarray(
        [
            index
            for index, record in enumerate(full_records)
            if record["REFNO"] in test_refnos
        ],
        dtype=np.int64,
    )
    train_refnos = {full_records[int(index)]["REFNO"] for index in train_indices}
    if train_refnos & test_refnos:
        raise AssertionError("random300 test REFNO remains in training")
    if not set(int(index) for index in test_indices).issubset(
        set(int(index) for index in excluded_indices)
    ):
        raise AssertionError("not every random300 row was excluded from training")

    full_alignment = auxiliary.sha256_records(full_records)
    numeric_arrays, numeric_manifest = numeric_sweep.validate_cache(
        paths["numeric_cache"], paths["numeric_manifest"]
    )
    own_arrays, own_manifest = own_eval.validate_own_cache(
        paths["own_cache"], paths["own_manifest"], len(full_records), full_alignment
    )
    component_arrays, component_manifest = component_eval.validate_component_cache(
        paths["component_cache"],
        paths["component_manifest"],
        len(full_records),
        full_alignment,
    )
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from full identity")
    matrices = {**numeric_arrays, **own_arrays, **component_arrays}
    experts = nested.expert_specs()
    router = paper_v1.router_spec()
    smoothers = paper_v1.smoothing_specs()
    if any(spec.matrix not in matrices for spec in experts):
        raise ValueError("an expert matrix is absent from frozen caches")

    freeze = {
        "schema_version": "r3_random300_extension_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_TEST_LABEL_SCORING",
        "cohort": {
            "path": str(paths["cohort"]),
            "sha256": base.file_sha256(paths["cohort"]),
            "sample_alignment_sha256": manifest["sample_alignment_sha256"],
        },
        "experts": [asdict(spec) for spec in experts],
        "router": asdict(router),
        "paper_smoothing_candidates": [asdict(spec) for spec in smoothers],
        "selection_protocol": {
            "label_free_early_exit": (
                "select identity when every candidate has the same test decisions"
            ),
            "noninvariant_branch": (
                "select only from fully nested REFNO-isolated training OOF"
            ),
            "test_labels_used_for_branch_or_selection": False,
        },
        "scope": {
            "test_rows": len(test_records),
            "test_paper_groups": len(test_refnos),
            "train_rows": len(train_indices),
            "train_paper_groups": len(train_refnos),
            "all_rows_excluded_for_test_papers": len(excluded_indices),
            "train_test_refno_overlap": 0,
        },
        "inputs": {
            name: {"path": str(path), "sha256": base.file_sha256(path)}
            for name, path in paths.items()
            if name != "pdf_dir" and name != "targets"
        },
        "feature_summary": {
            "numeric": numeric_manifest.get("matrices"),
            "own": own_manifest.get("matrices"),
            "component": component_manifest.get("matrices"),
        },
        "implementation_dependencies": {
            name: {
                "path": str(Path(module.__file__).resolve()),
                "sha256": base.file_sha256(Path(module.__file__).resolve()),
            }
            for name, module in {
                "nested_router": nested,
                "paper_consistency_v1": paper_v1,
                "paper_consistency_v2": paper_v2,
                "cohort_builder": cohort,
            }.items()
        },
        "runtime_boundary": {
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "test_refno_rows_excluded_from_every_fit": True,
            "test_labels_not_used_for_model_or_smoother_selection": True,
            "refno_used_only_for_group_isolation_and_paper_smoothing": True,
        },
        "canonical_runtime": runtime,
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    freeze_path = output_dir / "candidate_freeze.json"
    base.atomic_json(freeze_path, freeze)

    train_resultnos = {full_records[int(index)]["RESULTNO"] for index in train_indices}
    labels_by_resultno = base.load_training_labels(paths["targets"], train_resultnos)
    training_labels = np.full(len(full_records), -1, dtype=np.int8)
    for index in train_indices:
        training_labels[int(index)] = labels_by_resultno[
            full_records[int(index)]["RESULTNO"]
        ]
    if np.any(training_labels[train_indices] < 0) or np.any(
        training_labels[excluded_indices] >= 0
    ):
        raise AssertionError("test labels entered the prediction label array")

    canonical_targets = [
        base.canonical_target_identity(record["ROCKNAME"]) for record in full_records
    ]
    train_groups = [full_records[int(index)]["REFNO"] for index in train_indices]
    test_groups = [record["REFNO"] for record in test_records]
    inner_folds = nested.deterministic_group_folds(train_indices, train_groups)
    inner_probability = np.full(
        (len(full_records), len(experts)), np.nan, dtype=np.float64
    )
    expert_inner_audit: list[dict[str, Any]] = []
    for inner_fold, (inner_fit, inner_validation) in enumerate(inner_folds, start=1):
        fit_refnos = {full_records[int(index)]["REFNO"] for index in inner_fit}
        validation_refnos = {
            full_records[int(index)]["REFNO"] for index in inner_validation
        }
        if fit_refnos & validation_refnos or (fit_refnos | validation_refnos) & test_refnos:
            raise AssertionError("random300 expert inner cross-fit group leakage")
        weights = base.inverse_target_multiplicity_weights(
            inner_fit, full_records, canonical_targets
        )
        for column, expert in enumerate(experts):
            fitted = nested.fit_expert(
                expert,
                matrices[expert.matrix],
                inner_fit,
                training_labels,
                weights,
            )
            inner_probability[inner_validation, column] = fitted.predict(
                matrices[expert.matrix][inner_validation]
            )
        expert_inner_audit.append(
            {
                "inner_fold": inner_fold,
                "fit_rows": int(len(inner_fit)),
                "validation_rows": int(len(inner_validation)),
                "fit_validation_refno_overlap": 0,
                "test_refno_overlap": 0,
            }
        )
        print(
            f"inner expert fold {inner_fold}: fit={len(inner_fit)}, "
            f"validation={len(inner_validation)}",
            flush=True,
        )
    if not np.isfinite(inner_probability[train_indices]).all():
        raise AssertionError("random300 expert inner OOF is incomplete")

    train_weights = base.inverse_target_multiplicity_weights(
        train_indices, full_records, canonical_targets
    )
    test_expert_probability = np.zeros(
        (len(test_indices), len(experts)), dtype=np.float64
    )
    for column, expert in enumerate(experts):
        fitted = nested.fit_expert(
            expert,
            matrices[expert.matrix],
            train_indices,
            training_labels,
            train_weights,
        )
        test_expert_probability[:, column] = fitted.predict(
            matrices[expert.matrix][test_indices]
        )
    identity = nested.fit_predict_router(
        router,
        inner_probability[train_indices],
        test_expert_probability,
        own_arrays["own_target"][train_indices],
        own_arrays["own_target"][test_indices],
        component_arrays["component_bridge"][train_indices],
        component_arrays["component_bridge"][test_indices],
        training_labels[train_indices],
        train_weights,
    )

    invariant, decision_audit, fixed_candidates = paper_v2.label_free_decision_invariance(
        identity, test_groups, smoothers
    )
    router_level_audit: list[dict[str, Any]] = []
    selection_metrics = None
    if invariant:
        selected = smoothers[0]
        selection_branch = "LABEL_FREE_DECISION_INVARIANCE_EARLY_EXIT"
    else:
        router_oof, router_level_audit = paper_v1.fit_router_level_oof(
            spec=router,
            outer_train=train_indices,
            router_folds=inner_folds,
            experts=experts,
            matrices=matrices,
            full_labels=training_labels,
            full_records=full_records,
            canonical_targets=canonical_targets,
            own_target=own_arrays["own_target"],
            component_bridge=component_arrays["component_bridge"],
        )
        selected, selection_metrics = paper_v1.select_smoothing(
            training_labels[train_indices],
            router_oof[train_indices],
            train_groups,
            smoothers,
        )
        selection_branch = "FULLY_NESTED_TRAINING_OOF_SELECTION"
    probabilities = {
        "router_identity": identity,
        "nested_selected": fixed_candidates[selected.name],
        **{
            spec.name: fixed_candidates[spec.name]
            for spec in smoothers[1:]
        },
    }
    expert_diagnostic_probabilities = {
        expert.name: test_expert_probability[:, column]
        for column, expert in enumerate(experts)
    }
    expert_diagnostic_probabilities.update(
        {
            "fixed_expert_mean": test_expert_probability.mean(axis=1),
            "fixed_expert_median": np.median(test_expert_probability, axis=1),
        }
    )

    probability_path = output_dir / "holdout_probabilities.npz"
    write_probability_archive(probability_path, probabilities)
    expert_probability_path = output_dir / "expert_diagnostic_probabilities.npz"
    write_probability_archive(
        expert_probability_path, expert_diagnostic_probabilities
    )
    prediction_receipt = {
        "schema_version": "r3_random300_extension_prediction_receipt_v1",
        "status": "PREDICTIONS_FROZEN_BEFORE_TEST_SCORING",
        "cohort_sha256": base.file_sha256(paths["cohort"]),
        "candidate_freeze_sha256": base.file_sha256(freeze_path),
        "probabilities": {
            "path": str(probability_path),
            "sha256": base.file_sha256(probability_path),
        },
        "expert_diagnostic_probabilities": {
            "path": str(expert_probability_path),
            "sha256": base.file_sha256(expert_probability_path),
            "used_for_selection": False,
        },
        "selected_smoother": selected.name,
        "selection_branch": selection_branch,
        "test_labels_present": False,
        "rows": [
            {
                "sample_index": index,
                **record,
                "probability": float(probabilities["nested_selected"][index]),
                "prediction": int(probabilities["nested_selected"][index] >= 0.5),
            }
            for index, record in enumerate(test_records)
        ],
    }
    receipt_path = output_dir / "prediction_receipt.json"
    base.atomic_json(receipt_path, prediction_receipt)

    test_resultnos = {record["RESULTNO"] for record in test_records}
    test_labels_by_resultno = base.load_training_labels(paths["targets"], test_resultnos)
    test_labels = np.asarray(
        [test_labels_by_resultno[record["RESULTNO"]] for record in test_records],
        dtype=np.int8,
    )
    results: dict[str, Any] = {}
    for name, values in probabilities.items():
        results[name] = {
            "metrics": base.metric_record(test_labels, values),
            "transition_vs_identity": nested.transition_record(
                test_labels, identity, values
            ),
        }
    expert_diagnostics = {
        name: base.metric_record(test_labels, values)
        for name, values in expert_diagnostic_probabilities.items()
    }
    selected_probability = probabilities["nested_selected"]
    selected_prediction = (selected_probability >= 0.5).astype(np.int8)
    errors = [
        {
            "sample_index": int(index),
            **test_records[int(index)],
            "label": int(test_labels[int(index)]),
            "prediction": int(selected_prediction[int(index)]),
            "probability": float(selected_probability[int(index)]),
            "error_type": "FP" if selected_prediction[int(index)] == 1 else "FN",
        }
        for index in np.flatnonzero(selected_prediction != test_labels)
    ]
    selected_metrics = results["nested_selected"]["metrics"]
    report = {
        "schema_version": "r3_random300_extension_paper_holdout_v1",
        "status": "complete",
        "scope": {
            "evaluated_rows": len(test_records),
            "evaluated_paper_groups": len(test_refnos),
            "all_evaluated_rows_have_local_pdf": True,
            "training_rows": len(train_indices),
            "training_paper_groups": len(train_refnos),
            "all_rows_excluded_for_test_papers": len(excluded_indices),
            "additional_sibling_rows_excluded": len(excluded_indices) - len(test_records),
        },
        "compliance_audit": {
            "cohort_frozen_without_label_access": True,
            "cohort_papers_absent_from_canonical_exact300": True,
            "all_test_refno_rows_excluded_from_every_fit": True,
            "train_test_refno_overlap": 0,
            "prediction_receipt_written_before_test_scoring": True,
            "test_labels_used_for_model_or_smoother_selection": False,
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
        },
        "inputs": freeze["inputs"],
        "cohort": freeze["cohort"],
        "candidate_freeze": {
            "path": str(freeze_path),
            "sha256": base.file_sha256(freeze_path),
        },
        "prediction_receipt": {
            "path": str(receipt_path),
            "sha256": base.file_sha256(receipt_path),
        },
        "evaluation_labels": {
            "path": str(paths["targets"]),
            "sha256": base.file_sha256(paths["targets"]),
            "opened_for_test_scoring_after_prediction_receipt": True,
        },
        "model_audit": {
            "experts": [asdict(spec) for spec in experts],
            "router": asdict(router),
            "expert_inner_cross_fitting": expert_inner_audit,
            "selection_branch": selection_branch,
            "decision_invariance": invariant,
            "label_free_decision_audit": decision_audit,
            "router_level_cross_fitting": router_level_audit,
            "selected_smoother": selected.name,
            "selection_metrics": selection_metrics,
        },
        "models": results,
        "posthoc_expert_diagnostics": {
            "selection_effect": "NONE_DIAGNOSTIC_ONLY",
            "metrics": expert_diagnostics,
        },
        "best_observed": {
            "strategy": "nested_selected",
            **selected_metrics,
        },
        "errors": errors,
        "extension_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "passed": bool(selected_metrics["accuracy"] >= TARGET_ACCURACY),
            "decision": (
                "RANDOM300_EXTENSION_TARGET_REACHED"
                if selected_metrics["accuracy"] >= TARGET_ACCURACY
                else "RANDOM300_EXTENSION_TARGET_NOT_REACHED"
            ),
        },
        "interpretation_limit": (
            "Fit-time paper holdout from the frozen development partition; not a "
            "prospectively collected external corpus."
        ),
        "artifacts": {
            "probabilities": {
                "path": str(probability_path),
                "sha256": base.file_sha256(probability_path),
            },
            "expert_diagnostic_probabilities": {
                "path": str(expert_probability_path),
                "sha256": base.file_sha256(expert_probability_path),
            }
        },
    }
    report_path = output_dir / "report.json"
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "scope": report["scope"],
                "best_observed": report["best_observed"],
                "selected_smoother": selected.name,
                "selection_branch": selection_branch,
                "extension_gate": report["extension_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
