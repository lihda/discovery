#!/usr/bin/env python3
"""Evaluate preregistered conservative R3 fusion on the sealed third300 cohort."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier


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
import experiment_r3_exact300_nested_router_v2 as episode_v2  # noqa: E402
import experiment_r3_exact300_own_study_auxiliary as own_eval  # noqa: E402
import experiment_r3_exact300_paper_consistency as paper_v1  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import freeze_r3_random300_extension_cohort as random300  # noqa: E402
import freeze_r3_third300_validation_cohort as third300  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_third300_validation_v1/robust_holdout"
THIRD_MANIFEST_SHA256 = (
    "a275dd97b234a746ce3ba564157de3c7cf9766f37a445f4a8ae1b30307d2e3e1"
)
PRIMARY_STRATEGY = "conservative_soft_router_010"
HARD_ROUTER_WEIGHT = 0.10
TARGET_ACCURACY = 0.90


def write_probability_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    payload = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in sorted(arrays.items())
    }
    if any(
        values.shape != (third300.SAMPLE_ROWS,) or not np.isfinite(values).all()
        for values in payload.values()
    ):
        raise ValueError("third300 requires 300 finite probabilities per candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=third300.OUTPUT)
    parser.add_argument("--random300-manifest", type=Path, default=random300.OUTPUT)
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
    parser.add_argument("--field-cache", type=Path, default=episode_v2.DEFAULT_FIELD_CACHE)
    parser.add_argument(
        "--field-manifest", type=Path, default=episode_v2.DEFAULT_FIELD_MANIFEST
    )
    parser.add_argument("--field-freeze", type=Path, default=episode_v2.DEFAULT_FIELD_FREEZE)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = {
        "cohort": args.cohort.resolve(),
        "random300_manifest": args.random300_manifest.resolve(),
        "numeric_cache": args.numeric_cache.resolve(),
        "numeric_manifest": args.numeric_manifest.resolve(),
        "own_cache": args.own_cache.resolve(),
        "own_manifest": args.own_manifest.resolve(),
        "component_cache": args.component_cache.resolve(),
        "component_manifest": args.component_manifest.resolve(),
        "field_cache": args.field_cache.resolve(),
        "field_manifest": args.field_manifest.resolve(),
        "field_freeze": args.field_freeze.resolve(),
        "full_identity": args.full_identity.resolve(),
        "exact_identity": args.exact_identity.resolve(),
        "targets": args.targets.resolve(),
        "pdf_dir": args.pdf_dir.resolve(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    if runtime != paper_v1.CANONICAL_RUNTIME:
        raise ValueError(
            f"canonical runtime required: expected={paper_v1.CANONICAL_RUNTIME}, "
            f"observed={runtime}"
        )
    expected_hashes = {
        "cohort": THIRD_MANIFEST_SHA256,
        "random300_manifest": third300.RANDOM300_MANIFEST_SHA256,
        "full_identity": auxiliary.FULL_IDENTITY_SHA256,
        "exact_identity": auxiliary.EXACT_IDENTITY_SHA256,
        "targets": auxiliary.TARGETS_SHA256,
    }
    for name, digest in expected_hashes.items():
        if base.file_sha256(paths[name]) != digest:
            raise ValueError(f"frozen input hash mismatch: {name}")

    full_records, full_coverage = base.load_pdf_only_identity(
        paths["full_identity"], paths["pdf_dir"]
    )
    exact_records, _ = base.load_pdf_only_identity(
        paths["exact_identity"], paths["pdf_dir"]
    )
    random_manifest = json.loads(paths["random300_manifest"].read_text(encoding="utf-8"))
    random_records, _ = random300.validate_manifest(
        random_manifest, full_records, exact_records
    )
    cohort_manifest = json.loads(paths["cohort"].read_text(encoding="utf-8"))
    test_records, test_indices = third300.validate_manifest(
        cohort_manifest, full_records, exact_records, random_records
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
        raise AssertionError("third300 test paper remains in training")

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
    field_signatures, field_manifest = episode_v2.validate_field_cache(
        paths["field_cache"],
        paths["field_manifest"],
        paths["field_freeze"],
        len(full_records),
        full_alignment,
    )
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment mismatch")
    matrices = {**numeric_arrays, **own_arrays, **component_arrays}
    experts = nested.expert_specs()
    router = paper_v1.router_spec()
    if any(expert.matrix not in matrices for expert in experts):
        raise ValueError("a frozen baseline expert matrix is missing")

    component_names = json.loads(paths["component_manifest"].read_text(encoding="utf-8"))[
        "feature_names"
    ]["component_bridge"]
    direct_scope_column = component_names.index("target_direct_scope_count_log")
    candidate_declaration = [
        "mean6",
        "median6",
        "trimmed_mean6",
        "hard_router",
        PRIMARY_STRATEGY,
        "episode_signature_hgb",
        "episode_mean7",
        "full_evidence_hgb",
        "direct_scope_rescue_045",
    ]
    freeze = {
        "schema_version": "r3_third300_robust_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_TEST_LABEL_ACCESS",
        "primary_strategy": PRIMARY_STRATEGY,
        "candidate_declaration_order": candidate_declaration,
        "primary_formula": {
            "mean6_weight": 1.0 - HARD_ROUTER_WEIGHT,
            "hard_router_weight": HARD_ROUTER_WEIGHT,
            "decision_threshold": 0.5,
            "weights_selected_from_opened_development_cohorts_only": True,
        },
        "episode_enhancement": {
            "source": "frozen record-free field-signature cache",
            "runtime_pmagresult_access": False,
            "runtime_row_rockmag_access": False,
            "episode_mean7_selection_role": "diagnostic_only",
            "direct_scope_rescue": (
                "promote 0.45<=primary<0.5 only when source-derived direct target "
                "scope exists"
            ),
        },
        "experts": [asdict(expert) for expert in experts],
        "router": asdict(router),
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
            if name not in {"pdf_dir", "targets"}
        },
        "feature_summary": {
            "numeric": numeric_manifest.get("matrices"),
            "own": own_manifest.get("matrices"),
            "component": component_manifest.get("matrices"),
            "episode_signature": {
                "shape": list(field_signatures.shape),
                "sha256": field_manifest.get("matrix_sha256"),
            },
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "test_labels_used_for_candidate_or_weight_selection": False,
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "all_test_refno_rows_excluded_from_every_fit": True,
            "refno_used_only_for_group_isolation": True,
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
    if np.any(training_labels[excluded_indices] >= 0):
        raise AssertionError("test labels entered the training label array")

    canonical_targets = [
        base.canonical_target_identity(record["ROCKNAME"]) for record in full_records
    ]
    train_groups = [full_records[int(index)]["REFNO"] for index in train_indices]
    inner_folds = nested.deterministic_group_folds(train_indices, train_groups)
    inner_probability = np.full(
        (len(full_records), len(experts)), np.nan, dtype=np.float64
    )
    inner_audit: list[dict[str, Any]] = []
    for fold, (inner_fit, inner_validation) in enumerate(inner_folds, start=1):
        fit_refnos = {full_records[int(index)]["REFNO"] for index in inner_fit}
        validation_refnos = {
            full_records[int(index)]["REFNO"] for index in inner_validation
        }
        if fit_refnos & validation_refnos or (fit_refnos | validation_refnos) & test_refnos:
            raise AssertionError("third300 inner REFNO leakage")
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
        inner_audit.append(
            {
                "fold": fold,
                "fit_rows": len(inner_fit),
                "validation_rows": len(inner_validation),
                "fit_validation_refno_overlap": 0,
                "test_refno_overlap": 0,
            }
        )
        print(
            f"inner fold {fold}: fit={len(inner_fit)}, validation={len(inner_validation)}",
            flush=True,
        )
    if not np.isfinite(inner_probability[train_indices]).all():
        raise AssertionError("inner expert OOF is incomplete")

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
    hard_router = nested.fit_predict_router(
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
    mean6 = test_expert_probability.mean(axis=1)
    median6 = np.median(test_expert_probability, axis=1)
    trimmed6 = np.sort(test_expert_probability, axis=1)[:, 1:-1].mean(axis=1)
    primary = (1.0 - HARD_ROUTER_WEIGHT) * mean6 + HARD_ROUTER_WEIGHT * hard_router

    episode_spec = nested.ExpertSpec(
        "episode_signature_hgb", "field_signatures", "signature_hgb"
    )
    episode_model = episode_v2.fit_expert(
        episode_spec,
        field_signatures,
        train_indices,
        training_labels,
        train_weights,
    )
    episode_probability = episode_model.predict(field_signatures[test_indices])
    full_evidence = np.concatenate(
        (
            numeric_arrays["combined"],
            own_arrays["own_target"],
            component_arrays["component_bridge"],
            field_signatures,
        ),
        axis=1,
    )
    full_model = HistGradientBoostingClassifier(
        max_iter=260,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=12,
        l2_regularization=1.0,
        random_state=nested.RANDOM_STATE,
    )
    full_model.fit(
        full_evidence[train_indices],
        training_labels[train_indices],
        sample_weight=train_weights,
    )
    full_probability = np.asarray(
        full_model.predict_proba(full_evidence[test_indices])[:, 1],
        dtype=np.float64,
    )
    direct_scope = (
        component_arrays["component_bridge"][test_indices, direct_scope_column] > 0
    )
    direct_scope_rescue = primary.copy()
    rescue = (primary >= 0.45) & (primary < 0.5) & direct_scope
    direct_scope_rescue[rescue] = 0.500001
    probabilities = {
        "mean6": mean6,
        "median6": median6,
        "trimmed_mean6": trimmed6,
        "hard_router": hard_router,
        PRIMARY_STRATEGY: primary,
        "episode_signature_hgb": episode_probability,
        "episode_mean7": (6.0 * mean6 + episode_probability) / 7.0,
        "full_evidence_hgb": full_probability,
        "direct_scope_rescue_045": direct_scope_rescue,
    }
    if list(probabilities) != candidate_declaration:
        raise AssertionError("runtime candidate order differs from frozen declaration")
    probability_path = output_dir / "holdout_probabilities.npz"
    write_probability_archive(probability_path, probabilities)
    receipt = {
        "schema_version": "r3_third300_robust_prediction_receipt_v1",
        "status": "PREDICTIONS_FROZEN_BEFORE_TEST_SCORING",
        "primary_strategy": PRIMARY_STRATEGY,
        "cohort_sha256": base.file_sha256(paths["cohort"]),
        "candidate_freeze_sha256": base.file_sha256(freeze_path),
        "probabilities": {
            "path": str(probability_path),
            "sha256": base.file_sha256(probability_path),
        },
        "test_labels_present": False,
        "rows": [
            {
                "sample_index": index,
                **record,
                "probability": float(primary[index]),
                "prediction": int(primary[index] >= 0.5),
            }
            for index, record in enumerate(test_records)
        ],
    }
    receipt_path = output_dir / "prediction_receipt.json"
    base.atomic_json(receipt_path, receipt)

    test_resultnos = {record["RESULTNO"] for record in test_records}
    test_labels_by_resultno = base.load_training_labels(paths["targets"], test_resultnos)
    test_labels = np.asarray(
        [test_labels_by_resultno[record["RESULTNO"]] for record in test_records],
        dtype=np.int8,
    )
    results = {
        name: {
            "metrics": base.metric_record(test_labels, values),
            "transition_vs_primary": nested.transition_record(
                test_labels, primary, values
            ),
        }
        for name, values in probabilities.items()
    }
    primary_prediction = (primary >= 0.5).astype(np.int8)
    errors = [
        {
            "sample_index": int(index),
            **test_records[int(index)],
            "label": int(test_labels[int(index)]),
            "prediction": int(primary_prediction[int(index)]),
            "probability": float(primary[int(index)]),
            "error_type": "FP" if primary_prediction[int(index)] == 1 else "FN",
        }
        for index in np.flatnonzero(primary_prediction != test_labels)
    ]
    primary_metrics = results[PRIMARY_STRATEGY]["metrics"]
    report = {
        "schema_version": "r3_third300_robust_paper_holdout_v1",
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
            "third_cohort_frozen_without_label_access": True,
            "third_cohort_disjoint_from_both_opened_benchmarks": True,
            "candidate_family_and_primary_strategy_frozen_before_test_labels": True,
            "prediction_receipt_written_before_test_scoring": True,
            "test_labels_used_for_model_candidate_or_weight_selection": False,
            "all_test_refno_rows_excluded_from_every_fit": True,
            "train_test_refno_overlap": 0,
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
        },
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
            "opened_after_prediction_receipt": True,
        },
        "model_audit": {
            "inner_cross_fitting": inner_audit,
            "direct_scope_rescue_rows": int(rescue.sum()),
            "primary_formula": freeze["primary_formula"],
        },
        "models": results,
        "primary_result": {
            "strategy": PRIMARY_STRATEGY,
            **primary_metrics,
        },
        "errors": errors,
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "passed": bool(primary_metrics["accuracy"] >= TARGET_ACCURACY),
            "decision": (
                "THIRD300_TARGET_REACHED"
                if primary_metrics["accuracy"] >= TARGET_ACCURACY
                else "THIRD300_TARGET_NOT_REACHED"
            ),
        },
        "artifacts": {
            "probabilities": {
                "path": str(probability_path),
                "sha256": base.file_sha256(probability_path),
            }
        },
        "interpretation_limit": (
            "Prospectively sealed within the existing PDF-backed partition; not an "
            "independently collected external corpus."
        ),
        "source_scope": full_coverage,
    }
    report_path = output_dir / "report.json"
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "primary_result": report["primary_result"],
                "promotion_gate": report["promotion_gate"],
                "diagnostic_ranking": sorted(
                    (
                        {
                            "strategy": name,
                            "accuracy": value["metrics"]["accuracy"],
                            "macro_f1": value["metrics"]["macro_f1"],
                        }
                        for name, value in results.items()
                    ),
                    key=lambda item: (item["accuracy"], item["macro_f1"]),
                    reverse=True,
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
