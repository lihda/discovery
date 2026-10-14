#!/usr/bin/env python3
"""Evaluate frozen low-variance R3 expert means on the final sealed cohort."""

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
import freeze_r3_fourth300_validation_cohort as fourth300  # noqa: E402
import freeze_r3_random300_extension_cohort as random300  # noqa: E402
import freeze_r3_third300_validation_cohort as third300  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_fourth300_validation_v1/mean_holdout"
FOURTH_MANIFEST_SHA256 = (
    "1ad4d2a61fc8fbb09c67730d1b2c5a433abca3f5765211469615cd5a4c0cad5d"
)
PRIMARY_STRATEGY = "mean6"
TARGET_ACCURACY = 0.90


def write_probability_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    payload = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in sorted(arrays.items())
    }
    if any(
        values.shape != (fourth300.SAMPLE_ROWS,) or not np.isfinite(values).all()
        for values in payload.values()
    ):
        raise ValueError("fourth300 requires 300 finite candidate probabilities")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, default=fourth300.OUTPUT)
    parser.add_argument("--random300-manifest", type=Path, default=random300.OUTPUT)
    parser.add_argument("--third300-manifest", type=Path, default=third300.OUTPUT)
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
        "third300_manifest": args.third300_manifest.resolve(),
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
        "cohort": FOURTH_MANIFEST_SHA256,
        "random300_manifest": third300.RANDOM300_MANIFEST_SHA256,
        "third300_manifest": fourth300.THIRD300_MANIFEST_SHA256,
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
    third_manifest = json.loads(paths["third300_manifest"].read_text(encoding="utf-8"))
    third_records, _ = third300.validate_manifest(
        third_manifest, full_records, exact_records, random_records
    )
    cohort_manifest = json.loads(paths["cohort"].read_text(encoding="utf-8"))
    test_records, test_indices = fourth300.validate_manifest(
        cohort_manifest,
        full_records,
        [*exact_records, *random_records, *third_records],
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
        raise AssertionError("fourth300 test paper remains in training")

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
    if any(expert.matrix not in matrices for expert in experts):
        raise ValueError("a baseline expert matrix is missing")
    candidates = (
        "mean6",
        "median6",
        "trimmed_mean6",
        "episode_signature_hgb",
        "episode_mean7",
        "full_evidence_hgb",
    )
    freeze = {
        "schema_version": "r3_fourth300_mean_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_TEST_LABEL_ACCESS",
        "primary_strategy": PRIMARY_STRATEGY,
        "candidate_declaration_order": list(candidates),
        "selection_rationale": (
            "uniform mean minimized hard-router variance across three opened cohorts; "
            "episode candidates are preregistered diagnostics only"
        ),
        "experts": [asdict(expert) for expert in experts],
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
            "test_labels_used_for_candidate_selection": False,
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "all_test_refno_rows_excluded_from_every_fit": True,
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
    train_weights = base.inverse_target_multiplicity_weights(
        train_indices, full_records, canonical_targets
    )
    expert_probability = np.zeros(
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
        expert_probability[:, column] = fitted.predict(
            matrices[expert.matrix][test_indices]
        )
    mean6 = expert_probability.mean(axis=1)
    median6 = np.median(expert_probability, axis=1)
    trimmed6 = np.sort(expert_probability, axis=1)[:, 1:-1].mean(axis=1)
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
    probabilities = {
        "mean6": mean6,
        "median6": median6,
        "trimmed_mean6": trimmed6,
        "episode_signature_hgb": episode_probability,
        "episode_mean7": (6.0 * mean6 + episode_probability) / 7.0,
        "full_evidence_hgb": full_probability,
    }
    if tuple(probabilities) != candidates:
        raise AssertionError("candidate declaration changed after freeze")
    probability_path = output_dir / "holdout_probabilities.npz"
    write_probability_archive(probability_path, probabilities)
    receipt = {
        "schema_version": "r3_fourth300_mean_prediction_receipt_v1",
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
                "probability": float(mean6[index]),
                "prediction": int(mean6[index] >= 0.5),
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
                test_labels, mean6, values
            ),
        }
        for name, values in probabilities.items()
    }
    prediction = (mean6 >= 0.5).astype(np.int8)
    errors = [
        {
            "sample_index": int(index),
            **test_records[int(index)],
            "label": int(test_labels[int(index)]),
            "prediction": int(prediction[int(index)]),
            "probability": float(mean6[int(index)]),
            "error_type": "FP" if prediction[int(index)] == 1 else "FN",
        }
        for index in np.flatnonzero(prediction != test_labels)
    ]
    primary_metrics = results[PRIMARY_STRATEGY]["metrics"]
    report = {
        "schema_version": "r3_fourth300_mean_paper_holdout_v1",
        "status": "complete",
        "scope": {
            "evaluated_rows": len(test_records),
            "evaluated_paper_groups": len(test_refnos),
            "all_evaluated_rows_have_local_pdf": True,
            "training_rows": len(train_indices),
            "training_paper_groups": len(train_refnos),
            "all_rows_excluded_for_test_papers": len(excluded_indices),
        },
        "compliance_audit": {
            "fourth_cohort_disjoint_from_three_opened_benchmarks": True,
            "candidate_and_primary_strategy_frozen_before_test_labels": True,
            "prediction_receipt_written_before_test_scoring": True,
            "test_labels_used_for_model_or_candidate_selection": False,
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
        "models": results,
        "primary_result": {"strategy": PRIMARY_STRATEGY, **primary_metrics},
        "errors": errors,
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "passed": bool(primary_metrics["accuracy"] >= TARGET_ACCURACY),
            "decision": (
                "FOURTH300_TARGET_REACHED"
                if primary_metrics["accuracy"] >= TARGET_ACCURACY
                else "FOURTH300_TARGET_NOT_REACHED"
            ),
        },
        "artifacts": {
            "probabilities": {
                "path": str(probability_path),
                "sha256": base.file_sha256(probability_path),
            }
        },
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
                            "accuracy": result["metrics"]["accuracy"],
                            "macro_f1": result["metrics"]["macro_f1"],
                        }
                        for name, result in results.items()
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
