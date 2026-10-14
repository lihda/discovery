#!/usr/bin/env python3
"""Canonical exact300 R3 screen of frozen PDF metadata/structure features."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_exact300_auxiliary_numeric_cache as numeric_builder  # noqa: E402
import build_r3_exact300_pdf_metadata_cache as metadata_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_component_bridge_auxiliary as component_evaluation  # noqa: E402
import experiment_r3_exact300_own_study_auxiliary as shared_evaluation  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/pdf_metadata_auxiliary_v1"
TARGET_ACCURACY = 0.90
REFERENCE_ACCURACY = 0.87666667
RANDOM_STATE = 9357


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    matrix: str
    model: str
    uses_pdf_metadata: bool


def candidate_specs() -> tuple[CandidateSpec, ...]:
    spec = CandidateSpec
    return (
        spec("control_component_all_rf8", "component_all", "rf8", False),
        spec("control_component_all_hgb16", "component_all", "hgb16", False),
        spec("metadata_only_rf8", "metadata_only", "rf8", True),
        spec("metadata_only_hgb16", "metadata_only", "hgb16", True),
        spec("combined_metadata_rf4", "combined_metadata", "rf4", True),
        spec("combined_metadata_rf8", "combined_metadata", "rf8", True),
        spec("combined_metadata_rf12", "combined_metadata", "rf12", True),
        spec("combined_metadata_hgb10", "combined_metadata", "hgb10", True),
        spec("combined_metadata_hgb16", "combined_metadata", "hgb16", True),
        spec("component_all_metadata_rf4", "component_all_metadata", "rf4", True),
        spec("component_all_metadata_rf8", "component_all_metadata", "rf8", True),
        spec("component_all_metadata_rf12", "component_all_metadata", "rf12", True),
        spec("component_all_metadata_hgb10", "component_all_metadata", "hgb10", True),
        spec("component_all_metadata_hgb16", "component_all_metadata", "hgb16", True),
        spec("component_bridge_metadata_rf8", "component_bridge_metadata", "rf8", True),
        spec("component_bridge_metadata_hgb10", "component_bridge_metadata", "hgb10", True),
    )


def make_model(kind: str) -> Any:
    if kind == "hgb10":
        return HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    if kind == "hgb16":
        return HistGradientBoostingClassifier(
            max_iter=240,
            learning_rate=0.04,
            max_leaf_nodes=7,
            min_samples_leaf=16,
            l2_regularization=0.5,
            random_state=RANDOM_STATE,
        )
    if kind == "rf4":
        return RandomForestClassifier(
            n_estimators=600,
            max_features=0.25,
            min_samples_leaf=4,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if kind == "rf8":
        return RandomForestClassifier(
            n_estimators=600,
            max_features=0.5,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if kind == "rf12":
        return RandomForestClassifier(
            n_estimators=600,
            max_features=0.33,
            min_samples_leaf=12,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"unknown model: {kind}")


def validate_metadata_cache(
    cache_path: Path,
    manifest_path: Path,
    expected_rows: int,
    full_alignment_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    output = manifest.get("output", {})
    builder = manifest.get("builder", {})
    if (
        manifest.get("schema_version") != "r3_exact300_pdf_metadata_cache_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment_sha256
        or output.get("sha256") != base.file_sha256(cache_path)
        or builder.get("sha256") != base.file_sha256(Path(str(builder.get("path", ""))))
        or boundary.get("labels_read") is not False
        or boundary.get("targets_workbook_read") is not False
        or boundary.get("pmagresult_or_rockmag_read") is not False
        or boundary.get("prior_predictions_or_evaluations_read") is not False
        or boundary.get("identifiers_used_only_for_pdf_alignment") is not True
        or boundary.get("cache_contains_identifiers") is not False
        or boundary.get("cache_contains_paths_or_filenames") is not False
        or boundary.get("content_metadata_title_author_subject_keywords_ignored") is not True
    ):
        raise ValueError("PDF metadata cache manifest does not prove a label-free build")
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name], dtype=np.float32) for name in payload.files}
    audit = metadata_builder.validate_arrays(arrays, expected_rows)
    if audit != manifest.get("matrices"):
        raise ValueError("PDF metadata cache differs from its frozen manifest")
    if manifest.get("feature_names") != metadata_builder.feature_names():
        raise ValueError("PDF metadata feature names differ from the frozen schema")
    return arrays["pdf_metadata"], manifest


def build_matrices(
    numeric_arrays: Mapping[str, np.ndarray],
    component_arrays: Mapping[str, np.ndarray],
    metadata: np.ndarray,
) -> dict[str, np.ndarray]:
    matrices = {
        "component_all": component_arrays["combined_plus_component_all"],
        "metadata_only": metadata,
        "combined_metadata": np.concatenate((numeric_arrays["combined"], metadata), axis=1),
        "component_all_metadata": np.concatenate(
            (component_arrays["combined_plus_component_all"], metadata), axis=1
        ),
        "component_bridge_metadata": np.concatenate(
            (component_arrays["combined_plus_component_bridge"], metadata), axis=1
        ),
    }
    matrices = {name: np.asarray(value, dtype=np.float32) for name, value in matrices.items()}
    row_counts = {value.shape[0] for value in matrices.values()}
    if len(row_counts) != 1 or any(value.ndim != 2 or not np.isfinite(value).all() for value in matrices.values()):
        raise ValueError("candidate matrices are not finite aligned 2D arrays")
    return matrices


def staged_gate_receipt(
    output_dir: Path,
    specs: Sequence[CandidateSpec],
    metadata_cache_path: Path,
    metadata_manifest_path: Path,
) -> dict[str, Any]:
    report_path = output_dir / "folds_1_2_3_report.json"
    freeze_path = output_dir / "candidate_freeze_folds_1_2_3.json"
    if not report_path.is_file() or not freeze_path.is_file():
        raise ValueError("full evaluation requires a completed first-three-fold gate")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version") != "r3_exact300_pdf_metadata_auxiliary_v1"
        or report.get("mode") != "staged_screen"
        or report.get("scope", {}).get("evaluated_folds") != [1, 2, 3]
        or report.get("candidate_freeze", {}).get("sha256") != base.file_sha256(freeze_path)
        or freeze.get("candidates") != [asdict(spec) for spec in specs]
        or freeze.get("inputs", {}).get("metadata_cache", {}).get("sha256")
        != base.file_sha256(metadata_cache_path)
        or freeze.get("inputs", {}).get("metadata_manifest", {}).get("sha256")
        != base.file_sha256(metadata_manifest_path)
        or report.get("continuation_gate", {}).get("continue_to_full_five_folds") is not True
    ):
        raise ValueError("first-three-fold metadata gate did not authorize full evaluation")
    return {
        "report": {"path": str(report_path), "sha256": base.file_sha256(report_path)},
        "candidate_freeze": {"path": str(freeze_path), "sha256": base.file_sha256(freeze_path)},
    }


def fold3_negative_diagnostic(
    labels: np.ndarray,
    fold_ids: np.ndarray,
    control_probability: np.ndarray,
    candidate_probability: np.ndarray,
) -> dict[str, int | float | None]:
    subset = (fold_ids == 3) & (labels == 0) & (control_probability >= 0.5)
    count = int(subset.sum())
    recovered = int(((candidate_probability < 0.5) & subset).sum())
    return {
        "definition": "fold=3, label=0, canonical component control predicted positive",
        "rows": count,
        "candidate_recovered_as_negative": recovered,
        "recovery_rate": round(recovered / count, 8) if count else None,
    }


def evaluate(
    *,
    args: argparse.Namespace,
    specs: Sequence[CandidateSpec],
    matrices: Mapping[str, np.ndarray],
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_lookup: Mapping[tuple[str, str, str], int],
    fold_ids: np.ndarray,
    full_coverage: Mapping[str, Any],
    exact_coverage: Mapping[str, Any],
    metadata_manifest: Mapping[str, Any],
    metadata_cache_path: Path,
    metadata_manifest_path: Path,
    component_cache_path: Path,
    component_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
    prior_gate = (
        staged_gate_receipt(output_dir, specs, metadata_cache_path, metadata_manifest_path)
        if full_run
        else None
    )
    freeze = {
        "schema_version": "r3_exact300_pdf_metadata_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "evaluated_folds": list(args.folds),
        "candidates": [asdict(spec) for spec in specs],
        "threshold": 0.5,
        "inputs": {
            "metadata_cache": {"path": str(metadata_cache_path), "sha256": base.file_sha256(metadata_cache_path)},
            "metadata_manifest": {"path": str(metadata_manifest_path), "sha256": base.file_sha256(metadata_manifest_path)},
            "component_cache": {"path": str(component_cache_path), "sha256": base.file_sha256(component_cache_path)},
            "component_manifest": {"path": str(component_manifest_path), "sha256": base.file_sha256(component_manifest_path)},
            "fold_manifest": auxiliary.FOLD_MANIFEST_SHA256,
            "staged_gate": prior_gate,
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "pmagresult_or_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_feature": False,
            "identifiers_used_as_predictive_features": False,
        },
        "script": {"path": str(Path(__file__).resolve()), "sha256": base.file_sha256(Path(__file__).resolve())},
    }
    freeze_path = output_dir / (
        "candidate_freeze.json"
        if full_run
        else "candidate_freeze_folds_" + "_".join(map(str, args.folds)) + ".json"
    )
    base.atomic_json(freeze_path, freeze)

    labels_by_resultno = base.load_training_labels(
        args.targets.resolve(), {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray([labels_by_resultno[record["RESULTNO"]] for record in full_records], dtype=np.int8)
    exact_labels = np.asarray([labels_by_resultno[record["RESULTNO"]] for record in exact_records], dtype=np.int8)
    probabilities = {
        spec.name: np.full(auxiliary.EXPECTED_EXACT_ROWS, np.nan, dtype=np.float64)
        for spec in specs
    }
    fold_audit: list[dict[str, Any]] = []
    canonical_targets = [base.canonical_target_identity(record["ROCKNAME"]) for record in full_records]
    for fold in args.folds:
        exact_test = np.flatnonzero(fold_ids == fold)
        train_full, test_full, test_refnos = auxiliary.auxiliary_split(
            full_records, exact_records, exact_test, full_lookup
        )
        weights = base.inverse_target_multiplicity_weights(train_full, full_records, canonical_targets)
        for spec in specs:
            model = make_model(spec.model)
            model.fit(matrices[spec.matrix][train_full], full_labels[train_full], sample_weight=weights)
            probabilities[spec.name][exact_test] = model.predict_proba(matrices[spec.matrix][test_full])[:, 1]
        train_refnos = {full_records[int(index)]["REFNO"] for index in train_full}
        fold_audit.append(
            {
                "fold": fold,
                "test_rows": int(len(exact_test)),
                "test_papers": len(test_refnos),
                "train_rows": int(len(train_full)),
                "train_papers": len(train_refnos),
                "train_test_refno_overlap": len(train_refnos & test_refnos),
                "excluded_full_rows": int(len(full_records) - len(train_full)),
                "all_rows_of_each_test_refno_excluded": not bool(train_refnos & test_refnos),
            }
        )
        print(f"fold {fold}: candidates={len(specs)}, test_rows={len(exact_test)}", flush=True)

    covered = np.isin(fold_ids, np.asarray(args.folds, dtype=np.int8))
    preliminary: dict[str, dict[str, Any]] = {}
    for spec in specs:
        values = probabilities[spec.name]
        if not np.isfinite(values[covered]).all():
            raise AssertionError(f"candidate missing requested fold: {spec.name}")
        preliminary[spec.name] = {
            "spec": asdict(spec),
            "aggregate": base.metric_record(exact_labels[covered], values[covered]),
            "folds": [
                {"fold": fold, **base.metric_record(exact_labels[fold_ids == fold], values[fold_ids == fold])}
                for fold in args.folds
            ],
        }
    controls = [spec.name for spec in specs if not spec.uses_pdf_metadata]
    metadata_candidates = [spec.name for spec in specs if spec.uses_pdf_metadata]
    rank_key = lambda name: (
        preliminary[name]["aggregate"]["accuracy"],
        preliminary[name]["aggregate"]["macro_f1"],
        preliminary[name]["aggregate"]["balanced_accuracy"],
    )
    control_name = max(controls, key=rank_key)
    metadata_name = max(metadata_candidates, key=rank_key)
    ranking = sorted(preliminary, key=rank_key, reverse=True)
    control_probability = probabilities[control_name]
    results: dict[str, Any] = {}
    for spec in specs:
        values = probabilities[spec.name]
        result = dict(preliminary[spec.name])
        result["transition_vs_best_control"] = shared_evaluation.transition_record(
            exact_labels[covered], control_probability[covered], values[covered]
        )
        if 3 in args.folds:
            result["fold3_strong_evidence_label_zero"] = fold3_negative_diagnostic(
                exact_labels, fold_ids, control_probability, values
            )
        results[spec.name] = result

    control_accuracy = float(results[control_name]["aggregate"]["accuracy"])
    metadata_accuracy = float(results[metadata_name]["aggregate"]["accuracy"])
    metadata_non_regressing = metadata_accuracy >= control_accuracy
    oof_record = None
    if full_run:
        oof_path = output_dir / "oof_probabilities.npz"
        shared_evaluation.write_probabilities(oof_path, probabilities)
        oof_record = {"path": str(oof_path), "sha256": base.file_sha256(oof_path)}
    report = {
        "schema_version": "r3_exact300_pdf_metadata_auxiliary_v1",
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
            "metadata_cache_frozen_for_all_2172_rows_before_labels": True,
            "candidate_receipt_written_before_label_access": True,
            "all_outer_test_refno_rows_excluded_from_training": True,
            "pmagresult_or_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_feature": False,
            "identifiers_used_as_predictive_features": False,
            "pdf_path_or_filename_used_as_predictive_features": False,
            "model_threshold": 0.5,
        },
        "feature_summary": {
            "metadata_feature_count": len(metadata_builder.feature_names()),
            "metadata_feature_audit": metadata_manifest.get("feature_audit"),
            "matrix_shapes": {name: list(matrix.shape) for name, matrix in matrices.items()},
        },
        "candidate_freeze": {"path": str(freeze_path), "sha256": base.file_sha256(freeze_path)},
        "staged_gate_receipt": prior_gate,
        "outer_fold_audit": fold_audit,
        "ranking": ranking,
        "best_observed": {"strategy": ranking[0], **results[ranking[0]]},
        "best_control": {"strategy": control_name, **results[control_name]},
        "best_metadata_candidate": {"strategy": metadata_name, **results[metadata_name]},
        "models": results,
        "continuation_gate": {
            "rule": "best metadata candidate aggregate accuracy must be >= best same-run control on folds 1-3",
            "control_strategy": control_name,
            "control_accuracy_on_same_folds": control_accuracy,
            "metadata_strategy": metadata_name,
            "metadata_accuracy_on_same_folds": metadata_accuracy,
            "non_regressing": metadata_non_regressing,
            "continue_to_full_five_folds": bool(not full_run and metadata_non_regressing),
        },
        "promotion_gate": {
            "reference_accuracy": REFERENCE_ACCURACY,
            "target_accuracy": TARGET_ACCURACY,
            "full_coverage_required": True,
            "improves_reference": bool(full_run and results[ranking[0]]["aggregate"]["accuracy"] > REFERENCE_ACCURACY),
            "passed": bool(full_run and results[ranking[0]]["aggregate"]["accuracy"] >= TARGET_ACCURACY),
            "decision": (
                "EXACT300_TARGET_REACHED"
                if full_run and results[ranking[0]]["aggregate"]["accuracy"] >= TARGET_ACCURACY
                else "STAGED_GATE_PASSED"
                if not full_run and metadata_non_regressing
                else "STAGED_GATE_FAILED"
                if not full_run
                else "EXACT300_TARGET_NOT_REACHED"
            ),
        },
        "artifacts": {"oof_probabilities": oof_record},
    }
    report_path = output_dir / (
        "report.json" if full_run else "folds_" + "_".join(map(str, args.folds)) + "_report.json"
    )
    base.atomic_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-cache", type=Path, default=metadata_builder.DEFAULT_CACHE)
    parser.add_argument("--metadata-manifest", type=Path, default=metadata_builder.DEFAULT_MANIFEST)
    parser.add_argument("--component-cache", type=Path, default=component_evaluation.component_builder.DEFAULT_CACHE)
    parser.add_argument("--component-manifest", type=Path, default=component_evaluation.component_builder.DEFAULT_MANIFEST)
    parser.add_argument("--numeric-cache", type=Path, default=numeric_builder.DEFAULT_CACHE)
    parser.add_argument("--numeric-manifest", type=Path, default=numeric_builder.DEFAULT_MANIFEST)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--folds", type=shared_evaluation.parse_folds, default=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    metadata_cache_path = args.metadata_cache.resolve()
    metadata_manifest_path = args.metadata_manifest.resolve()
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
        raise ValueError("full auxiliary identity hash mismatch")
    if base.file_sha256(fold_manifest_path) != auxiliary.FOLD_MANIFEST_SHA256:
        raise ValueError("canonical fold manifest hash mismatch")
    if base.file_sha256(targets_path) != auxiliary.TARGETS_SHA256:
        raise ValueError("targets workbook hash mismatch")
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    full_lookup = auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(fold_manifest_path, exact_records)
    full_alignment = auxiliary.sha256_records(full_records)
    numeric_arrays, numeric_manifest = numeric_sweep.validate_cache(numeric_cache_path, numeric_manifest_path)
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from auxiliary identity")
    component_arrays, _component_manifest = component_evaluation.validate_component_cache(
        component_cache_path, component_manifest_path, len(full_records), full_alignment
    )
    metadata, metadata_manifest = validate_metadata_cache(
        metadata_cache_path, metadata_manifest_path, len(full_records), full_alignment
    )
    matrices = build_matrices(numeric_arrays, component_arrays, metadata)
    specs = candidate_specs()
    if any(spec.matrix not in matrices for spec in specs):
        raise ValueError("candidate matrix is absent")
    report = evaluate(
        args=args,
        specs=specs,
        matrices=matrices,
        full_records=full_records,
        exact_records=exact_records,
        full_lookup=full_lookup,
        fold_ids=fold_ids,
        full_coverage=full_coverage,
        exact_coverage=exact_coverage,
        metadata_manifest=metadata_manifest,
        metadata_cache_path=metadata_cache_path,
        metadata_manifest_path=metadata_manifest_path,
        component_cache_path=component_cache_path,
        component_manifest_path=component_manifest_path,
        output_dir=output_dir,
    )
    print(
        json.dumps(
            {
                "best_observed": report["best_observed"],
                "best_metadata_candidate": report["best_metadata_candidate"],
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
