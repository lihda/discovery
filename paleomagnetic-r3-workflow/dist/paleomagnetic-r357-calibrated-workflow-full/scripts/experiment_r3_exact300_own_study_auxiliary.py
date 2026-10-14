#!/usr/bin/env python3
"""Canonical exact300 R3 screen of source-derived own-study attribution features."""

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
import build_r3_exact300_own_study_cache as own_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_OWN_CACHE = own_builder.DEFAULT_CACHE
DEFAULT_OWN_MANIFEST = own_builder.DEFAULT_MANIFEST
DEFAULT_NUMERIC_CACHE = numeric_builder.DEFAULT_CACHE
DEFAULT_NUMERIC_MANIFEST = numeric_builder.DEFAULT_MANIFEST
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/own_study_auxiliary_v1"
TARGET_ACCURACY = 0.90
RANDOM_STATE = 9357


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    matrix: str
    model: str


def candidate_specs() -> tuple[CandidateSpec, ...]:
    specs = [
        CandidateSpec("control_combined_hgb10", "combined", "hgb10"),
        CandidateSpec(
            "curation_plus_own_article_hgb10",
            "curation_all_plus_own_article",
            "hgb10",
        ),
        CandidateSpec(
            "curation_plus_own_target_hgb10",
            "curation_all_plus_own_target",
            "hgb10",
        ),
        CandidateSpec(
            "curation_plus_own_all_hgb10", "curation_all_plus_own_all", "hgb10"
        ),
        CandidateSpec(
            "combined_plus_own_article_hgb10",
            "combined_plus_own_article",
            "hgb10",
        ),
        CandidateSpec(
            "combined_plus_own_target_hgb10",
            "combined_plus_own_target",
            "hgb10",
        ),
        CandidateSpec(
            "combined_plus_own_all_hgb10", "combined_plus_own_all", "hgb10"
        ),
        CandidateSpec(
            "combined_plus_own_article_hgb16",
            "combined_plus_own_article",
            "hgb16",
        ),
        CandidateSpec(
            "combined_plus_own_target_hgb16",
            "combined_plus_own_target",
            "hgb16",
        ),
        CandidateSpec(
            "combined_plus_own_all_hgb16", "combined_plus_own_all", "hgb16"
        ),
        CandidateSpec("control_combined_rf8", "combined", "rf8"),
        CandidateSpec(
            "combined_plus_own_article_rf8", "combined_plus_own_article", "rf8"
        ),
        CandidateSpec(
            "combined_plus_own_target_rf8", "combined_plus_own_target", "rf8"
        ),
        CandidateSpec("combined_plus_own_all_rf8", "combined_plus_own_all", "rf8"),
    ]
    return tuple(specs)


def parse_folds(value: str) -> tuple[int, ...]:
    if value.casefold() == "all":
        return tuple(range(1, auxiliary.N_SPLITS + 1))
    try:
        folds = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "folds must be 'all' or comma-separated integers"
        ) from exc
    if not folds or any(fold < 1 or fold > auxiliary.N_SPLITS for fold in folds):
        raise argparse.ArgumentTypeError("folds must be between 1 and 5")
    return folds


def validate_own_cache(
    cache_path: Path,
    manifest_path: Path,
    expected_rows: int,
    full_alignment_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    output = manifest.get("output", {})
    builder = manifest.get("builder", {})
    if (
        manifest.get("schema_version") != "r3_exact300_own_study_cache_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("scope", {}).get("full_alignment_sha256")
        != full_alignment_sha256
        or output.get("sha256") != base.file_sha256(cache_path)
        or builder.get("sha256")
        != base.file_sha256(Path(str(builder.get("path", ""))))
        or boundary.get("labels_read") is not False
        or boundary.get("targets_workbook_read") is not False
        or boundary.get("pmagresult_or_rockmag_read") is not False
        or boundary.get("prior_predictions_or_evaluations_read") is not False
        or boundary.get("numeric_cache_contains_identifiers") is not False
    ):
        raise ValueError("own-study manifest does not prove a label-free frozen build")
    numeric_input = manifest.get("inputs", {}).get("numeric_cache", {})
    numeric_manifest_input = manifest.get("inputs", {}).get(
        "numeric_cache_manifest", {}
    )
    if (
        numeric_input.get("sha256") != numeric_sweep.CACHE_SHA256
        or numeric_manifest_input.get("sha256")
        != numeric_sweep.CACHE_MANIFEST_SHA256
    ):
        raise ValueError("own-study cache is not bound to the frozen numeric cache")
    with np.load(cache_path, allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name], dtype=np.float32) for name in payload.files
        }
    audit = own_builder.validate_arrays(arrays, expected_rows)
    if audit != manifest.get("matrices"):
        raise ValueError("own-study cache arrays differ from the frozen manifest")
    return arrays, manifest


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
    if kind == "rf8":
        return RandomForestClassifier(
            n_estimators=600,
            max_features=0.5,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"unknown model: {kind}")


def transition_record(
    labels: np.ndarray, baseline_probability: np.ndarray, candidate_probability: np.ndarray
) -> dict[str, int]:
    baseline = baseline_probability >= 0.5
    candidate = candidate_probability >= 0.5
    baseline_error = baseline != labels
    candidate_error = candidate != labels
    return {
        "baseline_errors": int(baseline_error.sum()),
        "candidate_errors": int(candidate_error.sum()),
        "errors_resolved": int((baseline_error & ~candidate_error).sum()),
        "errors_introduced": int((~baseline_error & candidate_error).sum()),
        "baseline_false_positive": int((baseline & (labels == 0)).sum()),
        "candidate_false_positive": int((candidate & (labels == 0)).sum()),
        "baseline_false_negative": int((~baseline & (labels == 1)).sum()),
        "candidate_false_negative": int((~candidate & (labels == 1)).sum()),
    }


def write_probabilities(path: Path, probabilities: Mapping[str, np.ndarray]) -> None:
    if any(
        values.shape != (auxiliary.EXPECTED_EXACT_ROWS,)
        or not np.isfinite(values).all()
        for values in probabilities.values()
    ):
        raise ValueError("full OOF artifact requires 300 finite rows per candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(values, dtype=np.float32)
                for name, values in sorted(probabilities.items())
            },
        )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--own-cache", type=Path, default=DEFAULT_OWN_CACHE)
    parser.add_argument("--own-manifest", type=Path, default=DEFAULT_OWN_MANIFEST)
    parser.add_argument("--numeric-cache", type=Path, default=DEFAULT_NUMERIC_CACHE)
    parser.add_argument("--numeric-manifest", type=Path, default=DEFAULT_NUMERIC_MANIFEST)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("1,2,3"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    own_cache_path = args.own_cache.resolve()
    own_manifest_path = args.own_manifest.resolve()
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
    numeric_arrays, numeric_manifest = numeric_sweep.validate_cache(
        numeric_cache_path, numeric_manifest_path
    )
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from auxiliary identity")
    own_arrays, own_manifest = validate_own_cache(
        own_cache_path, own_manifest_path, len(full_records), full_alignment
    )
    matrices = {**numeric_arrays, **own_arrays}
    specs = candidate_specs()
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)) or any(spec.matrix not in matrices for spec in specs):
        raise ValueError("candidate declaration is invalid")

    # This receipt is committed before the first R3 label is opened.
    candidate_freeze = {
        "schema_version": "r3_exact300_own_study_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "evaluated_folds": list(args.folds),
        "candidates": [asdict(spec) for spec in specs],
        "threshold": 0.5,
        "inputs": {
            "own_cache": {
                "path": str(own_cache_path),
                "sha256": base.file_sha256(own_cache_path),
            },
            "own_manifest": {
                "path": str(own_manifest_path),
                "sha256": base.file_sha256(own_manifest_path),
            },
            "numeric_cache": {
                "path": str(numeric_cache_path),
                "sha256": base.file_sha256(numeric_cache_path),
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
        if args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
        else "candidate_freeze_folds_" + "_".join(map(str, args.folds)) + ".json"
    )
    base.atomic_json(freeze_path, candidate_freeze)

    labels_by_resultno = base.load_training_labels(
        targets_path, {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in full_records],
        dtype=np.int8,
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
                "excluded_full_rows": int(len(full_records) - len(train_full)),
            }
        )
        print(
            f"fold {fold}: candidates={len(specs)}, train_rows={len(train_full)}, "
            f"test_rows={len(exact_test)}",
            flush=True,
        )

    covered = np.isin(fold_ids, np.asarray(args.folds, dtype=np.int8))
    baseline_name = "control_combined_hgb10"
    baseline_probability = probabilities[baseline_name][covered]
    results: dict[str, Any] = {}
    for spec in specs:
        values = probabilities[spec.name]
        if not np.isfinite(values[covered]).all():
            raise AssertionError(f"candidate missing a requested fold: {spec.name}")
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
            "transition_vs_hgb_control": transition_record(
                exact_labels[covered], baseline_probability, values[covered]
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
    control_names = tuple(spec.name for spec in specs if spec.name.startswith("control_"))
    control_name = max(
        control_names,
        key=lambda name: (
            results[name]["aggregate"]["accuracy"],
            results[name]["aggregate"]["macro_f1"],
        ),
    )
    control_accuracy = float(results[control_name]["aggregate"]["accuracy"])
    best_accuracy = float(results[best]["aggregate"]["accuracy"])
    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
    oof_record = None
    if full_run:
        oof_path = output_dir / "oof_probabilities.npz"
        write_probabilities(oof_path, probabilities)
        oof_record = {"path": str(oof_path), "sha256": base.file_sha256(oof_path)}
    report = {
        "schema_version": "r3_exact300_own_study_auxiliary_v1",
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
        "inputs": {
            "own_cache": candidate_freeze["inputs"]["own_cache"],
            "own_manifest": candidate_freeze["inputs"]["own_manifest"],
            "numeric_cache": candidate_freeze["inputs"]["numeric_cache"],
            "targets_postfreeze": auxiliary.TARGETS_SHA256,
        },
        "feature_summary": {
            "event_summary": own_manifest.get("event_summary"),
            "matrices": own_manifest.get("matrices"),
        },
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
            "all_control_strategies": list(control_names),
            "control_accuracy_on_same_folds": control_accuracy,
            "best_accuracy_on_same_folds": best_accuracy,
            "strict_improvement": best_accuracy > control_accuracy,
            "continue_to_full_five_folds": (
                not full_run and best_accuracy > control_accuracy
            ),
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
        "artifacts": {
            "oof_probabilities": oof_record,
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
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
                "report": str(report_path),
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
