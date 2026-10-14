#!/usr/bin/env python3
"""Paper consistency with label-free decision-invariance early exit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
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
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/paper_consistency_v2"
TARGET_ACCURACY = 0.90


def decision_sha256(value: np.ndarray) -> str:
    decisions = np.asarray(value, dtype=np.uint8)
    return hashlib.sha256(decisions.tobytes()).hexdigest()


def label_free_decision_invariance(
    probability: np.ndarray,
    groups: Sequence[str],
    specs: Sequence[paper_v1.SmoothingSpec],
) -> tuple[bool, dict[str, Any], dict[str, np.ndarray]]:
    candidates = {
        spec.name: paper_v1.apply_smoothing(probability, groups, spec)
        for spec in specs
    }
    identity = candidates["identity"] >= 0.5
    audit = {
        name: {
            "decision_sha256": decision_sha256(values >= 0.5),
            "rows_changed_vs_identity": int(((values >= 0.5) != identity).sum()),
        }
        for name, values in candidates.items()
    }
    invariant = all(
        item["rows_changed_vs_identity"] == 0 for item in audit.values()
    )
    return invariant, audit, candidates


def write_probabilities(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    payload = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in sorted(arrays.items())
    }
    if any(
        value.shape != (auxiliary.EXPECTED_EXACT_ROWS,)
        or not np.isfinite(value).all()
        for value in payload.values()
    ):
        raise ValueError("full v2 OOF requires 300 finite rows per candidate")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numeric-cache", type=Path, default=numeric_builder.DEFAULT_CACHE)
    parser.add_argument("--numeric-manifest", type=Path, default=numeric_builder.DEFAULT_MANIFEST)
    parser.add_argument("--own-cache", type=Path, default=own_builder.DEFAULT_CACHE)
    parser.add_argument("--own-manifest", type=Path, default=own_builder.DEFAULT_MANIFEST)
    parser.add_argument("--component-cache", type=Path, default=component_builder.DEFAULT_CACHE)
    parser.add_argument(
        "--component-manifest", type=Path, default=component_builder.DEFAULT_MANIFEST
    )
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--folds", type=nested.parse_folds, default=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    paths = {
        "numeric_cache": args.numeric_cache.resolve(),
        "numeric_manifest": args.numeric_manifest.resolve(),
        "own_cache": args.own_cache.resolve(),
        "own_manifest": args.own_manifest.resolve(),
        "component_cache": args.component_cache.resolve(),
        "component_manifest": args.component_manifest.resolve(),
        "exact_identity": args.exact_identity.resolve(),
        "full_identity": args.full_identity.resolve(),
        "fold_manifest": args.fold_manifest.resolve(),
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
            f"canonical paper-consistency runtime required: "
            f"expected={paper_v1.CANONICAL_RUNTIME}, observed={runtime}"
        )
    if base.file_sha256(paths["exact_identity"]) != auxiliary.EXACT_IDENTITY_SHA256:
        raise ValueError("exact300 identity hash mismatch")
    if base.file_sha256(paths["full_identity"]) != auxiliary.FULL_IDENTITY_SHA256:
        raise ValueError("full identity hash mismatch")
    if base.file_sha256(paths["fold_manifest"]) != auxiliary.FOLD_MANIFEST_SHA256:
        raise ValueError("canonical fold manifest hash mismatch")
    if base.file_sha256(paths["targets"]) != auxiliary.TARGETS_SHA256:
        raise ValueError("targets workbook hash mismatch")

    full_records, full_coverage = base.load_pdf_only_identity(
        paths["full_identity"], paths["pdf_dir"]
    )
    exact_records, exact_coverage = base.load_pdf_only_identity(
        paths["exact_identity"], paths["pdf_dir"]
    )
    full_lookup = auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(paths["fold_manifest"], exact_records)
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

    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
    freeze = {
        "schema_version": "r3_exact300_paper_consistency_candidate_freeze_v2",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "evaluated_folds": list(args.folds),
        "experts": [asdict(spec) for spec in experts],
        "router": asdict(router),
        "paper_smoothing_candidates": [asdict(spec) for spec in smoothers],
        "selection_protocol": {
            "step_1": "recompute outer router prediction without prior OOF input",
            "step_2": (
                "before reading outer-test labels, compare all candidate binary "
                "decisions on the outer test rows"
            ),
            "label_free_early_exit": (
                "if every candidate decision equals identity, select identity because "
                "all candidates are mathematically evaluation-equivalent"
            ),
            "noninvariant_branch": (
                "otherwise select by accuracy on fully nested router-level "
                "outer-training OOF; exact ties prefer identity"
            ),
        },
        "inputs": {
            name: {"path": str(path), "sha256": base.file_sha256(path)}
            for name, path in paths.items()
            if name != "pdf_dir"
        },
        "implementation_dependencies": {
            "nested_router_v1": {
                "path": str(Path(nested.__file__).resolve()),
                "sha256": base.file_sha256(Path(nested.__file__).resolve()),
            },
            "paper_consistency_v1": {
                "path": str(Path(paper_v1.__file__).resolve()),
                "sha256": base.file_sha256(Path(paper_v1.__file__).resolve()),
            },
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "outer_test_labels_used_for_branch_or_smoother_selection": False,
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "refno_used_only_for_group_isolation_and_paper_smoothing": True,
        },
        "canonical_runtime": runtime,
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
        paths["targets"], {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in full_records],
        dtype=np.int8,
    )
    exact_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )
    names = [
        "router_identity",
        "nested_selected",
        *[spec.name for spec in smoothers[1:]],
    ]
    probabilities = {
        name: np.full(auxiliary.EXPECTED_EXACT_ROWS, np.nan, dtype=np.float64)
        for name in names
    }
    canonical_targets = [
        base.canonical_target_identity(record["ROCKNAME"]) for record in full_records
    ]
    outer_audit: list[dict[str, Any]] = []
    for fold in args.folds:
        exact_test = np.flatnonzero(fold_ids == fold)
        outer_train, outer_test, outer_test_refnos = auxiliary.auxiliary_split(
            full_records, exact_records, exact_test, full_lookup
        )
        train_groups = [full_records[int(index)]["REFNO"] for index in outer_train]
        test_groups = [full_records[int(index)]["REFNO"] for index in outer_test]
        inner_folds = nested.deterministic_group_folds(outer_train, train_groups)
        inner_probability = np.full(
            (len(full_records), len(experts)), np.nan, dtype=np.float64
        )
        expert_inner_audit: list[dict[str, Any]] = []
        for inner_fold, (inner_fit, inner_validation) in enumerate(
            inner_folds, start=1
        ):
            fit_refnos = {full_records[int(index)]["REFNO"] for index in inner_fit}
            validation_refnos = {
                full_records[int(index)]["REFNO"] for index in inner_validation
            }
            if fit_refnos & validation_refnos or (
                fit_refnos | validation_refnos
            ) & outer_test_refnos:
                raise AssertionError("base expert inner cross-fit group leakage")
            weights = base.inverse_target_multiplicity_weights(
                inner_fit, full_records, canonical_targets
            )
            for column, expert in enumerate(experts):
                fitted = nested.fit_expert(
                    expert,
                    matrices[expert.matrix],
                    inner_fit,
                    full_labels,
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
                    "outer_test_refno_overlap": 0,
                }
            )
        if not np.isfinite(inner_probability[outer_train]).all():
            raise AssertionError("base expert inner OOF is incomplete")

        outer_weights = base.inverse_target_multiplicity_weights(
            outer_train, full_records, canonical_targets
        )
        test_expert_probability = np.zeros(
            (len(outer_test), len(experts)), dtype=np.float64
        )
        for column, expert in enumerate(experts):
            fitted = nested.fit_expert(
                expert,
                matrices[expert.matrix],
                outer_train,
                full_labels,
                outer_weights,
            )
            test_expert_probability[:, column] = fitted.predict(
                matrices[expert.matrix][outer_test]
            )
        identity = nested.fit_predict_router(
            router,
            inner_probability[outer_train],
            test_expert_probability,
            own_arrays["own_target"][outer_train],
            own_arrays["own_target"][outer_test],
            component_arrays["component_bridge"][outer_train],
            component_arrays["component_bridge"][outer_test],
            full_labels[outer_train],
            outer_weights,
        )
        invariant, decision_audit, fixed_candidates = label_free_decision_invariance(
            identity, test_groups, smoothers
        )
        selection_metrics = None
        router_level_audit: list[dict[str, Any]] = []
        if invariant:
            selected = smoothers[0]
            branch = "LABEL_FREE_DECISION_INVARIANCE_EARLY_EXIT"
        else:
            router_oof, router_level_audit = paper_v1.fit_router_level_oof(
                spec=router,
                outer_train=outer_train,
                router_folds=inner_folds,
                experts=experts,
                matrices=matrices,
                full_labels=full_labels,
                full_records=full_records,
                canonical_targets=canonical_targets,
                own_target=own_arrays["own_target"],
                component_bridge=component_arrays["component_bridge"],
            )
            selected, selection_metrics = paper_v1.select_smoothing(
                full_labels[outer_train],
                router_oof[outer_train],
                train_groups,
                smoothers,
            )
            branch = "FULLY_NESTED_OUTER_TRAINING_OOF_SELECTION"

        probabilities["router_identity"][exact_test] = identity
        probabilities["nested_selected"][exact_test] = fixed_candidates[
            selected.name
        ]
        for spec in smoothers[1:]:
            probabilities[spec.name][exact_test] = fixed_candidates[spec.name]
        outer_audit.append(
            {
                "fold": fold,
                "outer_train_rows": int(len(outer_train)),
                "outer_test_rows": int(len(outer_test)),
                "outer_test_papers": len(outer_test_refnos),
                "train_test_refno_overlap": 0,
                "expert_inner_cross_fitting": expert_inner_audit,
                "selection_branch": branch,
                "decision_invariance": invariant,
                "label_free_decision_audit": decision_audit,
                "router_level_cross_fitting": router_level_audit,
                "selected_smoother": selected.name,
                "selection_metrics": selection_metrics,
                "outer_test_labels_used_for_branch_or_selection": False,
            }
        )
        print(
            f"fold {fold}: branch={branch}, selected={selected.name}, "
            f"test_rows={len(outer_test)}",
            flush=True,
        )

    covered = np.isin(fold_ids, np.asarray(args.folds, dtype=np.int8))
    results: dict[str, Any] = {}
    for name in names:
        values = probabilities[name]
        if not np.isfinite(values[covered]).all():
            raise AssertionError(f"v2 candidate incomplete: {name}")
        results[name] = {
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
            "transition_vs_identity": nested.transition_record(
                exact_labels[covered],
                probabilities["router_identity"][covered],
                values[covered],
            ),
        }
    identity_accuracy = float(results["router_identity"]["aggregate"]["accuracy"])
    selected_accuracy = float(results["nested_selected"]["aggregate"]["accuracy"])
    reproduction = {
        str(fold): {
            "expected_correct": paper_v1.CANONICAL_V1_CORRECT_ROWS[fold],
            "observed_correct": int(
                (
                    (probabilities["router_identity"][fold_ids == fold] >= 0.5)
                    == exact_labels[fold_ids == fold]
                ).sum()
            ),
        }
        for fold in args.folds
    }
    if any(
        item["observed_correct"] != item["expected_correct"]
        for item in reproduction.values()
    ):
        raise AssertionError(f"canonical v1 router reproduction failed: {reproduction}")
    oof_record = None
    if full_run:
        oof_path = output_dir / "oof_probabilities.npz"
        write_probabilities(oof_path, probabilities)
        oof_record = {"path": str(oof_path), "sha256": base.file_sha256(oof_path)}
    report = {
        "schema_version": "r3_exact300_paper_consistency_grouped_oof_v2",
        "status": "complete",
        "mode": "full_five_fold" if full_run else "staged_screen",
        "scope": {
            "evaluated_folds": list(args.folds),
            "evaluated_rows": int(covered.sum()),
            "full_coverage": full_run,
            "exact_membership": exact_coverage,
            "auxiliary_candidate_pool": full_coverage,
        },
        "compliance_audit": {
            "candidate_receipt_written_before_label_access": True,
            "all_outer_test_refno_rows_excluded_from_every_fit": True,
            "early_exit_uses_only_candidate_decision_invariance": True,
            "outer_test_labels_used_for_branch_or_smoother_selection": False,
            "noninvariant_smoother_selected_from_fully_nested_training_oof": True,
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
        },
        "inputs": freeze["inputs"],
        "candidate_freeze": {
            "path": str(freeze_path),
            "sha256": base.file_sha256(freeze_path),
        },
        "feature_summary": {
            "numeric": numeric_manifest.get("matrices"),
            "own": own_manifest.get("matrices"),
            "component": component_manifest.get("matrices"),
        },
        "outer_fold_audit": outer_audit,
        "canonical_v1_reproduction": reproduction,
        "models": results,
        "best_observed": {"strategy": "nested_selected", **results["nested_selected"]},
        "continuation_gate": {
            "reference": "router_identity recomputed in the same run",
            "identity_accuracy": identity_accuracy,
            "nested_selected_accuracy": selected_accuracy,
            "non_regressing": selected_accuracy >= identity_accuracy,
            "continue_to_full_five_folds": bool(
                not full_run and selected_accuracy >= 0.88333333
            ),
        },
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "full_coverage_required": True,
            "passed": bool(full_run and selected_accuracy >= TARGET_ACCURACY),
            "decision": (
                "EXACT300_TARGET_REACHED"
                if full_run and selected_accuracy >= TARGET_ACCURACY
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
                "report": str(report_path),
                "best_observed": report["best_observed"],
                "continuation_gate": report["continuation_gate"],
                "promotion_gate": report["promotion_gate"],
                "fold_branches": [
                    {
                        "fold": item["fold"],
                        "branch": item["selection_branch"],
                        "selected": item["selected_smoother"],
                    }
                    for item in outer_audit
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
