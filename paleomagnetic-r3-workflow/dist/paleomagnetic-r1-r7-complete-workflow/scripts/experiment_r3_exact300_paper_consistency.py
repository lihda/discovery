#!/usr/bin/env python3
"""Nested, REFNO-safe paper-consistency calibration for exact300 R3."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/paper_consistency_v1"
TARGET_ACCURACY = 0.90
ROUTER_NAME = "router_loss_hgb_hard"
CANONICAL_RUNTIME = {
    "python": "3.9.6",
    "numpy": "2.0.2",
    "sklearn": "1.6.1",
}
CANONICAL_V1_CORRECT_ROWS = {1: 55, 2: 55, 3: 49, 4: 54, 5: 55}


@dataclass(frozen=True)
class SmoothingSpec:
    name: str
    kind: str
    alpha: float


def smoothing_specs() -> tuple[SmoothingSpec, ...]:
    # Ordering is part of the freeze: exact metric ties prefer identity.
    return (
        SmoothingSpec("identity", "identity", 0.0),
        SmoothingSpec("paper_min_blend_050", "min", 0.50),
        SmoothingSpec("paper_min_blend_075", "min", 0.75),
        SmoothingSpec("paper_min_blend_100", "min", 1.00),
        SmoothingSpec("paper_mean", "mean", 1.00),
        SmoothingSpec("paper_majority", "majority", 1.00),
    )


def apply_smoothing(
    probability: np.ndarray,
    groups: Sequence[str],
    spec: SmoothingSpec,
) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.shape != (len(groups),) or not np.isfinite(values).all():
        raise ValueError("paper smoothing requires one finite probability per row")
    output = values.copy()
    members: dict[str, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(str(group), []).append(index)
    for indices in members.values():
        selected = values[indices]
        if spec.kind == "identity":
            continue
        if spec.kind == "min":
            output[indices] = (1.0 - spec.alpha) * selected + spec.alpha * float(
                selected.min()
            )
        elif spec.kind == "mean":
            output[indices] = (1.0 - spec.alpha) * selected + spec.alpha * float(
                selected.mean()
            )
        elif spec.kind == "majority":
            positive_fraction = float((selected >= 0.5).mean())
            if positive_fraction > 0.5:
                output[indices] = 1.0
            elif positive_fraction < 0.5:
                output[indices] = 0.0
            # An exact tie preserves each row's identity probability.
        else:
            raise ValueError(f"unknown smoothing kind: {spec.kind}")
    if not np.isfinite(output).all():
        raise AssertionError("paper smoothing produced non-finite probabilities")
    return np.clip(output, 0.0, 1.0)


def select_smoothing(
    labels: np.ndarray,
    router_oof: np.ndarray,
    groups: Sequence[str],
    specs: Sequence[SmoothingSpec],
) -> tuple[SmoothingSpec, dict[str, Any]]:
    metrics: dict[str, Any] = {}
    for spec in specs:
        values = apply_smoothing(router_oof, groups, spec)
        metrics[spec.name] = base.metric_record(labels, values)
    best_accuracy = max(metrics[spec.name]["accuracy"] for spec in specs)
    selected = next(
        spec for spec in specs if metrics[spec.name]["accuracy"] == best_accuracy
    )
    return selected, metrics


def router_spec() -> nested.RouterSpec:
    matches = [spec for spec in nested.router_specs() if spec.name == ROUTER_NAME]
    if len(matches) != 1:
        raise ValueError("nested router declaration changed")
    return matches[0]


def fit_router_level_oof(
    *,
    spec: nested.RouterSpec,
    outer_train: np.ndarray,
    router_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    experts: Sequence[nested.ExpertSpec],
    matrices: Mapping[str, np.ndarray],
    full_labels: np.ndarray,
    full_records: Sequence[dict[str, str]],
    canonical_targets: Sequence[str],
    own_target: np.ndarray,
    component_bridge: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    output = np.full(len(full_records), np.nan, dtype=np.float64)
    audit: list[dict[str, Any]] = []
    for fold, (router_fit, router_validation) in enumerate(router_folds, start=1):
        fit_refnos = {full_records[int(index)]["REFNO"] for index in router_fit}
        validation_refnos = {
            full_records[int(index)]["REFNO"] for index in router_validation
        }
        if fit_refnos & validation_refnos:
            raise AssertionError("router-level fit/validation REFNO overlap")

        # Build router-training expert inputs without using router-validation
        # papers anywhere in the nested expert fits.
        nested_groups = [full_records[int(index)]["REFNO"] for index in router_fit]
        expert_folds = nested.deterministic_group_folds(
            router_fit,
            nested_groups,
            seed=nested.RANDOM_STATE + 1000 + fold,
        )
        router_train_probability = np.full(
            (len(full_records), len(experts)), np.nan, dtype=np.float64
        )
        expert_crossfit_audit: list[dict[str, Any]] = []
        for expert_fold, (expert_fit, expert_validation) in enumerate(
            expert_folds, start=1
        ):
            expert_fit_refnos = {
                full_records[int(index)]["REFNO"] for index in expert_fit
            }
            expert_validation_refnos = {
                full_records[int(index)]["REFNO"] for index in expert_validation
            }
            if (
                expert_fit_refnos & expert_validation_refnos
                or (expert_fit_refnos | expert_validation_refnos)
                & validation_refnos
            ):
                raise AssertionError("nested expert/router validation group leakage")
            expert_weight = base.inverse_target_multiplicity_weights(
                expert_fit, full_records, canonical_targets
            )
            for column, expert in enumerate(experts):
                fitted = nested.fit_expert(
                    expert,
                    matrices[expert.matrix],
                    expert_fit,
                    full_labels,
                    expert_weight,
                )
                router_train_probability[expert_validation, column] = fitted.predict(
                    matrices[expert.matrix][expert_validation]
                )
            expert_crossfit_audit.append(
                {
                    "expert_fold": expert_fold,
                    "fit_rows": int(len(expert_fit)),
                    "validation_rows": int(len(expert_validation)),
                    "fit_validation_refno_overlap": 0,
                    "router_validation_refno_overlap": 0,
                }
            )
        if not np.isfinite(router_train_probability[router_fit]).all():
            raise AssertionError("nested router-training expert OOF is incomplete")

        router_validation_probability = np.zeros(
            (len(router_validation), len(experts)), dtype=np.float64
        )
        router_fit_weight = base.inverse_target_multiplicity_weights(
            router_fit, full_records, canonical_targets
        )
        for column, expert in enumerate(experts):
            fitted = nested.fit_expert(
                expert,
                matrices[expert.matrix],
                router_fit,
                full_labels,
                router_fit_weight,
            )
            router_validation_probability[:, column] = fitted.predict(
                matrices[expert.matrix][router_validation]
            )
        output[router_validation] = nested.fit_predict_router(
            spec,
            router_train_probability[router_fit],
            router_validation_probability,
            own_target[router_fit],
            own_target[router_validation],
            component_bridge[router_fit],
            component_bridge[router_validation],
            full_labels[router_fit],
            router_fit_weight,
        )
        audit.append(
            {
                "router_fold": fold,
                "fit_rows": int(len(router_fit)),
                "fit_papers": len(fit_refnos),
                "validation_rows": int(len(router_validation)),
                "validation_papers": len(validation_refnos),
                "fit_validation_refno_overlap": 0,
                "router_validation_excluded_from_all_nested_expert_fits": True,
                "expert_cross_fitting": expert_crossfit_audit,
            }
        )
    if not np.isfinite(output[outer_train]).all():
        raise AssertionError("router-level cross-fitted probabilities are incomplete")
    return output, audit


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
        raise ValueError("full paper-consistency OOF requires 300 finite rows")
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
    if runtime != CANONICAL_RUNTIME:
        raise ValueError(
            f"canonical paper-consistency runtime required: "
            f"expected={CANONICAL_RUNTIME}, observed={runtime}"
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
    router = router_spec()
    smoothers = smoothing_specs()
    if any(spec.matrix not in matrices for spec in experts):
        raise ValueError("an expert matrix is absent from frozen caches")

    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
    freeze = {
        "schema_version": "r3_exact300_paper_consistency_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "evaluated_folds": list(args.folds),
        "inner_experts": [asdict(spec) for spec in experts],
        "router": asdict(router),
        "router_level_cross_fitting": {
            "method": "same deterministic REFNO folds; router never scores its fit groups",
            "n_splits": nested.INNER_SPLITS,
            "seed": nested.RANDOM_STATE,
        },
        "paper_smoothing_candidates": [asdict(spec) for spec in smoothers],
        "selection": {
            "source": "router-level outer-training OOF only",
            "order": "accuracy, then declaration order for every exact tie",
            "exact_tie_preference": "identity",
        },
        "inputs": {
            name: {"path": str(path), "sha256": base.file_sha256(path)}
            for name, path in paths.items()
            if name != "pdf_dir"
        },
        "implementation_dependency": {
            "path": str(Path(nested.__file__).resolve()),
            "sha256": base.file_sha256(Path(nested.__file__).resolve()),
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "outer_test_labels_used_for_smoother_selection": False,
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
    names = ["router_identity", "nested_selected", *[spec.name for spec in smoothers[1:]]]
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
        inner_folds = nested.deterministic_group_folds(
            outer_train, train_groups
        )
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
                raise AssertionError("expert inner cross-fit group leakage")
            weights = base.inverse_target_multiplicity_weights(
                inner_fit, full_records, canonical_targets
            )
            for column, spec in enumerate(experts):
                fitted = nested.fit_expert(
                    spec,
                    matrices[spec.matrix],
                    inner_fit,
                    full_labels,
                    weights,
                )
                inner_probability[inner_validation, column] = fitted.predict(
                    matrices[spec.matrix][inner_validation]
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
            raise AssertionError("expert inner OOF is incomplete")

        router_oof, router_audit = fit_router_level_oof(
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
        selected, selection_metrics = select_smoothing(
            full_labels[outer_train],
            router_oof[outer_train],
            train_groups,
            smoothers,
        )

        outer_weights = base.inverse_target_multiplicity_weights(
            outer_train, full_records, canonical_targets
        )
        test_expert_probability = np.zeros(
            (len(outer_test), len(experts)), dtype=np.float64
        )
        for column, spec in enumerate(experts):
            fitted = nested.fit_expert(
                spec,
                matrices[spec.matrix],
                outer_train,
                full_labels,
                outer_weights,
            )
            test_expert_probability[:, column] = fitted.predict(
                matrices[spec.matrix][outer_test]
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
        probabilities["router_identity"][exact_test] = identity
        probabilities["nested_selected"][exact_test] = apply_smoothing(
            identity, test_groups, selected
        )
        for spec in smoothers[1:]:
            probabilities[spec.name][exact_test] = apply_smoothing(
                identity, test_groups, spec
            )
        outer_audit.append(
            {
                "fold": fold,
                "outer_train_rows": int(len(outer_train)),
                "outer_test_rows": int(len(outer_test)),
                "outer_test_papers": len(outer_test_refnos),
                "train_test_refno_overlap": 0,
                "expert_inner_cross_fitting": expert_inner_audit,
                "router_level_cross_fitting": router_audit,
                "smoother_selection_source": "router-level outer-training OOF",
                "selected_smoother": selected.name,
                "selection_metrics": selection_metrics,
                "outer_test_labels_used_for_selection": False,
            }
        )
        print(
            f"fold {fold}: selected={selected.name}, train_rows={len(outer_train)}, "
            f"test_rows={len(outer_test)}",
            flush=True,
        )

    covered = np.isin(fold_ids, np.asarray(args.folds, dtype=np.int8))
    results: dict[str, Any] = {}
    for name in names:
        values = probabilities[name]
        if not np.isfinite(values[covered]).all():
            raise AssertionError(f"paper candidate incomplete: {name}")
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
            "expected_correct": CANONICAL_V1_CORRECT_ROWS[fold],
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
        "schema_version": "r3_exact300_paper_consistency_grouped_oof_v1",
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
            "canonical_outer_fold_manifest_sha256": auxiliary.FOLD_MANIFEST_SHA256,
            "candidate_receipt_written_before_label_access": True,
            "expert_predictions_inner_cross_fitted_by_refno": True,
            "router_predictions_second_level_cross_fitted_by_refno": True,
            "smoother_selected_only_from_outer_training_router_oof": True,
            "outer_test_labels_used_for_selection": False,
            "all_outer_test_refno_rows_excluded_from_every_fit": True,
            "prior_oof_or_evaluation_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
        },
        "inputs": freeze["inputs"],
        "candidate_freeze": {
            "path": str(freeze_path),
            "sha256": base.file_sha256(freeze_path),
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
                "selected_smoothers": [
                    item["selected_smoother"] for item in outer_audit
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
