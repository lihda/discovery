#!/usr/bin/env python3
"""Nested exact300 R3 router with frozen field-signature and PDF metadata views.

This is an independent v2 experiment.  It imports reusable split/model helpers
from nested-router v1 but never reads the v1 report or OOF artifact.  The
record-free field-signature cache, structural PDF metadata cache, complete
expert family, router contexts, and fixed threshold are frozen before R3 is
opened.  Every outer-test REFNO is excluded in full, and router training uses
only REFNO-isolated inner cross-fitted expert probabilities.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_exact300_auxiliary_numeric_cache as numeric_builder  # noqa: E402
import build_r3_exact300_component_bridge_cache as component_builder  # noqa: E402
import build_r3_exact300_own_study_cache as own_builder  # noqa: E402
import build_r3_exact300_pdf_metadata_cache as metadata_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_component_bridge_auxiliary as component_eval  # noqa: E402
import experiment_r3_exact300_field_signatures as field_source  # noqa: E402
import experiment_r3_exact300_nested_router as v1  # noqa: E402
import experiment_r3_exact300_own_study_auxiliary as own_eval  # noqa: E402
import experiment_r3_exact300_pdf_metadata_auxiliary as metadata_eval  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/nested_router_v2"
FIELD_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/field_signatures_v1"
DEFAULT_FIELD_CACHE = FIELD_DIR / "signature_features.npz"
DEFAULT_FIELD_MANIFEST = FIELD_DIR / "signature_features.manifest.json"
DEFAULT_FIELD_FREEZE = FIELD_DIR / "feature_freeze.json"

FIELD_CACHE_SHA256 = (
    "dc057f42dac3904daa744e98f5bd7e5f7fdc04170aaedcc9c9303e6b64c6dce9"
)
FIELD_MANIFEST_SHA256 = (
    "99b0eb53465787bdc0adea27cd0c94a46465978be7da18e7faf15d5f4530afa2"
)
FIELD_FREEZE_SHA256 = (
    "5d9b18890966f714feac14f9ea68b4b3ec60b1be0836c00eb1a3709c36b77b26"
)
METADATA_CACHE_SHA256 = (
    "624809328b7c17f6f5f5c01a5b6507c4af8a09fe7e18ede75af556f76e9126c8"
)
METADATA_MANIFEST_SHA256 = (
    "0f0f25b9e175cfe6864f80647eb6d3285e7a72d4650f7dfb7645078f2303e6e0"
)
STAGED_REFERENCE_ACCURACY = 0.88333333
TARGET_ACCURACY = 0.90
SOURCE_MODEL_RANDOM_STATE = 9357


@dataclass(frozen=True)
class RouterSpec:
    name: str
    kind: str
    context_key: str


def expert_specs() -> tuple[v1.ExpertSpec, ...]:
    return (
        *v1.expert_specs(),
        v1.ExpertSpec(
            "expert_hgb_field_signature", "field_signatures", "signature_hgb"
        ),
        v1.ExpertSpec(
            "expert_hgb_canonical_signature",
            "canonical_plus_signature",
            "source_hgb10",
        ),
        v1.ExpertSpec(
            "expert_hgb_component_bridge_metadata",
            "component_bridge_metadata",
            "source_hgb10",
        ),
    )


def router_specs() -> tuple[RouterSpec, ...]:
    return (
        RouterSpec("fusion_mean_all", "mean", "probability"),
        RouterSpec("fusion_median_all", "median", "probability"),
        RouterSpec("stack_probability_hgb", "stack_hgb", "probability"),
        RouterSpec("stack_base_context_hgb", "stack_hgb", "base"),
        RouterSpec("stack_field_context_hgb", "stack_hgb", "field"),
        RouterSpec("stack_metadata_context_hgb", "stack_hgb", "metadata"),
        RouterSpec("stack_all_context_hgb", "stack_hgb", "all"),
        RouterSpec("stack_all_context_logreg", "stack_logreg", "all"),
        RouterSpec("router_loss_base_hard", "loss_hgb_hard", "base"),
        RouterSpec("router_loss_field_hard", "loss_hgb_hard", "field"),
        RouterSpec("router_loss_metadata_hard", "loss_hgb_hard", "metadata"),
        RouterSpec("router_loss_all_hard", "loss_hgb_hard", "all"),
        RouterSpec("router_loss_all_soft", "loss_hgb_soft", "all"),
        RouterSpec(
            "router_correctness_all_hard", "correctness_logreg_hard", "all"
        ),
    )


def validate_field_cache(
    cache_path: Path,
    manifest_path: Path,
    freeze_path: Path,
    expected_rows: int,
    full_alignment_sha256: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    if base.file_sha256(cache_path) != FIELD_CACHE_SHA256:
        raise ValueError("field-signature cache hash mismatch")
    if base.file_sha256(manifest_path) != FIELD_MANIFEST_SHA256:
        raise ValueError("field-signature manifest hash mismatch")
    if base.file_sha256(freeze_path) != FIELD_FREEZE_SHA256:
        raise ValueError("field-signature freeze hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    boundary = freeze.get("runtime_boundary", {})
    signature_input = freeze.get("signature_cache", {})
    if (
        manifest.get("schema_version") != "r3_exact300_field_signature_cache_v1"
        or manifest.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
        or manifest.get("rows") != expected_rows
        or manifest.get("features") != 531
        or manifest.get("alignment_sha256") != full_alignment_sha256
        or manifest.get("contains_labels") is not False
        or manifest.get("contains_record_identifiers") is not False
        or manifest.get("contains_source_quotes") is not False
        or freeze.get("schema_version")
        != "r3_exact300_field_signatures_feature_freeze_v1"
        or freeze.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
        or freeze.get("scope", {}).get("full_alignment_sha256")
        != full_alignment_sha256
        or signature_input.get("sha256") != FIELD_CACHE_SHA256
        or signature_input.get("manifest_sha256") != FIELD_MANIFEST_SHA256
        or boundary.get("labels_read") is not False
        or boundary.get("source_field_cells_read") is not False
        or boundary.get("prior_predictions_or_evaluations_read") is not False
        or boundary.get("identifiers_used_as_predictive_features") is not False
        or boundary.get("record_answer_mapping_loaded") is not False
        or boundary.get("source_quotes_cached") is not False
    ):
        raise ValueError("field-signature artifacts do not prove a record-free build")
    with np.load(cache_path, allow_pickle=False) as payload:
        if payload.files != ["field_signatures"]:
            raise ValueError("field-signature cache contains an unexpected array")
        matrix = np.asarray(payload["field_signatures"], dtype=np.float32)
    if (
        matrix.shape != (expected_rows, 531)
        or not np.isfinite(matrix).all()
        or field_source.sha256_array(matrix) != manifest.get("matrix_sha256")
        or field_source.sha256_array(matrix)
        != freeze.get("feature_matrices", {})
        .get("field_signatures", {})
        .get("sha256")
    ):
        raise ValueError("field-signature matrix differs from its frozen receipts")
    return matrix, manifest


def build_matrices(
    numeric_arrays: Mapping[str, np.ndarray],
    component_arrays: Mapping[str, np.ndarray],
    field_signatures: np.ndarray,
    metadata: np.ndarray,
) -> dict[str, np.ndarray]:
    matrices = {
        **numeric_arrays,
        **component_arrays,
        "field_signatures": field_signatures,
        "canonical_plus_signature": np.concatenate(
            (numeric_arrays["combined"], field_signatures), axis=1
        ),
        "component_bridge_metadata": np.concatenate(
            (component_arrays["combined_plus_component_bridge"], metadata), axis=1
        ),
    }
    output = {
        name: np.asarray(value, dtype=np.float32) for name, value in matrices.items()
    }
    if any(
        value.ndim != 2
        or value.shape[0] != field_signatures.shape[0]
        or not np.isfinite(value).all()
        for value in output.values()
    ):
        raise ValueError("v2 expert matrices are not aligned finite matrices")
    return output


def build_router_contexts(
    component_bridge: np.ndarray,
    field_signatures: np.ndarray,
    metadata: np.ndarray,
) -> dict[str, np.ndarray]:
    contexts = {
        "base": component_bridge,
        "field": np.concatenate((component_bridge, field_signatures), axis=1),
        "metadata": np.concatenate((component_bridge, metadata), axis=1),
        "all": np.concatenate(
            (component_bridge, field_signatures, metadata), axis=1
        ),
    }
    output = {
        name: np.asarray(value, dtype=np.float32) for name, value in contexts.items()
    }
    if any(value.ndim != 2 or not np.isfinite(value).all() for value in output.values()):
        raise ValueError("v2 router contexts are not finite matrices")
    return output


def fit_expert(
    spec: v1.ExpertSpec,
    matrix: np.ndarray,
    train_index: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
) -> v1.FittedExpert:
    if spec.kind not in {"signature_hgb", "source_hgb10"}:
        return v1.fit_expert(spec, matrix, train_index, labels, sample_weight)
    if spec.kind == "signature_hgb":
        model = HistGradientBoostingClassifier(
            max_iter=240,
            learning_rate=0.035,
            max_leaf_nodes=7,
            min_samples_leaf=12,
            l2_regularization=2.0,
            random_state=SOURCE_MODEL_RANDOM_STATE,
        )
    else:
        model = HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=SOURCE_MODEL_RANDOM_STATE,
        )
    model.fit(
        np.asarray(matrix[train_index], dtype=np.float64),
        labels[train_index],
        sample_weight=sample_weight,
    )
    return v1.FittedExpert(model=model, scaler=None)


def fit_predict_router(
    spec: RouterSpec,
    train_probability: np.ndarray,
    test_probability: np.ndarray,
    train_own_target: np.ndarray,
    test_own_target: np.ndarray,
    train_contexts: Mapping[str, np.ndarray],
    test_contexts: Mapping[str, np.ndarray],
    train_labels: np.ndarray,
    train_weight: np.ndarray,
) -> np.ndarray:
    context = "probability" if spec.context_key == "probability" else "full_context"
    empty_train = np.zeros((len(train_probability), 0), dtype=np.float64)
    empty_test = np.zeros((len(test_probability), 0), dtype=np.float64)
    component_train = (
        empty_train
        if spec.context_key == "probability"
        else train_contexts[spec.context_key]
    )
    component_test = (
        empty_test
        if spec.context_key == "probability"
        else test_contexts[spec.context_key]
    )
    own_train = empty_train if spec.context_key == "probability" else train_own_target
    own_test = empty_test if spec.context_key == "probability" else test_own_target
    return v1.fit_predict_router(
        v1.RouterSpec(spec.name, spec.kind, context),
        train_probability,
        test_probability,
        own_train,
        own_test,
        component_train,
        component_test,
        train_labels,
        train_weight,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--field-cache", type=Path, default=DEFAULT_FIELD_CACHE)
    parser.add_argument("--field-manifest", type=Path, default=DEFAULT_FIELD_MANIFEST)
    parser.add_argument("--field-freeze", type=Path, default=DEFAULT_FIELD_FREEZE)
    parser.add_argument("--metadata-cache", type=Path, default=metadata_builder.DEFAULT_CACHE)
    parser.add_argument(
        "--metadata-manifest", type=Path, default=metadata_builder.DEFAULT_MANIFEST
    )
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--folds", type=v1.parse_folds, default=(1, 2, 3))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    paths = {
        "numeric_cache": args.numeric_cache.resolve(),
        "numeric_manifest": args.numeric_manifest.resolve(),
        "own_cache": args.own_cache.resolve(),
        "own_manifest": args.own_manifest.resolve(),
        "component_cache": args.component_cache.resolve(),
        "component_manifest": args.component_manifest.resolve(),
        "field_cache": args.field_cache.resolve(),
        "field_manifest": args.field_manifest.resolve(),
        "field_freeze": args.field_freeze.resolve(),
        "metadata_cache": args.metadata_cache.resolve(),
        "metadata_manifest": args.metadata_manifest.resolve(),
        "exact_identity": args.exact_identity.resolve(),
        "full_identity": args.full_identity.resolve(),
        "fold_manifest": args.fold_manifest.resolve(),
        "targets": args.targets.resolve(),
        "pdf_dir": args.pdf_dir.resolve(),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

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
    field_signatures, field_manifest = validate_field_cache(
        paths["field_cache"],
        paths["field_manifest"],
        paths["field_freeze"],
        len(full_records),
        full_alignment,
    )
    if base.file_sha256(paths["metadata_cache"]) != METADATA_CACHE_SHA256:
        raise ValueError("metadata cache hash mismatch")
    if base.file_sha256(paths["metadata_manifest"]) != METADATA_MANIFEST_SHA256:
        raise ValueError("metadata manifest hash mismatch")
    metadata, metadata_manifest = metadata_eval.validate_metadata_cache(
        paths["metadata_cache"],
        paths["metadata_manifest"],
        len(full_records),
        full_alignment,
    )
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from full identity")

    matrices = build_matrices(
        numeric_arrays, component_arrays, field_signatures, metadata
    )
    contexts = build_router_contexts(
        component_arrays["component_bridge"], field_signatures, metadata
    )
    experts = expert_specs()
    routers = router_specs()
    if any(spec.matrix not in matrices for spec in experts):
        raise ValueError("a v2 expert matrix is absent")
    if len({spec.name for spec in experts}) != len(experts) or len(
        {spec.name for spec in routers}
    ) != len(routers):
        raise ValueError("duplicate v2 candidate declaration")

    # The candidate/context receipt is committed before the first R3 label read.
    freeze = {
        "schema_version": "r3_exact300_nested_router_v2_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "evaluated_folds": list(args.folds),
        "experts": [asdict(spec) for spec in experts],
        "routers": [asdict(spec) for spec in routers],
        "inner_split": {
            "method": "deterministic greedy row-balanced REFNO assignment",
            "n_splits": v1.INNER_SPLITS,
            "seed": v1.RANDOM_STATE,
        },
        "decision_threshold": 0.5,
        "staged_gate": {
            "reference_accuracy": STAGED_REFERENCE_ACCURACY,
            "rule": "best router must strictly exceed reference on folds 1,2,3",
        },
        "expert_matrix_shapes": {
            spec.name: list(matrices[spec.matrix].shape) for spec in experts
        },
        "router_context_shapes": {
            name: list(value.shape) for name, value in contexts.items()
        },
        "inputs": {
            name: {"path": str(path), "sha256": base.file_sha256(path)}
            for name, path in paths.items()
            if name != "pdf_dir"
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "source_field_cells_read": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_oof_or_evaluation_read": False,
            "identifiers_used_as_predictive_features": False,
            "refno_used_only_for_group_isolation": True,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
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

    names = [spec.name for spec in experts] + [spec.name for spec in routers]
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
        outer_train_refnos = {
            full_records[int(index)]["REFNO"] for index in outer_train
        }
        groups = [full_records[int(index)]["REFNO"] for index in outer_train]
        inner_folds = v1.deterministic_group_folds(outer_train, groups)
        inner_probability = np.full(
            (len(full_records), len(experts)), np.nan, dtype=np.float64
        )
        inner_audit: list[dict[str, Any]] = []
        for inner_fold, (inner_fit, inner_validation) in enumerate(inner_folds, start=1):
            fit_refnos = {full_records[int(index)]["REFNO"] for index in inner_fit}
            validation_refnos = {
                full_records[int(index)]["REFNO"] for index in inner_validation
            }
            if fit_refnos & validation_refnos:
                raise AssertionError("inner fit/validation REFNO overlap")
            if (fit_refnos | validation_refnos) & outer_test_refnos:
                raise AssertionError("outer-test REFNO entered inner cross-fitting")
            inner_weight = base.inverse_target_multiplicity_weights(
                inner_fit, full_records, canonical_targets
            )
            for column, spec in enumerate(experts):
                fitted = fit_expert(
                    spec,
                    matrices[spec.matrix],
                    inner_fit,
                    full_labels,
                    inner_weight,
                )
                inner_probability[inner_validation, column] = fitted.predict(
                    matrices[spec.matrix][inner_validation]
                )
            inner_audit.append(
                {
                    "inner_fold": inner_fold,
                    "fit_rows": int(len(inner_fit)),
                    "fit_papers": len(fit_refnos),
                    "validation_rows": int(len(inner_validation)),
                    "validation_papers": len(validation_refnos),
                    "fit_validation_refno_overlap": 0,
                    "outer_test_refno_overlap": 0,
                }
            )

        train_probability = inner_probability[outer_train]
        if not np.isfinite(train_probability).all():
            raise AssertionError("v2 inner expert probabilities are incomplete")
        outer_weight = base.inverse_target_multiplicity_weights(
            outer_train, full_records, canonical_targets
        )
        test_probability = np.zeros((len(outer_test), len(experts)), dtype=np.float64)
        for column, spec in enumerate(experts):
            fitted = fit_expert(
                spec,
                matrices[spec.matrix],
                outer_train,
                full_labels,
                outer_weight,
            )
            values = fitted.predict(matrices[spec.matrix][outer_test])
            test_probability[:, column] = values
            probabilities[spec.name][exact_test] = values

        train_contexts = {
            name: value[outer_train] for name, value in contexts.items()
        }
        test_contexts = {name: value[outer_test] for name, value in contexts.items()}
        for spec in routers:
            values = fit_predict_router(
                spec,
                train_probability,
                test_probability,
                own_arrays["own_target"][outer_train],
                own_arrays["own_target"][outer_test],
                train_contexts,
                test_contexts,
                full_labels[outer_train],
                outer_weight,
            )
            if values.shape != (len(outer_test),) or not np.isfinite(values).all():
                raise AssertionError(f"v2 router produced invalid probabilities: {spec.name}")
            probabilities[spec.name][exact_test] = np.clip(values, 0.0, 1.0)

        excluded_rows = sum(
            record["REFNO"] in outer_test_refnos for record in full_records
        )
        outer_audit.append(
            {
                "fold": fold,
                "test_rows": int(len(exact_test)),
                "test_papers": len(outer_test_refnos),
                "auxiliary_train_rows": int(len(outer_train)),
                "auxiliary_train_papers": len(outer_train_refnos),
                "excluded_full_rows": int(excluded_rows),
                "all_outer_test_refno_rows_excluded": True,
                "train_test_refno_overlap": len(
                    outer_train_refnos & outer_test_refnos
                ),
                "inner_cross_fitting": inner_audit,
            }
        )
        print(
            f"fold {fold}: train_rows={len(outer_train)}, test_rows={len(exact_test)}, "
            f"experts={len(experts)}, routers={len(routers)}",
            flush=True,
        )

    covered = np.isin(fold_ids, np.asarray(args.folds, dtype=np.int8))
    results: dict[str, Any] = {}
    for name in names:
        value = probabilities[name]
        if not np.isfinite(value[covered]).all():
            raise AssertionError(f"v2 candidate is incomplete: {name}")
        results[name] = {
            "role": "expert_control" if name.startswith("expert_") else "router",
            "aggregate": base.metric_record(exact_labels[covered], value[covered]),
            "folds": [
                {
                    "fold": fold,
                    **base.metric_record(
                        exact_labels[fold_ids == fold], value[fold_ids == fold]
                    ),
                }
                for fold in args.folds
            ],
        }

    expert_names = [spec.name for spec in experts]
    router_names = [spec.name for spec in routers]
    rank_key = lambda name: (  # noqa: E731
        results[name]["aggregate"]["accuracy"],
        results[name]["aggregate"]["macro_f1"],
        results[name]["aggregate"]["balanced_accuracy"],
    )
    best_expert = max(expert_names, key=rank_key)
    best_router = max(router_names, key=rank_key)
    ranking = sorted(names, key=rank_key, reverse=True)
    for name in names:
        results[name]["transition_vs_best_single_expert"] = v1.transition_record(
            exact_labels[covered],
            probabilities[best_expert][covered],
            probabilities[name][covered],
        )

    router_accuracy = float(results[best_router]["aggregate"]["accuracy"])
    expert_accuracy = float(results[best_expert]["aggregate"]["accuracy"])
    staged_passed = router_accuracy > STAGED_REFERENCE_ACCURACY
    oof_record = None
    if full_run:
        oof_path = output_dir / "oof_probabilities.npz"
        v1.write_probabilities(oof_path, probabilities)
        oof_record = {"path": str(oof_path), "sha256": base.file_sha256(oof_path)}

    report = {
        "schema_version": "r3_exact300_nested_router_v2_grouped_oof_v1",
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
            "all_feature_caches_frozen_without_labels": True,
            "field_signature_cache_record_free": True,
            "metadata_cache_record_free": True,
            "inner_expert_probabilities_cross_fitted_by_refno": True,
            "all_outer_test_refno_rows_excluded_from_every_fit": True,
            "all_inner_validation_refno_rows_excluded_from_inner_expert_fit": True,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_oof_or_evaluation_read": False,
            "identifiers_used_as_predictive_features": False,
            "fixed_decision_threshold": 0.5,
        },
        "inputs": freeze["inputs"],
        "candidate_freeze": {
            "path": str(freeze_path),
            "sha256": base.file_sha256(freeze_path),
        },
        "feature_summary": {
            "numeric_matrices": numeric_manifest.get("matrices"),
            "own_matrices": own_manifest.get("matrices"),
            "component_matrices": component_manifest.get("matrices"),
            "field_signature": {
                "shape": list(field_signatures.shape),
                "matrix_sha256": field_manifest.get("matrix_sha256"),
            },
            "metadata": metadata_manifest.get("matrices", {}).get("pdf_metadata"),
            "router_context_shapes": freeze["router_context_shapes"],
        },
        "outer_fold_audit": outer_audit,
        "ranking": ranking,
        "best_single_expert": {"strategy": best_expert, **results[best_expert]},
        "best_router": {"strategy": best_router, **results[best_router]},
        "models": results,
        "continuation_gate": {
            "reference_accuracy_on_folds_1_2_3": STAGED_REFERENCE_ACCURACY,
            "best_router": best_router,
            "router_accuracy": router_accuracy,
            "best_single_expert": best_expert,
            "single_expert_accuracy": expert_accuracy,
            "strictly_exceeds_reference": staged_passed,
            "continue_to_full_five_folds": bool(not full_run and staged_passed),
        },
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "full_coverage_required": True,
            "passed": bool(full_run and router_accuracy >= TARGET_ACCURACY),
            "decision": (
                "EXACT300_TARGET_REACHED"
                if full_run and router_accuracy >= TARGET_ACCURACY
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
                "best_single_expert": report["best_single_expert"],
                "best_router": report["best_router"],
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
