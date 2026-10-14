#!/usr/bin/env python3
"""Canonical exact300 evaluation of frozen target-conditioned BGE features.

The semantic cache is built without labels or record identifiers.  This
evaluation opens labels only after writing a candidate receipt.  In every
outer fold it removes every auxiliary row belonging to a test REFNO before
fitting PCA, scaling, or a classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_exact300_auxiliary_numeric_cache as numeric_builder  # noqa: E402
import build_r3_exact300_bge_semantic_cache as bge_builder  # noqa: E402
import build_r3_exact300_component_bridge_cache as component_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_component_bridge_auxiliary as component_eval  # noqa: E402
import experiment_r3_exact300_own_study_auxiliary as shared_eval  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_BGE_CACHE = bge_builder.DEFAULT_OUTPUT
DEFAULT_BGE_MANIFEST = bge_builder.DEFAULT_MANIFEST
DEFAULT_NUMERIC_CACHE = numeric_builder.DEFAULT_CACHE
DEFAULT_NUMERIC_MANIFEST = numeric_builder.DEFAULT_MANIFEST
DEFAULT_COMPONENT_CACHE = component_builder.DEFAULT_CACHE
DEFAULT_COMPONENT_MANIFEST = component_builder.DEFAULT_MANIFEST
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/bge_semantic_auxiliary_v1"

BGE_CACHE_SHA256 = "38e8826dbd7dab58ca21a1b0efe5eedbcf09c2b9b12c37b7576446d6f8cfd126"
BGE_MANIFEST_SHA256 = "bafe40529d05a836796dd7bd9817412cf4bf82a3062161e3c2cb52d2e12dd686"
BGE_INPUT_SHA256 = "f6d3b64bf14fc8200fa3555d132bb92d68d98d98aecb569c1ecb67fbb272ec62"
BGE_TEXT_ALIGNMENT_SHA256 = "2b8b2eacac8cb370dff898614a7a0cfdb054ea8b9766c7459a3f6b28affdce5d"
EXPECTED_BGE_VIEWS = {
    "target_query": 768,
    "abstract": 768,
    "target_local": 768,
    "target_paper": 768,
    "relation": 4,
}
CONTROL_MATRIX = "combined_plus_component_all"
CONTROL_NAME = "control_component_all_rf8"
RANDOM_STATE = 9357
TARGET_ACCURACY = 0.90


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    numeric_matrix: str | None
    semantic_views: tuple[str, ...]
    pca_components: int
    model: str
    regularization: float = 0.0


def candidate_specs() -> tuple[CandidateSpec, ...]:
    """Return the compact candidate family frozen before BGE label access."""
    spec = CandidateSpec
    candidates = (
        spec(CONTROL_NAME, CONTROL_MATRIX, (), 0, "rf8"),
        spec("bge_target_query_pca64_logreg", None, ("target_query",), 64, "logreg", 0.1),
        spec("bge_abstract_pca64_logreg", None, ("abstract",), 64, "logreg", 0.1),
        spec("bge_target_local_pca64_logreg", None, ("target_local",), 64, "logreg", 0.1),
        spec("bge_target_paper_pca64_logreg", None, ("target_paper",), 64, "logreg", 0.1),
        spec(
            "bge_evidence_pair_pca96_logreg",
            None,
            ("target_local", "target_paper"),
            96,
            "logreg",
            0.1,
        ),
        spec(
            "bge_all_views_pca128_logreg",
            None,
            ("target_query", "abstract", "target_local", "target_paper"),
            128,
            "logreg",
            0.1,
        ),
        spec("bge_relation_logreg", None, ("relation",), 0, "logreg", 0.1),
        spec("bge_relation_rf8", None, ("relation",), 0, "rf8"),
        spec("component_plus_relation_rf8", CONTROL_MATRIX, ("relation",), 0, "rf8"),
        spec("component_plus_relation_hgb10", CONTROL_MATRIX, ("relation",), 0, "hgb10"),
        spec(
            "component_plus_local_pca64_rf8",
            CONTROL_MATRIX,
            ("target_local",),
            64,
            "rf8",
        ),
        spec(
            "component_plus_paper_pca64_rf8",
            CONTROL_MATRIX,
            ("target_paper",),
            64,
            "rf8",
        ),
        spec(
            "component_plus_evidence_pca96_rf8",
            CONTROL_MATRIX,
            ("target_local", "target_paper"),
            96,
            "rf8",
        ),
        spec(
            "component_plus_evidence_pca96_logreg",
            CONTROL_MATRIX,
            ("target_local", "target_paper"),
            96,
            "logreg",
            0.01,
        ),
    )
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise AssertionError("duplicate BGE candidate name")
    return candidates


def parse_folds(value: str) -> tuple[int, ...]:
    if value.casefold() == "all":
        return tuple(range(1, auxiliary.N_SPLITS + 1))
    try:
        folds = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("folds must be 'all' or comma-separated integers") from exc
    if not folds or any(fold < 1 or fold > auxiliary.N_SPLITS for fold in folds):
        raise argparse.ArgumentTypeError("folds must be between 1 and 5")
    return folds


def validate_bge_cache(
    cache_path: Path,
    manifest_path: Path,
    full_identity_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if base.file_sha256(cache_path) != BGE_CACHE_SHA256:
        raise ValueError("BGE semantic cache hash mismatch")
    if base.file_sha256(manifest_path) != BGE_MANIFEST_SHA256:
        raise ValueError("BGE semantic manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    component_cache = manifest.get("inputs", {}).get("component_cache", {})
    component_manifest = manifest.get("inputs", {}).get("component_manifest", {})
    builder = manifest.get("builder", {})
    if (
        manifest.get("schema_version") != "r3_exact300_bge_semantic_cache_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("scope", {}).get("rows") != auxiliary.EXPECTED_FULL_PDF_ROWS
        or manifest.get("scope", {}).get("alignment_sha256") != BGE_TEXT_ALIGNMENT_SHA256
        or manifest.get("output", {}).get("sha256") != BGE_CACHE_SHA256
        or component_cache.get("sha256") != BGE_INPUT_SHA256
        or boundary.get("labels_read") is not False
        or boundary.get("targets_workbook_read") is not False
        or boundary.get("pmagresult_or_rockmag_read") is not False
        or boundary.get("prior_predictions_or_evaluations_read") is not False
        or boundary.get("cache_contains_identifiers") is not False
        or builder.get("sha256") != base.file_sha256(Path(str(builder.get("path", ""))))
    ):
        raise ValueError("BGE manifest does not prove a frozen label-free boundary")

    source_manifest_path = Path(str(component_manifest.get("path", ""))).resolve()
    if (
        not source_manifest_path.is_file()
        or component_manifest.get("sha256") != base.file_sha256(source_manifest_path)
    ):
        raise ValueError("BGE source text manifest hash mismatch")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    source_boundary = source_manifest.get("runtime_boundary", {})
    if (
        source_manifest.get("status") != "PASS"
        or source_manifest.get("inputs", {}).get("identity", {}).get("sha256")
        != auxiliary.FULL_IDENTITY_SHA256
        or base.file_sha256(full_identity_path) != auxiliary.FULL_IDENTITY_SHA256
        or source_manifest.get("inputs", {}).get("alignment_sha256")
        != BGE_TEXT_ALIGNMENT_SHA256
        or source_manifest.get("output", {}).get("sha256") != BGE_INPUT_SHA256
        or source_boundary.get("labels_read") is not False
        or source_boundary.get("pmagresult_or_rockmag_read") is not False
        or source_boundary.get("prior_predictions_or_evaluations_read") is not False
        or source_boundary.get("cache_contains_record_identifiers") is not False
    ):
        raise ValueError("BGE source text cache is not bound to the frozen auxiliary identity")

    with np.load(cache_path, allow_pickle=False) as payload:
        if set(payload.files) != set(EXPECTED_BGE_VIEWS):
            raise ValueError("BGE cache schema mismatch")
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    for name, width in EXPECTED_BGE_VIEWS.items():
        matrix = arrays[name]
        record = manifest.get("matrices", {}).get(name, {})
        if (
            matrix.shape != (auxiliary.EXPECTED_FULL_PDF_ROWS, width)
            or matrix.dtype != np.float32
            or not np.isfinite(matrix).all()
            or record.get("shape") != list(matrix.shape)
            or record.get("dtype") != "float32"
            or record.get("sha256") != auxiliary.sha256_array(matrix)
        ):
            raise ValueError(f"BGE matrix differs from manifest: {name}")
    norms = {
        name: np.linalg.norm(arrays[name], axis=1)
        for name in EXPECTED_BGE_VIEWS
        if name != "relation"
    }
    if any(not np.allclose(values, 1.0, atol=2e-5) for values in norms.values()):
        raise ValueError("BGE semantic embeddings are not normalized")
    return arrays, manifest


def make_model(spec: CandidateSpec) -> Any:
    if spec.model == "logreg":
        return LogisticRegression(
            C=spec.regularization,
            solver="liblinear",
            max_iter=3_000,
            random_state=RANDOM_STATE,
        )
    if spec.model == "rf8":
        return RandomForestClassifier(
            n_estimators=600,
            max_features=0.5,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if spec.model == "hgb10":
        return HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.035,
            max_leaf_nodes=7,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"unknown BGE model: {spec.model}")


def semantic_fold_block(
    views: tuple[str, ...],
    pca_components: int,
    arrays: Mapping[str, np.ndarray],
    train_index: np.ndarray,
    test_index: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if not views:
        return (
            np.empty((len(train_index), 0), dtype=np.float32),
            np.empty((len(test_index), 0), dtype=np.float32),
        )
    # Exact float64 SVD is deliberate here.  Accelerate-backed randomized SVD
    # emitted spurious overflow warnings for bounded normalized float32 BGE
    # vectors in the canonical sklearn 1.6 runtime.
    train = np.asarray(
        np.concatenate([arrays[name][train_index] for name in views], axis=1),
        dtype=np.float64,
    )
    test = np.asarray(
        np.concatenate([arrays[name][test_index] for name in views], axis=1),
        dtype=np.float64,
    )
    if pca_components:
        components = min(pca_components, len(train_index) - 1, train.shape[1])
        reducer = PCA(
            n_components=components,
            svd_solver="full",
        )
        train = reducer.fit_transform(train)
        test = reducer.transform(test)
    if not np.isfinite(train).all() or not np.isfinite(test).all():
        raise ValueError("BGE fold preprocessing produced non-finite values")
    return np.asarray(train, dtype=np.float32), np.asarray(test, dtype=np.float32)


def fit_fold_candidate(
    spec: CandidateSpec,
    numeric_arrays: Mapping[str, np.ndarray],
    bge_arrays: Mapping[str, np.ndarray],
    train_index: np.ndarray,
    test_index: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    semantic_cache: dict[tuple[tuple[str, ...], int], tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    key = (spec.semantic_views, spec.pca_components)
    if key not in semantic_cache:
        semantic_cache[key] = semantic_fold_block(
            spec.semantic_views,
            spec.pca_components,
            bge_arrays,
            train_index,
            test_index,
        )
    semantic_train, semantic_test = semantic_cache[key]
    train_parts: list[np.ndarray] = []
    test_parts: list[np.ndarray] = []
    if spec.numeric_matrix is not None:
        train_parts.append(numeric_arrays[spec.numeric_matrix][train_index])
        test_parts.append(numeric_arrays[spec.numeric_matrix][test_index])
    if semantic_train.shape[1]:
        train_parts.append(semantic_train)
        test_parts.append(semantic_test)
    if not train_parts:
        raise ValueError(f"candidate has no features: {spec.name}")
    train = np.concatenate(train_parts, axis=1)
    test = np.concatenate(test_parts, axis=1)
    if spec.model == "logreg":
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    model = make_model(spec)
    model.fit(train, labels[train_index], sample_weight=sample_weight)
    probability = np.asarray(model.predict_proba(test)[:, 1], dtype=np.float64)
    if probability.shape != (len(test_index),) or not np.isfinite(probability).all():
        raise AssertionError(f"invalid BGE probability vector: {spec.name}")
    return probability


def write_probabilities(path: Path, probabilities: Mapping[str, np.ndarray]) -> None:
    if any(
        values.shape != (auxiliary.EXPECTED_EXACT_ROWS,) or not np.isfinite(values).all()
        for values in probabilities.values()
    ):
        raise ValueError("full BGE OOF artifact requires 300 finite rows per candidate")
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


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        "bge_cache": args.bge_cache.resolve(),
        "bge_manifest": args.bge_manifest.resolve(),
        "numeric_cache": args.numeric_cache.resolve(),
        "numeric_manifest": args.numeric_manifest.resolve(),
        "component_cache": args.component_cache.resolve(),
        "component_manifest": args.component_manifest.resolve(),
        "exact_identity": args.exact_identity.resolve(),
        "full_identity": args.full_identity.resolve(),
        "fold_manifest": args.fold_manifest.resolve(),
        "targets": args.targets.resolve(),
        "pdf_dir": args.pdf_dir.resolve(),
        "output_dir": args.output_dir.resolve(),
    }
    paths["output_dir"].mkdir(parents=True, exist_ok=True)
    if base.file_sha256(paths["exact_identity"]) != auxiliary.EXACT_IDENTITY_SHA256:
        raise ValueError("exact300 identity hash mismatch")
    if base.file_sha256(paths["full_identity"]) != auxiliary.FULL_IDENTITY_SHA256:
        raise ValueError("full auxiliary identity hash mismatch")
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
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from full auxiliary identity")
    component_arrays, component_manifest = component_eval.validate_component_cache(
        paths["component_cache"],
        paths["component_manifest"],
        len(full_records),
        full_alignment,
    )
    bge_arrays, bge_manifest = validate_bge_cache(
        paths["bge_cache"], paths["bge_manifest"], paths["full_identity"]
    )
    frozen_numeric = {**numeric_arrays, **component_arrays}
    specs = candidate_specs()
    if any(
        spec.numeric_matrix is not None and spec.numeric_matrix not in frozen_numeric
        for spec in specs
    ):
        raise ValueError("candidate references an absent frozen numeric matrix")
    if any(view not in bge_arrays for spec in specs for view in spec.semantic_views):
        raise ValueError("candidate references an absent frozen BGE view")

    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
    freeze_path = paths["output_dir"] / (
        "candidate_freeze.json"
        if full_run
        else "candidate_freeze_folds_" + "_".join(map(str, args.folds)) + ".json"
    )
    candidate_freeze = {
        "schema_version": "r3_exact300_bge_semantic_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "evaluated_folds": list(args.folds),
        "candidates": [asdict(spec) for spec in specs],
        "threshold": 0.5,
        "inputs": {
            "bge_cache": {"path": str(paths["bge_cache"]), "sha256": BGE_CACHE_SHA256},
            "bge_manifest": {
                "path": str(paths["bge_manifest"]),
                "sha256": BGE_MANIFEST_SHA256,
            },
            "numeric_cache": {
                "path": str(paths["numeric_cache"]),
                "sha256": base.file_sha256(paths["numeric_cache"]),
            },
            "component_cache": {
                "path": str(paths["component_cache"]),
                "sha256": base.file_sha256(paths["component_cache"]),
            },
            "fold_manifest_sha256": auxiliary.FOLD_MANIFEST_SHA256,
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "field_table_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_feature": False,
            "identifiers_used_as_predictive_features": False,
            "all_preprocessing_fit_inside_outer_training_partition": True,
        },
        "script": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    base.atomic_json(freeze_path, candidate_freeze)

    # This is the first label access in the evaluation.
    labels_by_resultno = base.load_training_labels(
        paths["targets"], {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in full_records], dtype=np.int8
    )
    exact_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in exact_records], dtype=np.int8
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
        train_refnos = {full_records[int(index)]["REFNO"] for index in train_full}
        if train_refnos & test_refnos:
            raise AssertionError("outer test paper leaked into BGE auxiliary training")
        weights = base.inverse_target_multiplicity_weights(
            train_full, full_records, canonical_targets
        )
        fold_semantic_cache: dict[
            tuple[tuple[str, ...], int], tuple[np.ndarray, np.ndarray]
        ] = {}
        for spec in specs:
            probabilities[spec.name][exact_test] = fit_fold_candidate(
                spec,
                frozen_numeric,
                bge_arrays,
                train_full,
                test_full,
                full_labels,
                weights,
                fold_semantic_cache,
            )
        excluded_rows = sum(record["REFNO"] in test_refnos for record in full_records)
        fold_audit.append(
            {
                "fold": fold,
                "test_rows": int(len(exact_test)),
                "test_papers": len(test_refnos),
                "test_refno_set_sha256": auxiliary.sha256_strings(sorted(test_refnos)),
                "auxiliary_train_rows": int(len(train_full)),
                "auxiliary_train_papers": len(train_refnos),
                "all_test_paper_rows_excluded": int(excluded_rows),
                "train_test_refno_overlap": 0,
                "pca_fit_rows": int(len(train_full)),
                "pca_test_rows_seen_during_fit": 0,
            }
        )
        print(
            f"fold {fold}: candidates={len(specs)}, train_rows={len(train_full)}, "
            f"test_rows={len(exact_test)}, excluded_same_paper_rows={excluded_rows}",
            flush=True,
        )

    covered = np.isin(fold_ids, np.asarray(args.folds, dtype=np.int8))
    baseline = probabilities[CONTROL_NAME][covered]
    results: dict[str, Any] = {}
    for spec in specs:
        values = probabilities[spec.name]
        if not np.isfinite(values[covered]).all():
            raise AssertionError(f"candidate missing requested folds: {spec.name}")
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
            "transition_vs_control": shared_eval.transition_record(
                exact_labels[covered], baseline, values[covered]
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
    control_accuracy = float(results[CONTROL_NAME]["aggregate"]["accuracy"])
    best_accuracy = float(results[best]["aggregate"]["accuracy"])
    oof_record = None
    if full_run:
        oof_path = paths["output_dir"] / "oof_probabilities.npz"
        write_probabilities(oof_path, probabilities)
        oof_record = {"path": str(oof_path), "sha256": base.file_sha256(oof_path)}
    report = {
        "schema_version": "r3_exact300_bge_semantic_auxiliary_v1",
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
            "semantic_cache_frozen_without_labels_or_identifiers": True,
            "candidate_receipt_written_before_label_access": True,
            "all_outer_test_refno_rows_excluded_from_training": True,
            "pca_and_scaling_fit_only_on_outer_training_rows": True,
            "field_table_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_feature": False,
            "identifiers_used_as_predictive_features": False,
            "fixed_threshold": 0.5,
        },
        "inputs": {
            **candidate_freeze["inputs"],
            "targets_postfreeze_sha256": auxiliary.TARGETS_SHA256,
        },
        "feature_summary": {
            "encoder": bge_manifest.get("encoder"),
            "bge_matrices": bge_manifest.get("matrices"),
            "component_matrices": component_manifest.get("matrices"),
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
            "control_strategy": CONTROL_NAME,
            "control_accuracy_on_same_folds": control_accuracy,
            "best_accuracy_on_same_folds": best_accuracy,
            "strict_improvement": best_accuracy > control_accuracy,
            "continue_to_full_five_folds": bool(not full_run and best_accuracy > control_accuracy),
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
            "script": candidate_freeze["script"],
        },
    }
    report_path = paths["output_dir"] / (
        "report.json"
        if full_run
        else "folds_" + "_".join(map(str, args.folds)) + "_report.json"
    )
    base.atomic_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bge-cache", type=Path, default=DEFAULT_BGE_CACHE)
    parser.add_argument("--bge-manifest", type=Path, default=DEFAULT_BGE_MANIFEST)
    parser.add_argument("--numeric-cache", type=Path, default=DEFAULT_NUMERIC_CACHE)
    parser.add_argument("--numeric-manifest", type=Path, default=DEFAULT_NUMERIC_MANIFEST)
    parser.add_argument("--component-cache", type=Path, default=DEFAULT_COMPONENT_CACHE)
    parser.add_argument("--component-manifest", type=Path, default=DEFAULT_COMPONENT_MANIFEST)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("1,2,3"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    report = evaluate(args)
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
