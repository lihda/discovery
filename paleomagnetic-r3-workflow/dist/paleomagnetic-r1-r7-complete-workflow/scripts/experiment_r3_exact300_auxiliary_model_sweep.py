#!/usr/bin/env python3
"""Canonical exact300 sweep of compact non-boosted numeric model families."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_exact300_auxiliary_numeric_cache as cache_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_CACHE = cache_builder.DEFAULT_CACHE
DEFAULT_CACHE_MANIFEST = cache_builder.DEFAULT_MANIFEST
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/auxiliary_model_sweep_v1"
CACHE_SHA256 = "12b600c620ce069f3762a2741aff6b803f2295a2d3c834097eb5bdfbebbe07b5"
CACHE_MANIFEST_SHA256 = (
    "1dad1f75cbfdb1f1d906e9370951bc8005cb28963ac37ba2d58cf11416246cbb"
)
TARGET_ACCURACY = 0.90
RANDOM_STATE = 9357
REFERENCE_FOLD_ACCURACY = (0.90, 0.86666667, 0.78333333, 0.90, 0.91666667)
FIRST_THREE_FAMILY_MARGIN = 0.01
FAMILIES = ("linear", "kernel", "forest", "neighbors", "mlp")


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    family: str
    matrix: str
    kind: str
    parameter: float | int | str
    standardized: bool


def candidate_specs() -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = []
    for matrix in ("curation_all", "combined"):
        for value in (0.01, 0.1, 1.0):
            specs.append(
                CandidateSpec(
                    f"logreg_{matrix}_c{value:g}",
                    "linear",
                    matrix,
                    "logreg",
                    value,
                    True,
                )
            )
        for value in (0.001, 0.01, 0.1):
            specs.append(
                CandidateSpec(
                    f"linear_svc_{matrix}_c{value:g}",
                    "linear",
                    matrix,
                    "linear_svc",
                    value,
                    True,
                )
            )
        for value in (1.0, 10.0):
            specs.append(
                CandidateSpec(
                    f"rbf_svc_{matrix}_c{value:g}",
                    "kernel",
                    matrix,
                    "rbf_svc",
                    value,
                    True,
                )
            )
        specs.append(
            CandidateSpec(
                f"mlp_{matrix}_h32",
                "mlp",
                matrix,
                "mlp",
                32,
                True,
            )
        )
    specs.extend(
        (
            CandidateSpec(
                "rf_curation_all_leaf3_sqrt",
                "forest",
                "curation_all",
                "rf_leaf3_sqrt",
                3,
                False,
            ),
            CandidateSpec(
                "rf_combined_leaf3_sqrt",
                "forest",
                "combined",
                "rf_leaf3_sqrt",
                3,
                False,
            ),
            CandidateSpec(
                "rf_combined_leaf8_half",
                "forest",
                "combined",
                "rf_leaf8_half",
                8,
                False,
            ),
            CandidateSpec(
                "extra_combined_leaf1_sqrt",
                "forest",
                "combined",
                "extra_leaf1_sqrt",
                1,
                False,
            ),
            CandidateSpec(
                "extra_combined_leaf8_half",
                "forest",
                "combined",
                "extra_leaf8_half",
                8,
                False,
            ),
        )
    )
    for neighbors in (15, 31, 61):
        specs.append(
            CandidateSpec(
                f"knn_curation_all_k{neighbors}",
                "neighbors",
                "curation_all",
                "knn",
                neighbors,
                True,
            )
        )
    return tuple(specs)


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


def parse_families(value: str) -> tuple[str, ...]:
    if value.casefold() == "all":
        return FAMILIES
    families = tuple(sorted({item.strip() for item in value.split(",") if item.strip()}))
    unknown = set(families) - set(FAMILIES)
    if not families or unknown:
        raise argparse.ArgumentTypeError(
            f"families must be 'all' or a comma-separated subset of {FAMILIES}"
        )
    return families


def validate_cache(
    path: Path,
    manifest_path: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if base.file_sha256(path) != CACHE_SHA256:
        raise ValueError("auxiliary numeric cache hash mismatch")
    if base.file_sha256(manifest_path) != CACHE_MANIFEST_SHA256:
        raise ValueError("auxiliary numeric cache manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    if (
        manifest.get("schema_version") != "r3_exact300_auxiliary_numeric_cache_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("output", {}).get("sha256") != CACHE_SHA256
        or boundary.get("labels_read") is not False
        or boundary.get("targets_workbook_read") is not False
        or boundary.get("pmagresult_or_rockmag_read") is not False
        or boundary.get("prior_predictions_or_evaluations_read") is not False
        or boundary.get("numeric_cache_contains_identifiers") is not False
    ):
        raise ValueError("numeric cache manifest does not prove a label-free build")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {
            name: np.asarray(payload[name], dtype=np.float32)
            for name in payload.files
        }
    audit = cache_builder.validate_arrays(arrays)
    if audit != manifest.get("matrices"):
        raise ValueError("numeric cache arrays differ from the frozen manifest")
    return arrays, manifest


def make_model(spec: CandidateSpec) -> Any:
    value = spec.parameter
    if spec.kind == "logreg":
        return LogisticRegression(
            C=float(value),
            solver="liblinear",
            max_iter=2_000,
            random_state=RANDOM_STATE,
        )
    if spec.kind == "linear_svc":
        return LinearSVC(
            C=float(value), max_iter=10_000, random_state=RANDOM_STATE
        )
    if spec.kind == "rbf_svc":
        return SVC(
            C=float(value),
            kernel="rbf",
            gamma="scale",
            probability=False,
            cache_size=1_024,
            random_state=RANDOM_STATE,
        )
    if spec.kind == "rf_leaf3_sqrt":
        return RandomForestClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if spec.kind == "rf_leaf8_half":
        return RandomForestClassifier(
            n_estimators=600,
            max_features=0.5,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if spec.kind == "extra_leaf1_sqrt":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_features="sqrt",
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if spec.kind == "extra_leaf8_half":
        return ExtraTreesClassifier(
            n_estimators=600,
            max_features=0.5,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    if spec.kind == "knn":
        return KNeighborsClassifier(
            n_neighbors=int(value), weights="distance", p=2, n_jobs=-1
        )
    if spec.kind == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(int(value),),
            activation="relu",
            solver="adam",
            alpha=0.01,
            batch_size=64,
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=False,
            random_state=RANDOM_STATE,
        )
    raise ValueError(f"unknown candidate kind: {spec.kind}")


def predict_score(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return expit(np.asarray(model.decision_function(matrix), dtype=np.float64))


def fit_candidate(
    spec: CandidateSpec,
    matrix: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
) -> np.ndarray:
    train = matrix[train_index]
    test = matrix[test_index]
    if spec.standardized:
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        test = scaler.transform(test)
    model = make_model(spec)
    if spec.kind == "knn":
        model.fit(train, labels[train_index])
    else:
        model.fit(train, labels[train_index], sample_weight=sample_weight)
    probability = predict_score(model, test)
    if probability.shape != (len(test_index),) or not np.isfinite(probability).all():
        raise AssertionError(f"candidate produced invalid probabilities: {spec.name}")
    return probability


def family_gate(
    results: Mapping[str, Mapping[str, Any]],
    specs: Sequence[CandidateSpec],
    folds: Sequence[int],
) -> dict[str, Any]:
    reference = float(np.mean([REFERENCE_FOLD_ACCURACY[fold - 1] for fold in folds]))
    minimum = reference - FIRST_THREE_FAMILY_MARGIN
    output: dict[str, Any] = {}
    for family in FAMILIES:
        names = [spec.name for spec in specs if spec.family == family]
        if not names:
            continue
        best = max(
            names,
            key=lambda name: (
                results[name]["aggregate"]["accuracy"],
                results[name]["aggregate"]["macro_f1"],
            ),
        )
        accuracy = float(results[best]["aggregate"]["accuracy"])
        output[family] = {
            "best_candidate": best,
            "best_accuracy": accuracy,
            "reference_accuracy_on_same_folds": round(reference, 8),
            "continuation_minimum": round(minimum, 8),
            "continue_to_full_five_folds": accuracy >= minimum,
        }
    return output


def write_oof(path: Path, probabilities: Mapping[str, np.ndarray]) -> None:
    if any(values.shape != (auxiliary.EXPECTED_EXACT_ROWS,) for values in probabilities.values()):
        raise ValueError("full OOF artifact requires exactly 300 rows per candidate")
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise ValueError("full OOF artifact contains incomplete values")
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
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--folds", type=parse_folds, default=parse_folds("1,2,3"))
    parser.add_argument("--families", type=parse_families, default=FAMILIES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    cache_path = args.cache.resolve()
    cache_manifest_path = args.cache_manifest.resolve()
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
    arrays, cache_manifest = validate_cache(cache_path, cache_manifest_path)
    if cache_manifest.get("scope", {}).get("full_alignment_sha256") != auxiliary.sha256_records(
        full_records
    ):
        raise ValueError("numeric cache alignment differs from auxiliary identity")

    all_specs = candidate_specs()
    specs = tuple(spec for spec in all_specs if spec.family in set(args.families))
    if not specs:
        raise ValueError("no model candidates selected")

    # Labels are opened only after identities, canonical folds, cache, and candidates validate.
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
        train_weights = base.inverse_target_multiplicity_weights(
            train_full, full_records, canonical_targets
        )
        for spec in specs:
            probabilities[spec.name][exact_test] = fit_candidate(
                spec,
                arrays[spec.matrix],
                train_full,
                test_full,
                full_labels,
                train_weights,
            )
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
    results: dict[str, Any] = {}
    for spec in specs:
        values = probabilities[spec.name]
        if not np.isfinite(values[covered]).all():
            raise AssertionError(f"candidate missing a requested fold: {spec.name}")
        aggregate = base.metric_record(exact_labels[covered], values[covered])
        folds = [
            {
                "fold": fold,
                **base.metric_record(
                    exact_labels[fold_ids == fold], values[fold_ids == fold]
                ),
            }
            for fold in args.folds
        ]
        results[spec.name] = {
            "spec": asdict(spec),
            "aggregate": aggregate,
            "folds": folds,
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
    full_run = args.folds == tuple(range(1, auxiliary.N_SPLITS + 1))
    oof_record = None
    if full_run:
        oof_path = output_dir / "oof_probabilities.npz"
        write_oof(oof_path, probabilities)
        oof_record = {"path": str(oof_path), "sha256": base.file_sha256(oof_path)}
    report = {
        "schema_version": "r3_exact300_auxiliary_numeric_model_sweep_v1",
        "status": "complete",
        "mode": "full_five_fold" if full_run else "staged_family_screen",
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
            "labels_opened_after_cache_and_candidate_validation": True,
            "all_outer_test_refno_rows_excluded_from_training": True,
            "pmagresult_or_rockmag_used_as_feature": False,
            "prior_oof_predictions_used_as_feature": False,
            "identifiers_used_as_predictive_features": False,
            "hyperparameters_predeclared_before_label_access": True,
            "model_threshold": 0.5,
        },
        "inputs": {
            "cache": {"path": str(cache_path), "sha256": base.file_sha256(cache_path)},
            "cache_manifest": {
                "path": str(cache_manifest_path),
                "sha256": base.file_sha256(cache_manifest_path),
            },
            "exact_identity": auxiliary.EXACT_IDENTITY_SHA256,
            "full_identity": auxiliary.FULL_IDENTITY_SHA256,
            "fold_manifest": auxiliary.FOLD_MANIFEST_SHA256,
            "targets_postfreeze": auxiliary.TARGETS_SHA256,
        },
        "candidate_freeze": {
            "all_predeclared_candidates": [asdict(spec) for spec in all_specs],
            "selected_families_for_this_run": list(args.families),
            "evaluated_candidates": [spec.name for spec in specs],
        },
        "outer_fold_audit": fold_audit,
        "family_continuation_gate": family_gate(results, specs, args.folds),
        "ranking": ranking,
        "best_observed": {"strategy": best, **results[best]},
        "models": results,
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "full_coverage_required": True,
            "passed": bool(
                full_run and results[best]["aggregate"]["accuracy"] >= TARGET_ACCURACY
            ),
            "decision": (
                "EXACT300_TARGET_REACHED"
                if full_run and results[best]["aggregate"]["accuracy"] >= TARGET_ACCURACY
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
        "report.json" if full_run else "folds_" + "_".join(map(str, args.folds)) + "_report.json"
    )
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "best_observed": report["best_observed"],
                "family_continuation_gate": report["family_continuation_gate"],
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
