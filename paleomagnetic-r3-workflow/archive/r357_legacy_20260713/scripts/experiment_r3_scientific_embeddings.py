#!/usr/bin/env python3
"""Grouped OOF R3 evaluation of label-free SciNCL original-text features."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_scientific_embeddings as builder  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


DEFAULT_EMBEDDINGS = ROOT / "tmp/r3_scientific_embeddings/scincl_row_features.npz"
DEFAULT_MANIFEST = ROOT / "tmp/r3_scientific_embeddings/scincl_row_features.manifest.json"
DEFAULT_OUTPUT = ROOT / "tmp/r3_scientific_embeddings/scincl_grouped_oof.json"
DEFAULT_PROBABILITIES = ROOT / "tmp/r3_scientific_embeddings/scincl_grouped_oof.npz"
DEFAULT_SCREENING_PROTOCOL = ROOT / "tmp/r3_small_screen_v1/screening_protocol.json"
SCREENING_PROTOCOL_SHA256 = "2488bb8dc2167ae93bfedffa315f44f7968049c7895ef803a9e598aa066e291d"
EXPECTED_ARRAYS = {
    "abstract",
    "paper_mean",
    "paper_max",
    "paper_top1",
    "paper_top3",
    "local_mean",
    "local_max",
    "local_top1",
    "local_top3",
    "retrieval_scores",
    "engineered",
}


def load_feature_views(path: Path, row_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        names = set(payload.files)
        if names != EXPECTED_ARRAYS:
            raise ValueError(
                "SciNCL cache must contain only the expected numerical arrays; "
                f"missing={sorted(EXPECTED_ARRAYS - names)}, "
                f"extra={sorted(names - EXPECTED_ARRAYS)}"
            )
        arrays = {
            name: np.asarray(payload[name], dtype=np.float32) for name in names
        }
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in arrays.values()):
        raise ValueError("SciNCL feature arrays are not aligned to every PDF-only row")
    if any(not np.isfinite(matrix).all() for matrix in arrays.values()):
        raise ValueError("SciNCL feature cache contains non-finite values")

    def join(*names: str) -> np.ndarray:
        return np.concatenate([arrays[name] for name in names], axis=1)

    return {
        "abstract": arrays["abstract"],
        "paper_mean": arrays["paper_mean"],
        "paper_top3": arrays["paper_top3"],
        "paper_mean_top3": join("paper_mean", "paper_top3"),
        "local_mean": arrays["local_mean"],
        "local_top3": arrays["local_top3"],
        "local_mean_top3": join("local_mean", "local_top3"),
        "paper_semantic": join("abstract", "paper_mean", "paper_top3"),
        "target_semantic": join(
            "abstract", "paper_top3", "local_mean", "local_top3"
        ),
        "target_semantic_engineered": join(
            "abstract",
            "paper_top3",
            "local_mean",
            "local_top3",
            "retrieval_scores",
            "engineered",
        ),
        "target_all_pools": join(
            "abstract",
            "paper_mean",
            "paper_max",
            "paper_top1",
            "paper_top3",
            "local_mean",
            "local_max",
            "local_top1",
            "local_top3",
            "retrieval_scores",
            "engineered",
        ),
        "knowledge_scores": join("retrieval_scores", "engineered"),
        "engineered_only": arrays["engineered"],
    }


def model_specs() -> list[tuple[str, bool, Any]]:
    specs: list[tuple[str, bool, Any]] = []
    for c_value in (0.05, 0.25, 1.0, 4.0, 16.0):
        specs.append(
            (
                f"logreg_raw_c{c_value:g}",
                False,
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=base.RANDOM_STATE,
                ),
            )
        )
    for c_value in (0.002, 0.01, 0.05, 0.25):
        specs.append(
            (
                f"logreg_scaled_c{c_value:g}",
                True,
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=2_000,
                    random_state=base.RANDOM_STATE,
                ),
            )
        )
    for c_value in (0.05, 0.25, 1.0):
        specs.append(
            (
                f"linear_svc_raw_c{c_value:g}",
                False,
                LinearSVC(
                    C=c_value,
                    max_iter=8_000,
                    random_state=base.RANDOM_STATE,
                ),
            )
        )
    for c_value in (0.0005, 0.002, 0.01):
        specs.append(
            (
                f"linear_svc_scaled_c{c_value:g}",
                True,
                LinearSVC(
                    C=c_value,
                    max_iter=8_000,
                    random_state=base.RANDOM_STATE,
                ),
            )
        )
    return specs


def model_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return expit(np.asarray(model.decision_function(matrix), dtype=np.float64))


def inverse_target_weights(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    canonical_targets: Sequence[str],
) -> np.ndarray:
    counts = Counter(
        (records[index]["REFNO"], canonical_targets[index]) for index in indices
    )
    return np.asarray(
        [
            1.0 / counts[(records[index]["REFNO"], canonical_targets[index])]
            for index in indices
        ],
        dtype=np.float64,
    )


def run_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    views: Mapping[str, np.ndarray],
) -> tuple[
    dict[str, np.ndarray],
    dict[str, list[dict[str, Any]]],
    list[np.ndarray],
]:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    canonical_targets = [
        canonical_target_identity(record["ROCKNAME"]) for record in records
    ]
    specs = model_specs()
    probabilities = {
        f"{view_name}__{model_name}": np.full(len(records), np.nan, dtype=np.float64)
        for view_name in views
        for model_name, _, _ in specs
    }
    folds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_indices: list[np.ndarray] = []
    splitter = GroupKFold(n_splits=base.N_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(records)), labels, groups), start=1
    ):
        if set(groups[train_index]) & set(groups[test_index]):
            raise AssertionError("REFNO leaked across a SciNCL outer fold")
        fold_indices.append(test_index)
        weights = inverse_target_weights(train_index, records, canonical_targets)
        for view_name, matrix in views.items():
            raw_train = matrix[train_index]
            raw_test = matrix[test_index]
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(raw_train)
            scaled_test = scaler.transform(raw_test)
            for model_name, scaled, model in model_specs():
                train_matrix = scaled_train if scaled else raw_train
                test_matrix = scaled_test if scaled else raw_test
                model.fit(
                    train_matrix,
                    labels[train_index],
                    sample_weight=weights,
                )
                predicted = model_probability(model, test_matrix)
                key = f"{view_name}__{model_name}"
                probabilities[key][test_index] = predicted
                folds[key].append(
                    {"fold": fold, **base.metric_record(labels[test_index], predicted)}
                )
        print(
            f"fold {fold}/{base.N_SPLITS}: train_rows={len(train_index)}, "
            f"test_rows={len(test_index)}",
            flush=True,
        )
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("one or more SciNCL OOF arrays are incomplete")
    return probabilities, dict(folds), fold_indices


def paper_propagation_strategies(
    records: Sequence[Mapping[str, str]],
    base_probabilities: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Add label-free multi-target paper pooling at fixed blend weights."""

    members: dict[str, list[int]] = defaultdict(list)
    targets: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        members[record["REFNO"]].append(index)
        targets[record["REFNO"]].add(canonical_target_identity(record["ROCKNAME"]))
    multi = np.asarray([len(targets[record["REFNO"]]) > 1 for record in records])
    output = dict(base_probabilities)
    for name, values in base_probabilities.items():
        paper_mean = values.copy()
        for indices in members.values():
            paper_mean[indices] = float(values[indices].mean())
        for weight in (0.25, 0.50, 0.75, 1.00):
            blended = values.copy()
            blended[multi] = (
                (1.0 - weight) * values[multi] + weight * paper_mean[multi]
            )
            output[f"{name}__multi_paper_{int(weight * 100):03d}"] = blended
    return output


def fold_metrics(
    labels: np.ndarray,
    values: np.ndarray,
    indices: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    return [
        {"fold": fold, **base.metric_record(labels[index], values[index])}
        for fold, index in enumerate(indices, start=1)
    ]


def taxonomy_baseline(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    *,
    page_cache: Path,
    taxonomy: Path,
) -> dict[str, Any]:
    refnos = {record["REFNO"] for record in records}
    paper_text = base.load_page_text(page_cache, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(taxonomy, None)
    features = base.build_features(records, paper_text, concepts)
    raw, fold_indices, _, _ = base.run_grouped_oof(records, labels, features)
    strategies = base.build_strategies(records, features, raw)
    metrics = {
        name: base.metric_record(labels, value["probabilities"])
        for name, value in strategies.items()
    }


def load_taxonomy_baseline_report(
    path: Path,
    *,
    identity_path: Path,
    expected_rows: int,
) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    report_identity = report.get("inputs", {}).get("training_identity", {})
    if report_identity.get("sha256") != base.file_sha256(identity_path):
        raise ValueError("external taxonomy baseline uses a different identity partition")
    if report.get("data", {}).get("pdf_only_rows") != expected_rows:
        raise ValueError("external taxonomy baseline uses a different PDF-only row set")
    strict = report["best_strict_scope_candidate"]
    strategy = str(strict["strategy"])
    return {
        "source": {
            "path": str(path),
            "sha256": base.file_sha256(path),
            "reused_existing_grouped_oof": True,
        },
        "knowledge": report.get("knowledge"),
        "best_full_coverage": dict(strict),
        "best_folds": report["strategies"][strategy]["folds"],
        "diagnostic_unscoped_best": report.get("best_full_coverage"),
    }
    ranking = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["accuracy"],
            metrics[name]["macro_f1"],
            metrics[name]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best_name = ranking[0]
    return {
        "knowledge": knowledge_audit,
        "ranking": ranking,
        "best_full_coverage": {"model": best_name, **metrics[best_name]},
        "best_folds": fold_metrics(
            labels, strategies[best_name]["probabilities"], fold_indices
        ),
        "preregistered_full_baseline": {
            "model": "multi_all_article_50",
            **metrics["multi_all_article_50"],
        },
    }


def write_probability_npz(
    path: Path,
    probabilities: Mapping[str, np.ndarray],
) -> None:
    forbidden = re.compile(
        r"(?i)(?:^|__|_)(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)(?:$|__|_)"
    )
    invalid = sorted(name for name in probabilities if forbidden.search(name))
    if invalid:
        raise ValueError(f"OOF probability names contain forbidden fields: {invalid}")
    if any(np.asarray(value).ndim != 1 for value in probabilities.values()):
        raise ValueError("OOF probability cache accepts one-dimensional arrays only")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(value, dtype=np.float32)
                for name, value in sorted(probabilities.items())
            },
        )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=base.DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    parser.add_argument("--skip-taxonomy-baseline", action="store_true")
    parser.add_argument("--taxonomy-baseline-report", type=Path)
    parser.add_argument(
        "--screening-protocol", type=Path
    )
    args = parser.parse_args(argv)

    identity_path = args.identity.resolve()
    labels_path = args.targets.resolve()
    pdf_dir = args.pdf_dir.resolve()
    embeddings_path = args.embeddings.resolve()
    manifest_path = args.embedding_manifest.resolve()
    protocol_path = (
        args.screening_protocol.resolve()
        if args.screening_protocol is not None
        else None
    )
    protocol = None
    gate = None
    if protocol_path is not None:
        if base.file_sha256(protocol_path) != SCREENING_PROTOCOL_SHA256:
            raise ValueError("R3 small-screen protocol hash mismatch")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if protocol.get("selection", {}).get("identity_sha256") != base.file_sha256(identity_path):
            raise ValueError("R3 small-screen protocol does not match the requested identity")
        gate = protocol["promotion_gate"]
    all_records, coverage = base.load_pdf_only_identity(identity_path, pdf_dir)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    required_false = (
        "labels_read",
        "pmagresult_or_rockmag_read",
        "prior_predictions_or_evaluations_read",
        "identifier_features",
        "numeric_cache_contains_identifiers",
    )
    if any(boundary.get(name) is not False for name in required_false):
        raise ValueError("SciNCL manifest does not prove a strict label/identifier-free build")
    if manifest.get("inputs", {}).get("identity", {}).get("sha256") != base.file_sha256(identity_path):
        raise ValueError("SciNCL manifest identity hash mismatch")
    requested_papers = int(
        manifest.get("scope", {}).get("selection", {}).get("requested_papers", 0)
    )
    records, selection_audit = builder.source_stratified_pilot(
        all_records, requested_papers
    )
    if manifest.get("scope", {}).get("selection") != selection_audit:
        raise ValueError("SciNCL pilot selection audit mismatch")
    if manifest.get("inputs", {}).get("alignment_sha256") != builder.alignment_sha256(records):
        raise ValueError("SciNCL feature row alignment mismatch")
    if manifest.get("output", {}).get("sha256") != base.file_sha256(embeddings_path):
        raise ValueError("SciNCL numerical cache hash mismatch")
    views = load_feature_views(embeddings_path, len(records))

    # Labels are dereferenced only after PDF scope and the label-free cache pass.
    labels_by_resultno = base.load_training_labels(
        labels_path, {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    raw_oof, raw_folds, fold_indices = run_oof(records, labels, views)
    strategies = paper_propagation_strategies(records, raw_oof)
    metrics = {
        name: base.metric_record(labels, values) for name, values in strategies.items()
    }
    ranking = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["accuracy"],
            metrics[name]["macro_f1"],
            metrics[name]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best_name = ranking[0]
    models = {}
    for name in strategies:
        models[name] = {
            "aggregate": metrics[name],
            "folds": raw_folds.get(name)
            or fold_metrics(labels, strategies[name], fold_indices),
        }
    baseline_result = None
    pilot_gate = None
    if args.taxonomy_baseline_report is not None:
        baseline_result = load_taxonomy_baseline_report(
            args.taxonomy_baseline_report.resolve(),
            identity_path=identity_path,
            expected_rows=len(records),
        )
    elif not args.skip_taxonomy_baseline:
        baseline_result = taxonomy_baseline(
            records,
            labels,
            page_cache=args.page_cache.resolve(),
            taxonomy=args.taxonomy.resolve(),
        )
    if baseline_result is not None and protocol is not None and gate is not None:
        scientific_folds = models[best_name]["folds"]
        baseline_folds = baseline_result["best_folds"]
        fold_deltas = [
            scientific["accuracy"] - baseline_fold["accuracy"]
            for scientific, baseline_fold in zip(scientific_folds, baseline_folds)
        ]
        folds_not_worse = sum(delta >= 0.0 for delta in fold_deltas)
        improvement = (
            metrics[best_name]["accuracy"]
            - baseline_result["best_full_coverage"]["accuracy"]
        )
        pilot_gate = {
            "protocol_status": protocol["status"],
            "minimum_aggregate_accuracy": gate["minimum_aggregate_accuracy"],
            "minimum_absolute_improvement": gate["minimum_absolute_gain_over_strict_baseline"],
            "minimum_folds_not_worse": gate["minimum_folds_not_worse_than_strict_baseline"],
            "maximum_single_fold_regression": gate["maximum_allowed_accuracy_regression_in_any_fold"],
            "observed_absolute_improvement": round(float(improvement), 8),
            "observed_folds_not_worse": int(folds_not_worse),
            "fold_accuracy_deltas": [round(float(value), 8) for value in fold_deltas],
            "worst_fold_regression": round(float(min(fold_deltas)), 8),
            "continue_to_full_encoding": bool(
                metrics[best_name]["accuracy"] >= gate["minimum_aggregate_accuracy"]
                and improvement >= gate["minimum_absolute_gain_over_strict_baseline"]
                and folds_not_worse >= gate["minimum_folds_not_worse_than_strict_baseline"]
                and min(fold_deltas)
                >= -gate["maximum_allowed_accuracy_regression_in_any_fold"]
            ),
            "warning": "Both family rankings are posthoc within the pilot; the margin gate reduces but does not remove selection optimism.",
        }
    result = {
        "schema_version": "r3_scientific_embeddings_grouped_oof_v1",
        "status": "complete",
        "scope": "development-only PDF-backed fixed training partition",
        "guardrails": {
            "cross_validation": "5-fold GroupKFold by REFNO",
            "embedding_cache_label_free": True,
            "labels_opened_after_pdf_scope_and_cache_validation": True,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifier_features": False,
            "oof_probability_cache_contains_labels_or_ids": False,
            "training_weight": "inverse REFNO/canonical-target multiplicity",
        },
        "data": {
            **coverage,
            "selection": selection_audit,
            "labels": dict(sorted(Counter(int(value) for value in labels).items())),
        },
        "inputs": {
            "identity_sha256": base.file_sha256(identity_path),
            "labels_sha256": base.file_sha256(labels_path),
            "embeddings_sha256": base.file_sha256(embeddings_path),
            "embedding_manifest_sha256": base.file_sha256(manifest_path),
            "screening_protocol": (
                {
                    "path": str(protocol_path),
                    "sha256": base.file_sha256(protocol_path),
                }
                if protocol_path is not None
                else None
            ),
        },
        "embedding_manifest": manifest,
        "models": models,
        "ranking": ranking,
        "best_full_coverage": {"model": best_name, **metrics[best_name]},
        "same_subset_taxonomy_baseline": baseline_result,
        "pilot_progression_gate": pilot_gate,
        "interpretation": {
            "target_accuracy": 0.90,
            "target_reached": metrics[best_name]["accuracy"] >= 0.90,
            "small_screen_strict_baseline": (
                baseline_result["best_full_coverage"]["accuracy"]
                if baseline_result is not None
                else None
            ),
            "beats_small_screen_strict_baseline": (
                metrics[best_name]["accuracy"]
                > baseline_result["best_full_coverage"]["accuracy"]
                if baseline_result is not None
                else None
            ),
            "full_weighted_reference": 0.83103131,
            "beats_full_weighted_reference": metrics[best_name]["accuracy"] > 0.83103131,
            "model_ranking_is_posthoc": True,
            "external_refno_validation_required": True,
        },
    }
    base.atomic_json(args.output.resolve(), result)
    probability_path = args.probabilities.resolve()
    write_probability_npz(probability_path, strategies)
    print(json.dumps(result["best_full_coverage"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
