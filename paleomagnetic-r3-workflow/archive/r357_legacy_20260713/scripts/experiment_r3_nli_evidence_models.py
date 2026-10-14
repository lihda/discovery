#!/usr/bin/env python3
"""Evaluate label-free NLI evidence features with REFNO-grouped OOF models."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_citation_graph as relation  # noqa: E402


DEFAULT_FEATURES = ROOT / "tmp/r3_nli_evidence/features.npz"
DEFAULT_MANIFEST = ROOT / "tmp/r3_nli_evidence/manifest.json"
DEFAULT_OUTPUT = ROOT / "tmp/r3_nli_evidence/oof_models.json"
DEFAULT_PROBABILITIES = ROOT / "tmp/r3_nli_evidence/oof_probabilities.npz"


def paper_members(records: Sequence[dict[str, str]]) -> dict[str, list[int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        members[record["REFNO"]].append(index)
    return dict(members)


def load_nli_features(
    path: Path,
    expected_refnos: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        refnos = [str(value) for value in payload["refnos"]]
        matrix = np.asarray(payload["features"], dtype=np.float64)
        names = [str(value) for value in payload["feature_names"]]
    if refnos != list(expected_refnos):
        raise ValueError("NLI feature REFNO order does not match the strict identity")
    if matrix.shape != (len(refnos), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("NLI feature matrix is malformed or non-finite")
    return matrix, names


def model_factories() -> list[tuple[str, Callable[[], Any]]]:
    return [
        (
            "logreg_c0.25",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.25,
                    solver="liblinear",
                    max_iter=1600,
                    random_state=base.RANDOM_STATE,
                ),
            ),
        ),
        (
            "logreg_c1",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    max_iter=1600,
                    random_state=base.RANDOM_STATE,
                ),
            ),
        ),
        (
            "gradient_depth1",
            lambda: GradientBoostingClassifier(
                n_estimators=180,
                learning_rate=0.035,
                max_depth=1,
                min_samples_leaf=8,
                random_state=base.RANDOM_STATE,
            ),
        ),
        (
            "hist_leaf7",
            lambda: HistGradientBoostingClassifier(
                max_iter=220,
                learning_rate=0.04,
                max_leaf_nodes=7,
                min_samples_leaf=10,
                l2_regularization=0.7,
                random_state=base.RANDOM_STATE,
            ),
        ),
        (
            "extra_trees",
            lambda: ExtraTreesClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=4,
                n_jobs=1,
                random_state=base.RANDOM_STATE,
            ),
        ),
    ]


def fit_with_optional_weight(
    model: Any,
    matrix: np.ndarray,
    labels: np.ndarray,
    weight: np.ndarray | None,
) -> None:
    if weight is None:
        model.fit(matrix, labels)
        return
    if hasattr(model, "named_steps"):
        final_name = list(model.named_steps)[-1]
        model.fit(matrix, labels, **{f"{final_name}__sample_weight": weight})
    else:
        model.fit(matrix, labels, sample_weight=weight)


def run_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    matrices: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[np.ndarray], dict[str, list[dict[str, Any]]]]:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    members = paper_members(records)
    refnos = sorted(members)
    paper_position = {refno: position for position, refno in enumerate(refnos)}
    majority = np.asarray(
        [int(labels[members[refno]].mean() >= 0.5) for refno in refnos],
        dtype=np.int8,
    )
    row_weights = np.asarray([len(members[refno]) for refno in refnos], dtype=np.float64)
    probabilities = {
        f"{view}__{model_name}__{weight_name}": np.full(len(records), np.nan)
        for view in matrices
        for model_name, _ in model_factories()
        for weight_name in ("paper_weight", "row_weight")
    }
    folds: list[np.ndarray] = []
    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    splitter = GroupKFold(n_splits=base.N_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(records)), labels, groups), start=1
    ):
        folds.append(test_index)
        train_refnos = sorted(set(str(groups[index]) for index in train_index))
        test_refnos = sorted(set(str(groups[index]) for index in test_index))
        if set(train_refnos) & set(test_refnos):
            raise AssertionError("REFNO leaked across an outer fold")
        train_positions = np.asarray([paper_position[value] for value in train_refnos])
        test_positions = np.asarray([paper_position[value] for value in test_refnos])
        for view, matrix in matrices.items():
            for model_name, factory in model_factories():
                for weight_name, weight in (
                    ("paper_weight", None),
                    ("row_weight", row_weights[train_positions]),
                ):
                    model = factory()
                    fit_with_optional_weight(
                        model,
                        matrix[train_positions],
                        majority[train_positions],
                        weight,
                    )
                    predicted_paper = model.predict_proba(matrix[test_positions])[:, 1]
                    lookup = dict(zip(test_refnos, predicted_paper))
                    predicted_rows = np.asarray(
                        [lookup[str(groups[index])] for index in test_index], dtype=np.float64
                    )
                    name = f"{view}__{model_name}__{weight_name}"
                    probabilities[name][test_index] = predicted_rows
                    fold_metrics[name].append(
                        {"fold": fold, **base.metric_record(labels[test_index], predicted_rows)}
                    )
        print(f"fold {fold}/{base.N_SPLITS} complete", flush=True)
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("incomplete NLI OOF probabilities")
    return probabilities, folds, dict(fold_metrics)


def run_sparse_structured_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    paper_documents: Mapping[str, str],
    structured: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    """Fit text and NLI/knowledge features jointly inside every outer fold."""

    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    members = paper_members(records)
    refnos = sorted(members)
    position = {refno: index for index, refno in enumerate(refnos)}
    majority = np.asarray(
        [int(labels[members[refno]].mean() >= 0.5) for refno in refnos],
        dtype=np.int8,
    )
    row_weights = np.asarray([len(members[refno]) for refno in refnos], dtype=np.float64)
    scales = (("s020", 0.20), ("s100", 1.00))
    model_names = ("logreg_c2", "linear_svc_c0.5", "linear_svc_c2")
    names = tuple(
        f"knowledge_char_structured_{scale_name}__{model_name}"
        for scale_name, _ in scales
        for model_name in model_names
    )
    probabilities = {name: np.full(len(records), np.nan) for name in names}
    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    splitter = GroupKFold(n_splits=base.N_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(records)), labels, groups), start=1
    ):
        train_refnos = sorted(set(str(groups[index]) for index in train_index))
        test_refnos = sorted(set(str(groups[index]) for index in test_index))
        train_position = np.asarray([position[value] for value in train_refnos])
        test_position = np.asarray([position[value] for value in test_refnos])
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_df=1.0,
            max_features=180_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        text_train = vectorizer.fit_transform(
            [paper_documents[refno] for refno in train_refnos]
        )
        text_test = vectorizer.transform(
            [paper_documents[refno] for refno in test_refnos]
        )
        scaler = StandardScaler()
        numeric_train = scaler.fit_transform(structured[train_position]).astype(np.float32)
        numeric_test = scaler.transform(structured[test_position]).astype(np.float32)
        for scale_name, scale in scales:
            train_matrix = hstack(
                (text_train, csr_matrix(numeric_train * scale)), format="csr"
            )
            test_matrix = hstack(
                (text_test, csr_matrix(numeric_test * scale)), format="csr"
            )
            models: tuple[tuple[str, Any], ...] = (
                (
                    "logreg_c2",
                    LogisticRegression(
                        C=2.0,
                        solver="liblinear",
                        max_iter=1800,
                        random_state=base.RANDOM_STATE,
                    ),
                ),
                (
                    "linear_svc_c0.5",
                    LinearSVC(C=0.5, max_iter=6000, random_state=base.RANDOM_STATE),
                ),
                (
                    "linear_svc_c2",
                    LinearSVC(C=2.0, max_iter=6000, random_state=base.RANDOM_STATE),
                ),
            )
            for model_name, model in models:
                name = f"knowledge_char_structured_{scale_name}__{model_name}"
                model.fit(
                    train_matrix,
                    majority[train_position],
                    sample_weight=row_weights[train_position],
                )
                if hasattr(model, "predict_proba"):
                    paper_probability = model.predict_proba(test_matrix)[:, 1]
                else:
                    paper_probability = expit(model.decision_function(test_matrix))
                lookup = dict(zip(test_refnos, paper_probability))
                row_probability = np.asarray(
                    [lookup[str(groups[index])] for index in test_index], dtype=np.float64
                )
                probabilities[name][test_index] = row_probability
                fold_metrics[name].append(
                    {"fold": fold, **base.metric_record(labels[test_index], row_probability)}
                )
        print(f"sparse structured fold {fold}/{base.N_SPLITS} complete", flush=True)
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("incomplete sparse structured OOF probabilities")
    return probabilities, dict(fold_metrics)


def fixed_nli_rules(
    features: np.ndarray,
    feature_names: Sequence[str],
    records: Sequence[dict[str, str]],
) -> dict[str, np.ndarray]:
    lookup = {name: index for index, name in enumerate(feature_names)}
    refnos = sorted({record["REFNO"] for record in records})
    row_positions = np.asarray(
        [{refno: index for index, refno in enumerate(refnos)}[record["REFNO"]] for record in records]
    )
    method = np.maximum(
        features[:, lookup["own_study_generic__max"]],
        features[:, lookup["concept_specific_method__max"]],
    )
    carrier = features[:, lookup["carrier_identification__max"]]
    joint = np.sqrt(np.clip(method * carrier, 0.0, 1.0))
    return {
        "fixed_nli_method": method[row_positions],
        "fixed_nli_carrier": carrier[row_positions],
        "fixed_nli_joint_geometric": joint[row_positions],
        "fixed_nli_minimum": np.minimum(method, carrier)[row_positions],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=base.DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-probabilities", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    args = parser.parse_args(argv)

    records, coverage = base.load_pdf_only_identity(
        args.identity.resolve(), args.pdf_dir.resolve()
    )
    resultnos = {record["RESULTNO"] for record in records}
    label_map = base.load_training_labels(args.targets.resolve(), resultnos)
    labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
    refnos = sorted({record["REFNO"] for record in records})
    feature_manifest = json.loads(args.feature_manifest.resolve().read_text(encoding="utf-8"))
    boundary = feature_manifest.get("runtime_boundary", {})
    if feature_manifest.get("status") != "PASS" or any(
        boundary.get(key) is not False
        for key in (
            "labels_read",
            "pmagresult_or_rockmag_read",
            "prior_predictions_or_evaluations_read",
        )
    ):
        raise ValueError("NLI feature manifest does not prove a label-free boundary")
    declared_feature = feature_manifest.get("output", {}).get("features", {})
    if declared_feature.get("sha256") != base.file_sha256(args.features.resolve()):
        raise ValueError("NLI feature artifact does not match its frozen manifest")
    if feature_manifest.get("scope", {}).get("processed_papers") != len(refnos):
        raise ValueError("NLI feature manifest does not cover every evaluated PDF paper")
    nli, feature_names = load_nli_features(args.features.resolve(), refnos)

    paper_text = base.load_page_text(args.page_cache.resolve(), set(refnos))
    concepts, knowledge_audit = base.load_knowledge_patterns(
        args.taxonomy.resolve(), None
    )
    source_features = base.build_features(records, paper_text, concepts)
    article = np.asarray(
        [source_features["article_features"][refno] for refno in refnos],
        dtype=np.float64,
    )
    profiles, _ = relation.load_page_profiles(args.page_cache.resolve(), set(refnos))
    relation_features = relation.relation_count_features(
        [profiles[refno] for refno in refnos]
    )
    matrices = {
        "nli": nli,
        "nli_plus_knowledge": np.concatenate((nli, article), axis=1),
        "nli_plus_knowledge_relation": np.concatenate(
            (nli, article, relation_features), axis=1
        ),
    }
    probabilities, fold_indices, fold_metrics = run_oof(records, labels, matrices)
    sparse_probabilities, sparse_fold_metrics = run_sparse_structured_oof(
        records,
        labels,
        source_features["article_documents"],
        matrices["nli_plus_knowledge_relation"],
    )
    probabilities.update(sparse_probabilities)
    fold_metrics.update(sparse_fold_metrics)
    probabilities.update(fixed_nli_rules(nli, feature_names, records))

    fusion_status: dict[str, Any] | None = None
    if args.baseline_probabilities is not None:
        with np.load(args.baseline_probabilities.resolve(), allow_pickle=False) as payload:
            required = "multi_all_article_50"
            if required not in payload.files:
                raise ValueError(f"baseline NPZ does not contain {required}")
            baseline = np.asarray(payload[required], dtype=np.float64)
        if baseline.shape != labels.shape or not np.isfinite(baseline).all():
            raise ValueError("baseline probability array is not row-aligned")
        nli_candidates = [
            name
            for name in probabilities
            if name.startswith("nli_plus_knowledge__")
        ]
        for name in nli_candidates:
            for weight in (0.25, 0.50, 0.75):
                probabilities[
                    f"diagnostic_blend_{int(weight * 100):02d}__{name}"
                ] = (1.0 - weight) * baseline + weight * probabilities[name]
        fusion_status = {
            "baseline_path": str(args.baseline_probabilities.resolve()),
            "baseline_sha256": base.file_sha256(args.baseline_probabilities.resolve()),
            "warning": (
                "predefined arithmetic blends of two OOF sources; useful for development "
                "diagnosis but a final deployable model must be refit inside each outer fold"
            ),
        }

    metrics = {name: base.metric_record(labels, values) for name, values in probabilities.items()}
    ranked = sorted(metrics, key=lambda name: metrics[name]["accuracy"], reverse=True)
    homogeneous, mixed = base.paper_homogeneity_masks(records, labels)
    output = {
        "schema_version": "r3_nli_evidence_grouped_oof_v1",
        "status": "PASS",
        "scope": {**coverage, "rows": len(records), "papers": len(refnos)},
        "guardrails": {
            "outer_split": "5-fold REFNO GroupKFold",
            "labels_used_as_features": False,
            "nli_feature_builder_accepts_labels": False,
            "source_table_runtime_access": False,
            "full_coverage": True,
        },
        "inputs": {
            "identity": {"path": str(args.identity.resolve()), "sha256": base.file_sha256(args.identity.resolve())},
            "targets_post_gate": {"path": str(args.targets.resolve()), "sha256": base.file_sha256(args.targets.resolve())},
            "features": {"path": str(args.features.resolve()), "sha256": base.file_sha256(args.features.resolve())},
            "feature_manifest": {"path": str(args.feature_manifest.resolve()), "sha256": base.file_sha256(args.feature_manifest.resolve())},
        },
        "knowledge": knowledge_audit,
        "feature_views": {name: int(matrix.shape[1]) for name, matrix in matrices.items()},
        "models": {
            name: {
                "aggregate": metrics[name],
                "folds": fold_metrics.get(name),
                "homogeneous_paper_rows": base.subset_metric(labels, values, homogeneous),
                "mixed_paper_rows": base.subset_metric(labels, values, mixed),
            }
            for name, values in probabilities.items()
        },
        "ranking_by_full_coverage_accuracy": ranked,
        "best_full_coverage": {"strategy": ranked[0], **metrics[ranked[0]]},
        "fusion": fusion_status,
        "interpretation": {
            "target_accuracy": 0.90,
            "target_reached": metrics[ranked[0]]["accuracy"] >= 0.90,
            "model_and_strategy_selection_is_development_posthoc": True,
        },
    }
    probability_path = args.probabilities.resolve()
    probability_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        probability_path,
        **{name: probabilities[name] for name in sorted(probabilities)},
    )
    output["artifacts"] = {
        "probabilities": {
            "path": str(probability_path),
            "sha256": base.file_sha256(probability_path),
            "contains_labels": False,
            "contains_record_identifiers": False,
        }
    }
    base.atomic_json(args.output.resolve(), output)
    print(json.dumps(output["best_full_coverage"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
