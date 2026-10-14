#!/usr/bin/env python3
"""REFNO-grouped R3 models over label-free original-paper embeddings."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_EMBEDDINGS = ROOT / "tmp/r3_paper_embeddings/minilm_l6_v2.npz"
DEFAULT_MANIFEST = ROOT / "tmp/r3_paper_embeddings/minilm_l6_v2.manifest.json"
DEFAULT_OUTPUT = ROOT / "tmp/r3_embedding_models/minilm_l6_v2_oof.json"
DEFAULT_PROBABILITIES = ROOT / "tmp/r3_embedding_models/minilm_l6_v2_oof.npz"


def probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return expit(np.asarray(model.decision_function(matrix), dtype=np.float64))


def model_factories(dimension: int) -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = []
    for c_value in (0.05, 0.25, 1.0, 4.0, 16.0):
        models.append(
            (
                f"logreg_c{c_value:g}",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=base.RANDOM_STATE,
                ),
            )
        )
    for c_value in (0.05, 0.25, 1.0, 4.0):
        models.append(
            (
                f"linear_svc_c{c_value:g}",
                LinearSVC(C=c_value, max_iter=8000, random_state=base.RANDOM_STATE),
            )
        )
    for neighbors in (7, 15, 31, 63):
        models.append(
            (
                f"cosine_knn_k{neighbors}",
                KNeighborsClassifier(
                    n_neighbors=neighbors, weights="distance", metric="cosine"
                ),
            )
        )
    if dimension <= 800:
        for c_value in (0.5, 2.0, 8.0):
            models.append(
                (
                    f"rbf_svc_c{c_value:g}",
                    SVC(C=c_value, kernel="rbf", gamma="scale", probability=False),
                )
            )
        models.extend(
            [
                (
                    "hist_gradient",
                    HistGradientBoostingClassifier(
                        max_iter=260,
                        learning_rate=0.035,
                        max_leaf_nodes=9,
                        min_samples_leaf=12,
                        l2_regularization=1.0,
                        random_state=base.RANDOM_STATE,
                    ),
                ),
                (
                    "extra_trees",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        max_features="sqrt",
                        min_samples_leaf=4,
                        n_jobs=1,
                        random_state=base.RANDOM_STATE,
                    ),
                ),
            ]
        )
    return models


def load_embedding_views(
    path: Path,
) -> tuple[list[str], dict[str, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        refnos = [str(value) for value in payload["refnos"].tolist()]
        front_mean = np.asarray(payload["front_mean"], dtype=np.float32)
        front_max = np.asarray(payload["front_max"], dtype=np.float32)
        knowledge_mean = np.asarray(payload["knowledge_mean"], dtype=np.float32)
        knowledge_max = np.asarray(payload["knowledge_max"], dtype=np.float32)
        combined = np.asarray(payload["combined"], dtype=np.float32)
    views = {
        "front_mean": front_mean,
        "front_mean_max": np.concatenate((front_mean, front_max), axis=1),
        "knowledge_mean": knowledge_mean,
        "knowledge_mean_max": np.concatenate(
            (knowledge_mean, knowledge_max), axis=1
        ),
        "combined": combined,
    }
    if any(matrix.shape[0] != len(refnos) for matrix in views.values()):
        raise ValueError("embedding view row counts do not match REFNO values")
    if len(refnos) != len(set(refnos)):
        raise ValueError("embedding cache contains duplicate REFNO values")
    return refnos, views


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=base.DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    parser.add_argument("--embedding-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    args = parser.parse_args(argv)

    identity_path = args.identity.resolve()
    labels_path = args.targets.resolve()
    embeddings_path = args.embeddings.resolve()
    manifest_path = args.embedding_manifest.resolve()
    records, coverage = base.load_pdf_only_identity(identity_path, args.pdf_dir.resolve())
    labels_by_resultno = base.load_training_labels(
        labels_path, {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    embedding_refnos, embedding_views = load_embedding_views(embeddings_path)
    required_refnos = {record["REFNO"] for record in records}
    if set(embedding_refnos) != required_refnos:
        raise ValueError("embedding cache REFNO set does not match PDF-only identities")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("runtime_boundary", {}).get("labels_read") is not False:
        raise ValueError("embedding manifest does not prove a label-free build")
    if manifest.get("output", {}).get("sha256") != base.file_sha256(embeddings_path):
        raise ValueError("embedding cache hash does not match its manifest")

    paper_position = {refno: index for index, refno in enumerate(embedding_refnos)}
    members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        members[record["REFNO"]].append(index)
    majority = {
        refno: int(float(labels[indices].mean()) >= 0.5)
        for refno, indices in members.items()
    }
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)

    names = [
        f"{view_name}__{model_name}"
        for view_name, matrix in embedding_views.items()
        for model_name, _ in model_factories(matrix.shape[1])
    ]
    oof = {name: np.full(len(records), np.nan, dtype=np.float64) for name in names}
    folds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    splitter = GroupKFold(n_splits=base.N_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(records)), labels, groups), start=1
    ):
        train_refnos = sorted(set(str(groups[index]) for index in train_index))
        test_refnos = sorted(set(str(groups[index]) for index in test_index))
        if set(train_refnos) & set(test_refnos):
            raise AssertionError("REFNO leaked across an embedding outer fold")
        train_y = np.asarray([majority[refno] for refno in train_refnos], dtype=np.int8)
        for view_name, matrix in embedding_views.items():
            train_matrix = matrix[[paper_position[refno] for refno in train_refnos]]
            test_matrix = matrix[[paper_position[refno] for refno in test_refnos]]
            for model_name, model in model_factories(matrix.shape[1]):
                model.fit(train_matrix, train_y)
                paper_probability = dict(zip(test_refnos, probability(model, test_matrix)))
                row_probability = np.asarray(
                    [paper_probability[str(groups[index])] for index in test_index],
                    dtype=np.float64,
                )
                name = f"{view_name}__{model_name}"
                oof[name][test_index] = row_probability
                folds[name].append(
                    {"fold": fold, **base.metric_record(labels[test_index], row_probability)}
                )
        print(
            f"fold {fold}/{base.N_SPLITS}: train_papers={len(train_refnos)}, "
            f"test_papers={len(test_refnos)}",
            flush=True,
        )

    if any(not np.isfinite(values).all() for values in oof.values()):
        raise AssertionError("embedding OOF probabilities are incomplete")
    ensembles = {
        "all_embedding_mean": np.mean(list(oof.values()), axis=0),
        "all_linear_mean": np.mean(
            [value for name, value in oof.items() if "logreg" in name or "linear_svc" in name],
            axis=0,
        ),
        "knowledge_linear_mean": np.mean(
            [
                value
                for name, value in oof.items()
                if name.startswith("knowledge_")
                and ("logreg" in name or "linear_svc" in name)
            ],
            axis=0,
        ),
    }
    oof.update(ensembles)
    metrics = {name: base.metric_record(labels, values) for name, values in oof.items()}
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
    result = {
        "schema_version": "r3_embedding_models_grouped_oof_v1",
        "status": "complete",
        "scope": "development-only PDF-backed fixed training partition",
        "guardrails": {
            "cross_validation": "5-fold GroupKFold by REFNO",
            "embedding_cache_label_free": True,
            "labels_opened_after_fixed-training and local-PDF membership": True,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifier_features": False,
        },
        "data": {
            **coverage,
            "paper_examples": len(members),
            "labels": dict(sorted(Counter(int(value) for value in labels).items())),
        },
        "inputs": {
            "identity_sha256": base.file_sha256(identity_path),
            "labels_sha256": base.file_sha256(labels_path),
            "embeddings_sha256": base.file_sha256(embeddings_path),
            "embedding_manifest_sha256": base.file_sha256(manifest_path),
        },
        "embedding_manifest": manifest,
        "models": {
            name: {"aggregate": metrics[name], "folds": folds.get(name, [])}
            for name in metrics
        },
        "ranking": ranking,
        "best_full_coverage": {"model": best_name, **metrics[best_name]},
        "interpretation": {
            "target_accuracy": 0.9,
            "target_reached": metrics[best_name]["accuracy"] >= 0.9,
            "model_ranking_is_posthoc": True,
            "external_refno_validation_required": True,
        },
    }
    base.atomic_json(args.output.resolve(), result)
    args.probabilities.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.probabilities.resolve(), **{name: oof[name] for name in sorted(oof)}
    )
    print(json.dumps(result["best_full_coverage"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
