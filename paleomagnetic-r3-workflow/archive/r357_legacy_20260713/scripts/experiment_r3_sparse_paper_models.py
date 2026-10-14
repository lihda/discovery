#!/usr/bin/env python3
"""Grouped PDF-only R3 paper classifiers using sparse scientific text.

This is a development experiment, not a strict runtime scorer.  PDF membership
and the fixed training partition are established before R3 is dereferenced.
Every outer fold holds out complete REFNO groups.  Runtime-recomputable model
inputs are limited to original-paper text and frozen aggregate terminology.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_OUTPUT = ROOT / "tmp/r3_sparse_paper_models/aggregate.json"
DEFAULT_PROBABILITIES = ROOT / "tmp/r3_sparse_paper_models/oof_probabilities.npz"
REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:references|bibliography|literature\s+cited|works\s+cited)\s*$"
)
SPACE_RE = re.compile(r"\s+")


def trim_reference_section(text: str, *, cap: int = 180_000) -> str:
    """Remove a terminal bibliography without treating an early TOC as a cut."""

    minimum = int(len(text) * 0.35)
    starts = [match.start() for match in REFERENCE_HEADING_RE.finditer(text)]
    eligible = [start for start in starts if start >= minimum]
    body = text[: min(eligible)] if eligible else text
    if len(body) <= cap:
        return body
    # Preserve both the front matter/methods and the later results/discussion.
    front = int(cap * 0.68)
    return body[:front] + "\n[[BODY_TAIL]]\n" + body[-(cap - front) :]


def paper_groups(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
) -> tuple[dict[str, list[int]], dict[str, int]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        members[record["REFNO"]].append(index)
    majority = {
        refno: int(float(labels[indices].mean()) >= 0.5)
        for refno, indices in members.items()
    }
    return dict(members), majority


def document_views(
    records: Sequence[dict[str, str]],
    paper_text: Mapping[str, str],
    article_documents: Mapping[str, str],
) -> dict[str, dict[str, str]]:
    target_names: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["ROCKNAME"] not in target_names[record["REFNO"]]:
            target_names[record["REFNO"]].append(record["ROCKNAME"])

    views: dict[str, dict[str, str]] = defaultdict(dict)
    for refno, text in paper_text.items():
        body = trim_reference_section(text)
        knowledge = article_documents[refno]
        targets = " ; ".join(target_names[refno])
        views["knowledge"][refno] = f"TARGETS {targets}\n{knowledge}"
        views["body"][refno] = f"TARGETS {targets}\n{body}"
        views["front_knowledge"][refno] = (
            f"TARGETS {targets}\n{body[:14_000]}\n[[KNOWLEDGE]]\n{knowledge}"
        )
    return dict(views)


def model_probability(model: Any, matrix: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return expit(np.asarray(model.decision_function(matrix), dtype=np.float64))


def classifiers() -> list[tuple[str, Any]]:
    result: list[tuple[str, Any]] = []
    for c_value in (0.25, 1.0, 4.0):
        result.append(
            (
                f"logreg_c{c_value:g}",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    max_iter=1600,
                    random_state=base.RANDOM_STATE,
                ),
            )
        )
    for c_value in (0.1, 0.5, 2.0):
        result.append(
            (
                f"linear_svc_c{c_value:g}",
                LinearSVC(C=c_value, max_iter=6000, random_state=base.RANDOM_STATE),
            )
        )
    for alpha in (0.2, 1.0):
        result.append((f"complement_nb_a{alpha:g}", ComplementNB(alpha=alpha)))
    return result


def vectorizer_specs() -> dict[str, dict[str, Any]]:
    return {
        "knowledge_word": {
            "view": "knowledge",
            "analyzer": "word",
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.997,
            "max_features": 130_000,
            "strip_accents": "unicode",
        },
        "knowledge_char": {
            "view": "knowledge",
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "min_df": 2,
            "max_df": 1.0,
            "max_features": 180_000,
        },
        "body_word": {
            "view": "body",
            "analyzer": "word",
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.997,
            "max_features": 160_000,
            "strip_accents": "unicode",
        },
        "body_char": {
            "view": "body",
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "min_df": 3,
            "max_df": 1.0,
            "max_features": 200_000,
        },
        "front_knowledge_word": {
            "view": "front_knowledge",
            "analyzer": "word",
            "ngram_range": (1, 3),
            "min_df": 2,
            "max_df": 0.997,
            "max_features": 150_000,
            "strip_accents": "unicode",
        },
    }


def run_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    views: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    members, majority = paper_groups(records, labels)
    model_names = [name for name, _ in classifiers()]
    probability = {
        f"{vector_name}__{model_name}": np.full(len(records), np.nan)
        for vector_name in vectorizer_specs()
        for model_name in model_names
    }
    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    splitter = GroupKFold(n_splits=base.N_SPLITS)
    for fold, (train_index, test_index) in enumerate(
        splitter.split(np.zeros(len(records)), labels, groups), start=1
    ):
        train_refnos = sorted(set(str(groups[index]) for index in train_index))
        test_refnos = sorted(set(str(groups[index]) for index in test_index))
        if set(train_refnos) & set(test_refnos):
            raise AssertionError("REFNO leaked across an outer fold")
        train_y = np.asarray([majority[refno] for refno in train_refnos], dtype=np.int8)

        for vector_name, spec in vectorizer_specs().items():
            view = str(spec["view"])
            vectorizer = TfidfVectorizer(
                analyzer=spec["analyzer"],
                ngram_range=spec["ngram_range"],
                min_df=spec["min_df"],
                max_df=spec["max_df"],
                max_features=spec["max_features"],
                strip_accents=spec.get("strip_accents"),
                sublinear_tf=True,
                dtype=np.float32,
            )
            train_matrix = vectorizer.fit_transform(
                [views[view][refno] for refno in train_refnos]
            )
            test_matrix = vectorizer.transform(
                [views[view][refno] for refno in test_refnos]
            )
            for model_name, model in classifiers():
                model.fit(train_matrix, train_y)
                paper_probability = dict(
                    zip(test_refnos, model_probability(model, test_matrix))
                )
                row_probability = np.asarray(
                    [paper_probability[str(groups[index])] for index in test_index],
                    dtype=np.float64,
                )
                key = f"{vector_name}__{model_name}"
                probability[key][test_index] = row_probability
                fold_metrics[key].append(
                    {"fold": fold, **base.metric_record(labels[test_index], row_probability)}
                )
        print(
            f"fold {fold}/{base.N_SPLITS}: train_papers={len(train_refnos)}, "
            f"test_papers={len(test_refnos)}",
            flush=True,
        )

    if any(not np.isfinite(values).all() for values in probability.values()):
        raise AssertionError("one or more sparse paper OOF predictions are incomplete")
    return probability, dict(fold_metrics)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=base.DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--without-positive-kb", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    args = parser.parse_args(argv)

    base.require_experiment_dependencies()
    records, coverage = base.load_pdf_only_identity(
        args.identity.resolve(), args.pdf_dir.resolve()
    )
    labels_by_resultno = base.load_training_labels(
        args.targets.resolve(), {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    refnos = {record["REFNO"] for record in records}
    paper_text = base.load_page_text(args.page_cache.resolve(), refnos)
    positive_path = None if args.without_positive_kb else args.positive_kb.resolve()
    concepts, knowledge_audit = base.load_knowledge_patterns(
        args.taxonomy.resolve(), positive_path
    )
    base_features = base.build_features(records, paper_text, concepts)
    views = document_views(records, paper_text, base_features["article_documents"])
    probabilities, folds = run_oof(records, labels, views)

    aggregate = {
        name: base.metric_record(labels, values)
        for name, values in probabilities.items()
    }
    ensembles = {
        "all_model_mean": np.mean(list(probabilities.values()), axis=0),
        "all_linear_svc_mean": np.mean(
            [value for name, value in probabilities.items() if "linear_svc" in name],
            axis=0,
        ),
        "knowledge_model_mean": np.mean(
            [value for name, value in probabilities.items() if name.startswith("knowledge_")],
            axis=0,
        ),
        "body_model_mean": np.mean(
            [value for name, value in probabilities.items() if name.startswith("body_")],
            axis=0,
        ),
    }
    for name, values in ensembles.items():
        probabilities[name] = values
        aggregate[name] = base.metric_record(labels, values)

    ranking = sorted(
        aggregate,
        key=lambda name: (
            aggregate[name]["accuracy"],
            aggregate[name]["macro_f1"],
            aggregate[name]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best_name = ranking[0]
    members, _ = paper_groups(records, labels)
    result = {
        "schema_version": "r3_sparse_paper_models_oof_v1",
        "status": "complete",
        "scope": "development-only PDF-backed fixed training partition",
        "guardrails": {
            "cross_validation": "5-fold GroupKFold by REFNO",
            "labels_opened_after_training_and_local-PDF membership gates": True,
            "held_out_labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifier_features": False,
        },
        "data": {
            **coverage,
            "paper_examples": len(members),
            "row_labels": dict(sorted(Counter(int(value) for value in labels).items())),
        },
        "knowledge": knowledge_audit,
        "inputs": {
            "identity": base.file_sha256(args.identity.resolve()),
            "targets": base.file_sha256(args.targets.resolve()),
            "page_cache": base.file_sha256(args.page_cache.resolve()),
            "taxonomy": base.file_sha256(args.taxonomy.resolve()),
            **(
                {"positive_kb": base.file_sha256(positive_path)}
                if positive_path is not None
                else {}
            ),
        },
        "models": {
            name: {"aggregate": aggregate[name], "folds": folds.get(name, [])}
            for name in aggregate
        },
        "ranking": ranking,
        "best_full_coverage": {"model": best_name, **aggregate[best_name]},
        "interpretation": {
            "target_accuracy": 0.9,
            "target_reached": aggregate[best_name]["accuracy"] >= 0.9,
            "ranking_is_posthoc_development_analysis": True,
            "external_refno_validation_required": True,
        },
    }
    base.atomic_json(args.output.resolve(), result)
    args.probabilities.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.probabilities.resolve(),
        **{name: probabilities[name] for name in sorted(probabilities)},
    )
    print(json.dumps(result["best_full_coverage"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
