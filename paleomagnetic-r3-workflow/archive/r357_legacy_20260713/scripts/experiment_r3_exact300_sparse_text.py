#!/usr/bin/env python3
"""Leakage-safe sparse-text R3 models for the frozen exact300 screen.

For each exact300 outer fold, every row from a test REFNO is removed from the
2,172-row PDF-backed auxiliary training pool.  Models see only ROCKNAME and
text reconstructed from the frozen original-paper page cache and aggregate
knowledge patterns.  REFNO and RESULTNO are alignment keys, never features.

The feature views and candidate family are frozen and hashed before R3 labels
are opened.  Labels from an outer-test paper are used only for evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


FOLD_MANIFEST = ROOT / "tmp/r3_exact_300_resultno_v2/refno_fold_manifest.json"
FOLD_MANIFEST_SHA256 = (
    "8f6ab8f1a071878e14d4db8e19a7ec0519ed612be107b0f0fdf0489dd59aaf2d"
)
FOLD_ID_SHA256 = (
    "28552aa3627c2c808ff49ac50c7c9d67a6f32daaad0e9aff5ae6a96fbe61f08f"
)
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/sparse_text_v3_passage"
FREEZE_NAME = "feature_freeze.json"
REPORT_NAME = "report.json"
OOF_NAME = "oof_probabilities.npz"
EXPECTED_ROWS = auxiliary.EXPECTED_EXACT_ROWS
TARGET_ACCURACY = auxiliary.TARGET_ACCURACY

REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:references|bibliography|literature\s+cited|works\s+cited)\s*$"
)
SPACE_RE = re.compile(r"\s+")
FORBIDDEN_ARTIFACT_NAME = re.compile(
    r"(?i)(?:^|_)(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)(?:$|_)"
)


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    level: str
    view: str
    representation: str
    analyzer: str
    ngram_min: int
    ngram_max: int
    min_df: int
    max_features: int
    c_value: float
    pure_training_papers_only: bool = False
    top_k_passages: int = 0


CANDIDATES: tuple[CandidateSpec, ...] = (
    CandidateSpec(
        name="paper_evidence_word_nbsvm",
        level="paper",
        view="paper_evidence",
        representation="binary_count_nbsvm",
        analyzer="word",
        ngram_min=1,
        ngram_max=3,
        min_df=2,
        max_features=180_000,
        c_value=2.0,
    ),
    CandidateSpec(
        name="paper_evidence_word_nbsvm_pure",
        level="paper",
        view="paper_evidence",
        representation="binary_count_nbsvm",
        analyzer="word",
        ngram_min=1,
        ngram_max=3,
        min_df=2,
        max_features=180_000,
        c_value=2.0,
        pure_training_papers_only=True,
    ),
    CandidateSpec(
        name="paper_body_word_nbsvm_pure",
        level="paper",
        view="paper_body",
        representation="binary_count_nbsvm",
        analyzer="word",
        ngram_min=1,
        ngram_max=2,
        min_df=2,
        max_features=220_000,
        c_value=2.0,
        pure_training_papers_only=True,
    ),
    CandidateSpec(
        name="paper_evidence_char_calibrated_svc",
        level="paper",
        view="paper_evidence",
        representation="tfidf_calibrated_linear_svc",
        analyzer="char_wb",
        ngram_min=3,
        ngram_max=5,
        min_df=2,
        max_features=180_000,
        c_value=1.0,
    ),
    CandidateSpec(
        name="target_local_word_nbsvm",
        level="row",
        view="target_local",
        representation="binary_count_nbsvm",
        analyzer="word",
        ngram_min=1,
        ngram_max=3,
        min_df=2,
        max_features=180_000,
        c_value=2.0,
    ),
    CandidateSpec(
        name="paper_passage_word_nbsvm_top3",
        level="passage",
        view="paper_passages",
        representation="binary_count_nbsvm",
        analyzer="word",
        ngram_min=1,
        ngram_max=3,
        min_df=3,
        max_features=220_000,
        c_value=2.0,
        top_k_passages=3,
    ),
)

FIXED_FUSIONS: Mapping[str, tuple[tuple[str, float], ...]] = {
    "paper_sparse_ensemble": (
        ("paper_evidence_word_nbsvm", 1.0 / 3.0),
        ("paper_body_word_nbsvm_pure", 1.0 / 3.0),
        ("paper_evidence_char_calibrated_svc", 1.0 / 3.0),
    ),
    "paper_evidence_ensemble": (
        ("paper_evidence_word_nbsvm", 0.5),
        ("paper_evidence_char_calibrated_svc", 0.5),
    ),
    "paper_sparse__target_local_25": (
        ("paper_sparse_ensemble", 0.75),
        ("target_local_word_nbsvm", 0.25),
    ),
    "paper_evidence__passage_top3": (
        ("paper_evidence_word_nbsvm", 0.5),
        ("paper_passage_word_nbsvm_top3", 0.5),
    ),
    "paper_sparse__passage_top3_25": (
        ("paper_sparse_ensemble", 0.75),
        ("paper_passage_word_nbsvm_top3", 0.25),
    ),
}


def trim_reference_section(text: str, *, cap: int = 180_000) -> str:
    """Drop a terminal bibliography and cap text while retaining body tail."""

    minimum = int(len(text) * 0.35)
    eligible = [
        match.start()
        for match in REFERENCE_HEADING_RE.finditer(text)
        if match.start() >= minimum
    ]
    body = text[: min(eligible)] if eligible else text
    if len(body) <= cap:
        return body
    front = int(cap * 0.68)
    return body[:front] + "\n[[BODY TAIL]]\n" + body[-(cap - front) :]


def load_canonical_folds(
    path: Path,
    exact_records: Sequence[Mapping[str, str]],
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    observed_hash = base.file_sha256(path)
    if observed_hash != FOLD_MANIFEST_SHA256:
        raise ValueError("canonical fold manifest hash mismatch")
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if (
        artifact.get("schema_version")
        != "r3_exact300_label_free_refno_folds_v1"
        or artifact.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or artifact.get("fold_id_sha256") != FOLD_ID_SHA256
        or artifact.get("input", {}).get("identity_sha256")
        != auxiliary.EXACT_IDENTITY_SHA256
    ):
        raise ValueError("canonical fold manifest metadata is invalid")
    rows = artifact.get("rows")
    if not isinstance(rows, list) or len(rows) != len(exact_records):
        raise ValueError("canonical fold manifest row count mismatch")
    fold_ids = np.zeros(len(exact_records), dtype=np.int8)
    for index, (row, record) in enumerate(zip(rows, exact_records)):
        expected = (index, record["REFNO"], record["RESULTNO"])
        observed = (
            row.get("row_index"),
            str(row.get("REFNO", "")),
            str(row.get("RESULTNO", "")),
        )
        if observed != expected:
            raise ValueError(f"canonical fold row alignment mismatch at index {index}")
        fold_ids[index] = int(row.get("fold", 0))
    observed_fold_hash = hashlib.sha256(
        bytes(int(value) for value in fold_ids)
    ).hexdigest()
    if observed_fold_hash != FOLD_ID_SHA256:
        raise ValueError("canonical fold assignment digest mismatch")
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    refno_folds: dict[str, set[int]] = defaultdict(set)
    for record, fold in zip(exact_records, fold_ids):
        refno_folds[record["REFNO"]].add(int(fold))
    if any(len(values) != 1 for values in refno_folds.values()):
        raise ValueError("canonical fold manifest splits one REFNO across folds")
    for fold in range(1, auxiliary.N_SPLITS + 1):
        test = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        if len(test) != 60:
            raise ValueError("canonical exact300 fold does not contain 60 rows")
        folds.append((train, test))
    covered = np.concatenate([test for _, test in folds])
    if sorted(int(index) for index in covered) != list(range(len(exact_records))):
        raise ValueError("canonical folds do not cover exact300 exactly once")
    return folds, {
        "path": str(path),
        "sha256": observed_hash,
        "fold_id_sha256": observed_fold_hash,
        "rows": len(exact_records),
        "papers": len(refno_folds),
        "rows_per_fold": [int(len(test)) for _, test in folds],
        "source_sklearn_version": artifact.get("generator", {}).get(
            "sklearn_version"
        ),
        "loaded_independently_of_local_sklearn_group_tie_behavior": True,
    }


def build_document_views(
    records: Sequence[dict[str, str]],
    paper_text: Mapping[str, str],
    concepts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build every unlabeled view before the label workbook is dereferenced."""

    trimmed = {
        refno: trim_reference_section(text)
        for refno, text in paper_text.items()
    }
    features = base.build_features(records, trimmed, concepts)
    paper_body = {
        refno: SPACE_RE.sub(" ", text).strip()
        for refno, text in trimmed.items()
    }
    paper_evidence = {
        refno: features["article_documents"][refno]
        for refno in sorted(trimmed)
    }
    paper_passages = {
        refno: evidence_passages(paper_evidence[refno])
        for refno in sorted(trimmed)
    }
    views = {
        "paper_body": paper_body,
        "paper_evidence": paper_evidence,
        "paper_passages": paper_passages,
        "target_local": list(features["local_documents"]),
    }
    audit = {
        "paper_count": len(trimmed),
        "row_count": len(records),
        "paper_body_sha256": auxiliary.sha256_strings(
            [paper_body[refno] for refno in sorted(paper_body)]
        ),
        "paper_evidence_sha256": auxiliary.sha256_strings(
            [paper_evidence[refno] for refno in sorted(paper_evidence)]
        ),
        "paper_passages_sha256": auxiliary.sha256_strings(
            [
                "\n[[PASSAGE BOUNDARY]]\n".join(paper_passages[refno])
                for refno in sorted(paper_passages)
            ]
        ),
        "target_local_sha256": auxiliary.sha256_strings(views["target_local"]),
        "paper_body_empty": sum(not value for value in paper_body.values()),
        "paper_evidence_empty": sum(not value for value in paper_evidence.values()),
        "paper_passages_empty": sum(not value for value in paper_passages.values()),
        "target_local_empty": sum(not value for value in views["target_local"]),
        "passage_count": sum(len(value) for value in paper_passages.values()),
        "passage_count_min": min(len(value) for value in paper_passages.values()),
        "passage_count_max": max(len(value) for value in paper_passages.values()),
        "reference_trimming": "terminal heading after first 35% of article",
    }
    if any(
        audit[name] != 0
        for name in (
            "paper_body_empty",
            "paper_evidence_empty",
            "paper_passages_empty",
            "target_local_empty",
        )
    ):
        raise ValueError("sparse-text document views contain empty documents")
    return views, audit


def evidence_passages(
    evidence_text: str,
    *,
    chunk_size: int = 2_400,
    overlap: int = 300,
) -> list[str]:
    passages: list[str] = []
    step = chunk_size - overlap
    for raw_piece in evidence_text.splitlines():
        normalized = SPACE_RE.sub(" ", raw_piece).strip()
        if not normalized:
            continue
        for offset in range(0, len(normalized), step):
            chunk = normalized[offset : offset + chunk_size].strip()
            if chunk:
                passages.append(chunk)
            if offset + chunk_size >= len(normalized):
                break
    if not passages:
        fallback = SPACE_RE.sub(" ", evidence_text[:2_600]).strip()
        passages.append(fallback)
    return passages[:160]


def paper_training_examples(
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    train_indices: np.ndarray,
    *,
    pure_only: bool,
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index in train_indices:
        members[records[int(index)]["REFNO"]].append(int(index))
    ordered: list[str] = []
    paper_labels: list[int] = []
    mixed = 0
    for refno in sorted(members):
        values = labels[members[refno]]
        is_mixed = len(set(int(value) for value in values)) > 1
        mixed += int(is_mixed)
        if pure_only and is_mixed:
            continue
        ordered.append(refno)
        paper_labels.append(int(float(values.mean()) >= 0.5))
    output = np.asarray(paper_labels, dtype=np.int8)
    if len(set(int(value) for value in output)) != 2:
        raise ValueError("paper training examples do not contain both classes")
    return ordered, output, {
        "available_training_papers": len(members),
        "mixed_training_papers": mixed,
        "retained_training_papers": len(ordered),
        "pure_only": pure_only,
    }


def paper_balanced_row_weights(
    records: Sequence[Mapping[str, str]], train_indices: np.ndarray
) -> np.ndarray:
    counts = Counter(records[int(index)]["REFNO"] for index in train_indices)
    weights = np.asarray(
        [1.0 / counts[records[int(index)]["REFNO"]] for index in train_indices],
        dtype=np.float64,
    )
    return weights * (len(weights) / weights.sum())


def flatten_paper_passages(
    refnos: Sequence[str],
    passages: Mapping[str, Sequence[str]],
    labels: np.ndarray | None = None,
) -> tuple[list[str], np.ndarray | None, np.ndarray, list[slice]]:
    documents: list[str] = []
    repeated_labels: list[int] = []
    weights: list[float] = []
    bounds: list[slice] = []
    for position, refno in enumerate(refnos):
        items = list(passages[refno])
        if not items:
            raise ValueError("paper has no evidence passages")
        start = len(documents)
        documents.extend(items)
        if labels is not None:
            repeated_labels.extend([int(labels[position])] * len(items))
        weights.extend([1.0 / len(items)] * len(items))
        bounds.append(slice(start, len(documents)))
    sample_weight = np.asarray(weights, dtype=np.float64)
    sample_weight *= len(sample_weight) / sample_weight.sum()
    output_labels = (
        np.asarray(repeated_labels, dtype=np.int8) if labels is not None else None
    )
    return documents, output_labels, sample_weight, bounds


def aggregate_top_k_passages(
    probabilities: np.ndarray,
    bounds: Sequence[slice],
    top_k: int,
) -> np.ndarray:
    output: list[float] = []
    for bound in bounds:
        values = np.sort(np.asarray(probabilities[bound], dtype=np.float64))
        retained = values[-min(top_k, len(values)) :]
        output.append(float(retained.mean()))
    return np.asarray(output, dtype=np.float64)


def nb_log_count_ratio(
    matrix: Any,
    labels: np.ndarray,
    sample_weight: np.ndarray | None = None,
) -> np.ndarray:
    if sample_weight is None:
        sample_weight = np.ones(len(labels), dtype=np.float64)
    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    ratios: list[np.ndarray] = []
    for label in (1, 0):
        selected = labels == label
        weighted = matrix[selected].multiply(sample_weight[selected, None])
        numerator = 1.0 + np.asarray(weighted.sum(axis=0)).ravel()
        denominator = 1.0 + float(sample_weight[selected].sum())
        ratios.append(numerator / denominator)
    return np.log(ratios[0] / ratios[1])


def fit_nbsvm(
    train_documents: Sequence[str],
    test_documents: Sequence[str],
    labels: np.ndarray,
    spec: CandidateSpec,
    *,
    sample_weight: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    vectorizer = CountVectorizer(
        analyzer=spec.analyzer,
        ngram_range=(spec.ngram_min, spec.ngram_max),
        min_df=spec.min_df,
        max_df=0.997,
        max_features=spec.max_features,
        binary=True,
        strip_accents="unicode" if spec.analyzer == "word" else None,
        dtype=np.float64,
    )
    train_matrix = vectorizer.fit_transform(train_documents)
    test_matrix = vectorizer.transform(test_documents)
    ratio = nb_log_count_ratio(train_matrix, labels, sample_weight)
    model = LogisticRegression(
        C=spec.c_value,
        solver="liblinear",
        max_iter=2_000,
        random_state=base.RANDOM_STATE,
    )
    model.fit(train_matrix.multiply(ratio), labels, sample_weight=sample_weight)
    probability = model.predict_proba(test_matrix.multiply(ratio))[:, 1]
    return np.asarray(probability, dtype=np.float64), {
        "vocabulary_size": int(train_matrix.shape[1]),
        "train_documents": int(train_matrix.shape[0]),
    }


def fit_calibrated_svc(
    train_documents: Sequence[str],
    test_documents: Sequence[str],
    labels: np.ndarray,
    spec: CandidateSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    vectorizer = TfidfVectorizer(
        analyzer=spec.analyzer,
        ngram_range=(spec.ngram_min, spec.ngram_max),
        min_df=spec.min_df,
        max_df=1.0,
        max_features=spec.max_features,
        sublinear_tf=True,
        dtype=np.float64,
    )
    train_matrix = vectorizer.fit_transform(train_documents)
    test_matrix = vectorizer.transform(test_documents)
    calibration_cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=base.RANDOM_STATE
    )
    model = CalibratedClassifierCV(
        estimator=LinearSVC(
            C=spec.c_value,
            max_iter=8_000,
            random_state=base.RANDOM_STATE,
        ),
        method="sigmoid",
        cv=calibration_cv,
        n_jobs=1,
    )
    model.fit(train_matrix, labels)
    probability = model.predict_proba(test_matrix)[:, 1]
    return np.asarray(probability, dtype=np.float64), {
        "vocabulary_size": int(train_matrix.shape[1]),
        "train_documents": int(train_matrix.shape[0]),
        "calibration": "3-fold stratified Platt scaling over unique training papers",
    }


def add_fixed_fusions(probabilities: dict[str, np.ndarray]) -> None:
    for name, components in FIXED_FUSIONS.items():
        values = np.zeros_like(next(iter(probabilities.values())))
        for component, weight in components:
            values += probabilities[component] * weight
        probabilities[name] = values


def run_outer_oof(
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    views: Mapping[str, Any],
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[dict[str, Any]]]:
    full_lookup = {
        auxiliary.row_key(record): index
        for index, record in enumerate(full_records)
    }
    probabilities = {
        spec.name: np.full(len(exact_records), np.nan, dtype=np.float64)
        for spec in CANDIDATES
    }
    fold_indices: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    assignment_count = np.zeros(len(exact_records), dtype=np.int8)
    for fold, (_, exact_test) in enumerate(folds, start=1):
        train_full, test_full, test_refnos = auxiliary.auxiliary_split(
            full_records, exact_records, exact_test, full_lookup
        )
        fold_indices.append(exact_test)
        assignment_count[exact_test] += 1
        training_refnos = {full_records[int(index)]["REFNO"] for index in train_full}
        test_refno_order = [exact_records[int(index)]["REFNO"] for index in exact_test]
        candidate_audit: dict[str, Any] = {}
        for spec in CANDIDATES:
            if spec.level == "paper":
                paper_refnos, labels, membership = paper_training_examples(
                    full_records,
                    full_labels,
                    train_full,
                    pure_only=spec.pure_training_papers_only,
                )
                if set(paper_refnos) & test_refnos:
                    raise AssertionError("outer-test REFNO entered a paper classifier")
                train_documents = [views[spec.view][refno] for refno in paper_refnos]
                test_documents = [views[spec.view][refno] for refno in test_refno_order]
                if spec.representation == "binary_count_nbsvm":
                    predicted, model_audit = fit_nbsvm(
                        train_documents, test_documents, labels, spec
                    )
                else:
                    predicted, model_audit = fit_calibrated_svc(
                        train_documents, test_documents, labels, spec
                    )
                candidate_audit[spec.name] = {**membership, **model_audit}
            elif spec.level == "row":
                train_documents = [views[spec.view][int(index)] for index in train_full]
                test_documents = [views[spec.view][int(index)] for index in test_full]
                weights = paper_balanced_row_weights(full_records, train_full)
                predicted, model_audit = fit_nbsvm(
                    train_documents,
                    test_documents,
                    full_labels[train_full],
                    spec,
                    sample_weight=weights,
                )
                candidate_audit[spec.name] = {
                    **model_audit,
                    "paper_balanced_sample_weight": True,
                }
            elif spec.level == "passage":
                paper_refnos, labels, membership = paper_training_examples(
                    full_records,
                    full_labels,
                    train_full,
                    pure_only=spec.pure_training_papers_only,
                )
                if set(paper_refnos) & test_refnos:
                    raise AssertionError("outer-test REFNO entered a passage classifier")
                train_documents, passage_labels, weights, _ = flatten_paper_passages(
                    paper_refnos, views[spec.view], labels
                )
                test_refnos_ordered = list(test_refno_order)
                test_documents, _, _, test_bounds = flatten_paper_passages(
                    test_refnos_ordered, views[spec.view]
                )
                if passage_labels is None:
                    raise AssertionError("passage labels were not constructed")
                passage_probability, model_audit = fit_nbsvm(
                    train_documents,
                    test_documents,
                    passage_labels,
                    spec,
                    sample_weight=weights,
                )
                predicted = aggregate_top_k_passages(
                    passage_probability, test_bounds, spec.top_k_passages
                )
                candidate_audit[spec.name] = {
                    **membership,
                    **model_audit,
                    "paper_balanced_passage_weight": True,
                    "test_passages": len(test_documents),
                    "aggregation": f"mean of top {spec.top_k_passages} passages",
                }
            else:
                raise AssertionError(f"unknown candidate level: {spec.level}")
            probabilities[spec.name][exact_test] = predicted

        excluded_rows = sum(record["REFNO"] in test_refnos for record in full_records)
        audits.append(
            {
                "fold": fold,
                "exact_test_rows": int(len(exact_test)),
                "exact_test_papers": len(test_refnos),
                "auxiliary_train_rows": int(len(train_full)),
                "auxiliary_train_papers": len(training_refnos),
                "full_rows_excluded_for_test_refnos": int(excluded_rows),
                "train_test_refno_overlap": len(training_refnos & test_refnos),
                "all_test_paper_rows_excluded": all(
                    record["REFNO"] not in test_refnos
                    for record in (full_records[int(index)] for index in train_full)
                ),
                "test_refno_set_sha256": auxiliary.sha256_strings(sorted(test_refnos)),
                "candidate_fit_audit": candidate_audit,
            }
        )
        print(
            f"fold {fold}/5: train_rows={len(train_full)}, "
            f"test_rows={len(exact_test)}, test_papers={len(test_refnos)}",
            flush=True,
        )
    if not np.array_equal(assignment_count, np.ones(len(exact_records), dtype=np.int8)):
        raise AssertionError("exact300 rows were not assigned to exactly one test fold")
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("sparse OOF prediction coverage is incomplete")
    add_fixed_fusions(probabilities)
    return probabilities, fold_indices, audits


def write_probability_artifact(
    path: Path, probabilities: Mapping[str, np.ndarray]
) -> None:
    if any(FORBIDDEN_ARTIFACT_NAME.search(name) for name in probabilities):
        raise ValueError("OOF artifact key contains a label or record identifier")
    arrays = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in sorted(probabilities.items())
    }
    if any(values.shape != (EXPECTED_ROWS,) for values in arrays.values()):
        raise ValueError("OOF artifact must contain only exact300 probability arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument(
        "--screening-protocol", type=Path, default=auxiliary.SCREENING_PROTOCOL
    )
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    base.require_experiment_dependencies()

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    protocol_path = args.screening_protocol.resolve()
    fold_manifest_path = args.fold_manifest.resolve()
    targets_path = args.targets.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy_path = args.taxonomy.resolve()
    positive_kb_path = args.positive_kb.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_audit = auxiliary.validate_inputs(
        exact_identity,
        full_identity,
        protocol_path,
        fold_manifest_path,
        targets_path,
        page_cache_path,
        taxonomy_path,
        positive_kb_path,
    )
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    folds, fold_manifest_audit = load_canonical_folds(
        fold_manifest_path, exact_records
    )
    refnos = {record["REFNO"] for record in full_records}
    paper_text = base.load_page_text(page_cache_path, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )
    views, document_audit = build_document_views(full_records, paper_text, concepts)

    freeze_path = output_dir / FREEZE_NAME
    freeze = {
        "schema_version": "r3_exact300_sparse_text_feature_freeze_v3",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": {
            "evaluation_rows": len(exact_records),
            "evaluation_papers": len({record["REFNO"] for record in exact_records}),
            "auxiliary_rows": len(full_records),
            "auxiliary_papers": len(refnos),
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": auxiliary.sha256_records(full_records),
        },
        "candidate_specs": [asdict(spec) for spec in CANDIDATES],
        "fixed_fusions": {
            name: [[component, weight] for component, weight in components]
            for name, components in FIXED_FUSIONS.items()
        },
        "document_views": document_audit,
        "runtime_boundary": {
            "labels_read": False,
            "source_field_cells_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifiers_used_as_predictive_features": False,
        },
        "allowed_predictive_sources": [
            "ROCKNAME from frozen identity",
            "original-paper text from frozen page cache",
            "frozen aggregate terminology patterns",
        ],
        "inputs": input_audit,
        "canonical_fold_manifest": fold_manifest_audit,
        "knowledge": knowledge_audit,
    }
    base.atomic_json(freeze_path, freeze)
    freeze_sha256 = base.file_sha256(freeze_path)

    # Only now may R3 be read, and outer-test paper labels never enter fitting.
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
    probabilities, fold_indices, fold_audit = run_outer_oof(
        full_records, exact_records, full_labels, views, folds
    )
    metrics = {
        name: base.metric_record(exact_labels, values)
        for name, values in probabilities.items()
    }
    fold_metrics = {
        name: [
            {
                "fold": fold,
                **base.metric_record(exact_labels[index], values[index]),
            }
            for fold, index in enumerate(fold_indices, start=1)
        ]
        for name, values in probabilities.items()
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
    best = ranking[0]
    oof_path = output_dir / OOF_NAME
    write_probability_artifact(oof_path, probabilities)
    report = {
        "schema_version": "r3_exact300_sparse_text_grouped_oof_v3",
        "status": "complete",
        "scope": {
            "evaluated_rows": len(exact_records),
            "evaluated_papers": len({record["REFNO"] for record in exact_records}),
            "folds": len(fold_indices),
            "coverage": 1.0,
            "all_300_rows_predicted_once": True,
            "auxiliary_candidate_pool": full_coverage,
            "exact_membership": exact_coverage,
        },
        "guardrails": {
            "outer_split": "5-fold GroupKFold by exact300 REFNO",
            "canonical_fold_manifest": fold_manifest_audit,
            "per_fold_training_rule": (
                "all PDF-backed fixed-training rows except all rows belonging to "
                "an exact300 outer-test REFNO"
            ),
            "every_test_refno_fully_excluded_from_training": all(
                item["all_test_paper_rows_excluded"]
                and item["train_test_refno_overlap"] == 0
                for item in fold_audit
            ),
            "feature_freeze_before_label_access": True,
            "feature_freeze_sha256": freeze_sha256,
            "raw_source_fields_used_as_features": False,
            "prior_predictions_or_evaluations_used_as_features": False,
            "identifiers_used_as_predictive_features": False,
        },
        "data": {
            "exact_label_counts": dict(
                sorted(Counter(str(int(value)) for value in exact_labels).items())
            ),
            "auxiliary_label_counts": dict(
                sorted(Counter(str(int(value)) for value in full_labels).items())
            ),
            "document_views": document_audit,
        },
        "outer_fold_audit": fold_audit,
        "ranking": ranking,
        "best_full_coverage": {
            "strategy": best,
            **metrics[best],
            "folds": fold_metrics[best],
        },
        "models": {
            name: {"aggregate": metrics[name], "folds": fold_metrics[name]}
            for name in probabilities
        },
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "observed_accuracy": metrics[best]["accuracy"],
            "passed": metrics[best]["accuracy"] >= TARGET_ACCURACY,
            "decision": (
                "EXACT300_TARGET_REACHED"
                if metrics[best]["accuracy"] >= TARGET_ACCURACY
                else "EXACT300_TARGET_NOT_REACHED"
            ),
            "warning": (
                "Candidate ranking is development evidence on the frozen exact300 "
                "screen and is not independent external validation."
            ),
        },
        "artifacts": {
            "feature_freeze": {"path": str(freeze_path), "sha256": freeze_sha256},
            "oof_probabilities": {
                "path": str(oof_path),
                "sha256": base.file_sha256(oof_path),
                "rows_per_array": EXPECTED_ROWS,
                "contains_labels": False,
                "contains_identifiers": False,
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
    }
    report_path = output_dir / REPORT_NAME
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "best_full_coverage": report["best_full_coverage"],
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
