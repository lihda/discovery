#!/usr/bin/env python3
"""Evaluate exact300 R3 targets with leakage-safe auxiliary training rows.

The evaluation membership remains the frozen exact300 identity.  For each
outer fold, fitting may use every row in the PDF-backed fixed training
partition except all rows belonging to an outer-test REFNO.  Predictions and
metrics are emitted only for the 300 frozen targets.

Feature construction is completed and hashed before R3 is opened.  Runtime
features are limited to ROCKNAME, original-paper text/layout, local PDF size,
and the two frozen aggregate knowledge artifacts.  Existing predictions and
evaluations are not inputs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402


EXACT_IDENTITY = ROOT / "tmp/r3_exact_300_resultno_v2/identity.xlsx"
FULL_IDENTITY = ROOT / "tmp/r3_distilled_training_preflight/identity.xlsx"
SCREENING_PROTOCOL = ROOT / "tmp/r3_exact_300_resultno_v2/screening_protocol.json"
FOLD_MANIFEST = ROOT / "tmp/r3_exact_300_resultno_v2/refno_fold_manifest.json"
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/auxiliary_training_v1"
FREEZE_OUTPUT = OUTPUT_DIR / "feature_freeze.json"
REPORT_OUTPUT = OUTPUT_DIR / "report.json"
OOF_OUTPUT = OUTPUT_DIR / "oof_probabilities.npz"

EXACT_IDENTITY_SHA256 = (
    "93ea5dfba0a32501a564d06eb020386fd86b11f59ded0982043f90c064ab960b"
)
FULL_IDENTITY_SHA256 = (
    "7e45020df48b866096ac4bf567102f6a73ce3c12759b19c8550f78de299c72f1"
)
SCREENING_PROTOCOL_SHA256 = (
    "6412244a5d268e6dcabfaed63f6f529d4976dd17ef6c4313f70bdb6748c04b4f"
)
FOLD_MANIFEST_SHA256 = (
    "8f6ab8f1a071878e14d4db8e19a7ec0519ed612be107b0f0fdf0489dd59aaf2d"
)
TARGETS_SHA256 = (
    "378c17338f3b5cfeff3ccd496894e0f6183f747119b2624c4a9cefaeda02501e"
)
PAGE_CACHE_SHA256 = (
    "f8ec25c1148c04112ccfc75076743803417effbb833834a136c714dff4e146d7"
)
TAXONOMY_SHA256 = (
    "149df539a937678c8204aa6d6e60e77bd901e4a77975d26caa5362749aa18ad4"
)
POSITIVE_KB_SHA256 = (
    "b0fc28866fe6702938a7cc02fcf7b87b68abc9e3b4e391a53a53de1615b5e9ef"
)

EXPECTED_EXACT_ROWS = 300
EXPECTED_EXACT_PAPERS = 258
EXPECTED_FULL_PDF_ROWS = 2172
EXPECTED_FULL_PDF_PAPERS = 1174
N_SPLITS = 5
TARGET_ACCURACY = 0.90
FORBIDDEN_ARTIFACT_NAME = re.compile(
    r"(?i)^(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)$"
)


def row_key(record: Mapping[str, str]) -> tuple[str, str, str]:
    return record["REFNO"], record["RESULTNO"], record["ROCKNAME"]


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def sha256_strings(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        payload = value.encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def sha256_records(records: Sequence[Mapping[str, str]]) -> str:
    return sha256_strings(["\0".join(row_key(record)) for record in records])


def validate_inputs(
    exact_identity: Path,
    full_identity: Path,
    protocol_path: Path,
    fold_manifest_path: Path,
    targets_path: Path,
    page_cache_path: Path,
    taxonomy_path: Path,
    positive_kb_path: Path,
) -> dict[str, Any]:
    expected = {
        "exact_identity": (exact_identity, EXACT_IDENTITY_SHA256),
        "full_identity": (full_identity, FULL_IDENTITY_SHA256),
        "screening_protocol": (protocol_path, SCREENING_PROTOCOL_SHA256),
        "fold_manifest": (fold_manifest_path, FOLD_MANIFEST_SHA256),
        "targets": (targets_path, TARGETS_SHA256),
        "page_cache": (page_cache_path, PAGE_CACHE_SHA256),
        "taxonomy": (taxonomy_path, TAXONOMY_SHA256),
        "positive_kb": (positive_kb_path, POSITIVE_KB_SHA256),
    }
    audit: dict[str, Any] = {}
    for name, (path, expected_hash) in expected.items():
        if not path.is_file():
            raise ValueError(f"missing frozen input {name}: {path}")
        observed = base.file_sha256(path)
        if observed != expected_hash:
            raise ValueError(f"frozen input hash mismatch for {name}")
        audit[name] = {"path": str(path), "sha256": observed}
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_RESULTS"
        or protocol.get("selection", {}).get("identity_sha256")
        != EXACT_IDENTITY_SHA256
        or protocol.get("selection", {}).get("pdf_only_rows")
        != EXPECTED_EXACT_ROWS
        or protocol.get("selection", {}).get("pdf_only_paper_groups")
        != EXPECTED_EXACT_PAPERS
        or protocol.get("evaluation", {}).get("outer_split")
        != "5-fold REFNO GroupKFold"
        or protocol.get("promotion_gate", {}).get(
            "minimum_aggregate_accuracy"
        )
        != TARGET_ACCURACY
    ):
        raise ValueError("screening protocol does not bind the exact300 evaluation")
    return audit


def load_fold_manifest(
    path: Path,
    exact_records: Sequence[Mapping[str, str]],
) -> np.ndarray:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    if (
        manifest.get("schema_version") != "r3_exact300_label_free_refno_folds_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("input", {}).get("identity_sha256")
        != EXACT_IDENTITY_SHA256
        or manifest.get("input", {}).get("alignment_sha256")
        != sha256_records(exact_records)
        or boundary.get("labels_read") is not False
        or boundary.get("targets_workbook_read") is not False
        or boundary.get("prior_predictions_or_evaluations_read") is not False
    ):
        raise ValueError("fold manifest does not prove a label-free exact300 split")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(exact_records):
        raise ValueError("fold manifest row count differs from exact300")
    fold_ids = np.zeros(len(exact_records), dtype=np.int8)
    for index, (entry, record) in enumerate(zip(rows, exact_records)):
        if not isinstance(entry, Mapping) or (
            entry.get("row_index") != index
            or str(entry.get("REFNO")) != record["REFNO"]
            or str(entry.get("RESULTNO")) != record["RESULTNO"]
        ):
            raise ValueError("fold manifest row alignment differs from exact300")
        fold_ids[index] = int(entry.get("fold", 0))
    if hashlib.sha256(bytes(int(value) for value in fold_ids)).hexdigest() != manifest.get(
        "fold_id_sha256"
    ):
        raise ValueError("fold manifest fold-id digest mismatch")
    if Counter(int(value) for value in fold_ids) != Counter(
        {fold: 60 for fold in range(1, N_SPLITS + 1)}
    ):
        raise ValueError("fold manifest does not contain five 60-row folds")
    refno_folds: dict[str, set[int]] = defaultdict(set)
    for record, fold in zip(exact_records, fold_ids):
        refno_folds[record["REFNO"]].add(int(fold))
    if any(len(values) != 1 for values in refno_folds.values()):
        raise ValueError("fold manifest assigns one REFNO to multiple folds")
    return fold_ids


def validate_membership(
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
) -> dict[tuple[str, str, str], int]:
    if len(full_records) != EXPECTED_FULL_PDF_ROWS or len(
        {record["REFNO"] for record in full_records}
    ) != EXPECTED_FULL_PDF_PAPERS:
        raise ValueError("auxiliary candidate membership is not the frozen 2,172-row PDF set")
    if len(exact_records) != EXPECTED_EXACT_ROWS or len(
        {record["REFNO"] for record in exact_records}
    ) != EXPECTED_EXACT_PAPERS:
        raise ValueError("evaluation membership is not the frozen exact300 set")
    lookup = {row_key(record): index for index, record in enumerate(full_records)}
    if len(lookup) != len(full_records):
        raise ValueError("full PDF identity contains duplicate row identities")
    missing = [row_key(record) for record in exact_records if row_key(record) not in lookup]
    if missing:
        raise ValueError(f"exact300 rows are not a subset of the full PDF identity: {missing[:3]}")
    return lookup


def exact_group_folds(
    exact_records: Sequence[Mapping[str, str]],
    fold_ids: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([record["REFNO"] for record in exact_records], dtype=object)
    folds = [
        (np.flatnonzero(fold_ids != fold), np.flatnonzero(fold_ids == fold))
        for fold in range(1, N_SPLITS + 1)
    ]
    covered = np.concatenate([test for _, test in folds])
    if sorted(int(index) for index in covered) != list(range(len(exact_records))):
        raise AssertionError("exact300 outer folds do not cover every row once")
    for train, test in folds:
        if set(groups[train]) & set(groups[test]):
            raise AssertionError("REFNO leaked inside the exact300 fold definition")
    return folds


def auxiliary_split(
    full_records: Sequence[Mapping[str, str]],
    exact_records: Sequence[Mapping[str, str]],
    exact_test: np.ndarray,
    full_lookup: Mapping[tuple[str, str, str], int],
) -> tuple[np.ndarray, np.ndarray, set[str]]:
    test_refnos = {exact_records[int(index)]["REFNO"] for index in exact_test}
    train_full = np.asarray(
        [
            index
            for index, record in enumerate(full_records)
            if record["REFNO"] not in test_refnos
        ],
        dtype=np.int64,
    )
    test_full = np.asarray(
        [full_lookup[row_key(exact_records[int(index)])] for index in exact_test],
        dtype=np.int64,
    )
    train_refnos = {full_records[int(index)]["REFNO"] for index in train_full}
    if train_refnos & test_refnos:
        raise AssertionError("outer-test REFNO remains in auxiliary training")
    excluded_rows = {
        index
        for index, record in enumerate(full_records)
        if record["REFNO"] in test_refnos
    }
    if excluded_rows & set(int(index) for index in train_full):
        raise AssertionError("not every row of an outer-test paper was excluded")
    return train_full, test_full, test_refnos


def candidate_names(include_text: bool) -> tuple[str, ...]:
    numeric = (
        "aux_base_engineered_hgb",
        "aux_base_engineered_extra",
        "aux_base_engineered_gradient",
        "aux_curation_style_hgb",
        "aux_curation_knowledge_hgb",
        "aux_curation_all_hgb",
        "aux_curation_all_extra",
        "aux_curation_all_gradient",
        "aux_combined_hgb",
        "aux_combined_extra",
        "aux_combined_gradient",
        "aux_paper_style_hgb",
        "aux_paper_all_hgb",
        "aux_combined_numeric_nested_threshold_decision",
    )
    text = (
        "aux_target_local_char",
        "aux_target_local_word",
        "aux_rockname_char",
    )
    return numeric + (text if include_text else ())


def numeric_specs(
    base_features: Mapping[str, Any],
    curation_features: Mapping[str, Any],
) -> tuple[tuple[str, Callable[[], Any], np.ndarray], ...]:
    engineered = np.asarray(base_features["engineered_rows"], dtype=np.float64)
    style = np.asarray(curation_features["style"], dtype=np.float64)
    knowledge = np.asarray(curation_features["knowledge"], dtype=np.float64)
    all_features = np.asarray(curation_features["all"], dtype=np.float64)
    combined = np.concatenate((engineered, all_features), axis=1)
    return (
        ("aux_base_engineered_hgb", base.make_hist_gradient_boosting, engineered),
        ("aux_base_engineered_extra", base.make_extra_trees, engineered),
        ("aux_base_engineered_gradient", base.make_row_gradient_boosting, engineered),
        ("aux_curation_style_hgb", curation.make_hgb, style),
        ("aux_curation_knowledge_hgb", curation.make_hgb, knowledge),
        ("aux_curation_all_hgb", curation.make_hgb, all_features),
        ("aux_curation_all_extra", curation.make_extra, all_features),
        ("aux_curation_all_gradient", curation.make_gradient, all_features),
        ("aux_combined_hgb", curation.make_hgb, combined),
        ("aux_combined_extra", curation.make_extra, combined),
        ("aux_combined_gradient", curation.make_gradient, combined),
    )


def paper_arrays(
    full_records: Sequence[Mapping[str, str]],
    full_labels: np.ndarray,
    train_full: np.ndarray,
    test_refnos: set[str],
    curation_features: Mapping[str, Any],
    *,
    include_knowledge: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index in train_full:
        members[full_records[int(index)]["REFNO"]].append(int(index))
    train_refnos = sorted(members)
    ordered_test_refnos = sorted(test_refnos)

    def vector(refno: str) -> list[float]:
        values = list(curation_features["paper_style"][refno])
        if include_knowledge:
            values.extend(curation_features["paper_knowledge"][refno])
        return values

    labels = np.asarray(
        [int(full_labels[members[refno]].mean() >= 0.5) for refno in train_refnos],
        dtype=np.int8,
    )
    return (
        np.asarray([vector(refno) for refno in train_refnos], dtype=np.float64),
        labels,
        np.asarray([vector(refno) for refno in ordered_test_refnos], dtype=np.float64),
        ordered_test_refnos,
    )


def add_fixed_fusions(probabilities: dict[str, np.ndarray]) -> None:
    probabilities["aux_base_numeric_ensemble"] = np.mean(
        [
            probabilities["aux_base_engineered_hgb"],
            probabilities["aux_base_engineered_extra"],
            probabilities["aux_base_engineered_gradient"],
        ],
        axis=0,
    )
    probabilities["aux_curation_numeric_ensemble"] = np.mean(
        [
            probabilities["aux_curation_all_hgb"],
            probabilities["aux_curation_all_extra"],
            probabilities["aux_curation_all_gradient"],
        ],
        axis=0,
    )
    probabilities["aux_combined_numeric_ensemble"] = np.mean(
        [
            probabilities["aux_combined_hgb"],
            probabilities["aux_combined_extra"],
            probabilities["aux_combined_gradient"],
        ],
        axis=0,
    )
    row = probabilities["aux_combined_numeric_ensemble"]
    paper = probabilities["aux_paper_all_hgb"]
    for weight in (0.25, 0.50, 0.75):
        probabilities[f"aux_combined_numeric__paper_all_{int(weight * 100):02d}"] = (
            (1.0 - weight) * row + weight * paper
        )
    if "aux_target_local_char" in probabilities:
        text = np.mean(
            [
                probabilities["aux_target_local_char"],
                probabilities["aux_target_local_word"],
            ],
            axis=0,
        )
        probabilities["aux_target_text_ensemble"] = text
        for weight in (0.25, 0.50):
            probabilities[
                f"aux_combined_numeric__target_text_{int(weight * 100):02d}"
            ] = (1.0 - weight) * row + weight * text


def select_inner_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    candidates = []
    for threshold in np.arange(0.20, 0.801, 0.025):
        metric = base.metric_record(labels, probabilities, threshold=float(threshold))
        candidates.append(metric)
    selected = max(
        candidates,
        key=lambda item: (
            item["accuracy"],
            item["macro_f1"],
            item["balanced_accuracy"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    return float(selected["threshold"]), selected


def nested_numeric_threshold(
    full_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test_refnos: set[str],
    combined_matrix: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    """Choose an outer threshold only from group-isolated inner OOF rows."""

    groups = np.asarray(
        [full_records[int(index)]["REFNO"] for index in outer_train], dtype=object
    )
    target_names = [
        base.canonical_target_identity(record["ROCKNAME"]) for record in full_records
    ]
    inner_probability = np.full(len(outer_train), np.nan, dtype=np.float64)
    inner_audit: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=4)
    for inner_fold, (fit_position, validation_position) in enumerate(
        splitter.split(np.zeros(len(outer_train)), np.zeros(len(outer_train)), groups),
        start=1,
    ):
        fit_index = outer_train[fit_position]
        validation_index = outer_train[validation_position]
        fit_refnos = {full_records[int(index)]["REFNO"] for index in fit_index}
        validation_refnos = {
            full_records[int(index)]["REFNO"] for index in validation_index
        }
        if fit_refnos & validation_refnos:
            raise AssertionError("inner validation REFNO leaked into inner fitting")
        if (fit_refnos | validation_refnos) & outer_test_refnos:
            raise AssertionError("outer-test REFNO leaked into nested threshold selection")
        weights = base.inverse_target_multiplicity_weights(
            fit_index, full_records, target_names
        )
        members = []
        for factory in (curation.make_hgb, curation.make_extra, curation.make_gradient):
            model = factory()
            model.fit(
                combined_matrix[fit_index],
                full_labels[fit_index],
                sample_weight=weights,
            )
            members.append(model.predict_proba(combined_matrix[validation_index])[:, 1])
        inner_probability[validation_position] = np.mean(members, axis=0)
        inner_audit.append(
            {
                "inner_fold": inner_fold,
                "fit_rows": int(len(fit_index)),
                "fit_papers": len(fit_refnos),
                "validation_rows": int(len(validation_index)),
                "validation_papers": len(validation_refnos),
                "fit_validation_refno_overlap": 0,
                "outer_test_refno_overlap": 0,
            }
        )
    if not np.isfinite(inner_probability).all():
        raise AssertionError("inner OOF threshold probabilities are incomplete")
    threshold, metric = select_inner_threshold(
        full_labels[outer_train], inner_probability
    )
    return threshold, {
        "selection_rows": int(len(outer_train)),
        "selection_papers": len(set(str(value) for value in groups)),
        "inner_folds": inner_audit,
        "selected_threshold": threshold,
        "selected_inner_oof_metric": metric,
    }


def run_auxiliary_oof(
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    base_features: Mapping[str, Any],
    curation_features: Mapping[str, Any],
    fold_ids: np.ndarray,
    *,
    include_text: bool,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[np.ndarray]]:
    full_lookup = {row_key(record): index for index, record in enumerate(full_records)}
    folds = exact_group_folds(exact_records, fold_ids)
    probabilities = {
        name: np.full(len(exact_records), np.nan, dtype=np.float64)
        for name in candidate_names(include_text)
    }
    split_audit: list[dict[str, Any]] = []
    test_indices: list[np.ndarray] = []
    specs = numeric_specs(base_features, curation_features)
    canonical_targets = [item["canonical_target"] for item in base_features["row_meta"]]
    for fold, (_, exact_test) in enumerate(folds, start=1):
        test_indices.append(exact_test)
        train_full, test_full, test_refnos = auxiliary_split(
            full_records, exact_records, exact_test, full_lookup
        )
        weights = base.inverse_target_multiplicity_weights(
            train_full, full_records, canonical_targets
        )
        for name, factory, matrix in specs:
            model = factory()
            model.fit(matrix[train_full], full_labels[train_full], sample_weight=weights)
            probabilities[name][exact_test] = model.predict_proba(matrix[test_full])[:, 1]

        combined_matrix = next(
            matrix for name, _, matrix in specs if name == "aux_combined_hgb"
        )
        nested_threshold, nested_audit = nested_numeric_threshold(
            full_records,
            full_labels,
            train_full,
            test_refnos,
            combined_matrix,
        )
        combined_outer_probability = np.mean(
            [
                probabilities["aux_combined_hgb"][exact_test],
                probabilities["aux_combined_extra"][exact_test],
                probabilities["aux_combined_gradient"][exact_test],
            ],
            axis=0,
        )
        probabilities["aux_combined_numeric_nested_threshold_decision"][
            exact_test
        ] = (combined_outer_probability >= nested_threshold).astype(np.float64)

        for name, include_knowledge in (
            ("aux_paper_style_hgb", False),
            ("aux_paper_all_hgb", True),
        ):
            train_x, train_y, test_x, ordered_test_refnos = paper_arrays(
                full_records,
                full_labels,
                train_full,
                test_refnos,
                curation_features,
                include_knowledge=include_knowledge,
            )
            model = curation.make_hgb()
            model.fit(train_x, train_y)
            lookup = dict(zip(ordered_test_refnos, model.predict_proba(test_x)[:, 1]))
            probabilities[name][exact_test] = np.asarray(
                [lookup[exact_records[int(index)]["REFNO"]] for index in exact_test],
                dtype=np.float64,
            )

        if include_text:
            text_specs = (
                (
                    "aux_target_local_char",
                    base_features["local_documents"],
                    "char_wb",
                    (3, 5),
                    100_000,
                    2.0,
                ),
                (
                    "aux_target_local_word",
                    base_features["local_documents"],
                    "word",
                    (1, 2),
                    80_000,
                    2.0,
                ),
                (
                    "aux_rockname_char",
                    base_features["rockname_documents"],
                    "char_wb",
                    (3, 5),
                    30_000,
                    1.0,
                ),
            )
            for name, documents, analyzer, ngrams, cap, c_value in text_specs:
                predicted, _ = base.fit_row_text_model(
                    [documents[int(index)] for index in train_full],
                    [documents[int(index)] for index in test_full],
                    full_labels[train_full],
                    weights,
                    analyzer=analyzer,
                    ngram_range=ngrams,
                    max_features=cap,
                    c_value=c_value,
                )
                probabilities[name][exact_test] = predicted

        excluded_full_rows = sum(
            record["REFNO"] in test_refnos for record in full_records
        )
        split_audit.append(
            {
                "fold": fold,
                "exact_test_rows": int(len(exact_test)),
                "exact_test_papers": len(test_refnos),
                "auxiliary_train_rows": int(len(train_full)),
                "auxiliary_train_papers": len(
                    {full_records[int(index)]["REFNO"] for index in train_full}
                ),
                "full_rows_excluded_for_test_refnos": int(excluded_full_rows),
                "train_test_refno_overlap": 0,
                "nested_threshold": nested_audit,
            }
        )
        print(
            f"fold {fold}/{N_SPLITS}: train_rows={len(train_full)}, "
            f"test_rows={len(exact_test)}, excluded_same_paper_rows={excluded_full_rows}",
            flush=True,
        )
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("auxiliary OOF predictions are incomplete")
    add_fixed_fusions(probabilities)
    return probabilities, split_audit, test_indices


def write_probability_artifact(path: Path, probabilities: Mapping[str, np.ndarray]) -> None:
    if any(FORBIDDEN_ARTIFACT_NAME.search(name) for name in probabilities):
        raise ValueError("OOF artifact key contains a label or record identifier")
    arrays = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in sorted(probabilities.items())
    }
    if any(values.shape != (EXPECTED_EXACT_ROWS,) for values in arrays.values()):
        raise ValueError("OOF artifact must contain only full-coverage exact300 arrays")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=FULL_IDENTITY)
    parser.add_argument("--screening-protocol", type=Path, default=SCREENING_PROTOCOL)
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--skip-text", action="store_true")
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
    freeze_path = output_dir / FREEZE_OUTPUT.name
    report_path = output_dir / REPORT_OUTPUT.name
    oof_path = output_dir / OOF_OUTPUT.name

    input_audit = validate_inputs(
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
    validate_membership(full_records, exact_records)
    fold_ids = load_fold_manifest(fold_manifest_path, exact_records)

    refnos = {record["REFNO"] for record in full_records}
    paper_text = base.load_page_text(page_cache_path, refnos)
    pages = curation.load_pages(page_cache_path, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )
    base_features = base.build_features(full_records, paper_text, concepts)
    curation_features = curation.build_features(
        full_records, pages, concepts, pdf_dir
    )
    matrices = {
        "base_engineered": np.asarray(base_features["engineered_rows"]),
        "curation_style": np.asarray(curation_features["style"]),
        "curation_knowledge": np.asarray(curation_features["knowledge"]),
        "curation_all": np.asarray(curation_features["all"]),
    }
    include_text = not args.skip_text
    freeze = {
        "schema_version": "r3_exact300_auxiliary_training_feature_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": {
            "evaluation_rows": len(exact_records),
            "evaluation_papers": len({record["REFNO"] for record in exact_records}),
            "auxiliary_candidate_rows": len(full_records),
            "auxiliary_candidate_papers": len(refnos),
            "exact_alignment_sha256": sha256_records(exact_records),
            "full_alignment_sha256": sha256_records(full_records),
        },
        "protocol_interpretation": (
            "The frozen protocol fixes exactly 300 evaluation rows and REFNO-grouped "
            "outer tests; it does not restrict fitting to the other 240 sampled rows."
        ),
        "outer_fold_manifest": {
            "path": str(fold_manifest_path),
            "sha256": base.file_sha256(fold_manifest_path),
            "fold_id_sha256": hashlib.sha256(
                bytes(int(value) for value in fold_ids)
            ).hexdigest(),
        },
        "predeclared_base_candidates": list(candidate_names(include_text)),
        "predeclared_fixed_fusions": [
            "aux_base_numeric_ensemble",
            "aux_curation_numeric_ensemble",
            "aux_combined_numeric_ensemble",
            "aux_combined_numeric__paper_all_25",
            "aux_combined_numeric__paper_all_50",
            "aux_combined_numeric__paper_all_75",
            *(
                [
                    "aux_target_text_ensemble",
                    "aux_combined_numeric__target_text_25",
                    "aux_combined_numeric__target_text_50",
                ]
                if include_text
                else []
            ),
        ],
        "feature_matrices": {
            name: {"shape": list(matrix.shape), "sha256": sha256_array(matrix)}
            for name, matrix in matrices.items()
        },
        "text_features": {
            "enabled": include_text,
            "local_documents_sha256": sha256_strings(
                base_features["local_documents"]
            ),
            "rockname_documents_sha256": sha256_strings(
                base_features["rockname_documents"]
            ),
        },
        "allowed_feature_sources": [
            "ROCKNAME from frozen identities",
            "original-paper text/layout from frozen page cache",
            "local PDF byte size",
            "frozen aggregate R3 taxonomy",
            "frozen positive ROCKMAG expert vocabulary",
        ],
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifiers_used_as_numeric_features": False,
        },
        "inputs": input_audit,
        "knowledge": knowledge_audit,
    }
    base.atomic_json(freeze_path, freeze)
    freeze_sha = base.file_sha256(freeze_path)

    # R3 is opened only after identities, all feature candidates, and hashes freeze.
    full_labels_by_resultno = base.load_training_labels(
        targets_path, {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [full_labels_by_resultno[record["RESULTNO"]] for record in full_records],
        dtype=np.int8,
    )
    exact_labels = np.asarray(
        [full_labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )
    probabilities, split_audit, test_indices = run_auxiliary_oof(
        full_records,
        exact_records,
        full_labels,
        base_features,
        curation_features,
        fold_ids,
        include_text=include_text,
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
            for fold, index in enumerate(test_indices, start=1)
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
    write_probability_artifact(oof_path, probabilities)
    report = {
        "schema_version": "r3_exact300_auxiliary_training_grouped_oof_v1",
        "status": "complete",
        "scope": {
            "evaluated_rows": len(exact_records),
            "evaluated_papers": len({record["REFNO"] for record in exact_records}),
            "evaluated_folds": len(test_indices),
            "full_coverage": True,
            "full_2172_rows_evaluated": False,
            "auxiliary_candidate_pool": full_coverage,
            "exact_membership": exact_coverage,
        },
        "guardrails": {
            "evaluation_identity_unchanged": True,
            "outer_split": "5-fold GroupKFold by exact300 REFNO",
            "per_fold_training_rule": (
                "all 2,172 PDF-backed fixed-training rows except every row whose "
                "REFNO occurs in that exact300 outer-test fold"
            ),
            "every_test_refno_fully_excluded_from_training": True,
            "feature_freeze_before_label_access": True,
            "feature_freeze_sha256": freeze_sha,
            "pmagresult_or_rockmag_used_as_runtime_feature": False,
            "prior_predictions_or_evaluations_used_as_feature": False,
            "identifiers_used_as_model_features": False,
            "labels_used": (
                "R3 from PDF-backed fixed training rows only; in each fold labels from "
                "all outer-test REFNO rows are excluded from fitting"
            ),
        },
        "data": {
            "exact_label_counts": dict(
                sorted(Counter(str(int(value)) for value in exact_labels).items())
            ),
            "auxiliary_label_counts": dict(
                sorted(Counter(str(int(value)) for value in full_labels).items())
            ),
            "numeric_feature_shapes": {
                name: list(matrix.shape) for name, matrix in matrices.items()
            },
            "text_models_enabled": include_text,
        },
        "outer_fold_audit": split_audit,
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
                "Best-of-predeclared-candidate ranking is development evidence on the "
                "frozen exact300 screen, not independent external validation."
            ),
        },
        "artifacts": {
            "feature_freeze": {"path": str(freeze_path), "sha256": freeze_sha},
            "oof_probabilities": {
                "path": str(oof_path),
                "sha256": base.file_sha256(oof_path),
                "rows_per_array": EXPECTED_EXACT_ROWS,
                "contains_labels": False,
                "contains_identifiers": False,
            },
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
    }
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
