#!/usr/bin/env python3
"""Canonical two-stage R3 historical-missingness/conflict experiment.

Stage 1 reproduces a semantic/layout R3 score from frozen original-paper and
aggregate-knowledge features.  Stage 2 is trained only on Stage-1-positive
training rows and estimates whether strong evidence is likely to be absent
from the historical target annotation.  It sees publication, journal, layout,
section, title, and author-count style features plus their interaction with the
Stage-1 score.

Every outer test REFNO is fully removed from the 2,172-row auxiliary pool.
The correction gate is selected only from deterministic, REFNO-isolated inner
predictions within the corresponding outer training data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/historical_missingness_v2_strong_selection"
FREEZE_NAME = "feature_freeze.json"
REPORT_NAME = "report.json"
OOF_NAME = "oof_probabilities.npz"

EXPECTED_ROWS = auxiliary.EXPECTED_EXACT_ROWS
REFERENCE_ACCURACY = 0.87333333
TARGET_ACCURACY = auxiliary.TARGET_ACCURACY
INNER_FOLDS = 4
INNER_SPLIT_SALT = "r3-exact300-historical-missingness-inner-v1"
STRONG_STAGE1_THRESHOLDS = (0.20, 0.35, 0.50)
GATE_THRESHOLDS = (1.01, 0.95, 0.85, 0.75, 0.65, 0.55)
FORBIDDEN_ARTIFACT_NAME = re.compile(
    r"(?i)^(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)$"
)

TARGET_STYLE_FEATURES = {
    "target_count_log",
    "rows_in_paper_log",
    "rows_per_target_log",
    "rockname_characters_log",
    "rockname_words_log",
    "target_mention_count_log",
    "title_target_mention",
    "year_x_target_count",
    "year_x_rows_in_paper",
}


class ConstantMissingnessModel:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_missingness(self, rows: int) -> np.ndarray:
        return np.full(rows, self.probability, dtype=np.float64)


def historical_feature_indices(names: Sequence[str]) -> tuple[np.ndarray, list[str]]:
    selected = [
        (index, name)
        for index, name in enumerate(names)
        if name not in TARGET_STYLE_FEATURES
    ]
    retained_names = [name for _, name in selected]
    required_families = (
        "publication_year_",
        "journal_",
        "section_",
        "author_",
        "year_x_article_",
    )
    if any(
        not any(name.startswith(prefix) for name in retained_names)
        for prefix in required_families
    ):
        raise ValueError("historical feature set is missing a required style family")
    if any(name in TARGET_STYLE_FEATURES for name in retained_names):
        raise AssertionError("target-identity style feature entered missingness stage")
    return (
        np.asarray([index for index, _ in selected], dtype=np.int64),
        retained_names,
    )


def stage2_interaction_matrix(
    historical_matrix: np.ndarray,
    stage1_probability: np.ndarray,
) -> np.ndarray:
    score = np.asarray(stage1_probability, dtype=np.float64).reshape(-1, 1)
    history = np.asarray(historical_matrix, dtype=np.float64)
    if len(history) != len(score):
        raise ValueError("historical rows and Stage-1 scores are misaligned")
    return np.hstack((history, score, history * score))


def balanced_group_fold_ids(
    records: Sequence[Mapping[str, str]],
    indices: np.ndarray,
    *,
    n_splits: int = INNER_FOLDS,
    salt: str = INNER_SPLIT_SALT,
) -> np.ndarray:
    members: dict[str, list[int]] = defaultdict(list)
    for position, index in enumerate(indices):
        members[records[int(index)]["REFNO"]].append(position)

    def tie_key(refno: str) -> str:
        return hashlib.sha256(f"{salt}\0{refno}".encode("utf-8")).hexdigest()

    ordered = sorted(members, key=lambda value: (-len(members[value]), tie_key(value)))
    row_load = [0] * n_splits
    group_load = [0] * n_splits
    group_fold: dict[str, int] = {}
    for refno in ordered:
        fold = min(
            range(n_splits),
            key=lambda value: (row_load[value], group_load[value], value),
        )
        group_fold[refno] = fold
        row_load[fold] += len(members[refno])
        group_load[fold] += 1
    output = np.asarray(
        [group_fold[records[int(index)]["REFNO"]] for index in indices],
        dtype=np.int8,
    )
    if set(int(value) for value in output) != set(range(n_splits)):
        raise ValueError("deterministic inner folds are incomplete")
    refno_folds: dict[str, set[int]] = defaultdict(set)
    for index, fold in zip(indices, output):
        refno_folds[records[int(index)]["REFNO"]].add(int(fold))
    if any(len(values) != 1 for values in refno_folds.values()):
        raise AssertionError("deterministic inner split divides one REFNO")
    return output


def stage1_weights(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    canonical_targets: Sequence[str],
) -> np.ndarray:
    return base.inverse_target_multiplicity_weights(
        indices, records, canonical_targets
    )


def fit_stage1(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    canonical_targets: Sequence[str],
) -> Any:
    model = curation.make_hgb()
    model.fit(
        matrix[fit_indices],
        labels[fit_indices],
        sample_weight=stage1_weights(fit_indices, records, canonical_targets),
    )
    return model


def missingness_weights(
    indices: np.ndarray,
    missing_labels: np.ndarray,
    records: Sequence[Mapping[str, str]],
) -> np.ndarray:
    paper_counts = Counter(records[int(index)]["REFNO"] for index in indices)
    weights = np.asarray(
        [1.0 / paper_counts[records[int(index)]["REFNO"]] for index in indices],
        dtype=np.float64,
    )
    for value in (0, 1):
        selected = missing_labels == value
        if selected.any():
            weights[selected] *= 0.5 / float(weights[selected].sum())
    weights *= len(weights) / float(weights.sum())
    return weights


def make_missingness_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.035,
        max_leaf_nodes=5,
        min_samples_leaf=15,
        l2_regularization=2.0,
        random_state=base.RANDOM_STATE,
    )


def fit_missingness_model(
    historical_matrix: np.ndarray,
    stage1_probability: np.ndarray,
    labels: np.ndarray,
    fit_indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    strong_threshold: float,
) -> tuple[Any, dict[str, Any]]:
    strong_position = np.flatnonzero(
        stage1_probability >= strong_threshold
    )
    strong_indices = fit_indices[strong_position]
    missing_labels = (1 - labels[strong_indices]).astype(np.int8)
    counts = Counter(int(value) for value in missing_labels)
    if not strong_indices.size:
        raise ValueError("Stage 1 produced no strong-evidence training rows")
    if len(counts) < 2:
        probability = float(missing_labels.mean())
        model: Any = ConstantMissingnessModel(probability)
        kind = "constant_one_class_fallback"
    else:
        matrix = stage2_interaction_matrix(
            historical_matrix[strong_indices],
            stage1_probability[strong_position],
        )
        model = make_missingness_model()
        model.fit(
            matrix,
            missing_labels,
            sample_weight=missingness_weights(
                strong_indices, missing_labels, records
            ),
        )
        kind = "balanced_historical_hgb"
    return model, {
        "kind": kind,
        "strong_training_rows": int(len(strong_indices)),
        "strong_training_papers": len(
            {records[int(index)]["REFNO"] for index in strong_indices}
        ),
        "missing_proxy_class_counts": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        "strong_stage1_threshold": strong_threshold,
    }


def predict_missingness(
    model: Any,
    historical_rows: np.ndarray,
    stage1_probability: np.ndarray,
) -> np.ndarray:
    if isinstance(model, ConstantMissingnessModel):
        return model.predict_missingness(len(historical_rows))
    matrix = stage2_interaction_matrix(historical_rows, stage1_probability)
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)


def apply_gate(
    stage1_probability: np.ndarray,
    missing_probability: np.ndarray,
    gate_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(stage1_probability, dtype=np.float64).copy()
    flips = (probability >= 0.5) & (
        np.asarray(missing_probability, dtype=np.float64) >= gate_threshold
    )
    probability[flips] = np.minimum(probability[flips], 0.499999)
    return probability, flips


def select_gate(
    labels: np.ndarray,
    stage1_probability: np.ndarray,
    missing_probability: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for threshold in GATE_THRESHOLDS:
        adjusted, flips = apply_gate(
            stage1_probability, missing_probability, threshold
        )
        candidates.append(
            {
                "gate_threshold": threshold,
                "flipped_positive_rows": int(flips.sum()),
                **base.metric_record(labels, adjusted),
            }
        )
    selected = max(
        candidates,
        key=lambda item: (
            item["accuracy"],
            item["macro_f1"],
            item["balanced_accuracy"],
            -item["flipped_positive_rows"],
            item["gate_threshold"],
        ),
    )
    return float(selected["gate_threshold"]), {
        "selected": selected,
        "candidates": candidates,
    }


def select_strong_threshold_and_gate(
    labels: np.ndarray,
    stage1_probability: np.ndarray,
    missing_probabilities: Mapping[float, np.ndarray],
) -> tuple[float, float, dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for strong_threshold in STRONG_STAGE1_THRESHOLDS:
        gate_threshold, gate_audit = select_gate(
            labels,
            stage1_probability,
            missing_probabilities[strong_threshold],
        )
        selections.append(
            {
                "strong_training_threshold": strong_threshold,
                "selected_gate_threshold": gate_threshold,
                "selected_gate_metric": gate_audit["selected"],
                "gate_candidates": gate_audit["candidates"],
            }
        )
    selected = max(
        selections,
        key=lambda item: (
            item["selected_gate_metric"]["accuracy"],
            item["selected_gate_metric"]["macro_f1"],
            item["selected_gate_metric"]["balanced_accuracy"],
            -item["selected_gate_metric"]["flipped_positive_rows"],
            item["strong_training_threshold"],
            item["selected_gate_threshold"],
        ),
    )
    return (
        float(selected["strong_training_threshold"]),
        float(selected["selected_gate_threshold"]),
        {"selected": selected, "strong_threshold_candidates": selections},
    )


def nested_gate_selection(
    full_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    outer_train: np.ndarray,
    outer_test_refnos: set[str],
    stage1_matrix: np.ndarray,
    historical_matrix: np.ndarray,
    canonical_targets: Sequence[str],
) -> tuple[float, float, dict[str, Any]]:
    inner_fold_ids = balanced_group_fold_ids(full_records, outer_train)
    stage1_oof = np.full(len(outer_train), np.nan, dtype=np.float64)
    missing_oof = {
        threshold: np.full(len(outer_train), np.nan, dtype=np.float64)
        for threshold in STRONG_STAGE1_THRESHOLDS
    }
    inner_audit: list[dict[str, Any]] = []
    for inner_fold in range(INNER_FOLDS):
        fit_position = np.flatnonzero(inner_fold_ids != inner_fold)
        validation_position = np.flatnonzero(inner_fold_ids == inner_fold)
        fit_indices = outer_train[fit_position]
        validation_indices = outer_train[validation_position]
        fit_refnos = {full_records[int(index)]["REFNO"] for index in fit_indices}
        validation_refnos = {
            full_records[int(index)]["REFNO"] for index in validation_indices
        }
        if fit_refnos & validation_refnos:
            raise AssertionError("inner validation REFNO leaked into fitting")
        if (fit_refnos | validation_refnos) & outer_test_refnos:
            raise AssertionError("outer-test REFNO entered inner model selection")

        stage1_model = fit_stage1(
            stage1_matrix,
            full_labels,
            fit_indices,
            full_records,
            canonical_targets,
        )
        fit_stage1_probability = stage1_model.predict_proba(
            stage1_matrix[fit_indices]
        )[:, 1]
        validation_stage1_probability = stage1_model.predict_proba(
            stage1_matrix[validation_indices]
        )[:, 1]
        missing_audit: dict[str, Any] = {}
        for strong_threshold in STRONG_STAGE1_THRESHOLDS:
            missing_model, fit_audit = fit_missingness_model(
                historical_matrix,
                fit_stage1_probability,
                full_labels,
                fit_indices,
                full_records,
                strong_threshold,
            )
            validation_missing_probability = predict_missingness(
                missing_model,
                historical_matrix[validation_indices],
                validation_stage1_probability,
            )
            missing_oof[strong_threshold][
                validation_position
            ] = validation_missing_probability
            missing_audit[f"strong_{strong_threshold:.2f}"] = fit_audit
        stage1_oof[validation_position] = validation_stage1_probability
        inner_audit.append(
            {
                "inner_fold": inner_fold + 1,
                "fit_rows": int(len(fit_indices)),
                "fit_papers": len(fit_refnos),
                "validation_rows": int(len(validation_indices)),
                "validation_papers": len(validation_refnos),
                "fit_validation_refno_overlap": 0,
                "outer_test_refno_overlap": 0,
                "missingness_fit": missing_audit,
            }
        )
    if not np.isfinite(stage1_oof).all() or any(
        not np.isfinite(values).all() for values in missing_oof.values()
    ):
        raise AssertionError("nested two-stage predictions are incomplete")
    strong_threshold, gate_threshold, gate_audit = select_strong_threshold_and_gate(
        full_labels[outer_train], stage1_oof, missing_oof
    )
    return strong_threshold, gate_threshold, {
        "selection_rows": int(len(outer_train)),
        "selection_papers": len(
            {full_records[int(index)]["REFNO"] for index in outer_train}
        ),
        "inner_split": {
            "method": "label-free greedy row-balanced REFNO assignment",
            "salt": INNER_SPLIT_SALT,
            "folds": INNER_FOLDS,
            "fold_id_sha256": hashlib.sha256(
                bytes(int(value) for value in inner_fold_ids)
            ).hexdigest(),
        },
        "inner_folds": inner_audit,
        "joint_strong_threshold_and_gate_selection": gate_audit,
    }


def run_outer_oof(
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    fold_ids: np.ndarray,
    stage1_matrix: np.ndarray,
    historical_matrix: np.ndarray,
    canonical_targets: Sequence[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[np.ndarray], list[dict[str, Any]]]:
    full_lookup = {
        auxiliary.row_key(record): index
        for index, record in enumerate(full_records)
    }
    folds = auxiliary.exact_group_folds(exact_records, fold_ids)
    probabilities = {
        "stage1_combined_hgb": np.full(len(exact_records), np.nan),
        "historical_missingness_nested_gate": np.full(len(exact_records), np.nan),
    }
    missingness_proxy = np.full(len(exact_records), np.nan)
    fold_indices: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    assignment = np.zeros(len(exact_records), dtype=np.int8)
    for fold, (_, exact_test) in enumerate(folds, start=1):
        train_full, test_full, test_refnos = auxiliary.auxiliary_split(
            full_records, exact_records, exact_test, full_lookup
        )
        fold_indices.append(exact_test)
        assignment[exact_test] += 1
        strong_threshold, gate_threshold, nested_audit = nested_gate_selection(
            full_records,
            full_labels,
            train_full,
            test_refnos,
            stage1_matrix,
            historical_matrix,
            canonical_targets,
        )

        stage1_model = fit_stage1(
            stage1_matrix,
            full_labels,
            train_full,
            full_records,
            canonical_targets,
        )
        train_stage1_probability = stage1_model.predict_proba(
            stage1_matrix[train_full]
        )[:, 1]
        test_stage1_probability = stage1_model.predict_proba(
            stage1_matrix[test_full]
        )[:, 1]
        missing_model, missing_audit = fit_missingness_model(
            historical_matrix,
            train_stage1_probability,
            full_labels,
            train_full,
            full_records,
            strong_threshold,
        )
        test_missing_probability = predict_missingness(
            missing_model,
            historical_matrix[test_full],
            test_stage1_probability,
        )
        adjusted, flips = apply_gate(
            test_stage1_probability,
            test_missing_probability,
            gate_threshold,
        )
        probabilities["stage1_combined_hgb"][exact_test] = test_stage1_probability
        probabilities["historical_missingness_nested_gate"][exact_test] = adjusted
        missingness_proxy[exact_test] = test_missing_probability

        training_refnos = {
            full_records[int(index)]["REFNO"] for index in train_full
        }
        excluded_rows = sum(
            record["REFNO"] in test_refnos for record in full_records
        )
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
                "selected_gate_threshold": gate_threshold,
                "selected_strong_training_threshold": strong_threshold,
                "outer_test_positive_flips": int(flips.sum()),
                "missingness_fit": missing_audit,
                "nested_selection": nested_audit,
            }
        )
        print(
            f"fold {fold}/5: train_rows={len(train_full)}, "
            f"test_rows={len(exact_test)}, strong={strong_threshold:.2f}, "
            f"gate={gate_threshold:.2f}, "
            f"flips={int(flips.sum())}",
            flush=True,
        )
    if not np.array_equal(assignment, np.ones(len(exact_records), dtype=np.int8)):
        raise AssertionError("canonical exact300 rows were not tested exactly once")
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("two-stage OOF probabilities are incomplete")
    if not np.isfinite(missingness_proxy).all():
        raise AssertionError("missingness proxy coverage is incomplete")
    return probabilities, missingness_proxy, fold_indices, audits


def write_probability_artifact(
    path: Path,
    probabilities: Mapping[str, np.ndarray],
) -> None:
    if any(FORBIDDEN_ARTIFACT_NAME.search(name) for name in probabilities):
        raise ValueError("OOF artifact key contains a label or record identifier")
    arrays = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in sorted(probabilities.items())
    }
    if any(values.shape != (EXPECTED_ROWS,) for values in arrays.values()):
        raise ValueError("OOF artifact must contain exact300 arrays only")
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
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
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
    fold_ids = auxiliary.load_fold_manifest(fold_manifest_path, exact_records)

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
    stage1_matrix = np.concatenate(
        (
            np.asarray(base_features["engineered_rows"], dtype=np.float64),
            np.asarray(curation_features["all"], dtype=np.float64),
        ),
        axis=1,
    )
    historical_indices, historical_names = historical_feature_indices(
        curation_features["style_feature_names"]
    )
    historical_matrix = np.asarray(
        curation_features["style"], dtype=np.float64
    )[:, historical_indices]
    canonical_targets = [
        item["canonical_target"] for item in base_features["row_meta"]
    ]

    freeze_path = output_dir / FREEZE_NAME
    freeze = {
        "schema_version": "r3_exact300_historical_missingness_feature_freeze_v2",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": {
            "evaluation_rows": len(exact_records),
            "evaluation_papers": len({record["REFNO"] for record in exact_records}),
            "auxiliary_rows": len(full_records),
            "auxiliary_papers": len(refnos),
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": auxiliary.sha256_records(full_records),
        },
        "canonical_fold_manifest": {
            "path": str(fold_manifest_path),
            "sha256": base.file_sha256(fold_manifest_path),
            "fold_id_sha256": hashlib.sha256(
                bytes(int(value) for value in fold_ids)
            ).hexdigest(),
        },
        "candidate": {
            "stage1": "combined engineered + curation-all HistGradientBoosting",
            "stage2": "balanced historical-style HistGradientBoosting",
            "strong_stage1_threshold_candidates": list(STRONG_STAGE1_THRESHOLDS),
            "gate_threshold_candidates": list(GATE_THRESHOLDS),
            "gate_selection": "outer-train-only deterministic 4-fold REFNO OOF",
            "inner_split_salt": INNER_SPLIT_SALT,
            "history_semantic_interaction": True,
        },
        "feature_matrices": {
            "stage1": {
                "shape": list(stage1_matrix.shape),
                "sha256": auxiliary.sha256_array(stage1_matrix),
            },
            "historical_style": {
                "shape": list(historical_matrix.shape),
                "sha256": auxiliary.sha256_array(historical_matrix),
                "feature_names": historical_names,
            },
        },
        "allowed_predictive_sources": [
            "ROCKNAME-derived target context from frozen identity",
            "original-paper text and layout from frozen page cache",
            "local PDF size",
            "frozen aggregate terminology patterns",
        ],
        "runtime_boundary": {
            "labels_read": False,
            "source_field_cells_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifiers_used_as_predictive_features": False,
            "author_names_used_as_features": False,
        },
        "inputs": input_audit,
        "knowledge": knowledge_audit,
    }
    base.atomic_json(freeze_path, freeze)
    freeze_sha256 = base.file_sha256(freeze_path)

    # R3 is opened only after feature construction, candidate declaration, and freeze.
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
    probabilities, missingness_proxy, fold_indices, fold_audit = run_outer_oof(
        full_records,
        exact_records,
        full_labels,
        fold_ids,
        stage1_matrix,
        historical_matrix,
        canonical_targets,
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
    artifact_arrays = {
        **probabilities,
        "historical_missingness_proxy": missingness_proxy,
    }
    oof_path = output_dir / OOF_NAME
    write_probability_artifact(oof_path, artifact_arrays)
    adjusted_prediction = (
        probabilities["historical_missingness_nested_gate"] >= 0.5
    )
    stage1_prediction = probabilities["stage1_combined_hgb"] >= 0.5
    report = {
        "schema_version": "r3_exact300_historical_missingness_grouped_oof_v2",
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
            "canonical_fold_manifest_sha256": base.file_sha256(fold_manifest_path),
            "per_fold_training_rule": (
                "all PDF-backed auxiliary rows except every row belonging to an "
                "exact300 outer-test REFNO"
            ),
            "every_test_refno_fully_excluded_from_training": all(
                item["all_test_paper_rows_excluded"]
                and item["train_test_refno_overlap"] == 0
                for item in fold_audit
            ),
            "inner_gate_selection_refno_isolated": True,
            "feature_freeze_before_label_access": True,
            "feature_freeze_sha256": freeze_sha256,
            "raw_source_fields_used_as_features": False,
            "prior_predictions_or_evaluations_used_as_features": False,
            "identifiers_used_as_predictive_features": False,
            "author_names_used_as_features": False,
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
        "historical_conflict_diagnostic": {
            "rows_flipped_from_stage1_positive_to_negative": int(
                (stage1_prediction & ~adjusted_prediction).sum()
            ),
            "corrected_false_positives": int(
                (stage1_prediction & ~adjusted_prediction & (exact_labels == 0)).sum()
            ),
            "introduced_false_negatives": int(
                (stage1_prediction & ~adjusted_prediction & (exact_labels == 1)).sum()
            ),
            "proxy_mean_by_evaluation_label": {
                "0": round(float(missingness_proxy[exact_labels == 0].mean()), 8),
                "1": round(float(missingness_proxy[exact_labels == 1].mean()), 8),
            },
        },
        "reference_comparison": {
            "reference_strategy": "canonical aux_combined_hgb",
            "reference_accuracy": REFERENCE_ACCURACY,
            "candidate_accuracy": metrics["historical_missingness_nested_gate"][
                "accuracy"
            ],
            "absolute_gain": round(
                metrics["historical_missingness_nested_gate"]["accuracy"]
                - REFERENCE_ACCURACY,
                8,
            ),
            "improved": metrics["historical_missingness_nested_gate"]["accuracy"]
            > REFERENCE_ACCURACY,
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
                "reference_comparison": report["reference_comparison"],
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
