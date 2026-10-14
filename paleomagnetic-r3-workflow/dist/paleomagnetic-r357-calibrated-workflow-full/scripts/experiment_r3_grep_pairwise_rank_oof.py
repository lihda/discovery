#!/usr/bin/env python3
"""Nested OOF evaluation of GREP-inspired within-paper pairwise R3 ranking."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_all_labelled_pdf_grouped_oof as all_oof  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_exact300_paper_consistency as paper_v1  # noqa: E402
import experiment_r3_grep_entity_pair_oof as grep_oof  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


OOF_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
COMBINED_FEATURES = OOF_DIR / "combined_features.npz"
COMBINED_MANIFEST = OOF_DIR / "combined_features.manifest.json"
GREP_FEATURES = OOF_DIR / "grep_entity_pair_features_v1.npz"
GREP_MANIFEST = OOF_DIR / "grep_entity_pair_features_v1.manifest.json"
FOLD_MANIFEST = OOF_DIR / "refno_fold_manifest.json"
PRIOR_GREP_OOF = OOF_DIR / "grep_entity_pair_oof_v1.npz"
CANDIDATE_FREEZE = OOF_DIR / "grep_pairwise_rank_candidate_freeze_v2.json"
OOF_OUTPUT = OOF_DIR / "grep_pairwise_rank_oof_v2.npz"
PREDICTION_RECEIPT = OOF_DIR / "grep_pairwise_rank_prediction_receipt_v2.json"
REPORT_OUTPUT = OOF_DIR / "grep_pairwise_rank_report_v2.json"

PRIMARY_PAPER = {
    "title": "Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification",
    "authors": [
        "Ravi Patel",
        "Angus Brayne",
        "Rogier Hintzen",
        "Daniel Jaroslawicz",
        "Georgiana Neculae",
        "Dane S. Corneil",
    ],
    "venue": "Proceedings of ACL 2025, Volume 1: Long Papers",
    "publication_date": "2025-07",
    "url": "https://aclanthology.org/2025.acl-long.167/",
    "doi": "10.18653/v1/2025.acl-long.167",
}
SUPPORTING_PAPER = grep_oof.PAPER
PRIMARY_STRATEGY = "grep_pairwise_nested_router"
BASELINE_STRATEGY = "grep_expanded_mean4_recomputed"
INNER_SPLITS = 3
INNER_SEED = 35_720_260_731
STANDARDIZED_CLIP = 12.0


@dataclass(frozen=True)
class RankSpec:
    name: str
    c_value: float


@dataclass(frozen=True)
class GateSpec:
    name: str
    kind: str


@dataclass(frozen=True)
class FusionSpec:
    name: str
    rank: str | None
    gate: str | None
    gate_threshold: float
    alpha: float


def rank_specs() -> tuple[RankSpec, ...]:
    return (
        RankSpec("pair_logreg_c0001", 0.0001),
        RankSpec("pair_logreg_c001", 0.001),
        RankSpec("pair_logreg_c01", 0.01),
    )


def gate_specs() -> tuple[GateSpec, ...]:
    return (
        GateSpec("heterogeneity_logreg_c01", "logreg_c01"),
        GateSpec("heterogeneity_hgb_leaf20", "hgb_leaf20"),
    )


def fusion_specs() -> tuple[FusionSpec, ...]:
    specs = [FusionSpec("identity", None, None, 1.0, 0.0)]
    for rank in rank_specs():
        for gate in gate_specs():
            for threshold in (0.5, 0.7):
                for alpha in (0.35, 0.5, 0.75, 1.0):
                    specs.append(
                        FusionSpec(
                            f"{rank.name}__{gate.name}__t{threshold:.2f}__a{alpha:.2f}",
                            rank.name,
                            gate.name,
                            threshold,
                            alpha,
                        )
                    )
    return tuple(specs)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -40.0, 40.0)))


def rows_by_refno(records: Sequence[Mapping[str, str]]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        output[str(record["REFNO"])].append(index)
    return dict(output)


def build_pairwise_examples(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Build symmetric positive-minus-negative pairs, equally weighted by paper."""

    train_set = set(int(index) for index in train_indices)
    differences: list[np.ndarray] = []
    pair_labels: list[int] = []
    weights: list[float] = []
    mixed_papers = 0
    unordered_pairs = 0
    for indices in rows_by_refno(records).values():
        selected = [index for index in indices if index in train_set]
        positive = [index for index in selected if int(labels[index]) == 1]
        negative = [index for index in selected if int(labels[index]) == 0]
        paper_pairs = len(positive) * len(negative)
        if not paper_pairs:
            continue
        mixed_papers += 1
        unordered_pairs += paper_pairs
        example_weight = 0.5 / paper_pairs
        for positive_index in positive:
            for negative_index in negative:
                difference = np.asarray(
                    matrix[positive_index] - matrix[negative_index],
                    dtype=np.float64,
                )
                differences.extend((difference, -difference))
                pair_labels.extend((1, 0))
                weights.extend((example_weight, example_weight))
    if not differences:
        raise ValueError("pairwise training fold contains no mixed-label papers")
    return (
        np.asarray(differences, dtype=np.float64),
        np.asarray(pair_labels, dtype=np.int8),
        np.asarray(weights, dtype=np.float64),
        {"mixed_papers": mixed_papers, "unordered_pairs": unordered_pairs},
    )


@dataclass
class FittedRanker:
    scaler: StandardScaler
    model: LogisticRegression

    def score(self, matrix: np.ndarray) -> np.ndarray:
        transformed = np.clip(
            self.scaler.transform(np.asarray(matrix, dtype=np.float64)),
            -STANDARDIZED_CLIP,
            STANDARDIZED_CLIP,
        )
        with warnings.catch_warnings(), np.errstate(
            divide="ignore", over="ignore", invalid="ignore"
        ):
            warnings.simplefilter("ignore", RuntimeWarning)
            score = np.asarray(self.model.decision_function(transformed), dtype=np.float64)
        if score.ndim != 1 or not np.isfinite(score).all():
            raise AssertionError("pairwise ranker produced invalid scores")
        return score


def fit_ranker(
    spec: RankSpec,
    matrix: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
) -> tuple[FittedRanker, dict[str, int]]:
    differences, pair_labels, weights, audit = build_pairwise_examples(
        matrix, labels, train_indices, records
    )
    scaler = StandardScaler()
    transformed = np.clip(
        scaler.fit_transform(differences), -STANDARDIZED_CLIP, STANDARDIZED_CLIP
    )
    model = LogisticRegression(
        C=spec.c_value,
        solver="liblinear",
        fit_intercept=False,
        max_iter=3_000,
        random_state=nested.RANDOM_STATE,
    )
    model.fit(transformed, pair_labels, sample_weight=weights * len(weights))
    return FittedRanker(scaler, model), audit


def normalize_scores_by_paper(
    raw_scores: np.ndarray,
    records: Sequence[Mapping[str, str]],
    selected_indices: np.ndarray,
) -> np.ndarray:
    """Center and RMS-normalize ranks using only siblings in each complete paper."""

    values = np.asarray(raw_scores, dtype=np.float64)
    if values.shape != (len(records),) or not np.isfinite(values).all():
        raise ValueError("raw rank scores must align with records")
    selected = set(int(index) for index in selected_indices)
    output = np.zeros(len(records), dtype=np.float64)
    for indices in rows_by_refno(records).values():
        group = [index for index in indices if index in selected]
        if len(group) < 2:
            continue
        centered = values[group] - float(values[group].mean())
        scale = float(np.sqrt(np.mean(np.square(centered))))
        if scale > 1e-9:
            output[group] = np.clip(centered / scale, -3.0, 3.0)
    return output


def paper_aggregate_features(
    grep_matrix: np.ndarray,
    records: Sequence[Mapping[str, str]],
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Create label-free paper heterogeneity features from GREP row features."""

    by_refno = rows_by_refno(records)
    refnos = sorted(by_refno)
    features: list[np.ndarray] = []
    for refno in refnos:
        values = np.asarray(grep_matrix[by_refno[refno]], dtype=np.float64)
        features.append(
            np.concatenate(
                (
                    np.asarray([np.log1p(len(values))], dtype=np.float64),
                    values.mean(axis=0),
                    values.std(axis=0),
                    np.ptp(values, axis=0),
                )
            )
        )
    paper_matrix = np.asarray(features, dtype=np.float64)
    if not np.isfinite(paper_matrix).all():
        raise ValueError("paper aggregate matrix contains non-finite values")
    row_to_paper = np.asarray(
        [{refno: index for index, refno in enumerate(refnos)}[record["REFNO"]]
         for record in records],
        dtype=np.int64,
    )
    return refnos, paper_matrix, row_to_paper


@dataclass
class FittedGate:
    model: Any
    scaler: StandardScaler | None

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        values = np.asarray(matrix, dtype=np.float64)
        if self.scaler is not None:
            values = np.clip(
                self.scaler.transform(values), -STANDARDIZED_CLIP, STANDARDIZED_CLIP
            )
        probability = np.asarray(self.model.predict_proba(values)[:, 1], dtype=np.float64)
        if probability.ndim != 1 or not np.isfinite(probability).all():
            raise AssertionError("heterogeneity gate produced invalid probabilities")
        return probability


def fit_gate(
    spec: GateSpec,
    paper_matrix: np.ndarray,
    paper_labels: np.ndarray,
    train_papers: np.ndarray,
) -> FittedGate:
    train = np.asarray(paper_matrix[train_papers], dtype=np.float64)
    labels = paper_labels[train_papers]
    if set(int(value) for value in labels) != {0, 1}:
        raise ValueError("gate training fold must contain uniform and mixed papers")
    scaler: StandardScaler | None = None
    if spec.kind == "logreg_c01":
        scaler = StandardScaler()
        train = np.clip(
            scaler.fit_transform(train), -STANDARDIZED_CLIP, STANDARDIZED_CLIP
        )
        model: Any = LogisticRegression(
            C=0.01,
            solver="liblinear",
            max_iter=3_000,
            class_weight="balanced",
            random_state=nested.RANDOM_STATE,
        )
    elif spec.kind == "hgb_leaf20":
        model = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.04,
            max_leaf_nodes=5,
            min_samples_leaf=20,
            l2_regularization=2.0,
            class_weight="balanced",
            random_state=nested.RANDOM_STATE,
        )
    else:
        raise ValueError(f"unknown gate kind: {spec.kind}")
    model.fit(train, labels)
    return FittedGate(model, scaler)


def fit_baseline_mean4(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    canonical_targets: Sequence[str],
) -> np.ndarray:
    weights = base.inverse_target_multiplicity_weights(
        train_indices, records, canonical_targets
    )
    predictions = []
    for spec in grep_oof.model_specs():
        fitted = nested.fit_expert(spec, matrix, train_indices, labels, weights)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            predictions.append(fitted.predict(matrix[predict_indices]))
    return np.column_stack(predictions).mean(axis=1)


def multi_target_mask(
    records: Sequence[Mapping[str, str]],
    labelled_records: Sequence[Mapping[str, str]],
) -> np.ndarray:
    counts = Counter(record["REFNO"] for record in labelled_records)
    return np.asarray([counts[record["REFNO"]] > 1 for record in records], dtype=bool)


def mixed_label_mask(
    records: Sequence[Mapping[str, str]], labels: np.ndarray
) -> np.ndarray:
    by_refno = rows_by_refno(records)
    state = {
        refno: len({int(labels[index]) for index in indices}) > 1
        for refno, indices in by_refno.items()
    }
    return np.asarray([state[record["REFNO"]] for record in records], dtype=bool)


def paper_training_indices(
    row_indices: np.ndarray,
    refnos: Sequence[str],
    row_to_paper: np.ndarray,
    paper_target_counts: np.ndarray,
) -> np.ndarray:
    del refnos
    paper_indices = sorted({int(row_to_paper[index]) for index in row_indices})
    return np.asarray(
        [index for index in paper_indices if int(paper_target_counts[index]) > 1],
        dtype=np.int64,
    )


def apply_fusion(
    baseline: np.ndarray,
    rank_score: np.ndarray,
    gate_probability: np.ndarray,
    multi_mask: np.ndarray,
    spec: FusionSpec,
) -> np.ndarray:
    if spec.rank is None:
        return np.asarray(baseline, dtype=np.float64).copy()
    gate = (gate_probability >= spec.gate_threshold).astype(np.float64)
    correction = spec.alpha * rank_score * gate * multi_mask.astype(np.float64)
    return sigmoid(logit(baseline) + correction)


def accuracy(labels: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.0
    return float(np.mean((probability[mask] >= 0.5) == labels[mask]))


def clustered_bootstrap_delta(
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    *,
    iterations: int = 10_000,
    seed: int = 35_720_260_747,
) -> dict[str, Any]:
    """Paired accuracy delta with whole REFNO groups as resampling units."""

    by_refno: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        by_refno[records[int(index)]["REFNO"]].append(int(index))
    groups = list(by_refno.values())
    if not groups:
        raise ValueError("cluster bootstrap requires at least one REFNO group")
    baseline_correct = (np.asarray(baseline) >= 0.5) == labels
    candidate_correct = (np.asarray(candidate) >= 0.5) == labels
    correct_delta = np.asarray(
        [
            int(candidate_correct[indices].sum()) - int(baseline_correct[indices].sum())
            for indices in groups
        ],
        dtype=np.float64,
    )
    row_counts = np.asarray([len(indices) for indices in groups], dtype=np.float64)
    generator = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        chosen = generator.integers(0, len(groups), size=len(groups))
        samples[iteration] = correct_delta[chosen].sum() / row_counts[chosen].sum()
    observed = float(correct_delta.sum() / row_counts.sum())
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {
        "resampling_unit": "REFNO",
        "groups": len(groups),
        "rows": int(row_counts.sum()),
        "iterations": iterations,
        "observed_accuracy_delta": round(observed, 8),
        "percentile_95_interval": [round(float(lower), 8), round(float(upper), 8)],
        "bootstrap_probability_delta_positive": round(float(np.mean(samples > 0.0)), 8),
    }


def select_fusion(
    specs: Sequence[FusionSpec],
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    multi_mask: np.ndarray,
    mixed_mask: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[FusionSpec, dict[str, float]]:
    selected = np.zeros(len(labels), dtype=bool)
    selected[selected_indices] = True
    multi = selected & multi_mask
    mixed = selected & mixed_mask
    best_spec = specs[0]
    best_metrics = {
        "multi_target_accuracy": accuracy(labels, probabilities[best_spec.name], multi),
        "mixed_label_accuracy": accuracy(labels, probabilities[best_spec.name], mixed),
        "overall_accuracy": accuracy(labels, probabilities[best_spec.name], selected),
    }
    best_key = (
        best_metrics["multi_target_accuracy"],
        best_metrics["overall_accuracy"],
        best_metrics["mixed_label_accuracy"],
    )
    for spec in specs[1:]:
        metrics = {
            "multi_target_accuracy": accuracy(labels, probabilities[spec.name], multi),
            "mixed_label_accuracy": accuracy(labels, probabilities[spec.name], mixed),
            "overall_accuracy": accuracy(labels, probabilities[spec.name], selected),
        }
        key = (
            metrics["multi_target_accuracy"],
            metrics["overall_accuracy"],
            metrics["mixed_label_accuracy"],
        )
        if key > best_key:
            best_spec, best_metrics, best_key = spec, metrics, key
    return best_spec, best_metrics


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=all_oof.IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--combined-features", type=Path, default=COMBINED_FEATURES)
    parser.add_argument("--combined-manifest", type=Path, default=COMBINED_MANIFEST)
    parser.add_argument("--grep-features", type=Path, default=GREP_FEATURES)
    parser.add_argument("--grep-manifest", type=Path, default=GREP_MANIFEST)
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OOF_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / CANDIDATE_FREEZE.name
    oof_path = output_dir / OOF_OUTPUT.name
    receipt_path = output_dir / PREDICTION_RECEIPT.name
    report_path = output_dir / REPORT_OUTPUT.name

    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    if runtime != paper_v1.CANONICAL_RUNTIME:
        raise ValueError(
            f"canonical runtime required: expected={paper_v1.CANONICAL_RUNTIME}, observed={runtime}"
        )

    labelled_records = all_oof.load_identity(args.identity.resolve())
    records, coverage = all_oof.pdf_only_records(labelled_records, args.pdf_dir.resolve())
    alignment_sha256 = all_oof.record_sha256(records)
    fold_ids = grep_oof.load_fold_ids(args.fold_manifest.resolve(), records)
    folds = grep_oof.fold_indices(fold_ids)

    combined_manifest = json.loads(args.combined_manifest.resolve().read_text("utf-8"))
    grep_manifest = json.loads(args.grep_manifest.resolve().read_text("utf-8"))
    for name, manifest in (("combined", combined_manifest), ("grep", grep_manifest)):
        if manifest.get("alignment_sha256") != alignment_sha256:
            raise ValueError(f"{name} feature alignment hash changed")
    combined = np.load(args.combined_features.resolve())["combined"].astype(
        np.float32, copy=False
    )
    grep_matrix = np.load(args.grep_features.resolve())["grep"].astype(
        np.float32, copy=False
    )
    if combined.shape[0] != len(records) or grep_matrix.shape[0] != len(records):
        raise ValueError("feature rows do not align with evaluated identity")
    expanded = np.concatenate((combined, grep_matrix), axis=1).astype(
        np.float32, copy=False
    )
    refnos, paper_matrix, row_to_paper = paper_aggregate_features(grep_matrix, records)
    by_refno = rows_by_refno(records)
    paper_target_counts = np.asarray([len(by_refno[refno]) for refno in refnos])

    candidates = fusion_specs()
    candidate_freeze = {
        "schema_version": "r3_grep_pairwise_rank_candidate_freeze_v2",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "primary_paper_reference": PRIMARY_PAPER,
        "supporting_paper_reference": SUPPORTING_PAPER,
        "adaptation": {
            "evidence_only_candidate_representation": "R2E-style target scoring from retrieved evidence features rather than RESULTNO identity",
            "candidate_ranking": "R2E-style comparison of all sibling targets by supporting evidence",
            "global_relations": "recomputed expanded GREP/global baseline",
            "entity_pair_reasoning": "GREP-style positive-minus-negative sibling-target feature differences",
            "heterogeneity_gate": "label-free paper aggregates decide whether rank correction is applied",
            "nested_selection": "candidate selected only by inner grouped OOF of each outer training fold",
        },
        "rank_specs": [asdict(spec) for spec in rank_specs()],
        "gate_specs": [asdict(spec) for spec in gate_specs()],
        "fusion_specs": [asdict(spec) for spec in candidates],
        "inner_splits": INNER_SPLITS,
        "inner_seed": INNER_SEED,
        "outer_folds": all_oof.N_SPLITS,
        "threshold": 0.5,
        "selection_objective": "multi-target accuracy, then overall accuracy, then mixed-label accuracy",
        "coverage": coverage,
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_model_input": False,
            "refno_used_for_group_isolation_and_sibling_reasoning": True,
            "resultno_used_only_for_alignment": True,
        },
        "inputs": {
            "combined_manifest": {
                "path": str(args.combined_manifest.resolve()),
                "sha256": base.file_sha256(args.combined_manifest.resolve()),
            },
            "grep_manifest": {
                "path": str(args.grep_manifest.resolve()),
                "sha256": base.file_sha256(args.grep_manifest.resolve()),
            },
            "fold_manifest": {
                "path": str(args.fold_manifest.resolve()),
                "sha256": base.file_sha256(args.fold_manifest.resolve()),
            },
        },
        "canonical_runtime": runtime,
    }
    base.atomic_json(candidate_path, candidate_freeze)

    labels_by_resultno = all_oof.load_labels(
        args.targets.resolve(), {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    canonical_targets = [canonical_target_identity(record["ROCKNAME"]) for record in records]
    multi_mask = multi_target_mask(records, labelled_records)
    mixed_mask = mixed_label_mask(records, labels)
    paper_labels = np.asarray(
        [int(len({int(labels[index]) for index in by_refno[refno]}) > 1) for refno in refnos],
        dtype=np.int8,
    )

    baseline_oof = np.full(len(records), np.nan, dtype=np.float64)
    primary_oof = np.full(len(records), np.nan, dtype=np.float64)
    rank_oof = np.zeros(len(records), dtype=np.float64)
    gate_oof = np.zeros(len(records), dtype=np.float64)
    selections: list[dict[str, Any]] = []

    rank_by_name = {spec.name: spec for spec in rank_specs()}
    gate_by_name = {spec.name: spec for spec in gate_specs()}
    groups = [record["REFNO"] for record in records]

    for outer_fold, (outer_train, outer_test) in enumerate(folds, start=1):
        train_refnos = {records[int(index)]["REFNO"] for index in outer_train}
        test_refnos = {records[int(index)]["REFNO"] for index in outer_test}
        if train_refnos & test_refnos:
            raise AssertionError("outer fold splits a REFNO")

        baseline_oof[outer_test] = fit_baseline_mean4(
            expanded, labels, outer_train, outer_test, records, canonical_targets
        )

        inner_folds = nested.deterministic_group_folds(
            outer_train,
            [groups[int(index)] for index in outer_train],
            n_splits=INNER_SPLITS,
            seed=INNER_SEED + outer_fold,
        )
        inner_baseline = np.full(len(records), np.nan, dtype=np.float64)
        inner_rank = {
            spec.name: np.zeros(len(records), dtype=np.float64) for spec in rank_specs()
        }
        inner_gate = {
            spec.name: np.zeros(len(records), dtype=np.float64) for spec in gate_specs()
        }

        for inner_train, inner_validation in inner_folds:
            inner_baseline[inner_validation] = fit_baseline_mean4(
                expanded,
                labels,
                inner_train,
                inner_validation,
                records,
                canonical_targets,
            )
            for spec in rank_specs():
                fitted, _ = fit_ranker(spec, combined, labels, inner_train, records)
                raw = fitted.score(combined)
                normalized = normalize_scores_by_paper(raw, records, inner_validation)
                inner_rank[spec.name][inner_validation] = normalized[inner_validation]

            train_papers = paper_training_indices(
                inner_train, refnos, row_to_paper, paper_target_counts
            )
            validation_papers = np.asarray(
                sorted({int(row_to_paper[index]) for index in inner_validation}),
                dtype=np.int64,
            )
            for spec in gate_specs():
                fitted_gate = fit_gate(spec, paper_matrix, paper_labels, train_papers)
                paper_probability = fitted_gate.predict(paper_matrix[validation_papers])
                mapping = dict(zip(validation_papers, paper_probability))
                inner_gate[spec.name][inner_validation] = [
                    mapping[int(row_to_paper[index])] for index in inner_validation
                ]

        if not np.isfinite(inner_baseline[outer_train]).all():
            raise AssertionError("inner baseline OOF is incomplete")
        candidate_probabilities: dict[str, np.ndarray] = {}
        for spec in candidates:
            if spec.rank is None:
                candidate_probabilities[spec.name] = inner_baseline.copy()
            else:
                candidate_probabilities[spec.name] = apply_fusion(
                    inner_baseline,
                    inner_rank[spec.rank],
                    inner_gate[spec.gate],
                    multi_mask,
                    spec,
                )
        selected_spec, inner_metrics = select_fusion(
            candidates,
            labels,
            candidate_probabilities,
            multi_mask,
            mixed_mask,
            outer_train,
        )

        if selected_spec.rank is None:
            primary_oof[outer_test] = baseline_oof[outer_test]
            pair_audit = {"mixed_papers": 0, "unordered_pairs": 0}
        else:
            ranker, pair_audit = fit_ranker(
                rank_by_name[selected_spec.rank], combined, labels, outer_train, records
            )
            raw_rank = ranker.score(combined)
            normalized_rank = normalize_scores_by_paper(raw_rank, records, outer_test)
            train_papers = paper_training_indices(
                outer_train, refnos, row_to_paper, paper_target_counts
            )
            test_papers = np.asarray(
                sorted({int(row_to_paper[index]) for index in outer_test}), dtype=np.int64
            )
            gate = fit_gate(
                gate_by_name[selected_spec.gate], paper_matrix, paper_labels, train_papers
            )
            test_paper_probability = gate.predict(paper_matrix[test_papers])
            mapping = dict(zip(test_papers, test_paper_probability))
            row_gate = np.zeros(len(records), dtype=np.float64)
            row_gate[outer_test] = [
                mapping[int(row_to_paper[index])] for index in outer_test
            ]
            primary = apply_fusion(
                baseline_oof,
                normalized_rank,
                row_gate,
                multi_mask,
                selected_spec,
            )
            primary_oof[outer_test] = primary[outer_test]
            rank_oof[outer_test] = normalized_rank[outer_test]
            gate_oof[outer_test] = row_gate[outer_test]

        selections.append(
            {
                "outer_fold": outer_fold,
                "selected": asdict(selected_spec),
                "inner_selection_metrics": inner_metrics,
                "outer_train_mixed_label_papers": int(
                    len({records[int(index)]["REFNO"] for index in outer_train if mixed_mask[index]})
                ),
                "pair_training_audit": pair_audit,
            }
        )
        print(
            f"pairwise outer fold {outer_fold}: selected={selected_spec.name}, "
            f"train_rows={len(outer_train)}, test_rows={len(outer_test)}",
            flush=True,
        )

    if not np.isfinite(baseline_oof).all() or not np.isfinite(primary_oof).all():
        raise AssertionError("outer OOF predictions are incomplete")
    arrays = {
        BASELINE_STRATEGY: baseline_oof,
        PRIMARY_STRATEGY: primary_oof,
        "selected_pairwise_rank_score": rank_oof,
        "selected_heterogeneity_gate_probability": gate_oof,
    }
    grep_oof.atomic_npz(oof_path, **arrays)

    receipt = {
        "schema_version": "r3_grep_pairwise_rank_prediction_receipt_v2",
        "status": "NESTED_OOF_PREDICTIONS_FROZEN_BEFORE_AGGREGATE_SCORING",
        "primary_strategy": PRIMARY_STRATEGY,
        "candidate_freeze_sha256": base.file_sha256(candidate_path),
        "oof": {"path": str(oof_path), "sha256": base.file_sha256(oof_path)},
        "each_row_predicted_by_model_excluding_its_refno": True,
        "each_outer_candidate_selected_without_outer_test_labels": True,
        "rows": [
            {
                "row_index": index,
                **record,
                "fold": int(fold_ids[index]),
                "baseline_probability": float(baseline_oof[index]),
                "probability": float(primary_oof[index]),
                "prediction": int(primary_oof[index] >= 0.5),
            }
            for index, record in enumerate(records)
        ],
    }
    base.atomic_json(receipt_path, receipt)

    masks = grep_oof.segment_masks(records, labelled_records, labels)
    baseline_report = grep_oof.strategy_report(
        records, labels, baseline_oof, masks
    )
    primary_report = grep_oof.strategy_report(records, labels, primary_oof, masks)

    prior_difference: float | None = None
    if PRIOR_GREP_OOF.exists():
        prior = np.load(PRIOR_GREP_OOF)["grep_expanded_mean4"].astype(np.float64)
        prior_difference = float(np.max(np.abs(prior - baseline_oof)))
    selected_counts = Counter(item["selected"]["name"] for item in selections)
    report = {
        "schema_version": "r3_grep_pairwise_rank_nested_grouped_oof_v2",
        "status": "complete_development_nested_oof",
        "primary_paper_reference": PRIMARY_PAPER,
        "supporting_paper_reference": SUPPORTING_PAPER,
        "primary_strategy": PRIMARY_STRATEGY,
        "claim_boundary": {
            "internal_refno_grouped_nested_oof": True,
            "independent_external_validation": False,
            "candidate_family_informed_by_prior_development_analysis": True,
            "feature_and_candidate_artifacts_frozen_before_label_access": True,
            "outer_test_labels_excluded_from_training_and_selection": True,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_oof_or_evaluation_used_as_model_input": False,
            "paper_method_adapted_not_architecture_reproduced": True,
        },
        "baseline_reproduction": {
            "prior_oof_path": str(PRIOR_GREP_OOF),
            "maximum_absolute_probability_difference": prior_difference,
            "within_numeric_tolerance_1e_7": (
                prior_difference is not None and prior_difference <= 1e-7
            ),
        },
        "nested_selection": {
            "selected_candidate_counts": dict(selected_counts),
            "outer_folds": selections,
        },
        "strategies": {
            BASELINE_STRATEGY: baseline_report,
            PRIMARY_STRATEGY: primary_report,
        },
        "delta_vs_recomputed_expanded_baseline": {
            "overall_accuracy": round(
                primary_report["overall"]["accuracy"]
                - baseline_report["overall"]["accuracy"],
                8,
            ),
            "multi_target_accuracy": round(
                primary_report["segments"]["multi_target"]["accuracy"]
                - baseline_report["segments"]["multi_target"]["accuracy"],
                8,
            ),
            "mixed_label_accuracy": round(
                primary_report["segments"]["multi_target_mixed_labels"]["accuracy"]
                - baseline_report["segments"]["multi_target_mixed_labels"]["accuracy"],
                8,
            ),
            "uniform_multi_target_accuracy": round(
                primary_report["segments"]["multi_target_uniform_labels"]["accuracy"]
                - baseline_report["segments"]["multi_target_uniform_labels"]["accuracy"],
                8,
            ),
        },
        "paired_refno_cluster_bootstrap": {
            "overall": clustered_bootstrap_delta(
                records,
                labels,
                baseline_oof,
                primary_oof,
                np.ones(len(records), dtype=bool),
            ),
            "multi_target": clustered_bootstrap_delta(
                records, labels, baseline_oof, primary_oof, masks["multi_target"]
            ),
            "multi_target_mixed_labels": clustered_bootstrap_delta(
                records,
                labels,
                baseline_oof,
                primary_oof,
                masks["multi_target_mixed_labels"],
            ),
        },
        "promotion_decision": {
            "promote_to_current_supervised_evaluation": bool(
                primary_report["overall"]["accuracy"] > baseline_report["overall"]["accuracy"]
                and primary_report["segments"]["multi_target"]["accuracy"]
                > baseline_report["segments"]["multi_target"]["accuracy"]
            ),
            "promote_to_strict_production_scoring": False,
            "rule": "supervised evaluation promotion requires strict nested-OOF improvement in overall and multi-target accuracy; strict production remains label-free",
        },
        "artifacts": {
            "candidate_freeze": {
                "path": str(candidate_path),
                "sha256": base.file_sha256(candidate_path),
            },
            "prediction_receipt": {
                "path": str(receipt_path),
                "sha256": base.file_sha256(receipt_path),
            },
            "oof_probabilities": {
                "path": str(oof_path),
                "sha256": base.file_sha256(oof_path),
            },
        },
    }
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "baseline": baseline_report["overall"],
                "primary": primary_report["overall"],
                "primary_multi_target": primary_report["segments"]["multi_target"],
                "primary_mixed_label": primary_report["segments"]["multi_target_mixed_labels"],
                "delta": report["delta_vs_recomputed_expanded_baseline"],
                "promotion": report["promotion_decision"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
