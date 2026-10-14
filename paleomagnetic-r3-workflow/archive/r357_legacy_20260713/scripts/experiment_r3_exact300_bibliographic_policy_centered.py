#!/usr/bin/env python3
"""Bounded centered-log-odds follow-up for bibliographic R3 policy effects.

This v2 reuses the frozen original-text profiles and semantic matrix builder
from the v1 experiment.  It converts each empirical-Bayes posterior into an
effect relative to the allowed outer-training prior, then applies that effect
to the semantic logit.  Lambda and an optional confidence gate are selected
only from four REFNO-isolated inner OOF folds.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_bibliographic_policy as v1  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/bibliographic_policy_centered_v2"
V1_CANDIDATE_PROTOCOL = v1.candidate_protocol
LAMBDAS = (0.25, 0.50, 1.00)
SUPPORT_GATES = (0.0, 0.5)
EPSILON = 1e-6


@dataclass(frozen=True)
class CenteredCandidateSpec:
    name: str
    smoothing: float | None
    fusion_weight: float
    support_gate: float


def candidate_specs() -> tuple[CenteredCandidateSpec, ...]:
    return (
        CenteredCandidateSpec("semantic_only", None, 0.0, 0.0),
        *(
            CenteredCandidateSpec(
                (
                    f"centered_a{int(alpha)}_lambda{int(weight * 100):03d}"
                    f"_gate{int(gate * 100):02d}"
                ),
                alpha,
                weight,
                gate,
            )
            for alpha in v1.SMOOTHING_STRENGTHS
            for weight in LAMBDAS
            for gate in SUPPORT_GATES
        ),
    )


def logit(values: np.ndarray | float) -> np.ndarray:
    array = np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(array / (1.0 - array))


def sigmoid(values: np.ndarray | float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(array, -40.0, 40.0)))


def centered_effect_probability(
    posterior: np.ndarray,
    prior: float,
) -> np.ndarray:
    """Encode a zero-centered log-odds effect as a probability around 0.5."""

    return sigmoid(logit(posterior) - float(logit(prior)))


def transform_centered_policy_rows(
    encoder: v1.PolicyEncoder,
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    profiles: Mapping[str, Any],
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    posterior, confidence, audit = v1.transform_policy_rows(
        encoder, indices, records, profiles, smoothing
    )
    centered = centered_effect_probability(posterior, encoder.prior)
    output_audit = {
        **audit,
        "outer_allowed_training_prior": round(float(encoder.prior), 8),
        "calibration": "logit(posterior) - logit(training-paper prior)",
        "neutral_effect_probability": 0.5,
        "unsupported_rows_have_zero_confidence": bool(
            np.all(confidence[confidence == 0] == 0)
        ),
    }
    return centered, confidence, output_audit


def centered_fuse_probability(
    semantic: np.ndarray,
    centered_policy: np.ndarray,
    confidence: np.ndarray,
    weight: float,
    support_gate: float,
) -> np.ndarray:
    effective_confidence = np.asarray(confidence, dtype=np.float64).copy()
    if support_gate > 0:
        effective_confidence[effective_confidence < support_gate] = 0.0
    effect = logit(centered_policy)
    return sigmoid(logit(semantic) + float(weight) * effective_confidence * effect)


def select_candidate(
    labels: np.ndarray,
    semantic: np.ndarray,
    policy_by_alpha: Mapping[float, np.ndarray],
    confidence_by_alpha: Mapping[float, np.ndarray],
) -> tuple[CenteredCandidateSpec, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    specs = candidate_specs()
    for order, spec in enumerate(specs):
        probability = (
            semantic
            if spec.smoothing is None
            else centered_fuse_probability(
                semantic,
                policy_by_alpha[spec.smoothing],
                confidence_by_alpha[spec.smoothing],
                spec.fusion_weight,
                spec.support_gate,
            )
        )
        metric = v1.base.metric_record(labels, probability, threshold=0.5)
        records.append(
            {
                "candidate": spec.name,
                "candidate_order": order,
                "smoothing": spec.smoothing,
                "lambda": spec.fusion_weight,
                "support_gate": spec.support_gate,
                "metric": metric,
            }
        )
    selected = max(
        records,
        key=lambda item: (
            item["metric"]["accuracy"],
            item["metric"]["macro_f1"],
            item["metric"]["balanced_accuracy"],
            v1.specificity(item["metric"]),
            item["support_gate"],
            -item["lambda"],
            -item["candidate_order"],
        ),
    )
    return next(spec for spec in specs if spec.name == selected["candidate"]), records


def run_fold(
    fold: int,
    *,
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    fold_ids: np.ndarray,
    semantic_matrix: np.ndarray,
    canonical_targets: Sequence[str],
    profiles: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Produce one centered outer fold without accepting exact labels."""

    exact_test = np.flatnonzero(fold_ids == fold).astype(np.int64)
    full_lookup = {
        v1.auxiliary.row_key(record): index
        for index, record in enumerate(full_records)
    }
    outer_train, test_full, test_refnos = v1.auxiliary.auxiliary_split(
        full_records, exact_records, exact_test, full_lookup
    )
    inner_ids = v1.balanced_group_fold_ids(full_records, outer_train)
    inner_semantic = np.full(len(outer_train), np.nan, dtype=np.float64)
    inner_policy = {
        alpha: np.full(len(outer_train), np.nan, dtype=np.float64)
        for alpha in v1.SMOOTHING_STRENGTHS
    }
    inner_confidence = {
        alpha: np.full(len(outer_train), np.nan, dtype=np.float64)
        for alpha in v1.SMOOTHING_STRENGTHS
    }
    inner_audits: list[dict[str, Any]] = []
    for inner_fold in range(v1.INNER_FOLDS):
        fit_position = np.flatnonzero(inner_ids != inner_fold)
        validation_position = np.flatnonzero(inner_ids == inner_fold)
        fit_indices = outer_train[fit_position]
        validation_indices = outer_train[validation_position]
        fit_refnos = {full_records[int(index)]["REFNO"] for index in fit_indices}
        validation_refnos = {
            full_records[int(index)]["REFNO"] for index in validation_indices
        }
        if (
            fit_refnos & validation_refnos
            or fit_refnos & test_refnos
            or validation_refnos & test_refnos
        ):
            raise AssertionError("REFNO leaked into centered policy inner selection")
        semantic_model = v1.fit_semantic_model(
            semantic_matrix,
            full_labels,
            fit_indices,
            full_records,
            canonical_targets,
        )
        inner_semantic[validation_position] = semantic_model.predict_proba(
            semantic_matrix[validation_indices]
        )[:, 1]
        encoder = v1.fit_policy_encoder(
            sorted(fit_refnos),
            v1.paper_label_map(fit_indices, full_records, full_labels),
            profiles,
        )
        alpha_audits: dict[str, Any] = {}
        for alpha in v1.SMOOTHING_STRENGTHS:
            probability, confidence, transform_audit = transform_centered_policy_rows(
                encoder,
                validation_indices,
                full_records,
                profiles,
                alpha,
            )
            inner_policy[alpha][validation_position] = probability
            inner_confidence[alpha][validation_position] = confidence
            alpha_audits[f"alpha_{int(alpha)}"] = transform_audit
        inner_audits.append(
            {
                "inner_fold": inner_fold + 1,
                "fit_rows": int(len(fit_indices)),
                "fit_papers": len(fit_refnos),
                "validation_rows": int(len(validation_indices)),
                "validation_papers": len(validation_refnos),
                "fit_validation_refno_overlap": len(
                    fit_refnos & validation_refnos
                ),
                "outer_test_refno_overlap": len(
                    (fit_refnos | validation_refnos) & test_refnos
                ),
                "policy_transform": alpha_audits,
            }
        )
    if not np.isfinite(inner_semantic).all() or any(
        not np.isfinite(values).all()
        for mapping in (inner_policy, inner_confidence)
        for values in mapping.values()
    ):
        raise AssertionError("centered policy inner OOF coverage is incomplete")
    selected, candidate_audit = select_candidate(
        full_labels[outer_train], inner_semantic, inner_policy, inner_confidence
    )

    semantic_model = v1.fit_semantic_model(
        semantic_matrix,
        full_labels,
        outer_train,
        full_records,
        canonical_targets,
    )
    semantic_probability = np.asarray(
        semantic_model.predict_proba(semantic_matrix[test_full])[:, 1],
        dtype=np.float64,
    )
    smoothing = selected.smoothing or v1.SMOOTHING_STRENGTHS[-1]
    train_refnos = {full_records[int(index)]["REFNO"] for index in outer_train}
    encoder = v1.fit_policy_encoder(
        sorted(train_refnos),
        v1.paper_label_map(outer_train, full_records, full_labels),
        profiles,
    )
    policy_probability, policy_confidence, policy_audit = (
        transform_centered_policy_rows(
            encoder, test_full, full_records, profiles, smoothing
        )
    )
    fused_probability = (
        semantic_probability
        if selected.smoothing is None
        else centered_fuse_probability(
            semantic_probability,
            policy_probability,
            policy_confidence,
            selected.fusion_weight,
            selected.support_gate,
        )
    )
    excluded_rows = sum(record["REFNO"] in test_refnos for record in full_records)
    audit = {
        "fold": fold,
        "exact_test_rows": int(len(exact_test)),
        "exact_test_papers": len(test_refnos),
        "outer_train_rows": int(len(outer_train)),
        "outer_train_papers": len(train_refnos),
        "full_rows_excluded_for_test_refnos": excluded_rows,
        "outer_train_test_refno_overlap": len(train_refnos & test_refnos),
        "outer_train_exact_test_row_overlap": len(
            set(int(index) for index in outer_train)
            & set(int(index) for index in test_full)
        ),
        "excluded_test_refno_rows_remaining_in_outer_train": sum(
            full_records[int(index)]["REFNO"] in test_refnos
            for index in outer_train
        ),
        "inner_split": {
            "method": "label-free deterministic row-balanced REFNO assignment",
            "salt": v1.INNER_SPLIT_SALT,
            "folds": v1.INNER_FOLDS,
            "fold_id_sha256": v1.hashlib.sha256(bytes(inner_ids)).hexdigest(),
        },
        "inner_folds": inner_audits,
        "selection_source": "four-fold REFNO-isolated inner OOF only",
        "selection_rule": (
            "accuracy, macro-F1, balanced accuracy, specificity, stronger support "
            "gate, lower lambda, declaration order"
        ),
        "selected_candidate": asdict(selected),
        "candidate_metrics": candidate_audit,
        "outer_policy_transform": policy_audit,
        "outer_test_labels_received_by_run_fold": False,
        "centered_log_odds_calibration": True,
    }
    return exact_test, {
        "bibliographic_policy_fusion": fused_probability,
        "semantic_hgb": semantic_probability,
        "policy_probability": policy_probability,
        "policy_confidence": policy_confidence,
    }, audit


def candidate_protocol() -> dict[str, Any]:
    protocol = V1_CANDIDATE_PROTOCOL()
    protocol["candidates"] = [asdict(spec) for spec in candidate_specs()]
    protocol["centered_calibration"] = {
        "policy_effect": "logit(policy posterior) - logit(allowed training-paper prior)",
        "semantic_update": (
            "semantic_logit + lambda * effective_confidence * policy_effect"
        ),
        "lambdas": list(LAMBDAS),
        "support_gates": list(SUPPORT_GATES),
        "unsupported_policy_effect_applied": False,
        "outer_threshold": 0.5,
    }
    protocol["fusion"] = {
        "rule": (
            "sigmoid(logit(semantic) + lambda * gated_confidence * "
            "(logit(policy)-logit(training prior)))"
        ),
        "lambdas": list(LAMBDAS),
        "support_gates": list(SUPPORT_GATES),
        "threshold": 0.5,
        "outer_label_threshold_selection": False,
    }
    return protocol


def output_dir_from_argv(argv: Sequence[str] | None) -> Path:
    values = list(argv if argv is not None else sys.argv[1:])
    if "--output-dir" in values:
        position = values.index("--output-dir")
        if position + 1 < len(values):
            return Path(values[position + 1]).resolve()
    return OUTPUT_DIR.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    # Install the bounded v2 behavior before v1 constructs or hashes its freeze.
    v1.OUTPUT_DIR = OUTPUT_DIR
    v1.__file__ = __file__
    v1.candidate_specs = candidate_specs
    v1.candidate_protocol = candidate_protocol
    v1.run_fold = run_fold
    result = v1.main(argv)

    # Clarify the shared report structure without changing the already-frozen inputs.
    report_path = output_dir_from_argv(argv) / v1.REPORT_NAME
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["schema_version"] = "r3_exact300_bibliographic_policy_centered_grouped_oof_v2"
    report["calibration"] = candidate_protocol()["centered_calibration"]
    report["primary_full_coverage_result"]["strategy"] = (
        "inner-selected centered-log-odds bibliographic policy correction"
    )
    report["guardrails"]["centered_candidate_protocol_frozen_before_label_access"] = True
    report["guardrails"]["unsupported_policy_effect_applied"] = False
    v1.base.atomic_json(report_path, report)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
