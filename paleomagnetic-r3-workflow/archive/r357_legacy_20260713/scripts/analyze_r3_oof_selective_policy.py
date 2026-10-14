#!/usr/bin/env python3
"""Analyze a selective R3 policy from frozen grouped-OOF probabilities.

This is a development evaluator, not a runtime scorer. It may read labels only
for the fixed training partition and never opens PMAGRESULT, ROCKMAG, held-out
labels, or the guarded holdout predictions. A high known accuracy is always
reported together with its automatic-decision coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

try:
    from .experiment_r3_grouped_oof_text_model import (
        DEFAULT_CHECKPOINT,
        DEFAULT_IDENTITY,
        DEFAULT_OUTPUT as DEFAULT_AGGREGATE,
        DEFAULT_TARGETS,
        file_sha256,
        load_training_identity,
        load_training_labels,
    )
except ImportError:
    from experiment_r3_grouped_oof_text_model import (
        DEFAULT_CHECKPOINT,
        DEFAULT_IDENTITY,
        DEFAULT_OUTPUT as DEFAULT_AGGREGATE,
        DEFAULT_TARGETS,
        file_sha256,
        load_training_identity,
        load_training_labels,
    )


DEFAULT_OUTPUT = DEFAULT_AGGREGATE.with_name("selective_policy_analysis.json")
DEFAULT_THRESHOLDS = (
    0.500,
    0.600,
    0.700,
    0.750,
    0.775,
    0.800,
    0.813,
    0.825,
    0.850,
    0.875,
    0.900,
    0.925,
)
DEFAULT_NEGATIVE_THRESHOLDS = tuple(round(value * 0.025, 3) for value in range(2, 21))
DEFAULT_POSITIVE_THRESHOLDS = tuple(round(value * 0.025, 3) for value in range(20, 39))


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def selective_metric(
    labels: np.ndarray,
    probabilities: np.ndarray,
    negative_threshold: float,
    positive_threshold: float,
) -> dict[str, Any]:
    if not 0.0 <= negative_threshold <= 0.5 <= positive_threshold <= 1.0:
        raise ValueError("selective thresholds must straddle 0.5")
    known = (probabilities <= negative_threshold) | (
        probabilities >= positive_threshold
    )
    predictions = (probabilities >= positive_threshold).astype(np.int8)
    if not known.any():
        return {
            "negative_threshold": negative_threshold,
            "positive_threshold": positive_threshold,
            "known_rows": 0,
            "coverage": 0.0,
            "known_accuracy": None,
            "macro_f1": None,
            "f1_0": None,
            "f1_1": None,
            "confusion": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        }
    tn, fp, fn, tp = confusion_matrix(
        labels[known], predictions[known], labels=[0, 1]
    ).ravel()
    f1_0, f1_1 = f1_score(
        labels[known],
        predictions[known],
        labels=[0, 1],
        average=None,
        zero_division=0,
    )
    return {
        "negative_threshold": negative_threshold,
        "positive_threshold": positive_threshold,
        "known_rows": int(known.sum()),
        "coverage": round(float(known.mean()), 8),
        "known_accuracy": round(float((predictions[known] == labels[known]).mean()), 8),
        "macro_f1": round(float((f1_0 + f1_1) / 2.0), 8),
        "f1_0": round(float(f1_0), 8),
        "f1_1": round(float(f1_1), 8),
        "confusion": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=DEFAULT_AGGREGATE)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-known-accuracy", type=float, default=0.90)
    parser.add_argument("--minimum-class-f1", type=float, default=0.50)
    args = parser.parse_args(argv)
    if not 0.5 < args.target_known_accuracy <= 1.0:
        raise SystemExit("target-known-accuracy must be in (0.5, 1.0]")
    if not 0.0 <= args.minimum_class_f1 <= 1.0:
        raise SystemExit("minimum-class-f1 must be in [0.0, 1.0]")

    aggregate = _load_json(args.aggregate.resolve())
    best = aggregate.get("highest_accuracy")
    if not isinstance(best, Mapping) or best.get("candidate") != "nonlinear_knowledge_fusion":
        raise ValueError("aggregate does not contain the expected nonlinear OOF fusion")
    weights = best.get("weights")
    if not isinstance(weights, Mapping) or not weights:
        raise ValueError("aggregate highest_accuracy.weights is missing")

    identity = load_training_identity(args.identity.resolve())
    labels_by_resultno = load_training_labels(
        args.targets.resolve(),
        {record["RESULTNO"] for record in identity},
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in identity],
        dtype=np.int8,
    )
    groups = np.asarray([record["REFNO"] for record in identity], dtype=object)
    checkpoint = np.load(args.checkpoint.resolve())
    probabilities = np.zeros(len(labels), dtype=np.float64)
    for name, raw_weight in weights.items():
        if name not in checkpoint.files:
            raise ValueError(f"checkpoint is missing OOF probability array {name}")
        array = checkpoint[name]
        if array.shape != labels.shape or not np.isfinite(array).all():
            raise ValueError(f"invalid OOF probability array {name}")
        probabilities += float(raw_weight) * array
    if not np.isclose(sum(float(value) for value in weights.values()), 1.0):
        raise ValueError("OOF fusion weights must sum to one")

    symmetric_curve = [
        selective_metric(labels, probabilities, 1.0 - threshold, threshold)
        for threshold in DEFAULT_THRESHOLDS
    ]
    asymmetric_grid = [
        selective_metric(labels, probabilities, negative, positive)
        for negative in DEFAULT_NEGATIVE_THRESHOLDS
        for positive in DEFAULT_POSITIVE_THRESHOLDS
    ]
    eligible = [
        metric
        for metric in asymmetric_grid
        if metric["known_accuracy"] is not None
        and metric["known_accuracy"] >= args.target_known_accuracy
        and metric["f1_0"] >= args.minimum_class_f1
        and metric["f1_1"] >= args.minimum_class_f1
    ]
    selected = max(eligible, key=lambda metric: metric["coverage"]) if eligible else None
    selected_folds: list[dict[str, Any]] = []
    if selected is not None:
        for fold, (_, test_index) in enumerate(
            GroupKFold(n_splits=5).split(
                np.zeros(len(labels)),
                labels,
                groups,
            ),
            start=1,
        ):
            metric = selective_metric(
                labels[test_index],
                probabilities[test_index],
                selected["negative_threshold"],
                selected["positive_threshold"],
            )
            metric["fold"] = fold
            selected_folds.append(metric)
    result = {
        "schema_version": "r3_training_oof_selective_policy_analysis_v1",
        "status": "complete",
        "scope": "development-only fixed training partition",
        "target_known_accuracy": args.target_known_accuracy,
        "minimum_class_f1": args.minimum_class_f1,
        "probability_source": {
            "candidate": best["candidate"],
            "weights": dict(weights),
            "selection_warning": (
                "Base-model weights and the displayed confidence threshold were selected "
                "post-hoc on grouped OOF training predictions. A new external REFNO set is "
                "required before deployment."
            ),
        },
        "selected_operating_point": selected,
        "selected_operating_point_folds": selected_folds,
        "symmetric_curve": symmetric_curve,
        "asymmetric_grid": {
            "negative_thresholds": list(DEFAULT_NEGATIVE_THRESHOLDS),
            "positive_thresholds": list(DEFAULT_POSITIVE_THRESHOLDS),
            "candidate_count": len(asymmetric_grid),
            "selection": (
                "maximum coverage subject to target known accuracy and both class-F1 floors"
            ),
        },
        "interpretation": {
            "full_coverage_target_reached": bool(
                symmetric_curve[0]["known_accuracy"] >= args.target_known_accuracy
            ),
            "selective_target_reached_in_development": selected is not None,
            "selective_target_reached_in_every_fold": bool(
                selected_folds
                and all(
                    metric["known_accuracy"] >= args.target_known_accuracy
                    for metric in selected_folds
                )
            ),
            "abstained_rows_at_selected_point": (
                len(labels) - selected["known_rows"] if selected is not None else len(labels)
            ),
            "formal_scorer_modified": False,
            "external_validation_required": True,
        },
        "runtime_recomputable_feature_sources": [
            "ROCKNAME",
            "original-paper text",
            "frozen R3 taxonomy",
        ],
        "forbidden_sources_not_read": [
            "held-out R3 labels",
            "PMAGRESULT",
            "ROCKMAG",
            "guarded holdout predictions",
        ],
        "inputs": {
            "aggregate": {
                "path": str(args.aggregate.resolve()),
                "sha256": file_sha256(args.aggregate.resolve()),
            },
            "checkpoint": {
                "path": str(args.checkpoint.resolve()),
                "sha256": file_sha256(args.checkpoint.resolve()),
            },
            "training_identity": {
                "path": str(args.identity.resolve()),
                "sha256": file_sha256(args.identity.resolve()),
            },
            "training_labels": {
                "path": str(args.targets.resolve()),
                "sha256": file_sha256(args.targets.resolve()),
                "held_out_cells_dereferenced": False,
            },
        },
        "artifacts": {
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": file_sha256(Path(__file__).resolve()),
            "output": str(args.output.resolve()),
        },
        "analysis_sha256": hashlib.sha256(probabilities.tobytes()).hexdigest(),
    }
    _atomic_json(args.output.resolve(), result)
    print(json.dumps(result["selected_operating_point"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
