#!/usr/bin/env python3
"""Train and audit the frozen R5 original-paper Bernoulli calibration.

Historical labels are used only by this offline tool. Papers are isolated by
REFNO, target terms are masked, and the score threshold is selected from
five-fold out-of-fold percentiles without consulting the held-out test split.
The frozen runtime artifact contains no REFNO/RESULTNO answer lookup.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

try:
    from .paper_text_calibration import (
        bernoulli_features,
        build_calibration_input,
        clean,
    )
except ImportError:
    from paper_text_calibration import (
        bernoulli_features,
        build_calibration_input,
        clean,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_PAGE_CACHE = ROOT / "outputs/r357_latest_full_07111019/r3_page_source_cache.jsonl"
DEFAULT_OUTPUT = ROOT / "data/calibration/r5_paper_text_model_v1.json"


def stable_split(refno: str) -> str:
    value = int(hashlib.sha256(refno.encode("utf-8")).hexdigest()[:12], 16)
    fraction = value / float(16**12 - 1)
    if fraction < 0.20:
        return "test"
    if fraction < 0.35:
        return "dev"
    return "train"


def oof_fold(refno: str, folds: int = 5) -> int:
    value = hashlib.sha256(f"oof:{refno}".encode("utf-8")).hexdigest()[:8]
    return int(value, 16) % folds


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_page_cache(path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            refno = clean(row.get("REFNO"))
            if refno:
                cache[refno] = row
    return cache


def build_records(targets_path: Path, page_cache_path: Path) -> list[dict[str, Any]]:
    frame = pd.read_excel(
        targets_path,
        usecols=["REFNO", "RESULTNO", "ROCKNAME", "R5"],
        dtype={"REFNO": str, "RESULTNO": str, "ROCKNAME": str},
    ).fillna("")
    cache = load_page_cache(page_cache_path)
    records: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        refno = clean(row["REFNO"])
        target = clean(row["ROCKNAME"])
        source = build_calibration_input(cache.get(refno, {"pages": []}), target)
        records.append(
            {
                "REFNO": refno,
                "RESULTNO": clean(row["RESULTNO"]),
                "TARGET": target,
                "split": stable_split(refno),
                "oof_fold": oof_fold(refno),
                "gold": int(float(row["R5"])),
                "source_available": bool(source["source_available"]),
                "features": bernoulli_features(source["text"], source["pattern_flags"]),
            }
        )
    return records


def fit_bernoulli(
    records: list[dict[str, Any]],
    *,
    min_document_frequency: int,
    max_features: int,
    alpha: float,
) -> dict[str, Any]:
    n0 = sum(record["gold"] == 0 for record in records)
    n1 = sum(record["gold"] == 1 for record in records)
    if not n0 or not n1:
        raise ValueError("Bernoulli training requires both R5 classes")
    df0: Counter[str] = Counter()
    df1: Counter[str] = Counter()
    for record in records:
        (df0 if record["gold"] == 0 else df1).update(record["features"])
    candidates: list[tuple[float, str, float, float, float]] = []
    for feature in set(df0) | set(df1):
        total = df0[feature] + df1[feature]
        if total < min_document_frequency or total > 0.96 * len(records):
            continue
        p0 = (df0[feature] + alpha) / (n0 + 2.0 * alpha)
        p1 = (df1[feature] + alpha) / (n1 + 2.0 * alpha)
        weight = math.log(p0 / (1.0 - p0)) - math.log(p1 / (1.0 - p1))
        candidates.append((abs(weight), feature, weight, p0, p1))
    candidates.sort(reverse=True)
    selected = candidates[:max_features]
    prior0 = (n0 + alpha) / (len(records) + 2.0 * alpha)
    prior1 = 1.0 - prior0
    base = math.log(prior0 / prior1)
    for _, _, _, p0, p1 in selected:
        base += math.log((1.0 - p0) / (1.0 - p1))
    return {
        "base": base,
        "weights": {feature: weight for _, feature, weight, _, _ in selected},
        "rows": len(records),
        "class_zero": n0,
        "class_one": n1,
    }


def risk_score(model: dict[str, Any], features: Iterable[str]) -> float:
    weights = model["weights"]
    return float(model["base"] + sum(weights.get(feature, 0.0) for feature in features))


def metrics(gold: list[int], predictions: list[int]) -> dict[str, Any]:
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in zip(gold, predictions))
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in zip(gold, predictions))
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in zip(gold, predictions))
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in zip(gold, predictions))

    def f1(label: int) -> float:
        correct = sum(actual == label and predicted == label for actual, predicted in zip(gold, predictions))
        false_positive = sum(actual != label and predicted == label for actual, predicted in zip(gold, predictions))
        false_negative = sum(actual == label and predicted != label for actual, predicted in zip(gold, predictions))
        denominator = 2 * correct + false_positive + false_negative
        return 2 * correct / denominator if denominator else 0.0

    return {
        "rows": len(gold),
        "accuracy": (tp + tn) / len(gold) if gold else None,
        "macro_f1": (f1(0) + f1(1)) / 2.0 if gold else None,
        "f1_0": f1(0) if gold else None,
        "f1_1": f1(1) if gold else None,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "predicted_zero": sum(predicted == 0 for predicted in predictions),
    }


def select_percentile(percentiles: list[float], gold: list[int]) -> dict[str, Any]:
    candidates = [max(percentiles) + 1.0] + sorted(set(percentiles))
    best: tuple[tuple[float, float, float], float, dict[str, Any]] | None = None
    for threshold in candidates:
        predictions = [0 if score >= threshold else 1 for score in percentiles]
        result = metrics(gold, predictions)
        key = (
            float(result["accuracy"]),
            float(result["f1_0"]),
            float(result["macro_f1"]),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), result)
    assert best is not None
    return {"percentile_threshold": best[1], **best[2]}


def oof_calibration(
    records: list[dict[str, Any]],
    *,
    folds: int,
    min_document_frequency: int,
    max_features: int,
    alpha: float,
) -> dict[str, Any]:
    percentiles: dict[int, float] = {}
    fold_audit: list[dict[str, Any]] = []
    for fold in range(folds):
        train = [record for record in records if record["oof_fold"] != fold]
        validation = [record for record in records if record["oof_fold"] == fold]
        model = fit_bernoulli(
            train,
            min_document_frequency=min_document_frequency,
            max_features=max_features,
            alpha=alpha,
        )
        training_scores = sorted(risk_score(model, record["features"]) for record in train)
        for record in validation:
            score = risk_score(model, record["features"])
            percentiles[id(record)] = bisect_right(training_scores, score) / len(training_scores)
        fold_audit.append(
            {
                "fold": fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_refnos": len({record["REFNO"] for record in train}),
                "validation_refnos": len({record["REFNO"] for record in validation}),
            }
        )
    ordered = [percentiles[id(record)] for record in records]
    selection = select_percentile(ordered, [record["gold"] for record in records])
    return {"selection": selection, "folds": fold_audit, "percentiles": ordered}


def evaluate(
    records: list[dict[str, Any]],
    model: dict[str, Any],
    raw_threshold: float,
) -> dict[str, Any]:
    scores = [risk_score(model, record["features"]) for record in records]
    model_predictions = [0 if score >= raw_threshold else 1 for score in scores]
    deployed = [
        prediction if record["source_available"] else 1
        for record, prediction in zip(records, model_predictions)
    ]
    available = [index for index, record in enumerate(records) if record["source_available"]]
    gold = [record["gold"] for record in records]
    return {
        "deployed_missing_source_defaults_to_one": metrics(gold, deployed),
        "model_all_rows_for_audit_only": metrics(gold, model_predictions),
        "source_available_only": metrics(
            [gold[index] for index in available],
            [model_predictions[index] for index in available],
        ),
        "source_coverage": len(available) / len(records) if records else 0.0,
        "unknown_rows": len(records) - len(available),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oof-folds", type=int, default=5)
    parser.add_argument("--min-document-frequency", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=8000)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()

    records = build_records(args.targets, args.page_cache)
    calibration = [record for record in records if record["split"] != "test"]
    test = [record for record in records if record["split"] == "test"]
    oof = oof_calibration(
        calibration,
        folds=args.oof_folds,
        min_document_frequency=args.min_document_frequency,
        max_features=args.max_features,
        alpha=args.alpha,
    )
    model = fit_bernoulli(
        calibration,
        min_document_frequency=args.min_document_frequency,
        max_features=args.max_features,
        alpha=args.alpha,
    )
    training_scores = sorted(risk_score(model, record["features"]) for record in calibration)
    percentile_threshold = float(oof["selection"]["percentile_threshold"])
    rank = max(1, min(len(training_scores), math.ceil(percentile_threshold * len(training_scores))))
    raw_threshold = float(training_scores[rank - 1])

    partitions = {
        "calibration": calibration,
        "held_out_test": test,
        "all": records,
    }
    evaluation = {
        name: evaluate(partition, model, raw_threshold)
        for name, partition in partitions.items()
    }
    weighted_features = sorted(model["weights"].items(), key=lambda item: item[1], reverse=True)
    artifact = {
        "version": "r5_paper_text_model_v1",
        "description": "Target-masked Bernoulli calibration over original-paper evidence windows.",
        "models": {
            "R5": {
                "scoring": "bernoulli_presence",
                "base": round(float(model["base"]), 12),
                "bias": round(float(model["base"]), 12),
                "threshold": round(raw_threshold, 12),
                "percentile_threshold": percentile_threshold,
                "feature_parameters": [
                    [feature, round(float(weight), 12)]
                    for feature, weight in model["weights"].items()
                ],
            }
        },
        "audit": {
            "recommendation": (
                "Paper-only shadow/AB candidate. Held-out accuracy improvement is small; "
                "promote only after a new external REFNO test set confirms it."
            ),
            "runtime_input_columns": ["REFNO", "RESULTNO", "ROCKNAME"],
            "runtime_scoring_sources": [
                "identity manifest for source lookup and target masking",
                "target-centered original paper page text",
                "this frozen calibration artifact",
            ],
            "forbidden_runtime_scoring_sources": [
                "PMAGRESULT",
                "ROCKUNIT",
                "R3/R5/R7 labels",
                "previous prediction workbooks",
                "REFNO/RESULTNO answer lookup",
            ],
            "feature_builder": {
                "target_name_feature": "masked as TARGETTERM",
                "dropped_document_lines": 2,
                "token_pattern": "[a-z][a-z0-9'-]{2,}",
                "features": ["binary word", "binary adjacent bigram", "generic R5 pattern flag"],
                "explicit_identity_features": False,
            },
            "calibration_only_sources": [str(args.targets), str(args.page_cache)],
            "targets_sha256": file_sha256(args.targets),
            "page_cache_sha256": file_sha256(args.page_cache),
            "split": "SHA256(REFNO): held-out test=20%; remaining 80% used for OOF calibration",
            "oof_fold": "int(SHA256('oof:' + REFNO)[:8], 16) % 5",
            "held_out_test_not_used_for_feature_or_threshold_selection": True,
            "hyperparameters": {
                "alpha": args.alpha,
                "min_document_frequency": args.min_document_frequency,
                "max_features": args.max_features,
                "oof_folds": args.oof_folds,
            },
            "rows": {
                "calibration": len(calibration),
                "calibration_refnos": len({record["REFNO"] for record in calibration}),
                "held_out_test": len(test),
                "held_out_test_refnos": len({record["REFNO"] for record in test}),
                "all": len(records),
            },
            "class_counts": {
                "calibration_zero": model["class_zero"],
                "calibration_one": model["class_one"],
            },
            "oof_selection": oof["selection"],
            "oof_folds": oof["folds"],
            "raw_threshold_rank_one_based": rank,
            "training_score_distribution": [round(float(score), 10) for score in training_scores],
            "evaluation": evaluation,
            "top_risk_features": weighted_features[:40],
            "top_support_features": weighted_features[-40:][::-1],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "features": len(model["weights"]),
                "base": model["base"],
                "percentile_threshold": percentile_threshold,
                "raw_threshold": raw_threshold,
                "held_out_test": evaluation["held_out_test"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
