#!/usr/bin/env python3
"""Evaluate literature-only Apple sentence embeddings on grouped R5/R7 splits."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from run_target_text_classifier import choose_threshold, metric  # noqa: E402


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def checked_matmul(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    # Apple Accelerate can emit spurious floating-point warnings for finite,
    # normalized arrays. Validate the result explicitly instead.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = left @ right
    if not np.isfinite(result).all():
        raise ValueError("Embedding matrix multiplication produced non-finite values")
    return result


def knn_risk(query: np.ndarray, train: np.ndarray, train_gold: np.ndarray, k: int) -> np.ndarray:
    similarity = checked_matmul(query, train.T)
    k = min(k, train.shape[0])
    indices = np.argpartition(similarity, -k, axis=1)[:, -k:]
    selected_similarity = np.take_along_axis(similarity, indices, axis=1)
    selected_gold = train_gold[indices]
    weights = np.exp((selected_similarity - selected_similarity.max(axis=1, keepdims=True)) / 0.08)
    zero = (selected_gold == 0).astype(float)
    return (weights * zero).sum(axis=1) / weights.sum(axis=1)


def evaluate_indicator(rows: list[dict[str, Any]], indicator: str) -> dict[str, Any]:
    matrix = normalize(np.asarray([row["embedding"] for row in rows], dtype=float))
    gold = np.asarray([int(row[f"gold_{indicator}"]) for row in rows], dtype=int)
    splits = np.asarray([row["split"] for row in rows])
    train_mask = splits == "train"; dev_mask = splits == "dev"; test_mask = splits == "test"
    train = matrix[train_mask]; train_gold = gold[train_mask]
    centroid_zero = train[train_gold == 0].mean(axis=0)
    centroid_one = train[train_gold == 1].mean(axis=0)
    centroid_zero /= max(np.linalg.norm(centroid_zero), 1e-12)
    centroid_one /= max(np.linalg.norm(centroid_one), 1e-12)

    def scores(query: np.ndarray, k: int) -> dict[str, np.ndarray]:
        diff = checked_matmul(query, centroid_zero) - checked_matmul(query, centroid_one)
        centroid = sigmoid(diff / 0.08)
        knn = knn_risk(query, train, train_gold, k)
        return {"centroid": centroid, f"knn_{k}": knn, f"hybrid_{k}": 0.35 * centroid + 0.65 * knn}

    candidates: dict[str, dict[str, Any]] = {}
    for k in (5, 11, 21, 41):
        dev_scores = scores(matrix[dev_mask], k)
        test_scores = scores(matrix[test_mask], k)
        for name, values in dev_scores.items():
            if name in candidates:
                continue
            threshold = choose_threshold(values.tolist(), gold[dev_mask].tolist())
            precision_threshold = choose_threshold(values.tolist(), gold[dev_mask].tolist(), min_zero_precision=0.75)
            test_values = test_scores[name]
            pred = [0 if score >= threshold["threshold"] else 1 for score in test_values]
            pred_precision = [0 if score >= precision_threshold["threshold"] else 1 for score in test_values]
            candidates[name] = {
                "threshold": threshold,
                "precision_threshold": precision_threshold,
                "test": metric(gold[test_mask].tolist(), pred),
                "test_precision_gate": metric(gold[test_mask].tolist(), pred_precision),
            }

    best_name = max(candidates, key=lambda name: (candidates[name]["threshold"]["accuracy"], candidates[name]["threshold"]["zero_recall"]))
    return {
        "rows": {"train": int(train_mask.sum()), "dev": int(dev_mask.sum()), "test": int(test_mask.sum())},
        "test_all_one": metric(gold[test_mask].tolist(), [1] * int(test_mask.sum())),
        "selected_on_dev": best_name,
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.embeddings)
    summary = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "method": "Apple NaturalLanguage sentence embeddings with prototype and grouped kNN calibration",
        "embedding_dimensions": len(rows[0]["embedding"]) if rows else 0,
        "rows": len(rows),
        "indicators": {indicator: evaluate_indicator(rows, indicator) for indicator in ("R5", "R7")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "embedding_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Literature-Only Embedding Prototype Experiment", "",
        f"Generated: {summary['created_at']}", "",
        "Apple NaturalLanguage English sentence embeddings are averaged over target-centered evidence windows.",
        "The same REFNO never appears in both train and test.", "",
        "| Indicator | Dev-selected method | Test accuracy | Macro-F1 | F1(0) | FP | FN | Pred 0 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for indicator, values in summary["indicators"].items():
        name = values["selected_on_dev"]
        result = values["candidates"][name]["test"]
        lines.append(f"| {indicator} | {name} | {result['accuracy']:.4f} | {result['macro_f1']:.4f} | {result['f1_0']:.4f} | {result['fp']} | {result['fn']} | {result['pred_zero']} |")
    (args.output_dir / "EMBEDDING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
