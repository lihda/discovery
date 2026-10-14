#!/usr/bin/env python3
"""Measure R3 OOF performance by the number and label mix of paper targets."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_all_labelled_pdf_grouped_oof as all_oof  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OOF_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
OUTPUT = OOF_DIR / "target_multiplicity_report.json"


def metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
    refnos: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    selected_labels = labels[mask]
    prediction = probabilities[mask] >= threshold
    tn = int(((selected_labels == 0) & ~prediction).sum())
    fp = int(((selected_labels == 0) & prediction).sum())
    fn = int(((selected_labels == 1) & ~prediction).sum())
    tp = int(((selected_labels == 1) & prediction).sum())
    return {
        "rows": len(selected_labels),
        "paper_groups": len(set(refnos[mask])),
        "positive_labels": int((selected_labels == 1).sum()),
        "negative_labels": int((selected_labels == 0).sum()),
        "accuracy": round(float((selected_labels == prediction).mean()), 8),
        "macro_f1": round(
            float(f1_score(selected_labels, prediction, average="macro")), 8
        ),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(selected_labels, prediction)), 8
        ),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "threshold": threshold,
    }


def main() -> int:
    receipt_path = OOF_DIR / "prediction_receipt.json"
    archive_path = OOF_DIR / "oof_probabilities.npz"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))["rows"]
    probabilities = np.load(archive_path)
    records = [
        {key: str(row[key]) for key in ("REFNO", "RESULTNO", "ROCKNAME")}
        for row in receipt
    ]
    identity = all_oof.load_identity(all_oof.IDENTITY)
    target_counts = Counter(row["REFNO"] for row in identity)
    labels_by_resultno = all_oof.load_labels(
        base.DEFAULT_TARGETS, {row["RESULTNO"] for row in records}
    )
    labels = np.asarray(
        [labels_by_resultno[row["RESULTNO"]] for row in records], dtype=np.int8
    )
    refnos = np.asarray([row["REFNO"] for row in records])
    multiplicity = np.asarray([target_counts[refno] for refno in refnos])
    primary = probabilities[all_oof.PRIMARY_STRATEGY].astype(np.float64)

    row_indices_by_refno: dict[str, list[int]] = defaultdict(list)
    for index, refno in enumerate(refnos):
        row_indices_by_refno[str(refno)].append(index)
    paper_kind: dict[str, str] = {}
    for refno, indices in row_indices_by_refno.items():
        if len(indices) == 1:
            paper_kind[refno] = "single_target"
        elif len({int(labels[index]) for index in indices}) == 1:
            paper_kind[refno] = "multi_target_uniform_labels"
        else:
            paper_kind[refno] = "multi_target_mixed_labels"

    masks = {
        "single_target": multiplicity == 1,
        "multi_target": multiplicity >= 2,
        "two_targets": multiplicity == 2,
        "three_or_four_targets": (multiplicity >= 3) & (multiplicity <= 4),
        "five_or_more_targets": multiplicity >= 5,
        "multi_target_uniform_labels": np.asarray(
            [paper_kind[str(refno)] == "multi_target_uniform_labels" for refno in refnos]
        ),
        "multi_target_mixed_labels": np.asarray(
            [paper_kind[str(refno)] == "multi_target_mixed_labels" for refno in refnos]
        ),
    }
    segments = {
        name: metrics(labels, primary, mask, refnos) for name, mask in masks.items()
    }

    single_model_results = {
        name: metrics(
            labels,
            probabilities[name].astype(np.float64),
            masks["single_target"],
            refnos,
        )
        for name in probabilities.files
    }
    single_labels = labels[masks["single_target"]]
    single_probabilities = primary[masks["single_target"]]
    candidates = np.concatenate(
        (
            [np.nextafter(single_probabilities.min(), -np.inf)],
            np.unique(single_probabilities),
            [np.nextafter(single_probabilities.max(), np.inf)],
        )
    )
    accuracies = np.asarray(
        [
            float((single_labels == (single_probabilities >= threshold)).mean())
            for threshold in candidates
        ]
    )
    best = float(accuracies.max())
    tied = candidates[np.isclose(accuracies, best, rtol=0.0, atol=1e-15)]
    oracle_threshold = float(
        min(tied, key=lambda value: (abs(float(value) - 0.5), float(value)))
    )

    mixed_refnos = [
        refno
        for refno, kind in paper_kind.items()
        if kind == "multi_target_mixed_labels"
    ]
    mixed_exact = 0
    mixed_constant = 0
    for refno in mixed_refnos:
        indices = row_indices_by_refno[refno]
        prediction = (primary[indices] >= 0.5).astype(np.int8)
        mixed_exact += int(np.array_equal(prediction, labels[indices]))
        mixed_constant += int(len(set(int(value) for value in prediction)) == 1)

    report = {
        "schema_version": "r3_target_multiplicity_oof_analysis_v1",
        "status": "complete",
        "definition": {
            "single_target": "REFNO occurs exactly once in the complete 3,173-row R3 identity manifest",
            "mixed_labels": "a PDF-backed REFNO has at least one R3=0 and one R3=1 target",
        },
        "primary_strategy": all_oof.PRIMARY_STRATEGY,
        "segments": segments,
        "single_target_models": single_model_results,
        "single_target_posthoc_oracle": {
            "warning": "threshold selected after reading all single-target labels",
            "threshold": oracle_threshold,
            "result": metrics(
                labels,
                primary,
                masks["single_target"],
                refnos,
                threshold=oracle_threshold,
            ),
        },
        "mixed_label_paper_diagnostics": {
            "papers": len(mixed_refnos),
            "all_targets_correct_papers": mixed_exact,
            "all_targets_correct_rate": round(mixed_exact / len(mixed_refnos), 8),
            "constant_prediction_papers": mixed_constant,
            "constant_prediction_rate": round(mixed_constant / len(mixed_refnos), 8),
        },
        "inputs": {
            "identity": {
                "path": str(all_oof.IDENTITY),
                "sha256": base.file_sha256(all_oof.IDENTITY),
            },
            "targets": {
                "path": str(base.DEFAULT_TARGETS),
                "sha256": base.file_sha256(base.DEFAULT_TARGETS),
            },
            "prediction_receipt": {
                "path": str(receipt_path),
                "sha256": base.file_sha256(receipt_path),
            },
            "oof_probabilities": {
                "path": str(archive_path),
                "sha256": base.file_sha256(archive_path),
            },
        },
    }
    base.atomic_json(OUTPUT, report)
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "segments": segments,
                "mixed_label_paper_diagnostics": report[
                    "mixed_label_paper_diagnostics"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
