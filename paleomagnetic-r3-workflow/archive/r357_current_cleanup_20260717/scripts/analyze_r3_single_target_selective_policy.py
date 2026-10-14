#!/usr/bin/env python3
"""Evaluate the fixed selective R3 policy on PDF-backed single-target papers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_all_labelled_pdf_grouped_oof as all_oof  # noqa: E402


OOF_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
INPUT = OOF_DIR / "oof_probabilities.npz"
RECEIPT = OOF_DIR / "prediction_receipt.json"
PREDICTION_OUTPUT = OOF_DIR / "single_target_selective_prediction_receipt.json"
OUTPUT = OOF_DIR / "single_target_selective_policy_report.json"
NEGATIVE_THRESHOLD = 0.30
POSITIVE_THRESHOLD = 0.70


def selective_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    known = eligible & (
        (probabilities <= NEGATIVE_THRESHOLD)
        | (probabilities >= POSITIVE_THRESHOLD)
    )
    selected_labels = labels[known]
    prediction = probabilities[known] >= POSITIVE_THRESHOLD
    tn, fp, fn, tp = confusion_matrix(
        selected_labels, prediction, labels=[0, 1]
    ).ravel()
    return {
        "eligible_rows": int(eligible.sum()),
        "decided_rows": int(known.sum()),
        "abstained_rows": int(eligible.sum() - known.sum()),
        "coverage": round(float(known.sum() / eligible.sum()), 8),
        "decided_accuracy": round(float((selected_labels == prediction).mean()), 8),
        "decided_macro_f1": round(
            float(f1_score(selected_labels, prediction, average="macro")), 8
        ),
        "decided_balanced_accuracy": round(
            float(balanced_accuracy_score(selected_labels, prediction)), 8
        ),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    archive = np.load(INPUT)
    probabilities = archive["mean4"].astype(np.float64)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))["rows"]
    identity = all_oof.load_identity(all_oof.IDENTITY)
    multiplicity = Counter(row["REFNO"] for row in identity)
    single = np.asarray(
        [multiplicity[str(row["REFNO"])] == 1 for row in receipt], dtype=bool
    )
    if len(probabilities) != len(receipt) or int(single.sum()) != 872:
        raise ValueError("single-target selective inputs do not align")
    fold_ids = np.asarray([int(row["fold"]) for row in receipt], dtype=np.int8)
    original_training = np.asarray(
        [base.is_training_refno(str(row["REFNO"])) for row in receipt], dtype=bool
    )

    selective_rows = []
    for index in np.flatnonzero(single):
        probability = float(probabilities[index])
        decision = (
            "DOES_NOT_MEET_PRACTICE_PROXY"
            if probability <= NEGATIVE_THRESHOLD
            else "MEETS_PRACTICE_PROXY"
            if probability >= POSITIVE_THRESHOLD
            else "ABSTAIN"
        )
        selective_rows.append(
            {
                "row_index": int(index),
                "REFNO": str(receipt[index]["REFNO"]),
                "RESULTNO": str(receipt[index]["RESULTNO"]),
                "ROCKNAME": str(receipt[index]["ROCKNAME"]),
                "fold": int(fold_ids[index]),
                "probability": probability,
                "decision": decision,
            }
        )
    prediction_receipt = {
        "schema_version": "r3_single_target_selective_prediction_receipt_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "negative_threshold": NEGATIVE_THRESHOLD,
        "positive_threshold": POSITIVE_THRESHOLD,
        "rows": selective_rows,
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "identity_columns": ["REFNO", "RESULTNO", "ROCKNAME"],
            "probability_source": "REFNO-grouped OOF archive",
        },
    }
    base.atomic_json(PREDICTION_OUTPUT, prediction_receipt)

    labels_by_resultno = all_oof.load_labels(
        base.DEFAULT_TARGETS, {str(row["RESULTNO"]) for row in receipt}
    )
    labels = np.asarray(
        [labels_by_resultno[str(row["RESULTNO"])] for row in receipt], dtype=np.int8
    )
    full_prediction = probabilities >= 0.5
    report = {
        "schema_version": "r3_single_target_selective_policy_evaluation_v1",
        "status": "complete_development_oof",
        "policy": {
            "negative": f"probability <= {NEGATIVE_THRESHOLD:.2f}",
            "positive": f"probability >= {POSITIVE_THRESHOLD:.2f}",
            "abstain": f"{NEGATIVE_THRESHOLD:.2f} < probability < {POSITIVE_THRESHOLD:.2f}",
            "no_pdf": "SKIP_NO_SOURCE",
        },
        "scope": {
            "definition": "REFNO occurs once in the complete 3,173-row R3 identity manifest",
            "pdf_backed_single_target_rows": int(single.sum()),
            "paper_groups": int(single.sum()),
        },
        "full_coverage_reference": {
            "threshold": 0.5,
            "rows": int(single.sum()),
            "correct": int((labels[single] == full_prediction[single]).sum()),
            "accuracy": round(
                float((labels[single] == full_prediction[single]).mean()), 8
            ),
        },
        "selective_result": selective_metrics(labels, probabilities, single),
        "folds": [
            {
                "fold": fold,
                **selective_metrics(
                    labels, probabilities, single & (fold_ids == fold)
                ),
            }
            for fold in range(1, 6)
        ],
        "historical_partitions": {
            "original_training": selective_metrics(
                labels, probabilities, single & original_training
            ),
            "original_holdout": selective_metrics(
                labels, probabilities, single & ~original_training
            ),
        },
        "claim_boundary": {
            "selective_decisions_frozen_before_label_access": True,
            "outer_predictions_are_refno_grouped_oof": True,
            "each_prediction_excludes_its_paper_from_model_fit": True,
            "thresholds_selected_during_single_target_development": True,
            "independent_external_validation": False,
            "full_coverage_accuracy_above_90_percent": False,
            "decided_subset_accuracy_above_90_percent": True,
            "coverage_must_accompany_accuracy": True,
        },
        "inputs": {
            "probabilities": {"path": str(INPUT), "sha256": base.file_sha256(INPUT)},
            "receipt": {"path": str(RECEIPT), "sha256": base.file_sha256(RECEIPT)},
            "selective_prediction_receipt": {
                "path": str(PREDICTION_OUTPUT),
                "sha256": base.file_sha256(PREDICTION_OUTPUT),
            },
            "labels_postfreeze": {
                "path": str(base.DEFAULT_TARGETS),
                "sha256": base.file_sha256(base.DEFAULT_TARGETS),
            },
        },
    }
    base.atomic_json(OUTPUT, report)
    print(json.dumps({"output": str(OUTPUT), **report}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
