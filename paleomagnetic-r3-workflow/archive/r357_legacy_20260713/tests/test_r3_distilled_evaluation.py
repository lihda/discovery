from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import scripts.evaluate_r3_distilled_predictions as evaluator


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_refnos(count: int = 4) -> tuple[list[str], list[str]]:
    holdout: list[str] = []
    training: list[str] = []
    candidate = 1
    while len(holdout) < count or len(training) < count:
        refno = str(candidate)
        destination = holdout if evaluator.is_holdout_refno(refno) else training
        if len(destination) < count:
            destination.append(refno)
        candidate += 1
    return holdout, training


class R3DistilledEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.predictions = self.root / "r3_predictions.jsonl"
        self.labels = self.root / "labels.xlsx"
        self.output = self.root / "evaluation.json"
        self.receipt = self.output.with_suffix(
            self.output.suffix + ".prediction_hashes.json"
        )
        self.holdout, self.training = split_refnos()

    def tearDown(self):
        self.temporary.cleanup()

    def write_labels(self) -> list[dict]:
        rows = [
            {"REFNO": self.holdout[0], "RESULTNO": "h0", "R3": 0},
            {"REFNO": self.holdout[1], "RESULTNO": "h1", "R3": 0},
            {"REFNO": self.holdout[2], "RESULTNO": "h2", "R3": 1},
            {"REFNO": self.holdout[3], "RESULTNO": "h3", "R3": 1},
            {"REFNO": self.training[0], "RESULTNO": "t0", "R3": 1},
        ]
        pd.DataFrame(rows).to_excel(self.labels, index=False)
        return rows

    def prediction_rows(self) -> list[dict]:
        return [
            {"REFNO": self.holdout[0], "RESULTNO": "h0", "R3": 0, "outcome": "NEGATIVE"},
            {
                "REFNO": self.holdout[1],
                "RESULTNO": "h1",
                "R3": 1,
                "R3_RESULT_JSON": {"outcome": "ABSTAIN"},
            },
            {"REFNO": self.holdout[2], "RESULTNO": "h2", "R3": 0, "OUTCOME": "NEGATIVE"},
            {"REFNO": self.holdout[3], "RESULTNO": "h3", "R3": 1, "outcome": "POSITIVE"},
        ]

    def write_predictions(self, rows: list[dict] | None = None) -> None:
        rows = self.prediction_rows() if rows is None else rows
        self.predictions.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

    def test_evaluates_only_fixed_holdout_and_persists_receipt_before_labels_open(self):
        self.write_labels()
        self.write_predictions()
        prediction_hash = sha256(self.predictions)
        original_read_excel = evaluator.pd.read_excel

        def guarded_read_excel(*args, **kwargs):
            self.assertTrue(self.receipt.is_file())
            receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt["phase"], "before_labels_opened")
            self.assertEqual(receipt["prediction"]["sha256"], prediction_hash)
            return original_read_excel(*args, **kwargs)

        with patch.object(evaluator.pd, "read_excel", side_effect=guarded_read_excel):
            report = evaluator.evaluate(self.predictions, self.labels, self.output)

        self.assertEqual(report["split"]["salt"], "r3-rockmag-distillation-v1")
        self.assertEqual(report["split"]["fraction"], 0.20)
        self.assertEqual(report["split"]["all_rows"], 5)
        self.assertEqual(report["split"]["holdout_rows"], 4)
        metrics = report["metrics"]
        self.assertEqual(
            metrics["confusion_matrix"],
            {
                "true_negative": 1,
                "false_positive": 1,
                "false_negative": 1,
                "true_positive": 1,
            },
        )
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["balanced_accuracy"], 0.5)
        self.assertEqual(metrics["coverage"], 0.75)
        self.assertAlmostEqual(metrics["known_accuracy"], 2 / 3)
        self.assertEqual(metrics["macro_f1"], 0.5)
        self.assertEqual(metrics["f1_0"], 0.5)
        self.assertEqual(metrics["f1_1"], 0.5)
        self.assertEqual(
            report["sources"]["predictions"]["sha256_before_labels"],
            prediction_hash,
        )
        self.assertEqual(
            report["sources"]["predictions"]["sha256_after_evaluation"],
            prediction_hash,
        )
        self.assertEqual(report["prediction_hash_receipt"]["sha256"], sha256(self.receipt))
        serialized = json.dumps(report)
        self.assertNotIn('"RESULTNO"', serialized)
        self.assertNotIn('"REFNO"', serialized)
        self.assertTrue(self.output.is_file())

    def test_rejects_non_holdout_prediction_before_labels_are_opened(self):
        self.write_labels()
        rows = self.prediction_rows()
        rows.append(
            {"REFNO": self.training[0], "RESULTNO": "t0", "R3": 1, "outcome": "POSITIVE"}
        )
        self.write_predictions(rows)

        with patch.object(evaluator.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(evaluator.EvaluationError, "non-holdout"):
                evaluator.evaluate(self.predictions, self.labels, self.output)

        read_excel.assert_not_called()
        self.assertTrue(self.receipt.is_file())
        self.assertFalse(self.output.exists())

    def test_rejects_duplicate_predictions_before_labels_are_opened(self):
        self.write_labels()
        rows = self.prediction_rows()
        rows.append(dict(rows[0]))
        self.write_predictions(rows)

        with patch.object(evaluator.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(evaluator.EvaluationError, "must be unique"):
                evaluator.evaluate(self.predictions, self.labels, self.output)

        read_excel.assert_not_called()
        self.assertTrue(self.receipt.is_file())

    def test_rejects_refno_mismatch_and_incomplete_holdout(self):
        self.write_labels()
        rows = self.prediction_rows()
        rows[0]["REFNO"] = self.holdout[1]
        self.write_predictions(rows)
        with self.assertRaisesRegex(evaluator.EvaluationError, "REFNO disagrees"):
            evaluator.evaluate(self.predictions, self.labels, self.output)

        self.write_predictions(self.prediction_rows()[:-1])
        with self.assertRaisesRegex(evaluator.EvaluationError, "must equal"):
            evaluator.evaluate(self.predictions, self.labels, self.output)

    def test_rejects_prediction_changed_after_receipt(self):
        self.write_labels()
        self.write_predictions()
        original_read_excel = evaluator.pd.read_excel

        def tampering_read_excel(*args, **kwargs):
            self.assertTrue(self.receipt.is_file())
            self.predictions.write_text(
                self.predictions.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            return original_read_excel(*args, **kwargs)

        with patch.object(evaluator.pd, "read_excel", side_effect=tampering_read_excel):
            with self.assertRaisesRegex(evaluator.EvaluationIntegrityError, "changed"):
                evaluator.evaluate(self.predictions, self.labels, self.output)

        self.assertTrue(self.receipt.is_file())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
