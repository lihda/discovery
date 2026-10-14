from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import scripts.evaluate_r3_training_posthoc as evaluator


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition_refnos(training_count: int) -> tuple[list[str], str]:
    training: list[str] = []
    holdout = ""
    candidate = 1
    while len(training) < training_count or not holdout:
        refno = str(candidate)
        if evaluator.is_holdout_refno(refno):
            holdout = holdout or refno
        elif len(training) < training_count:
            training.append(refno)
        candidate += 1
    return training, holdout


class R3TrainingPosthocEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.training_refnos, cls.holdout_refno = partition_refnos(
            evaluator.EXPECTED_TRAINING_ROWS
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.prediction_a = self.root / "sentence_only.jsonl"
        self.prediction_b = self.root / "exclusive_scope.jsonl"
        self.labels = self.root / "labels.xlsx"
        self.page_cache = self.root / "page_cache.jsonl"
        self.output = self.root / "evaluation.json"
        self.receipt = self.output.with_suffix(
            self.output.suffix + ".prediction_hashes.json"
        )

    def tearDown(self):
        self.temporary.cleanup()

    def label_rows(self) -> list[dict]:
        rows = [
            {
                "REFNO": refno,
                "RESULTNO": f"r{index}",
                "R3": index % 2,
            }
            for index, refno in enumerate(self.training_refnos)
        ]
        rows.append(
            {"REFNO": self.holdout_refno, "RESULTNO": "holdout", "R3": 1}
        )
        return rows

    def write_labels(self, rows: list[dict] | None = None) -> None:
        pd.DataFrame(self.label_rows() if rows is None else rows).to_excel(
            self.labels, index=False
        )

    def prediction_rows(self, variant: str = "a") -> list[dict]:
        rows: list[dict] = []
        for index, refno in enumerate(self.training_refnos):
            truth = index % 2
            prediction = truth
            outcome = "POSITIVE" if prediction else "NEGATIVE"
            if variant == "a" and index == 0:
                prediction = 1
                outcome = "ABSTAIN"
            if variant == "b" and index == 2:
                prediction = 1
                outcome = "POSITIVE"
            rows.append(
                {
                    "REFNO": refno,
                    "RESULTNO": f"r{index}",
                    "R3": prediction,
                    "OUTCOME": outcome,
                }
            )
        return rows

    def write_predictions(
        self, path: Path, rows: list[dict] | None = None, *, variant: str = "a"
    ) -> None:
        selected = self.prediction_rows(variant) if rows is None else rows
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in selected),
            encoding="utf-8",
        )

    def write_page_cache(self, eligible_count: int) -> None:
        records = []
        for index, refno in enumerate(self.training_refnos):
            eligible = index < eligible_count
            records.append(
                {
                    "REFNO": refno,
                    "source": {
                        "pdf_path": f"/papers/{refno}.pdf" if eligible else None,
                        "source_quality_flags": [] if eligible else ["no_local_pdf"],
                    },
                    "pages": [],
                }
            )
        self.page_cache.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )

    def test_receipts_all_predictions_before_labels_and_reports_aggregate_metrics(self):
        self.write_labels()
        self.write_predictions(self.prediction_a, variant="a")
        self.write_predictions(self.prediction_b, variant="b")
        prediction_hashes = [sha256(self.prediction_a), sha256(self.prediction_b)]
        labels_hash = sha256(self.labels)
        original_read_excel = evaluator.pd.read_excel

        def guarded_read_excel(*args, **kwargs):
            self.assertTrue(self.receipt.is_file())
            receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
            self.assertEqual(receipt["phase"], "before_labels_opened")
            self.assertEqual(
                [item["sha256"] for item in receipt["predictions"]],
                prediction_hashes,
            )
            return original_read_excel(*args, **kwargs)

        with patch.object(evaluator.pd, "read_excel", side_effect=guarded_read_excel):
            report = evaluator.evaluate(
                [self.prediction_a, self.prediction_b],
                self.labels,
                self.output,
                include_change_counts=True,
            )

        self.assertEqual(report["split"]["training_rows"], 2473)
        self.assertEqual(report["split"]["holdout_rows_excluded"], 1)
        metrics_a = report["prediction_sets"][0]["metrics"]
        self.assertEqual(metrics_a["labelled_rows"], 2473)
        self.assertEqual(metrics_a["correct_rows"], 2472)
        self.assertEqual(metrics_a["known_rows"], 2472)
        self.assertEqual(metrics_a["coverage"], 2472 / 2473)
        self.assertEqual(metrics_a["known_accuracy"], 1.0)
        changes = report["prediction_change_counts"]
        self.assertEqual(len(changes), 1)
        self.assertEqual(changes[0]["r3_changed_rows"], 2)
        self.assertEqual(changes[0]["r3_zero_to_one_rows"], 1)
        self.assertEqual(changes[0]["r3_one_to_zero_rows"], 1)
        self.assertEqual(changes[0]["known_status_changed_rows"], 1)

        self.assertEqual(sha256(self.prediction_a), prediction_hashes[0])
        self.assertEqual(sha256(self.prediction_b), prediction_hashes[1])
        self.assertEqual(sha256(self.labels), labels_hash)
        serialized = json.dumps(report)
        self.assertNotIn('"REFNO"', serialized)
        self.assertNotIn('"RESULTNO"', serialized)
        self.assertTrue(self.output.is_file())

    def test_rejects_wrong_count_before_labels_are_opened(self):
        self.write_labels()
        self.write_predictions(
            self.prediction_a, self.prediction_rows()[:-1]
        )

        with patch.object(evaluator.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(
                evaluator.TrainingEvaluationError, "exactly 2473"
            ):
                evaluator.evaluate(self.prediction_a, self.labels, self.output)

        read_excel.assert_not_called()
        self.assertTrue(self.receipt.is_file())
        self.assertFalse(self.output.exists())

    def test_rejects_holdout_assignment_before_labels_are_opened(self):
        self.write_labels()
        rows = self.prediction_rows()
        rows[0]["REFNO"] = self.holdout_refno
        self.write_predictions(self.prediction_a, rows)

        with patch.object(evaluator.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(
                evaluator.TrainingEvaluationError, "holdout assignments"
            ):
                evaluator.evaluate([self.prediction_a], self.labels, self.output)

        read_excel.assert_not_called()
        self.assertTrue(self.receipt.is_file())

    def test_requires_exact_labelled_training_identity_set(self):
        rows = self.label_rows()
        rows[0]["RESULTNO"] = "different"
        self.write_labels(rows)
        self.write_predictions(self.prediction_a)

        with self.assertRaisesRegex(
            evaluator.TrainingEvaluationError, "exactly equal"
        ):
            evaluator.evaluate([self.prediction_a], self.labels, self.output)

        self.assertTrue(self.receipt.is_file())
        self.assertFalse(self.output.exists())

    def test_rejects_prediction_changed_after_receipt(self):
        self.write_labels()
        self.write_predictions(self.prediction_a)
        original_read_excel = evaluator.pd.read_excel

        def tampering_read_excel(*args, **kwargs):
            frame = original_read_excel(*args, **kwargs)
            self.prediction_a.write_text(
                self.prediction_a.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            return frame

        with patch.object(
            evaluator.pd, "read_excel", side_effect=tampering_read_excel
        ):
            with self.assertRaisesRegex(
                evaluator.TrainingEvaluationIntegrityError, "changed"
            ):
                evaluator.evaluate([self.prediction_a], self.labels, self.output)

        self.assertTrue(self.receipt.is_file())
        self.assertFalse(self.output.exists())

    def test_rejects_label_changed_while_opening(self):
        self.write_labels()
        self.write_predictions(self.prediction_a)
        original_read_excel = evaluator.pd.read_excel

        def tampering_read_excel(*args, **kwargs):
            frame = original_read_excel(*args, **kwargs)
            with self.labels.open("ab") as handle:
                handle.write(b"tampered")
            return frame

        with patch.object(
            evaluator.pd, "read_excel", side_effect=tampering_read_excel
        ):
            with self.assertRaisesRegex(
                evaluator.TrainingEvaluationIntegrityError, "Label workbook changed"
            ):
                evaluator.evaluate([self.prediction_a], self.labels, self.output)

        self.assertTrue(self.receipt.is_file())
        self.assertFalse(self.output.exists())

    def test_pdf_only_scope_is_frozen_before_labels_and_skips_no_pdf_rows(self):
        self.write_labels()
        self.write_predictions(self.prediction_a, variant="a")
        self.write_page_cache(eligible_count=1200)
        page_cache_hash = sha256(self.page_cache)
        original_read_excel = evaluator.pd.read_excel

        def guarded_read_excel(*args, **kwargs):
            receipt = json.loads(self.receipt.read_text(encoding="utf-8"))
            scope = receipt["pdf_only_scope"]
            self.assertTrue(scope["enabled"])
            self.assertEqual(scope["page_cache_sha256"], page_cache_hash)
            return original_read_excel(*args, **kwargs)

        with patch.object(evaluator.pd, "read_excel", side_effect=guarded_read_excel):
            report = evaluator.evaluate(
                self.prediction_a,
                self.labels,
                self.output,
                page_cache_path=self.page_cache,
                pdf_only=True,
            )

        scope = report["evaluation_scope"]
        self.assertTrue(scope["pdf_only"])
        self.assertEqual(scope["pdf_eligible_papers"], 1200)
        self.assertEqual(scope["evaluated_rows"], 1200)
        self.assertEqual(scope["skipped_rows"], 1273)
        self.assertEqual(report["prediction_sets"][0]["metrics"]["labelled_rows"], 1200)
        self.assertEqual(sha256(self.page_cache), page_cache_hash)

    def test_pdf_only_cache_mismatch_is_rejected_before_labels_open(self):
        self.write_labels()
        self.write_predictions(self.prediction_a)
        self.write_page_cache(eligible_count=1200)
        lines = self.page_cache.read_text(encoding="utf-8").splitlines()
        self.page_cache.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        with patch.object(evaluator.pd, "read_excel") as read_excel:
            with self.assertRaisesRegex(
                evaluator.TrainingEvaluationError,
                "REFNO set must match",
            ):
                evaluator.evaluate(
                    self.prediction_a,
                    self.labels,
                    self.output,
                    page_cache_path=self.page_cache,
                    pdf_only=True,
                )

        read_excel.assert_not_called()
        self.assertTrue(self.receipt.is_file())
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
