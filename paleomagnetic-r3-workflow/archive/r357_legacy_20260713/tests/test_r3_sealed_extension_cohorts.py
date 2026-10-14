from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import freeze_r3_fourth300_validation_cohort as fourth300  # noqa: E402
import freeze_r3_random300_extension_cohort as random300  # noqa: E402
import freeze_r3_third300_validation_cohort as third300  # noqa: E402


class SealedExtensionCohortTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full_records, _ = base.load_pdf_only_identity(
            auxiliary.FULL_IDENTITY, base.DEFAULT_PDF_DIR
        )
        cls.exact_records, _ = base.load_pdf_only_identity(
            auxiliary.EXACT_IDENTITY, base.DEFAULT_PDF_DIR
        )
        cls.random_manifest = json.loads(random300.OUTPUT.read_text(encoding="utf-8"))
        cls.random_records, _ = random300.validate_manifest(
            cls.random_manifest, cls.full_records, cls.exact_records
        )
        cls.third_manifest = json.loads(third300.OUTPUT.read_text(encoding="utf-8"))
        cls.third_records, _ = third300.validate_manifest(
            cls.third_manifest,
            cls.full_records,
            cls.exact_records,
            cls.random_records,
        )
        cls.fourth_manifest = json.loads(fourth300.OUTPUT.read_text(encoding="utf-8"))
        cls.fourth_records, _ = fourth300.validate_manifest(
            cls.fourth_manifest,
            cls.full_records,
            [*cls.exact_records, *cls.random_records, *cls.third_records],
        )

    def test_all_four_cohorts_are_paper_disjoint(self) -> None:
        cohorts = [
            self.exact_records,
            self.random_records,
            self.third_records,
            self.fourth_records,
        ]
        refno_sets = [{record["REFNO"] for record in cohort} for cohort in cohorts]
        self.assertTrue(all(len(cohort) == 300 for cohort in cohorts))
        for left in range(len(refno_sets)):
            for right in range(left + 1, len(refno_sets)):
                self.assertFalse(refno_sets[left] & refno_sets[right])

    def test_sealed_receipts_contain_no_test_labels(self) -> None:
        for path in (
            ROOT
            / "tmp/r3_third300_validation_v1/robust_holdout/prediction_receipt.json",
            ROOT
            / "tmp/r3_fourth300_validation_v1/mean_holdout/prediction_receipt.json",
        ):
            receipt = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["status"], "PREDICTIONS_FROZEN_BEFORE_TEST_SCORING"
            )
            self.assertFalse(receipt["test_labels_present"])
            self.assertEqual(len(receipt["rows"]), 300)
            for row in receipt["rows"]:
                keys = {str(key).casefold() for key in row}
                self.assertFalse({"label", "answer", "target", "r3"} & keys)

    def test_third_primary_metrics_recompute(self) -> None:
        output = ROOT / "tmp/r3_third300_validation_v1/robust_holdout"
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        labels_by_resultno = base.load_training_labels(
            base.DEFAULT_TARGETS,
            {record["RESULTNO"] for record in self.third_records},
        )
        labels = np.asarray(
            [labels_by_resultno[record["RESULTNO"]] for record in self.third_records],
            dtype=np.int8,
        )
        with np.load(output / "holdout_probabilities.npz") as archive:
            observed = base.metric_record(
                labels, archive["conservative_soft_router_010"]
            )
        self.assertEqual(
            {"strategy": "conservative_soft_router_010", **observed},
            report["primary_result"],
        )
        self.assertEqual(observed["accuracy"], 0.87)

    def test_fourth_primary_metrics_recompute(self) -> None:
        output = ROOT / "tmp/r3_fourth300_validation_v1/mean_holdout"
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        labels_by_resultno = base.load_training_labels(
            base.DEFAULT_TARGETS,
            {record["RESULTNO"] for record in self.fourth_records},
        )
        labels = np.asarray(
            [labels_by_resultno[record["RESULTNO"]] for record in self.fourth_records],
            dtype=np.int8,
        )
        with np.load(output / "holdout_probabilities.npz") as archive:
            observed = base.metric_record(labels, archive["mean6"])
        self.assertEqual(
            {"strategy": "mean6", **observed}, report["primary_result"]
        )
        self.assertEqual(observed["accuracy"], 0.82333333)


if __name__ == "__main__":
    unittest.main()
