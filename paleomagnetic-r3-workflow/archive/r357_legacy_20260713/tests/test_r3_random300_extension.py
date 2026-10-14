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
import freeze_r3_random300_extension_cohort as cohort  # noqa: E402


class Random300ExtensionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_path = ROOT / "tmp/r3_random300_extension_v1/cohort_manifest.json"
        cls.output_dir = ROOT / "tmp/r3_random300_extension_v1/holdout"
        cls.manifest = json.loads(cls.manifest_path.read_text(encoding="utf-8"))
        cls.full_records, _ = base.load_pdf_only_identity(
            auxiliary.FULL_IDENTITY, base.DEFAULT_PDF_DIR
        )
        cls.exact_records, _ = base.load_pdf_only_identity(
            auxiliary.EXACT_IDENTITY, base.DEFAULT_PDF_DIR
        )

    def test_cohort_is_deterministic_and_paper_disjoint(self) -> None:
        records, full_indices = cohort.validate_manifest(
            self.manifest, self.full_records, self.exact_records
        )
        self.assertEqual(len(records), 300)
        self.assertEqual(len(set(int(index) for index in full_indices)), 300)
        self.assertEqual(len({record["REFNO"] for record in records}), 263)
        self.assertFalse(
            {record["REFNO"] for record in records}
            & {record["REFNO"] for record in self.exact_records}
        )
        rebuilt = cohort.build_manifest(
            auxiliary.FULL_IDENTITY,
            auxiliary.EXACT_IDENTITY,
            base.DEFAULT_PDF_DIR,
        )
        self.assertEqual(rebuilt["rows"], self.manifest["rows"])
        self.assertEqual(
            rebuilt["sample_alignment_sha256"],
            self.manifest["sample_alignment_sha256"],
        )

    def test_every_sample_paper_is_excluded_from_fit(self) -> None:
        records, _ = cohort.validate_manifest(
            self.manifest, self.full_records, self.exact_records
        )
        test_refnos = {record["REFNO"] for record in records}
        train = [
            record for record in self.full_records if record["REFNO"] not in test_refnos
        ]
        excluded = [
            record for record in self.full_records if record["REFNO"] in test_refnos
        ]
        self.assertEqual(len(train), 1652)
        self.assertEqual(len({record["REFNO"] for record in train}), 911)
        self.assertEqual(len(excluded), 520)
        self.assertFalse({record["REFNO"] for record in train} & test_refnos)

    def test_prediction_receipt_precedes_label_scoring_contract(self) -> None:
        receipt = json.loads(
            (self.output_dir / "prediction_receipt.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            receipt["status"], "PREDICTIONS_FROZEN_BEFORE_TEST_SCORING"
        )
        self.assertFalse(receipt["test_labels_present"])
        self.assertEqual(len(receipt["rows"]), 300)
        forbidden = {"label", "target", "answer", "r3"}
        for row in receipt["rows"]:
            self.assertFalse(forbidden & {str(key).casefold() for key in row})

    def test_report_metrics_recompute_from_frozen_probabilities(self) -> None:
        report = json.loads((self.output_dir / "report.json").read_text(encoding="utf-8"))
        records, _ = cohort.validate_manifest(
            self.manifest, self.full_records, self.exact_records
        )
        labels_by_resultno = base.load_training_labels(
            base.DEFAULT_TARGETS, {record["RESULTNO"] for record in records}
        )
        labels = np.asarray(
            [labels_by_resultno[record["RESULTNO"]] for record in records],
            dtype=np.int8,
        )
        archive = np.load(self.output_dir / "holdout_probabilities.npz")
        observed = base.metric_record(labels, archive["nested_selected"])
        self.assertEqual(
            {"strategy": "nested_selected", **observed}, report["best_observed"]
        )
        self.assertEqual(observed["accuracy"], 0.84666667)
        self.assertEqual(observed["confusion"], {"tn": 71, "fp": 25, "fn": 21, "tp": 183})


if __name__ == "__main__":
    unittest.main()
