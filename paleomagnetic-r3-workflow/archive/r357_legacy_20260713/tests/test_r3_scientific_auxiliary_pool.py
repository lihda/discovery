from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import experiment_r3_exact300_auxiliary_training as membership
from scripts import experiment_r3_pdf_only_strategies as base
from scripts import experiment_r3_scientific_auxiliary_pool as experiment


ROOT = Path(__file__).resolve().parents[1]


class R3ScientificAuxiliaryPoolTests(unittest.TestCase):
    def test_frozen_protocol_and_label_free_full_pool_cache(self) -> None:
        protocol = experiment.DEFAULT_PROTOCOL
        cache = experiment.DEFAULT_INPUTS
        manifest = experiment.DEFAULT_INPUT_MANIFEST
        self.assertEqual(base.file_sha256(protocol), experiment.AUX_PROTOCOL_SHA256)
        self.assertEqual(
            base.file_sha256(cache),
            "f6d3b64bf14fc8200fa3555d132bb92d68d98d98aecb569c1ecb67fbb272ec62",
        )
        self.assertEqual(
            base.file_sha256(manifest),
            "507e9b437eee61f9f336fa82cc07568af4b39fa99a1df2c7fb6534fb1720b835",
        )
        with np.load(cache, allow_pickle=False) as payload:
            self.assertEqual(set(payload.files), experiment.EXPECTED_COMPONENTS)
            self.assertTrue(all(payload[name].shape == (2172,) for name in payload.files))
            self.assertTrue(
                all(all(str(value).strip() for value in payload[name]) for name in payload.files)
            )

    def test_real_fold1_excludes_every_test_paper_row(self) -> None:
        full, _ = base.load_pdf_only_identity(
            experiment.DEFAULT_FULL_IDENTITY, base.DEFAULT_PDF_DIR
        )
        exact, _ = base.load_pdf_only_identity(
            experiment.DEFAULT_EXACT_IDENTITY, base.DEFAULT_PDF_DIR
        )
        lookup = membership.validate_membership(full, exact)
        fold_ids = experiment.load_canonical_fold_ids(
            experiment.DEFAULT_FOLD_MANIFEST, exact
        )
        exact_test = np.flatnonzero(fold_ids == 1)
        train, test, test_refnos = membership.auxiliary_split(
            full, exact, exact_test, lookup
        )

        self.assertEqual(len(exact_test), 60)
        self.assertEqual(len(test_refnos), 51)
        self.assertEqual(len(train), 2035)
        self.assertEqual(len(test), 60)
        self.assertEqual(
            sum(record["REFNO"] in test_refnos for record in full), 137
        )
        self.assertFalse(
            {full[index]["REFNO"] for index in train} & test_refnos
        )

    def test_paper_weights_equalize_class_mass(self) -> None:
        records = [
            {"REFNO": "A"},
            {"REFNO": "A"},
            {"REFNO": "B"},
            {"REFNO": "C"},
        ]
        labels = np.asarray([0, 0, 0, 1], dtype=np.int8)
        indices = np.arange(4, dtype=np.int64)

        weights, audit = experiment.paper_class_balanced_weights(
            indices, records, labels
        )

        self.assertAlmostEqual(float(weights[labels == 0].sum()), 2.0, places=6)
        self.assertAlmostEqual(float(weights[labels == 1].sum()), 2.0, places=6)
        self.assertEqual(audit["papers"], 3)
        self.assertEqual(audit["balanced_class_mass"], {"0": 2.0, "1": 2.0})

    def test_inner_threshold_avoids_one_class_decision(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.asarray([0.20, 0.40, 0.60, 0.80], dtype=np.float64)

        threshold, metric, stable, count = experiment.select_inner_threshold(
            labels, probabilities
        )

        self.assertTrue(stable)
        self.assertGreaterEqual(count, len(experiment.THRESHOLD_GRID))
        self.assertEqual(metric["accuracy"], 1.0)
        self.assertEqual(metric["predicted_positive"], 2)
        self.assertGreater(threshold, 0.40)
        self.assertLessEqual(threshold, 0.60)

    def test_inner_split_is_deterministic_and_refno_isolated(self) -> None:
        records = [
            {"REFNO": str(refno)}
            for refno in range(20)
            for _ in range(1 + refno % 3)
        ]
        indices = np.arange(len(records), dtype=np.int64)

        first_fit, first_validation = experiment.deterministic_inner_split(
            indices, records
        )
        second_fit, second_validation = experiment.deterministic_inner_split(
            indices, records
        )

        np.testing.assert_array_equal(first_fit, second_fit)
        np.testing.assert_array_equal(first_validation, second_validation)
        self.assertFalse(
            {records[index]["REFNO"] for index in first_fit}
            & {records[index]["REFNO"] for index in first_validation}
        )

    def test_constant_inner_probability_is_marked_collapsed(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.full(4, 0.7, dtype=np.float64)

        _, _, stable, _ = experiment.select_inner_threshold(labels, probabilities)

        self.assertFalse(stable)

    def test_fold1_artifact_contains_no_labels_or_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fold1.npz"
            experiment.write_fold1_artifact(
                path,
                np.linspace(0.1, 0.9, 60),
                np.asarray([0] * 30 + [1] * 30, dtype=np.int8),
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(
                    set(payload.files), {"fold_1_probability", "fold_1_decision"}
                )

    def test_training_entry_has_no_prior_prediction_or_source_table_input(self) -> None:
        source = inspect.getsource(experiment.main)
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("PMAGRESULT", source)
        self.assertNotIn("ROCKMAG.xlsx", source)


if __name__ == "__main__":
    unittest.main()
