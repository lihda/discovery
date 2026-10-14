from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import experiment_r3_exact300_auxiliary_training as experiment


class R3Exact300AuxiliaryTrainingTests(unittest.TestCase):
    def test_auxiliary_split_excludes_every_row_of_test_paper(self) -> None:
        full = [
            {"REFNO": "A", "RESULTNO": "1", "ROCKNAME": "alpha"},
            {"REFNO": "A", "RESULTNO": "2", "ROCKNAME": "beta"},
            {"REFNO": "B", "RESULTNO": "3", "ROCKNAME": "gamma"},
            {"REFNO": "C", "RESULTNO": "4", "ROCKNAME": "delta"},
        ]
        exact = [full[0], full[2], full[3]]
        lookup = {experiment.row_key(record): index for index, record in enumerate(full)}

        train, test, test_refnos = experiment.auxiliary_split(
            full, exact, np.asarray([0]), lookup
        )

        self.assertEqual(test_refnos, {"A"})
        self.assertEqual(test.tolist(), [0])
        self.assertEqual(train.tolist(), [2, 3])
        self.assertFalse(any(full[index]["REFNO"] == "A" for index in train))

    def test_membership_requires_exact_rows_to_be_full_subset(self) -> None:
        original = (
            experiment.EXPECTED_FULL_PDF_ROWS,
            experiment.EXPECTED_FULL_PDF_PAPERS,
            experiment.EXPECTED_EXACT_ROWS,
            experiment.EXPECTED_EXACT_PAPERS,
        )
        experiment.EXPECTED_FULL_PDF_ROWS = 1
        experiment.EXPECTED_FULL_PDF_PAPERS = 1
        experiment.EXPECTED_EXACT_ROWS = 1
        experiment.EXPECTED_EXACT_PAPERS = 1
        try:
            full = [{"REFNO": "A", "RESULTNO": "1", "ROCKNAME": "alpha"}]
            exact = [{"REFNO": "A", "RESULTNO": "9", "ROCKNAME": "alpha"}]
            with self.assertRaisesRegex(ValueError, "not a subset"):
                experiment.validate_membership(full, exact)
        finally:
            (
                experiment.EXPECTED_FULL_PDF_ROWS,
                experiment.EXPECTED_FULL_PDF_PAPERS,
                experiment.EXPECTED_EXACT_ROWS,
                experiment.EXPECTED_EXACT_PAPERS,
            ) = original

    def test_probability_artifact_rejects_identifier_or_label_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.npz"
            with self.assertRaisesRegex(ValueError, "label or record identifier"):
                experiment.write_probability_artifact(
                    path,
                    {"labels": np.zeros(experiment.EXPECTED_EXACT_ROWS)},
                )

    def test_probability_artifact_allows_descriptive_model_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.npz"
            experiment.write_probability_artifact(
                path,
                {"aux_rockname_char": np.zeros(experiment.EXPECTED_EXACT_ROWS)},
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(payload.files, ["aux_rockname_char"])

    def test_fold_manifest_controls_outer_membership(self) -> None:
        records = [
            {
                "REFNO": str(index),
                "RESULTNO": str(index),
                "ROCKNAME": f"target {index}",
            }
            for index in range(experiment.EXPECTED_EXACT_ROWS)
        ]
        fold_ids = np.repeat(
            np.arange(1, experiment.N_SPLITS + 1, dtype=np.int8), 60
        )
        manifest = {
            "schema_version": "r3_exact300_label_free_refno_folds_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "input": {
                "identity_sha256": experiment.EXACT_IDENTITY_SHA256,
                "alignment_sha256": experiment.sha256_records(records),
            },
            "runtime_boundary": {
                "labels_read": False,
                "targets_workbook_read": False,
                "prior_predictions_or_evaluations_read": False,
            },
            "fold_id_sha256": hashlib.sha256(bytes(fold_ids)).hexdigest(),
            "rows": [
                {
                    "row_index": index,
                    "REFNO": record["REFNO"],
                    "RESULTNO": record["RESULTNO"],
                    "fold": int(fold_ids[index]),
                }
                for index, record in enumerate(records)
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            loaded = experiment.load_fold_manifest(path, records)
        folds = experiment.exact_group_folds(records, loaded)
        self.assertEqual(folds[0][1].tolist(), list(range(60)))
        self.assertEqual(folds[4][1].tolist(), list(range(240, 300)))

    def test_inner_threshold_is_selected_without_test_input(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probability = np.asarray([0.1, 0.3, 0.4, 0.9], dtype=np.float64)
        threshold, metric = experiment.select_inner_threshold(labels, probability)
        self.assertAlmostEqual(threshold, 0.4)
        self.assertEqual(metric["accuracy"], 1.0)

    def test_experiment_has_no_prior_prediction_input(self) -> None:
        source = inspect.getsource(experiment.main)
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("grouped_oof.npz", source)
        self.assertNotIn("PMAGRESULT", source)
        self.assertNotIn("ROCKMAG source", source)


if __name__ == "__main__":
    unittest.main()
