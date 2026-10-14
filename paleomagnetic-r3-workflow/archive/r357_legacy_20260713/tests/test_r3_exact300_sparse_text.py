from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import csr_matrix

from scripts import experiment_r3_exact300_sparse_text as experiment


class R3Exact300SparseTextTests(unittest.TestCase):
    def test_paper_examples_exclude_outer_test_and_optionally_mixed(self) -> None:
        records = [
            {"REFNO": "A"},
            {"REFNO": "A"},
            {"REFNO": "B"},
            {"REFNO": "C"},
        ]
        labels = np.asarray([0, 1, 1, 0], dtype=np.int8)
        retained, values, audit = experiment.paper_training_examples(
            records,
            labels,
            np.asarray([0, 1, 2, 3]),
            pure_only=True,
        )
        self.assertEqual(retained, ["B", "C"])
        self.assertEqual(values.tolist(), [1, 0])
        self.assertEqual(audit["mixed_training_papers"], 1)

    def test_nb_ratio_points_class_specific_terms_in_expected_directions(self) -> None:
        matrix = csr_matrix(
            np.asarray(
                [
                    [5, 0, 1],
                    [3, 0, 1],
                    [0, 4, 1],
                    [0, 2, 1],
                ],
                dtype=np.float64,
            )
        )
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        ratio = experiment.nb_log_count_ratio(matrix, labels)
        self.assertGreater(ratio[0], 0.0)
        self.assertLess(ratio[1], 0.0)
        self.assertAlmostEqual(ratio[2], 0.0)

    def test_candidates_never_use_alignment_identifiers_as_views(self) -> None:
        self.assertTrue(experiment.CANDIDATES)
        for candidate in experiment.CANDIDATES:
            self.assertIn(candidate.level, {"paper", "row", "passage"})
            self.assertIn(
                candidate.view,
                {
                    "paper_body",
                    "paper_evidence",
                    "paper_passages",
                    "target_local",
                },
            )
            self.assertNotIn("refno", candidate.view.casefold())
            self.assertNotIn("resultno", candidate.view.casefold())

    def test_passage_flattening_balances_each_paper_and_top_k_aggregation(self) -> None:
        documents, labels, weights, bounds = experiment.flatten_paper_passages(
            ["A", "B"],
            {"A": ["a1", "a2", "a3"], "B": ["b1"]},
            np.asarray([0, 1], dtype=np.int8),
        )
        self.assertEqual(documents, ["a1", "a2", "a3", "b1"])
        self.assertEqual(labels.tolist(), [0, 0, 0, 1])
        self.assertAlmostEqual(float(weights[bounds[0]].sum()), 2.0)
        self.assertAlmostEqual(float(weights[bounds[1]].sum()), 2.0)
        aggregated = experiment.aggregate_top_k_passages(
            np.asarray([0.1, 0.7, 0.4, 0.2]), bounds, 2
        )
        self.assertTrue(np.allclose(aggregated, [0.55, 0.2]))

    def test_canonical_fold_loader_rejects_misaligned_row(self) -> None:
        records = [
            {"REFNO": str(index), "RESULTNO": str(index), "ROCKNAME": "target"}
            for index in range(experiment.EXPECTED_ROWS)
        ]
        rows = [
            {
                "row_index": index,
                "REFNO": record["REFNO"],
                "RESULTNO": record["RESULTNO"],
                "fold": index // 60 + 1,
            }
            for index, record in enumerate(records)
        ]
        artifact = {
            "schema_version": "r3_exact300_label_free_refno_folds_v1",
            "status": "FROZEN_WITHOUT_LABEL_ACCESS",
            "fold_id_sha256": experiment.FOLD_ID_SHA256,
            "input": {"identity_sha256": experiment.auxiliary.EXACT_IDENTITY_SHA256},
            "rows": rows,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "folds.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            original = experiment.FOLD_MANIFEST_SHA256
            experiment.FOLD_MANIFEST_SHA256 = experiment.base.file_sha256(path)
            try:
                rows[0]["RESULTNO"] = "wrong"
                path.write_text(json.dumps(artifact), encoding="utf-8")
                experiment.FOLD_MANIFEST_SHA256 = experiment.base.file_sha256(path)
                with self.assertRaisesRegex(ValueError, "alignment mismatch"):
                    experiment.load_canonical_folds(path, records)
            finally:
                experiment.FOLD_MANIFEST_SHA256 = original

    def test_probability_artifact_rejects_identifier_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.npz"
            with self.assertRaisesRegex(ValueError, "record identifier"):
                experiment.write_probability_artifact(
                    path, {"resultnos": np.zeros(experiment.EXPECTED_ROWS)}
                )

    def test_main_has_no_prior_prediction_input(self) -> None:
        source = inspect.getsource(experiment.main).casefold()
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("grouped_oof.npz", source)
        self.assertNotIn("prior-prediction", source)


if __name__ == "__main__":
    unittest.main()
