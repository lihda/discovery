from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import experiment_r3_exact300_historical_missingness as experiment


class R3Exact300HistoricalMissingnessTests(unittest.TestCase):
    def test_historical_features_keep_required_families_and_drop_target_style(self) -> None:
        names = [
            "publication_year_normalized",
            "journal_gji",
            "section_methods_present",
            "author_count_log",
            "year_x_article_method_total",
            "target_count_log",
            "rockname_characters_log",
        ]
        indices, retained = experiment.historical_feature_indices(names)
        self.assertEqual(indices.tolist(), [0, 1, 2, 3, 4])
        self.assertNotIn("target_count_log", retained)
        self.assertNotIn("rockname_characters_log", retained)

    def test_balanced_inner_folds_keep_every_refno_together(self) -> None:
        records = [
            {"REFNO": refno}
            for refno in ("A", "A", "A", "B", "B", "C", "D", "E", "F", "G")
        ]
        indices = np.arange(len(records), dtype=np.int64)
        folds = experiment.balanced_group_fold_ids(records, indices)
        self.assertEqual(set(int(value) for value in folds), {0, 1, 2, 3})
        by_refno: dict[str, set[int]] = {}
        for record, fold in zip(records, folds):
            by_refno.setdefault(record["REFNO"], set()).add(int(fold))
        self.assertTrue(all(len(values) == 1 for values in by_refno.values()))

    def test_gate_can_only_turn_positive_predictions_negative(self) -> None:
        stage1 = np.asarray([0.2, 0.6, 0.8, 0.9])
        missing = np.asarray([0.9, 0.4, 0.8, 0.95])
        adjusted, flips = experiment.apply_gate(stage1, missing, 0.85)
        self.assertEqual(flips.tolist(), [False, False, False, True])
        self.assertLess(adjusted[3], 0.5)
        self.assertFalse(bool(flips[stage1 < 0.5].any()))
        self.assertTrue(np.array_equal(adjusted[stage1 < 0.5], stage1[stage1 < 0.5]))

    def test_inner_gate_selection_can_select_a_conservative_correction(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        stage1 = np.asarray([0.9, 0.8, 0.8, 0.7])
        missing = np.asarray([0.1, 0.2, 0.9, 0.8])
        threshold, audit = experiment.select_gate(labels, stage1, missing)
        adjusted, _ = experiment.apply_gate(stage1, missing, threshold)
        self.assertEqual((adjusted >= 0.5).astype(int).tolist(), labels.tolist())
        self.assertEqual(audit["selected"]["accuracy"], 1.0)

    def test_joint_selection_uses_only_predeclared_strong_thresholds(self) -> None:
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        stage1 = np.asarray([0.9, 0.8, 0.8, 0.7])
        missing = {
            threshold: np.asarray([0.1, 0.2, 0.9, 0.8])
            for threshold in experiment.STRONG_STAGE1_THRESHOLDS
        }
        strong, gate, audit = experiment.select_strong_threshold_and_gate(
            labels, stage1, missing
        )
        self.assertIn(strong, experiment.STRONG_STAGE1_THRESHOLDS)
        self.assertIn(gate, experiment.GATE_THRESHOLDS)
        self.assertEqual(audit["selected"]["selected_gate_metric"]["accuracy"], 1.0)

    def test_probability_artifact_rejects_identifier_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.npz"
            with self.assertRaisesRegex(ValueError, "record identifier"):
                experiment.write_probability_artifact(
                    path, {"refnos": np.zeros(experiment.EXPECTED_ROWS)}
                )

    def test_main_has_no_prior_prediction_input(self) -> None:
        source = inspect.getsource(experiment.main).casefold()
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("oof_probabilities.npz", source)
        self.assertNotIn("grouped_oof.npz", source)


if __name__ == "__main__":
    unittest.main()
