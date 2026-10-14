from __future__ import annotations

import inspect
import unittest

import numpy as np

from scripts import experiment_r3_exact300_paper_consistency as experiment


class R3Exact300PaperConsistencyTests(unittest.TestCase):
    def test_candidate_order_prefers_identity_on_exact_tie(self) -> None:
        specs = experiment.smoothing_specs()
        self.assertEqual(specs[0].name, "identity")
        self.assertEqual(len({spec.name for spec in specs}), len(specs))
        labels = np.asarray([0, 1], dtype=np.int8)
        probability = np.asarray([0.2, 0.8], dtype=np.float64)
        selected, metrics = experiment.select_smoothing(
            labels, probability, ["a", "b"], specs
        )
        self.assertEqual(selected.name, "identity")
        self.assertTrue(all(item["accuracy"] == 1.0 for item in metrics.values()))

    def test_min_blend_and_majority_operate_only_within_paper(self) -> None:
        probability = np.asarray([0.8, 0.4, 0.7, 0.6, 0.2], dtype=np.float64)
        groups = ["a", "a", "b", "b", "b"]
        minimum = experiment.apply_smoothing(
            probability,
            groups,
            experiment.SmoothingSpec("min", "min", 1.0),
        )
        self.assertTrue(np.allclose(minimum, [0.4, 0.4, 0.2, 0.2, 0.2]))
        majority = experiment.apply_smoothing(
            probability,
            groups,
            experiment.SmoothingSpec("majority", "majority", 1.0),
        )
        self.assertTrue(np.allclose(majority[:2], probability[:2]))
        self.assertTrue(np.allclose(majority[2:], 1.0))

    def test_invalid_probability_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite probability"):
            experiment.apply_smoothing(
                np.asarray([np.nan]),
                ["a"],
                experiment.SmoothingSpec("identity", "identity", 0.0),
            )

    def test_main_has_no_prior_prediction_input(self) -> None:
        source = inspect.getsource(experiment.main)
        self.assertNotIn('"--oof-probabilities"', source)
        self.assertNotIn('"--prior-report"', source)
        self.assertIn("FROZEN_BEFORE_LABEL_ACCESS", source)

    def test_router_oof_declares_nested_group_exclusion(self) -> None:
        source = inspect.getsource(experiment.fit_router_level_oof)
        self.assertIn(
            "router_validation_excluded_from_all_nested_expert_fits", source
        )
        self.assertIn("nested expert/router validation group leakage", source)


if __name__ == "__main__":
    unittest.main()
