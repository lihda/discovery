from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts import experiment_r3_single872_base_weight_nested_oof_v1 as experiment


ROOT = Path(__file__).resolve().parents[1]


class Single872BaseWeightExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.v2_config = json.loads(
            (ROOT / "data/knowledge/r3_style_fusion_evaluation_v2.json").read_text()
        )
        cls.v3_config = json.loads(
            (ROOT / "data/knowledge/r3_fulltext_fusion_evaluation_v3.json").read_text()
        )
        cls.current = np.asarray(
            [
                cls.v2_config["base"]["weights"][name]
                for name in experiment.COMPONENTS
            ]
        )

    def test_grid_covers_simplex_and_exact_current_weights(self) -> None:
        candidates = experiment.weight_candidates(self.current)
        self.assertEqual(candidates.shape, (5152, 3))
        np.testing.assert_allclose(candidates.sum(axis=1), 1.0)
        self.assertTrue((candidates >= 0.0).all())
        self.assertTrue(
            any(
                np.allclose(row, self.current, rtol=0.0, atol=1e-12)
                for row in candidates
            )
        )

    def test_base_probability_uses_declared_component_order(self) -> None:
        matrix = np.asarray([[0.2, 0.4, 0.8], [0.8, 0.4, 0.2]])
        observed = experiment.base_probability(matrix, np.asarray([0.5, 0.25, 0.25]))
        np.testing.assert_allclose(observed, [0.4, 0.55])

    def test_selection_uses_only_supplied_labels_and_predictions(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        perfect = np.asarray([0.1, 0.9, 0.1, 0.9])
        reversed_values = 1.0 - perfect
        components = np.column_stack((perfect, reversed_values, reversed_values))
        neutral = np.full(len(labels), 0.5)
        candidates = experiment.weight_candidates(self.current, step=0.5)

        weights, record = experiment.select_weights(
            candidates,
            labels,
            components,
            neutral,
            neutral,
            self.current,
            self.v2_config,
            self.v3_config,
        )

        np.testing.assert_allclose(weights, [1.0, 0.0, 0.0])
        self.assertEqual(record["final_correct"], len(labels))


if __name__ == "__main__":
    unittest.main()
