from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts import evaluate_r3_style_fusion_single872_oof_v2 as evaluation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/knowledge/r3_style_fusion_evaluation_v2.json"


class StyleFusionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_contract(self) -> None:
        evaluation.validate_config(self.config)
        self.assertFalse(
            self.config["runtime_boundary"]["pmagresult_or_rockmag_read"]
        )
        self.assertFalse(
            self.config["runtime_boundary"]["outer_test_labels_used_for_training_or_selection"]
        )

    def test_no_strict_base_is_declared_convex_combination(self) -> None:
        components = {
            "domain_mean4": np.asarray([0.2, 0.8]),
            "query_pca64_logreg": np.asarray([0.4, 0.6]),
            "front_word_logreg": np.asarray([0.7, 0.3]),
        }
        observed = evaluation.base_fusion(components, self.config)
        weights = self.config["base"]["weights"]
        expected = sum(weights[name] * value for name, value in components.items())
        np.testing.assert_allclose(observed, expected)

    def test_primary_fusion_is_fixed_at_fifteen_percent_knowledge(self) -> None:
        base = np.asarray([0.2, 0.8])
        knowledge = np.asarray([0.6, 0.4])
        observed = evaluation.primary_fusion(base, knowledge, self.config)
        np.testing.assert_allclose(observed, 0.85 * base + 0.15 * knowledge)

    def test_knowledge_experts_fit_only_supplied_rows(self) -> None:
        rng = np.random.default_rng(357)
        matrix = rng.normal(size=(30, 5))
        labels = np.asarray([0, 1] * 15, dtype=np.int8)
        train = np.arange(24)
        test = np.arange(24, 30)
        output = evaluation.fit_knowledge_experts(
            matrix, train, test, labels, self.config
        )
        self.assertEqual(set(output), set(evaluation.KNOWLEDGE_EXPERTS))
        for probability in output.values():
            self.assertEqual(probability.shape, (6,))
            self.assertTrue(np.isfinite(probability).all())

    def test_config_rejects_posthoc_threshold_search(self) -> None:
        changed = json.loads(json.dumps(self.config))
        changed["runtime_boundary"]["posthoc_threshold_search_used"] = True
        with self.assertRaises(evaluation.StyleFusionError):
            evaluation.validate_config(changed)


if __name__ == "__main__":
    unittest.main()
