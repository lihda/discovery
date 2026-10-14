from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts import evaluate_r3_fulltext_fusion_single872_oof_v3 as evaluation


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/knowledge/r3_fulltext_fusion_evaluation_v3.json"


class FulltextFusionEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_frozen_config_contract(self) -> None:
        evaluation.validate_config(self.config)
        self.assertEqual(
            tuple(self.config["fulltext_expert"]["C_values"]), (0.5, 1.0, 2.0)
        )
        self.assertFalse(
            self.config["runtime_boundary"]["pmagresult_or_rockmag_read"]
        )

    def test_primary_is_fixed_five_percent_fulltext(self) -> None:
        v2 = np.asarray([0.2, 0.8])
        text = np.asarray([0.6, 0.4])
        observed = evaluation.primary_fusion(v2, text, self.config)
        np.testing.assert_allclose(observed, 0.95 * v2 + 0.05 * text)

    def test_fulltext_experts_fit_only_supplied_documents(self) -> None:
        documents = [
            ("magnetite carrier remanence " if index % 2 else "thermal treatment ")
            + f"sample document {index}"
            for index in range(30)
        ]
        labels = np.asarray([index % 2 for index in range(30)], dtype=np.int8)
        output = evaluation.fit_fulltext_experts(
            documents,
            np.arange(24),
            np.arange(24, 30),
            labels,
            self.config,
        )
        self.assertEqual(set(output), set(evaluation.FULLTEXT_EXPERTS))
        for probability in output.values():
            self.assertEqual(probability.shape, (6,))
            self.assertTrue(np.isfinite(probability).all())

    def test_config_rejects_test_selection(self) -> None:
        changed = json.loads(json.dumps(self.config))
        changed["protocol"]["outer_test_selection"] = True
        with self.assertRaises(evaluation.FulltextFusionError):
            evaluation.validate_config(changed)


if __name__ == "__main__":
    unittest.main()
