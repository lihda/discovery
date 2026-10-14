from __future__ import annotations

import inspect
import unittest

import numpy as np

from scripts import experiment_r3_exact300_paper_consistency as v1
from scripts import experiment_r3_exact300_paper_consistency_v2 as v2


class R3Exact300PaperConsistencyV2Tests(unittest.TestCase):
    def test_invariant_branch_is_label_free_and_returns_identity_equivalence(self) -> None:
        probability = np.asarray([0.8, 0.7, 0.2], dtype=np.float64)
        invariant, audit, candidates = v2.label_free_decision_invariance(
            probability, ["a", "b", "c"], v1.smoothing_specs()
        )
        self.assertTrue(invariant)
        identity = candidates["identity"] >= 0.5
        self.assertTrue(
            all(np.array_equal(value >= 0.5, identity) for value in candidates.values())
        )
        self.assertTrue(
            all(item["rows_changed_vs_identity"] == 0 for item in audit.values())
        )

    def test_noninvariant_branch_is_detected_without_labels(self) -> None:
        probability = np.asarray([0.8, 0.4], dtype=np.float64)
        invariant, audit, _ = v2.label_free_decision_invariance(
            probability, ["a", "a"], v1.smoothing_specs()
        )
        self.assertFalse(invariant)
        self.assertGreater(audit["paper_min_blend_100"]["rows_changed_vs_identity"], 0)

    def test_runtime_and_v1_reproduction_are_frozen(self) -> None:
        self.assertEqual(
            v1.CANONICAL_RUNTIME,
            {"python": "3.9.6", "numpy": "2.0.2", "sklearn": "1.6.1"},
        )
        self.assertEqual(v1.CANONICAL_V1_CORRECT_ROWS, {1: 55, 2: 55, 3: 49, 4: 54, 5: 55})

    def test_v2_does_not_accept_prior_predictions(self) -> None:
        source = inspect.getsource(v2.main)
        self.assertNotIn('"--oof-probabilities"', source)
        self.assertNotIn('"--prior-report"', source)
        self.assertIn("outer_test_labels_used_for_branch_or_smoother_selection", source)


if __name__ == "__main__":
    unittest.main()
