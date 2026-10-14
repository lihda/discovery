from __future__ import annotations

import inspect
import unittest

import numpy as np

from scripts import experiment_r3_exact300_bibliographic_policy_centered as experiment


class R3Exact300BibliographicPolicyCenteredTests(unittest.TestCase):
    def test_candidate_grid_is_small_and_predeclared(self) -> None:
        specs = experiment.candidate_specs()
        self.assertEqual(len(specs), 13)
        self.assertEqual(specs[0].name, "semantic_only")
        self.assertEqual(
            {spec.fusion_weight for spec in specs[1:]}, set(experiment.LAMBDAS)
        )
        self.assertEqual(
            {spec.support_gate for spec in specs[1:]},
            set(experiment.SUPPORT_GATES),
        )

    def test_posterior_equal_to_prior_is_neutral(self) -> None:
        posterior = np.asarray([0.7, 0.7])
        centered = experiment.centered_effect_probability(posterior, 0.7)
        np.testing.assert_allclose(centered, [0.5, 0.5])

    def test_centered_effect_moves_semantic_in_both_directions(self) -> None:
        semantic = np.asarray([0.6, 0.6])
        centered = np.asarray([0.8, 0.2])
        confidence = np.ones(2)
        fused = experiment.centered_fuse_probability(
            semantic, centered, confidence, 0.5, 0.0
        )
        self.assertGreater(fused[0], semantic[0])
        self.assertLess(fused[1], semantic[1])

    def test_absent_or_below_gate_support_is_not_applied(self) -> None:
        semantic = np.asarray([0.3, 0.7])
        centered = np.asarray([0.9, 0.1])
        confidence = np.asarray([0.0, 0.4])
        fused = experiment.centered_fuse_probability(
            semantic, centered, confidence, 1.0, 0.5
        )
        np.testing.assert_allclose(fused, semantic)

    def test_run_fold_does_not_accept_exact_labels(self) -> None:
        signature = inspect.signature(experiment.run_fold)
        self.assertNotIn("exact_labels", signature.parameters)
        self.assertNotIn("metric_record", inspect.getsource(experiment.run_fold))

    def test_protocol_declares_no_outer_threshold_selection(self) -> None:
        protocol = experiment.candidate_protocol()
        self.assertFalse(protocol["fusion"]["outer_label_threshold_selection"])
        self.assertFalse(
            protocol["centered_calibration"]["unsupported_policy_effect_applied"]
        )


if __name__ == "__main__":
    unittest.main()
