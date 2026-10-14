from __future__ import annotations

import inspect
import unittest

import numpy as np

from scripts import experiment_r3_exact300_bge_semantic_auxiliary as experiment


class R3Exact300BgeSemanticAuxiliaryTests(unittest.TestCase):
    def test_candidate_family_is_compact_unique_and_covers_every_view(self) -> None:
        specs = experiment.candidate_specs()
        names = [spec.name for spec in specs]
        self.assertEqual(len(names), len(set(names)))
        self.assertLessEqual(len(specs), 16)
        seen = {view for spec in specs for view in spec.semantic_views}
        self.assertEqual(seen, set(experiment.EXPECTED_BGE_VIEWS))
        self.assertTrue(any(spec.pca_components for spec in specs))
        self.assertEqual({spec.model for spec in specs}, {"logreg", "rf8", "hgb10"})

    def test_control_is_frozen_component_bridge_baseline(self) -> None:
        controls = [
            spec for spec in experiment.candidate_specs() if spec.name == experiment.CONTROL_NAME
        ]
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0].numeric_matrix, experiment.CONTROL_MATRIX)
        self.assertEqual(controls[0].semantic_views, ())
        self.assertEqual(controls[0].model, "rf8")

    def test_fold_parser(self) -> None:
        self.assertEqual(experiment.parse_folds("all"), (1, 2, 3, 4, 5))
        self.assertEqual(experiment.parse_folds("3,1,3"), (1, 3))
        with self.assertRaises(Exception):
            experiment.parse_folds("0,2")

    def test_pca_is_fit_only_from_supplied_training_rows(self) -> None:
        rng = np.random.default_rng(7)
        arrays = {
            "target_local": rng.normal(size=(12, 8)).astype(np.float32),
        }
        train = np.arange(9, dtype=np.int64)
        test = np.arange(9, 12, dtype=np.int64)
        first_train, first_test = experiment.semantic_fold_block(
            ("target_local",), 4, arrays, train, test
        )
        arrays["target_local"][test] += 1000.0
        second_train, second_test = experiment.semantic_fold_block(
            ("target_local",), 4, arrays, train, test
        )
        np.testing.assert_allclose(first_train, second_train, atol=1e-5)
        self.assertFalse(np.allclose(first_test, second_test))

    def test_evaluator_has_no_prior_oof_input(self) -> None:
        parameters = inspect.signature(experiment.evaluate).parameters
        self.assertEqual(tuple(parameters), ("args",))
        source = inspect.getsource(experiment.evaluate).casefold()
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("load_prior", source)
        self.assertNotIn("prior_oof", source)


if __name__ == "__main__":
    unittest.main()
