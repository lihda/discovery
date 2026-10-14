from __future__ import annotations

import inspect
import unittest

from scripts import experiment_r3_exact300_auxiliary_model_sweep as experiment


class R3Exact300AuxiliaryModelSweepTests(unittest.TestCase):
    def test_candidate_family_is_compact_unique_and_predeclared(self) -> None:
        specs = experiment.candidate_specs()
        names = [spec.name for spec in specs]
        self.assertEqual(len(names), len(set(names)))
        self.assertLessEqual(len(names), 30)
        self.assertEqual({spec.family for spec in specs}, set(experiment.FAMILIES))
        self.assertTrue(all(spec.matrix in {"curation_all", "combined"} for spec in specs))

    def test_fold_parser(self) -> None:
        self.assertEqual(experiment.parse_folds("all"), (1, 2, 3, 4, 5))
        self.assertEqual(experiment.parse_folds("3,1,3"), (1, 3))
        with self.assertRaises(Exception):
            experiment.parse_folds("0,2")

    def test_family_parser(self) -> None:
        self.assertEqual(experiment.parse_families("all"), experiment.FAMILIES)
        self.assertEqual(experiment.parse_families("linear,kernel"), ("kernel", "linear"))
        with self.assertRaises(Exception):
            experiment.parse_families("unknown")

    def test_main_has_no_prior_prediction_input(self) -> None:
        source = inspect.getsource(experiment.main)
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("PMAGRESULT", source)
        self.assertNotIn("ROCKMAG", source)


if __name__ == "__main__":
    unittest.main()
