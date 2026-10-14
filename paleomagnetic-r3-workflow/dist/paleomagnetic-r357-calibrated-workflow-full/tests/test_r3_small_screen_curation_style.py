from __future__ import annotations

import inspect
import unittest

from scripts import experiment_r3_small_screen_curation_style as experiment


class R3SmallScreenCurationStyleTests(unittest.TestCase):
    def test_publication_year_prefers_journal_header(self) -> None:
        year, source = experiment.publication_year(
            "Geophys. J. R. astr. Soc. (1979) 58, 309-336\n"
            "Received 1978 December 4\n"
            "Palaeomagnetism in the Coronation Geosyncline"
        )
        self.assertEqual(year, 1979)
        self.assertEqual(source, "journal_parenthetical")

    def test_publication_year_falls_back_to_received(self) -> None:
        year, source = experiment.publication_year(
            "Relative age from paleomagnetism\nReceived December 14, 1989\n"
        )
        self.assertEqual(year, 1989)
        self.assertEqual(source, "received")

    def test_decade_bins_are_explicit(self) -> None:
        self.assertEqual(experiment.decade_name(1968), "pre1970")
        self.assertEqual(experiment.decade_name(1979), "1970s")
        self.assertEqual(experiment.decade_name(2015), "2010s")
        self.assertEqual(experiment.decade_name(None), "missing")

    def test_promotion_gate_requires_aggregate_and_fold_stability(self) -> None:
        protocol = {
            "baseline": {
                "strict_accuracy": 0.79375,
                "strict_fold_accuracy": [0.67, 0.80, 0.79, 0.81, 0.88],
            },
            "promotion_gate": {
                "minimum_aggregate_accuracy": 0.81375,
                "minimum_absolute_gain_over_strict_baseline": 0.02,
                "minimum_folds_not_worse_than_strict_baseline": 3,
                "maximum_allowed_accuracy_regression_in_any_fold": 0.05,
            },
        }
        passing = {
            "aggregate": {"accuracy": 0.82},
            "folds": [
                {"accuracy": 0.70}, {"accuracy": 0.81}, {"accuracy": 0.80},
                {"accuracy": 0.82}, {"accuracy": 0.86},
            ],
        }
        failing = {
            "aggregate": {"accuracy": 0.82},
            "folds": [
                {"accuracy": 0.80}, {"accuracy": 0.81}, {"accuracy": 0.80},
                {"accuracy": 0.82}, {"accuracy": 0.70},
            ],
        }
        self.assertTrue(experiment.promotion_record(passing, protocol)["passed"])
        failed = experiment.promotion_record(failing, protocol)
        self.assertFalse(failed["passed"])
        self.assertFalse(
            failed["checks"]["maximum_allowed_accuracy_regression_in_any_fold"]
        )

    def test_experiment_has_no_prior_prediction_input(self) -> None:
        parameters = inspect.signature(experiment.main).parameters
        self.assertNotIn("predictions", parameters)
        self.assertNotIn("evaluation", parameters)
        source = inspect.getsource(experiment.main)
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("baseline.json", source)


if __name__ == "__main__":
    unittest.main()
