from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import experiment_r3_exact300_field_signatures as experiment


class R3Exact300FieldSignatureTests(unittest.TestCase):
    def test_kb_boundary_rejects_record_answer_mapping(self) -> None:
        artifact = {
            "schema_version": "r3_positive_rockmag_expert_kb_v2",
            "runtime_policy": {
                "contains_record_answer_mapping": True,
                "source_table_access": False,
                "label_access": False,
            },
            "provenance": {
                "raw_field_values_retained": False,
                "row_identity_mapping_retained": False,
                "source_quotes_retained": False,
                "held_out_field_cells_opened": False,
                "held_out_labels_opened": False,
                "prior_predictions_or_evaluations_used": False,
            },
        }
        with self.assertRaisesRegex(ValueError, "record-free"):
            experiment.validate_kb_boundary(artifact)

    def test_ambiguous_abbreviation_requires_magnetic_context(self) -> None:
        abbreviations = [
            {
                "abbreviation": "IRM",
                "concept_id": "irm_acquisition",
                "pattern": experiment.re.compile(r"\bIRM\b", experiment.re.I),
            }
        ]
        retained, suppressed = experiment.guarded_abbreviation_counts(
            "The IRM office issued a report.", abbreviations
        )
        self.assertEqual(retained, {})
        self.assertEqual(suppressed, 1)
        retained, suppressed = experiment.guarded_abbreviation_counts(
            "IRM acquisition curves were measured to 1 tesla.", abbreviations
        )
        self.assertEqual(retained, {"IRM": 1})
        self.assertEqual(suppressed, 0)

    def test_nonqualifying_carrier_and_unblocking_without_method(self) -> None:
        configured = {
            "carrier_without_method",
            "tub_unblocking_interval_without_independent_method",
        }
        classes = experiment.passage_nonqualifying_classes(
            "Magnetite was noted and Tub > 600 C.", {"carrier"}, configured
        )
        self.assertEqual(
            classes,
            {
                "carrier_without_method",
                "tub_unblocking_interval_without_independent_method",
            },
        )
        classes = experiment.passage_nonqualifying_classes(
            "IRM acquisition identified magnetite; Tub > 600 C.",
            {"method", "carrier"},
            configured,
        )
        self.assertEqual(classes, set())

    def test_scope_assignment_does_not_borrow_another_target(self) -> None:
        passages = [
            {
                "assigned_target": "alpha",
                "target_distances": {"alpha": 10, "beta": 20},
                "own_likely": True,
                "external_only": False,
            },
            {
                "assigned_target": "",
                "target_distances": {"beta": 100},
                "own_likely": True,
                "external_only": False,
            },
        ]
        beta = experiment.target_scope_passages(passages, "beta")
        self.assertEqual(beta["target_local"], [passages[1]])

    def test_first_three_stop_gate_requires_mean_regression_and_no_fold3_gain(self) -> None:
        self.assertTrue(experiment.first3_clear_regression([0.81, 0.80, 0.78]))
        self.assertFalse(experiment.first3_clear_regression([0.84, 0.84, 0.78]))
        self.assertFalse(experiment.first3_clear_regression([0.80, 0.80, 0.80]))

    def test_cache_rejects_identifier_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            with self.assertRaisesRegex(ValueError, "record identifier"):
                experiment.write_npz(path, {"resultnos": np.zeros(3)})

    def test_main_has_no_prior_prediction_input(self) -> None:
        source = inspect.getsource(experiment.main).casefold()
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("grouped_oof.npz", source)
        self.assertNotIn("auxiliary_training_canonical_v2", source)


if __name__ == "__main__":
    unittest.main()
