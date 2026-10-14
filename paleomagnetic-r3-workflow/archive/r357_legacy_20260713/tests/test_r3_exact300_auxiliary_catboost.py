from __future__ import annotations

import inspect
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import experiment_r3_exact300_auxiliary_catboost as experiment


class R3Exact300AuxiliaryCatBoostTests(unittest.TestCase):
    def test_candidate_family_is_compact_and_predeclared(self) -> None:
        specs = experiment.candidate_specs()
        self.assertEqual(len(specs), 12)
        self.assertEqual(len({spec.name for spec in specs}), 12)
        self.assertEqual({spec.matrix for spec in specs}, {"curation_all", "combined"})
        self.assertEqual({spec.depth for spec in specs}, {4, 6, 8})
        self.assertEqual(
            {spec.weight_mode for spec in specs},
            {"inverse_paper", "inverse_paper_class_balanced"},
        )
        protocol = experiment.freeze_protocol()
        self.assertEqual(protocol["candidate_count"], 12)
        self.assertFalse(protocol["inner_split"]["labels_used_to_assign_groups"])

    def test_inner_split_is_deterministic_and_refno_isolated(self) -> None:
        records = [
            {"REFNO": refno, "RESULTNO": str(index), "ROCKNAME": "x"}
            for index, refno in enumerate(("A", "A", "B", "C", "D", "E", "F"))
        ]
        indices = np.arange(len(records), dtype=np.int64)
        first_fit, first_validation = experiment.deterministic_inner_split(indices, records)
        second_fit, second_validation = experiment.deterministic_inner_split(indices, records)
        self.assertEqual(first_fit.tolist(), second_fit.tolist())
        self.assertEqual(first_validation.tolist(), second_validation.tolist())
        fit_refnos = {records[index]["REFNO"] for index in first_fit}
        validation_refnos = {records[index]["REFNO"] for index in first_validation}
        self.assertFalse(fit_refnos & validation_refnos)
        source = inspect.getsource(experiment.deterministic_inner_split)
        self.assertNotIn("labels", source)

    def test_inverse_paper_weights_equalize_paper_mass(self) -> None:
        records = [
            {"REFNO": "A", "RESULTNO": "1", "ROCKNAME": "x"},
            {"REFNO": "A", "RESULTNO": "2", "ROCKNAME": "y"},
            {"REFNO": "B", "RESULTNO": "3", "ROCKNAME": "z"},
            {"REFNO": "C", "RESULTNO": "4", "ROCKNAME": "w"},
        ]
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        indices = np.arange(4, dtype=np.int64)
        weights, audit = experiment.paper_weights(
            indices, records, labels, "inverse_paper"
        )
        self.assertAlmostEqual(float(weights[:2].sum()), float(weights[2]))
        self.assertAlmostEqual(float(weights[2]), float(weights[3]))
        self.assertAlmostEqual(audit["paper_mass_min"], audit["paper_mass_max"])

    def test_balanced_inverse_paper_weights_equalize_class_mass(self) -> None:
        records = [
            {"REFNO": "A", "RESULTNO": "1", "ROCKNAME": "x"},
            {"REFNO": "A", "RESULTNO": "2", "ROCKNAME": "y"},
            {"REFNO": "B", "RESULTNO": "3", "ROCKNAME": "z"},
            {"REFNO": "C", "RESULTNO": "4", "ROCKNAME": "w"},
        ]
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        weights, audit = experiment.paper_weights(
            np.arange(4, dtype=np.int64),
            records,
            labels,
            "inverse_paper_class_balanced",
        )
        self.assertAlmostEqual(float(weights[labels == 0].sum()), 2.0, places=6)
        self.assertAlmostEqual(float(weights[labels == 1].sum()), 2.0, places=6)
        self.assertEqual(audit["final_class_mass"], {"0": 2.0, "1": 2.0})

    def test_threshold_selection_uses_only_fixed_grid(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = np.asarray([0.1, 0.4, 0.6, 0.9], dtype=np.float64)
        threshold, metric = experiment.select_inner_threshold(labels, probabilities)
        self.assertIn(threshold, experiment.THRESHOLDS)
        self.assertEqual(metric["accuracy"], 1.0)
        self.assertEqual(metric["predicted_positive"], 2)

    def test_run_fold_does_not_accept_outer_labels(self) -> None:
        signature = inspect.signature(experiment.run_fold)
        self.assertNotIn("exact_labels", signature.parameters)
        source = inspect.getsource(experiment.run_fold)
        self.assertNotIn("metric_record", source)

    def test_prediction_artifact_has_no_labels_or_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            experiment.write_prediction_artifact(
                path,
                np.asarray([0.2, 0.8], dtype=np.float64),
                np.asarray([0, 1], dtype=np.int8),
                expected_rows=2,
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(
                    set(payload.files),
                    {"catboost_probability", "catboost_inner_selected_decision"},
                )

    def test_canonical_manifest_hash_and_runtime_boundary_are_frozen(self) -> None:
        self.assertEqual(
            experiment.CANONICAL_FOLD_MANIFEST_SHA256,
            "8f6ab8f1a071878e14d4db8e19a7ec0519ed612be107b0f0fdf0489dd59aaf2d",
        )
        source = inspect.getsource(experiment.main)
        freeze_position = source.index("base.atomic_json(freeze_path, freeze)")
        label_position = source.index("base.load_training_labels")
        self.assertLess(freeze_position, label_position)
        self.assertNotIn("baseline_probabilities", source)
        self.assertNotIn("oof_probabilities.npz", source)


if __name__ == "__main__":
    unittest.main()
