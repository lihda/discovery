from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts import evaluate_r3_residual_gate_single872_oof as residual


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "data/knowledge/r3_residual_gate_evaluation_v1.json"


def components(rows: int = 4) -> dict[str, np.ndarray]:
    base = {
        "v4_random_forest": np.asarray([0.2, 0.8, 0.4, 0.6]),
        "domain_hgb": np.asarray([0.3, 0.7, 0.45, 0.55]),
        "domain_rf": np.asarray([0.25, 0.75, 0.4, 0.6]),
        "domain_extra": np.asarray([0.35, 0.65, 0.48, 0.52]),
        "domain_logreg": np.asarray([0.2, 0.8, 0.42, 0.58]),
        "domain_mean4": np.asarray([0.275, 0.725, 0.4375, 0.5625]),
        "query_pca64_logreg": np.asarray([0.1, 0.9, 0.7, 0.3]),
        "query_compact_hgb": np.asarray([0.15, 0.85, 0.65, 0.35]),
        "front_word_logreg": np.asarray([0.05, 0.95, 0.75, 0.25]),
        "trusted_hgb": np.asarray([0.2, 0.8, 0.8, 0.2]),
        "trusted_extra": np.asarray([0.25, 0.75, 0.9, 0.1]),
    }
    return {name: value[:rows].copy() for name, value in base.items()}


class ResidualGateEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_frozen_config_contract(self) -> None:
        residual.validate_config(self.config)
        self.assertEqual(tuple(self.config["candidate_order"]), residual.CANDIDATE_ORDER)
        self.assertFalse(
            self.config["runtime_boundary"]["pmagresult_or_rockmag_read"]
        )

    def test_fixed_fusion_is_the_declared_convex_combination(self) -> None:
        values = components()
        observed = residual.fixed_fusion(values, self.config)
        weights = self.config["fixed_fusion"]["weights"]
        trusted = (values["trusted_hgb"] + values["trusted_extra"]) / 2
        expected = (
            weights["domain_mean4"] * values["domain_mean4"]
            + weights["query_pca64_logreg"] * values["query_pca64_logreg"]
            + weights["front_word_logreg"] * values["front_word_logreg"]
            + weights["trusted_expert_mean"] * trusted
        )
        np.testing.assert_allclose(observed, expected)

    def test_component_meta_matrix_is_low_dimensional_and_finite(self) -> None:
        trusted_meta = np.zeros((4, 8), dtype=np.float64)
        matrix = residual.component_meta_features(
            components(), trusted_meta, self.config["stack"]["probability_clip"]
        )
        self.assertEqual(matrix.shape, (4, 24))
        self.assertTrue(np.isfinite(matrix).all())

    def test_evidence_probability_is_a_fixed_median(self) -> None:
        values = components()
        observed = residual.evidence_probability(
            values, self.config["residual_gate"]
        )
        selected = np.column_stack(
            [
                values[name]
                for name in self.config["residual_gate"]["evidence_components"]
            ]
        )
        np.testing.assert_allclose(observed, np.median(selected, axis=1))

    def test_gate_never_overrides_a_high_margin_base(self) -> None:
        spec = dict(self.config["residual_gate"])
        spec["gate_probability_threshold"] = 0.5
        base = np.asarray([0.1, 0.45])
        evidence = np.asarray([0.9, 0.9])
        meta = np.zeros((2, 24), dtype=np.float64)
        fitted = residual.GateFit(
            residual.LogisticFit(None, None, constant=1.0), True, 30
        )
        output, switched, quality = residual.apply_gate(
            fitted, meta, base, evidence, np.ones(2, dtype=bool), spec
        )
        self.assertFalse(switched[0])
        self.assertTrue(switched[1])
        self.assertEqual(output[0], base[0])
        self.assertGreaterEqual(quality[1], 0.5)

    def test_disabled_gate_is_exactly_the_base(self) -> None:
        base = np.asarray([0.4, 0.6])
        output, switched, quality = residual.apply_gate(
            residual.GateFit(None, False, 2),
            np.zeros((2, 24)),
            base,
            np.asarray([0.8, 0.2]),
            np.ones(2, dtype=bool),
            self.config["residual_gate"],
        )
        np.testing.assert_array_equal(output, base)
        self.assertFalse(switched.any())
        self.assertTrue((quality == 0).all())

    def test_candidate_guard_requires_minimum_gain(self) -> None:
        labels = np.asarray([0, 1, 0, 1], dtype=np.int8)
        baseline = np.asarray([0.1, 0.9, 0.1, 0.9])
        candidates = {
            "domain_mean4": baseline,
            "fixed_fusion": baseline.copy(),
            "low_dimensional_stack": baseline.copy(),
            "bounded_residual_gate": baseline.copy(),
        }
        selected, _, audit = residual.choose_candidate(
            labels, candidates, self.config
        )
        self.assertEqual(selected, "domain_mean4")
        self.assertFalse(audit["accepted_non_baseline"])

    def test_gate_training_refuses_too_few_disagreements(self) -> None:
        spec = dict(self.config["residual_gate"])
        spec["minimum_disagreement_rows"] = 4
        gate = residual.fit_gate(
            np.zeros((3, 24)),
            np.asarray([0.4, 0.4, 0.4]),
            np.asarray([0.6, 0.6, 0.6]),
            np.asarray([1, 0, 1]),
            spec,
        )
        self.assertFalse(gate.enabled)
        self.assertEqual(gate.disagreement_rows, 3)


if __name__ == "__main__":
    unittest.main()
