from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_r3_query_adaptive_single872_oof import (
    CandidateRef,
    apply_candidate,
    candidate_refs,
    paired_change,
    raw_specs,
    select_candidate,
    validate_evaluation_contract,
)


class QueryAdaptiveEvaluationTests(unittest.TestCase):
    def test_candidate_contract_covers_hardcoded_evaluator_values(self) -> None:
        config = {
            "training_scopes": ["all_pdf_training", "single_target_training"],
            "feature_families": [
                "v4",
                "retrieval_compact",
                "v4_plus_retrieval_compact",
                "retrieval_pooled",
                "retrieval_compact_plus_pooled",
                "v4_plus_retrieval_compact_plus_pooled",
            ],
            "inner_folds": 3,
            "inner_seed": 35720260761,
            "threshold": 0.5,
            "logistic_c_values": [0.001, 0.01, 0.1, 1.0],
            "pca_dimensions": [32, 64, 128],
            "random_forest": {},
            "extra_trees": {},
            "hist_gradient_boosting": {},
            "fusion_alphas": [0.2],
        }
        validate_evaluation_contract(
            config, raw_specs(), config["feature_families"]
        )
        bad = dict(config, threshold=0.55)
        with self.assertRaises(ValueError):
            validate_evaluation_contract(
                bad, raw_specs(), config["feature_families"]
            )

    def test_candidate_family_is_fixed_and_contains_both_training_scopes(self) -> None:
        specs = raw_specs()
        candidates = candidate_refs(specs, [0.2, 0.5, 0.8])
        self.assertEqual(len({spec.name for spec in specs}), len(specs))
        self.assertEqual(
            {candidate.scope for candidate in candidates},
            {"all_pdf_training", "single_target_training"},
        )
        self.assertTrue(any(candidate.kind == "fusion" for candidate in candidates))
        self.assertFalse(
            any("pca64" in candidate.raw_name for candidate in candidates)
        )

    def test_fusion_and_inner_selection_use_only_given_rows(self) -> None:
        raw = {
            "all_pdf_training": {
                "v4_rf": np.asarray([0.9, 0.1, 0.1, 0.9]),
                "retrieval": np.asarray([0.9, 0.9, 0.1, 0.1]),
            }
        }
        fusion = CandidateRef(
            "fusion", "all_pdf_training", "fusion", "retrieval", 0.5
        )
        np.testing.assert_allclose(
            apply_candidate(fusion, raw), np.asarray([0.9, 0.5, 0.1, 0.5])
        )
        candidates = (
            CandidateRef("base", "all_pdf_training", "raw", "v4_rf"),
            CandidateRef("retrieval", "all_pdf_training", "raw", "retrieval"),
        )
        labels = np.asarray([1, 1, 0, 0], dtype=np.int8)
        selected, _ = select_candidate(
            candidates, raw, labels, np.asarray([0, 1, 2], dtype=np.int64)
        )
        self.assertEqual(selected.name, "retrieval")

    def test_paired_change_counts_repairs_and_degradations(self) -> None:
        labels = np.asarray([1, 0, 1, 0], dtype=np.int8)
        baseline = np.asarray([0.6, 0.8, 0.4, 0.2])
        candidate = np.asarray([0.7, 0.2, 0.8, 0.9])
        result = paired_change(labels, baseline, candidate)
        self.assertEqual(result["repairs"], 2)
        self.assertEqual(result["degradations"], 1)
        self.assertEqual(result["net_repairs"], 1)


if __name__ == "__main__":
    unittest.main()
