from __future__ import annotations

import inspect
import unittest

import numpy as np

from scripts import build_r3_exact300_component_bridge_cache as component_builder
from scripts import build_r3_exact300_own_study_cache as own_builder
from scripts import experiment_r3_exact300_component_bridge_auxiliary as component_eval
from scripts import experiment_r3_exact300_component_xgboost as xgb_eval
from scripts import experiment_r3_exact300_own_study_auxiliary as own_eval


class R3Exact300SourceAttributionFeatureTests(unittest.TestCase):
    def test_own_cache_schema_requires_all_float32_finite_matrices(self) -> None:
        arrays = {
            name: np.zeros((2, 1), dtype=np.float32)
            for name in own_builder.EXPECTED_ARRAY_NAMES
        }
        audit = own_builder.validate_arrays(arrays, 2)
        self.assertEqual(set(audit), own_builder.EXPECTED_ARRAY_NAMES)
        arrays["own_target"] = np.asarray([[0.0], [np.nan]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            own_builder.validate_arrays(arrays, 2)

    def test_component_cache_schema_requires_all_float32_matrices(self) -> None:
        arrays = {
            name: np.zeros((2, 1), dtype=np.float32)
            for name in component_builder.EXPECTED_ARRAY_NAMES
        }
        audit = component_builder.validate_arrays(arrays, 2)
        self.assertEqual(set(audit), component_builder.EXPECTED_ARRAY_NAMES)
        arrays["component_all"] = np.zeros((2, 1), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "not float32"):
            component_builder.validate_arrays(arrays, 2)

    def test_component_alignment_uses_exact_frozen_rockname(self) -> None:
        records = [
            {
                "REFNO": "1",
                "RESULTNO": "10",
                "ROCKNAME": "St. Audrie’s Bay Sediments",
            }
        ]
        papers = [
            {
                "refno": "1",
                "page_count": 2,
                "target_count": 1,
                "article": {
                    "relevant_scope_count": 2,
                    "knowledge_scope_count": 1,
                    "shared_knowledge_scope_count": 0,
                    "collective_knowledge_scope_count": 0,
                    "has_shared_evidence": False,
                    "concepts": ["carrier:magnetite"],
                },
                "targets": [
                    {
                        "canonical_target": "st audrie s bay sediments",
                        "rocknames": ["St. Audrie’s Bay Sediments"],
                        "resultnos": ["10"],
                        "direct_concepts": ["carrier:magnetite"],
                        "bridged_concepts": [],
                        "components": ["chrm"],
                        "features": {
                            "direct_scope_count": 1,
                            "linked_scope_count": 1,
                            "table_scope_count": 0,
                            "caption_scope_count": 0,
                            "sample_id_count": 0,
                            "component_count": 1,
                            "direct_concept_count": 1,
                            "bridged_concept_count": 0,
                        },
                        "gate": {
                            "sample_bridge": False,
                            "component_bridge": False,
                            "collective_target_scope": False,
                            "eligible_for_article_propagation": False,
                        },
                        "aliases": {
                            "identity_derived": [],
                            "source_confirmed_identity": [],
                            "source_derived": [],
                        },
                        "scope_links": {
                            "sample_bridge_scope_ids": [],
                            "component_bridge_scope_ids": [],
                            "bridged_scope_ids": [],
                        },
                    }
                ],
            }
        ]
        all_matrix, bridge_matrix, all_names, bridge_names, audit = (
            component_builder.build_component_features(records, papers)
        )
        self.assertEqual(all_matrix.shape, (1, len(all_names)))
        self.assertEqual(bridge_matrix.shape, (1, len(bridge_names)))
        self.assertEqual(audit["manifest_row_links"], 1)

    def test_candidate_sets_are_small_unique_and_predeclared(self) -> None:
        for specs, maximum in (
            (own_eval.candidate_specs(), 16),
            (component_eval.candidate_specs(), 12),
            (xgb_eval.candidate_specs(), 8),
        ):
            names = [spec.name for spec in specs]
            self.assertEqual(len(names), len(set(names)))
            self.assertLessEqual(len(names), maximum)

    def test_label_free_builders_do_not_open_training_labels(self) -> None:
        for builder in (own_builder, component_builder):
            source = inspect.getsource(builder.main)
            self.assertNotIn("load_training_labels", source)
            self.assertNotIn("DEFAULT_TARGETS", source)

    def test_evaluators_do_not_accept_prior_oof_input(self) -> None:
        for evaluator in (own_eval, component_eval, xgb_eval):
            source = inspect.getsource(evaluator.main)
            self.assertNotIn('"--baseline-probabilities"', source)
            self.assertNotIn('"--oof-probabilities"', source)


if __name__ == "__main__":
    unittest.main()
