from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np
import scipy.sparse as sp

from scripts import build_r3_evidence_mil_bags as mil_freeze
from scripts.evaluate_r3_evidence_mil_single872_oof import (
    COMPONENT_NAMES,
    EXPECTED_STACK_FEATURES,
    EvidencePrediction,
    MilEvaluationError,
    aggregate_instance_probabilities,
    bag_feature_mean_max,
    build_hard_em_training_rows,
    build_stack_features,
    fit_hard_em,
    load_evaluation_config,
    select_top_k_instances,
    sparse_matrix_sha256,
    validate_bag_alignment,
    validate_trusted_instance,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/knowledge/r3_evidence_mil_evaluation_v1.json"


class EvidenceMilEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_evaluation_config(CONFIG)

    def test_config_freezes_sources_and_fixed_stack_contract(self) -> None:
        self.assertEqual(
            [item["source_id"] for item in self.config["source_refs"]],
            [
                "PATEL_ET_AL_ACL_2025_R2E",
                "TAN_ET_AL_SIGIR_2025_EDU_FILTERING",
                "DU_ET_AL_ICLR_2025_PSMIL",
            ],
        )
        self.assertEqual(
            tuple(self.config["stack"]["component_logits"]), COMPONENT_NAMES
        )
        self.assertEqual(
            self.config["stack"]["expected_dimensions"], EXPECTED_STACK_FEATURES
        )
        self.assertTrue(
            all(
                "not" in item["adaptation_boundary"]
                and "direct reproduction" in item["adaptation_boundary"]
                for item in self.config["source_refs"]
            )
        )

    def test_hard_em_top_k_selection_is_stable_and_bag_local(self) -> None:
        indptr = np.asarray([0, 3, 5], dtype=np.int64)
        scores = np.asarray([0.9, 0.9, 0.2, 0.3, 0.8], dtype=np.float64)
        selected = select_top_k_instances(scores, indptr, [0, 1], 2)
        np.testing.assert_array_equal(selected[0], np.asarray([0, 1]))
        np.testing.assert_array_equal(selected[1], np.asarray([4, 3]))

    def test_hard_em_training_rows_give_every_nonempty_bag_total_weight_one(self) -> None:
        indptr = np.asarray([0, 2, 5, 6], dtype=np.int64)
        labels = np.asarray([0, 1, 1], dtype=np.int8)
        selected = {
            1: np.asarray([3, 4], dtype=np.int64),
            2: np.asarray([5], dtype=np.int64),
        }
        rows = build_hard_em_training_rows([0, 1, 2], labels, indptr, selected)
        np.testing.assert_array_equal(rows.instance_indices, [0, 1, 3, 4, 5])
        np.testing.assert_array_equal(rows.labels, [0, 0, 1, 1, 1])
        for bag in (0, 1, 2):
            self.assertAlmostEqual(
                float(rows.sample_weight[rows.bag_indices == bag].sum()), 1.0
            )

    def test_instance_aggregation_supports_max_top2mean_and_empty_bags(self) -> None:
        indptr = np.asarray([0, 2, 5, 5], dtype=np.int64)
        probability = np.asarray([0.1, 0.8, 0.2, 0.9, 0.7], dtype=np.float64)
        maximum = aggregate_instance_probabilities(
            probability, indptr, [0, 1, 2], "max"
        )
        top2 = aggregate_instance_probabilities(
            probability, indptr, [0, 1, 2], "top2mean"
        )
        np.testing.assert_allclose(maximum.probability, [0.8, 0.9, 0.5])
        np.testing.assert_allclose(top2.probability, [0.45, 0.8, 0.5])
        np.testing.assert_allclose(maximum.instance_mean, [0.45, 0.6, 0.5])
        np.testing.assert_allclose(maximum.instance_max, [0.8, 0.9, 0.5])
        np.testing.assert_array_equal(maximum.top_global_index, [1, 3, -1])

    def test_three_round_hard_em_fit_returns_bag_local_top_instance(self) -> None:
        names = mil_freeze.instance_feature_names()
        features = np.zeros((8, len(names)), dtype=np.float32)
        features[4:, names.index("trusted_any")] = 1.0
        features[4:, names.index("trusted_category_fraction")] = 0.5
        features[4:, names.index("event_score_scaled")] = 0.8
        indptr = np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        prediction = fit_hard_em(
            features=features,
            feature_names=names,
            bag_indptr=indptr,
            bag_labels=labels,
            train_bags=np.asarray([0, 1, 2], dtype=np.int64),
            test_bags=np.asarray([3], dtype=np.int64),
            spec=self.config["components"]["hard_em"][0],
            common=self.config["components"]["hard_em_common"],
        )
        self.assertEqual(prediction.probability.shape, (1,))
        self.assertTrue(np.isfinite(prediction.probability).all())
        self.assertIn(int(prediction.top_global_index[0]), (6, 7))

    def test_static_bag_mean_max_uses_zero_for_an_empty_bag(self) -> None:
        features = np.asarray([[1.0, 3.0], [3.0, 1.0]], dtype=np.float32)
        indptr = np.asarray([0, 2, 2], dtype=np.int64)
        output = bag_feature_mean_max(features, indptr)
        np.testing.assert_allclose(output[0], [2.0, 2.0, 3.0, 3.0])
        np.testing.assert_array_equal(output[1], np.zeros(4))

    def test_sparse_hash_is_canonical_and_detects_content_changes(self) -> None:
        left = sp.csr_matrix(np.asarray([[0.0, 1.0], [2.0, 0.0]]))
        right = left.copy()
        self.assertEqual(sparse_matrix_sha256(left), sparse_matrix_sha256(right))
        right = sp.csr_matrix(np.asarray([[1.0, 1.0], [2.0, 0.0]]))
        self.assertNotEqual(sparse_matrix_sha256(left), sparse_matrix_sha256(right))

    def test_bag_alignment_rejects_fold_or_identity_drift(self) -> None:
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "Alpha"},
            {"REFNO": "2", "RESULTNO": "20", "ROCKNAME": "Beta"},
        ]
        indptr = np.asarray([0, 1, 3], dtype=np.int64)
        arrays = {
            "bag_indptr": indptr,
            "instance_features": np.zeros((3, 61), dtype=np.float32),
            "fold_ids": np.asarray([1, 2], dtype=np.int8),
            "source_row_indices": np.asarray([0, 1], dtype=np.int32),
            "refnos": np.asarray(["1", "2"]),
            "resultnos": np.asarray(["10", "20"]),
            "rocknames": np.asarray(["Alpha", "Beta"]),
            "document_groups": np.asarray(["a" * 64, "b" * 64]),
            "instance_bag_indices": np.asarray([0, 1, 1], dtype=np.int32),
            "instance_pages": np.asarray([1, 2, 3], dtype=np.int32),
            "instance_scale_ids": np.asarray([0, 0, 1], dtype=np.int8),
            "instance_route_ids": np.asarray([0, 1, 2], dtype=np.int8),
            "instance_chunk_ids": np.asarray(["c" * 64, "d" * 64, "e" * 64]),
            "instance_text_sha256": np.asarray(
                ["f" * 64, "0" * 64, "1" * 64]
            ),
        }
        folds = np.asarray([1, 2], dtype=np.int8)
        groups = np.asarray(["a" * 64, "b" * 64], dtype=object)
        validate_bag_alignment(arrays, records, folds, groups, [f"f{i}" for i in range(61)])
        bad = dict(arrays, fold_ids=np.asarray([2, 2], dtype=np.int8))
        with self.assertRaisesRegex(MilEvaluationError, "align"):
            validate_bag_alignment(
                bad, records, folds, groups, [f"f{i}" for i in range(61)]
            )

    def test_trusted_constraints_are_recomputed_from_receipt(self) -> None:
        names = mil_freeze.instance_feature_names()
        feature = np.zeros(len(names), dtype=np.float32)
        feature[names.index("trusted_method")] = 1.0
        feature[names.index("trusted_category_fraction")] = 0.25
        feature[names.index("trusted_any")] = 1.0
        instance = {
            "section_class": "methods",
            "lexical_counts": {
                "method": 1,
                "diagnostic": 0,
                "carrier": 0,
                "bridge": 0,
            },
            "counter_counts": {
                "demagnetization_only": 0,
                "external_attribution": 0,
                "future_recommendation": 0,
            },
            "event_signals": {
                "external": False,
                "future": False,
                "method_negation": False,
                "first_person": True,
                "passive_execution": False,
                "procedure": False,
                "result_verb": False,
                "quantitative": False,
                "own_likely": True,
                "citation": False,
            },
            "trusted_categories": {
                "method": True,
                "diagnostic": False,
                "carrier": False,
                "bridge": False,
            },
            "exclusion_codes": [],
        }
        validate_trusted_instance(instance, feature, names)
        bad = dict(instance, trusted_categories=dict(instance["trusted_categories"], method=False))
        with self.assertRaisesRegex(MilEvaluationError, "trusted-category"):
            validate_trusted_instance(bad, feature, names)

    def test_stack_input_is_exactly_131_dimensions(self) -> None:
        components = {
            name: np.asarray([0.5, 0.8], dtype=np.float64)
            for name in COMPONENT_NAMES
        }
        em = {
            name: EvidencePrediction(
                probability=components[name],
                instance_mean=np.asarray([0.5, 0.2]),
                instance_max=np.asarray([0.5, 0.9]),
                top_global_index=np.asarray([-1, 2], dtype=np.int32),
                top_probability=np.asarray([0.5, 0.9]),
            )
            for name in COMPONENT_NAMES[-2:]
        }
        static = np.arange(2 * 122, dtype=np.float64).reshape(2, 122) / 244.0
        matrix = build_stack_features(components, em, static, self.config)
        self.assertEqual(matrix.shape, (2, EXPECTED_STACK_FEATURES))
        np.testing.assert_allclose(matrix[0, :5], np.zeros(5), atol=1e-12)
        np.testing.assert_allclose(matrix[0, 5:9], [0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(matrix[:, 9:], static)


if __name__ == "__main__":
    unittest.main()
