import unittest

import numpy as np

from scripts.build_r3_trace_page_graph_features import deterministic_refno_fold_ids
from scripts.experiment_r3_trace_clean_nested_oof import (
    _pairwise_examples,
    apply_pairwise_route,
    fusion_probabilities,
    indistinguishable_identity_ceiling,
    monotonic_probability,
    select_fusion,
    select_pairwise_route,
)


class TraceCleanNestedOofTests(unittest.TestCase):
    def test_monotonic_formula_is_fixed_and_bounded(self):
        names = ["method", "carrier", "bridge"]
        matrix = np.asarray([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])
        formula = {
            "intercept": 0.1,
            "method": 0.3,
            "carrier": 0.3,
            "bridge": 0.4,
        }
        observed = monotonic_probability(matrix, names, formula)
        np.testing.assert_allclose(observed, [1.0, 0.1])

    def test_fusion_candidates_match_frozen_names(self):
        values = np.asarray([0.2, 0.8])
        output = fusion_probabilities(
            values, values, values, values, values, values
        )
        self.assertEqual(
            set(output),
            {
                "monotonic_criterion",
                "structured",
                "hgb",
                "random_forest",
                "word",
                "char",
                "mean_hgb_random_forest",
                "mean_structured_hgb",
                "mean_structured_random_forest",
                "mean_structured_hgb_random_forest",
                "mean_structured_word",
                "mean_structured_char",
                "mean_hgb_char",
                "mean_random_forest_char",
                "mean_structured_hgb_char",
                "mean_word_char",
                "mean_all",
                "weighted_structured_50_word_25_char_25",
            },
        )
        for predicted in output.values():
            np.testing.assert_allclose(predicted, values)

    def test_refno_folds_never_split_a_paper(self):
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "A"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "B"},
            {"REFNO": "2", "RESULTNO": "12", "ROCKNAME": "C"},
            {"REFNO": "3", "RESULTNO": "13", "ROCKNAME": "D"},
            {"REFNO": "4", "RESULTNO": "14", "ROCKNAME": "E"},
        ]
        folds = deterministic_refno_fold_ids(records, n_splits=3, seed=7)
        self.assertEqual(folds[0], folds[1])
        self.assertEqual(set(map(int, folds)), {1, 2, 3})

    def test_identical_pdf_groups_share_a_fold_across_refnos(self):
        records = [
            {"REFNO": str(index), "RESULTNO": str(index), "ROCKNAME": chr(64 + index)}
            for index in range(1, 6)
        ]
        groups = {"1": "same-pdf", "2": "same-pdf", "3": "c", "4": "d", "5": "e"}
        folds = deterministic_refno_fold_ids(
            records, n_splits=3, seed=9, group_key_by_refno=groups
        )
        self.assertEqual(folds[0], folds[1])

    def test_identical_identity_ceiling_exposes_unresolvable_rows(self):
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "Alpha"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "Alpha"},
            {"REFNO": "2", "RESULTNO": "12", "ROCKNAME": "Beta"},
        ]
        observed = indistinguishable_identity_ceiling(
            records, np.asarray([0, 1, 1], dtype=np.int8)
        )
        self.assertEqual(observed["mixed_label_indistinguishable_groups"], 1)
        self.assertEqual(observed["mixed_label_indistinguishable_rows"], 2)
        self.assertAlmostEqual(observed["oracle_majority_accuracy_ceiling"], 2 / 3)

    def test_pairwise_examples_are_symmetric_and_paper_balanced(self):
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "A"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "B"},
            {"REFNO": "2", "RESULTNO": "12", "ROCKNAME": "C"},
            {"REFNO": "2", "RESULTNO": "13", "ROCKNAME": "D"},
            {"REFNO": "2", "RESULTNO": "14", "ROCKNAME": "E"},
        ]
        matrix = np.asarray([[2.0], [0.0], [3.0], [1.0], [0.0]])
        labels = np.asarray([1, 0, 1, 0, 0], dtype=np.int8)
        differences, pair_labels, weights = _pairwise_examples(
            matrix, labels, np.arange(len(records)), records
        )
        self.assertEqual(len(differences), 6)
        np.testing.assert_allclose(differences[0], -differences[1])
        np.testing.assert_array_equal(pair_labels[0:2], [1, 0])
        self.assertAlmostEqual(float(weights[:2].sum()), 1.0)
        self.assertAlmostEqual(float(weights[2:].sum()), 1.0)

    def test_pairwise_route_only_changes_enabled_multi_target_papers(self):
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "A"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "B"},
            {"REFNO": "2", "RESULTNO": "12", "ROCKNAME": "C"},
        ]
        baseline = np.asarray([0.5, 0.5, 0.6])
        rank_score = np.asarray([1.0, -1.0, 3.0])
        observed = apply_pairwise_route(
            baseline,
            rank_score,
            np.arange(3),
            records,
            {"alpha": 1.0, "minimum_distinct_target_count": 2},
        )
        self.assertGreater(observed[0], baseline[0])
        self.assertLess(observed[1], baseline[1])
        self.assertEqual(observed[2], baseline[2])

    def test_pairwise_gate_ignores_duplicate_result_rows(self):
        records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "A"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "A"},
        ]
        baseline = np.asarray([0.5, 0.5])
        observed = apply_pairwise_route(
            baseline,
            np.asarray([2.0, -2.0]),
            np.arange(2),
            records,
            {"alpha": 1.0, "minimum_distinct_target_count": 2},
        )
        np.testing.assert_allclose(observed, baseline)

    def test_pairwise_route_requires_material_inner_gain(self):
        specs = [
            {"name": "identity"},
            {"name": "pair"},
        ]
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = {
            "identity": np.asarray([0.2, 0.6, 0.7, 0.8]),
            "pair": np.asarray([0.2, 0.4, 0.7, 0.8]),
        }
        selected, audit = select_pairwise_route(
            specs,
            probabilities,
            labels,
            {
                "minimum_accuracy_gain_over_identity": 0.30,
                "require_non_decreasing_macro_f1": True,
            },
        )
        self.assertEqual(selected["name"], "identity")
        self.assertFalse(audit["accepted_non_identity"])

    def test_fusion_guard_keeps_hgb_without_material_gain(self):
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        probabilities = {
            "hgb": np.asarray([0.2, 0.6, 0.7, 0.8]),
            "mean_structured_hgb": np.asarray([0.2, 0.4, 0.7, 0.8]),
        }
        selected, audit = select_fusion(
            ["hgb", "mean_structured_hgb"],
            probabilities,
            labels,
            {
                "baseline": "hgb",
                "minimum_accuracy_gain_over_baseline": 0.30,
                "require_non_decreasing_macro_f1": True,
            },
        )
        self.assertEqual(selected, "hgb")
        self.assertFalse(audit["accepted_non_baseline"])


if __name__ == "__main__":
    unittest.main()
