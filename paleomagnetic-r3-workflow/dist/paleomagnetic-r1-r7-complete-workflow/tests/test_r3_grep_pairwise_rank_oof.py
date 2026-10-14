import unittest

import numpy as np

from scripts.experiment_r3_grep_pairwise_rank_oof import (
    build_pairwise_examples,
    normalize_scores_by_paper,
    paper_aggregate_features,
)


class GrepPairwiseRankTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"REFNO": "1", "RESULTNO": "10", "ROCKNAME": "Alpha"},
            {"REFNO": "1", "RESULTNO": "11", "ROCKNAME": "Beta"},
            {"REFNO": "2", "RESULTNO": "12", "ROCKNAME": "Gamma"},
        ]

    def test_pairwise_examples_use_only_mixed_training_papers(self):
        matrix = np.asarray([[3.0, 1.0], [1.0, 2.0], [9.0, 9.0]])
        labels = np.asarray([1, 0, 1])
        differences, pair_labels, weights, audit = build_pairwise_examples(
            matrix, labels, np.asarray([0, 1, 2]), self.records
        )
        np.testing.assert_allclose(differences, [[2.0, -1.0], [-2.0, 1.0]])
        np.testing.assert_array_equal(pair_labels, [1, 0])
        np.testing.assert_allclose(weights, [0.5, 0.5])
        self.assertEqual(audit, {"mixed_papers": 1, "unordered_pairs": 1})

    def test_normalization_is_centered_within_complete_paper(self):
        result = normalize_scores_by_paper(
            np.asarray([3.0, 1.0, 7.0]), self.records, np.asarray([0, 1, 2])
        )
        np.testing.assert_allclose(result, [1.0, -1.0, 0.0])

    def test_paper_aggregates_are_label_free_and_aligned(self):
        refnos, matrix, row_to_paper = paper_aggregate_features(
            np.asarray([[1.0, 2.0], [3.0, 2.0], [4.0, 5.0]]), self.records
        )
        self.assertEqual(refnos, ["1", "2"])
        self.assertEqual(matrix.shape, (2, 7))
        np.testing.assert_array_equal(row_to_paper, [0, 0, 1])
        np.testing.assert_allclose(matrix[0, -2:], [2.0, 0.0])


if __name__ == "__main__":
    unittest.main()
