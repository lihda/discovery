from __future__ import annotations

import sys
from pathlib import Path
import re
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_deberta_target_nli_embeddings as builder
import experiment_r3_deberta_target_nli as experiment
import freeze_r3_exact300_deberta_features as freezer


def passage(
    text: str,
    *,
    score: int = 0,
    method: int = 0,
    carrier: int = 0,
    start: int = 0,
) -> dict[str, object]:
    return {
        "text": text,
        "start": start,
        "end": start + len(text),
        "concepts": [],
        "method_hits": method,
        "diagnostic_hits": 0,
        "carrier_hits": carrier,
        "own_study_hits": 0,
        "result_hits": 0,
        "citation_heavy": False,
        "retrieval_score": score,
    }


class R3DebertaTargetNliTest(unittest.TestCase):
    def test_exact300_scope_constants_are_frozen(self) -> None:
        self.assertEqual(builder.FROZEN_ROWS, 300)
        self.assertEqual(builder.FROZEN_PAPERS, 258)
        self.assertEqual(
            builder.FROZEN_IDENTITY_SHA256,
            "93ea5dfba0a32501a564d06eb020386fd86b11f59ded0982043f90c064ab960b",
        )
        self.assertEqual(
            builder.FROZEN_ALIGNMENT_SHA256,
            "16fba124ba12920ed31aed22e93a4b1ace5db6ca35b7a63724a6e933e73f0a48",
        )

    def test_target_matching_ignores_generic_geological_words(self) -> None:
        exact, hits, coverage = builder.target_match_features(
            "Measurements on the Seton Formation at Kahochella identify magnetite.",
            "Seton Formation, Kahochella Group",
        )

        self.assertEqual(exact, 0)
        self.assertEqual(hits, 2)
        self.assertEqual(coverage, 1.0)
        self.assertNotIn("formation", builder.informative_target_tokens("Seton Formation"))

    def test_passage_selection_keeps_expert_and_target_bound_contexts(self) -> None:
        expert = passage("IRM and hysteresis identify magnetite.", score=12, method=2, carrier=1)
        target = passage(
            "The Wadi Abu Shihat specimens were measured.", score=3, start=100
        )
        noise = passage("regional geology", score=1, start=200)

        selected = builder.select_target_passages(
            [noise, target, expert], "Diorite dykes, Wadi Abu Shihat", 2
        )

        self.assertIn(expert, selected)
        self.assertIn(target, selected)

    def test_pair_aggregation_uses_highest_entailment_for_top(self) -> None:
        hypothesis_count = len(builder.HYPOTHESIS_NAMES)
        owners: list[int] = []
        hypotheses: list[int] = []
        mean: list[list[float]] = []
        cls: list[list[float]] = []
        probabilities: list[list[float]] = []
        for hypothesis in range(hypothesis_count):
            owners.extend([0, 0])
            hypotheses.extend([hypothesis, hypothesis])
            mean.extend([[1.0, 0.0], [0.0, 1.0]])
            cls.extend([[1.0, 0.0], [0.0, 1.0]])
            probabilities.extend([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])

        aggregated = builder.aggregate_target_pairs(
            np.asarray(mean, dtype=np.float32),
            np.asarray(cls, dtype=np.float32),
            np.asarray(probabilities, dtype=np.float32),
            owners,
            hypotheses,
            1,
        )

        np.testing.assert_allclose(aggregated["target_criterion__top"], [[0.0, 1.0]])
        self.assertEqual(set(aggregated), experiment.expected_arrays() - {"structured"})

    def test_merge_preserves_richer_positive_kb_metadata(self) -> None:
        reused = passage("IRM identifies magnetite", score=4, method=1, start=10)
        reused.update(
            {
                "concepts": ["irm_acquisition"],
                "matched_sources": ["frozen_taxonomy"],
            }
        )
        fresh = dict(reused)
        fresh.update(
            {
                "concepts": ["irm_acquisition", "magnetite"],
                "matched_sources": [
                    "frozen_taxonomy",
                    "positive_rockmag_expert_kb",
                ],
                "method_hits": 5,
                "retrieval_score": 12,
            }
        )

        merged = builder.merge_passage_sets([reused], [fresh])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["concepts"], ["irm_acquisition", "magnetite"])
        self.assertEqual(merged[0]["method_hits"], 5.0)
        self.assertIn("positive_rockmag_expert_kb", merged[0]["matched_sources"])
        self.assertEqual(
            merged[0]["retrieval_origins"], ["fresh_expert_retrieval", "reused"]
        )

    def test_enrichment_tracks_actual_matching_knowledge_source(self) -> None:
        concepts = [
            {
                "source": "frozen_taxonomy",
                "category": "method_taxonomy",
                "concept_id": "irm",
                "patterns": [re.compile(r"\bIRM\b")],
            },
            {
                "source": "positive_rockmag_expert_kb",
                "category": "carrier_taxonomy",
                "concept_id": "hematite",
                "patterns": [re.compile(r"\bhematite\b")],
            },
        ]

        enriched = builder.enrich_passage(
            {"text": "IRM acquisition curves were measured."}, concepts
        )

        self.assertEqual(enriched["matched_sources"], ["frozen_taxonomy"])
        self.assertEqual(enriched["method_hits"], 1)
        self.assertEqual(enriched["carrier_hits"], 0)

    def test_target_local_passage_binds_target_and_method(self) -> None:
        concepts = [
            {
                "source": "positive_rockmag_expert_kb",
                "category": "method_taxonomy",
                "concept_id": "irm_acquisition",
                "patterns": [re.compile(r"\bIRM\b")],
            }
        ]
        text = (
            "Regional context. The Seton Formation samples were selected. "
            "IRM acquisition curves were measured and interpreted. End."
        )

        rows = builder.target_local_passages(
            text, "Seton Formation, Kahochella Group", concepts, maximum=2, radius=80
        )

        self.assertEqual(len(rows), 1)
        self.assertIn("Seton Formation", rows[0]["text"])
        self.assertEqual(rows[0]["method_hits"], 1)

    def test_feature_cache_rejects_identifier_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            arrays = {
                name: np.zeros((2, 3), dtype=np.float32)
                for name in experiment.expected_arrays()
            }
            arrays["refnos"] = np.asarray(["1", "2"])
            np.savez_compressed(path, **arrays)

            with self.assertRaisesRegex(ValueError, "extra=.*refnos"):
                experiment.load_feature_views(path, 2)

    def test_feature_freezer_requires_all_300_rows_and_no_identifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.npz"
            arrays = {
                name: np.zeros((300, 1), dtype=np.float32)
                for name in experiment.expected_arrays()
            }
            np.savez_compressed(good, **arrays)
            shapes = freezer.validate_numeric_cache(good)
            self.assertTrue(all(shape[0] == 300 for shape in shapes.values()))

            bad = Path(directory) / "bad.npz"
            arrays["resultnos"] = np.arange(300)
            np.savez_compressed(bad, **arrays)
            with self.assertRaisesRegex(ValueError, "extra=.*resultnos"):
                freezer.validate_numeric_cache(bad)

    def test_inverse_target_weight_removes_duplicate_row_multiplicity(self) -> None:
        records = [
            {"REFNO": "1", "ROCKNAME": "Alpha Formation"},
            {"REFNO": "1", "ROCKNAME": "Alpha Formation"},
            {"REFNO": "1", "ROCKNAME": "Beta Formation"},
        ]
        weights = experiment.inverse_target_weights(np.arange(3), records)

        np.testing.assert_allclose(weights, [0.5, 0.5, 1.0])

    def test_multi_target_propagation_does_not_change_single_target_paper(self) -> None:
        records = [
            {"REFNO": "a", "ROCKNAME": "Alpha"},
            {"REFNO": "a", "ROCKNAME": "Beta"},
            {"REFNO": "b", "ROCKNAME": "Gamma"},
        ]
        strategies = experiment.propagation_strategies(
            records, {"candidate": np.asarray([0.2, 0.8, 0.7])}
        )

        np.testing.assert_allclose(
            strategies["candidate__multi_paper_50"], [0.35, 0.65, 0.7]
        )

    def test_outer_folds_cover_every_row_once_without_group_overlap(self) -> None:
        records = [
            {"REFNO": str(paper), "ROCKNAME": f"Target {paper}"}
            for paper in range(10)
            for _ in range(2)
        ]
        labels = np.asarray([paper % 2 for paper in range(10) for _ in range(2)])

        folds = experiment.outer_folds(records, labels)

        test_rows = np.concatenate([test for _, test in folds])
        np.testing.assert_array_equal(np.sort(test_rows), np.arange(len(records)))
        for train, test in folds:
            train_groups = {records[index]["REFNO"] for index in train}
            test_groups = {records[index]["REFNO"] for index in test}
            self.assertFalse(train_groups & test_groups)

    def test_paper_relative_features_compare_unique_competing_targets(self) -> None:
        records = [
            {"REFNO": "a", "ROCKNAME": "Alpha"},
            {"REFNO": "a", "ROCKNAME": "Alpha"},
            {"REFNO": "a", "ROCKNAME": "Beta"},
            {"REFNO": "b", "ROCKNAME": "Gamma"},
        ]
        matrix = np.asarray([[1.0], [1.0], [3.0], [5.0]], dtype=np.float32)

        relative = experiment.paper_relative_features(records, matrix)

        np.testing.assert_allclose(relative[:3], [[-1.0, 0.0], [-1.0, 0.0], [1.0, 1.0]])
        np.testing.assert_allclose(relative[3], [0.0, 0.5])

    def test_oof_npz_contains_only_probabilities_and_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "oof.npz"
            experiment.write_oof_npz(
                path,
                {"candidate": np.asarray([0.2, 0.8], dtype=np.float64)},
                expected_rows=2,
            )
            with np.load(path, allow_pickle=False) as payload:
                self.assertEqual(
                    set(payload.files), {"prob__candidate", "decision__candidate"}
                )
                np.testing.assert_array_equal(payload["decision__candidate"], [0, 1])

            with self.assertRaises(ValueError):
                experiment.write_oof_npz(
                    path, {"labels": np.asarray([0.0, 1.0], dtype=np.float64)}
                )
            with self.assertRaisesRegex(ValueError, "full row coverage"):
                experiment.write_oof_npz(
                    path,
                    {"candidate": np.asarray([0.2], dtype=np.float64)},
                    expected_rows=2,
                )

    def test_dense_reference_strategy_is_generated_by_fixed_model_and_propagation(self) -> None:
        records = [
            {"REFNO": "a", "ROCKNAME": "Alpha"},
            {"REFNO": "a", "ROCKNAME": "Beta"},
        ]
        base_name = "dense__target_chain_taxonomy__linear_svc_scaled_c0.001"
        strategies = experiment.propagation_strategies(
            records, {base_name: np.asarray([0.2, 0.8])}
        )
        self.assertIn(experiment.PRIMARY_STRATEGY, strategies)

    def test_repair_primary_is_fixed_tree_ensemble_plus_paper_style(self) -> None:
        probabilities = {
            "joint_tree__paper_style__hgb": np.asarray([0.4, 0.8]),
        }
        model_values = {
            "hgb": np.asarray([0.2, 0.5]),
            "extra": np.asarray([0.3, 0.6]),
            "gradient": np.asarray([0.4, 0.7]),
        }
        for name, values in model_values.items():
            probabilities[
                f"joint_tree__curation_deberta_taxonomy_relative__{name}"
            ] = values

        experiment.add_joint_repair_strategies(
            probabilities, ["curation_deberta_taxonomy_relative"]
        )

        np.testing.assert_allclose(
            probabilities[experiment.REPAIR_PRIMARY_STRATEGY],
            0.75 * np.asarray([0.3, 0.6]) + 0.25 * np.asarray([0.4, 0.8]),
        )

    def test_alignment_hash_changes_with_row_order(self) -> None:
        first = {"REFNO": "1", "RESULTNO": "2", "ROCKNAME": "Alpha"}
        second = {"REFNO": "1", "RESULTNO": "3", "ROCKNAME": "Beta"}
        self.assertNotEqual(
            builder.alignment_sha256([first, second]),
            builder.alignment_sha256([second, first]),
        )


if __name__ == "__main__":
    unittest.main()
