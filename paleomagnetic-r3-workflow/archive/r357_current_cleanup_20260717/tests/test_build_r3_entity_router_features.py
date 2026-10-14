from __future__ import annotations

import unittest

from scripts.build_r3_entity_router_features import (
    CRITERION_FEATURE_NAMES,
    criterion_counterfactual_features,
    document_chunks,
)
from scripts.r3_paleomag_entity_graph import build_entity_graph


CONCEPTS = (
    {"category": "method", "search_patterns": [r"\bIRM\b"]},
    {"category": "diagnostic", "search_patterns": [r"saturat"]},
    {"category": "carrier", "search_patterns": [r"\bmagnetite\b"]},
    {"category": "bridge", "search_patterns": [r"carries?\s+the\s+ChRM"]},
)


class EntityRouterFeatureBuilderTests(unittest.TestCase):
    def test_document_chunks_are_multiscale_and_stop_at_references(self):
        pages = [
            {
                "page": 1,
                "blocks": [
                    {
                        "block_id": "b1",
                        "text": "Alpha Formation was tested by IRM.",
                        "lines": [
                            {
                                "line_id": "l1",
                                "text": "Alpha Formation was tested by IRM.",
                            }
                        ],
                    }
                ],
                "tables": [
                    {
                        "table_id": "t1",
                        "rows": [
                            {
                                "row_id": "r1",
                                "cells": [{"text": "Alpha Formation"}, {"text": "IRM"}],
                            }
                        ],
                    }
                ],
                "captions": [{"caption_id": "c1", "text": "Alpha Formation"}],
            },
            {
                "page": 2,
                "blocks": [{"block_id": "b2", "text": "References", "lines": []}],
                "tables": [],
                "captions": [],
            },
            {
                "page": 3,
                "blocks": [{"block_id": "b3", "text": "Alpha Formation", "lines": []}],
                "tables": [],
                "captions": [],
            },
        ]
        chunks = document_chunks(pages)
        self.assertEqual(
            {item["kind"] for item in chunks},
            {"sentence", "paragraph", "table", "caption", "section"},
        )
        self.assertTrue(all(not item["chunk_id"].startswith("p3:") for item in chunks))
        self.assertEqual(len({item["chunk_id"] for item in chunks}), len(chunks))

    def test_criterion_features_bind_practice_and_normative_chains(self):
        chunks = [
            {
                "chunk_id": "p1",
                "kind": "paragraph",
                "text": (
                    "Alpha Formation was tested by IRM; magnetite carries the ChRM."
                ),
            }
        ]
        targets = {"1": "Alpha Formation", "2": "Beta Formation"}
        graph = build_entity_graph(targets, chunks)
        features = criterion_counterfactual_features(
            targets, chunks, graph, CONCEPTS
        )
        self.assertEqual(set(features["1"]), set(CRITERION_FEATURE_NAMES))
        self.assertGreater(features["1"]["entity_cf_practice_chain_score"], 0.5)
        self.assertGreater(features["1"]["entity_cf_normative_chain_score"], 0.5)
        self.assertLess(
            features["2"]["entity_cf_practice_chain_score"],
            features["1"]["entity_cf_practice_chain_score"],
        )
        self.assertGreater(
            features["1"]["entity_cf_exclusive_event_count_log"], 0.0
        )

    def test_duplicate_canonical_targets_remain_bitwise_equal(self):
        chunks = [
            {
                "chunk_id": "p1",
                "kind": "paragraph",
                "text": "Alpha Formation was tested by IRM and contains magnetite.",
            }
        ]
        targets = {"1": "Alpha Formation", "2": "Alpha Formation"}
        graph = build_entity_graph(targets, chunks)
        features = criterion_counterfactual_features(
            targets, chunks, graph, CONCEPTS
        )
        self.assertEqual(graph["target_features"]["1"], graph["target_features"]["2"])
        self.assertEqual(features["1"], features["2"])
        self.assertIsNone(graph["assignments"][0]["top_target"])

    def test_multiscale_duplicate_text_does_not_sum_support(self):
        text = "Alpha Formation was tested by IRM."
        targets = {"1": "Alpha Formation", "2": "Beta Formation"}
        one = [{"chunk_id": "a", "kind": "paragraph", "text": text}]
        two = [
            *one,
            {"chunk_id": "b", "kind": "paragraph", "text": text},
        ]
        first = criterion_counterfactual_features(
            targets, one, build_entity_graph(targets, one), CONCEPTS
        )
        second = criterion_counterfactual_features(
            targets, two, build_entity_graph(targets, two), CONCEPTS
        )
        self.assertEqual(
            first["1"]["entity_cf_method_support_max"],
            second["1"]["entity_cf_method_support_max"],
        )
        self.assertEqual(
            first["1"]["entity_cf_method_scale_coverage"],
            second["1"]["entity_cf_method_scale_coverage"],
        )


if __name__ == "__main__":
    unittest.main()
