from __future__ import annotations

import json
from pathlib import Path
import unittest

from scripts.r3_scientific_evidence_chunks import (
    SCALES,
    build_scientific_evidence_chunks,
    compile_category_patterns,
    reciprocal_rank_fusion,
    robust_distractor_statistics,
    select_distractor_chunks,
)


ROOT = Path(__file__).resolve().parents[1]


class ScientificEvidenceChunkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = json.loads(
            (ROOT / "data/knowledge/r3_query_adaptive_retrieval_v1.json").read_text(
                encoding="utf-8"
            )
        )["chunking"]
        self.patterns = compile_category_patterns(
            [
                {
                    "concept_id": "irm",
                    "category": "method",
                    "search_patterns": [r"\bIRM\b"],
                },
                {
                    "concept_id": "curie",
                    "category": "diagnostic",
                    "search_patterns": [r"Curie temperature"],
                },
                {
                    "concept_id": "magnetite",
                    "category": "carrier",
                    "search_patterns": [r"magnetite"],
                },
                {
                    "concept_id": "origin",
                    "category": "bridge",
                    "search_patterns": [r"primary origin"],
                },
            ]
        )

    def test_multiscale_chunks_exclude_references_and_keep_evidence(self) -> None:
        pages = [
            {
                "page": 1,
                "section_hints": ["METHODS"],
                "text": (
                    "METHODS\nWe performed IRM acquisition on all studied specimens. "
                    "The measurements were repeated after heating.\n\n"
                    "RESULTS\nThe Curie temperature indicates magnetite. "
                    "This carrier supports a primary origin of the remanence."
                ),
            },
            {
                "page": 2,
                "section_hints": ["REFERENCES"],
                "text": "REFERENCES\nSmith (2020) magnetite IRM Curie temperature.",
            },
        ]
        chunks, audit = build_scientific_evidence_chunks(
            "10", pages, self.patterns, self.config
        )
        self.assertTrue(chunks)
        self.assertTrue({chunk.scale for chunk in chunks}.issubset(set(SCALES)))
        self.assertGreaterEqual(len({chunk.scale for chunk in chunks}), 3)
        self.assertTrue(any(chunk.lexical("method") for chunk in chunks))
        self.assertTrue(any(chunk.lexical("carrier") for chunk in chunks))
        self.assertFalse(any("Smith (2020)" in chunk.text for chunk in chunks))
        self.assertGreaterEqual(audit["reference_pages_excluded"], 1)
        self.assertLessEqual(len(chunks), self.config["max_chunks_per_document"])

    def test_cap_and_distractor_selection_are_deterministic(self) -> None:
        text = "\n\n".join(
            f"Paragraph {index}. We measured IRM and the result indicates magnetite."
            for index in range(200)
        )
        pages = [{"page": 1, "section_hints": ["RESULTS"], "text": text}]
        first, _ = build_scientific_evidence_chunks(
            "11", pages, self.patterns, self.config
        )
        second, _ = build_scientific_evidence_chunks(
            "11", pages, self.patterns, self.config
        )
        self.assertEqual([item.chunk_id for item in first], [item.chunk_id for item in second])
        self.assertLessEqual(len(first), self.config["max_chunks_per_document"])
        selected = select_distractor_chunks(first, 12)
        self.assertLessEqual(len(selected), 12)
        self.assertEqual(
            [item.chunk_id for item in selected],
            [item.chunk_id for item in select_distractor_chunks(first, 12)],
        )

    def test_rrf_and_robust_statistics(self) -> None:
        fused = reciprocal_rank_fusion([0.9, 0.7, 0.2], [0.0, 3.0, 1.0], k=60)
        self.assertEqual(len(fused), 3)
        self.assertGreater(fused[1], fused[2])
        stats = robust_distractor_statistics([0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertAlmostEqual(stats["median"], 0.3)
        self.assertGreater(stats["q95"], stats["q90"])
        self.assertGreater(stats["mad"], 0.0)

    def test_zero_lexical_scores_do_not_create_order_signal(self) -> None:
        fused = reciprocal_rank_fusion([0.5, 0.5, 0.2], [0.0, 0.0, 0.0], k=60)
        self.assertEqual(fused[0], fused[1])
        self.assertAlmostEqual(fused[0], 1.0 / 61.0)

    def test_untitled_later_pages_are_not_forced_into_abstract(self) -> None:
        pages = [
            {
                "page": 1,
                "section_hints": [],
                "text": "An abstract-like opening with enough words to form one evidence chunk.",
            },
            {
                "page": 8,
                "section_hints": [],
                "text": "We measured IRM on specimens and the result indicates magnetite grains.",
            },
        ]
        chunks, _ = build_scientific_evidence_chunks(
            "12", pages, self.patterns, self.config
        )
        later = [chunk for chunk in chunks if chunk.page == 8]
        self.assertTrue(later)
        self.assertTrue(all(chunk.section == "unknown" for chunk in later))

    def test_page_windows_keep_their_local_section(self) -> None:
        filler = " ".join(["measurement"] * 40)
        pages = [
            {
                "page": 4,
                "section_hints": [],
                "text": (
                    f"METHODS\nWe performed IRM experiments. {filler}\n\n"
                    f"RESULTS\nThe Curie temperature indicates magnetite. {filler}"
                ),
            }
        ]
        chunks, _ = build_scientific_evidence_chunks(
            "13", pages, self.patterns, self.config
        )
        sections = {
            chunk.section for chunk in chunks if chunk.scale == "page_window"
        }
        self.assertIn("methods", sections)
        self.assertIn("results", sections)


if __name__ == "__main__":
    unittest.main()
