from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from create_r1234567_total_review_html import json_evidence, r3_readable_conditions, render_html


class TotalReviewHtmlTests(unittest.TestCase):
    def test_r3_knowledge_conflict_relations_are_readable(self) -> None:
        conditions = r3_readable_conditions(
            {
                "knowledge_conflict_analysis": {
                    "relation": "MIXED",
                    "confidence": 0.7,
                    "conflict_types": ["EVIDENCE_DISAGREEMENT"],
                    "decision": {"reason": "Supporting and conflicting claims coexist."},
                    "relations": [
                        {
                            "rule_id": "R3.METHOD.CARRIER_DIAGNOSTIC",
                            "relation": "SUPPORT",
                            "confidence": 0.91,
                            "conflict_type": "NONE",
                            "claim": {"en": "A carrier diagnostic was reported."},
                            "evidence": {
                                "quote": "IRM acquisition curves were measured.",
                                "page": 4,
                                "section": "Rock magnetism",
                            },
                        }
                    ],
                }
            }
        )

        self.assertEqual(conditions[0]["label"], "Expert knowledge relation")
        self.assertEqual(conditions[0]["status"], "MIXED")
        self.assertIsNone(conditions[0]["satisfied"])
        self.assertEqual(conditions[1]["status"], "SUPPORT")
        self.assertIn("page 4", conditions[1]["quote"])

    def test_unknown_outcome_is_rendered_as_source_unavailable(self) -> None:
        evidence = json_evidence(
            {
                "R3": 0,
                "outcome": "UNKNOWN",
                "summary": "No original-paper text was available.",
            },
            "R3",
        )
        self.assertEqual(evidence["outcome"], "UNKNOWN")
        self.assertEqual(evidence["source_status"], "UNKNOWN · source unavailable")

        row = {
            "REFNO": "1",
            "RESULTNO": "2",
            "TARGET": "Example unit",
            "ROCKUNITNO": "3",
            "scores": {f"R{index}": 0 for index in range(1, 8)},
            "R_SUM": 0,
            "evidence": {
                f"R{index}": evidence if index == 3 else {"status": "R=0", "outcome": "DECIDED"}
                for index in range(1, 8)
            },
            "source_values": {},
        }
        summary = {
            "rows": 1,
            "unique_refnos": 1,
            "unique_resultnos": 1,
            "score_counts": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "review.html"
            render_html([row], summary, output, "Review")
            html = output.read_text(encoding="utf-8")

        self.assertIn("UNKNOWN · source unavailable", html)
        self.assertIn('"outcome": "UNKNOWN"', html)


if __name__ == "__main__":
    unittest.main()
