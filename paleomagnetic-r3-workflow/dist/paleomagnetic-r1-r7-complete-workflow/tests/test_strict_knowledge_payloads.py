import copy
import json
from pathlib import Path
import unittest

import pandas as pd

from scripts.build_r5_r7_source_recovered_predictions import (
    build_rows,
    fuse_r7_decisions,
    r5_knowledge_payload,
    r7_knowledge_payload,
)
from scripts.expert_conflict_engine import DEFAULT_KNOWLEDGE_BASE, ExpertConflictEngine


def analysis(value=1, outcome="POSITIVE"):
    return {
        "schema_version": "1.0",
        "engine_version": "test",
        "kb_version": "test",
        "indicator": "R5",
        "target": "Alpha Formation",
        "source_available": True,
        "relation": "SUPPORT",
        "conflict_types": [],
        "confidence": 0.9,
        "evidence_mass": {"support": 0.9, "conflict": 0.0, "unknown": 0.1, "contradiction": 0.0},
        "decision": {
            "recommended_value": value,
            "outcome": outcome,
            "reason": "Synthetic expert relation.",
            "support_groups": [],
            "required_support_groups": [],
            "missing_support_groups": [],
        },
        "relations": [],
        "provenance": {"label_supervision": False},
    }


class StrictKnowledgePayloadTests(unittest.TestCase):
    def test_r7_build_rows_runs_original_text_extraction_and_spherical_reasoner(self):
        knowledge = json.loads(DEFAULT_KNOWLEDGE_BASE.read_text(encoding="utf-8"))
        knowledge = copy.deepcopy(knowledge)
        knowledge["r7_quantitative_knowledge"]["reference_poles"] = {
            "poles": [
                {
                    "pole_id": "young-kalahari",
                    "lat": 12.0,
                    "lon": 24.0,
                    "A95": 4.0,
                    "age": 520.0,
                    "period": "Cambrian",
                    "craton": "Kalahari Craton",
                    "stable_connected_region": "Kalahari Craton",
                    "reliability_R": 4,
                    "source": {"citation": "Synthetic frozen pole catalogue"},
                }
            ],
            "source": {"citation": "Synthetic frozen pole catalogue"},
        }
        targets = pd.DataFrame(
            [{"REFNO": "paper", "RESULTNO": "1", "ROCKNAME": "Alpha Formation"}]
        )
        cache_row = {
            "REFNO": "paper",
            "source": {"source_kind": "synthetic", "source_quality_flags": []},
            "pages": [
                {
                    "page": 4,
                    "text": (
                        "The Alpha Formation paleomagnetic pole has Plat = 10.0 N, "
                        "Plong = 20.0 E, A95 = 5.0 degrees, age = 1700 Ma, and is "
                        "Statherian within the stable connected Kalahari Craton."
                    ),
                    "text_sha256": "synthetic",
                    "section_hints": ["Results"],
                }
            ],
        }

        rows = build_rows(
            "R7",
            targets,
            Path("unused"),
            Path("unused"),
            {},
            {},
            "knowledge_strict",
            "knowledge_strict",
            page_cache_rows={"paper": cache_row},
            knowledge_engine=ExpertConflictEngine(knowledge),
        )

        payload = json.loads(rows[0]["R7_RESULT_JSON"])
        self.assertEqual(rows[0]["R7"], 0)
        self.assertEqual(payload["outcome"], "NEGATIVE")
        self.assertEqual(
            payload["quantitative_pole_conflict_analysis"]["relation"], "CONFLICT"
        )
        self.assertEqual(
            payload["quantitative_extraction_audit"]["accepted_counts"]["target"],
            1,
        )

    def test_r7_quantitative_decision_has_priority(self):
        semantic = analysis(value=1, outcome="POSITIVE")
        semantic["indicator"] = "R7"
        quantitative = {
            "relation": "CONFLICT",
            "decision": {
                "recommended_value": 0,
                "reason_code": "UNRESCUED_YOUNGER_A95_OVERLAP",
            },
        }

        fused = fuse_r7_decisions(semantic, quantitative)

        self.assertEqual(fused["recommended_value"], 0)
        self.assertEqual(fused["outcome"], "NEGATIVE")
        self.assertEqual(fused["decision_source"], "quantitative_a95_reasoner")

    def test_r7_mixed_quantitative_evidence_abstains(self):
        semantic = analysis(value=0, outcome="NEGATIVE")
        semantic["indicator"] = "R7"
        quantitative = {
            "relation": "MIXED",
            "decision": {"recommended_value": None, "reason_code": "UNRESOLVED_SCOPE"},
        }

        fused = fuse_r7_decisions(semantic, quantitative)

        self.assertIsNone(fused["recommended_value"])
        self.assertEqual(fused["binary_compatibility_value"], 1)
        self.assertTrue(fused["manual_review"])

    def test_r5_indistinguishable_group_preserves_value_but_abstains(self):
        payload = r5_knowledge_payload(
            pd.Series({"RESULTNO": "1", "ROCKNAME": "Alpha Formation"}),
            "page_cache",
            {"provenance": {}},
            {"target_anchor": {}, "top_risk": [], "top_rescue": [], "candidate_score": 1, "conflict": False},
            analysis(),
            identity_ambiguous=True,
        )

        self.assertEqual(payload["R5"], 1)
        self.assertEqual(payload["outcome"], "UNKNOWN")
        self.assertTrue(payload["manual_review"])
        self.assertIn("component locator", payload["final_decision_reasoning"])

    def test_r7_indistinguishable_group_preserves_value_but_abstains(self):
        source = analysis()
        source["indicator"] = "R7"
        payload = r7_knowledge_payload(
            pd.Series({"RESULTNO": "2", "ROCKNAME": "Alpha Formation"}),
            "page_cache",
            source,
            identity_ambiguous=True,
        )

        self.assertEqual(payload["R7"], 1)
        self.assertEqual(payload["outcome"], "UNKNOWN")
        self.assertTrue(payload["manual_review"])
        self.assertIn("component locator", payload["final_decision_reasoning"])

    def test_r7_indistinguishable_conflict_uses_unknown_compatibility_default(self):
        source = analysis(value=0, outcome="NEGATIVE")
        source["indicator"] = "R7"
        source["relation"] = "CONFLICT"
        payload = r7_knowledge_payload(
            pd.Series({"RESULTNO": "3", "ROCKNAME": "Alpha Formation"}),
            "page_cache",
            source,
            identity_ambiguous=True,
        )

        self.assertEqual(payload["R7"], 1)
        self.assertEqual(payload["outcome"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
