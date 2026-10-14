import unittest

import pandas as pd

from scripts.build_r5_r7_source_recovered_predictions import r5_payload
from scripts.paper_text_retrieval import retrieve_descriptions


class R5PaperOnlyPredictionTests(unittest.TestCase):
    def test_frozen_paper_calibration_decides_without_source_tables(self):
        retrieval = {
            "source_available": True,
            "target_anchor": {"status": "exact", "strength": "strong"},
            "categories": {"risk": [], "rescue": []},
            "provenance": "original paper page cache only",
        }
        calibration = {
            "candidate_score": 0,
            "source_available": True,
            "risk_score": 0.8,
            "threshold": 0.5,
            "margin": 0.3,
            "risk_contributions": [],
        }

        payload = r5_payload(
            pd.Series({"RESULTNO": "1", "ROCKNAME": "Alpha Formation"}),
            "Original paper text",
            "page_cache#1",
            {},
            "paper_only",
            paper_retrieval=retrieval,
            paper_calibration=calibration,
        )

        self.assertEqual(payload["R5"], 0)
        self.assertEqual(payload["outcome"], "NEGATIVE")
        self.assertEqual(payload["source_metadata"], {})
        self.assertTrue(
            all(
                "PMAGRESULT" not in source and "ROCKUNIT" not in source
                for source in payload["runtime_scoring_sources"]
            )
        )

    def test_missing_original_text_is_unknown_with_compatible_default(self):
        payload = r5_payload(
            pd.Series({"RESULTNO": "2", "ROCKNAME": "Missing Formation"}),
            "",
            "missing",
            {},
            "paper_only",
            paper_retrieval={
                "source_available": False,
                "target_anchor": {"status": "missing", "strength": "missing"},
                "categories": {"risk": [], "rescue": []},
                "provenance": "original paper page cache only",
            },
        )

        self.assertEqual(payload["R5"], 1)
        self.assertEqual(payload["outcome"], "UNKNOWN")
        self.assertTrue(payload["manual_review"])

    def test_target_bound_tilting_risk_overrides_calibrated_default(self):
        cache_row = {
            "pages": [
                {
                    "page": 1,
                    "source": "synthetic",
                    "section_hints": [],
                    "text": (
                        "Jalokoski Gabbro (R5: 0). Possible tilting could not be ruled out."
                    ),
                }
            ]
        }
        retrieval = retrieve_descriptions(cache_row, "Jalokoski Gabbro", False, "R5")
        calibration = {
            "candidate_score": 1,
            "source_available": True,
            "risk_score": -0.8,
            "threshold": 0.5,
            "margin": 1.3,
            "risk_contributions": [],
        }

        payload = r5_payload(
            pd.Series({"RESULTNO": "3", "ROCKNAME": "Jalokoski Gabbro"}),
            cache_row["pages"][0]["text"],
            "synthetic",
            {},
            "paper_only",
            paper_retrieval=retrieval,
            paper_calibration=calibration,
        )

        self.assertEqual(payload["R5"], 0)
        self.assertTrue(payload["paper_only_candidate_score"] == 0)


if __name__ == "__main__":
    unittest.main()
