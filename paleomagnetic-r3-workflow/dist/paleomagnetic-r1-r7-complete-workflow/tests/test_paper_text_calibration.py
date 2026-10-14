import json
from pathlib import Path
import tempfile
import unittest

from scripts.paper_text_calibration import FrozenPaperTextModel, build_target_document
from scripts.paper_text_retrieval import retrieve_descriptions


class PaperTextCalibrationTests(unittest.TestCase):
    def test_frozen_r5_artifact_keeps_grouped_holdout_and_identity_boundary(self):
        path = Path(__file__).resolve().parents[1] / "data/calibration/r5_paper_text_model_v1.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        audit = artifact["audit"]
        model = artifact["models"]["R5"]
        held_out = audit["evaluation"]["held_out_test"]["deployed_missing_source_defaults_to_one"]
        features = [row[0] for row in model["feature_parameters"]]

        self.assertTrue(audit["held_out_test_not_used_for_feature_or_threshold_selection"])
        self.assertGreaterEqual(held_out["accuracy"], 0.749)
        self.assertGreaterEqual(held_out["f1_0"], 0.37)
        self.assertEqual(len(features), 8000)
        self.assertFalse(any("refno" in feature or "resultno" in feature for feature in features))
        self.assertIn("REFNO/RESULTNO answer lookup", audit["forbidden_runtime_scoring_sources"])

    def test_prediction_keeps_original_page_evidence(self):
        cache_row = {
            "pages": [
                {
                    "page": 2,
                    "source": "synthetic-paper",
                    "text": (
                        "The Alpha Formation belongs to an allochthonous mobile belt. "
                        "Its structural attitude remains uncertain."
                    ),
                    "section_hints": ["Geology"],
                }
            ]
        }
        retrieval = retrieve_descriptions(cache_row, "Alpha Formation", False, "R5")
        document = build_target_document(cache_row, "Alpha Formation", retrieval)
        artifact = {
            "version": "test-model",
            "models": {
                "R5": {
                    "bias": 0.0,
                    "threshold": 1.0,
                    "feature_parameters": [["w:allochthonous", 1.0, 2.0]],
                }
            },
            "audit": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            result = FrozenPaperTextModel(path).predict(document)

        self.assertEqual(result["candidate_score"], 0)
        self.assertEqual(result["risk_contributions"][0]["feature"], "w:allochthonous")
        evidence = result["risk_contributions"][0]["evidence"]
        self.assertIn("page 2", evidence["section_or_page"])
        self.assertEqual(evidence["source"], "synthetic-paper")
        self.assertNotEqual(evidence["passage_kind"], "identity")

    def test_missing_page_text_is_explicit_in_document(self):
        retrieval = {
            "target_anchor": {"status": "missing", "strength": "missing", "pages": []},
            "hits": [],
        }

        document = build_target_document({"pages": []}, "Missing Formation", retrieval)

        self.assertFalse(document["source_available"])
        self.assertEqual(len(document["passages"]), 1)
        self.assertEqual(document["passages"][0]["kind"], "identity")


if __name__ == "__main__":
    unittest.main()
