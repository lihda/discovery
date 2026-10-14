import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_r3_pdf_only_error_forensics as forensics


class _FakeRetriever:
    def __init__(self, cache_row):
        self.cache_row = cache_row

    def retrieve(self, *args, **kwargs):
        return {
            "source_available": True,
            "target_anchor": {
                "found": True,
                "status": "token",
                "strength": "strong",
                "specificity": 0.9,
            },
            "hits": [],
            "categories": {},
        }


class _FakeReasoner:
    def __init__(self, artifact):
        self.artifact = artifact

    def evaluate(self, retrieval, **kwargs):
        return {
            "rockmag_proxy_r3": {"binary_value": 0, "outcome": "NEGATIVE"},
            "reason_code": "NO_TARGET_BOUND_STRONG_ROCKMAG_EVIDENCE",
            "conditions": {
                "accepted_method": False,
                "carrier_identification": False,
                "diagnostic_result": False,
                "alteration_result": False,
                "connected_positive_path": False,
                "result_component_identity_ambiguous": kwargs.get(
                    "identity_ambiguous", False
                ),
                "source_quality_sufficient": True,
            },
            "evidence_counts": {},
            "rejected_evidence_counts": {},
            "selected_evidence": {},
        }


class _PoisonPath:
    def open(self, *args, **kwargs):
        raise AssertionError("post-freeze source was touched during candidate generation")

    def read_text(self, *args, **kwargs):
        raise AssertionError("post-freeze source was touched during candidate generation")


class R3PdfOnlyErrorForensicsTests(unittest.TestCase):
    def test_shared_detector_requires_executed_method_and_explicit_quantifier(self):
        patterns = [("irm_acquisition", re.compile(r"\bIRM\b", re.I))]
        positive = {
            "pages": [
                {
                    "text": (
                        "All samples were measured in progressively increasing fields. "
                        "IRM acquisition curves were then recorded."
                    )
                }
            ]
        }
        unshared = {
            "pages": [
                {
                    "text": (
                        "Several samples are discussed. IRM acquisition curves were "
                        "then recorded."
                    )
                }
            ]
        }
        references_only = {
            "pages": [
                {
                    "text": (
                        "No rock magnetic experiment was reported.\n"
                        "References\n"
                        "All samples were measured and IRM curves were recorded."
                    )
                }
            ]
        }

        self.assertTrue(forensics.explicit_shared_episode(positive, patterns)["found"])
        self.assertFalse(forensics.explicit_shared_episode(unshared, patterns)["found"])
        self.assertFalse(
            forensics.explicit_shared_episode(references_only, patterns)["found"]
        )

    def test_candidate_phase_does_not_touch_labels_or_prior_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = root / "identity.xlsx"
            page_cache = root / "cache.jsonl"
            artifact = root / "artifact.json"
            candidates_path = root / "candidates.jsonl"
            receipt_path = root / "receipt.json"
            identity.write_bytes(b"identity")
            page_cache.write_text("cache\n", encoding="utf-8")
            artifact.write_text("{}\n", encoding="utf-8")
            identity_rows = [
                {"REFNO": "10", "RESULTNO": "1", "ROCKNAME": "Alpha Formation"}
            ]
            cache = {
                "10": {
                    "REFNO": "10",
                    "source": {"source_quality_flags": []},
                    "pages": [{"page": 1, "text": "No diagnostic method."}],
                }
            }

            with (
                patch.object(forensics, "IDENTITY", identity),
                patch.object(forensics, "PAGE_CACHE", page_cache),
                patch.object(forensics, "ARTIFACT", artifact),
                patch.object(forensics, "CANDIDATES", candidates_path),
                patch.object(forensics, "RECEIPT", receipt_path),
                patch.object(forensics, "LABELS", _PoisonPath()),
                patch.object(forensics, "LATEST", _PoisonPath()),
                patch.object(forensics, "PREVIOUS", _PoisonPath()),
                patch.object(forensics, "load_identity", return_value=identity_rows),
                patch.object(forensics, "load_pdf_cache", return_value=cache),
                patch.object(forensics, "method_patterns", return_value=[]),
                patch.object(
                    forensics,
                    "explicit_shared_episode",
                    return_value={"found": False},
                ),
                patch.object(forensics, "PaperTextRetriever", _FakeRetriever),
                patch.object(forensics, "R3RockmagDistilledReasoner", _FakeReasoner),
            ):
                candidates, receipt = forensics.build_candidates()

            self.assertEqual(len(candidates), 1)
            self.assertEqual(
                receipt["phase"],
                "candidates_frozen_before_labels_or_predictions_opened",
            )
            self.assertTrue(candidates_path.is_file())
            frozen = json.loads(candidates_path.read_text(encoding="utf-8"))
            self.assertFalse(frozen["runtime_boundary"]["labels_read"])
            self.assertFalse(frozen["runtime_boundary"]["prior_predictions_read"])


if __name__ == "__main__":
    unittest.main()
