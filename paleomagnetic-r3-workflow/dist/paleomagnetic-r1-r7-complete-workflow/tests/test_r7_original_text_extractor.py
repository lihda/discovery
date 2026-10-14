import unittest

from scripts.r7_original_text_extractor import (
    R7OriginalTextBoundaryError,
    extract_r7_original_text_evidence,
)
from scripts.r7_pole_overlap_reasoner import evaluate_r7_pole_overlap


def page_row(*paragraphs: str):
    return {
        "pages": [
            {
                "page": 4,
                "text": "\n\n".join(paragraphs),
                "text_sha256": "synthetic-page-hash",
                "section_hints": ["Paleomagnetic results"],
            }
        ]
    }


def reference_pole(**overrides):
    pole = {
        "pole_id": "young-kalahari",
        "lat": 12.0,
        "lon": 24.0,
        "A95": 4.0,
        "age": 520.0,
        "period": "Cambrian",
        "craton": "Kalahari Craton",
        "stable_connected_region": "Kalahari Craton",
        "reliability_R": 4,
        "source": {"citation": "Frozen synthetic pole catalogue v1"},
    }
    pole.update(overrides)
    return pole


class R7OriginalTextExtractorTests(unittest.TestCase):
    def test_extracts_explicit_target_candidate_search_and_field_rescue(self):
        row = page_row(
            "The Alpha Formation paleomagnetic pole has Plat = 10.0 N, "
            "Plong = 20.0 E, A95 = 5.0 degrees, age = 1700 Ma, and is "
            "Statherian within the stable connected Kalahari Craton.",
            "For the Alpha Formation, a positive fold test establishes that the "
            "magnetization is older than 1000 Ma and predates the young-kalahari pole.",
        )
        search = {
            "qualified_search_complete": True,
            "declaration": (
                "A complete search included all poles with R >= 3 in the stable "
                "connected Kalahari Craton."
            ),
            "scope": {
                "quantifier": "all",
                "minimum_reliability_R": 3,
                "stable_connected_region": "Kalahari Craton",
            },
            "source": {"catalogue": "Frozen synthetic search manifest v1"},
        }

        extracted = extract_r7_original_text_evidence(
            "Alpha Formation", row, [reference_pole()], search
        )

        self.assertEqual(extracted["target"]["lat"], 10.0)
        self.assertEqual(extracted["target"]["A95"], 5.0)
        self.assertEqual(extracted["target"]["period"], "Statherian")
        self.assertEqual(extracted["candidates"][0]["reliability_R"], 4.0)
        self.assertTrue(extracted["qualified_search"]["qualified_search_complete"])
        self.assertEqual(
            extracted["field_test_age_constraints"][0]["candidate_id"],
            "young-kalahari",
        )
        self.assertEqual(extracted["extraction_audit"]["status"], "COMPLETE")

        reasoned = evaluate_r7_pole_overlap(
            extracted["target"],
            extracted["candidates"],
            extracted["field_test_age_constraints"],
            extracted["qualified_search"],
        )
        self.assertEqual(reasoned["relation"], "SUPPORT")
        self.assertEqual(reasoned["relations"][0]["reason_code"], "FIELD_EVIDENCE_RESCUE")

    def test_missing_target_fields_remain_missing_and_are_audited(self):
        row = page_row(
            "The Alpha Formation pole has Plat = 10 N, Plong = 20 E, age = "
            "1700 Ma, is Statherian, and lies in the stable connected Kalahari Craton."
        )

        extracted = extract_r7_original_text_evidence("Alpha Formation", row)

        self.assertNotIn("A95", extracted["target"])
        self.assertEqual(extracted["extraction_audit"]["status"], "INCOMPLETE")
        target_event = next(
            event
            for event in extracted["extraction_audit"]["events"]
            if event["kind"] == "target_pole"
        )
        self.assertIn("MISSING_A95", target_event["reason_codes"])

    def test_does_not_transfer_another_targets_pole_values(self):
        row = page_row(
            "The Alpha Formation is discussed here without a reported pole.",
            "The Beta Formation pole has Plat = 10 N, Plong = 20 E, A95 = 5 "
            "degrees, age = 1700 Ma, is Statherian, and lies in the stable "
            "connected Kalahari Craton.",
        )

        extracted = extract_r7_original_text_evidence("Alpha Formation", row)

        self.assertIsNone(extracted["target"])
        self.assertIn(
            "NO_EXPLICIT_TARGET_POLE_VALUES",
            extracted["extraction_audit"]["events"][0]["reason_codes"],
        )

    def test_rejects_orogenic_reference_poles(self):
        row = page_row(
            "The Alpha Formation pole has Plat = 10 N, Plong = 20 E, A95 = 5 "
            "degrees, age = 1700 Ma, is Statherian, and lies in the stable "
            "connected Kalahari Craton."
        )

        extracted = extract_r7_original_text_evidence(
            "Alpha Formation",
            row,
            [
                reference_pole(
                    pole_id="orogenic-pole",
                    craton="Damara orogenic belt",
                    stable_connected_region="Damara orogenic belt",
                )
            ],
        )

        self.assertEqual(extracted["candidates"], [])
        rejected = next(
            event
            for event in extracted["extraction_audit"]["events"]
            if event["kind"] == "reference_candidate"
        )
        self.assertIn("OROGENIC_SCOPE", rejected["reason_codes"])

    def test_search_completeness_requires_all_gates_and_source(self):
        target = (
            "The Alpha Formation pole has Plat = 10 N, Plong = 20 E, A95 = 5 "
            "degrees, age = 1700 Ma, is Statherian, and lies in the stable "
            "connected Kalahari Craton."
        )
        incomplete = page_row(
            target,
            "For the Alpha Formation, several younger poles with R >= 3 were "
            "compared in the stable connected Kalahari Craton.",
        )
        complete = page_row(
            target,
            "For the Alpha Formation, a complete search compared all younger "
            "poles with R >= 3 in the stable connected Kalahari Craton.",
        )

        incomplete_result = extract_r7_original_text_evidence("Alpha Formation", incomplete)
        complete_result = extract_r7_original_text_evidence("Alpha Formation", complete)

        self.assertIsNone(incomplete_result["qualified_search"])
        self.assertTrue(complete_result["qualified_search"]["qualified_search_complete"])
        incomplete_event = next(
            event
            for event in incomplete_result["extraction_audit"]["events"]
            if event["kind"] == "qualified_search"
        )
        self.assertIn("MISSING_COMPLETE_QUANTIFIER", incomplete_event["reason_codes"])
        self.assertIn("MISSING_ALL_QUANTIFIER", incomplete_event["reason_codes"])

    def test_rejects_record_ids_and_label_fields_recursively(self):
        with self.assertRaises(R7OriginalTextBoundaryError):
            extract_r7_original_text_evidence(
                "Alpha Formation", {"REFNO": "100", "pages": []}
            )
        with self.assertRaises(R7OriginalTextBoundaryError):
            extract_r7_original_text_evidence(
                "Alpha Formation",
                {"pages": []},
                [{"pole_id": "x", "expected_label": 0}],
            )


if __name__ == "__main__":
    unittest.main()
