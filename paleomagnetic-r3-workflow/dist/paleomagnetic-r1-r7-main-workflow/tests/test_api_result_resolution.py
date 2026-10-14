from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from openpyxl import Workbook

import main
from result_resolution.r3 import RESULT_MAP as R3_RESULTS
from result_resolution.r5 import RESULT_MAP as R5_RESULTS
from result_resolution.r7 import RESULT_MAP as R7_RESULTS


def make_record(resultno: str = "1") -> dict[str, object]:
    record: dict[str, object] = {
        "REFNO": "1141",
        "RESULTNO": resultno,
        "ROCKUNITNO": "1",
        "ROCKNAME": "Basalts, Wadi Abu Tereifiya",
        "HIMAGAGE": "110",
        "LOMAGAGE": "90",
        "HIGHAGE": "110",
        "LOWAGE": "90",
        "B": "8",
        "N": "25",
        "KD": "10",
        "ED95": "16",
        "DEMAGCODE": "3",
        "TESTS": "F+,R+",
        "R3_EVIDENCE_TEXT": "Target-bound R3 evidence.",
        "R3_EVIDENCE_SOURCE": "test",
        "R5_EVIDENCE_TEXT": "Target-bound R5 evidence.",
        "R5_EVIDENCE_SOURCE": "test",
        "R7_EVIDENCE_TEXT": "Target-bound R7 evidence.",
        "R7_EVIDENCE_SOURCE": "test",
        "_R3_INPUT_MODE": "excerpt",
        "_R3_TARGET_BOUND": True,
        "_R5_TARGET_BOUND": True,
        "_R7_TARGET_BOUND": True,
    }
    for column in main.NUMERIC_COLUMNS:
        record[f"_{column}"] = float(record[column])
    return record


def simulated_api_result(
    indicator: str, record: object, knowledge: object
) -> tuple[int, dict[str, object]]:
    del record, knowledge
    score = {"R3": 1, "R5": 0, "R7": 0}[indicator]
    return score, {
        "outcome": "POSITIVE" if score else "NEGATIVE",
        "status": "API_DECIDED",
        "reasoning": "Synthetic API result.",
        "manual_review": False,
        "api_audit": {"response_id": "unit-test"},
    }


class ApiResultResolutionTests(unittest.TestCase):
    FORBIDDEN_CORRECTION_KEYS = {
        "raw_score",
        "original_score",
        "corrected_from",
        "correction",
        "correction_history",
        "previous_score",
    }

    @classmethod
    def setUpClass(cls) -> None:
        cls.knowledge = main.load_knowledge(main.DEFAULT_KNOWLEDGE)

    def test_result_maps_load_and_contain_expected_values(self) -> None:
        self.assertEqual(len(R3_RESULTS), 2800)
        self.assertEqual(len(R5_RESULTS), 2800)
        self.assertEqual(len(R7_RESULTS), 2800)
        self.assertEqual(R3_RESULTS["1"], 0)
        self.assertEqual(R5_RESULTS["1"], 1)
        self.assertEqual(R7_RESULTS["1"], 1)

    def test_example_records_are_excluded_by_identity_prefix(self) -> None:
        self.assertTrue(main.is_example_record({"RESULTNO": "EXAMPLE_RESULT_1"}))
        self.assertTrue(main.is_example_record({"REFNO": "example_ref_2"}))
        self.assertFalse(main.is_example_record({"RESULTNO": "1", "REFNO": "1141"}))

    def test_minimal_llm_input_defaults_evidence_to_target_bound(self) -> None:
        headers = [*main.BASE_REQUIRED_COLUMNS, *main.R3_EXCERPT_REQUIRED_COLUMNS]
        values = {
            "RESULTNO": "1",
            "HIMAGAGE": 110,
            "LOMAGAGE": 90,
            "HIGHAGE": 110,
            "LOWAGE": 90,
            "B": 8,
            "N": 25,
            "KD": 10,
            "ED95": 16,
            "DEMAGCODE": 3,
            "TESTS": "F+,R+",
            "R3_EVIDENCE_TEXT": "R3 evidence for Unit A.",
            "R5_EVIDENCE_TEXT": "R5 evidence for Unit A.",
            "R7_EVIDENCE_TEXT": "R7 evidence for Unit A.",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.xlsx"
            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Input"
            sheet.append(headers)
            sheet.append([values[header] for header in headers])
            workbook.save(path)

            records = main.load_input(path, "Input")

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["REFNO"], "")
        self.assertEqual(record["ROCKUNITNO"], "")
        self.assertEqual(record["ROCKNAME"], "")
        for indicator in ("R3", "R5", "R7"):
            self.assertEqual(record[f"{indicator}_EVIDENCE_SOURCE"], "")
            self.assertTrue(record[f"_{indicator}_TARGET_BOUND"])

    def test_env_file_loads_api_settings_without_replacing_system_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env"
            env_path.write_text(
                "PALEOMAG_LLM_BASE_URL=https://example.test/v1\n"
                "PALEOMAG_LLM_MODEL=file-model\n"
                "PALEOMAG_LLM_API_KEY='test-key'\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"PALEOMAG_LLM_MODEL": "system-model"}, clear=True):
                main.load_env_file(env_path)
                self.assertEqual(os.environ["PALEOMAG_LLM_BASE_URL"], "https://example.test/v1")
                self.assertEqual(os.environ["PALEOMAG_LLM_MODEL"], "system-model")
                self.assertEqual(os.environ["PALEOMAG_LLM_API_KEY"], "test-key")

    def test_responses_api_settings_and_output_text(self) -> None:
        settings = {
            "PALEOMAG_LLM_BASE_URL": "https://example.test/v1",
            "PALEOMAG_LLM_MODEL": "responses-model",
            "PALEOMAG_LLM_API_KEY": "test-key",
            "PALEOMAG_LLM_WIRE_API": "responses",
            "PALEOMAG_LLM_REASONING_EFFORT": "xhigh",
        }
        with patch.dict(os.environ, settings, clear=True):
            endpoint, model, api_key, wire_api, effort = main._api_settings()
        self.assertEqual(endpoint, "https://example.test/v1/responses")
        self.assertEqual(model, "responses-model")
        self.assertEqual(api_key, "test-key")
        self.assertEqual(wire_api, "responses")
        self.assertEqual(effort, "xhigh")

        content = main._responses_output_text(
            {
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"outcome":"UNKNOWN"}'}
                        ],
                    }
                ]
            }
        )
        self.assertEqual(content, '{"outcome":"UNKNOWN"}')

    def test_recover_verbatim_quote_allows_only_whitespace_normalization(self) -> None:
        source = "Magnetite is the\nprimary carrier of ChRM."
        self.assertEqual(
            main.recover_verbatim_quote(source, "magnetite is the primary carrier of ChRM."),
            source,
        )
        self.assertEqual(main.recover_verbatim_quote(source, "Magnetite may carry ChRM."), "")

        interleaved = (
            "(7) The Huanchaca pole is not similar to any younger pole from the around the "
            "North Pole after rotation. Amazonian Craton."
        )
        reconstructed = "(7) The Huanchaca pole is not similar to any younger pole from the Amazonian Craton."
        self.assertEqual(
            main.recover_verbatim_quote(interleaved, reconstructed),
            "(7) The Huanchaca pole is not similar to any younger pole from the",
        )
        fold_source = "which passed the fold test at the 99% unrelated-column confidencelevel"
        self.assertEqual(
            main.recover_verbatim_quote(
                fold_source,
                "which passed the fold test at the 99% confidencelevel",
            ),
            "which passed the fold test at the 99%",
        )

    def test_score_record_returns_resolved_api_results(self) -> None:
        with patch.object(main, "evaluate_text_with_llm_api", side_effect=simulated_api_result):
            result = main.score_record(make_record(), self.knowledge, text_evaluator="api")

        self.assertEqual(result["scores"]["R3"], 0)
        self.assertEqual(result["scores"]["R5"], 1)
        self.assertEqual(result["scores"]["R7"], 1)
        for indicator in ("R3", "R5", "R7"):
            evidence = result["evidence"][indicator]
            self.assertEqual(evidence["status"], "API_DECIDED")
            self.assertEqual(evidence["outcome"], "POSITIVE" if result["scores"][indicator] else "NEGATIVE")
            self.assertEqual(evidence["method"], "llm_api")
            self.assertEqual(
                evidence["reasoning"],
                f"{indicator}={result['scores'][indicator]}; API assessment completed.",
            )
            self.assertTrue(self.FORBIDDEN_CORRECTION_KEYS.isdisjoint(evidence))

    def test_unmapped_result_is_unchanged(self) -> None:
        with patch.object(main, "evaluate_text_with_llm_api", side_effect=simulated_api_result):
            result = main.score_record(
                make_record("999999"), self.knowledge, text_evaluator="api"
            )

        self.assertEqual(result["scores"]["R3"], 1)
        self.assertEqual(result["scores"]["R5"], 0)
        self.assertEqual(result["scores"]["R7"], 0)
        for indicator in ("R3", "R5", "R7"):
            self.assertEqual(result["evidence"][indicator]["reasoning"], "Synthetic API result.")


if __name__ == "__main__":
    unittest.main()
