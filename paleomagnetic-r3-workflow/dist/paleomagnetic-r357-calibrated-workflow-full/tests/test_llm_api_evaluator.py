from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from llm_api_evaluator import (  # noqa: E402
    ApiSettings,
    LLMApiEvaluator,
    OutputContractError,
    load_api_settings,
    prepare_record,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class LLMApiEvaluatorTests(unittest.TestCase):
    def settings(self) -> ApiSettings:
        return ApiSettings(
            provider="openai_compatible_chat",
            base_url="https://model.example/v1",
            chat_path="/chat/completions",
            model="test-model",
            api_key="secret-value",
            max_retries=0,
        )

    def record(self) -> dict:
        return {
            "indicator": "R3",
            "RESULTNO": "101",
            "REFNO": "22",
            "ROCKNAME": "Unit A",
            "R3": 0,
            "parameters": {
                "R3_METHOD_EVIDENCE": True,
                "nested": {"STANDARD_R3": 1},
            },
            "evidence": [
                {
                    "page": 8,
                    "source": "paper.pdf",
                    "text": "Magnetite carries the stable ChRM in Unit A.",
                }
            ],
        }

    def test_prepare_record_removes_answer_fields_but_keeps_scientific_fields(self) -> None:
        prepared, stripped = prepare_record(self.record())
        self.assertNotIn("R3", prepared)
        self.assertTrue(prepared["parameters"]["R3_METHOD_EVIDENCE"])
        self.assertNotIn("STANDARD_R3", prepared["parameters"]["nested"])
        self.assertEqual(stripped, ["R3", "parameters.nested.STANDARD_R3"])

    def test_load_settings_uses_environment_without_persisting_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "api.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "provider": "openai_compatible_chat",
                        "base_url": "",
                        "base_url_env": "BASE",
                        "model": "",
                        "model_env": "MODEL",
                        "api_key_env": "KEY",
                        "supported_indicators": ["R3", "R5", "R7"],
                    }
                ),
                encoding="utf-8",
            )
            settings = load_api_settings(
                path,
                environ={"BASE": "https://example.test/v1", "MODEL": "m", "KEY": "top-secret"},
            )
        self.assertEqual(settings.endpoint, "https://example.test/v1/chat/completions")
        self.assertNotIn("top-secret", repr(settings))
        self.assertNotIn("top-secret", json.dumps(settings.audit()))

    def test_evaluate_calls_api_and_returns_audited_result(self) -> None:
        captured = {}

        def opener(request, timeout):
            captured["url"] = request.full_url
            captured["timeout"] = timeout
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            model_result = {
                "indicator": "R3",
                "score": 1,
                "outcome": "POSITIVE",
                "confidence": 0.91,
                "reasoning": "原文明确将磁铁矿与稳定特征剩磁联系起来。",
                "evidence_used": [
                    {
                        "quote": "Magnetite carries the stable ChRM in Unit A.",
                        "source": "paper.pdf",
                        "page": 8,
                    }
                ],
                "manual_review": False,
            }
            return FakeResponse(
                {
                    "id": "response-1",
                    "choices": [{"message": {"content": json.dumps(model_result)}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20},
                }
            )

        result = LLMApiEvaluator(self.settings(), opener=opener).evaluate(self.record())
        self.assertEqual(result["score"], 1)
        self.assertEqual(result["RESULTNO"], "101")
        self.assertEqual(captured["url"], "https://model.example/v1/chat/completions")
        self.assertEqual(captured["authorization"], "Bearer secret-value")
        sent = json.dumps(captured["body"], ensure_ascii=False)
        self.assertNotIn('"R3": 0', sent)
        self.assertNotIn("STANDARD_R3", sent)
        self.assertIn("R3_METHOD_EVIDENCE", sent)
        audit_text = json.dumps(result["api_audit"], ensure_ascii=False)
        self.assertNotIn("secret-value", audit_text)

    def test_non_verbatim_model_quote_is_rejected(self) -> None:
        def opener(_request, _timeout=None, **_kwargs):
            model_result = {
                "indicator": "R3",
                "score": 1,
                "outcome": "POSITIVE",
                "reasoning": "unsupported",
                "evidence_used": [{"quote": "This quote was invented."}],
            }
            return FakeResponse(
                {"choices": [{"message": {"content": json.dumps(model_result)}}]}
            )

        with self.assertRaises(OutputContractError):
            LLMApiEvaluator(self.settings(), opener=opener).evaluate(self.record())


if __name__ == "__main__":
    unittest.main()
