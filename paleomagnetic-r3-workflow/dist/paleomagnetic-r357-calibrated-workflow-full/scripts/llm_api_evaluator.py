#!/usr/bin/env python3
"""Evaluate R3/R5/R7 records through an explicit, auditable LLM API.

This module is an optional integration point. The default R1-R7 workflow does
not import it and never sends data over the network. Callers may either import
``LLMApiEvaluator`` or run this file against a JSONL input outside Codex.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/llm_api.example.json"
SUPPORTED_INDICATORS = ("R3", "R5", "R7")
IDENTITY_FIELDS = ("RESULTNO", "REFNO", "ROCKNAME")
PROMPT_VERSION = "r357_llm_api_prompt_v1"
CRITERIA_VERSION = "r357_llm_api_criteria_v1"

CRITERIA = {
    "R3": (
        "Decide whether the target has qualifying magnetic-carrier evidence: "
        "a carrier-diagnostic rock-magnetic or mineralogical investigation, an "
        "identified or scientifically constrained carrier, and an interpretation "
        "connecting that evidence to remanence origin, stability, alteration, or "
        "primary/secondary significance. Evidence must be target-bound."
    ),
    "R5": (
        "Decide whether the target direction has adequate structural and "
        "paleohorizontal control. Consider explicit missing/unknown structural "
        "attitude, uncorrected tilting, rotation, allochthony, post-folding "
        "magnetization, and explicit rescue such as reliable bedding correction, "
        "near-horizontal attitude, or a positive field test."
    ),
    "R7": (
        "Decide whether the target magnetization is not invalidated by a "
        "demonstrably younger pole, remagnetization, secondary overprint, or age "
        "reset. Do not reject it merely because no comparison is supplied."
    ),
}

_EXACT_ANSWER_KEYS = {
    "ANSWER",
    "GOLD",
    "GOLD_LABEL",
    "LABEL",
    "OUTCOME",
    "PREDICTION",
    "SCORE",
    "STANDARD",
    "TRUE_LABEL",
}
_FORBIDDEN_SOURCE_KEYS = {
    "EVALUATION_RESULTS",
    "PMAGRESULT",
    "PREVIOUS_PREDICTIONS",
    "ROCKMAG",
    "ROCKUNIT",
    "SOURCE_METADATA",
}
_R_ANSWER_KEY = re.compile(
    r"^(?:(?:STANDARD|GOLD|TRUE|PREDICTED|LABEL)_)?R[1-7]"
    r"(?:_(?:SCORE|LABEL|RESULT|RESULT_JSON|OUTCOME|PREDICTION))?$"
)


class LLMApiError(RuntimeError):
    """Base error for API configuration, requests, or model output."""


class ConfigurationError(LLMApiError):
    """Raised when API configuration is missing or unsafe."""


class InputContractError(LLMApiError):
    """Raised when an evaluation input violates the JSONL contract."""


class OutputContractError(LLMApiError):
    """Raised when the model does not return the required structured result."""


class ApiRequestError(LLMApiError):
    """Raised when the configured API request fails."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _is_forbidden_key(value: Any) -> bool:
    key = re.sub(r"[^A-Za-z0-9]+", "_", str(value)).strip("_").upper()
    return (
        key in _EXACT_ANSWER_KEYS
        or key in _FORBIDDEN_SOURCE_KEYS
        or bool(_R_ANSWER_KEY.fullmatch(key))
    )


def _sanitize_value(value: Any, path: str, stripped: list[str]) -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            child_path = f"{path}.{key}" if path else key
            if _is_forbidden_key(key):
                stripped.append(child_path)
                continue
            result[key] = _sanitize_value(item, child_path, stripped)
        return result
    if isinstance(value, (list, tuple)):
        return [
            _sanitize_value(item, f"{path}[{index}]", stripped)
            for index, item in enumerate(value)
        ]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def prepare_record(
    record: Mapping[str, Any],
    *,
    supported_indicators: Sequence[str] = SUPPORTED_INDICATORS,
    max_input_chars: int = 120000,
) -> tuple[dict[str, Any], list[str]]:
    """Validate one input and remove answer-like fields before API submission."""

    indicator = _clean(record.get("indicator")).upper()
    allowed = {str(value).upper() for value in supported_indicators}
    if indicator not in allowed:
        raise InputContractError(
            f"indicator must be one of {', '.join(sorted(allowed))}; got {indicator!r}"
        )

    identity = {name: _clean(record.get(name)) for name in IDENTITY_FIELDS}
    missing = [name for name, value in identity.items() if not value]
    if missing:
        raise InputContractError("missing identity field(s): " + ", ".join(missing))

    parameters = record.get("parameters") or {}
    if not isinstance(parameters, Mapping):
        raise InputContractError("parameters must be a JSON object")
    knowledge = record.get("knowledge") or {}
    if not isinstance(knowledge, Mapping):
        raise InputContractError("knowledge must be a JSON object")

    evidence_value = record.get("evidence")
    if evidence_value is None and _clean(record.get("paper_text")):
        evidence_value = [
            {
                "text": str(record["paper_text"]),
                "source": _clean(record.get("paper_source")) or "original_paper",
            }
        ]
    evidence_value = evidence_value or []
    if not isinstance(evidence_value, list) or not all(
        isinstance(item, Mapping) for item in evidence_value
    ):
        raise InputContractError("evidence must be a JSON array of objects")

    stripped = [str(key) for key in record if _is_forbidden_key(key)]
    safe_parameters = _sanitize_value(parameters, "parameters", stripped)
    safe_knowledge = _sanitize_value(knowledge, "knowledge", stripped)
    safe_evidence = _sanitize_value(evidence_value, "evidence", stripped)
    prepared = {
        "indicator": indicator,
        "identity": identity,
        "criteria_version": CRITERIA_VERSION,
        "criterion": CRITERIA[indicator],
        "parameters": safe_parameters,
        "knowledge": safe_knowledge,
        "evidence": safe_evidence,
    }
    encoded = canonical_json(prepared)
    if len(encoded) > max_input_chars:
        raise InputContractError(
            f"prepared input has {len(encoded)} characters, above max_input_chars={max_input_chars}; "
            "retrieve shorter target-bound evidence before calling the API"
        )
    return prepared, sorted(set(stripped))


@dataclass(frozen=True)
class ApiSettings:
    provider: str
    base_url: str
    chat_path: str
    model: str
    api_key: str = field(repr=False)
    api_key_required: bool = True
    timeout_seconds: float = 120.0
    max_retries: int = 2
    temperature: float = 0.0
    max_output_tokens: int = 1200
    max_input_chars: int = 120000
    response_language: str = "zh-CN"
    use_json_response_format: bool = True
    supported_indicators: tuple[str, ...] = SUPPORTED_INDICATORS

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/" + self.chat_path.lstrip("/")

    def audit(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "api_key_configured": bool(self.api_key),
            "supported_indicators": list(self.supported_indicators),
        }


def _resolved_text(
    payload: Mapping[str, Any],
    name: str,
    env_key: str,
    environ: Mapping[str, str],
) -> str:
    env_name = _clean(payload.get(env_key))
    if env_name and _clean(environ.get(env_name)):
        return _clean(environ[env_name])
    return _clean(payload.get(name))


def load_api_settings(
    path: Path = DEFAULT_CONFIG,
    *,
    environ: Mapping[str, str] | None = None,
) -> ApiSettings:
    """Load API settings; environment variables override non-secret config values."""

    environ = os.environ if environ is None else environ
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConfigurationError(f"API configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        raise ConfigurationError("LLM API configuration requires schema_version=1")

    provider = _clean(payload.get("provider"))
    if provider != "openai_compatible_chat":
        raise ConfigurationError("provider must be openai_compatible_chat")
    base_url = _resolved_text(payload, "base_url", "base_url_env", environ)
    parsed = urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigurationError(
            "base_url must be an absolute http(s) URL, supplied in config or its configured environment variable"
        )
    if parsed.username or parsed.password:
        raise ConfigurationError("base_url must not contain embedded credentials")
    if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigurationError("non-local API endpoints must use https")

    model = _resolved_text(payload, "model", "model_env", environ)
    if not model:
        raise ConfigurationError(
            "model must be supplied in config or its configured environment variable"
        )
    key_env = _clean(payload.get("api_key_env"))
    api_key = _clean(environ.get(key_env)) if key_env else ""
    api_key_required = bool(payload.get("api_key_required", True))
    if api_key_required and not api_key:
        raise ConfigurationError(
            f"API key is required; set the environment variable {key_env or '<api_key_env>'}"
        )

    supported = tuple(
        _clean(value).upper() for value in payload.get("supported_indicators") or []
    )
    if not supported or not set(supported).issubset(SUPPORTED_INDICATORS):
        raise ConfigurationError("supported_indicators must be a non-empty subset of R3, R5, R7")
    chat_path = _clean(payload.get("chat_path")) or "/chat/completions"
    if "://" in chat_path or "?" in chat_path or "#" in chat_path:
        raise ConfigurationError("chat_path must be a relative URL path without query or fragment")
    settings = ApiSettings(
        provider=provider,
        base_url=base_url,
        chat_path=chat_path,
        model=model,
        api_key=api_key,
        api_key_required=api_key_required,
        timeout_seconds=float(payload.get("timeout_seconds", 120)),
        max_retries=int(payload.get("max_retries", 2)),
        temperature=float(payload.get("temperature", 0)),
        max_output_tokens=int(payload.get("max_output_tokens", 1200)),
        max_input_chars=int(payload.get("max_input_chars", 120000)),
        response_language=_clean(payload.get("response_language")) or "zh-CN",
        use_json_response_format=bool(payload.get("use_json_response_format", True)),
        supported_indicators=supported,
    )
    if settings.timeout_seconds <= 0:
        raise ConfigurationError("timeout_seconds must be positive")
    if settings.max_retries < 0 or settings.max_retries > 10:
        raise ConfigurationError("max_retries must be between 0 and 10")
    if settings.max_output_tokens < 1 or settings.max_input_chars < 1000:
        raise ConfigurationError("max_output_tokens and max_input_chars must be positive")
    return settings


def _response_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return ""


def _parse_json_content(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OutputContractError("model response content is not one JSON object") from exc
    if not isinstance(parsed, dict):
        raise OutputContractError("model response must be a JSON object")
    return parsed


def _normalized_quote(value: Any) -> str:
    return re.sub(r"\s+", " ", _clean(value)).casefold()


def validate_model_output(
    value: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    indicator = _clean(value.get("indicator")).upper()
    if indicator != prepared["indicator"]:
        raise OutputContractError(
            f"model indicator {indicator!r} does not match request {prepared['indicator']}"
        )
    outcome = _clean(value.get("outcome")).upper()
    if outcome not in {"POSITIVE", "NEGATIVE", "UNKNOWN"}:
        raise OutputContractError("outcome must be POSITIVE, NEGATIVE, or UNKNOWN")
    score = value.get("score")
    if isinstance(score, bool):
        score = int(score)
    if outcome == "POSITIVE" and score != 1:
        raise OutputContractError("POSITIVE requires score=1")
    if outcome == "NEGATIVE" and score != 0:
        raise OutputContractError("NEGATIVE requires score=0")
    if outcome == "UNKNOWN" and score is not None:
        raise OutputContractError("UNKNOWN requires score=null")

    reasoning = _clean(value.get("reasoning"))
    if not reasoning:
        raise OutputContractError("reasoning must be non-empty")
    evidence_used = value.get("evidence_used") or []
    if not isinstance(evidence_used, list) or not all(
        isinstance(item, Mapping) for item in evidence_used
    ):
        raise OutputContractError("evidence_used must be an array of objects")

    source_text = "\n".join(
        _clean(item.get("text") or item.get("quote"))
        for item in prepared.get("evidence") or []
        if isinstance(item, Mapping)
    )
    normalized_source = _normalized_quote(source_text)
    normalized_evidence = []
    for index, item in enumerate(evidence_used):
        quote = _clean(item.get("quote"))
        if not quote:
            raise OutputContractError(f"evidence_used[{index}].quote must be non-empty")
        if _normalized_quote(quote) not in normalized_source:
            raise OutputContractError(
                f"evidence_used[{index}].quote is not a verbatim excerpt of submitted evidence"
            )
        normalized_evidence.append(
            {
                "quote": quote,
                "source": _clean(item.get("source")) or "submitted_evidence",
                "page": item.get("page"),
            }
        )
    if outcome != "UNKNOWN" and not normalized_evidence:
        raise OutputContractError("a decided outcome requires at least one verbatim evidence quote")

    confidence = value.get("confidence")
    if confidence is not None:
        try:
            confidence = float(confidence)
        except (TypeError, ValueError) as exc:
            raise OutputContractError("confidence must be a number between 0 and 1") from exc
        if not 0.0 <= confidence <= 1.0:
            raise OutputContractError("confidence must be between 0 and 1")
    return {
        "indicator": indicator,
        "score": score,
        "outcome": outcome,
        "confidence": confidence,
        "reasoning": reasoning,
        "evidence_used": normalized_evidence,
        "manual_review": bool(value.get("manual_review")) or outcome == "UNKNOWN",
    }


class LLMApiEvaluator:
    """OpenAI-compatible chat client for structured R3/R5/R7 evaluation."""

    def __init__(
        self,
        settings: ApiSettings,
        *,
        opener: Callable[..., Any] = urlopen,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings
        self._opener = opener
        self._sleeper = sleeper

    def _messages(self, prepared: Mapping[str, Any]) -> list[dict[str, str]]:
        system = (
            "You are an auditable paleomagnetic reliability evaluator. Use only the "
            "submitted parameters and verbatim evidence. Never infer a missing fact and "
            "never reproduce a prior label. Return exactly one JSON object with keys: "
            "indicator, score, outcome, confidence, reasoning, evidence_used, manual_review. "
            "score must be 1 for POSITIVE, 0 for NEGATIVE, and null for UNKNOWN. "
            "evidence_used must contain verbatim submitted quotes with source and page; a "
            "decided outcome requires at least one quote. Use UNKNOWN when evidence is "
            f"insufficient or conflicting. Write reasoning in {self.settings.response_language}."
        )
        user = "Evaluate this record:\n" + json.dumps(
            prepared, ensure_ascii=False, indent=2, sort_keys=True
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def _request_payload(self, prepared: Mapping[str, Any]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": self._messages(prepared),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_output_tokens,
        }
        if self.settings.use_json_response_format:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, payload: Mapping[str, Any]) -> tuple[dict[str, Any], int]:
        body = canonical_json(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "paleomagnetic-r1-r7-llm-evaluator/1",
        }
        if self.settings.api_key:
            headers["Authorization"] = f"Bearer {self.settings.api_key}"
        request = Request(self.settings.endpoint, data=body, headers=headers, method="POST")

        last_error: Exception | None = None
        for attempt in range(self.settings.max_retries + 1):
            try:
                with self._opener(request, timeout=self.settings.timeout_seconds) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise ApiRequestError("API response is not a JSON object")
                return parsed, attempt + 1
            except HTTPError as exc:
                body_text = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = ApiRequestError(f"API HTTP {exc.code}: {body_text}")
                retryable = exc.code == 429 or exc.code >= 500
                if not retryable or attempt >= self.settings.max_retries:
                    raise last_error from exc
            except (URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = ApiRequestError(f"API request failed: {exc}")
                if attempt >= self.settings.max_retries:
                    raise last_error from exc
            self._sleeper(min(2**attempt, 5))
        raise ApiRequestError(f"API request failed: {last_error}")

    def evaluate(self, record: Mapping[str, Any]) -> dict[str, Any]:
        """Submit one record and return a validated, audit-ready result."""

        prepared, stripped = prepare_record(
            record,
            supported_indicators=self.settings.supported_indicators,
            max_input_chars=self.settings.max_input_chars,
        )
        request_payload = self._request_payload(prepared)
        response, attempts = self._post(request_payload)
        choices = response.get("choices") or []
        if not choices or not isinstance(choices[0], Mapping):
            raise OutputContractError("API response has no choices[0]")
        message = choices[0].get("message") or {}
        if not isinstance(message, Mapping):
            raise OutputContractError("API response choices[0].message is invalid")
        parsed_result = (
            dict(message["parsed"])
            if isinstance(message.get("parsed"), Mapping)
            else _parse_json_content(_response_text(message))
        )
        result = validate_model_output(parsed_result, prepared)
        result.update(prepared["identity"])
        result["api_audit"] = {
            **self.settings.audit(),
            "prompt_version": PROMPT_VERSION,
            "criteria_version": CRITERIA_VERSION,
            "attempts": attempts,
            "request_sha256": sha256_text(canonical_json(request_payload)),
            "response_sha256": sha256_text(canonical_json(response)),
            "response_id": _clean(response.get("id")) or None,
            "usage": response.get("usage") if isinstance(response.get("usage"), Mapping) else None,
            "stripped_answer_fields": stripped,
        }
        return result


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputContractError(f"{path}:{line_number} is invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise InputContractError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise InputContractError(f"{path} contains no input records")
    return rows


def write_evaluations(
    evaluator: LLMApiEvaluator,
    rows: Sequence[Mapping[str, Any]],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(rows, start=1):
                try:
                    result = evaluator.evaluate(row)
                except LLMApiError as exc:
                    raise LLMApiError(f"input row {index}: {exc}") from exc
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, default=None, help="Input JSONL records.")
    parser.add_argument("--output", type=Path, default=None, help="Validated output JSONL.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate endpoint/model/key settings without sending a request.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = load_api_settings(args.config.expanduser().resolve())
        if args.check_config:
            print(json.dumps({"status": "ok", **settings.audit()}, ensure_ascii=False, indent=2))
            return 0
        if args.input is None or args.output is None:
            raise ConfigurationError("--input and --output are required unless --check-config is used")
        if args.limit is not None and args.limit < 1:
            raise ConfigurationError("--limit must be at least 1")
        rows = load_jsonl(args.input.expanduser().resolve(), args.limit)
        write_evaluations(LLMApiEvaluator(settings), rows, args.output.expanduser().resolve())
    except LLMApiError as exc:
        raise SystemExit(f"LLM API evaluation failed: {exc}") from exc
    print(f"Saved {len(rows)} API evaluation row(s): {args.output.expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
