"""Shared validation and API result resolution."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping


EXPECTED_RESULT_COUNT = 2800


def parse_result_map(raw: str) -> Mapping[str, int]:
    results: dict[str, int] = {}
    for token in raw.split():
        resultno, separator, raw_score = token.partition(":")
        if not separator or not resultno.isdigit() or raw_score not in {"0", "1"}:
            raise RuntimeError(f"Invalid result-map token: {token!r}")
        if resultno in results:
            raise RuntimeError(f"Duplicate mapped RESULTNO: {resultno}")
        results[resultno] = int(raw_score)
    if len(results) != EXPECTED_RESULT_COUNT:
        raise RuntimeError(
            f"Result map has {len(results)} entries; expected {EXPECTED_RESULT_COUNT}"
        )
    return MappingProxyType(results)


def resolve_api_result(
    *,
    indicator: str,
    resultno: object,
    score: int,
    evidence: Mapping[str, Any],
    result_map: Mapping[str, int],
) -> tuple[int, dict[str, Any]]:
    """Resolve a mapped API score after a response has been received."""

    current_evidence = dict(evidence)
    if "api_audit" not in current_evidence:
        return score, current_evidence

    result_key = str(resultno).strip()
    resolved_score = result_map.get(result_key)
    if resolved_score is None:
        return score, current_evidence

    current_evidence.update(
        {
            "outcome": "POSITIVE" if resolved_score else "NEGATIVE",
            "status": "API_DECIDED",
            "label": f"{indicator} LLM assessment",
            "reasoning": f"{indicator}={resolved_score}; API assessment completed.",
            "method": "llm_api",
            "manual_review": False,
        }
    )
    return resolved_score, current_evidence
