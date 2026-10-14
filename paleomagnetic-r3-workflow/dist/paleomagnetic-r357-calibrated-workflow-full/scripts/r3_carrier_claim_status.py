#!/usr/bin/env python3
"""Classify whether a carrier-identification claim is usable R3 evidence."""

from __future__ import annotations

from collections.abc import Mapping
import re
from typing import Any


AFFIRMED = "AFFIRMED"
NEGATED = "NEGATED"
UNRESOLVED = "UNRESOLVED"

_MINERAL = (
    r"magnetite|titanomagnetite|titano[- ]?magnetite|hematite|haematite|"
    r"maghemite|pyrrhotite|greigite|goethite|ilmenite"
)
_CARRIER = (
    rf"(?:{_MINERAL}|(?:magnetic\s+|remanence\s+)?carrier(?:s)?)"
)
_STATUS_WORD = r"(?:unknown|unclear|uncertain|unresolved|inconclusive|ambiguous|indeterminate)"
_IDENTIFY_VERB = r"(?:identif(?:y|ied)|determin(?:e|ed)|establish(?:ed)?|resolv(?:e|ed)|confirm(?:ed)?|constrain(?:ed)?)"

_NEGATED_RE = re.compile(
    rf"\b{_CARRIER}\b.{{0,65}}\b(?:is|are|was|were)\s+not\s+"
    rf"(?:the\s+)?(?:magnetic\s+|remanence\s+)?carrier\b|"
    rf"\b(?:exclude[sd]?|rule[sd]?\s+out|reject(?:ed|s)?)\b.{{0,90}}\b{_CARRIER}\b|"
    rf"\bno\s+evidence\b.{{0,110}}\b(?:carr(?:y|ies|ied)|{_CARRIER})\b|"
    rf"\b(?:data|results|measurements|observations)\b.{{0,35}}\b"
    rf"(?:do|does|did)\s+not\s+support\b.{{0,90}}\b{_CARRIER}\b",
    re.I | re.S,
)

_UNRESOLVED_RE = re.compile(
    rf"\b{_CARRIER}\b.{{0,100}}\b(?:remain(?:s|ed)?\s+)?{_STATUS_WORD}\b|"
    rf"\b{_STATUS_WORD}\b.{{0,90}}\b(?:which\s+)?"
    rf"(?:mineral(?:s)?\b.{{0,30}}\b(?:carr(?:y|ies|ied)|carrier)|"
    rf"(?:magnetic\s+|remanence\s+)?carrier(?:s)?)\b|"
    rf"\b{_CARRIER}\b.{{0,70}}\b(?:has|have|had)\s+yet\s+to\s+be\s+{_IDENTIFY_VERB}\b|"
    rf"\b{_CARRIER}\b.{{0,70}}\b(?:could|can|was|were|is|are|has|have|had)\s+"
    rf"(?:not|never)\s+(?:be\s+)?{_IDENTIFY_VERB}\b|"
    rf"\b{_CARRIER}\b.{{0,70}}\bcannot\s+(?:be\s+)?{_IDENTIFY_VERB}\b|"
    rf"\bno\s+(?:specific\s+|definite\s+|particular\s+)?{_CARRIER}\b.{{0,55}}"
    rf"\b(?:(?:was|were|is|are)\s+)?(?:{_IDENTIFY_VERB}|detected|observed|found)\b|"
    rf"\b(?:data|results|measurements|observations)\b.{{0,35}}\b"
    rf"(?:do|does|did)\s+not\s+(?:constrain|identify|determine|establish|resolve|confirm)\b"
    rf".{{0,90}}\b(?:which\s+mineral|{_CARRIER})\b|"
    rf"\b(?:cannot|can't|could\s+not|unable\s+to|fail(?:ed|s)?\s+to)\s+"
    rf"(?:identify|determine|establish|resolve|confirm)\b.{{0,90}}\b"
    rf"(?:which\s+mineral|{_CARRIER})\b",
    re.I | re.S,
)
_CARRIER_CLAUSE_RE = re.compile(
    rf"\b{_CARRIER}\b|\bwhich\s+mineral(?:s)?\b.{{0,35}}\bcarr(?:y|ies|ied)\b",
    re.I | re.S,
)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+|,?\s+(?=(?:but|although|however|whereas)\b)",
    re.I,
)
_ADVERSATIVE_UNRESOLVED_RE = re.compile(
    rf"\b{_CARRIER}\b.{{0,100}}\bbut\b.{{0,70}}\b"
    rf"(?:results?|data|identity|carrier|mineral(?:s)?)\b.{{0,35}}\b"
    rf"{_STATUS_WORD}\b",
    re.I | re.S,
)


def _clean(value: Any) -> str:
    return " ".join("" if value is None else str(value).split())


def classify_carrier_claim(
    text: Any,
    *,
    affirmed: bool | None = None,
    executed: bool | None = None,
) -> str:
    """Return AFFIRMED, NEGATED, or UNRESOLVED for one local carrier claim."""

    if affirmed is False or executed is False:
        return NEGATED
    claim = _clean(text)
    if not claim:
        return UNRESOLVED
    if _ADVERSATIVE_UNRESOLVED_RE.search(claim):
        return UNRESOLVED
    relevant_clauses = [
        clause.strip(" ,")
        for clause in _CLAUSE_SPLIT_RE.split(claim)
        if _CARRIER_CLAUSE_RE.search(clause)
    ]
    if not relevant_clauses:
        return UNRESOLVED
    if any(_NEGATED_RE.search(clause) for clause in relevant_clauses):
        return NEGATED
    if any(_UNRESOLVED_RE.search(clause) for clause in relevant_clauses):
        return UNRESOLVED
    return AFFIRMED


def classify_carrier_hit(hit: Mapping[str, Any] | None) -> str:
    if not isinstance(hit, Mapping):
        return UNRESOLVED
    declared_status = re.sub(
        r"[^a-z]",
        "",
        _clean(hit.get("claim_status") or hit.get("status")).casefold(),
    )
    if declared_status in {"negated", "rejected", "notexecuted"}:
        return NEGATED
    if declared_status in {"unknown", "uncertain", "unresolved", "hypothetical"}:
        return UNRESOLVED
    text = hit.get("claim_text") or hit.get("quote") or hit.get("matched_text")
    return classify_carrier_claim(
        text,
        affirmed=hit.get("affirmed"),
        executed=hit.get("executed"),
    )


def carrier_claim_is_affirmed(hit: Mapping[str, Any] | None) -> bool:
    return classify_carrier_hit(hit) == AFFIRMED


__all__ = [
    "AFFIRMED",
    "NEGATED",
    "UNRESOLVED",
    "carrier_claim_is_affirmed",
    "classify_carrier_claim",
    "classify_carrier_hit",
]
