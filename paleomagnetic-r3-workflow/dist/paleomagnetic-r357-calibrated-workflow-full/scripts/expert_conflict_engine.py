#!/usr/bin/env python3
"""Label-free expert-knowledge conflict detection over original-paper evidence.

The module is deliberately independent from the current R3/R5/R7 scorers. It
accepts either their retrieval-hit dictionaries or page-cache text and emits a
compact, provenance-bearing support/conflict analysis. No source database row
or historical indicator label is part of the API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_BASE = ROOT / "data/knowledge/r357_expert_kb_v1.json"
ENGINE_VERSION = "expert_conflict_engine_v4"

RELATIONS = {"SUPPORT", "CONFLICT", "UNKNOWN", "MIXED"}
INDICATORS = {"R3", "R5", "R7"}
CONFLICT_TYPES = {
    "NONE",
    "POLARITY",
    "VALUE_RANGE",
    "TEMPORAL_ORDER",
    "TARGET_SCOPE",
    "COMPONENT_SCOPE",
    "MODALITY",
    "PROVENANCE",
    "EVIDENCE_DISAGREEMENT",
}
_FORBIDDEN_KEY_NAMES = {
    "refno",
    "resultno",
    "answer",
    "answers",
    "answermap",
    "answerlookup",
    "rowanswers",
    "targetanswers",
    "gold",
    "goldlabel",
    "benchmarklabel",
    "standardr3",
    "standardr5",
    "standardr7",
}
_FORBIDDEN_VALUE_RE = re.compile(r"\b(?:REFNO|RESULTNO)\b", re.I)
_GENERIC_TARGET_TOKENS = {
    "and",
    "complex",
    "formation",
    "group",
    "member",
    "rock",
    "rocks",
    "series",
    "suite",
    "unit",
}


class KnowledgeBaseValidationError(ValueError):
    """Raised when a knowledge artifact violates the strict scoring boundary."""


@dataclass(frozen=True)
class _CompiledRule:
    raw: Mapping[str, Any]
    patterns: tuple[re.Pattern[str], ...]
    exclusions: tuple[re.Pattern[str], ...]
    hit_subtypes: frozenset[str]
    claim_scope: str
    require_target_in_claim: bool
    require_exact_target_in_claim: bool
    maximum_target_distance_chars: int | None
    required_claim_patterns: tuple[re.Pattern[str], ...]
    reference_sections_unknown: bool
    context_veto: str


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", "" if value is None else str(value)).strip()


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", "" if value is None else str(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _sha256_text(value: Any) -> str:
    return hashlib.sha256(_clean(value).encode("utf-8")).hexdigest()


def _safe_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return result


def _safe_page(value: Any, fallback: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _walk_for_leakage(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, member in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if normalized in _FORBIDDEN_KEY_NAMES:
                raise KnowledgeBaseValidationError(
                    f"forbidden identifier or answer key at {child_path}"
                )
            if normalized == "labelsupervision" and member is not False:
                raise KnowledgeBaseValidationError(
                    f"label_supervision must be false at {child_path}"
                )
            _walk_for_leakage(member, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, member in enumerate(value):
            _walk_for_leakage(member, f"{path}[{index}]")
        return
    if isinstance(value, str) and _FORBIDDEN_VALUE_RE.search(value):
        raise KnowledgeBaseValidationError(
            f"forbidden row identifier reference in text at {path}"
        )


def validate_knowledge_base(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a mutable copy of a strict, label-free knowledge base."""

    if not isinstance(data, Mapping):
        raise KnowledgeBaseValidationError("knowledge base must be a JSON object")
    _walk_for_leakage(data)
    copy = json.loads(json.dumps(data, ensure_ascii=False))
    if not _clean(copy.get("schema_version")):
        raise KnowledgeBaseValidationError("schema_version is required")
    if not _clean(copy.get("kb_version")):
        raise KnowledgeBaseValidationError("kb_version is required")
    provenance = copy.get("provenance")
    if not isinstance(provenance, Mapping) or provenance.get("label_supervision") is not False:
        raise KnowledgeBaseValidationError(
            "top-level provenance.label_supervision=false is required"
        )
    if provenance.get("contains_row_answer_mapping") is not False:
        raise KnowledgeBaseValidationError(
            "top-level provenance.contains_row_answer_mapping=false is required"
        )

    indicators = copy.get("indicators")
    if not isinstance(indicators, Mapping) or set(indicators) != INDICATORS:
        raise KnowledgeBaseValidationError("indicators must define exactly R3, R5, and R7")
    support_requirements: dict[str, tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]] = {}
    for indicator, definition in indicators.items():
        where = f"indicators.{indicator}.decision_policy"
        policy = definition.get("decision_policy") if isinstance(definition, Mapping) else None
        if not isinstance(policy, Mapping):
            raise KnowledgeBaseValidationError(f"{where} must be an object")
        for boolean_key in (
            "conflict_overrides_incomplete_support",
            "conflict_overrides_support",
        ):
            if boolean_key in policy and not isinstance(policy[boolean_key], bool):
                raise KnowledgeBaseValidationError(
                    f"{where}.{boolean_key} must be boolean"
                )
        required_groups = policy.get("required_support_groups") or []
        if not isinstance(required_groups, list) or any(
            not _clean(group) for group in required_groups
        ):
            raise KnowledgeBaseValidationError(
                f"{where}.required_support_groups must be a list of non-empty group names"
            )
        normalized_required = tuple(_clean(group) for group in required_groups)
        if len(set(normalized_required)) != len(normalized_required):
            raise KnowledgeBaseValidationError(
                f"{where}.required_support_groups must not contain duplicates"
            )
        raw_sets = policy.get("required_support_sets") or []
        if not isinstance(raw_sets, list):
            raise KnowledgeBaseValidationError(f"{where}.required_support_sets must be a list")
        normalized_sets: list[tuple[str, ...]] = []
        for set_index, raw_set in enumerate(raw_sets):
            if not isinstance(raw_set, list) or not raw_set:
                raise KnowledgeBaseValidationError(
                    f"{where}.required_support_sets[{set_index}] must be a non-empty list"
                )
            normalized_set = tuple(_clean(group) for group in raw_set)
            if any(not group for group in normalized_set) or len(set(normalized_set)) != len(
                normalized_set
            ):
                raise KnowledgeBaseValidationError(
                    f"{where}.required_support_sets[{set_index}] must contain unique "
                    "non-empty group names"
                )
            normalized_sets.append(normalized_set)
        if len({tuple(sorted(member)) for member in normalized_sets}) != len(normalized_sets):
            raise KnowledgeBaseValidationError(
                f"{where}.required_support_sets must not contain duplicate alternatives"
            )
        support_requirements[indicator] = (normalized_required, tuple(normalized_sets))
    rules = copy.get("rules")
    if not isinstance(rules, list) or not rules:
        raise KnowledgeBaseValidationError("rules must be a non-empty list")

    seen: set[str] = set()
    seen_indicators: set[str] = set()
    support_groups: dict[str, set[str]] = {indicator: set() for indicator in INDICATORS}
    defeasible_references: list[tuple[str, str, tuple[str, ...]]] = []
    for index, rule in enumerate(rules):
        where = f"rules[{index}]"
        if not isinstance(rule, Mapping):
            raise KnowledgeBaseValidationError(f"{where} must be an object")
        rule_id = _clean(rule.get("rule_id"))
        if not rule_id or rule_id in seen:
            raise KnowledgeBaseValidationError(f"{where}.rule_id must be unique and non-empty")
        seen.add(rule_id)
        indicator = _clean(rule.get("indicator")).upper()
        if indicator not in INDICATORS:
            raise KnowledgeBaseValidationError(f"{where}.indicator is invalid")
        seen_indicators.add(indicator)
        claim = rule.get("claim")
        if not isinstance(claim, Mapping) or not _clean(claim.get("en")) or not _clean(claim.get("zh")):
            raise KnowledgeBaseValidationError(f"{where}.claim requires readable en and zh text")
        effect = rule.get("effect")
        if not isinstance(effect, Mapping):
            raise KnowledgeBaseValidationError(f"{where}.effect must be an object")
        relation = _clean(effect.get("relation")).upper()
        value = effect.get("value")
        if relation not in {"SUPPORT", "CONFLICT", "UNKNOWN"} or (
            value not in {0, 1} and value is not None
        ):
            raise KnowledgeBaseValidationError(
                f"{where}.effect must contain SUPPORT/CONFLICT with a binary value "
                "or UNKNOWN with a null value"
            )
        if (
            (relation == "SUPPORT" and value != 1)
            or (relation == "CONFLICT" and value != 0)
            or (relation == "UNKNOWN" and value is not None)
        ):
            raise KnowledgeBaseValidationError(
                f"{where}.effect relation and value are inconsistent"
            )
        group = _clean(rule.get("group"))
        if not group:
            raise KnowledgeBaseValidationError(f"{where}.group must be non-empty")
        if relation == "SUPPORT":
            support_groups[indicator].add(group)
        unless_support_groups = rule.get("unless_support_groups")
        unless_support_policy = _clean(rule.get("unless_support_policy") or "ANY").upper()
        if unless_support_groups is not None:
            if relation != "CONFLICT":
                raise KnowledgeBaseValidationError(
                    f"{where}.unless_support_groups is only valid for CONFLICT rules"
                )
            if not isinstance(unless_support_groups, list) or not unless_support_groups:
                raise KnowledgeBaseValidationError(
                    f"{where}.unless_support_groups must be a non-empty list"
                )
            normalized_groups = tuple(_clean(member) for member in unless_support_groups)
            if any(not member for member in normalized_groups) or len(set(normalized_groups)) != len(
                normalized_groups
            ):
                raise KnowledgeBaseValidationError(
                    f"{where}.unless_support_groups must contain unique non-empty group names"
                )
            if unless_support_policy not in {"ANY", "ALL"}:
                raise KnowledgeBaseValidationError(
                    f"{where}.unless_support_policy must be ANY or ALL"
                )
            defeasible_references.append((where, indicator, normalized_groups))
        elif "unless_support_policy" in rule:
            raise KnowledgeBaseValidationError(
                f"{where}.unless_support_policy requires unless_support_groups"
            )
        conflict_type = _clean(rule.get("conflict_type") or "NONE").upper()
        if conflict_type not in CONFLICT_TYPES:
            raise KnowledgeBaseValidationError(f"{where}.conflict_type is invalid")
        base_confidence = _safe_float(rule.get("base_confidence"), -1.0)
        if not 0.0 < base_confidence <= 1.0:
            raise KnowledgeBaseValidationError(f"{where}.base_confidence must be in (0, 1]")
        rule_provenance = rule.get("provenance")
        if not isinstance(rule_provenance, Mapping) or rule_provenance.get("label_supervision") is not False:
            raise KnowledgeBaseValidationError(
                f"{where}.provenance.label_supervision=false is required"
            )
        evidence = rule.get("evidence")
        if not isinstance(evidence, Mapping):
            raise KnowledgeBaseValidationError(f"{where}.evidence must be an object")
        claim_scope = _clean(evidence.get("claim_scope") or "window").casefold()
        if claim_scope not in {"window", "same_sentence"}:
            raise KnowledgeBaseValidationError(
                f"{where}.evidence.claim_scope must be window or same_sentence"
            )
        for boolean_key in (
            "require_target_in_claim",
            "require_exact_target_in_claim",
            "reference_sections_unknown",
        ):
            if boolean_key in evidence and not isinstance(evidence[boolean_key], bool):
                raise KnowledgeBaseValidationError(
                    f"{where}.evidence.{boolean_key} must be boolean"
                )
        maximum_target_distance = evidence.get("maximum_target_distance_chars")
        if maximum_target_distance is not None and (
            isinstance(maximum_target_distance, bool)
            or not isinstance(maximum_target_distance, int)
            or maximum_target_distance < 0
        ):
            raise KnowledgeBaseValidationError(
                f"{where}.evidence.maximum_target_distance_chars must be a "
                "non-negative integer"
            )
        if maximum_target_distance is not None and not evidence.get(
            "require_exact_target_in_claim", False
        ):
            raise KnowledgeBaseValidationError(
                f"{where}.evidence.maximum_target_distance_chars requires "
                "require_exact_target_in_claim=true"
            )
        context_veto = _clean(evidence.get("context_veto"))
        if context_veto not in {"", "structural_correction_unnecessary"}:
            raise KnowledgeBaseValidationError(
                f"{where}.evidence.context_veto is invalid"
            )
        minimum_target_binding = evidence.get("minimum_target_binding")
        if minimum_target_binding is not None:
            if isinstance(minimum_target_binding, bool) or not isinstance(
                minimum_target_binding, (int, float)
            ):
                raise KnowledgeBaseValidationError(
                    f"{where}.evidence.minimum_target_binding must be numeric"
                )
            if not math.isfinite(float(minimum_target_binding)) or not (
                0.0 <= float(minimum_target_binding) <= 1.0
            ):
                raise KnowledgeBaseValidationError(
                    f"{where}.evidence.minimum_target_binding must be in [0, 1]"
                )
        patterns = evidence.get("patterns") or []
        hit_subtypes = evidence.get("hit_subtypes") or []
        if not patterns and not hit_subtypes:
            raise KnowledgeBaseValidationError(
                f"{where}.evidence requires patterns or hit_subtypes"
            )
        try:
            for expression in [
                *patterns,
                *(evidence.get("exclude_patterns") or []),
                *(evidence.get("required_claim_patterns") or []),
            ]:
                re.compile(str(expression), re.I | re.S)
        except re.error as exc:
            raise KnowledgeBaseValidationError(f"invalid regex in {where}: {exc}") from exc
    if seen_indicators != INDICATORS:
        raise KnowledgeBaseValidationError("at least one rule is required for each indicator")
    for where, indicator, referenced_groups in defeasible_references:
        missing = sorted(set(referenced_groups) - support_groups[indicator])
        if missing:
            raise KnowledgeBaseValidationError(
                f"{where}.unless_support_groups references missing SUPPORT groups for "
                f"{indicator}: {', '.join(missing)}"
            )
    for indicator, (required_groups, required_sets) in support_requirements.items():
        referenced = set(required_groups)
        for required_set in required_sets:
            referenced.update(required_set)
        missing = sorted(referenced - support_groups[indicator])
        if missing:
            raise KnowledgeBaseValidationError(
                f"indicators.{indicator}.decision_policy references missing SUPPORT groups: "
                f"{', '.join(missing)}"
            )
    return copy


def load_knowledge_base(path: Path | str = DEFAULT_KNOWLEDGE_BASE) -> dict[str, Any]:
    """Load and validate a knowledge artifact from disk."""

    source = Path(path)
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseValidationError(f"invalid JSON in {source}: {exc}") from exc
    return validate_knowledge_base(data)


def _compile_rule(rule: Mapping[str, Any]) -> _CompiledRule:
    evidence = rule.get("evidence") or {}
    return _CompiledRule(
        raw=rule,
        patterns=tuple(
            re.compile(str(expression), re.I | re.S)
            for expression in evidence.get("patterns") or []
        ),
        exclusions=tuple(
            re.compile(str(expression), re.I | re.S)
            for expression in evidence.get("exclude_patterns") or []
        ),
        hit_subtypes=frozenset(
            _clean(value).casefold() for value in evidence.get("hit_subtypes") or []
        ),
        claim_scope=_clean(evidence.get("claim_scope") or "window").casefold(),
        require_target_in_claim=bool(evidence.get("require_target_in_claim", False)),
        require_exact_target_in_claim=bool(
            evidence.get("require_exact_target_in_claim", False)
        ),
        maximum_target_distance_chars=evidence.get("maximum_target_distance_chars"),
        required_claim_patterns=tuple(
            re.compile(str(expression), re.I | re.S)
            for expression in evidence.get("required_claim_patterns") or []
        ),
        reference_sections_unknown=bool(evidence.get("reference_sections_unknown", False)),
        context_veto=_clean(evidence.get("context_veto")),
    )


def _target_tokens(target: str) -> tuple[str, ...]:
    tokens = tuple(dict.fromkeys(re.findall(r"[a-z0-9]+", _fold(target))))
    specific = tuple(
        token for token in tokens if len(token) > 2 and token not in _GENERIC_TARGET_TOKENS
    )
    return specific or tokens


def _target_binding(target: str, quote: str, page_text: str, multi_target: bool) -> tuple[float, str]:
    target_folded = _fold(target)
    quote_folded = _fold(quote)
    page_folded = _fold(page_text)
    if target_folded and re.search(
        r"(?<![a-z0-9])" + re.escape(target_folded) + r"(?![a-z0-9])",
        quote_folded,
    ):
        return 1.0, "target_same_evidence_window"
    tokens = _target_tokens(target)
    if tokens:
        hits = sum(
            bool(re.search(r"(?<![a-z0-9])" + re.escape(token) + r"(?![a-z0-9])", quote_folded))
            for token in tokens
        )
        if hits >= max(1, math.ceil(len(tokens) * 0.6)):
            return 0.86, "target_tokens_same_evidence_window"
    if target_folded and target_folded in page_folded:
        return (0.48 if multi_target else 0.7), "target_same_page"
    return (0.16 if multi_target else 0.52), (
        "unbound_multi_target" if multi_target else "single_target_document_assumption"
    )


def _source_quality(source: Mapping[str, Any] | None, text_available: bool = True) -> float:
    if not text_available:
        return 0.0
    source = source or {}
    flags = set(str(flag) for flag in source.get("source_quality_flags") or [])
    value = _safe_float(source.get("source_quality"), 0.94)
    if "glyph_or_cid_text" in flags:
        value -= 0.25
    if "short_text" in flags or "possibly_incomplete_extraction" in flags:
        value -= 0.16
    if "no_extractable_text" in flags:
        return 0.05
    return max(0.1, min(1.0, value))


def _quote_window(text: str, start: int, end: int, radius: int = 260) -> str:
    paragraph_start = text.rfind("\n\n", 0, start)
    paragraph_end = text.find("\n\n", end)
    left = paragraph_start + 2 if paragraph_start >= 0 else max(0, start - radius)
    right = paragraph_end if paragraph_end >= 0 else min(len(text), end + radius)
    if right - left > 1100:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
    return _clean(text[left:right])[:1100]


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n\s*\n+")
_REFERENCE_SECTION_RE = re.compile(
    r"\b(?:references?|bibliography|literature\s+cited|works\s+cited)\b", re.I
)
_NEGLIGIBLE_DEFORMATION_RE = re.compile(
    r"\bundeformed\b"
    r"|\b(?:no|without)\s+(?:(?:discernible|significant|appreciable|tectonic|structural)\s+){0,2}"
    r"(?:disturbance|deformation)\b"
    r"|\b(?:negligible|minimal|insignificant|very\s+small)\s+(?:bedding\s+)?(?:tilt|dip|deformation)\b"
    r"|\b(?:tilt|dip|deformation)\s+(?:was|is|remains?|is\s+considered)?\s*"
    r"(?:negligible|minimal|insignificant)\b",
    re.I,
)
_LOW_TILT_AFTER_RE = re.compile(
    r"\b(?:tilt|dip)\b.{0,35}?"
    r"(?P<comparator>(?:less\s+than|below|under|<)\s*)?"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?:degrees?|deg\.?|°)",
    re.I | re.S,
)
_LOW_TILT_BEFORE_RE = re.compile(
    r"(?P<comparator>(?:less\s+than|below|under|<)\s*)?"
    r"(?<![\d.])(?P<value>\d+(?:\.\d+)?)\s*(?:degrees?|deg\.?|°)"
    r".{0,20}?\b(?:tilt|dip)\b",
    re.I | re.S,
)
_ADJACENT_CAUSAL_RE = re.compile(
    r"\b(?:because|since|therefore|thus|hence|consequently|accordingly|so|given)\b"
    r"|\bdue\s+to\b|\bowing\s+to\b"
    r"|\b(?:correction|correcting)\b.{0,25}\b(?:unnecessary|not\s+(?:needed|required|necessary))\b",
    re.I | re.S,
)


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = boundary.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = boundary.end()
    if text[start:].strip():
        spans.append((start, len(text)))
    return spans or [(0, len(text))]


def _claim_for_span(
    text: str, start: int, end: int
) -> tuple[str, bool, int | None, int, int]:
    for index, (sentence_start, sentence_end) in enumerate(_sentence_spans(text)):
        if start >= sentence_start and end <= sentence_end:
            return (
                _clean(text[sentence_start:sentence_end]),
                True,
                index,
                sentence_start,
                sentence_end,
            )
    claim_start = max(0, start - 180)
    claim_end = min(len(text), end + 180)
    return _clean(text[claim_start:claim_end]), False, None, claim_start, claim_end


def _exact_target_spans(target: str, text: str, offset: int = 0) -> list[tuple[int, int]]:
    words = re.findall(r"[A-Za-z0-9]+", unicodedata.normalize("NFKD", target))
    if not words:
        return []
    expression = r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]+".join(
        re.escape(word) for word in words
    ) + r"(?![A-Za-z0-9])"
    return [
        (offset + match.start(), offset + match.end())
        for match in re.finditer(expression, text, re.I)
    ]


def _interval_distance(
    first_start: int,
    first_end: int,
    second_start: int,
    second_end: int,
) -> int:
    if first_end < second_start:
        return second_start - first_end
    if second_end < first_start:
        return first_start - second_end
    return 0


def _has_negligible_deformation(text: str) -> bool:
    if _NEGLIGIBLE_DEFORMATION_RE.search(text):
        return True
    for pattern in (_LOW_TILT_AFTER_RE, _LOW_TILT_BEFORE_RE):
        for match in pattern.finditer(text):
            value = _safe_float(match.group("value"), 90.0)
            if value < 10.0 or (match.group("comparator") and value <= 10.0):
                return True
    return False


def _structural_correction_unnecessary(text: str, start: int, end: int) -> bool:
    spans = _sentence_spans(text)
    sentence_index = next(
        (
            index
            for index, (sentence_start, sentence_end) in enumerate(spans)
            if start >= sentence_start and end <= sentence_end
        ),
        None,
    )
    if sentence_index is None:
        return False
    current_start, current_end = spans[sentence_index]
    current = text[current_start:current_end]
    if _has_negligible_deformation(current):
        return True
    for neighbor_index in (sentence_index - 1, sentence_index + 1):
        if not 0 <= neighbor_index < len(spans):
            continue
        neighbor_start, neighbor_end = spans[neighbor_index]
        neighbor = text[neighbor_start:neighbor_end]
        if not _has_negligible_deformation(neighbor):
            continue
        combined_start = min(current_start, neighbor_start)
        combined_end = max(current_end, neighbor_end)
        if _ADJACENT_CAUSAL_RE.search(text[combined_start:combined_end]):
            return True
    return False


def _is_reference_section(section: str) -> bool:
    return bool(_REFERENCE_SECTION_RE.search(_clean(section)))


def _claim_scope_gate(
    rule: _CompiledRule,
    *,
    text: str,
    start: int,
    end: int,
    target: str,
    section: str,
) -> tuple[bool, str, dict[str, Any], str]:
    claim, same_sentence, _, claim_start, claim_end = _claim_for_span(text, start, end)
    metadata: dict[str, Any] = {
        "policy": rule.claim_scope,
        "passed": True,
        "reason": "claim_scope_satisfied",
        "text": claim,
        "text_sha256": _sha256_text(claim),
    }
    if rule.reference_sections_unknown and _is_reference_section(section):
        metadata.update(passed=False, reason="reference_section")
        return False, "PROVENANCE", metadata, claim
    if rule.claim_scope == "same_sentence" and not same_sentence:
        metadata.update(passed=False, reason="cross_sentence_match")
        return False, "COMPONENT_SCOPE", metadata, claim
    if rule.require_target_in_claim:
        claim_binding, claim_binding_method = _target_binding(target, claim, claim, True)
        metadata["target_binding"] = {
            "method": claim_binding_method,
            "score": round(claim_binding, 6),
        }
        if claim_binding < 0.86:
            metadata.update(passed=False, reason="target_not_in_same_claim")
            return False, "TARGET_SCOPE", metadata, claim
    if rule.require_exact_target_in_claim:
        exact_spans = _exact_target_spans(
            target,
            text[claim_start:claim_end],
            claim_start,
        )
        if not exact_spans:
            metadata.update(passed=False, reason="exact_target_not_in_same_claim")
            return False, "TARGET_SCOPE", metadata, claim
        distance = min(
            _interval_distance(target_start, target_end, start, end)
            for target_start, target_end in exact_spans
        )
        metadata["exact_target_distance_chars"] = distance
        maximum = rule.maximum_target_distance_chars
        if maximum is not None and distance > maximum:
            metadata.update(
                passed=False,
                reason="target_too_far_from_rule_match",
                maximum_target_distance_chars=maximum,
            )
            return False, "TARGET_SCOPE", metadata, claim
    if rule.context_veto == "structural_correction_unnecessary" and (
        _structural_correction_unnecessary(text, start, end)
    ):
        metadata.update(passed=False, reason="structural_correction_unnecessary")
        return False, "MODALITY", metadata, claim
    missing = [
        pattern.pattern
        for pattern in rule.required_claim_patterns
        if pattern.search(claim) is None
    ]
    if missing:
        metadata.update(
            passed=False,
            reason="missing_required_claim_element",
            missing_required_patterns=missing,
        )
        return False, "COMPONENT_SCOPE", metadata, claim
    return True, "NONE", metadata, claim


def _claim_policy_active(rule: _CompiledRule) -> bool:
    return bool(
        rule.claim_scope != "window"
        or rule.require_target_in_claim
        or rule.require_exact_target_in_claim
        or rule.maximum_target_distance_chars is not None
        or rule.required_claim_patterns
        or rule.reference_sections_unknown
        or rule.context_veto
    )


def _rule_excluded(rule: _CompiledRule, context: str) -> bool:
    return any(pattern.search(context) for pattern in rule.exclusions)


def _minimum_binding(rule: _CompiledRule) -> float:
    evidence = rule.raw.get("evidence") or {}
    return max(0.0, min(1.0, _safe_float(evidence.get("minimum_target_binding"), 0.4)))


def _quality_score(
    rule: _CompiledRule,
    *,
    source_quality: float,
    binding: float,
    match_quality: float,
) -> float:
    base = _safe_float(rule.raw.get("base_confidence"), 0.8)
    target_factor = 0.45 + 0.55 * max(0.0, min(1.0, binding))
    score = base * source_quality * target_factor * match_quality
    return round(max(0.01, min(0.99, score)), 6)


def _evidence_record(
    *,
    quote: str,
    matched_text: str,
    page: int | None,
    section: str,
    source: str,
    page_text: str,
    start: int | None,
    end: int | None,
    binding: float,
    binding_method: str,
    extraction_method: str,
) -> dict[str, Any]:
    return {
        "quote": _clean(quote),
        "matched_text": _clean(matched_text),
        "page": page,
        "section": _clean(section) or "Unknown",
        "source": _clean(source) or "original_paper",
        "start": start,
        "end": end,
        "quote_sha256": _sha256_text(quote),
        "page_text_sha256": _sha256_text(page_text or quote),
        "target_binding": {
            "method": binding_method,
            "score": round(binding, 6),
        },
        "extraction_method": extraction_method,
    }


def _relation_record(
    rule: _CompiledRule,
    evidence: Mapping[str, Any],
    confidence: float,
    relation_override: str | None = None,
    conflict_type_override: str | None = None,
) -> dict[str, Any]:
    effect = rule.raw["effect"]
    relation = relation_override or str(effect["relation"])
    conflict_type = conflict_type_override or str(rule.raw.get("conflict_type") or "NONE")
    record_id = _sha256_text(
        f"{rule.raw['rule_id']}|{relation}|{evidence.get('page')}|{evidence.get('quote_sha256')}"
    )[:20]
    record = {
        "relation_id": record_id,
        "rule_id": rule.raw["rule_id"],
        "indicator": rule.raw["indicator"],
        "group": rule.raw.get("group"),
        "relation": relation,
        "decision_value": effect.get("value") if relation in {"SUPPORT", "CONFLICT"} else None,
        "conflict_type": conflict_type,
        "confidence": round(confidence, 6),
        "claim": {
            "en": rule.raw["claim"]["en"],
            "zh": rule.raw["claim"]["zh"],
            "predicate": rule.raw.get("predicate"),
            "temporal": rule.raw.get("temporal"),
        },
        "evidence": dict(evidence),
    }
    unless_support_groups = tuple(
        _clean(group) for group in rule.raw.get("unless_support_groups") or []
    )
    if unless_support_groups:
        record["defeasible"] = {
            "status": "ACTIVE" if relation == "CONFLICT" else "NOT_APPLICABLE",
            "unless_support_groups": list(unless_support_groups),
            "unless_support_policy": _clean(
                rule.raw.get("unless_support_policy") or "ANY"
            ).upper(),
            "matched_support_groups": [],
            "exception_relations": [],
        }
    return record


def _iter_retrieval_hits(retrieval: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    if not retrieval:
        return []
    direct = retrieval.get("hits")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return [hit for hit in direct if isinstance(hit, Mapping)]
    hits: list[Mapping[str, Any]] = []
    categories = retrieval.get("categories") or {}
    if isinstance(categories, Mapping):
        for members in categories.values():
            if isinstance(members, Sequence) and not isinstance(members, (str, bytes)):
                hits.extend(hit for hit in members if isinstance(hit, Mapping))
    return hits


def _normalize_pages(
    pages: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    if pages is None:
        return [], {}
    if isinstance(pages, Mapping):
        raw = pages.get("pages") or []
        source = pages.get("source") or {}
    else:
        raw = pages
        source = {}
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("pages must be a page-cache mapping or a sequence of page mappings")
    return [page for page in raw if isinstance(page, Mapping)], source if isinstance(source, Mapping) else {}


def _matches_from_hits(
    rule: _CompiledRule,
    hits: Sequence[Mapping[str, Any]],
    target: str,
    multi_target: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for hit in hits:
        quote = _clean(hit.get("quote") or hit.get("matched_text"))
        matched_text = _clean(hit.get("matched_text") or quote)
        if not quote:
            continue
        subtype = _clean(hit.get("subtype")).casefold()
        subtype_match = bool(subtype and subtype in rule.hit_subtypes)
        lexical_matches = [pattern.search(quote) for pattern in rule.patterns]
        lexical_match = next((match for match in lexical_matches if match), None)
        if not subtype_match and lexical_match is None:
            continue
        if lexical_match is not None:
            match_start, match_end = lexical_match.span()
            evidence_match = lexical_match.group(0)
        else:
            match_start = quote.casefold().find(matched_text.casefold())
            if match_start < 0:
                match_start, match_end = 0, len(quote)
            else:
                match_end = match_start + len(matched_text)
            evidence_match = matched_text
        section = _clean(hit.get("section"))
        if hit.get("reference_noise"):
            binding = _safe_float(hit.get("binding"), 0.25)
            evidence = _evidence_record(
                quote=quote,
                matched_text=matched_text,
                page=_safe_page(hit.get("page")),
                section=_clean(hit.get("section")) or "References",
                source=_clean(hit.get("source")),
                page_text=quote,
                start=_safe_page(hit.get("start")),
                end=_safe_page(hit.get("end")),
                binding=binding,
                binding_method="reference_or_citation_context",
                extraction_method="existing_retrieval_hit",
            )
            output.append(
                _relation_record(
                    rule,
                    evidence,
                    min(0.35, binding),
                    relation_override="UNKNOWN",
                    conflict_type_override="PROVENANCE",
                )
            )
            continue
        scope_ok, scope_conflict_type, scope_metadata, claim = _claim_scope_gate(
            rule,
            text=quote,
            start=match_start,
            end=match_end,
            target=target,
            section=section,
        )
        if scope_ok and _rule_excluded(
            rule, claim if rule.claim_scope == "same_sentence" else quote
        ):
            continue
        supplied_binding = hit.get("binding")
        if supplied_binding is None:
            supplied_binding = hit.get("target_relevance")
        if supplied_binding is None:
            binding, binding_method = _target_binding(target, quote, quote, multi_target)
        else:
            binding = max(0.0, min(1.0, _safe_float(supplied_binding, 0.5)))
            binding_method = _clean(hit.get("target_anchor")) or "retrieval_binding"
        evidence = _evidence_record(
            quote=quote,
            matched_text=evidence_match,
            page=_safe_page(hit.get("page")),
            section=section,
            source=_clean(hit.get("source")),
            page_text=quote,
            start=_safe_page(hit.get("start")),
            end=_safe_page(hit.get("end")),
            binding=binding,
            binding_method=binding_method,
            extraction_method="existing_retrieval_hit",
        )
        if _claim_policy_active(rule):
            evidence["claim_scope"] = scope_metadata
        if not scope_ok:
            output.append(
                _relation_record(
                    rule,
                    evidence,
                    round(min(0.65, max(0.01, binding)), 6),
                    relation_override="UNKNOWN",
                    conflict_type_override=scope_conflict_type,
                )
            )
            continue
        if binding < _minimum_binding(rule):
            output.append(
                _relation_record(
                    rule,
                    evidence,
                    round(binding, 6),
                    relation_override="UNKNOWN",
                    conflict_type_override="TARGET_SCOPE",
                )
            )
            continue
        confidence = _quality_score(
            rule,
            source_quality=0.94,
            binding=binding,
            match_quality=1.0 if subtype_match else 0.94,
        )
        output.append(_relation_record(rule, evidence, confidence))
    return output


def _matches_from_pages(
    rule: _CompiledRule,
    pages: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Any],
    target: str,
    multi_target: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    quality = _source_quality(source_metadata, bool(pages))
    for fallback, page in enumerate(pages, start=1):
        text = str(page.get("text") or "")
        if not text.strip():
            continue
        page_number = _safe_page(page.get("page"), fallback)
        section_hints = page.get("section_hints") or []
        section = _clean(section_hints[-1]) if section_hints else "Unknown"
        for pattern in rule.patterns:
            count = 0
            for match in pattern.finditer(text):
                context = _quote_window(text, match.start(), match.end())
                scope_ok, scope_conflict_type, scope_metadata, claim = _claim_scope_gate(
                    rule,
                    text=text,
                    start=match.start(),
                    end=match.end(),
                    target=target,
                    section=section,
                )
                if scope_ok and _rule_excluded(
                    rule, claim if rule.claim_scope == "same_sentence" else context
                ):
                    continue
                binding, binding_method = _target_binding(target, context, text, multi_target)
                evidence = _evidence_record(
                    quote=context,
                    matched_text=match.group(0),
                    page=page_number,
                    section=section,
                    source=_clean(page.get("source")) or _clean(source_metadata.get("source_kind")),
                    page_text=text,
                    start=match.start(),
                    end=match.end(),
                    binding=binding,
                    binding_method=binding_method,
                    extraction_method="knowledge_pattern_scan",
                )
                if _claim_policy_active(rule):
                    evidence["claim_scope"] = scope_metadata
                if not scope_ok:
                    output.append(
                        _relation_record(
                            rule,
                            evidence,
                            round(min(0.65, max(0.01, binding)), 6),
                            relation_override="UNKNOWN",
                            conflict_type_override=scope_conflict_type,
                        )
                    )
                elif binding < _minimum_binding(rule):
                    output.append(
                        _relation_record(
                            rule,
                            evidence,
                            round(binding, 6),
                            relation_override="UNKNOWN",
                            conflict_type_override="TARGET_SCOPE",
                        )
                    )
                else:
                    confidence = _quality_score(
                        rule,
                        source_quality=quality,
                        binding=binding,
                        match_quality=0.94,
                    )
                    output.append(_relation_record(rule, evidence, confidence))
                count += 1
                if count >= 5:
                    break
    return output


def _deduplicate_relations(relations: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str, Any, str], dict[str, Any]] = {}
    for relation in relations:
        evidence = relation.get("evidence") or {}
        key = (
            str(relation.get("rule_id")),
            str(relation.get("relation")),
            evidence.get("page"),
            str(evidence.get("quote_sha256") or _sha256_text(evidence.get("quote"))),
        )
        candidate = dict(relation)
        current = selected.get(key)
        if current is None or _safe_float(candidate.get("confidence"), 0.0) > _safe_float(
            current.get("confidence"), 0.0
        ):
            selected[key] = candidate
    return sorted(
        selected.values(),
        key=lambda relation: (
            {"CONFLICT": 0, "SUPPORT": 1, "UNKNOWN": 2}.get(str(relation.get("relation")), 3),
            -_safe_float(relation.get("confidence"), 0.0),
            str(relation.get("rule_id")),
            (relation.get("evidence") or {}).get("page") or 0,
        ),
    )


def _apply_defeasible_exceptions(
    relations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Downgrade a conflict when its declared, target-bound exception is supported.

    Exception rules are ordinary SUPPORT relations, so they pass the same source,
    claim-scope, and target-binding gates as every other item.  The resulting
    relation graph retains both directions of each defeat edge for auditability.
    """

    resolved = [dict(relation) for relation in relations]
    strongest_by_group: dict[str, dict[str, Any]] = {}
    for relation in resolved:
        if relation.get("relation") != "SUPPORT":
            continue
        group = _clean(relation.get("group"))
        if not group:
            continue
        current = strongest_by_group.get(group)
        if current is None or _safe_float(relation.get("confidence"), 0.0) > _safe_float(
            current.get("confidence"), 0.0
        ):
            strongest_by_group[group] = relation

    defeat_edges: dict[str, list[dict[str, str]]] = {}
    for relation in resolved:
        defeasible = relation.get("defeasible")
        if relation.get("relation") != "CONFLICT" or not isinstance(defeasible, Mapping):
            continue
        declared_groups = [
            _clean(group) for group in defeasible.get("unless_support_groups") or []
        ]
        matched_groups = [group for group in declared_groups if group in strongest_by_group]
        policy = _clean(defeasible.get("unless_support_policy") or "ANY").upper()
        exception_applies = bool(matched_groups) if policy == "ANY" else (
            bool(declared_groups) and len(matched_groups) == len(declared_groups)
        )
        if not exception_applies:
            continue

        exception_relations = [
            {
                "relation_id": str(strongest_by_group[group].get("relation_id")),
                "rule_id": str(strongest_by_group[group].get("rule_id")),
                "group": group,
                "page": (strongest_by_group[group].get("evidence") or {}).get("page"),
                "quote_sha256": str(
                    (strongest_by_group[group].get("evidence") or {}).get("quote_sha256")
                    or ""
                ),
            }
            for group in matched_groups
        ]
        relation["relation"] = "UNKNOWN"
        relation["decision_value"] = None
        relation["defeasible"] = {
            **dict(defeasible),
            "status": "DEFEATED",
            "resolution": "DOWNGRADED_TO_UNKNOWN",
            "original_relation": "CONFLICT",
            "original_decision_value": 0,
            "original_conflict_type": str(relation.get("conflict_type") or "NONE"),
            "matched_support_groups": matched_groups,
            "exception_relations": exception_relations,
        }
        for exception in exception_relations:
            defeat_edges.setdefault(exception["relation_id"], []).append(
                {
                    "relation_id": str(relation.get("relation_id")),
                    "rule_id": str(relation.get("rule_id")),
                }
            )

    for relation in resolved:
        edges = defeat_edges.get(str(relation.get("relation_id")))
        if edges:
            relation["defeasible_effects"] = {
                "effect": "DEFEATS",
                "relations": sorted(edges, key=lambda item: (item["rule_id"], item["relation_id"])),
            }
    return resolved


def _independent_relations(relations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Use the strongest item per rule so repeated excerpts are not independent votes."""

    best: dict[str, Mapping[str, Any]] = {}
    for relation in relations:
        if relation.get("relation") not in {"SUPPORT", "CONFLICT"}:
            continue
        rule_id = str(relation.get("rule_id"))
        current = best.get(rule_id)
        if current is None or _safe_float(relation.get("confidence"), 0.0) > _safe_float(
            current.get("confidence"), 0.0
        ):
            best[rule_id] = relation
    return list(best.values())


def _combine_masses(relations: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Combine simple support/conflict masses and retain unnormalized conflict."""

    support = 0.0
    conflict = 0.0
    unknown = 1.0
    contradiction = 0.0
    for relation in _independent_relations(relations):
        confidence = max(0.0, min(0.99, _safe_float(relation.get("confidence"), 0.0)))
        item_support = confidence if relation.get("relation") == "SUPPORT" else 0.0
        item_conflict = confidence if relation.get("relation") == "CONFLICT" else 0.0
        item_unknown = 1.0 - confidence
        next_support = (
            support * item_support
            + support * item_unknown
            + unknown * item_support
        )
        next_conflict = (
            conflict * item_conflict
            + conflict * item_unknown
            + unknown * item_conflict
        )
        next_unknown = unknown * item_unknown
        next_contradiction = contradiction + support * item_conflict + conflict * item_support
        support, conflict, unknown, contradiction = (
            next_support,
            next_conflict,
            next_unknown,
            next_contradiction,
        )
    values = [support, conflict, unknown, contradiction]
    total = sum(values)
    if total <= 0:
        values = [0.0, 0.0, 1.0, 0.0]
        total = 1.0
    return {
        "support": round(values[0] / total, 6),
        "conflict": round(values[1] / total, 6),
        "unknown": round(values[2] / total, 6),
        "contradiction": round(values[3] / total, 6),
    }


class ExpertConflictEngine:
    """Compare original-paper evidence with a validated expert knowledge base."""

    def __init__(self, knowledge_base: Mapping[str, Any], *, kb_sha256: str | None = None) -> None:
        self.knowledge_base = validate_knowledge_base(knowledge_base)
        canonical = json.dumps(
            self.knowledge_base,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.kb_sha256 = kb_sha256 or hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        self.rules = tuple(_compile_rule(rule) for rule in self.knowledge_base["rules"])

    @classmethod
    def from_path(cls, path: Path | str = DEFAULT_KNOWLEDGE_BASE) -> "ExpertConflictEngine":
        source = Path(path)
        raw = source.read_bytes()
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise KnowledgeBaseValidationError(f"invalid JSON in {source}: {exc}") from exc
        return cls(data, kb_sha256=hashlib.sha256(raw).hexdigest())

    def evaluate(
        self,
        indicator: str,
        target: str,
        *,
        retrieval: Mapping[str, Any] | None = None,
        pages: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
        multi_target: bool = False,
    ) -> dict[str, Any]:
        """Return a qualifier-aware knowledge relation for one target and indicator."""

        indicator_name = _clean(indicator).upper()
        if indicator_name not in INDICATORS:
            raise ValueError("indicator must be R3, R5, or R7")
        target_name = _clean(target)
        if not target_name:
            raise ValueError("target must be non-empty")
        retrieval_hits = _iter_retrieval_hits(retrieval)
        page_rows, source_metadata = _normalize_pages(pages)
        source_available = bool(
            any(_clean(hit.get("quote") or hit.get("matched_text")) for hit in retrieval_hits)
            or any(_clean(page.get("text")) for page in page_rows)
        )

        collected: list[dict[str, Any]] = []
        for rule in self.rules:
            if rule.raw["indicator"] != indicator_name:
                continue
            collected.extend(
                _matches_from_hits(rule, retrieval_hits, target_name, bool(multi_target))
            )
            collected.extend(
                _matches_from_pages(
                    rule,
                    page_rows,
                    source_metadata,
                    target_name,
                    bool(multi_target),
                )
            )
        relations = _apply_defeasible_exceptions(_deduplicate_relations(collected))
        independent = _independent_relations(relations)
        support_relations = [item for item in independent if item["relation"] == "SUPPORT"]
        conflict_relations = [item for item in independent if item["relation"] == "CONFLICT"]
        evidence_status = (
            "BOTH"
            if support_relations and conflict_relations
            else "SUPPORTED_ONLY"
            if support_relations
            else "REFUTED_ONLY"
            if conflict_relations
            else "NEITHER"
        )
        defeated_relations = [
            item
            for item in relations
            if (item.get("defeasible") or {}).get("status") == "DEFEATED"
        ]
        undecided_relations = [
            item
            for item in relations
            if item.get("relation") == "UNKNOWN"
            and (item.get("defeasible") or {}).get("status") != "DEFEATED"
        ]
        masses = _combine_masses(relations)
        policy = self.knowledge_base["indicators"][indicator_name]["decision_policy"]
        conflict_overrides_support = bool(policy.get("conflict_overrides_support", False))

        if not source_available or not independent:
            overall_relation = "UNKNOWN"
        elif conflict_relations and conflict_overrides_support:
            overall_relation = "CONFLICT"
        elif support_relations and conflict_relations:
            overall_relation = "MIXED"
        elif support_relations and masses["support"] >= 0.35:
            overall_relation = "SUPPORT"
        elif conflict_relations and masses["conflict"] >= 0.35:
            overall_relation = "CONFLICT"
        else:
            overall_relation = "UNKNOWN"

        confidence = (
            masses["support"]
            if overall_relation == "SUPPORT"
            else max(
                _safe_float(item.get("confidence"), 0.0) for item in conflict_relations
            )
            if overall_relation == "CONFLICT" and conflict_overrides_support
            else masses["conflict"]
            if overall_relation == "CONFLICT"
            else min(1.0, masses["contradiction"] + min(masses["support"], masses["conflict"]))
            if overall_relation == "MIXED"
            else masses["unknown"]
        )
        required_groups = set(policy.get("required_support_groups") or [])
        required_support_sets = [
            set(group_set) for group_set in policy.get("required_support_sets") or []
        ]
        support_groups = {str(item.get("group")) for item in support_relations}
        missing_groups = sorted(required_groups - support_groups)
        satisfied_support_set = next(
            (
                sorted(group_set)
                for group_set in required_support_sets
                if group_set.issubset(support_groups)
            ),
            None,
        )
        missing_support_sets = [
            {
                "required": sorted(group_set),
                "missing": sorted(group_set - support_groups),
            }
            for group_set in required_support_sets
        ]
        support_requirements_met = not missing_groups and (
            not required_support_sets or satisfied_support_set is not None
        )
        if overall_relation == "CONFLICT":
            recommended_value: int | None = 0
            outcome = "NEGATIVE"
            reason = (
                "An undefeated policy-priority conflict cannot be cancelled by generic support."
                if conflict_overrides_support and support_relations
                else "Target-bound original-paper evidence conflicts with the positive criterion."
            )
        elif overall_relation == "SUPPORT" and support_requirements_met:
            recommended_value = 1
            outcome = "POSITIVE"
            reason = "Target-bound original-paper evidence supports the expert criterion."
        elif overall_relation == "SUPPORT":
            recommended_value = None
            outcome = "UNKNOWN"
            reason = "Supporting evidence is present, but required knowledge groups are incomplete."
        elif overall_relation == "MIXED":
            recommended_value = None
            outcome = "UNKNOWN"
            reason = "Strong support and conflict evidence coexist and must not be collapsed."
        else:
            recommended_value = None
            outcome = "UNKNOWN"
            reason = (
                "Original-paper source is unavailable."
                if not source_available
                else "No sufficiently target-bound knowledge relation was established."
            )

        conflict_types = sorted(
            {
                str(item.get("conflict_type"))
                for item in relations
                if item.get("relation") == "CONFLICT" and item.get("conflict_type") != "NONE"
            }
        )
        if overall_relation == "MIXED":
            conflict_types.append("EVIDENCE_DISAGREEMENT")
            conflict_types = sorted(set(conflict_types))
        runtime_sources = ["expert_knowledge_base"]
        if retrieval_hits:
            runtime_sources.append("target_bound_retrieval_hits")
        if page_rows:
            runtime_sources.append("original_paper_pages")

        return {
            "schema_version": "1.0",
            "engine_version": ENGINE_VERSION,
            "kb_version": self.knowledge_base["kb_version"],
            "indicator": indicator_name,
            "target": target_name,
            "source_available": source_available,
            "relation": overall_relation,
            "evidence_status": evidence_status,
            "rule_status_summary": {
                "active_support": len(support_relations),
                "active_conflict": len(conflict_relations),
                "defeated": len(defeated_relations),
                "undecided": len(undecided_relations),
            },
            "conflict_types": conflict_types,
            "confidence": round(confidence, 6),
            "evidence_mass": masses,
            "decision": {
                "recommended_value": recommended_value,
                "policy_decision": (
                    recommended_value if recommended_value in {0, 1} else "ABSTAIN"
                ),
                "outcome": outcome,
                "reason": reason,
                "support_groups": sorted(support_groups),
                "required_support_groups": sorted(required_groups),
                "missing_support_groups": missing_groups,
                "required_support_sets": [
                    sorted(group_set) for group_set in required_support_sets
                ],
                "satisfied_support_set": satisfied_support_set,
                "missing_support_sets": missing_support_sets,
                "conflict_overrides_support": conflict_overrides_support,
            },
            "relations": relations,
            "provenance": {
                "kb_sha256": self.kb_sha256,
                "label_supervision": False,
                "runtime_sources": runtime_sources,
                "multi_target": bool(multi_target),
                "aggregation": (
                    "target-bound defeasible exceptions followed by unnormalized conjunctive "
                    "support/conflict masses with retained contradiction"
                ),
            },
        }


def compact_analysis(
    analysis: Mapping[str, Any],
    *,
    relation_limit: int = 6,
    quote_limit: int = 560,
) -> dict[str, Any]:
    """Return an Excel-safe conflict summary while retaining evidence lineage."""

    relations: list[dict[str, Any]] = []
    for source in list(analysis.get("relations") or [])[:relation_limit]:
        relation = dict(source)
        evidence = dict(relation.get("evidence") or {})
        quote = _clean(evidence.get("quote"))
        if len(quote) > quote_limit:
            evidence["quote"] = quote[: quote_limit - 3].rstrip() + "..."
        relation["evidence"] = evidence
        relations.append(relation)
    return {
        key: analysis.get(key)
        for key in (
            "schema_version",
            "engine_version",
            "kb_version",
            "indicator",
            "target",
            "source_available",
            "relation",
            "evidence_status",
            "rule_status_summary",
            "conflict_types",
            "confidence",
            "evidence_mass",
            "decision",
            "provenance",
        )
    } | {
        "relations": relations,
        "relations_retained": len(relations),
        "relations_total": len(analysis.get("relations") or []),
    }


def evaluate_conflicts(
    indicator: str,
    target: str,
    *,
    retrieval: Mapping[str, Any] | None = None,
    pages: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    multi_target: bool = False,
    knowledge_base: Path | str = DEFAULT_KNOWLEDGE_BASE,
) -> dict[str, Any]:
    """Convenience API for a one-off conflict evaluation."""

    return ExpertConflictEngine.from_path(knowledge_base).evaluate(
        indicator,
        target,
        retrieval=retrieval,
        pages=pages,
        multi_target=multi_target,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indicator", choices=sorted(INDICATORS), required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--knowledge-base", type=Path, default=DEFAULT_KNOWLEDGE_BASE)
    parser.add_argument("--multi-target", action="store_true")
    args = parser.parse_args()
    source = json.loads(args.input_json.read_text(encoding="utf-8"))
    retrieval = source.get("retrieval") if isinstance(source, Mapping) else None
    pages = source.get("pages") if isinstance(source, Mapping) else source
    result = evaluate_conflicts(
        args.indicator,
        args.target,
        retrieval=retrieval,
        pages=pages,
        multi_target=args.multi_target,
        knowledge_base=args.knowledge_base,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
