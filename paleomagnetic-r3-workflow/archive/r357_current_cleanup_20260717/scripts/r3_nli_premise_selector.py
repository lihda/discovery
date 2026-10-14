#!/usr/bin/env python3
"""Label-blind premise selection for single-target R3 NLI inference.

The selector accepts only original-paper pages, a frozen concept snapshot, and
one target identity.  It never opens labels, source tables, prior predictions,
or evaluation artifacts.  All ranking rules are fixed structural heuristics.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Pattern, Sequence

try:
    from .experiment_r3_own_study_events import classify_heading, event_signals
    from .r3_trace_discourse_roles import (
        DISCOURSE_ROLE_NAMES,
        annotate_discourse_window,
    )
except ImportError:
    from experiment_r3_own_study_events import classify_heading, event_signals
    from r3_trace_discourse_roles import (
        DISCOURSE_ROLE_NAMES,
        annotate_discourse_window,
    )


SCHEMA_VERSION = "r3_nli_premise_selection_v1"
PREMISE_GROUPS = (
    "METHOD_DIAGNOSTIC",
    "IDENTIFY",
    "SIGNIFICANCE_COUNTER",
)
DEFAULT_MAX_PREMISES_PER_GROUP = 2
DEFAULT_MAX_PREMISE_CHARACTERS = 700

_PAGE_FIELDS = frozenset(
    ("page", "text", "section_hints", "source", "text_sha256")
)
_CONCEPT_FIELDS = frozenset(
    ("concept_id", "category", "search_patterns", "source_refs")
)
_TARGET_FIELDS = frozenset(("target_id", "name", "aliases"))
_CONCEPT_CATEGORIES = frozenset(("method", "diagnostic", "carrier", "bridge"))
_FORBIDDEN_FIELD_RE = re.compile(
    r"(?:^|_)(?:r3|r5|r7|pmag|pmagresult|rockmag|labels?|ground_truth|"
    r"outcomes?|predictions?|oof|evaluation|metrics?)(?:$|_)",
    re.I,
)
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
_SPACE_RE = re.compile(r"\s+")
_ADVERSE_ROLES = frozenset(
    ("background_citation_prior_work", "negated", "prospective", "conditional")
)
_OWN_ROLES = frozenset(
    ("performed_execution", "reported_result", "interpretation")
)
_INCONCLUSIVE_CONCEPT_IDS = frozenset(("explicit_inconclusive_carrier_attempt",))


@dataclass(frozen=True)
class _FrozenConcept:
    concept_id: str
    category: str
    patterns: tuple[Pattern[str], ...]


@dataclass(frozen=True)
class _Paragraph:
    page: str | int
    page_index: int
    section: str
    start: int
    end: int
    text: str


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", _clean(value).casefold()).strip("_")


def _reject_forbidden_fields(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for raw_name, child in value.items():
            name = _field_name(raw_name)
            if _FORBIDDEN_FIELD_RE.search(name):
                raise ValueError(f"forbidden supervised field at {path}.{raw_name}")
            _reject_forbidden_fields(child, f"{path}.{raw_name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _target_names(target: str | Mapping[str, Any]) -> tuple[str, ...]:
    _reject_forbidden_fields(target, "target")
    if isinstance(target, str):
        names = (target,)
    elif isinstance(target, Mapping):
        extra = set(target) - _TARGET_FIELDS
        if extra:
            raise ValueError(f"unsupported target fields: {sorted(map(str, extra))}")
        aliases = target.get("aliases") or ()
        if isinstance(aliases, str) or not isinstance(aliases, Sequence):
            raise TypeError("target aliases must be a sequence")
        names = (_clean(target.get("name")), *(_clean(value) for value in aliases))
    else:
        raise TypeError("target must be a string or a narrow mapping")
    output = tuple(dict.fromkeys(name for name in names if name))
    if not output:
        raise ValueError("target must contain a nonempty name")
    return output


def _load_concepts(concepts: Sequence[Mapping[str, Any]]) -> tuple[_FrozenConcept, ...]:
    if isinstance(concepts, (str, bytes, bytearray)) or not isinstance(concepts, Sequence):
        raise TypeError("concepts must be a sequence of narrow mappings")
    _reject_forbidden_fields(concepts, "concepts")
    output: list[_FrozenConcept] = []
    seen: set[str] = set()
    for index, raw in enumerate(concepts):
        if not isinstance(raw, Mapping):
            raise TypeError(f"concepts[{index}] must be a mapping")
        extra = set(raw) - _CONCEPT_FIELDS
        if extra:
            raise ValueError(
                f"unsupported concept fields at concepts[{index}]: {sorted(map(str, extra))}"
            )
        concept_id = _clean(raw.get("concept_id"))
        category = _clean(raw.get("category")).casefold()
        expressions = raw.get("search_patterns") or ()
        if (
            not concept_id
            or concept_id in seen
            or category not in _CONCEPT_CATEGORIES
            or isinstance(expressions, (str, bytes, bytearray))
            or not isinstance(expressions, Sequence)
        ):
            raise ValueError(f"invalid frozen concept at concepts[{index}]")
        patterns = tuple(re.compile(_clean(value), re.I) for value in expressions if _clean(value))
        if not patterns:
            raise ValueError(f"concept {concept_id!r} has no search patterns")
        output.append(_FrozenConcept(concept_id, category, patterns))
        seen.add(concept_id)
    if not output:
        raise ValueError("at least one frozen concept is required")
    return tuple(output)


def _validated_pages(pages: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    if isinstance(pages, (str, bytes, bytearray)) or not isinstance(pages, Sequence):
        raise TypeError("pages must be a sequence of narrow mappings")
    _reject_forbidden_fields(pages, "pages")
    output = []
    for index, raw in enumerate(pages):
        if not isinstance(raw, Mapping):
            raise TypeError(f"pages[{index}] must be a mapping")
        extra = set(raw) - _PAGE_FIELDS
        if extra:
            raise ValueError(
                f"unsupported page fields at pages[{index}]: {sorted(map(str, extra))}"
            )
        text = raw.get("text")
        if not isinstance(text, str):
            raise TypeError(f"pages[{index}].text must be a string")
        hints = raw.get("section_hints") or ()
        if isinstance(hints, str) or not isinstance(hints, Sequence):
            raise TypeError(f"pages[{index}].section_hints must be a sequence")
        output.append(
            {
                "page": raw.get("page", index + 1),
                "text": text,
                "section_hints": tuple(_clean(value) for value in hints if _clean(value)),
            }
        )
    if not output:
        raise ValueError("at least one page is required")
    return tuple(output)


def _trim_span(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _paragraphs(
    pages: Sequence[Mapping[str, Any]],
) -> tuple[list[_Paragraph], dict[str, int]]:
    paragraphs: list[_Paragraph] = []
    current_section = "abstract"
    references_started = False
    reference_pages = 0
    reference_characters = 0
    for page_index, page in enumerate(pages):
        text = str(page["text"])
        if current_section == "abstract" and page_index > 0:
            current_section = "unknown"
        for hint in page["section_hints"]:
            detected = classify_heading(hint)
            if detected == "references":
                references_started = True
                break
            if detected and detected != "abstract":
                current_section = detected
        if references_started:
            reference_pages += 1
            reference_characters += len(text)
            continue

        segment_start = 0

        def add_segment(start: int, end: int, section: str) -> None:
            for boundary in _PARAGRAPH_BREAK_RE.finditer(text, start, end):
                paragraph_start, paragraph_end = _trim_span(text, start, boundary.start())
                if paragraph_start < paragraph_end:
                    paragraphs.append(
                        _Paragraph(
                            page["page"],
                            page_index,
                            section,
                            paragraph_start,
                            paragraph_end,
                            text[paragraph_start:paragraph_end],
                        )
                    )
                start = boundary.end()
            paragraph_start, paragraph_end = _trim_span(text, start, end)
            if paragraph_start < paragraph_end:
                paragraphs.append(
                    _Paragraph(
                        page["page"],
                        page_index,
                        section,
                        paragraph_start,
                        paragraph_end,
                        text[paragraph_start:paragraph_end],
                    )
                )

        for line in re.finditer(r"[^\n]*(?:\n|$)", text):
            if line.start() == line.end():
                continue
            detected = classify_heading(line.group(0).strip())
            if not detected:
                continue
            add_segment(segment_start, line.start(), current_section)
            if detected == "references":
                references_started = True
                reference_pages += 1
                reference_characters += len(text) - line.start()
                segment_start = len(text)
                break
            current_section = detected
            segment_start = line.end()
        if not references_started:
            add_segment(segment_start, len(text), current_section)
    return paragraphs, {
        "reference_pages_excluded": reference_pages,
        "reference_characters_excluded": reference_characters,
    }


def _concept_matches(
    text: str, concepts: Sequence[_FrozenConcept]
) -> tuple[set[str], set[str]]:
    concept_ids: set[str] = set()
    categories: set[str] = set()
    for concept in concepts:
        if any(pattern.search(text) for pattern in concept.patterns):
            concept_ids.add(concept.concept_id)
            categories.add(concept.category)
    return concept_ids, categories


def _target_present(text: str, target_names: Sequence[str]) -> bool:
    folded = _SPACE_RE.sub(" ", text).casefold()
    return any(_SPACE_RE.sub(" ", name).casefold() in folded for name in target_names)


def _bounded_sentence_window(
    paragraph: _Paragraph,
    annotations: Sequence[Any],
    focal_index: int,
    max_characters: int,
    focal_anchor: int,
) -> tuple[int, int, bool]:
    left = max(0, focal_index - 1)
    right = min(len(annotations) - 1, focal_index + 1)
    start = int(annotations[left].start)
    end = int(annotations[right].end)
    if end - start <= max_characters:
        return start, end, False

    focal_start = int(annotations[focal_index].start)
    focal_end = int(annotations[focal_index].end)
    if focal_end - focal_start >= max_characters:
        start = max(focal_start, focal_anchor - max_characters // 2)
        start = min(start, focal_end - max_characters)
        return start, start + max_characters, True

    remaining = max_characters - (focal_end - focal_start)
    before = min(focal_start, remaining // 2)
    after = min(len(paragraph.text) - focal_end, remaining - before)
    before = min(focal_start, remaining - after)
    return focal_start - before, focal_end + after, True


def _groups_for_candidate(
    concept_ids: set[str], categories: set[str], roles: set[str], signals: Mapping[str, Any]
) -> tuple[str, ...]:
    groups = []
    if categories & {"method", "diagnostic"}:
        groups.append("METHOD_DIAGNOSTIC")
    if "carrier" in categories or concept_ids & _INCONCLUSIVE_CONCEPT_IDS:
        groups.append("IDENTIFY")
    if (
        "bridge" in categories
        or roles & _ADVERSE_ROLES
        or signals.get("external")
        or signals.get("future")
        or signals.get("method_negation")
    ):
        groups.append("SIGNIFICANCE_COUNTER")
    return tuple(groups)


def _evidence_mode(roles: set[str], signals: Mapping[str, Any]) -> str:
    adverse = bool(
        roles & _ADVERSE_ROLES
        or signals.get("external")
        or signals.get("future")
        or signals.get("method_negation")
    )
    own = bool(roles & _OWN_ROLES or signals.get("own_likely"))
    if adverse and own:
        return "mixed"
    if adverse:
        return "counter"
    if own:
        return "support"
    return "context"


def _base_score(
    *,
    concepts: set[str],
    categories: set[str],
    roles: set[str],
    section: str,
    signals: Mapping[str, Any],
    mode: str,
) -> float:
    score = 1.25 * len(concepts) + 0.65 * len(categories)
    score += 4.0 * int(bool(signals.get("own_strong")))
    score += 1.75 * int(bool(signals.get("own_likely")))
    score += 1.5 * int(bool(signals.get("first_person")))
    score += 1.5 * int(bool(signals.get("passive_execution")))
    score += 0.9 * int(bool(signals.get("procedure")))
    score += 1.3 * int(bool(signals.get("result_verb")))
    score += 0.6 * int(bool(signals.get("quantitative")))
    score += 0.35 * int(bool(signals.get("target")))
    score += 0.55 * int(section in {"methods", "results", "discussion", "conclusion"})
    score += 1.8 * int("performed_execution" in roles)
    score += 1.8 * int("reported_result" in roles)
    score += 1.5 * int("interpretation" in roles)
    score += 0.35 * int("cause" in roles)
    score -= 2.8 * int(bool(signals.get("external")))
    score -= 1.3 * int(bool(signals.get("citation")) and not signals.get("own_strong"))
    score -= 2.4 * int(bool(signals.get("future")))
    score -= 2.7 * int(bool(signals.get("method_negation")))
    score -= 1.7 * int("background_citation_prior_work" in roles)
    score -= 1.8 * int("prospective" in roles)
    score -= 0.8 * int("conditional" in roles)
    score -= 0.7 * int("negated" in roles)
    if mode == "counter":
        score -= 0.5
    return score


def _candidate_group_score(candidate: Mapping[str, Any], group: str) -> float:
    categories = set(candidate["categories"])
    concept_ids = set(candidate["concepts"])
    fit = 0.0
    if group == "METHOD_DIAGNOSTIC":
        fit = 1.4 * len(categories & {"method", "diagnostic"})
    elif group == "IDENTIFY":
        fit = 2.2 * int("carrier" in categories) + 1.4 * int(
            bool(concept_ids & _INCONCLUSIVE_CONCEPT_IDS)
        )
    elif group == "SIGNIFICANCE_COUNTER":
        fit = 2.2 * int("bridge" in categories) + 1.1 * int(
            candidate["evidence_mode"] in {"counter", "mixed"}
        )
    return float(candidate["base_score"]) + fit


def _token_jaccard(left: str, right: str) -> float:
    left_tokens = set(token.casefold() for token in _TOKEN_RE.findall(left))
    right_tokens = set(token.casefold() for token in _TOKEN_RE.findall(right))
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def _build_candidates(
    pages: Sequence[Mapping[str, Any]],
    paragraphs: Sequence[_Paragraph],
    concepts: Sequence[_FrozenConcept],
    target_names: Sequence[str],
    max_characters: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates_by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    focal_sentences = 0
    capped_windows = 0
    sentence_count = 0
    for paragraph in paragraphs:
        annotations = annotate_discourse_window(paragraph.text)
        sentence_count += len(annotations)
        for focal_index, annotation in enumerate(annotations):
            focal_concepts, _ = _concept_matches(annotation.text, concepts)
            if not focal_concepts:
                continue
            focal_sentences += 1
            focal_match_starts = [
                match.start()
                for concept in concepts
                if concept.concept_id in focal_concepts
                for pattern in concept.patterns
                for match in pattern.finditer(annotation.text)
            ]
            focal_anchor = int(annotation.start) + min(focal_match_starts)
            relative_start, relative_end, capped = _bounded_sentence_window(
                paragraph,
                annotations,
                focal_index,
                max_characters,
                focal_anchor,
            )
            capped_windows += int(capped)
            relative_start, relative_end = _trim_span(
                paragraph.text, relative_start, relative_end
            )
            raw_text = paragraph.text[relative_start:relative_end]
            premise_text = _SPACE_RE.sub(" ", raw_text).strip()
            if not premise_text:
                continue
            premise_concepts, categories = _concept_matches(premise_text, concepts)
            premise_annotations = annotate_discourse_window(premise_text)
            roles = {
                role
                for item in premise_annotations
                for role in item.roles
            }
            signals = event_signals(
                premise_text,
                paragraph.section,
                _target_present(premise_text, target_names),
            )
            groups = _groups_for_candidate(premise_concepts, categories, roles, signals)
            if not groups:
                continue
            mode = _evidence_mode(roles, signals)
            absolute_start = paragraph.start + relative_start
            absolute_end = paragraph.start + relative_end
            key = (paragraph.page_index, absolute_start, absolute_end)
            candidate = {
                "page": paragraph.page,
                "page_index": paragraph.page_index,
                "section": paragraph.section,
                "offset_start": absolute_start,
                "offset_end": absolute_end,
                "text": premise_text,
                "text_sha256": hashlib.sha256(premise_text.encode("utf-8")).hexdigest(),
                "concepts": sorted(premise_concepts),
                "categories": sorted(categories),
                "roles": [role for role in DISCOURSE_ROLE_NAMES if role in roles],
                "eligible_groups": list(groups),
                "evidence_mode": mode,
                "base_score": _base_score(
                    concepts=premise_concepts,
                    categories=categories,
                    roles=roles,
                    section=paragraph.section,
                    signals=signals,
                    mode=mode,
                ),
            }
            prior = candidates_by_key.get(key)
            if prior is None or candidate["base_score"] > prior["base_score"]:
                candidates_by_key[key] = candidate
    return list(candidates_by_key.values()), {
        "sentence_count": sentence_count,
        "concept_focal_sentence_count": focal_sentences,
        "capped_candidate_window_count": capped_windows,
    }


def _select_candidates(
    candidates: Sequence[Mapping[str, Any]], max_per_group: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    selected_concepts: set[str] = set()
    selected_categories: set[str] = set()
    selected_pages: set[int] = set()
    candidate_counts = Counter(
        group for candidate in candidates for group in candidate["eligible_groups"]
    )
    duplicate_rejections = 0
    for _round in range(max_per_group):
        for group in PREMISE_GROUPS:
            pool = [
                candidate
                for candidate in candidates
                if group in candidate["eligible_groups"]
                and candidate["text_sha256"] not in selected_hashes
            ]
            duplicate_rejections += sum(
                group in candidate["eligible_groups"]
                and candidate["text_sha256"] in selected_hashes
                for candidate in candidates
            )
            if not pool:
                continue

            def rank(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
                similarity = max(
                    (_token_jaccard(candidate["text"], item["text"]) for item in selected),
                    default=0.0,
                )
                novelty = 1.1 * len(set(candidate["concepts"]) - selected_concepts)
                novelty += 0.45 * len(set(candidate["categories"]) - selected_categories)
                novelty += 0.3 * int(candidate["page_index"] not in selected_pages)
                mmr_score = _candidate_group_score(candidate, group) + novelty - 3.0 * similarity
                return (
                    mmr_score,
                    _candidate_group_score(candidate, group),
                    -similarity,
                    -int(candidate["page_index"]),
                    -int(candidate["offset_start"]),
                    candidate["text_sha256"],
                )

            winner = max(pool, key=rank)
            result = {
                name: winner[name]
                for name in (
                    "page",
                    "section",
                    "offset_start",
                    "offset_end",
                    "text",
                    "text_sha256",
                    "concepts",
                    "roles",
                    "evidence_mode",
                )
            }
            result["group"] = group
            result["score"] = round(_candidate_group_score(winner, group), 6)
            selected.append(result)
            selected_hashes.add(winner["text_sha256"])
            selected_concepts.update(winner["concepts"])
            selected_categories.update(winner["categories"])
            selected_pages.add(int(winner["page_index"]))
    return selected, {
        "duplicate_candidate_rejections": duplicate_rejections,
        **{
            f"candidate_{group.lower()}": int(candidate_counts[group])
            for group in PREMISE_GROUPS
        },
    }


def select_nli_premises(
    pages: Sequence[Mapping[str, Any]],
    concepts: Sequence[Mapping[str, Any]],
    target: str | Mapping[str, Any],
    *,
    max_premises_per_group: int = DEFAULT_MAX_PREMISES_PER_GROUP,
    max_premise_characters: int = DEFAULT_MAX_PREMISE_CHARACTERS,
) -> dict[str, Any]:
    """Return at most six auditable NLI premises for one single-target paper."""

    if isinstance(max_premises_per_group, bool) or not isinstance(
        max_premises_per_group, int
    ):
        raise TypeError("max_premises_per_group must be an integer")
    if max_premises_per_group < 1 or max_premises_per_group > 2:
        raise ValueError("max_premises_per_group must be between one and two")
    if isinstance(max_premise_characters, bool) or not isinstance(
        max_premise_characters, int
    ):
        raise TypeError("max_premise_characters must be an integer")
    if max_premise_characters < 160 or max_premise_characters > 700:
        raise ValueError("max_premise_characters must be between 160 and 700")

    clean_pages = _validated_pages(pages)
    clean_concepts = _load_concepts(concepts)
    names = _target_names(target)
    paragraphs, reference_audit = _paragraphs(clean_pages)
    candidates, candidate_audit = _build_candidates(
        clean_pages,
        paragraphs,
        clean_concepts,
        names,
        max_premise_characters,
    )
    selected, selection_audit = _select_candidates(candidates, max_premises_per_group)
    selected_counts = Counter(item["group"] for item in selected)
    unresolved = [group for group in PREMISE_GROUPS if not selected_counts[group]]
    source_digest = hashlib.sha256()
    for page in clean_pages:
        source_digest.update(_clean(page["page"]).encode("utf-8"))
        source_digest.update(b"\0")
        source_digest.update(str(page["text"]).encode("utf-8"))
        source_digest.update(b"\0")
    audit = {
        "schema_version": SCHEMA_VERSION + "_audit",
        "page_count": len(clean_pages),
        "source_text_sha256": source_digest.hexdigest(),
        "paragraph_count": len(paragraphs),
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_support_count": sum(
            item["evidence_mode"] == "support" for item in selected
        ),
        "selected_counter_or_mixed_count": sum(
            item["evidence_mode"] in {"counter", "mixed"} for item in selected
        ),
        "selected_group_counts": {
            group: int(selected_counts[group]) for group in PREMISE_GROUPS
        },
        "unresolved_groups": unresolved,
        "complete_group_coverage": not unresolved,
        "max_premises_per_group": max_premises_per_group,
        "max_premise_characters": max_premise_characters,
        "selected_character_count": sum(len(item["text"]) for item in selected),
        "selected_concepts": sorted(
            {concept for item in selected for concept in item["concepts"]}
        ),
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
        },
        **reference_audit,
        **candidate_audit,
        **selection_audit,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "premises": selected,
        "audit": audit,
    }


__all__ = [
    "DEFAULT_MAX_PREMISE_CHARACTERS",
    "DEFAULT_MAX_PREMISES_PER_GROUP",
    "PREMISE_GROUPS",
    "SCHEMA_VERSION",
    "select_nli_premises",
]
