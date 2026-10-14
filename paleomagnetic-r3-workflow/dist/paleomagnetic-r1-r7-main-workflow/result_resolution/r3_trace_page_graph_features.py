#!/usr/bin/env python3
"""TRACE-inspired label-free page-graph evidence-chain features for R3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .paper_text_retrieval import canonical_target_identity, fold_text_with_position_map
    from .r3_paleomag_scope_features import (
        anchor_positions,
        resolve_document_target_anchors,
        target_profile,
    )
    from .r3_trace_discourse_roles import discourse_role_counts, discourse_role_features
    from .r3_trace_coreference_events import analyze_coreference_events
except ImportError:
    from paper_text_retrieval import canonical_target_identity, fold_text_with_position_map
    from r3_paleomag_scope_features import (
        anchor_positions,
        resolve_document_target_anchors,
        target_profile,
    )
    from r3_trace_discourse_roles import discourse_role_counts, discourse_role_features
    from r3_trace_coreference_events import analyze_coreference_events


QUERY_CATEGORIES = ("method", "diagnostic", "carrier", "bridge")
SEMANTIC_EDGE_THRESHOLD = 0.22
MAX_PHYSICAL_HOPS = 2
TOP_PAGES_PER_QUERY = 5
CONTEXT_PAGES_PER_QUERY = 3
CONTEXT_WINDOWS_PER_PAGE = 2
CONTEXT_WINDOW_RADIUS = 520
MAX_TARGET_CONTEXT_CHARACTERS = 20_000
REFERENCE_RE = re.compile(
    r"(?im)^\s*(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography|literature cited|works cited)\s*:?[ \t]*$"
)

# ROCKNAME often encodes the row-level paleomagnetic entity as a material plus
# a polarity, component, group, or directional qualifier.  These cues are
# scientific identity descriptors, not labels, and are needed to avoid giving
# N/R or A/B rows from the same unit identical evidence assignments.
QUALIFIER_SPECS: tuple[
    tuple[str, re.Pattern[str], re.Pattern[str]], ...
] = (
    (
        "normal_polarity",
        re.compile(r"(?:\bnormal(?:\s+polarity)?|\(\s*N\s*\)|\bN\s+(?:directions?|components?|comp\.?))", re.I),
        re.compile(r"(?:\bnormal(?:\s+polarity|\s+component|\s+directions?)?|\bN[- ]polarity\b|\bN\s+(?:directions?|components?|comp\.?))", re.I),
    ),
    (
        "reverse_polarity",
        re.compile(r"(?:\brevers(?:e|ed)(?:\s+polarity)?|\(\s*R\s*\)|\bR\s+(?:directions?|components?|comp\.?))", re.I),
        re.compile(r"(?:\brevers(?:e|ed)(?:\s+polarity|\s+component|\s+directions?)?|\bR[- ]polarity\b|\bR\s+(?:directions?|components?|comp\.?))", re.I),
    ),
    (
        "component_a",
        re.compile(r"(?:\bA\s+(?:components?|comp\.?|magneti[sz]ation)|\b(?:components?|comp\.?|group|locality|section)\s*A\b)", re.I),
        re.compile(r"(?:\bA\s+(?:components?|comp\.?|magneti[sz]ation)|\b(?:components?|comp\.?|group|locality|section)\s*A\b)", re.I),
    ),
    (
        "component_b",
        re.compile(r"(?:\bB\s+(?:components?|comp\.?|magneti[sz]ation)|\b(?:components?|comp\.?|group|locality|section)\s*B\b)", re.I),
        re.compile(r"(?:\bB\s+(?:components?|comp\.?|magneti[sz]ation)|\b(?:components?|comp\.?|group|locality|section)\s*B\b)", re.I),
    ),
    (
        "component_c",
        re.compile(r"(?:\bC\s+(?:components?|comp\.?)|\b(?:components?|comp\.?|group)\s*C\b)", re.I),
        re.compile(r"(?:\bC\s+(?:components?|comp\.?)|\b(?:components?|comp\.?|group)\s*C\b)", re.I),
    ),
    (
        "component_d_or_e",
        re.compile(r"(?:\b[DE]\s+(?:components?|comp\.?)|\b(?:components?|comp\.?|group)\s*[DE]\b)", re.I),
        re.compile(r"(?:\b[DE]\s+(?:components?|comp\.?)|\b(?:components?|comp\.?|group)\s*[DE]\b)", re.I),
    ),
    (
        "medium_temperature_component",
        re.compile(r"\bMT\s+(?:components?|comp\.?)", re.I),
        re.compile(r"(?:\bMT\s+(?:components?|comp\.?)|medium[- ]temperature\s+component)", re.I),
    ),
    (
        "high_temperature_component",
        re.compile(r"(?:\bHTC\b|high[- ]temperature\s+component)", re.I),
        re.compile(r"(?:\bHTC\b|high[- ]temperature\s+component)", re.I),
    ),
    (
        "low_temperature_component",
        re.compile(r"(?:\bLTC\b|low[- ]temperature\s+component)", re.I),
        re.compile(r"(?:\bLTC\b|low[- ]temperature\s+component)", re.I),
    ),
    (
        "west_direction",
        re.compile(r"(?:\bwest(?:ern)?\s+directions?|\bW\s+directions?)", re.I),
        re.compile(r"(?:\bwest(?:ern)?\s+directions?|\bW\s+directions?)", re.I),
    ),
    (
        "directional_trend",
        re.compile(r"\b(?:N|S|E|W|NE|NW|SE|SW)(?:[- ](?:trending|directions?))\b", re.I),
        re.compile(r"\b(?:N|S|E|W|NE|NW|SE|SW)(?:[- ](?:trending|directions?))\b", re.I),
    ),
    (
        "upper_unit",
        re.compile(r"\bupper\s+(?:part|zone|unit|section)\b", re.I),
        re.compile(r"\bupper\s+(?:part|zone|unit|section)\b", re.I),
    ),
    (
        "lower_unit",
        re.compile(r"\blower\s+(?:part|zone|unit|section)\b", re.I),
        re.compile(r"\blower\s+(?:part|zone|unit|section)\b", re.I),
    ),
    (
        "combined_result",
        re.compile(r"\bcombined\b", re.I),
        re.compile(r"\b(?:combined|overall|grand\s+mean|mean\s+of\s+all)\b", re.I),
    ),
)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _fold(value: Any) -> str:
    # Keep the same public normalization contract as target grouping.
    return canonical_target_identity(clean(value))


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@dataclass(frozen=True)
class Concept:
    concept_id: str
    category: str
    patterns: tuple[re.Pattern[str], ...]


@dataclass(frozen=True)
class QualifierProfile:
    families: tuple[str, ...]
    patterns: tuple[re.Pattern[str], ...]


def qualifier_profile(value: Any) -> QualifierProfile:
    raw = clean(value)
    matched = [
        (name, page_pattern)
        for name, identity_pattern, page_pattern in QUALIFIER_SPECS
        if identity_pattern.search(raw)
    ]
    return QualifierProfile(
        tuple(name for name, _ in matched),
        tuple(pattern for _, pattern in matched),
    )


@dataclass
class PageNode:
    index: int
    number: int
    text: str
    folded: str
    folded_to_original: tuple[int, ...]
    section_hints: tuple[str, ...]
    reference_noise: bool
    concept_counts: dict[str, int]
    concept_positions: dict[str, tuple[int, ...]]
    concepts: set[str]
    category_counts: dict[str, int]


def load_trace_concepts(path: Path) -> tuple[Concept, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    concepts = []
    concept_ids: set[str] = set()
    for item in payload.get("concepts") or []:
        category = clean(item.get("category")).casefold()
        if category not in QUERY_CATEGORIES:
            continue
        concept_id = clean(item.get("concept_id"))
        if not concept_id:
            raise ValueError("TRACE concept_id must be nonempty")
        if concept_id in concept_ids:
            raise ValueError(f"duplicate TRACE concept_id: {concept_id}")
        patterns = tuple(
            re.compile(str(expression), re.I)
            for expression in item.get("search_patterns") or []
            if clean(expression)
        )
        if patterns:
            concepts.append(Concept(concept_id, category, patterns))
            concept_ids.add(concept_id)
    if not concepts:
        raise ValueError("TRACE graph requires nonempty frozen R3 concepts")
    categories = {concept.category for concept in concepts}
    missing = set(QUERY_CATEGORIES) - categories
    if missing:
        raise ValueError(f"TRACE graph lacks frozen query categories: {sorted(missing)}")
    return tuple(concepts)


def _prepare_pages(
    cache_row: Mapping[str, Any], concepts: Sequence[Concept]
) -> list[PageNode]:
    pages = []
    references_started = False
    for offset, raw_page in enumerate(cache_row.get("pages") or [], start=1):
        if not isinstance(raw_page, Mapping):
            continue
        text = clean(raw_page.get("text"))
        heading = REFERENCE_RE.search(text)
        reference_noise = references_started
        if heading:
            text = text[: heading.start()]
            references_started = True
        hints = tuple(clean(item) for item in raw_page.get("section_hints") or [] if clean(item))
        if any(_fold(item) in {"references", "bibliography", "literature cited", "works cited"} for item in hints):
            reference_noise = True
            references_started = True
        concept_counts: dict[str, int] = {}
        concept_positions: dict[str, tuple[int, ...]] = {}
        category_counts = {category: 0 for category in QUERY_CATEGORIES}
        for concept in concepts:
            positions = tuple(
                sorted(
                    {
                        (match.start() + match.end()) // 2
                        for pattern in concept.patterns
                        for match in pattern.finditer(text)
                    }
                )
            )
            count = len(positions)
            if count:
                concept_counts[concept.concept_id] = count
                concept_positions[concept.concept_id] = positions
                category_counts[concept.category] += count
        folded, folded_to_original = fold_text_with_position_map(text)
        pages.append(
            PageNode(
                index=len(pages),
                number=int(raw_page.get("page") or offset),
                text=text,
                folded=folded,
                folded_to_original=folded_to_original,
                section_hints=hints,
                reference_noise=reference_noise,
                concept_counts=concept_counts,
                concept_positions=concept_positions,
                concepts=set(concept_counts),
                category_counts=category_counts,
            )
        )
    return pages


def _weighted_semantic_similarity(
    left: PageNode,
    right: PageNode,
    idf: Mapping[str, float],
) -> float:
    shared = left.concepts & right.concepts
    if not shared:
        return 0.0
    numerator = sum(idf[concept] ** 2 for concept in shared)
    left_norm = math.sqrt(sum(idf[concept] ** 2 for concept in left.concepts))
    right_norm = math.sqrt(sum(idf[concept] ** 2 for concept in right.concepts))
    return numerator / max(1e-12, left_norm * right_norm)


def build_page_graph(
    pages: Sequence[PageNode],
) -> tuple[dict[int, set[int]], dict[tuple[int, int], float], dict[str, float]]:
    physical: dict[int, set[int]] = {page.index: set() for page in pages}
    for left, right in zip(pages, pages[1:]):
        if right.number - left.number <= 1:
            physical[left.index].add(right.index)
            physical[right.index].add(left.index)
    document_frequency: dict[str, int] = defaultdict(int)
    for page in pages:
        for concept in page.concepts:
            document_frequency[concept] += 1
    idf = {
        concept: math.log((len(pages) + 1.0) / (frequency + 0.5)) + 1.0
        for concept, frequency in document_frequency.items()
    }
    semantic: dict[tuple[int, int], float] = {}
    for left, right in itertools.combinations(pages, 2):
        if right.index in physical[left.index]:
            continue
        similarity = _weighted_semantic_similarity(left, right, idf)
        if similarity >= SEMANTIC_EDGE_THRESHOLD:
            semantic[(left.index, right.index)] = similarity
            semantic[(right.index, left.index)] = similarity
    return physical, semantic, idf


def _anchor_state_positions(
    page: PageNode,
    profile: Any,
    anchor_state: Mapping[str, Any] | None,
) -> tuple[list[int], list[int], list[int]]:
    if anchor_state is None:
        exact, clustered = anchor_positions(page.folded, profile)
        return exact, clustered, []
    return (
        list(anchor_state.get("exact") or []),
        list(anchor_state.get("cluster") or []),
        list(anchor_state.get("fuzzy") or []),
    )


def _direct_target_relevance(
    page: PageNode,
    profile: Any,
    anchor_state: Mapping[str, Any] | None,
) -> tuple[float, int, int, int]:
    exact, clustered, fuzzy = _anchor_state_positions(page, profile, anchor_state)
    if exact:
        return min(1.0, 0.82 + 0.04 * len(exact)), len(exact), len(clustered), 0
    if clustered:
        return min(0.78, 0.58 + 0.035 * len(clustered)), 0, len(clustered), 0
    if fuzzy:
        fuzzy_score = float((anchor_state or {}).get("fuzzy_score") or 0.0)
        return min(0.74, 0.40 + 0.34 * fuzzy_score), 0, 0, len(fuzzy)
    return 0.0, 0, 0, 0


def _target_relevance(
    pages: Sequence[PageNode],
    profile: Any,
    anchor_states: Mapping[int, Mapping[str, Any]],
    physical: Mapping[int, set[int]],
    semantic: Mapping[tuple[int, int], float],
) -> tuple[np.ndarray, dict[str, float]]:
    relevance = np.zeros(len(pages), dtype=np.float64)
    exact_total = cluster_total = fuzzy_total = ambiguous_pages = 0
    fuzzy_scores: list[float] = []
    seeds = []
    for page in pages:
        state = anchor_states.get(page.index)
        value, exact, clustered, fuzzy = _direct_target_relevance(
            page, profile, state
        )
        relevance[page.index] = value
        exact_total += exact
        cluster_total += clustered
        fuzzy_total += fuzzy
        if state and state.get("ambiguous"):
            ambiguous_pages += 1
        if state and state.get("fuzzy_score"):
            fuzzy_scores.append(float(state["fuzzy_score"]))
        if value > 0:
            seeds.append(page.index)
    frontier = {(seed, 0) for seed in seeds}
    visited = set(frontier)
    while frontier:
        node, distance = frontier.pop()
        if distance >= MAX_PHYSICAL_HOPS:
            continue
        for neighbor in physical[node]:
            state = (neighbor, distance + 1)
            relevance[neighbor] = max(
                relevance[neighbor], relevance[node] * (0.70 if distance == 0 else 0.62)
            )
            if state not in visited:
                visited.add(state)
                frontier.add(state)
    for seed in seeds:
        for page in pages:
            similarity = semantic.get((seed, page.index), 0.0)
            if similarity:
                relevance[page.index] = max(
                    relevance[page.index], relevance[seed] * similarity * 0.86
                )
    return relevance, {
        "anchor_exact_count_log": _log1p(exact_total),
        "anchor_cluster_count_log": _log1p(cluster_total),
        "anchor_fuzzy_count_log": _log1p(fuzzy_total),
        "anchor_fuzzy_page_count_log": _log1p(len(fuzzy_scores)),
        "anchor_fuzzy_score_max": max(fuzzy_scores, default=0.0),
        "anchor_fuzzy_score_mean": _mean(fuzzy_scores),
        "anchor_fuzzy_ambiguous_page_count_log": _log1p(ambiguous_pages),
        "anchor_page_count_log": _log1p(len(seeds)),
        "propagated_page_count_log": _log1p(int(np.sum((relevance > 0) & (relevance < 0.58)))),
        "target_relevance_max": float(relevance.max(initial=0.0)),
        "target_relevance_mean": float(relevance.mean()) if len(relevance) else 0.0,
    }


def _qualifier_page_strength(
    pages: Sequence[PageNode], profile: QualifierProfile
) -> np.ndarray:
    strength = np.zeros(len(pages), dtype=np.float64)
    if not profile.patterns:
        return strength
    for page in pages:
        counts = [len(pattern.findall(page.text)) for pattern in profile.patterns]
        if any(counts):
            strength[page.index] = min(1.0, math.log1p(sum(counts)) / math.log(4.0))
    return strength


def _qualifier_features(
    pages: Sequence[PageNode],
    profile: QualifierProfile,
    target_relevance: np.ndarray,
    competitor_profiles: Sequence[QualifierProfile],
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    own = _qualifier_page_strength(pages, profile)
    competitor = np.zeros(len(pages), dtype=np.float64)
    for sibling in competitor_profiles:
        competitor = np.maximum(competitor, _qualifier_page_strength(pages, sibling))
    own_pages = own > 0
    evidence_pages = np.asarray(
        [sum(page.category_counts.values()) > 0 for page in pages], dtype=bool
    )
    anchored_pages = target_relevance > 0
    exclusive_pages = own_pages & (competitor <= 0)
    values = {
        "identity_qualifier_present": float(bool(profile.families)),
        "identity_qualifier_family_count_log": _log1p(len(profile.families)),
        "qualifier_page_count_log": _log1p(int(own_pages.sum())),
        "qualifier_anchor_page_count_log": _log1p(int((own_pages & anchored_pages).sum())),
        "qualifier_evidence_page_count_log": _log1p(int((own_pages & evidence_pages).sum())),
        "qualifier_exclusive_page_count_log": _log1p(int(exclusive_pages.sum())),
        "qualifier_competitor_page_count_log": _log1p(int((competitor > 0).sum())),
        "qualifier_competitor_overlap_ratio": (
            float(np.mean(competitor[own_pages] > 0)) if own_pages.any() else 0.0
        ),
        "qualifier_strength_max": float(own.max(initial=0.0)),
        "qualifier_strength_mean": float(own[own_pages].mean()) if own_pages.any() else 0.0,
        "qualifier_target_relevance_mean": (
            float(target_relevance[own_pages].mean()) if own_pages.any() else 0.0
        ),
    }
    for category in QUERY_CATEGORIES:
        category_pages = np.asarray(
            [page.category_counts.get(category, 0) > 0 for page in pages], dtype=bool
        )
        values[f"qualifier_{category}_page_count_log"] = _log1p(
            int((own_pages & category_pages).sum())
        )
        values[f"qualifier_{category}_exclusive_page_count_log"] = _log1p(
            int((exclusive_pages & category_pages).sum())
        )
    return values, own, competitor


def _entity_positions(
    page: PageNode,
    profile: Any,
    qualifier: QualifierProfile,
    anchor_state: Mapping[str, Any] | None,
) -> tuple[int, ...]:
    exact, clustered, fuzzy = _anchor_state_positions(page, profile, anchor_state)
    anchor_centers = [
        page.folded_to_original[min(position, len(page.folded_to_original) - 1)]
        for position in exact + clustered + fuzzy
        if page.folded_to_original
    ]
    qualifier_positions = [
        (match.start() + match.end()) // 2
        for pattern in qualifier.patterns
        for match in pattern.finditer(page.text)
    ]
    return tuple(sorted(set(anchor_centers + qualifier_positions)))


def _concept_event_relation_features(
    pages: Sequence[PageNode],
    concepts: Sequence[Concept],
    profile: Any,
    qualifier: QualifierProfile,
    anchor_states: Mapping[int, Mapping[str, Any]],
    competitor_profiles: Sequence[
        tuple[Any, QualifierProfile, Mapping[int, Mapping[str, Any]]]
    ],
) -> dict[str, float]:
    entity_positions = {
        page.index: _entity_positions(
            page, profile, qualifier, anchor_states.get(page.index)
        )
        for page in pages
    }
    competitor_positions = {
        page.index: [
            _entity_positions(
                page,
                competitor,
                competitor_qualifier,
                competitor_anchors.get(page.index),
            )
            for competitor, competitor_qualifier, competitor_anchors in competitor_profiles
        ]
        for page in pages
    }
    output: dict[str, float] = {}
    for concept in concepts:
        mass = winner = close = exclusive = 0.0
        distance_scores: list[float] = []
        occurrence_count = 0
        paper_occurrence_count = sum(
            len(page.concept_positions.get(concept.concept_id, ()))
            for page in pages
            if not page.reference_noise
        )
        for page in pages:
            if page.reference_noise:
                continue
            for center in page.concept_positions.get(concept.concept_id, ()):
                own = min(
                    (abs(center - position) for position in entity_positions[page.index]),
                    default=10**9,
                )
                sibling_distances = [
                    min((abs(center - position) for position in positions), default=10**9)
                    for positions in competitor_positions[page.index]
                ]
                distances = [own, *sibling_distances]
                weights = [
                    math.exp(-distance / 180.0) if distance <= 1_200 else 0.0
                    for distance in distances
                ]
                if not any(weights):
                    continue
                occurrence_count += 1
                mass += weights[0] / sum(weights)
                minimum = min(distances)
                tied = sum(distance == minimum for distance in distances)
                if own == minimum:
                    winner += 1.0 / tied
                close += float(own <= 300)
                sibling_minimum = min(sibling_distances, default=10**9)
                exclusive += float(own + 60 < sibling_minimum and own <= 900)
                if own <= 1_200:
                    distance_scores.append(1.0 / (1.0 + own / 120.0))
        prefix = f"event_{concept.category}_{concept.concept_id}_"
        output.update(
            {
                prefix + "paper_occurrence_count_log": _log1p(paper_occurrence_count),
                prefix + "occurrence_count_log": _log1p(occurrence_count),
                prefix + "target_mass_count_log": _log1p(mass),
                prefix + "winner_count_log": _log1p(winner),
                prefix + "close_count_log": _log1p(close),
                prefix + "exclusive_ratio": exclusive / max(1, occurrence_count),
                prefix + "mean_distance_score": _mean(distance_scores),
            }
        )
    return output


def _query_page_scores(
    pages: Sequence[PageNode],
    target_relevance: np.ndarray,
    competitor_direct: np.ndarray,
    category: str,
    qualifier_strength: np.ndarray,
    competitor_qualifier_strength: np.ndarray,
) -> list[tuple[int, float, float, bool]]:
    output = []
    for page in pages:
        count = page.category_counts.get(category, 0)
        if not count or page.reference_noise or target_relevance[page.index] <= 0:
            continue
        category_strength = min(1.0, math.log1p(count) / math.log(5.0))
        qualifier_exclusive = (
            qualifier_strength[page.index] > 0
            and competitor_qualifier_strength[page.index] <= 0
        )
        exclusive = competitor_direct[page.index] <= 0.0 or qualifier_exclusive
        critic = 1.0 if exclusive else 0.58
        if qualifier_strength[page.index] > 0:
            critic *= 1.0 + 0.22 * qualifier_strength[page.index]
        elif competitor_qualifier_strength[page.index] > 0:
            critic *= 0.62
        score = target_relevance[page.index] * category_strength * critic
        output.append((page.index, score, target_relevance[page.index], exclusive))
    return sorted(output, key=lambda item: (-item[1], item[0]))[:TOP_PAGES_PER_QUERY]


def _category_windows(
    page: PageNode,
    concepts: Sequence[Concept],
    category: str,
) -> list[str]:
    spans = sorted(
        (match.start(), match.end())
        for concept in concepts
        if concept.category == category
        for pattern in concept.patterns
        for match in pattern.finditer(page.text)
    )
    intervals: list[list[int]] = []
    for start, end in spans:
        left = max(0, start - CONTEXT_WINDOW_RADIUS)
        right = min(len(page.text), end + CONTEXT_WINDOW_RADIUS)
        if intervals and left <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], right)
        else:
            intervals.append([left, right])
    return [
        re.sub(r"\s+", " ", page.text[start:end]).strip()
        for start, end in intervals[:CONTEXT_WINDOWS_PER_PAGE]
        if page.text[start:end].strip()
    ]


def _discourse_tags(text: str) -> str:
    counts = discourse_role_counts(text)
    return " ".join(
        f"[ROLE_{name.upper()}]" for name, count in counts.items() if count
    )


def _target_context_document(
    pages: Sequence[PageNode],
    concepts: Sequence[Concept],
    profile: Any,
    qualifier: QualifierProfile,
    anchor_states: Mapping[int, Mapping[str, Any]],
    competitor_profiles: Sequence[
        tuple[Any, QualifierProfile, Mapping[int, Mapping[str, Any]]]
    ],
    physical: Mapping[int, set[int]],
    semantic: Mapping[tuple[int, int], float],
) -> str:
    relevance, _ = _target_relevance(
        pages, profile, anchor_states, physical, semantic
    )
    qualifier_strength = _qualifier_page_strength(pages, qualifier)
    competitor_qualifier = np.zeros(len(pages), dtype=np.float64)
    competitor_direct = np.zeros(len(pages), dtype=np.float64)
    for competitor, competitor_component, competitor_anchors in competitor_profiles:
        competitor_qualifier = np.maximum(
            competitor_qualifier,
            _qualifier_page_strength(pages, competitor_component),
        )
        for page in pages:
            value, _, _, _ = _direct_target_relevance(
                page, competitor, competitor_anchors.get(page.index)
            )
            competitor_direct[page.index] = max(
                competitor_direct[page.index], value
            )
    query_pages = {
        category: _query_page_scores(
            pages,
            relevance,
            competitor_direct,
            category,
            qualifier_strength,
            competitor_qualifier,
        )
        for category in QUERY_CATEGORIES
    }
    parts = [
        "[TARGET] " + clean(profile.raw),
        "[TARGET_QUALIFIERS] " + " ".join(qualifier.families),
    ]
    seen: set[str] = set()
    for category in QUERY_CATEGORIES:
        for page_index, _, _, exclusive in query_pages[category][
            :CONTEXT_PAGES_PER_QUERY
        ]:
            page = pages[page_index]
            scope = "EXCLUSIVE" if exclusive else "SHARED"
            heading = " | ".join(page.section_hints)
            for window in _category_windows(page, concepts, category):
                key = _fold(window)
                if not key or key in seen:
                    continue
                seen.add(key)
                roles = _discourse_tags(window)
                parts.append(
                    f"[{category.upper()}_{scope}] [PAGE {page.number}] "
                    f"[SECTION {heading}] {roles} {window}"
                )
    # A target can be absent from OCR text even though the paper has decisive
    # R3 evidence. Preserve a small, category-balanced global fallback while
    # keeping target-bound passages first in the document.
    for category in QUERY_CATEGORIES:
        ranked_pages = sorted(
            (
                page
                for page in pages
                if not page.reference_noise and page.category_counts.get(category, 0) > 0
            ),
            key=lambda page: (-page.category_counts.get(category, 0), page.index),
        )
        for page in ranked_pages[:1]:
            heading = " | ".join(page.section_hints)
            for window in _category_windows(page, concepts, category)[:1]:
                key = _fold(window)
                if not key or key in seen:
                    continue
                seen.add(key)
                roles = _discourse_tags(window)
                parts.append(
                    f"[GLOBAL_{category.upper()}] [PAGE {page.number}] "
                    f"[SECTION {heading}] {roles} {window}"
                )
    document = "\n".join(parts)
    return document[:MAX_TARGET_CONTEXT_CHARACTERS]


def _edge_kind(
    left: int,
    right: int,
    physical: Mapping[int, set[int]],
    semantic: Mapping[tuple[int, int], float],
) -> tuple[str, float]:
    if left == right:
        return "same_page", 1.0
    if right in physical[left]:
        return "physical", 0.88
    if (left, right) in semantic:
        return "semantic", 0.72 + 0.20 * semantic[(left, right)]
    return "none", 0.0


def _connected_three(nodes: Sequence[int], edges: Sequence[tuple[int, int]]) -> bool:
    unique = set(nodes)
    if len(unique) <= 1:
        return True
    adjacency = {node: set() for node in unique}
    for left, right in edges:
        if left in unique and right in unique:
            adjacency[left].add(right)
            adjacency[right].add(left)
    visited = {next(iter(unique))}
    frontier = list(visited)
    while frontier:
        node = frontier.pop()
        for neighbor in adjacency[node] - visited:
            visited.add(neighbor)
            frontier.append(neighbor)
    return visited == unique


def _best_chain(
    pages: Sequence[PageNode],
    query_pages: Mapping[str, Sequence[tuple[int, float, float, bool]]],
    physical: Mapping[int, set[int]],
    semantic: Mapping[tuple[int, int], float],
) -> dict[str, float]:
    candidates = [query_pages.get(category) or [] for category in ("method", "carrier", "bridge")]
    if any(not values for values in candidates):
        return {
            "trace_chain_complete": 0.0,
            "trace_chain_connected": 0.0,
            "trace_chain_score": 0.0,
            "trace_chain_physical_edge_count": 0.0,
            "trace_chain_semantic_edge_count": 0.0,
            "trace_chain_same_page_edge_count": 0.0,
            "trace_chain_page_span_score": 0.0,
            "trace_chain_min_target_relevance": 0.0,
            "trace_chain_all_pages_exclusive": 0.0,
        }
    best: tuple[float, tuple[Any, ...]] | None = None
    for combination in itertools.product(*candidates):
        nodes = [item[0] for item in combination]
        pair_edges = []
        kinds = []
        strengths = []
        for left, right in itertools.combinations(nodes, 2):
            kind, strength = _edge_kind(left, right, physical, semantic)
            if strength:
                pair_edges.append((left, right))
            kinds.append(kind)
            strengths.append(strength)
        connected = _connected_three(nodes, pair_edges)
        node_score = _mean([item[1] for item in combination])
        connectivity = max(strengths, default=0.0) if len(set(nodes)) <= 2 else _mean(
            sorted(strengths, reverse=True)[:2]
        )
        score = node_score * (0.52 + 0.48 * connectivity) * (1.0 if connected else 0.62)
        state = (score, combination, kinds, connected)
        if best is None or state[0] > best[0]:
            best = state
    assert best is not None
    score, combination, kinds, connected = best
    nodes = [item[0] for item in combination]
    page_numbers = [pages[node].number for node in nodes]
    return {
        "trace_chain_complete": 1.0,
        "trace_chain_connected": float(connected),
        "trace_chain_score": float(score),
        "trace_chain_physical_edge_count": float(kinds.count("physical")),
        "trace_chain_semantic_edge_count": float(kinds.count("semantic")),
        "trace_chain_same_page_edge_count": float(kinds.count("same_page")),
        "trace_chain_page_span_score": 1.0 / (1.0 + max(page_numbers) - min(page_numbers)),
        "trace_chain_min_target_relevance": min(item[2] for item in combination),
        "trace_chain_all_pages_exclusive": float(all(item[3] for item in combination)),
    }


def _target_graph_features(
    pages: Sequence[PageNode],
    concepts: Sequence[Concept],
    profile: Any,
    qualifier: QualifierProfile,
    anchor_states: Mapping[int, Mapping[str, Any]],
    competitor_profiles: Sequence[
        tuple[Any, QualifierProfile, Mapping[int, Mapping[str, Any]]]
    ],
    physical: Mapping[int, set[int]],
    semantic: Mapping[tuple[int, int], float],
) -> dict[str, float]:
    relevance, features = _target_relevance(
        pages, profile, anchor_states, physical, semantic
    )
    qualifier_values, qualifier_strength, competitor_qualifier_strength = (
        _qualifier_features(
            pages,
            qualifier,
            relevance,
            [item[1] for item in competitor_profiles],
        )
    )
    features.update(qualifier_values)
    features.update(
        _concept_event_relation_features(
            pages,
            concepts,
            profile,
            qualifier,
            anchor_states,
            competitor_profiles,
        )
    )
    competitor_direct = np.zeros(len(pages), dtype=np.float64)
    for competitor, _, competitor_anchors in competitor_profiles:
        for page in pages:
            value, _, _, _ = _direct_target_relevance(
                page, competitor, competitor_anchors.get(page.index)
            )
            competitor_direct[page.index] = max(competitor_direct[page.index], value)
    query_pages = {
        category: _query_page_scores(
            pages,
            relevance,
            competitor_direct,
            category,
            qualifier_strength,
            competitor_qualifier_strength,
        )
        for category in QUERY_CATEGORIES
    }
    for category, candidates in query_pages.items():
        scores = [item[1] for item in candidates]
        features.update(
            {
                f"query_{category}_available": float(bool(candidates)),
                f"query_{category}_top_score": max(scores, default=0.0),
                f"query_{category}_top3_mean_score": _mean(scores[:3]),
                f"query_{category}_accepted_page_count_log": _log1p(len(candidates)),
                f"query_{category}_exclusive_page_ratio": _mean(
                    [float(item[3]) for item in candidates]
                ),
                f"query_{category}_target_relevance_mean": _mean(
                    [item[2] for item in candidates]
                ),
            }
        )
    accepted_sets = [set(item[0] for item in query_pages[category]) for category in QUERY_CATEGORIES]
    pair_jaccards = []
    for left, right in itertools.combinations(accepted_sets, 2):
        union = left | right
        pair_jaccards.append(len(left & right) / len(union) if union else 0.0)
    top_pages = [values[0][0] for values in query_pages.values() if values]
    features.update(
        {
            "query_consensus_mean_jaccard": _mean(pair_jaccards),
            "query_consensus_same_top_page_ratio": (
                max((top_pages.count(page) for page in set(top_pages)), default=0) / len(top_pages)
                if top_pages else 0.0
            ),
            "query_whitelist_page_count_log": _log1p(len(set().union(*accepted_sets))),
            "query_all_four_available": float(all(query_pages[category] for category in QUERY_CATEGORIES)),
        }
    )
    features.update(_best_chain(pages, query_pages, physical, semantic))
    return features


def _add_sibling_contrasts(feature_maps: dict[str, dict[str, float]]) -> None:
    targets = list(feature_maps)
    names = list(next(iter(feature_maps.values()))) if targets else []
    for target in targets:
        siblings = [item for item in targets if item != target]
        for name in names:
            value = feature_maps[target][name]
            sibling_values = [feature_maps[item][name] for item in siblings]
            maximum = max(sibling_values, default=0.0)
            mean = _mean(sibling_values)
            total = value + sum(sibling_values)
            feature_maps[target][f"contrast_{name}_minus_sibling_max"] = value - maximum
            feature_maps[target][f"contrast_{name}_minus_sibling_mean"] = value - mean
            feature_maps[target][f"contrast_{name}_paper_share"] = value / total if total > 0 else 0.0


def build_trace_page_graph_features(
    identity: Sequence[Mapping[str, str]],
    page_cache_rows: Mapping[str, Mapping[str, Any]],
    concepts: Sequence[Concept],
) -> dict[str, Any]:
    allowed = {"REFNO", "RESULTNO", "ROCKNAME"}
    records = []
    for row_number, record in enumerate(identity, start=1):
        extra = set(record) - allowed
        if extra:
            raise ValueError(f"identity row {row_number} contains forbidden fields: {sorted(extra)}")
        row = {name: clean(record.get(name)) for name in allowed}
        if not all(row.values()):
            raise ValueError("identity requires nonempty REFNO, RESULTNO, and ROCKNAME")
        records.append(row)
    by_refno: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_refno[record["REFNO"]].append(index)

    rows: list[list[float] | None] = [None] * len(records)
    target_context_documents: list[str | None] = [None] * len(records)
    metadata: list[dict[str, Any] | None] = [None] * len(records)
    feature_names: list[str] | None = None
    audits = []
    for paper_number, refno in enumerate(sorted(by_refno), start=1):
        indices = by_refno[refno]
        cache = page_cache_rows.get(refno)
        if cache is None:
            raise ValueError(f"missing page cache row for REFNO {refno}")
        target_names: dict[str, str] = {}
        row_multiplicity: dict[str, int] = defaultdict(int)
        for index in indices:
            canonical = canonical_target_identity(records[index]["ROCKNAME"])
            target_names.setdefault(canonical, records[index]["ROCKNAME"])
            row_multiplicity[canonical] += 1
        profiles = {target: target_profile(name) for target, name in target_names.items()}
        qualifiers = {
            target: qualifier_profile(name) for target, name in target_names.items()
        }
        pages = _prepare_pages(cache, concepts)
        resolved_anchors = resolve_document_target_anchors(
            [{"page": page.number, "text": page.text} for page in pages],
            profiles,
            reference_pages={page.number for page in pages if page.reference_noise},
        )
        anchor_states = {
            target: {
                page.index: resolved_anchors[target][page.number] for page in pages
            }
            for target in profiles
        }
        physical, semantic, _ = build_page_graph(pages)
        feature_maps = {
            target: _target_graph_features(
                pages,
                concepts,
                profile,
                qualifiers[target],
                anchor_states[target],
                [
                    (
                        profiles[sibling],
                        qualifiers[sibling],
                        anchor_states[sibling],
                    )
                    for sibling in profiles
                    if sibling != target
                ],
                physical,
                semantic,
            )
            for target, profile in profiles.items()
        }
        coreference_text = "\n\n".join(
            page.text for page in pages if page.text and not page.reference_noise
        )
        coreference = analyze_coreference_events(
            coreference_text,
            [
                {"target_id": target, "name": target_names[target], "aliases": ()}
                for target in profiles
            ],
            concepts,
            include_event_text=False,
        )
        for target, values in feature_maps.items():
            values.update(coreference.target_features[target])
        for target, values in feature_maps.items():
            values.update(
                {
                    "paper_result_row_count_log": _log1p(len(indices)),
                    "paper_distinct_target_count_log": _log1p(len(profiles)),
                    "identity_duplicate_row_count_log": _log1p(
                        row_multiplicity[target] - 1
                    ),
                    "identity_result_row_share": row_multiplicity[target]
                    / len(indices),
                }
            )
        context_documents = {
            target: _target_context_document(
                pages,
                concepts,
                profile,
                qualifiers[target],
                anchor_states[target],
                [
                    (
                        profiles[sibling],
                        qualifiers[sibling],
                        anchor_states[sibling],
                    )
                    for sibling in profiles
                    if sibling != target
                ],
                physical,
                semantic,
            )
            for target, profile in profiles.items()
        }
        for target, values in feature_maps.items():
            values.update(discourse_role_features(context_documents[target]))
        _add_sibling_contrasts(feature_maps)
        for index in indices:
            canonical = canonical_target_identity(records[index]["ROCKNAME"])
            values = feature_maps[canonical]
            names = list(values)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise AssertionError("TRACE feature order changed")
            rows[index] = [float(values[name]) for name in names]
            target_context_documents[index] = context_documents[canonical]
            metadata[index] = {
                "canonical_target": canonical,
                "qualifier_families": list(qualifiers[canonical].families),
                "identity_duplicate_rows": row_multiplicity[canonical],
                "paper_pages": len(pages),
                "physical_edges": sum(len(value) for value in physical.values()) // 2,
                "semantic_edges": len(semantic) // 2,
                "atomic_events": len(coreference.events),
                "resolved_atomic_events": sum(
                    event.target_id is not None for event in coreference.events
                ),
                "fuzzy_anchor_pages": sum(
                    bool(state.get("fuzzy"))
                    for state in anchor_states[canonical].values()
                ),
                "ambiguous_fuzzy_anchor_pages": sum(
                    bool(state.get("ambiguous"))
                    for state in anchor_states[canonical].values()
                ),
            }
        audits.append(
            {
                "REFNO": refno,
                "targets": len(profiles),
                "targets_with_qualifier": sum(
                    bool(profile.families) for profile in qualifiers.values()
                ),
                "indistinguishable_identity_rows": len(indices) - len(profiles),
                "pages": len(pages),
                "physical_edges": sum(len(value) for value in physical.values()) // 2,
                "semantic_edges": len(semantic) // 2,
                "atomic_events": len(coreference.events),
                "resolved_atomic_events": sum(
                    event.target_id is not None for event in coreference.events
                ),
                "targets_with_atomic_events": sum(
                    coreference.target_features[target][
                        "coref_assigned_event_count_log"
                    ]
                    > 0
                    for target in profiles
                ),
                "targets_with_fuzzy_anchor": sum(
                    any(state.get("fuzzy") for state in anchor_states[target].values())
                    for target in profiles
                ),
                "targets_with_ambiguous_fuzzy_anchor": sum(
                    any(
                        state.get("ambiguous")
                        for state in anchor_states[target].values()
                    )
                    for target in profiles
                ),
            }
        )
        if paper_number % 250 == 0:
            print(f"TRACE page graph features: papers={paper_number}/{len(by_refno)}", flush=True)
    if (
        feature_names is None
        or any(row is None for row in rows)
        or any(document is None for document in target_context_documents)
    ):
        raise AssertionError("TRACE page graph feature construction is incomplete")
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise AssertionError("TRACE page graph feature matrix is invalid")
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "target_context_documents": target_context_documents,
        "row_metadata": metadata,
        "paper_audit": audits,
        "paper_count": len(by_refno),
    }


__all__ = [
    "Concept",
    "build_page_graph",
    "build_trace_page_graph_features",
    "load_trace_concepts",
    "qualifier_profile",
]
