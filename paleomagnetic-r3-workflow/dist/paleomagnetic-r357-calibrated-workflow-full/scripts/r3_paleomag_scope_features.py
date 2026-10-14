#!/usr/bin/env python3
"""Label-free paleomagnetic document-structure features for R3 targets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
import unicodedata
from typing import Any, Collection, Mapping, Sequence

import numpy as np

try:
    from .paper_text_retrieval import canonical_target_identity
except ImportError:
    from paper_text_retrieval import canonical_target_identity


GENERIC_TARGET_TOKENS = {
    "and", "basalt", "basalts", "batholith", "complex", "combined", "directions",
    "dyke", "dykes", "formation", "gabbro", "granite", "group", "intrusion",
    "lava", "lavas", "pluton", "rock", "rocks", "sediment", "sediments", "sequence",
    "sill", "sills", "suite", "volcanic", "volcanics", "west", "east", "north", "south",
}
RESULT_QUALIFIER_RE = re.compile(
    r"(?:\s*[,;:/-]\s*|\s+)(?:combined(?:\s*,.*)?|[NSEW]\s+directions?|"
    r"(?:normal|reverse|reversed|[ABNSEW]|MT)\s+(?:components?|comp\.?)|"
    r"[AB]\s+magneti[sz]ation)\s*$",
    re.I,
)
HEADING_CUE_RE = re.compile(
    r"\b(?:results?|rock\s+magnet(?:ism|ic)|magnetic\s+mineralogy|"
    r"thermo[- ]?magnetic|petrography|mineralogy|component\s+[a-z0-9]+)\b",
    re.I,
)
CAPTION_RE = re.compile(r"\b(?:fig(?:ure)?s?|tables?)\s*[a-z0-9.:-]*", re.I)

PATTERNS = {
    "rockmag": re.compile(
        r"\b(?:IRM|SIRM|FORC|Lowrie|hystere(?:sis|tic)|back[- ]field|"
        r"thermo[- ]?magnetic|Curie\s+(?:point|temperature)|Day\s+plot|Henkel\s+plot|"
        r"coerciv(?:e|ity)|Hcr|Bcr|Mrs|Jrs|M[oö]ssbauer|SEM|TEM|EPMA|EDS|XRD|"
        r"ore\s+microscop|reflected[- ]light|magnetic\s+mineralogy|"
        r"rock[- ]magnetic|susceptibility.{0,40}temperature|Js[- ]?T|kappa[- ]?T)\b",
        re.I,
    ),
    "diagnostic": re.compile(
        r"\b(?:saturat(?:e|ed|es|ion)|does\s+not\s+saturate|Curie\s+(?:point|temperature)|"
        r"single[- ]domain|pseudo[- ]single[- ]domain|multi[- ]domain|SD|PSD|MD|"
        r"Verwey|Morin|Hcr\s*/\s*Hc|Bcr\s*/\s*Bc|Mrs\s*/\s*Ms|Jrs\s*/\s*Js)\b",
        re.I,
    ),
    "carrier": re.compile(
        r"\b(?:titano[- ]?magnetite|titano[- ]?maghemite|magnetite|hematite|haematite|"
        r"maghemite|goethite|pyrrhotite|greigite|ilmenite|ilmenomagnetite|"
        r"Fe[- ]?Ti\s+oxide|magnetic\s+carrier|remanence\s+carrier)\b",
        re.I,
    ),
    "bridge": re.compile(
        r"\b(?:carr(?:y|ies|ied|ying)|carried\s+by|carrier(?:s)?\s+(?:of|for)|"
        r"attributed\s+to|due\s+to|resides?\s+in|primary|secondary|overprint|"
        r"remagneti[sz](?:ed|ation)|detrital|chemical|authigenic|alteration)\b",
        re.I,
    ),
    "demag": re.compile(
        r"\b(?:thermal|AF|alternating[- ]field)\s+(?:cleaning|demagneti[sz]ation)|"
        r"\bdemagneti[sz](?:e|ed|ation)|\bunblocking\s+temperature|"
        r"\bZijderveld|\borthogonal\s+(?:vector\s+)?plots?|\bvector\s+diagrams?",
        re.I,
    ),
    "component": re.compile(
        r"\b(?:ChRM|characteristic\s+remanent\s+magneti[sz]ation|"
        r"component\s+(?:[A-Z][0-9]?|HTC|LTC)|(?:high|low)[- ]temperature\s+component|"
        r"(?:high|low)[- ]coercivity\s+component|HTC|LTC|blocking\s+(?:range|spectrum))\b",
        re.I,
    ),
    "sample": re.compile(
        r"\b(?:samples?|specimens?|sites?|localit(?:y|ies)|cores?|drill(?:ed|ing)?|"
        r"sample\s+group|site\s+mean)\b",
        re.I,
    ),
    "direction": re.compile(
        r"\b(?:normal|reverse|reversed)\s+polarity|\b(?:declination|inclination)|"
        r"\bD\s*/\s*I\b|\ba95\b|\bpalaeomagnetic\s+pole|\bpaleomagnetic\s+pole|"
        r"\bfold\s+test|\btilt[- ]corrected|\bmean\s+direction",
        re.I,
    ),
}
POSITIVE_CATEGORIES = ("rockmag", "diagnostic", "carrier", "bridge")
CONTEXT_RADIUS = 520
ANCHOR_CLUSTER_CHARS = 180
RESPONSIBILITY_MAX_DISTANCE = 900
SOFTMAX_DISTANCE = 180.0
FUZZY_TOKEN_SIMILARITY = 0.86
FUZZY_MULTI_MEAN_SIMILARITY = 0.90
FUZZY_SINGLE_SIMILARITY = 0.92
FUZZY_SIBLING_MARGIN = 0.08
FUZZY_LONG_TOKEN_CHARS = 8


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", clean(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


@dataclass(frozen=True)
class TargetProfile:
    raw: str
    folded: str
    material_folded: str
    token_groups: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class _TokenUnit:
    """One folded word or a two-word compound, with folded-text coordinates."""

    text: str
    token_start: int
    token_end: int
    char_start: int
    char_end: int


@dataclass(frozen=True)
class _FuzzyTokenMatch:
    target_start: int
    target_end: int
    document_start: int
    document_end: int
    char_start: int
    char_end: int
    target_text: str
    document_text: str
    similarity: float
    exact: bool


@dataclass(frozen=True)
class _FuzzyAnchorCandidate:
    position: int
    start: int
    end: int
    score: float
    coverage: float


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(re.findall(r"[a-z0-9]+", value)))


def _specific_tokens(value: str) -> tuple[str, ...]:
    tokens = tuple(
        token for token in _tokens(value)
        if len(token) > 2 and token not in GENERIC_TARGET_TOKENS
    )
    return tokens or tuple(token for token in _tokens(value) if len(token) > 2)


def target_profile(value: Any) -> TargetProfile:
    raw = clean(value)
    folded = _fold(raw)
    material_raw = RESULT_QUALIFIER_RE.sub("", raw).strip(" ,;:/-")
    material_folded = _fold(material_raw) or folded
    groups = []
    for candidate in (folded, material_folded):
        group = _specific_tokens(candidate)
        if group and group not in groups:
            groups.append(group)
    return TargetProfile(raw, folded, material_folded, tuple(groups))


def _word_positions(text: str, token: str) -> list[int]:
    variants = {token}
    if token.endswith("ies") and len(token) > 4:
        variants.add(token[:-3] + "y")
    elif token.endswith("s") and len(token) > 3:
        variants.add(token[:-1])
    expression = "|".join(re.escape(item) for item in sorted(variants, key=len, reverse=True))
    return [
        (match.start() + match.end()) // 2
        for match in re.finditer(r"(?<![a-z0-9])(?:" + expression + r")(?![a-z0-9])", text)
    ]


def _cluster_positions(text: str, group: Sequence[str]) -> list[int]:
    hits = sorted(
        (position, token)
        for token in group
        for position in _word_positions(text, token)
    )
    required = max(1, math.ceil(len(group) * 0.6))
    positions: list[int] = []
    left = 0
    for right, (position, _) in enumerate(hits):
        while hits[left][0] < position - ANCHOR_CLUSTER_CHARS:
            left += 1
        window = hits[left : right + 1]
        if len({token for _, token in window}) >= required:
            positions.append(round(_mean([item[0] for item in window])))
    output: list[int] = []
    for position in positions:
        if not output or position - output[-1] > 60:
            output.append(position)
    return output


def anchor_positions(text: str, profile: TargetProfile) -> tuple[list[int], list[int]]:
    exact: list[int] = []
    for phrase in dict.fromkeys((profile.folded, profile.material_folded)):
        if not phrase:
            continue
        exact.extend(
            (match.start() + match.end()) // 2
            for match in re.finditer(
                r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", text
            )
        )
    exact = sorted(set(exact))
    clustered = sorted(
        set(position for group in profile.token_groups for position in _cluster_positions(text, group))
    )
    clustered = [
        position for position in clustered
        if all(abs(position - exact_position) > 60 for exact_position in exact)
    ]
    return exact, clustered


def _page_rows(
    page_texts: Sequence[str | Mapping[str, Any]] | Mapping[int, str],
) -> list[tuple[int, str]]:
    """Normalize supported page inputs without ever joining page boundaries."""

    rows: list[tuple[int, str]] = []
    if isinstance(page_texts, Mapping):
        source = page_texts.items()
        for page_number, text in source:
            rows.append((int(page_number), _fold(text)))
    elif isinstance(page_texts, Sequence) and not isinstance(page_texts, (str, bytes)):
        for offset, item in enumerate(page_texts, start=1):
            if isinstance(item, Mapping):
                page_number = int(item.get("page") or offset)
                text = item.get("text")
            else:
                page_number = offset
                text = item
            rows.append((page_number, _fold(text)))
    else:
        raise TypeError("page_texts must be a page mapping or a sequence of page texts")
    if len({page_number for page_number, _ in rows}) != len(rows):
        raise ValueError("page_texts contains duplicate page numbers")
    return sorted(rows)


def _document_token_units(text: str) -> list[_TokenUnit]:
    words = [
        (match.group(), match.start(), match.end())
        for match in re.finditer(r"[a-z0-9]+", text)
    ]
    units = [
        _TokenUnit(word, index, index, start, end)
        for index, (word, start, end) in enumerate(words)
    ]
    for index in range(len(words) - 1):
        compound = words[index][0] + words[index + 1][0]
        if len(compound) >= 5:
            units.append(
                _TokenUnit(
                    compound,
                    index,
                    index + 1,
                    words[index][1],
                    words[index + 1][2],
                )
            )
    return units


def _target_token_units(tokens: Sequence[str]) -> list[_TokenUnit]:
    units = [
        _TokenUnit(token, index, index, index, index + 1)
        for index, token in enumerate(tokens)
    ]
    for index in range(len(tokens) - 1):
        compound = tokens[index] + tokens[index + 1]
        if len(compound) >= 5:
            units.append(_TokenUnit(compound, index, index + 1, index, index + 2))
    return units


def _token_similarity(left: str, right: str) -> float | None:
    if left == right:
        return 1.0
    if min(len(left), len(right)) < 4 or left[:2] != right[:2]:
        return None
    maximum_length = max(len(left), len(right))
    length_limit = max(1, min(3, math.ceil(maximum_length * 0.2)))
    if abs(len(left) - len(right)) > length_limit:
        return None
    similarity = SequenceMatcher(None, left, right).ratio()
    return similarity if similarity >= FUZZY_TOKEN_SIMILARITY else None


def _fuzzy_token_matches(text: str, tokens: Sequence[str]) -> list[_FuzzyTokenMatch]:
    document_units = _document_token_units(text)
    by_prefix: dict[str, list[_TokenUnit]] = defaultdict(list)
    for unit in document_units:
        if len(unit.text) >= 2:
            by_prefix[unit.text[:2]].append(unit)
    matches: list[_FuzzyTokenMatch] = []
    for target_unit in _target_token_units(tokens):
        for document_unit in by_prefix.get(target_unit.text[:2], []):
            similarity = _token_similarity(target_unit.text, document_unit.text)
            if similarity is None:
                continue
            matches.append(
                _FuzzyTokenMatch(
                    target_start=target_unit.token_start,
                    target_end=target_unit.token_end,
                    document_start=document_unit.token_start,
                    document_end=document_unit.token_end,
                    char_start=document_unit.char_start,
                    char_end=document_unit.char_end,
                    target_text=target_unit.text,
                    document_text=document_unit.text,
                    similarity=similarity,
                    exact=target_unit.text == document_unit.text,
                )
            )
    return matches


def _select_nonoverlapping_matches(
    matches: Sequence[_FuzzyTokenMatch],
) -> list[_FuzzyTokenMatch]:
    selected: list[_FuzzyTokenMatch] = []
    occupied_target: set[int] = set()
    occupied_document: set[int] = set()
    ordered = sorted(
        matches,
        key=lambda item: (
            -item.similarity,
            -(item.target_end - item.target_start + 1),
            item.document_end - item.document_start,
            item.char_start,
            item.target_start,
        ),
    )
    for match in ordered:
        target_indices = set(range(match.target_start, match.target_end + 1))
        document_indices = set(range(match.document_start, match.document_end + 1))
        if target_indices & occupied_target or document_indices & occupied_document:
            continue
        selected.append(match)
        occupied_target.update(target_indices)
        occupied_document.update(document_indices)
    return selected


def _candidate_from_window(
    matches: Sequence[_FuzzyTokenMatch],
    token_count: int,
) -> _FuzzyAnchorCandidate | None:
    selected = _select_nonoverlapping_matches(matches)
    if not selected:
        return None
    if (
        max(match.char_end for match in selected)
        - min(match.char_start for match in selected)
        > ANCHOR_CLUSTER_CHARS
    ):
        return None
    covered = sum(match.target_end - match.target_start + 1 for match in selected)
    coverage = covered / max(1, token_count)
    if coverage < 0.6:
        return None
    weighted_similarity = sum(
        match.similarity * (match.target_end - match.target_start + 1)
        for match in selected
    ) / covered
    if token_count == 1:
        strongest = max(selected, key=lambda item: item.similarity)
        if (
            weighted_similarity < FUZZY_SINGLE_SIMILARITY
            or max(len(strongest.target_text), len(strongest.document_text)) < 6
        ):
            return None
    else:
        if weighted_similarity < FUZZY_MULTI_MEAN_SIMILARITY:
            return None
        corroborated = any(
            match.exact
            or (
                max(len(match.target_text), len(match.document_text))
                >= FUZZY_LONG_TOKEN_CHARS
                and match.similarity >= FUZZY_SINGLE_SIMILARITY
            )
            for match in selected
        )
        if not corroborated:
            return None
    start = min(match.char_start for match in selected)
    end = max(match.char_end for match in selected)
    weighted_centers = [
        ((match.char_start + match.char_end) / 2, match.target_end - match.target_start + 1)
        for match in selected
    ]
    position = round(
        sum(center * weight for center, weight in weighted_centers)
        / sum(weight for _, weight in weighted_centers)
    )
    return _FuzzyAnchorCandidate(position, start, end, weighted_similarity, coverage)


def _fuzzy_anchor_candidates(text: str, profile: TargetProfile) -> list[_FuzzyAnchorCandidate]:
    candidates: list[_FuzzyAnchorCandidate] = []
    for tokens in profile.token_groups:
        matches = sorted(
            _fuzzy_token_matches(text, tokens),
            key=lambda item: (item.char_start, item.char_end, item.target_start),
        )
        for seed in matches:
            window = [
                match for match in matches
                if match.char_start >= seed.char_start
                and match.char_end - seed.char_start <= ANCHOR_CLUSTER_CHARS
            ]
            candidate = _candidate_from_window(window, len(tokens))
            if candidate is not None:
                candidates.append(candidate)
    unique: dict[tuple[int, int, int], _FuzzyAnchorCandidate] = {}
    for candidate in candidates:
        key = (candidate.position, candidate.start, candidate.end)
        previous = unique.get(key)
        if previous is None or (candidate.score, candidate.coverage) > (
            previous.score,
            previous.coverage,
        ):
            unique[key] = candidate
    selected: list[_FuzzyAnchorCandidate] = []
    for candidate in sorted(
        unique.values(),
        key=lambda item: (-item.score, -item.coverage, item.position, item.start, item.end),
    ):
        if any(abs(candidate.position - kept.position) <= 60 for kept in selected):
            continue
        selected.append(candidate)
    return sorted(selected, key=lambda item: (item.position, item.start, item.end))


def _windows_overlap(left: _FuzzyAnchorCandidate, right: _FuzzyAnchorCandidate) -> bool:
    return left.start < right.end and right.start < left.end


def resolve_document_target_anchors(
    page_texts: Sequence[str | Mapping[str, Any]] | Mapping[int, str],
    profiles: Mapping[str, TargetProfile | str],
    reference_pages: Collection[int] = (),
) -> dict[str, dict[int, dict[str, Any]]]:
    """Resolve folded target coordinates with a conservative fuzzy fallback.

    Reference pages return no anchors. Fuzzy matching is available only when a
    target has no exact or 180-character token-cluster anchor on any remaining
    page. It never crosses pages and jointly arbitrates overlapping candidates
    from sibling targets. A score gap below 0.08 marks the page ambiguous and
    assigns neither candidate; otherwise only the leading sibling retains the
    fuzzy coordinate.

    The return value is ``target -> page -> fields``. ``exact``, ``cluster``, and
    ``fuzzy`` are center offsets in the page's folded text. ``fuzzy_scores`` is
    aligned with ``fuzzy``; ``fuzzy_score`` is their maximum (or zero), and
    ``ambiguous`` records a rejected near-tie.
    """

    pages = _page_rows(page_texts)
    normalized_profiles = {
        clean(target): (
            profile if isinstance(profile, TargetProfile) else target_profile(profile)
        )
        for target, profile in profiles.items()
    }
    if any(not target for target in normalized_profiles):
        raise ValueError("profile keys must be nonempty target identifiers")
    references = {int(page_number) for page_number in reference_pages}
    output: dict[str, dict[int, dict[str, Any]]] = {
        target: {
            page_number: {
                "exact": [],
                "cluster": [],
                "fuzzy": [],
                "fuzzy_scores": [],
                "fuzzy_score": 0.0,
                "ambiguous": False,
            }
            for page_number, _ in pages
        }
        for target in normalized_profiles
    }
    has_body_anchor: dict[str, bool] = {target: False for target in normalized_profiles}
    for target, profile in normalized_profiles.items():
        for page_number, folded in pages:
            if page_number in references:
                continue
            exact, clustered = anchor_positions(folded, profile)
            output[target][page_number]["exact"] = exact
            output[target][page_number]["cluster"] = clustered
            if exact or clustered:
                has_body_anchor[target] = True

    fuzzy_by_page: dict[int, dict[str, list[_FuzzyAnchorCandidate]]] = defaultdict(dict)
    for target, profile in normalized_profiles.items():
        if has_body_anchor[target]:
            continue
        for page_number, folded in pages:
            if page_number in references:
                continue
            fuzzy_by_page[page_number][target] = _fuzzy_anchor_candidates(folded, profile)

    for page_number, target_candidates in fuzzy_by_page.items():
        flattened = [
            (target, candidate)
            for target, candidates in target_candidates.items()
            for candidate in candidates
        ]
        retained: dict[str, list[_FuzzyAnchorCandidate]] = defaultdict(list)
        for target, candidate in flattened:
            competitors = [
                other for sibling, other in flattened
                if sibling != target and _windows_overlap(candidate, other)
            ]
            near_tie = any(
                abs(candidate.score - competitor.score) < FUZZY_SIBLING_MARGIN
                for competitor in competitors
            )
            if near_tie:
                output[target][page_number]["ambiguous"] = True
                continue
            if any(
                competitor.score > candidate.score
                for competitor in competitors
            ):
                continue
            retained[target].append(candidate)
        for target, candidates in retained.items():
            candidates = sorted(candidates, key=lambda item: item.position)
            scores = [float(candidate.score) for candidate in candidates]
            output[target][page_number]["fuzzy"] = [
                candidate.position for candidate in candidates
            ]
            output[target][page_number]["fuzzy_scores"] = scores
            output[target][page_number]["fuzzy_score"] = max(scores, default=0.0)
    return output


def _merge_intervals(intervals: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    output: list[list[int]] = []
    for start, end in sorted(intervals):
        if not output or start > output[-1][1]:
            output.append([start, end])
        else:
            output[-1][1] = max(output[-1][1], end)
    return [(start, end) for start, end in output]


def _line_intervals(text: str) -> list[tuple[int, int]]:
    intervals = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        intervals.append((start, end))
        start = end
    if start < len(text):
        intervals.append((start, len(text)))
    return intervals


def _contains_anchor(text: str, profile: TargetProfile) -> bool:
    exact, clustered = anchor_positions(text, profile)
    return bool(exact or clustered)


def _context_feature_map(
    contexts: Sequence[tuple[str, bool, bool, bool]],
) -> dict[str, float]:
    output: dict[str, float] = {"local_context_count_log": _log1p(len(contexts))}
    hits_by_category: dict[str, list[int]] = {category: [] for category in PATTERNS}
    exclusive_by_category: dict[str, int] = {category: 0 for category in PATTERNS}
    heading_by_category: dict[str, int] = {category: 0 for category in PATTERNS}
    caption_by_category: dict[str, int] = {category: 0 for category in PATTERNS}
    complete = method_diagnostic = method_carrier = carrier_bridge = demag_only = 0
    positive_without_demag = 0
    for context, exclusive, heading, caption in contexts:
        present = {category: len(pattern.findall(context)) for category, pattern in PATTERNS.items()}
        for category, count in present.items():
            hits_by_category[category].append(count)
            exclusive_by_category[category] += int(exclusive and count > 0)
            heading_by_category[category] += int(heading and count > 0)
            caption_by_category[category] += int(caption and count > 0)
        method_diagnostic += int(present["rockmag"] > 0 and present["diagnostic"] > 0)
        method_carrier += int(present["rockmag"] > 0 and present["carrier"] > 0)
        carrier_bridge += int(present["carrier"] > 0 and present["bridge"] > 0)
        complete += int(
            present["rockmag"] > 0 and present["carrier"] > 0 and present["bridge"] > 0
        )
        demag_only += int(
            present["demag"] > 0 and not any(present[name] for name in POSITIVE_CATEGORIES)
        )
        positive_without_demag += int(
            any(present[name] for name in POSITIVE_CATEGORIES) and present["demag"] == 0
        )
    for category in PATTERNS:
        counts = hits_by_category[category]
        output[f"local_{category}_count_log"] = _log1p(sum(counts))
        output[f"local_{category}_context_ratio"] = _mean([float(value > 0) for value in counts])
        output[f"exclusive_{category}_context_count_log"] = _log1p(exclusive_by_category[category])
        output[f"heading_{category}_context_count_log"] = _log1p(heading_by_category[category])
        output[f"caption_{category}_context_count_log"] = _log1p(caption_by_category[category])
    output.update(
        {
            "local_method_diagnostic_context_count_log": _log1p(method_diagnostic),
            "local_method_carrier_context_count_log": _log1p(method_carrier),
            "local_carrier_bridge_context_count_log": _log1p(carrier_bridge),
            "local_complete_chain_context_count_log": _log1p(complete),
            "local_demag_only_context_count_log": _log1p(demag_only),
            "local_positive_without_demag_context_count_log": _log1p(positive_without_demag),
            "local_positive_to_demag_ratio": (
                sum(sum(hits_by_category[name]) for name in POSITIVE_CATEGORIES)
                / max(1.0, sum(hits_by_category["demag"]))
            ),
        }
    )
    return output


def _sibling_contrasts(feature_maps: dict[str, dict[str, float]]) -> None:
    targets = list(feature_maps)
    names = list(next(iter(feature_maps.values()))) if targets else []
    for target in targets:
        siblings = [item for item in targets if item != target]
        for name in names:
            value = feature_maps[target][name]
            sibling_values = [feature_maps[item][name] for item in siblings]
            sibling_maximum = max(sibling_values, default=0.0)
            sibling_mean = _mean(sibling_values)
            total = value + sum(sibling_values)
            feature_maps[target][f"contrast_{name}_minus_sibling_max"] = value - sibling_maximum
            feature_maps[target][f"contrast_{name}_minus_sibling_mean"] = value - sibling_mean
            feature_maps[target][f"contrast_{name}_paper_share"] = value / total if total > 0 else 0.0


def build_paleomag_scope_features(
    identity: Sequence[Mapping[str, str]],
    page_cache_rows: Mapping[str, Mapping[str, Any]],
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
    metadata: list[dict[str, Any] | None] = [None] * len(records)
    feature_names: list[str] | None = None
    paper_audit = []
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
        pages = []
        for offset, page in enumerate(cache.get("pages") or [], start=1):
            if not isinstance(page, Mapping):
                continue
            raw = clean(page.get("text"))
            pages.append((int(page.get("page") or offset), raw, _fold(raw)))
        anchors: dict[str, dict[int, list[int]]] = {target: {} for target in profiles}
        exact_counts = CounterLike()
        cluster_counts = CounterLike()
        for target, profile in profiles.items():
            for page_number, _, folded in pages:
                exact, clustered = anchor_positions(folded, profile)
                anchors[target][page_number] = sorted(set(exact + clustered))
                exact_counts[target] += len(exact)
                cluster_counts[target] += len(clustered)

        feature_maps: dict[str, dict[str, float]] = {}
        for target, profile in profiles.items():
            contexts: list[tuple[str, bool, bool, bool]] = []
            heading_count = caption_count = table_count = exclusive_count = 0
            anchor_pages = 0
            for page_number, raw, folded in pages:
                positions = anchors[target].get(page_number) or []
                if positions:
                    anchor_pages += 1
                competitor_positions = sorted(
                    position
                    for sibling in profiles
                    if sibling != target
                    for position in anchors[sibling].get(page_number, [])
                )
                bounded_intervals = []
                for position in positions:
                    preceding = [item for item in competitor_positions if item < position]
                    following = [item for item in competitor_positions if item > position]
                    heading_anchor = bool(
                        HEADING_CUE_RE.search(
                            folded[max(0, position - 100) : min(len(folded), position + 120)]
                        )
                    )
                    if heading_anchor:
                        start = max(0, position - len(profile.material_folded) // 2 - 12)
                        end = min(len(folded), position + 1200)
                        if preceding:
                            start = max(start, preceding[-1] + 20)
                        if following:
                            end = min(end, following[0] - 20)
                    else:
                        start = max(0, position - CONTEXT_RADIUS)
                        end = min(len(folded), position + CONTEXT_RADIUS)
                        if preceding:
                            start = max(start, (preceding[-1] + position) // 2)
                        if following:
                            end = min(end, (following[0] + position) // 2)
                    bounded_intervals.append((start, end))
                intervals = _merge_intervals(
                    bounded_intervals
                )
                for start, end in intervals:
                    context = folded[start:end]
                    competitors = sum(
                        bool(
                            [position for position in anchors[sibling].get(page_number, []) if start <= position <= end]
                        )
                        for sibling in profiles if sibling != target
                    )
                    exclusive = competitors == 0
                    heading = bool(HEADING_CUE_RE.search(context[: min(len(context), 650)]))
                    caption = bool(CAPTION_RE.search(context))
                    contexts.append((context, exclusive, heading, caption))
                    exclusive_count += int(exclusive)
                    heading_count += int(heading)
                    caption_count += int(caption)
                for line_start, line_end in _line_intervals(folded):
                    line = folded[line_start:line_end]
                    if not line or not _contains_anchor(line, profile):
                        continue
                    table_count += int("\t" in raw[line_start:line_end])
            features = {
                "identity_duplicate_row_count_log": _log1p(row_multiplicity[target] - 1),
                "identity_material_alias_changed": float(profile.material_folded != profile.folded),
                "identity_specific_token_count_log": _log1p(max(len(group) for group in profile.token_groups)),
                "anchor_exact_count_log": _log1p(exact_counts[target]),
                "anchor_cluster_count_log": _log1p(cluster_counts[target]),
                "anchor_page_count_log": _log1p(anchor_pages),
                "anchor_context_count_log": _log1p(len(contexts)),
                "anchor_exclusive_context_count_log": _log1p(exclusive_count),
                "anchor_heading_context_count_log": _log1p(heading_count),
                "anchor_caption_context_count_log": _log1p(caption_count),
                "anchor_table_line_count_log": _log1p(table_count),
            }
            features.update(_context_feature_map(contexts))
            feature_maps[target] = features

        # Allocate each domain occurrence to the closest target anchor on the same page.
        for category, pattern in PATTERNS.items():
            winner_counts = {target: 0.0 for target in profiles}
            mass_counts = {target: 0.0 for target in profiles}
            distance_scores = {target: [] for target in profiles}
            total_occurrences = 0
            for page_number, _, folded in pages:
                for match in pattern.finditer(folded):
                    center = (match.start() + match.end()) // 2
                    distances = {
                        target: min(
                            (abs(center - position) for position in anchors[target].get(page_number, [])),
                            default=10**9,
                        )
                        for target in profiles
                    }
                    minimum = min(distances.values(), default=10**9)
                    if minimum > RESPONSIBILITY_MAX_DISTANCE:
                        continue
                    total_occurrences += 1
                    weights = {
                        target: math.exp(-min(distance, 5000) / SOFTMAX_DISTANCE)
                        for target, distance in distances.items()
                    }
                    denominator = sum(weights.values()) or 1.0
                    winners = [target for target, distance in distances.items() if distance == minimum]
                    for target in profiles:
                        mass_counts[target] += weights[target] / denominator
                        if target in winners:
                            winner_counts[target] += 1.0 / len(winners)
                        if distances[target] <= RESPONSIBILITY_MAX_DISTANCE:
                            distance_scores[target].append(1.0 / (1.0 + distances[target] / 120.0))
            for target in profiles:
                feature_maps[target][f"responsibility_{category}_winner_count_log"] = _log1p(winner_counts[target])
                feature_maps[target][f"responsibility_{category}_mass_share"] = mass_counts[target] / max(1, total_occurrences)
                feature_maps[target][f"responsibility_{category}_mean_distance_score"] = _mean(distance_scores[target])

        _sibling_contrasts(feature_maps)
        for index in indices:
            canonical = canonical_target_identity(records[index]["ROCKNAME"])
            values = feature_maps[canonical]
            names = list(values)
            if feature_names is None:
                feature_names = names
            elif names != feature_names:
                raise AssertionError("paleomagnetic scope feature order changed")
            rows[index] = [float(values[name]) for name in feature_names]
            metadata[index] = {
                "canonical_target": canonical,
                "paper_target_count": len(profiles),
                "identity_duplicate_rows": row_multiplicity[canonical],
                "anchor_occurrences": exact_counts[canonical] + cluster_counts[canonical],
            }
        paper_audit.append(
            {
                "REFNO": refno,
                "targets": len(profiles),
                "pages": len(pages),
                "targets_without_anchor": sum(
                    not any(anchors[target].values()) for target in profiles
                ),
            }
        )
        if paper_number % 250 == 0:
            print(f"paleomag scope features: papers={paper_number}/{len(by_refno)}", flush=True)

    if feature_names is None or any(row is None for row in rows):
        raise AssertionError("paleomagnetic scope feature construction is incomplete")
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise AssertionError("paleomagnetic scope feature matrix is invalid")
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "row_metadata": metadata,
        "paper_audit": paper_audit,
        "paper_count": len(by_refno),
    }


class CounterLike(defaultdict[str, int]):
    def __init__(self) -> None:
        super().__init__(int)


__all__ = [
    "anchor_positions",
    "build_paleomag_scope_features",
    "resolve_document_target_anchors",
    "target_profile",
]
