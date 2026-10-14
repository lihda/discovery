#!/usr/bin/env python3
"""Conservative target coreference and atomic R3 evidence events.

The module has a deliberately narrow boundary: callers provide original text,
clean target identities, and frozen concept patterns.  It resolves only exact
target mentions and a small paleomagnetic vocabulary of demonstratives,
components, and polarities.  An event receives target responsibility only when
that target is unique within the sentence or within a same-paragraph,
two-sentence antecedent window.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import unicodedata
from typing import Any, Mapping, Pattern, Sequence

try:
    from .r3_trace_discourse_roles import (
        DISCOURSE_ROLE_NAMES,
        annotate_discourse_window,
    )
except ImportError:
    from r3_trace_discourse_roles import (
        DISCOURSE_ROLE_NAMES,
        annotate_discourse_window,
    )


EVENT_CATEGORIES = ("method", "diagnostic", "carrier", "bridge")
MAX_ANTECEDENT_SENTENCES = 2

_DIRECT_ASSIGNMENTS = frozenset(("direct_explicit", "direct_qualified"))
_INHERITED_ASSIGNMENTS = frozenset(("inherited_reference", "qualified_context"))
_ADVERSE_ROLES = frozenset(
    (
        "background_citation_prior_work",
        "negated",
        "prospective",
        "conditional",
    )
)
_OWN_EVIDENCE_ROLES = frozenset(
    ("performed_execution", "reported_result", "interpretation")
)

_BASE_FEATURE_NAMES = (
    "coref_assigned_sentence_count_log",
    "coref_direct_sentence_count_log",
    "coref_qualified_sentence_count_log",
    "coref_inherited_sentence_count_log",
    "coref_assigned_event_count_log",
    "coref_direct_event_count_log",
    "coref_inherited_event_count_log",
    "coref_responsibility_mass_log",
    "coref_assignment_confidence_mean",
    "coref_direct_event_ratio",
    "coref_inherited_event_ratio",
    "coref_antecedent_distance_mean",
    "coref_document_responsibility_share",
)
_CATEGORY_FEATURE_NAMES = tuple(
    name
    for category in EVENT_CATEGORIES
    for name in (
        f"coref_{category}_event_present",
        f"coref_{category}_event_count_log",
        f"coref_{category}_direct_event_count_log",
        f"coref_{category}_inherited_event_count_log",
        f"coref_{category}_responsibility_mass_log",
    )
)
_ROLE_FEATURE_NAMES = tuple(
    name
    for role in DISCOURSE_ROLE_NAMES
    for name in (
        f"coref_role_{role}_event_present",
        f"coref_role_{role}_event_count_log",
    )
)
_CHAIN_FEATURE_NAMES = (
    "coref_method_diagnostic_present",
    "coref_method_carrier_present",
    "coref_carrier_bridge_present",
    "coref_complete_atomic_chain_present",
    "coref_own_evidence_event_ratio",
    "coref_adverse_discourse_event_ratio",
)
COREFERENCE_FEATURE_NAMES = (
    *_BASE_FEATURE_NAMES,
    *_CATEGORY_FEATURE_NAMES,
    *_ROLE_FEATURE_NAMES,
    *_CHAIN_FEATURE_NAMES,
)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", _clean(value))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _phrase_pattern(value: str) -> Pattern[str] | None:
    tokens = re.findall(r"[A-Za-z0-9]+", value)
    if not tokens:
        return None
    body = r"[^A-Za-z0-9]+".join(re.escape(token) for token in tokens)
    return re.compile(r"(?<![A-Za-z0-9])" + body + r"(?![A-Za-z0-9])", re.I)


@dataclass(frozen=True)
class TargetIdentity:
    """A clean target identifier and the names allowed to anchor it in text."""

    target_id: str
    name: str
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        target_id = _clean(self.target_id)
        name = _clean(self.name)
        if not target_id or not name:
            raise ValueError("target_id and name must be nonempty")
        aliases = tuple(
            dict.fromkeys(_clean(alias) for alias in self.aliases if _clean(alias))
        )
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)


@dataclass(frozen=True)
class EventConcept:
    """One frozen, category-specific concept used to form atomic events."""

    concept_id: str
    category: str
    patterns: tuple[Pattern[str], ...]


@dataclass(frozen=True)
class AtomicEvent:
    """One concept observed in one sentence with conservative responsibility."""

    paragraph_index: int
    sentence_index: int
    sentence_index_in_paragraph: int
    category: str
    concept_id: str
    matches: tuple[str, ...]
    discourse_roles: tuple[str, ...]
    target_id: str | None
    candidate_target_ids: tuple[str, ...]
    assignment: str
    responsibility: float
    antecedent_sentence_distance: int | None
    text: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "paragraph_index": self.paragraph_index,
            "sentence_index": self.sentence_index,
            "sentence_index_in_paragraph": self.sentence_index_in_paragraph,
            "category": self.category,
            "concept_id": self.concept_id,
            "matches": list(self.matches),
            "discourse_roles": list(self.discourse_roles),
            "target_id": self.target_id,
            "candidate_target_ids": list(self.candidate_target_ids),
            "assignment": self.assignment,
            "responsibility": self.responsibility,
            "antecedent_sentence_distance": self.antecedent_sentence_distance,
            "text": self.text,
        }


@dataclass(frozen=True)
class CoreferenceEventAnalysis:
    """Fixed-schema per-target features and optional auditable event records."""

    target_features: dict[str, dict[str, float]]
    events: tuple[AtomicEvent, ...]
    sentence_count: int

    @property
    def feature_names(self) -> tuple[str, ...]:
        return COREFERENCE_FEATURE_NAMES

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(COREFERENCE_FEATURE_NAMES),
            "target_features": self.target_features,
            "sentence_count": self.sentence_count,
            "events": [event.as_dict() for event in self.events],
        }


# Letter components are intentionally case-sensitive.  Treating the English
# article "a" as component A creates frequent false target assignments.
_QUALIFIER_PATTERNS: tuple[tuple[str, Pattern[str]], ...] = (
    (
        "component_a",
        re.compile(
            r"(?:\bA\s+(?i:components?|comp\.?|magneti[sz]ation)\b|"
            r"\b(?i:components?|comp\.?)\s+A\b)"
        ),
    ),
    (
        "component_b",
        re.compile(
            r"(?:\bB\s+(?i:components?|comp\.?|magneti[sz]ation)\b|"
            r"\b(?i:components?|comp\.?)\s+B\b)"
        ),
    ),
    (
        "component_c",
        re.compile(
            r"(?:\bC\s+(?i:components?|comp\.?|magneti[sz]ation)\b|"
            r"\b(?i:components?|comp\.?)\s+C\b)"
        ),
    ),
    (
        "high_temperature_component",
        re.compile(r"\b(?:HTC|high[- ]temperature\s+component)\b", re.I),
    ),
    (
        "low_temperature_component",
        re.compile(r"\b(?:LTC|low[- ]temperature\s+component)\b", re.I),
    ),
    (
        "normal_component",
        re.compile(
            r"(?:\bnormal(?:[- ]polarity)?\s+(?:component|directions?)\b|"
            r"\bN\s+(?i:component|comp\.?|directions?)\b)"
        ),
    ),
    (
        "reversed_component",
        re.compile(
            r"(?:\brevers(?:e|ed)(?:[- ]polarity)?\s+(?:component|directions?)\b|"
            r"\bR\s+(?i:component|comp\.?|directions?)\b)",
            re.I,
        ),
    ),
)

_QUALIFIER_SUFFIX_RE = re.compile(
    r"\s*[,;:/-]\s*(?:"
    r"[ABC]\s+(?:components?|comp\.?|magneti[sz]ation)|"
    r"(?:components?|comp\.?)\s+[ABC]|"
    r"HTC|LTC|"
    r"(?:high|low)[- ]temperature\s+component|"
    r"(?:normal|reverse|reversed)(?:[- ]polarity)?(?:\s+component)?|"
    r"[NR]\s+(?:component|comp\.?|directions?)"
    r")\s*$",
    re.I,
)

_RESTRICTED_REFERENCE_RE = re.compile(
    r"\b(?:these|those)\s+(?:samples?|specimens?)\b|"
    r"\b(?:this|that)\s+(?:unit|formation)\b",
    re.I,
)
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class _TargetProfile:
    identity: TargetIdentity
    full_patterns: tuple[Pattern[str], ...]
    material_pattern: Pattern[str] | None
    material_key: str
    qualifiers: frozenset[str]


@dataclass(frozen=True)
class _Sentence:
    paragraph_index: int
    index: int
    index_in_paragraph: int
    text: str
    roles: tuple[str, ...]


@dataclass(frozen=True)
class _MentionState:
    target_id: str | None
    candidates: tuple[str, ...]
    assignment: str
    responsibility: float
    antecedent_distance: int | None
    material_candidates: tuple[str, ...]


def _coerce_target(value: Any) -> TargetIdentity:
    if isinstance(value, TargetIdentity):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("targets must be TargetIdentity objects or narrow mappings")
    allowed = {"target_id", "name", "aliases"}
    extra = set(value) - allowed
    if extra:
        raise ValueError(f"unsupported target fields: {sorted(extra)}")
    aliases = value.get("aliases") or ()
    if isinstance(aliases, str) or not isinstance(aliases, Sequence):
        raise TypeError("target aliases must be a sequence of strings")
    return TargetIdentity(
        _clean(value.get("target_id")),
        _clean(value.get("name")),
        tuple(aliases),
    )


def _qualifiers(value: str) -> frozenset[str]:
    return frozenset(
        name for name, pattern in _QUALIFIER_PATTERNS if pattern.search(value)
    )


def _target_profiles(targets: Sequence[Any]) -> tuple[_TargetProfile, ...]:
    identities = tuple(_coerce_target(target) for target in targets)
    if not identities:
        raise ValueError("at least one target identity is required")
    ids = [target.target_id for target in identities]
    if len(set(ids)) != len(ids):
        raise ValueError("target_id values must be unique")

    profiles = []
    for target in identities:
        names = tuple(dict.fromkeys((target.name, *target.aliases)))
        patterns = tuple(
            pattern for name in names if (pattern := _phrase_pattern(name)) is not None
        )
        material = _QUALIFIER_SUFFIX_RE.sub("", target.name).strip(" ,;:/-")
        material_pattern = _phrase_pattern(material)
        profiles.append(
            _TargetProfile(
                target,
                patterns,
                material_pattern,
                _fold(material),
                _qualifiers(target.name),
            )
        )
    return tuple(profiles)


def _coerce_concepts(concepts: Sequence[Any]) -> tuple[EventConcept, ...]:
    output = []
    seen: set[tuple[str, str]] = set()
    for value in concepts:
        if isinstance(value, EventConcept):
            concept = value
        else:
            if isinstance(value, Mapping):
                concept_id = _clean(value.get("concept_id"))
                category = _clean(value.get("category")).casefold()
                raw_patterns = value.get("search_patterns") or value.get("patterns") or ()
            else:
                concept_id = _clean(getattr(value, "concept_id", ""))
                category = _clean(getattr(value, "category", "")).casefold()
                raw_patterns = getattr(value, "patterns", ())
            if isinstance(raw_patterns, (str, re.Pattern)):
                raw_patterns = (raw_patterns,)
            compiled = tuple(
                pattern
                if isinstance(pattern, re.Pattern)
                else re.compile(str(pattern), re.I)
                for pattern in raw_patterns
                if _clean(getattr(pattern, "pattern", pattern))
            )
            concept = EventConcept(concept_id, category, compiled)
        if not concept.concept_id or concept.category not in EVENT_CATEGORIES:
            raise ValueError("event concepts require an id and a supported category")
        if not concept.patterns:
            raise ValueError(f"event concept has no patterns: {concept.concept_id}")
        key = (concept.category, concept.concept_id)
        if key in seen:
            raise ValueError(f"duplicate event concept: {concept.category}/{concept.concept_id}")
        seen.add(key)
        output.append(concept)
    if not output:
        raise ValueError("at least one event concept is required")
    return tuple(output)


def _target_abbreviations(profiles: Sequence[_TargetProfile]) -> frozenset[str]:
    return frozenset(
        match.group(0).casefold()
        for profile in profiles
        for value in (profile.identity.name, *profile.identity.aliases)
        for match in re.finditer(r"\b[A-Za-z]{1,6}\.", value)
    )


def _sentences(
    text: str, profiles: Sequence[_TargetProfile]
) -> tuple[_Sentence, ...]:
    output = []
    global_index = 0
    target_abbreviations = _target_abbreviations(profiles)
    paragraphs = [part for part in _PARAGRAPH_BREAK_RE.split(text) if part.strip()]
    for paragraph_index, paragraph in enumerate(paragraphs):
        raw_annotations = list(annotate_discourse_window(paragraph))
        merged: list[tuple[str, tuple[str, ...]]] = []
        raw_index = 0
        while raw_index < len(raw_annotations):
            annotation = raw_annotations[raw_index]
            pieces = [annotation.text]
            roles = set(annotation.roles)
            while raw_index + 1 < len(raw_annotations):
                last_token = re.search(r"\b[A-Za-z]{1,6}\.$", pieces[-1].strip())
                if not last_token or last_token.group(0).casefold() not in target_abbreviations:
                    break
                raw_index += 1
                continuation = raw_annotations[raw_index]
                pieces.append(continuation.text)
                roles.update(continuation.roles)
            merged.append(
                (
                    " ".join(piece.strip() for piece in pieces),
                    tuple(role for role in DISCOURSE_ROLE_NAMES if role in roles),
                )
            )
            raw_index += 1

        for local_index, (sentence_text, roles) in enumerate(merged):
            output.append(
                _Sentence(
                    paragraph_index,
                    global_index,
                    local_index,
                    sentence_text,
                    roles,
                )
            )
            global_index += 1
    return tuple(output)


def _sentence_candidates(
    sentence: str, profiles: Sequence[_TargetProfile]
) -> tuple[set[str], set[str], set[str]]:
    full = {
        profile.identity.target_id
        for profile in profiles
        if any(pattern.search(sentence) for pattern in profile.full_patterns)
    }
    material = {
        profile.identity.target_id
        for profile in profiles
        if profile.material_pattern and profile.material_pattern.search(sentence)
    }
    qualifier_names = {
        name for name, pattern in _QUALIFIER_PATTERNS if pattern.search(sentence)
    }
    qualifier = {
        profile.identity.target_id
        for profile in profiles
        if profile.qualifiers & qualifier_names
    }
    return full, material, qualifier


def _nearest_prior_state(
    sentence: _Sentence,
    states: Sequence[_MentionState],
    sentences: Sequence[_Sentence],
) -> tuple[_MentionState | None, int | None]:
    start = max(0, sentence.index - MAX_ANTECEDENT_SENTENCES)
    for prior_index in range(sentence.index - 1, start - 1, -1):
        prior_sentence = sentences[prior_index]
        if prior_sentence.paragraph_index != sentence.paragraph_index:
            break
        state = states[prior_index]
        if state.target_id is None and state.candidates:
            return state, sentence.index - prior_index
        if state.target_id is not None:
            distance = sentence.index - prior_index
            if (
                state.assignment == "inherited_reference"
                and state.antecedent_distance is not None
            ):
                distance += state.antecedent_distance
            if distance <= MAX_ANTECEDENT_SENTENCES:
                return state, distance
    return None, None


def _resolve_sentence(
    sentence: _Sentence,
    profiles: Sequence[_TargetProfile],
    sentences: Sequence[_Sentence],
    states: Sequence[_MentionState],
) -> _MentionState:
    full, material, qualifier = _sentence_candidates(sentence.text, profiles)
    has_reference = bool(_RESTRICTED_REFERENCE_RE.search(sentence.text))

    if len(full) == 1:
        target_id = next(iter(full))
        return _MentionState(
            target_id,
            (target_id,),
            "direct_explicit",
            1.0,
            None,
            tuple(sorted(material)),
        )
    if len(full) > 1:
        return _MentionState(
            None,
            tuple(sorted(full)),
            "ambiguous_explicit",
            0.0,
            None,
            tuple(sorted(material)),
        )

    if material and qualifier:
        narrowed = material & qualifier
        if len(narrowed) == 1:
            target_id = next(iter(narrowed))
            return _MentionState(
                target_id,
                (target_id,),
                "direct_qualified",
                0.95,
                None,
                tuple(sorted(material)),
            )
        candidates = narrowed or material | qualifier
        return _MentionState(
            None,
            tuple(sorted(candidates)),
            "ambiguous_qualifier",
            0.0,
            None,
            tuple(sorted(material)),
        )

    if len(material) == 1:
        target_id = next(iter(material))
        return _MentionState(target_id, (target_id,), "direct_explicit", 0.95, None, (target_id,))
    if len(material) > 1 and not qualifier:
        return _MentionState(
            None,
            tuple(sorted(material)),
            "ambiguous_material",
            0.0,
            None,
            tuple(sorted(material)),
        )

    if qualifier:
        if len(qualifier) == 1:
            target_id = next(iter(qualifier))
            return _MentionState(target_id, (target_id,), "direct_qualified", 0.85, None, ())
        prior, distance = _nearest_prior_state(sentence, states, sentences)
        if prior is not None:
            context = set(prior.material_candidates or prior.candidates)
            narrowed = context & qualifier
            if len(narrowed) == 1:
                target_id = next(iter(narrowed))
                confidence = 0.80 if distance == 1 else 0.70
                return _MentionState(
                    target_id,
                    (target_id,),
                    "qualified_context",
                    confidence,
                    distance,
                    (),
                )
            if context:
                return _MentionState(
                    None,
                    tuple(sorted(context | qualifier)),
                    "blocked_ambiguous",
                    0.0,
                    distance,
                    (),
                )
        return _MentionState(None, tuple(sorted(qualifier)), "ambiguous_qualifier", 0.0, None, ())

    if has_reference:
        prior, distance = _nearest_prior_state(sentence, states, sentences)
        if prior is None:
            return _MentionState(None, (), "unresolved_reference", 0.0, None, ())
        if prior.target_id is None:
            return _MentionState(None, prior.candidates, "blocked_ambiguous", 0.0, distance, ())
        confidence = 0.78 if distance == 1 else 0.64
        return _MentionState(
            prior.target_id,
            (prior.target_id,),
            "inherited_reference",
            confidence,
            distance,
            (),
        )

    return _MentionState(None, (), "unassigned", 0.0, None, ())


def _feature_map(
    target_id: str,
    events: Sequence[AtomicEvent],
    resolved_event_count: int,
) -> dict[str, float]:
    assigned = [event for event in events if event.target_id == target_id]
    direct = [event for event in assigned if event.assignment in _DIRECT_ASSIGNMENTS]
    inherited = [event for event in assigned if event.assignment in _INHERITED_ASSIGNMENTS]
    assigned_sentence_ids = {event.sentence_index for event in assigned}
    direct_sentence_ids = {event.sentence_index for event in direct}
    qualified_sentence_ids = {
        event.sentence_index
        for event in assigned
        if event.assignment in {"direct_qualified", "qualified_context"}
    }
    inherited_sentence_ids = {event.sentence_index for event in inherited}
    responsibilities = [event.responsibility for event in assigned]
    distances = [
        float(event.antecedent_sentence_distance)
        for event in inherited
        if event.antecedent_sentence_distance is not None
    ]
    count = len(assigned)

    values: dict[str, float] = {
        "coref_assigned_sentence_count_log": _log1p(len(assigned_sentence_ids)),
        "coref_direct_sentence_count_log": _log1p(len(direct_sentence_ids)),
        "coref_qualified_sentence_count_log": _log1p(len(qualified_sentence_ids)),
        "coref_inherited_sentence_count_log": _log1p(len(inherited_sentence_ids)),
        "coref_assigned_event_count_log": _log1p(count),
        "coref_direct_event_count_log": _log1p(len(direct)),
        "coref_inherited_event_count_log": _log1p(len(inherited)),
        "coref_responsibility_mass_log": _log1p(sum(responsibilities)),
        "coref_assignment_confidence_mean": _mean(responsibilities),
        "coref_direct_event_ratio": len(direct) / count if count else 0.0,
        "coref_inherited_event_ratio": len(inherited) / count if count else 0.0,
        "coref_antecedent_distance_mean": _mean(distances),
        "coref_document_responsibility_share": (
            count / resolved_event_count if resolved_event_count else 0.0
        ),
    }

    categories_present = set()
    for category in EVENT_CATEGORIES:
        category_events = [event for event in assigned if event.category == category]
        category_direct = [
            event for event in category_events if event.assignment in _DIRECT_ASSIGNMENTS
        ]
        category_inherited = [
            event for event in category_events if event.assignment in _INHERITED_ASSIGNMENTS
        ]
        if category_events:
            categories_present.add(category)
        values[f"coref_{category}_event_present"] = float(bool(category_events))
        values[f"coref_{category}_event_count_log"] = _log1p(len(category_events))
        values[f"coref_{category}_direct_event_count_log"] = _log1p(len(category_direct))
        values[f"coref_{category}_inherited_event_count_log"] = _log1p(len(category_inherited))
        values[f"coref_{category}_responsibility_mass_log"] = _log1p(
            sum(event.responsibility for event in category_events)
        )

    for role in DISCOURSE_ROLE_NAMES:
        role_events = [event for event in assigned if role in event.discourse_roles]
        values[f"coref_role_{role}_event_present"] = float(bool(role_events))
        values[f"coref_role_{role}_event_count_log"] = _log1p(len(role_events))

    own_count = sum(
        bool(set(event.discourse_roles) & _OWN_EVIDENCE_ROLES)
        and not bool(set(event.discourse_roles) & _ADVERSE_ROLES)
        for event in assigned
    )
    adverse_count = sum(
        bool(set(event.discourse_roles) & _ADVERSE_ROLES) for event in assigned
    )
    values.update(
        {
            "coref_method_diagnostic_present": float(
                {"method", "diagnostic"} <= categories_present
            ),
            "coref_method_carrier_present": float(
                {"method", "carrier"} <= categories_present
            ),
            "coref_carrier_bridge_present": float(
                {"carrier", "bridge"} <= categories_present
            ),
            "coref_complete_atomic_chain_present": float(
                set(EVENT_CATEGORIES) <= categories_present
            ),
            "coref_own_evidence_event_ratio": own_count / count if count else 0.0,
            "coref_adverse_discourse_event_ratio": adverse_count / count if count else 0.0,
        }
    )
    return {name: float(values[name]) for name in COREFERENCE_FEATURE_NAMES}


def analyze_coreference_events(
    text: Any,
    targets: Sequence[Any],
    concepts: Sequence[Any],
    *,
    include_event_text: bool = False,
) -> CoreferenceEventAnalysis:
    """Resolve conservative target responsibility for original-text events.

    Each concept contributes at most one event per sentence.  Exact target
    names, material-plus-qualifier combinations, and unique component or
    polarity qualifiers are direct anchors.  Restricted demonstratives and
    ambiguous-unit component mentions can look back at most two sentences in
    the same paragraph.  Ambiguous assignments remain uncredited.
    """

    if not isinstance(text, str):
        raise TypeError("coreference-event input must be an original-text string")
    profiles = _target_profiles(targets)
    frozen_concepts = _coerce_concepts(concepts)
    sentences = _sentences(text, profiles)

    states: list[_MentionState] = []
    events: list[AtomicEvent] = []
    for sentence in sentences:
        state = _resolve_sentence(sentence, profiles, sentences, states)
        states.append(state)
        for concept in frozen_concepts:
            matches = tuple(
                dict.fromkeys(
                    match.group(0)
                    for pattern in concept.patterns
                    for match in pattern.finditer(sentence.text)
                )
            )
            if not matches:
                continue
            events.append(
                AtomicEvent(
                    paragraph_index=sentence.paragraph_index,
                    sentence_index=sentence.index,
                    sentence_index_in_paragraph=sentence.index_in_paragraph,
                    category=concept.category,
                    concept_id=concept.concept_id,
                    matches=matches,
                    discourse_roles=sentence.roles,
                    target_id=state.target_id,
                    candidate_target_ids=state.candidates,
                    assignment=state.assignment,
                    responsibility=state.responsibility if state.target_id else 0.0,
                    antecedent_sentence_distance=state.antecedent_distance,
                    text=sentence.text if include_event_text else None,
                )
            )

    resolved_count = sum(event.target_id is not None for event in events)
    features = {
        profile.identity.target_id: _feature_map(
            profile.identity.target_id, events, resolved_count
        )
        for profile in profiles
    }
    return CoreferenceEventAnalysis(features, tuple(events), len(sentences))


__all__ = [
    "COREFERENCE_FEATURE_NAMES",
    "EVENT_CATEGORIES",
    "MAX_ANTECEDENT_SENTENCES",
    "AtomicEvent",
    "CoreferenceEventAnalysis",
    "EventConcept",
    "TargetIdentity",
    "analyze_coreference_events",
]
