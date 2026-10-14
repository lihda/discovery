#!/usr/bin/env python3
"""Label-free sentence-leading target attribution for paleomagnetic R3 evidence.

The resolver models a recurring writing convention in paleomagnetic papers:
an evidence passage commonly starts with a rock-unit name and continues with
sentences led by local references such as "these samples" or "the coercivity
curves".  Only target identity, original page text, and frozen R3 concepts are
accepted.  Database evidence fields, labels, and predictions are deliberately
outside this module's API.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import re
import unicodedata
from typing import Any, Mapping, Sequence

try:
    from .r3_trace_discourse_roles import annotate_discourse_window
except ImportError:
    from r3_trace_discourse_roles import annotate_discourse_window


GENERIC_UNIT_WORDS = frozenset(
    {
        "area",
        "basic",
        "belt",
        "block",
        "combined",
        "complex",
        "conglomerate",
        "country",
        "dacite",
        "dyke",
        "dykes",
        "formation",
        "formations",
        "gabbro",
        "granite",
        "group",
        "intrusion",
        "intrusions",
        "intrusives",
        "lava",
        "lavas",
        "magmatic",
        "massif",
        "member",
        "metamorphic",
        "pluton",
        "porphyrites",
        "quartzite",
        "range",
        "rhyolite",
        "rock",
        "rocks",
        "sediment",
        "sediments",
        "sequence",
        "sill",
        "schists",
        "supergroup",
        "tuff",
        "unit",
        "volcanics",
    }
)
_STOPWORDS = GENERIC_UNIT_WORDS | frozenset(
    {
        "and",
        "de",
        "del",
        "des",
        "east",
        "frio",
        "hill",
        "las",
        "lower",
        "middle",
        "mount",
        "north",
        "peninsula",
        "sierra",
        "south",
        "upper",
        "west",
    }
)
_UNIT_SUFFIX_RE = re.compile(
    r"\b(formations?|groups?|supergroup|complex|sediments?|volcanics?|"
    r"intrusions?|intrusives?|pluton|massif|members?|sequence|dykes?|sill|"
    r"gabbro|granites?|lavas?|rhyolites?|quartzite|conglomerate|tuffs?|"
    r"schists?|rocks?|porphyrites?)\b",
    re.I,
)
_REFERENCE_LEAD_RE = re.compile(
    r"^(?:"
    r"it|they|its|their|"
    r"this|that|these|those|"
    r"this\s+(?:unit|formation|group|complex|rock|material|sample|specimen)|"
    r"that\s+(?:unit|formation|group|complex|rock|material|sample|specimen)|"
    r"these\s+(?:rocks?|samples?|specimens?|units?|materials?|formations?)|"
    r"those\s+(?:rocks?|samples?|specimens?|units?|materials?|formations?)|"
    r"the\s+(?:samples?|specimens?|rocks?|unit|formation|group|complex|"
    r"material|remanence|magneti[sz]ation|magnetic\s+minerals?|carriers?|"
    r"blocking\s+temperatures?|curie\s+temperatures?|coercivit(?:y|ies)|curves?|results?|"
    r"measurements?|experiments?|analyses?|stability\s+spectrum|"
    r"(?:very\s+)?(?:small|large|fine|coarse)\s+grains?)|"
    r"(?:electron\s+microprobe|microscopic|petrographic|thermomagnetic|"
    r"rock[- ]magnetic)\s+analys(?:is|es)\s+of\s+"
    r"(?:(?:one|some|all)\s+of\s+)?(?:these|those|the)\s+|"
    r"(?:measurements?|analyses?|observations?)\s+of\s+"
    r"(?:(?:one|some|all)\s+of\s+)?(?:these|those|the)\s+"
    r")\b",
    re.I,
)
_FORMER_LATTER_RE = re.compile(r"^the\s+(former|latter)\b", re.I)
_LEAD_PREFIX_RE = re.compile(
    r"^(?:(?:in|for|within|from|of)\s+(?:the\s+)?|"
    r"(?:samples?|specimens?|rocks?|materials?)\s+(?:from|of|in)\s+(?:the\s+)?|"
    r"the\s+)",
    re.I,
)
_REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:references|bibliography|literature cited|works cited)\s*:?[ \t]*$"
)
_ADVERSE_ROLES = frozenset(("negated", "prospective", "conditional"))
_MAX_FOCUS_DISTANCE = 3


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def fold_text(value: Any) -> str:
    text = clean(value)
    # Some PDF extractors separate a combining mark from the base word:
    # "Va ¨sterbotten", "Ja ¨mtland", but also "Ulvo ¨ Complex".  Join only
    # when the mark interrupts a lowercase word; otherwise remove the mark and
    # retain the word boundary.
    text = re.sub(
        r"(?<=[A-Za-z])\s*[¨´`^~˚]\s*(?=[a-z])",
        "",
        text,
    )
    text = re.sub(r"[¨´`^~˚]", " ", text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _tokens(value: Any) -> tuple[str, ...]:
    return tuple(fold_text(value).split())


def _singular_suffix(value: str) -> str:
    mapping = {
        "formations": "formation",
        "groups": "group",
        "intrusions": "intrusion",
        "intrusives": "intrusive",
        "sediments": "sediment",
        "volcanics": "volcanic",
        "dykes": "dyke",
        "lavas": "lava",
        "rhyolites": "rhyolite",
        "granites": "granite",
        "tuffs": "tuff",
        "schists": "schist",
        "rocks": "rock",
        "members": "member",
        "porphyrites": "porphyrite",
    }
    return mapping.get(value.casefold(), value)


def _raw_aliases(name: str) -> set[str]:
    """Generate conservative aliases from one ROCKNAME without corpus outcomes."""

    source = re.sub(r"\s+", " ", clean(name))
    aliases = {source}
    parenthetical = tuple(re.findall(r"\(([^()]{2,80})\)", source))
    without_parenthetical = re.sub(r"\s*\([^()]*\)", "", source).strip(" ,")
    if without_parenthetical:
        aliases.add(without_parenthetical)
    aliases.update(parenthetical)

    suffixes = list(_UNIT_SUFFIX_RE.finditer(without_parenthetical))
    final_suffix = suffixes[-1].group(1) if suffixes else ""
    if final_suffix and re.search(r"\b(?:and|&)\b|,", without_parenthetical, re.I):
        prefix = without_parenthetical[: suffixes[-1].start()].strip(" ,")
        parts = [
            part.strip(" ,")
            for part in re.split(r"\s*(?:,|\band\b|&)\s*", prefix, flags=re.I)
            if part.strip(" ,")
        ]
        if 1 < len(parts) <= 4:
            aliases.update(f"{part} {_singular_suffix(final_suffix)}" for part in parts)
            aliases.update(parts)

    comma_parts = [part.strip() for part in source.split(",") if part.strip()]
    if 1 < len(comma_parts) <= 3:
        aliases.update(comma_parts)

    # Database identities often use "Dykes A and C", whereas the paper uses
    # "Group A dikes".  The A/B/C designator is an identity qualifier, so this
    # transformation does not require labels or any evidence-table field.
    qualifier = re.search(
        r"\b(?:dykes?|dikes?|groups?)\s+([A-E])(?:\s+(?:and|&)\s+([A-E]))?\b",
        source,
        re.I,
    )
    if qualifier:
        for letter in (qualifier.group(1), qualifier.group(2)):
            if not letter:
                continue
            aliases.update(
                {
                    f"group {letter}",
                    f"group {letter} dikes",
                    f"group {letter} dykes",
                    f"{letter} dikes",
                    f"{letter} dykes",
                }
            )

    # A unit name without its generic lithology suffix is often the paper's
    # short form ("the Duffer", "Takiyuak", "Matozero").
    for alias in tuple(aliases):
        stripped = _UNIT_SUFFIX_RE.sub("", alias).strip(" ,-")
        if stripped and len(_tokens(stripped)) <= 5:
            aliases.add(stripped)

    return {fold_text(alias) for alias in aliases if len(fold_text(alias)) >= 3}


@dataclass(frozen=True)
class TargetGroup:
    group_id: str
    name: str
    target_ids: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class LeadEvent:
    page: int
    sentence_index: int
    group_id: str
    assignment: str
    category: str
    concept_id: str
    trusted: bool
    roles: tuple[str, ...]
    text: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "page": self.page,
            "sentence_index": self.sentence_index,
            "group_id": self.group_id,
            "assignment": self.assignment,
            "category": self.category,
            "concept_id": self.concept_id,
            "trusted": self.trusted,
            "roles": list(self.roles),
            "text": self.text,
        }


@dataclass(frozen=True)
class SentenceLeadAnalysis:
    groups: tuple[TargetGroup, ...]
    group_features: dict[str, dict[str, Any]]
    events: tuple[LeadEvent, ...]
    sentence_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "groups": [
                {
                    "group_id": group.group_id,
                    "name": group.name,
                    "target_ids": list(group.target_ids),
                    "aliases": list(group.aliases),
                }
                for group in self.groups
            ],
            "group_features": self.group_features,
            "events": [event.as_dict() for event in self.events],
            "sentence_count": self.sentence_count,
        }


def build_target_groups(targets: Sequence[Mapping[str, Any]]) -> tuple[TargetGroup, ...]:
    """Group indistinguishable RESULTNO rows and derive paper-local aliases."""

    if not targets:
        raise ValueError("at least one target is required")
    by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for target in targets:
        extra = set(target) - {"target_id", "name"}
        if extra:
            raise ValueError(f"unsupported target fields: {sorted(extra)}")
        target_id = clean(target.get("target_id"))
        name = clean(target.get("name"))
        if not target_id or not name:
            raise ValueError("target_id and name are required")
        by_name[fold_text(name)].append((target_id, name))

    raw_by_group: dict[str, set[str]] = {}
    names: dict[str, str] = {}
    ids: dict[str, tuple[str, ...]] = {}
    for key, values in by_name.items():
        raw_by_group[key] = _raw_aliases(values[0][1])
        names[key] = values[0][1]
        ids[key] = tuple(value[0] for value in values)

    token_owners: dict[str, set[str]] = defaultdict(set)
    for key, aliases in raw_by_group.items():
        for alias in aliases:
            for token in alias.split():
                if len(token) >= 5 and token not in _STOPWORDS:
                    token_owners[token].add(key)
    for token, owners in token_owners.items():
        if len(owners) == 1:
            raw_by_group[next(iter(owners))].add(token)

    groups = []
    for index, key in enumerate(by_name):
        aliases = tuple(
            sorted(raw_by_group[key], key=lambda value: (-len(value.split()), -len(value), value))
        )
        groups.append(TargetGroup(f"g{index:02d}", names[key], ids[key], aliases))
    return tuple(groups)


def _prepare_page_text(value: Any) -> str:
    text = clean(value).replace("\r\n", "\n").replace("\r", "\n")
    reference = _REFERENCE_HEADING_RE.search(text)
    if reference:
        text = text[: reference.start()]
    text = re.sub(r"(?<=[A-Za-z])-[ \t]*\n[ \t]*(?=[a-z])", "", text)
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def _concept_matches(sentence: str, concepts: Sequence[Any]) -> list[tuple[str, str]]:
    output = []
    for concept in concepts:
        category = clean(getattr(concept, "category", "")).casefold()
        concept_id = clean(getattr(concept, "concept_id", ""))
        patterns = getattr(concept, "patterns", ())
        if category not in {"method", "diagnostic", "carrier", "bridge"}:
            continue
        if any(pattern.search(sentence) for pattern in patterns):
            output.append((category, concept_id))
    return output


def _leading_alias_candidates(
    sentence: str, groups: Sequence[TargetGroup]
) -> tuple[str, ...]:
    folded = fold_text(sentence)
    variants = {folded}
    prefix = _LEAD_PREFIX_RE.match(folded)
    if prefix:
        variants.add(folded[prefix.end() :])
    matches: list[tuple[int, str]] = []
    for group in groups:
        for alias in group.aliases:
            if any(value == alias or value.startswith(alias + " ") for value in variants):
                matches.append((len(alias.split()), group.group_id))
                break
    if not matches:
        return ()
    maximum = max(length for length, _ in matches)
    return tuple(sorted({group_id for length, group_id in matches if length == maximum}))


def _mentioned_groups(sentence: str, groups: Sequence[TargetGroup]) -> tuple[str, ...]:
    folded = " " + fold_text(sentence) + " "
    mentioned = set()
    for group in groups:
        for alias in group.aliases:
            if len(alias.split()) < 2 and len(alias) < 6:
                continue
            if f" {alias} " in folded:
                mentioned.add(group.group_id)
                break
    return tuple(sorted(mentioned))


def _event_is_trusted(roles: Sequence[str]) -> bool:
    role_set = set(roles)
    if role_set & _ADVERSE_ROLES:
        return False
    if "background_citation_prior_work" in role_set and not (
        {"performed_execution", "reported_result"} & role_set
    ):
        return False
    return True


def analyze_sentence_lead_attribution(
    pages: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    concepts: Sequence[Any],
) -> SentenceLeadAnalysis:
    """Attribute frozen R3 concepts using sentence-leading target focus."""

    groups = build_target_groups(targets)
    group_by_id = {group.group_id: group for group in groups}
    anchor_counts = Counter()
    ambiguous_anchor_counts = Counter()
    reference_counts = Counter()
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    trusted_category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    events: list[LeadEvent] = []
    sentence_total = 0

    for page_offset, raw_page in enumerate(pages, start=1):
        if not isinstance(raw_page, Mapping):
            continue
        page_number = int(raw_page.get("page") or page_offset)
        annotations = annotate_discourse_window(_prepare_page_text(raw_page.get("text")))
        focus: str | None = None
        focus_origin: int | None = None
        pair_focus: tuple[str, ...] = ()
        for local_index, annotation in enumerate(annotations):
            sentence_total += 1
            text = annotation.text.strip()
            lead = _leading_alias_candidates(text, groups)
            mentioned = _mentioned_groups(text, groups)
            assignment = "unassigned"
            resolved: str | None = None

            if len(lead) == 1:
                resolved = lead[0]
                focus = resolved
                focus_origin = local_index
                pair_focus = ()
                anchor_counts[resolved] += 1
                assignment = "direct_sentence_lead"
            elif len(lead) > 1:
                focus = None
                focus_origin = local_index
                pair_focus = lead
                for group_id in lead:
                    ambiguous_anchor_counts[group_id] += 1
                assignment = "ambiguous_sentence_lead"
            else:
                folded = fold_text(text)
                former_latter = _FORMER_LATTER_RE.match(folded)
                if former_latter and len(pair_focus) == 2:
                    resolved = pair_focus[0 if former_latter.group(1).casefold() == "former" else 1]
                    assignment = "former_latter_reference"
                elif (
                    _REFERENCE_LEAD_RE.match(folded)
                    and focus is not None
                    and focus_origin is not None
                    and local_index - focus_origin <= _MAX_FOCUS_DISTANCE
                ):
                    resolved = focus
                    assignment = "inherited_sentence_lead_reference"
                elif mentioned:
                    # A sibling mention away from sentence start blocks local focus;
                    # it is never promoted into a positive assignment.
                    focus = None
                    focus_origin = None
                    pair_focus = mentioned if len(mentioned) == 2 else ()
                    assignment = "blocked_nonleading_mention"

            if resolved is not None and assignment != "direct_sentence_lead":
                reference_counts[resolved] += 1
            matches = _concept_matches(text, concepts)
            if resolved is None or not matches:
                continue
            roles = tuple(annotation.roles)
            trusted = _event_is_trusted(roles)
            for category, concept_id in matches:
                category_counts[resolved][category] += 1
                if trusted:
                    trusted_category_counts[resolved][category] += 1
                events.append(
                    LeadEvent(
                        page_number,
                        local_index,
                        resolved,
                        assignment,
                        category,
                        concept_id,
                        trusted,
                        roles,
                        text,
                    )
                )

    features: dict[str, dict[str, Any]] = {}
    for group_id, group in group_by_id.items():
        trusted = trusted_category_counts[group_id]
        trusted_event_count = sum(trusted.values())
        method_support = trusted["method"] > 0
        diagnostic_carrier_support = trusted["diagnostic"] > 0 and trusted["carrier"] > 0
        strong_support = method_support or diagnostic_carrier_support
        features[group_id] = {
            "target_ids": list(group.target_ids),
            "explicit_anchor_count": int(anchor_counts[group_id]),
            "ambiguous_anchor_count": int(ambiguous_anchor_counts[group_id]),
            "reference_sentence_count": int(reference_counts[group_id]),
            "category_event_counts": {
                category: int(category_counts[group_id][category])
                for category in ("method", "diagnostic", "carrier", "bridge")
            },
            "trusted_category_event_counts": {
                category: int(trusted[category])
                for category in ("method", "diagnostic", "carrier", "bridge")
            },
            "strong_support": bool(strong_support),
            "carrier_only": bool(
                trusted["carrier"] > 0 and not method_support and not diagnostic_carrier_support
            ),
            "candidate_signal": (
                1.0
                if strong_support
                else 0.0
                if trusted_event_count > 0
                else None
            ),
        }
    return SentenceLeadAnalysis(groups, features, tuple(events), sentence_total)


__all__ = [
    "LeadEvent",
    "SentenceLeadAnalysis",
    "TargetGroup",
    "analyze_sentence_lead_attribution",
    "build_target_groups",
    "fold_text",
]
