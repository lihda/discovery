#!/usr/bin/env python3
"""Label-blind paleomagnetic entity inventory and sibling assignment.

The public boundary is intentionally small::

    build_entity_graph({target_id: rockname}, chunks) -> dict

Chunks contain only ``chunk_id``, ``kind``, ``text``, and optional
``parent_id``.  No criterion label, prediction, metric, or PMAGRESULT field is
accepted.  The implementation uses identity-derived aliases and explicit
parenthetical abbreviations; it never learns aliases from evaluated outcomes.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "r3_paleomag_entity_graph_v1"
ALLOWED_CHUNK_FIELDS = frozenset(("chunk_id", "kind", "text", "parent_id"))
ALLOWED_CHUNK_KINDS = frozenset(
    ("sentence", "paragraph", "section", "table", "caption")
)
FORBIDDEN_FIELD_FRAGMENTS = (
    "pmagresult",
    "rockmag",
    "label",
    "prediction",
    "metric",
)

GENERIC_TOKENS = frozenset(
    (
        "and", "andesite", "andesites", "amphibolite", "area", "basalt",
        "basalts", "basin", "bed",
        "beds", "belt", "block", "breccia", "carbonate", "carbonates",
        "complex", "diabase", "diorite", "direction", "directions",
        "dolerite", "dolerites", "dolomite", "dyke", "dykes", "east",
        "formation", "gabbro", "gneiss", "granite", "group", "intrusion",
        "kimberlite", "kimberlites", "lava", "lavas", "limestone",
        "lithology", "locality", "massif", "member", "mudstone", "north", "of",
        "picrite", "pluton", "porphyrite", "porphyrites", "province",
        "quartzite", "redbed", "redbeds", "rhyolite", "rhyolites", "rock",
        "rocks", "schist", "schists", "sediment", "sediments", "sequence",
        "series", "sill", "sills", "site", "sites", "south", "suite",
        "supergroup", "the", "tuff", "tuffs", "unit", "volcanic",
        "volcanics", "west",
    )
)
UNIT_WORDS = frozenset(
    (
        "basin", "belt", "block", "complex", "formation", "group",
        "intrusion", "massif", "member", "pluton", "province", "sequence",
        "locality", "series", "site", "suite", "supergroup", "unit",
    )
)
LITHOLOGY_WORDS = frozenset(
    (
        "amphibolite", "andesite", "andesites", "basalt", "basalts",
        "breccia", "carbonate", "carbonates", "diabase", "diorite",
        "dolerite", "dolerites", "dolomite", "dyke", "dykes", "gabbro",
        "gneiss", "granite", "kimberlite", "kimberlites", "lava", "lavas",
        "limestone", "mudstone", "picrite", "porphyrite", "porphyrites",
        "quartzite", "redbed", "redbeds", "rhyolite", "rhyolites", "rock",
        "rocks", "schist", "schists", "sediment", "sediments", "sill",
        "sills", "tuff", "tuffs", "volcanic", "volcanics",
    )
)
ABBREVIATIONS = {
    "fm": "formation",
    "fmn": "formation",
    "gp": "group",
    "grp": "group",
    "mbr": "member",
}
UNIT_SHORT = {
    "formation": "fm",
    "group": "gp",
    "member": "mbr",
}

FEATURE_NAMES = (
    "entity_inventory_alias_count_log",
    "entity_inventory_unique_alias_count_log",
    "entity_inventory_conflict_alias_count_log",
    "entity_duplicate_identity",
    "entity_chunk_count_log",
    "entity_exact_chunk_count_log",
    "entity_alias_chunk_count_log",
    "entity_cluster_chunk_count_log",
    "entity_component_match_count_log",
    "entity_polarity_match_count_log",
    "entity_positive_chunk_count_log",
    "entity_target_mass_mean",
    "entity_target_mass_max",
    "entity_target_mass_sum_log",
    "entity_winner_count_log",
    "entity_margin_mean",
    "entity_entropy_mean",
    "entity_swap_delta_mean",
    "entity_unresolved_mass_mean",
    "entity_global_mass_mean",
)

_COMPONENT_PATTERNS = (
    ("a", re.compile(r"(?:\bA\s+(?i:components?|comp\.?|magneti[sz]ation)\b|\b(?i:components?|comp\.?)\s+A\b)")),
    ("b", re.compile(r"(?:\bB\s+(?i:components?|comp\.?|magneti[sz]ation)\b|\b(?i:components?|comp\.?)\s+B\b)")),
    ("c", re.compile(r"(?:\bC\s+(?i:components?|comp\.?|magneti[sz]ation)\b|\b(?i:components?|comp\.?)\s+C\b)")),
    ("htc", re.compile(r"\b(?:HTC|high[- ]temperature\s+component)\b", re.I)),
    ("ltc", re.compile(r"\b(?:LTC|low[- ]temperature\s+component)\b", re.I)),
    ("chrm", re.compile(r"\b(?:ChRM|characteristic\s+remanent\s+magneti[sz]ation)\b", re.I)),
)
_POLARITY_PATTERNS = (
    ("n", re.compile(r"\b(?:normal\s+(?:polarity|component|direction)|N[- ](?:polarity|component|direction))\b", re.I)),
    ("r", re.compile(r"\b(?:(?:reverse|reversed)\s+(?:polarity|component|direction)|R[- ](?:polarity|component|direction))\b", re.I)),
)
_TARGET_TRAILING_COMPONENT = re.compile(
    r"(?:[,;/:-]\s*)(A|B|C|HTC|LTC|ChRM)\s*$", re.I
)
_TARGET_TRAILING_POLARITY = re.compile(
    r"(?:[,;/:-]\s*)(N|R|normal|reverse|reversed)\s*$", re.I
)
_QUALIFIER_REMOVALS = (
    re.compile(r"\b(?:A|B|C)\s+(?:components?|comp\.?|magneti[sz]ation)\b"),
    re.compile(r"\b(?:components?|comp\.?)\s+(?:A|B|C)\b", re.I),
    re.compile(r"\b(?:HTC|LTC|ChRM)\b", re.I),
    re.compile(r"\b(?:high|low)[- ]temperature\s+component\b", re.I),
    re.compile(r"\bcharacteristic\s+remanent\s+magneti[sz]ation\b", re.I),
    re.compile(r"\b(?:normal|reverse|reversed)\s+(?:polarity|component|direction)\b", re.I),
    re.compile(r"\b[NR][- ](?:polarity|component|direction)\b", re.I),
)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _fold(value: Any) -> str:
    text = _clean(value).translate(
        str.maketrans(
            {
                "¨": "",
                "´": "",
                "`": "",
                "ı": "i",
                "ø": "o",
                "Ø": "O",
                "ł": "l",
                "Ł": "L",
                "đ": "d",
                "Đ": "D",
                "ß": "ss",
                "æ": "ae",
                "Æ": "AE",
                "œ": "oe",
                "Œ": "OE",
            }
        )
    )
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = text.encode("ascii", "ignore").decode("ascii").casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _normalize_tokens(value: Any) -> str:
    return " ".join(ABBREVIATIONS.get(token, token) for token in _fold(value).split())


def _qualifiers(value: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    components = {name for name, pattern in _COMPONENT_PATTERNS if pattern.search(value)}
    polarities = {name for name, pattern in _POLARITY_PATTERNS if pattern.search(value)}
    trailing = _TARGET_TRAILING_COMPONENT.search(value)
    if trailing:
        components.add(trailing.group(1).casefold())
    trailing = _TARGET_TRAILING_POLARITY.search(value)
    if trailing:
        polarity = trailing.group(1).casefold()
        polarities.add("n" if polarity in ("n", "normal") else "r")
    return tuple(sorted(components)), tuple(sorted(polarities))


def _material_name(value: str) -> str:
    text = value
    for pattern in _QUALIFIER_REMOVALS:
        text = pattern.sub(" ", text)
    text = _TARGET_TRAILING_COMPONENT.sub(" ", text)
    text = _TARGET_TRAILING_POLARITY.sub(" ", text)
    return _normalize_tokens(text).strip()


def _is_generic(alias: str) -> bool:
    tokens = alias.split()
    return not tokens or all(token in GENERIC_TOKENS or token.isdigit() for token in tokens)


def _initialism(full_name: str) -> str:
    ignored = {"and", "of", "the"}
    return "".join(token[0] for token in full_name.split() if token not in ignored)


def _phrase_pattern(phrase: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", phrase)
    if not tokens:
        return r"(?!)"
    return r"(?<![A-Za-z0-9])" + r"[^A-Za-z0-9]+".join(
        re.escape(token) for token in tokens
    ) + r"(?![A-Za-z0-9])"


def _contains_phrase(folded_text: str, phrase: str) -> bool:
    return bool(
        re.search(
            r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])",
            folded_text,
        )
    )


def _edit_distance_at_most_one(left: str, right: str) -> bool:
    if left == right:
        return True
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) > len(right):
        left, right = right, left
    if len(left) == len(right):
        return sum(a != b for a, b in zip(left, right)) <= 1
    offset = 0
    differences = 0
    for index, character in enumerate(right):
        if index - offset < len(left) and left[index - offset] == character:
            continue
        differences += 1
        offset += 1
        if differences > 1:
            return False
    return True


def _cluster_score(material: str, folded_text: str) -> float:
    target_tokens = material.split()
    text_tokens = folded_text.split()
    specific = [
        token for token in target_tokens if token not in GENERIC_TOKENS and not token.isdigit()
    ]
    if not specific:
        return 0.0

    def present(token: str) -> bool:
        if token in text_tokens:
            return True
        return len(token) >= 6 and any(
            len(candidate) >= 5 and _edit_distance_at_most_one(token, candidate)
            for candidate in text_tokens
        )

    present_specific = [token for token in specific if present(token)]
    if len(present_specific) != len(specific):
        return 0.0
    discriminators = [
        token
        for token in target_tokens
        if token in UNIT_WORDS.union(LITHOLOGY_WORDS) or token.isdigit()
    ]
    discriminator_hits = sum(token in text_tokens for token in discriminators)
    locality_code = re.search(r"\b(?:locality|site)\s+(\d+[a-z]?)\b", material)
    if locality_code and re.search(
        rf"\b(?:locality|site)\s+{re.escape(locality_code.group(1))}\b", folded_text
    ):
        return 3.0
    if discriminator_hits:
        return 2.4 + 0.25 * min(2, discriminator_hits)
    if len(specific) >= 2:
        return 1.7
    return 0.0


def _validate_inputs(
    targets: Mapping[str, str], chunks: Sequence[Mapping[str, Any]]
) -> tuple[list[tuple[str, str]], list[dict[str, str | None]]]:
    if not isinstance(targets, Mapping) or not targets:
        raise TypeError("targets must be a nonempty {target_id: rockname} mapping")
    target_rows: list[tuple[str, str]] = []
    for target_id, rockname in targets.items():
        if not isinstance(target_id, str) or not target_id.strip():
            raise ValueError("every target_id must be a nonempty string")
        if not isinstance(rockname, str) or not rockname.strip():
            raise ValueError("every rockname must be a nonempty string")
        target_rows.append((target_id.strip(), rockname.strip()))
    if len({target_id for target_id, _ in target_rows}) != len(target_rows):
        raise ValueError("target_id values must be unique")

    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)):
        raise TypeError("chunks must be a sequence of chunk mappings")
    chunk_rows: list[dict[str, str | None]] = []
    seen: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, Mapping):
            raise TypeError(f"chunk {index} must be a mapping")
        keys = set(chunk)
        normalized_keys = {re.sub(r"[^a-z0-9]+", "", str(key).casefold()) for key in keys}
        if any(
            fragment in key
            for key in normalized_keys
            for fragment in FORBIDDEN_FIELD_FRAGMENTS
        ) or normalized_keys.intersection(("r3", "r5", "r7")):
            raise ValueError(f"chunk {index} contains a forbidden evaluation field")
        unsupported = keys - ALLOWED_CHUNK_FIELDS
        if unsupported:
            raise ValueError(f"chunk {index} has unsupported fields: {sorted(unsupported)}")
        chunk_id = _clean(chunk.get("chunk_id"))
        kind = _clean(chunk.get("kind")).casefold()
        text = chunk.get("text")
        parent_id = _clean(chunk.get("parent_id")) or None
        if not chunk_id:
            raise ValueError(f"chunk {index} has an empty chunk_id")
        if chunk_id in seen:
            raise ValueError(f"duplicate chunk_id: {chunk_id}")
        if kind not in ALLOWED_CHUNK_KINDS:
            raise ValueError(f"chunk {chunk_id} has unsupported kind: {kind}")
        if not isinstance(text, str):
            raise TypeError(f"chunk {chunk_id} text must be an original-text string")
        seen.add(chunk_id)
        chunk_rows.append(
            {"chunk_id": chunk_id, "kind": kind, "text": text, "parent_id": parent_id}
        )
    return target_rows, chunk_rows


def _profile(target_id: str, rockname: str) -> dict[str, Any]:
    components, polarities = _qualifiers(rockname)
    material = _material_name(rockname) or _normalize_tokens(rockname)
    normalized_full = _normalize_tokens(rockname)
    qualifier_key = ",".join((*components, *polarities))
    canonical_identity = material + ("::" + qualifier_key if qualifier_key else "")
    candidates: dict[str, set[str]] = defaultdict(set)
    candidates[_fold(rockname)].add("identity_exact")
    candidates[normalized_full].add("identity_normalized")
    candidates[material].add("identity_material")
    tokens = material.split()
    if tokens and tokens[-1] in UNIT_SHORT:
        abbreviated = " ".join((*tokens[:-1], UNIT_SHORT[tokens[-1]]))
        candidates[abbreviated].add("identity_unit_abbreviation")
    if tokens and tokens[-1] in UNIT_WORDS.union(LITHOLOGY_WORDS):
        stem = " ".join(tokens[:-1])
        if stem:
            provenance = (
                "identity_unit_stem"
                if tokens[-1] in UNIT_WORDS
                else "identity_lithology_stem"
            )
            candidates[stem].add(provenance)
    if any(token in UNIT_WORDS for token in tokens):
        candidates[material].add("identity_geological_unit")
    if any(token in LITHOLOGY_WORDS for token in tokens):
        candidates[material].add("identity_lithology")
    raw_fragments = re.split(r"[,;()]|\s+\band\b\s+", rockname, flags=re.I)
    for raw_fragment in raw_fragments:
        fragment = _normalize_tokens(raw_fragment)
        if not fragment or fragment == material or _is_generic(fragment):
            continue
        fragment_tokens = fragment.split()
        if not any(token not in GENERIC_TOKENS and not token.isdigit() for token in fragment_tokens):
            continue
        candidates[fragment].add("identity_composite_fragment")
        if fragment_tokens[-1] in UNIT_WORDS.union(LITHOLOGY_WORDS):
            stem = " ".join(fragment_tokens[:-1])
            if stem and not _is_generic(stem):
                candidates[stem].add("identity_fragment_stem")
    return {
        "target_id": target_id,
        "rockname": rockname,
        "canonical_identity": canonical_identity,
        "canonical_material": material,
        "components": components,
        "polarities": polarities,
        "candidates": candidates,
    }


def _mine_abbreviations(
    profiles: Sequence[Mapping[str, Any]], chunks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, tuple[str, tuple[str, ...]]], list[dict[str, Any]]]:
    material_names = sorted({str(profile["canonical_material"]) for profile in profiles})
    observed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for chunk in chunks:
        text = str(chunk["text"])
        chunk_id = str(chunk["chunk_id"])
        for material in material_names:
            if len(material.split()) < 2:
                continue
            full = _phrase_pattern(material)
            full_case_insensitive = "(?i:" + full + ")"
            patterns = (
                re.compile(
                    full_case_insensitive
                    + r"\s*\(\s*([A-Z][A-Z0-9-]{1,9})\s*\)"
                ),
                re.compile(
                    r"(?<![A-Za-z0-9])([A-Z][A-Z0-9-]{1,9})\s*\(\s*"
                    + full_case_insensitive
                    + r"\s*\)"
                ),
            )
            expected = _initialism(material)
            for pattern in patterns:
                for match in pattern.finditer(text):
                    acronym = re.sub(r"[^A-Za-z0-9]+", "", match.group(1)).casefold()
                    if acronym == expected and len(acronym) >= 2:
                        observed[(material, acronym)].add(chunk_id)

    acronym_to_materials: dict[str, set[str]] = defaultdict(set)
    material_to_acronyms: dict[str, set[str]] = defaultdict(set)
    for material, acronym in observed:
        acronym_to_materials[acronym].add(material)
        material_to_acronyms[material].add(acronym)

    accepted: dict[str, tuple[str, tuple[str, ...]]] = {}
    audit: list[dict[str, Any]] = []
    for (material, acronym), source_ids in sorted(observed.items()):
        unique = (
            len(acronym_to_materials[acronym]) == 1
            and len(material_to_acronyms[material]) == 1
        )
        audit.append(
            {
                "full_name": material,
                "acronym": acronym,
                "source_chunk_ids": sorted(source_ids),
                "accepted": unique,
                "reason": "mutual_unique" if unique else "non_unique_mapping",
            }
        )
        if unique:
            accepted[material] = (acronym, tuple(sorted(source_ids)))
    return accepted, audit


def _build_inventory(
    profiles: list[dict[str, Any]], chunks: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    abbreviations, abbreviation_audit = _mine_abbreviations(profiles, chunks)
    grouped_candidates: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    identity_profiles: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in profiles:
        identity = profile["canonical_identity"]
        identity_profiles[identity].append(profile)
        for alias, provenance in profile["candidates"].items():
            if alias:
                grouped_candidates[identity][alias].update(provenance)
    acronym_sources: dict[tuple[str, str], tuple[str, ...]] = {}
    for identity, members in identity_profiles.items():
        material = members[0]["canonical_material"]
        if material in abbreviations:
            acronym, source_ids = abbreviations[material]
            grouped_candidates[identity][acronym].add("document_parenthetical_acronym")
            acronym_sources[(identity, acronym)] = source_ids

    alias_identities: dict[str, set[str]] = defaultdict(set)
    alias_materials: dict[str, set[str]] = defaultdict(set)
    for identity, candidates in grouped_candidates.items():
        material = identity_profiles[identity][0]["canonical_material"]
        for alias in candidates:
            alias_identities[alias].add(identity)
            alias_materials[alias].add(material)

    target_ids_by_identity = {
        identity: tuple(sorted(member["target_id"] for member in members))
        for identity, members in identity_profiles.items()
    }
    targets_by_alias: dict[str, tuple[str, ...]] = {}
    for alias, identities in alias_identities.items():
        targets_by_alias[alias] = tuple(
            sorted(
                target_id
                for identity in identities
                for target_id in target_ids_by_identity[identity]
            )
        )

    inventories: dict[str, dict[str, Any]] = {}
    conflicts: dict[str, dict[str, Any]] = {}
    for identity, members in identity_profiles.items():
        aliases = []
        for alias in sorted(grouped_candidates[identity]):
            if _is_generic(alias):
                status = "generic"
            elif len(alias_identities[alias]) == 1:
                status = "unique"
            elif len(alias_materials[alias]) == 1:
                status = "shared_material"
            else:
                status = "conflict"
            record = {
                "text": alias,
                "provenance": sorted(grouped_candidates[identity][alias]),
                "status": status,
                "owner_target_ids": list(targets_by_alias[alias]),
                "source_chunk_ids": list(acronym_sources.get((identity, alias), ())),
            }
            aliases.append(record)
            if status in ("generic", "conflict"):
                conflicts[alias] = {
                    "status": status,
                    "owner_target_ids": list(targets_by_alias[alias]),
                }
        duplicate_ids = target_ids_by_identity[identity]
        template = members[0]
        for member in members:
            inventories[member["target_id"]] = {
                "rockname": member["rockname"],
                "canonical_identity": identity,
                "canonical_material": template["canonical_material"],
                "components": list(template["components"]),
                "polarities": list(template["polarities"]),
                "duplicate_target_ids": list(duplicate_ids if len(duplicate_ids) > 1 else ()),
                "aliases": aliases,
            }
    audit = {
        "abbreviation_links": abbreviation_audit,
        "alias_conflicts": conflicts,
        "duplicate_groups": [
            list(target_ids)
            for target_ids in target_ids_by_identity.values()
            if len(target_ids) > 1
        ],
    }
    return inventories, audit


def _softmax(logits: Sequence[float]) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    values = values - np.max(values)
    exponentials = np.exp(values)
    return exponentials / np.sum(exponentials)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _score_chunk(
    chunk: Mapping[str, Any], target_ids: Sequence[str], inventories: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    text = str(chunk["text"])
    folded = _fold(text)
    chunk_components, chunk_polarities = _qualifiers(text)
    raw_scores: dict[str, float] = {}
    signals: dict[str, dict[str, Any]] = {}

    for target_id in target_ids:
        inventory = inventories[target_id]
        exact: list[str] = []
        aliases: list[str] = []
        acronyms: list[str] = []
        conflicts: list[str] = []
        cluster_match = False
        base_score = 0.0
        for record in inventory["aliases"]:
            alias = record["text"]
            if not _contains_phrase(folded, alias):
                continue
            provenance = set(record["provenance"])
            status = record["status"]
            if status in ("generic", "conflict"):
                conflicts.append(alias)
                continue
            is_exact = bool(provenance.intersection(("identity_exact", "identity_normalized")))
            is_acronym = "document_parenthetical_acronym" in provenance
            if is_exact:
                exact.append(alias)
            else:
                aliases.append(alias)
            if is_acronym:
                acronyms.append(alias)
            if status == "shared_material":
                weight = 1.5
            elif is_exact:
                weight = 5.0
            elif is_acronym:
                weight = 3.5
            elif "identity_material" in provenance:
                weight = 3.0
            elif "identity_unit_stem" in provenance:
                weight = 1.8
            else:
                weight = 2.5
            base_score = max(base_score, weight)

        cluster = _cluster_score(str(inventory["canonical_material"]), folded)
        if cluster > base_score:
            base_score = cluster
            cluster_match = True

        target_components = set(inventory["components"])
        target_polarities = set(inventory["polarities"])
        component_match = bool(set(chunk_components).intersection(target_components))
        polarity_match = bool(set(chunk_polarities).intersection(target_polarities))
        score = base_score
        if base_score > 0.0 and chunk_components and target_components:
            score += 2.0 if component_match else -1.0
        if base_score > 0.0 and chunk_polarities and target_polarities:
            score += 2.0 if polarity_match else -1.0
        score = max(0.0, score)
        raw_scores[target_id] = score
        signals[target_id] = {
            "exact_mentions": sorted(set(exact)),
            "alias_mentions": sorted(set(aliases)),
            "acronym_mentions": sorted(set(acronyms)),
            "conflicting_alias_mentions": sorted(set(conflicts)),
            "token_cluster_match": cluster_match,
            "component_mentions": list(chunk_components),
            "component_match": component_match,
            "polarity_mentions": list(chunk_polarities),
            "polarity_match": polarity_match,
        }

    number_targets = len(target_ids)
    positive = [target_id for target_id in target_ids if raw_scores[target_id] > 0.0]
    maximum = max(raw_scores.values(), default=0.0)
    top_ids = [
        target_id
        for target_id in target_ids
        if maximum > 0.0 and math.isclose(raw_scores[target_id], maximum, abs_tol=1e-12)
    ]
    duplicate_top = bool(
        len(top_ids) > 1
        and len({inventories[target_id]["canonical_identity"] for target_id in top_ids}) == 1
    )
    target_logits = [
        raw_scores[target_id]
        if raw_scores[target_id] > 0.0
        else -6.0 - math.log(max(1, number_targets))
        for target_id in target_ids
    ]
    global_logit = -1.0 if positive else (0.5 if chunk_components or chunk_polarities else 0.0)
    if not positive:
        unresolved_logit = 3.5 if any(
            signals[target_id]["conflicting_alias_mentions"] for target_id in target_ids
        ) else 3.0
    elif len(top_ids) > 1:
        unresolved_logit = maximum + (1.0 if duplicate_top else 0.75)
    else:
        unresolved_logit = 0.0
    probabilities = _softmax((*target_logits, global_logit, unresolved_logit))
    target_probabilities = {
        target_id: float(probabilities[index]) for index, target_id in enumerate(target_ids)
    }
    global_mass = float(probabilities[-2])
    unresolved_mass = float(probabilities[-1])
    ordered_target_mass = sorted(target_probabilities.values(), reverse=True)
    margin = (
        ordered_target_mass[0] - ordered_target_mass[1]
        if len(ordered_target_mass) > 1
        else ordered_target_mass[0]
    )
    entropy = float(-np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    entropy /= math.log(len(probabilities))
    swap_delta: dict[str, float] = {}
    for target_id in target_ids:
        sibling_mass = [
            target_probabilities[other] for other in target_ids if other != target_id
        ]
        swap_delta[target_id] = (
            target_probabilities[target_id] - max(sibling_mass) if sibling_mass else 0.0
        )

    top_target = top_ids[0] if len(top_ids) == 1 else None
    if not positive:
        status = (
            "unresolved_alias_conflict"
            if any(signals[target_id]["conflicting_alias_mentions"] for target_id in target_ids)
            else "unresolved_no_entity"
        )
        top_target = None
    elif duplicate_top:
        status = "duplicate_canonical_identity"
        top_target = None
    elif len(top_ids) > 1:
        status = "unresolved_sibling_tie"
        top_target = None
    elif target_probabilities[top_target] <= max(global_mass, unresolved_mass):
        status = "unresolved_low_margin"
    elif signals[top_target]["component_match"] or signals[top_target]["polarity_match"]:
        status = "assigned_qualified"
    elif signals[top_target]["exact_mentions"]:
        status = "assigned_exact"
    else:
        status = "assigned_alias"

    return {
        "chunk_id": chunk["chunk_id"],
        "kind": chunk["kind"],
        "parent_id": chunk["parent_id"],
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "top_target": top_target,
        "status": status,
        "target_scores": raw_scores,
        "target_probabilities": target_probabilities,
        "global_mass": global_mass,
        "unresolved_mass": unresolved_mass,
        "margin": float(margin if positive else 0.0),
        "entropy": entropy,
        "swap_delta": swap_delta,
        "signals": signals,
    }


def _aggregate_features(
    target_ids: Sequence[str], inventories: Mapping[str, Mapping[str, Any]], assignments: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for target_id in target_ids:
        inventory = inventories[target_id]
        aliases = inventory["aliases"]
        masses = [float(row["target_probabilities"][target_id]) for row in assignments]
        signals = [row["signals"][target_id] for row in assignments]
        positive = [float(row["target_scores"][target_id]) > 0.0 for row in assignments]
        target_rows = [row for row, is_positive in zip(assignments, positive) if is_positive]
        values = {
            "entity_inventory_alias_count_log": math.log1p(len(aliases)),
            "entity_inventory_unique_alias_count_log": math.log1p(sum(alias["status"] == "unique" for alias in aliases)),
            "entity_inventory_conflict_alias_count_log": math.log1p(sum(alias["status"] in ("generic", "conflict") for alias in aliases)),
            "entity_duplicate_identity": float(bool(inventory["duplicate_target_ids"])),
            "entity_chunk_count_log": math.log1p(len(assignments)),
            "entity_exact_chunk_count_log": math.log1p(sum(bool(signal["exact_mentions"]) for signal in signals)),
            "entity_alias_chunk_count_log": math.log1p(sum(bool(signal["alias_mentions"] or signal["acronym_mentions"]) for signal in signals)),
            "entity_cluster_chunk_count_log": math.log1p(sum(bool(signal["token_cluster_match"]) for signal in signals)),
            "entity_component_match_count_log": math.log1p(sum(bool(signal["component_match"]) for signal in signals)),
            "entity_polarity_match_count_log": math.log1p(sum(bool(signal["polarity_match"]) for signal in signals)),
            "entity_positive_chunk_count_log": math.log1p(sum(positive)),
            "entity_target_mass_mean": _mean(masses),
            "entity_target_mass_max": max(masses, default=0.0),
            "entity_target_mass_sum_log": math.log1p(sum(masses)),
            "entity_winner_count_log": math.log1p(sum(row["top_target"] == target_id for row in assignments)),
            "entity_margin_mean": _mean([float(row["margin"]) for row in target_rows]),
            "entity_entropy_mean": _mean([float(row["entropy"]) for row in target_rows]),
            "entity_swap_delta_mean": _mean([float(row["swap_delta"][target_id]) for row in target_rows]),
            "entity_unresolved_mass_mean": _mean([float(row["unresolved_mass"]) for row in target_rows]),
            "entity_global_mass_mean": _mean([float(row["global_mass"]) for row in target_rows]),
        }
        output[target_id] = {name: float(values[name]) for name in FEATURE_NAMES}
    return output


def build_entity_graph(
    targets: Mapping[str, str], chunks: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build a conservative, auditable entity graph from original text only."""

    target_rows, chunk_rows = _validate_inputs(targets, chunks)
    profiles = [_profile(target_id, rockname) for target_id, rockname in target_rows]
    inventories, inventory_audit = _build_inventory(profiles, chunk_rows)
    target_ids = [target_id for target_id, _ in target_rows]
    assignments = [
        _score_chunk(chunk, target_ids, inventories) for chunk in chunk_rows
    ]
    target_features = _aggregate_features(target_ids, inventories, assignments)
    return {
        "feature_names": list(FEATURE_NAMES),
        "target_features": target_features,
        "assignments": assignments,
        "inventories": inventories,
        "audit": {
            "schema_version": SCHEMA_VERSION,
            "label_blind_input_fields": {
                "targets": "{target_id: rockname}",
                "chunks": ["chunk_id", "kind", "text", "parent_id"],
            },
            "softmax_classes": [*target_ids, "global", "unresolved"],
            **inventory_audit,
        },
    }


__all__ = ("FEATURE_NAMES", "SCHEMA_VERSION", "build_entity_graph")
