#!/usr/bin/env python3
"""Label-blind single-page layout graph features for target-bound R3 evidence."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import re
from typing import Any, Mapping, Pattern, Sequence

import numpy as np

try:
    from .paper_text_retrieval import canonical_target_identity, fold_text_with_position_map
    from .r3_paleomag_scope_features import TargetProfile, anchor_positions, target_profile
    from .r3_trace_page_graph_features import Concept
except ImportError:
    from paper_text_retrieval import canonical_target_identity, fold_text_with_position_map
    from r3_paleomag_scope_features import TargetProfile, anchor_positions, target_profile
    from r3_trace_page_graph_features import Concept


CATEGORIES = ("method", "diagnostic", "carrier", "bridge")
RELATIONS = ("same_line", "same_block", "same_table_row", "same_caption")
RELATION_STRENGTH = {
    "same_line": 1.0,
    "same_table_row": 0.95,
    "same_caption": 0.90,
    "same_block": 0.72,
}
CLUSTER_ANCHOR_STRENGTH = 0.78
SIBLING_MARGIN = 0.08
FORBIDDEN_FIELD_RE = re.compile(
    r"^(?:r[357]|pmagresult|rockmag|gold|ground_truth|"
    r"predictions?|oof|evaluation|metrics?)$",
    re.I,
)


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _normalized_entropy(values: Sequence[float], maximum_classes: int) -> float:
    total = float(sum(values))
    if total <= 0.0 or maximum_classes <= 1:
        return 0.0
    probabilities = [float(value) / total for value in values if value > 0.0]
    entropy = -sum(value * math.log(value) for value in probabilities)
    return min(1.0, entropy / math.log(maximum_classes))


def _validate_label_blind(value: Any, path: str = "page") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = _clean(key)
            if FORBIDDEN_FIELD_RE.fullmatch(name):
                raise ValueError(f"forbidden outcome field at {path}.{name}")
            _validate_label_blind(nested, f"{path}.{name}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _validate_label_blind(nested, f"{path}[{index}]")


def _confidence(
    value: Mapping[str, Any], inherited: tuple[float, bool] = (0.0, False)
) -> tuple[float, bool]:
    for key in ("layout_confidence", "confidence", "score", "ocr_confidence"):
        if key not in value or value.get(key) is None:
            continue
        raw = value.get(key)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"layout confidence {key} must be numeric")
        score = float(raw)
        if not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError(f"layout confidence {key} must be in [0, 1]")
        return score, True
    return inherited


def _item_id(value: Mapping[str, Any], fallback: str) -> str:
    identifier = _clean(
        value.get("id")
        or value.get("block_id")
        or value.get("line_id")
        or value.get("table_id")
        or value.get("row_id")
        or value.get("caption_id")
    )
    return identifier or fallback


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        return _clean(value.get("text") or value.get("content") or value.get("value"))
    return _clean(value)


def _cell_texts(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        cells = value.get("cells")
        if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes, bytearray)):
            return [text for item in cells if (text := _text(item))]
        text = _text(value)
        return [text] if text else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [text for item in value if (text := _text(item))]
    text = _text(value)
    return [text] if text else []


@dataclass(frozen=True)
class _Atom:
    source_id: str
    kind: str
    text: str
    block_id: str | None
    confidence: float
    confidence_available: bool


def _layout_atoms(page: Mapping[str, Any]) -> list[_Atom]:
    page_confidence = _confidence(page)
    atoms: list[_Atom] = []
    for block_index, raw_block in enumerate(page.get("blocks") or [], start=1):
        if not isinstance(raw_block, Mapping):
            raise ValueError("layout blocks must be mappings")
        block_id = _item_id(raw_block, f"block-{block_index}")
        block_confidence = _confidence(raw_block, page_confidence)
        raw_lines = raw_block.get("lines")
        lines = (
            list(raw_lines)
            if isinstance(raw_lines, Sequence)
            and not isinstance(raw_lines, (str, bytes, bytearray))
            else []
        )
        if lines:
            for line_index, raw_line in enumerate(lines, start=1):
                line = raw_line if isinstance(raw_line, Mapping) else {"text": raw_line}
                text = _text(line)
                if not text:
                    continue
                confidence, available = _confidence(line, block_confidence)
                line_id = _item_id(line, f"line-{line_index}")
                atoms.append(
                    _Atom(
                        f"block:{block_id}:line:{line_id}",
                        "line",
                        text,
                        block_id,
                        confidence,
                        available,
                    )
                )
        else:
            text = _text(raw_block)
            if text:
                atoms.append(
                    _Atom(
                        f"block:{block_id}",
                        "block",
                        text,
                        block_id,
                        block_confidence[0],
                        block_confidence[1],
                    )
                )

    for table_index, raw_table in enumerate(page.get("tables") or [], start=1):
        if not isinstance(raw_table, Mapping):
            raise ValueError("layout tables must be mappings")
        table_id = _item_id(raw_table, f"table-{table_index}")
        table_confidence = _confidence(raw_table, page_confidence)
        rows = raw_table.get("rows") or []
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes, bytearray)):
            raise ValueError("layout table rows must be a sequence")
        for row_index, raw_row in enumerate(rows, start=1):
            row = raw_row if isinstance(raw_row, Mapping) else {"cells": raw_row}
            texts = _cell_texts(raw_row)
            if not texts:
                continue
            confidence, available = _confidence(row, table_confidence)
            row_id = _item_id(row, f"row-{row_index}")
            atoms.append(
                _Atom(
                    f"table:{table_id}:row:{row_id}",
                    "table_row",
                    " | ".join(texts),
                    None,
                    confidence,
                    available,
                )
            )
        caption = raw_table.get("caption")
        caption_text = _text(caption)
        if caption_text:
            caption_map = caption if isinstance(caption, Mapping) else {"text": caption}
            confidence, available = _confidence(caption_map, table_confidence)
            atoms.append(
                _Atom(
                    f"table:{table_id}:caption",
                    "caption",
                    caption_text,
                    None,
                    confidence,
                    available,
                )
            )

    for caption_index, raw_caption in enumerate(page.get("captions") or [], start=1):
        caption = raw_caption if isinstance(raw_caption, Mapping) else {"text": raw_caption}
        text = _text(caption)
        if not text:
            continue
        confidence, available = _confidence(caption, page_confidence)
        caption_id = _item_id(caption, f"caption-{caption_index}")
        atoms.append(
            _Atom(
                f"caption:{caption_id}",
                "caption",
                text,
                None,
                confidence,
                available,
            )
        )
    if len({atom.source_id for atom in atoms}) != len(atoms):
        raise ValueError("layout source identifiers must be unique within a page")
    return atoms


@dataclass(frozen=True)
class _FrozenConcept:
    concept_id: str
    category: str
    patterns: tuple[Pattern[str], ...]


def _concepts(values: Sequence[Any]) -> tuple[_FrozenConcept, ...]:
    output: list[_FrozenConcept] = []
    seen: set[str] = set()
    for index, value in enumerate(values, start=1):
        if isinstance(value, Mapping):
            concept_id = _clean(value.get("concept_id"))
            category = _clean(value.get("category")).casefold()
            raw_patterns = value.get("patterns") or value.get("search_patterns") or []
        else:
            concept_id = _clean(getattr(value, "concept_id", ""))
            category = _clean(getattr(value, "category", "")).casefold()
            raw_patterns = getattr(value, "patterns", ())
        if not concept_id or concept_id in seen:
            raise ValueError(f"layout concept {index} has an empty or duplicate concept_id")
        if category not in CATEGORIES:
            raise ValueError(f"layout concept {concept_id} has invalid category {category!r}")
        patterns = tuple(
            item if hasattr(item, "finditer") else re.compile(_clean(item), re.I)
            for item in raw_patterns
            if _clean(getattr(item, "pattern", item))
        )
        if not patterns:
            raise ValueError(f"layout concept {concept_id} has no patterns")
        output.append(_FrozenConcept(concept_id, category, patterns))
        seen.add(concept_id)
    missing = set(CATEGORIES) - {concept.category for concept in output}
    if missing:
        raise ValueError(f"layout concepts lack categories: {sorted(missing)}")
    return tuple(output)


@dataclass(frozen=True)
class _Target:
    key: str
    profile: TargetProfile
    canonical: str


def _targets(values: Mapping[Any, Any]) -> tuple[list[_Target], dict[str, list[_Target]]]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("target_profiles must be a nonempty mapping")
    output: list[_Target] = []
    seen_keys: set[str] = set()
    by_canonical: dict[str, list[_Target]] = defaultdict(list)
    for raw_key, raw_profile in values.items():
        key = _clean(raw_key)
        if not key or key in seen_keys:
            raise ValueError("target profile keys must be nonempty and unique")
        profile = raw_profile if isinstance(raw_profile, TargetProfile) else target_profile(raw_profile)
        canonical = canonical_target_identity(profile.raw)
        if not canonical:
            raise ValueError(f"target profile {key} has an empty canonical identity")
        target = _Target(key, profile, canonical)
        output.append(target)
        by_canonical[canonical].append(target)
        seen_keys.add(key)
    return output, dict(by_canonical)


def _anchor_kind(text: str, profiles: Sequence[TargetProfile]) -> tuple[str, float]:
    folded, _ = fold_text_with_position_map(text)
    best_kind, best = "none", 0.0
    for profile in profiles:
        exact, cluster = anchor_positions(folded, profile)
        if exact:
            return "exact", 1.0
        if cluster and best < CLUSTER_ANCHOR_STRENGTH:
            best_kind, best = "cluster", CLUSTER_ANCHOR_STRENGTH
    return best_kind, best


@dataclass
class _Accumulator:
    anchors: Counter[str] = field(default_factory=Counter)
    relation_mass: dict[tuple[str, str], float] = field(default_factory=lambda: defaultdict(float))
    category_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    exclusive_category_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    shared_category_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    ambiguous_category_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    evidence_mass: float = 0.0
    exclusive_mass: float = 0.0
    shared_mass: float = 0.0
    ambiguous_mass: float = 0.0
    unresolved_mass: float = 0.0
    unique_count: int = 0
    shared_count: int = 0
    ambiguous_count: int = 0
    margins: list[float] = field(default_factory=list)
    assignment_entropy_mass: float = 0.0
    confidence_available_mass: float = 0.0
    confidence_weighted_mass: float = 0.0
    confidence_values: list[float] = field(default_factory=list)

def _add_resolved(
    accumulator: _Accumulator,
    *,
    category: str,
    relation: str,
    mass: float,
    status: str,
    margin: float,
    entropy: float,
    atom: _Atom,
) -> None:
    accumulator.relation_mass[(category, relation)] += mass
    accumulator.category_mass[category] += mass
    accumulator.evidence_mass += mass
    accumulator.margins.append(margin)
    accumulator.assignment_entropy_mass += mass * entropy
    if status == "exclusive":
        accumulator.exclusive_mass += mass
        accumulator.exclusive_category_mass[category] += mass
        accumulator.unique_count += 1
    else:
        accumulator.shared_mass += mass
        accumulator.shared_category_mass[category] += mass
        accumulator.shared_count += 1
    if atom.confidence_available:
        accumulator.confidence_available_mass += mass
        accumulator.confidence_weighted_mass += mass * atom.confidence
        accumulator.confidence_values.append(atom.confidence)


def _base_feature_map() -> dict[str, float]:
    values: dict[str, float] = {
        "layout_resolved_identity": 0.0,
        "layout_unresolved_duplicate_identity": 0.0,
        "layout_duplicate_identity_size_log": 0.0,
        "layout_anchor_exact_scope_count_log": 0.0,
        "layout_anchor_cluster_scope_count_log": 0.0,
        "layout_anchor_scope_count_log": 0.0,
    }
    for category in CATEGORIES:
        for relation in RELATIONS:
            values[f"layout_{category}_{relation}_available"] = 0.0
            values[f"layout_{category}_{relation}_mass"] = 0.0
        values[f"layout_{category}_evidence_mass"] = 0.0
        values[f"layout_{category}_exclusive_mass"] = 0.0
        values[f"layout_{category}_shared_mass"] = 0.0
        values[f"layout_{category}_ambiguous_mass"] = 0.0
    values.update(
        {
            "layout_complete_chain": 0.0,
            "layout_exclusive_complete_chain": 0.0,
            "layout_exclusive_complete_chain_with_diagnostic": 0.0,
            "layout_exclusive_chain_category_fraction": 0.0,
            "layout_exclusive_chain_min_mass": 0.0,
            "layout_evidence_mass": 0.0,
            "layout_evidence_share": 0.0,
            "layout_evidence_mass_minus_sibling_max": 0.0,
            "layout_exclusive_evidence_mass": 0.0,
            "layout_shared_evidence_mass": 0.0,
            "layout_ambiguous_evidence_mass": 0.0,
            "layout_unresolved_evidence_mass": 0.0,
            "layout_unique_assignment_count_log": 0.0,
            "layout_shared_assignment_count_log": 0.0,
            "layout_ambiguous_assignment_count_log": 0.0,
            "layout_has_shared_evidence": 0.0,
            "layout_has_ambiguous_evidence": 0.0,
            "layout_unique_sibling_margin_min": 0.0,
            "layout_unique_sibling_margin_mean": 0.0,
            "layout_unique_sibling_margin_max": 0.0,
            "layout_evidence_category_entropy": 0.0,
            "layout_assignment_entropy_mean": 0.0,
            "layout_ambiguity_ratio": 0.0,
            "layout_confidence": 0.0,
            "layout_confidence_mean": 0.0,
            "layout_confidence_min": 0.0,
            "layout_confidence_available_ratio": 0.0,
            "layout_confidence_weighted_evidence_mass": 0.0,
        }
    )
    return values


FEATURE_NAMES = tuple(_base_feature_map())


def build_layout_graph_target_features(
    page: Mapping[str, Any],
    target_profiles: Mapping[Any, Any],
    concepts: Sequence[Concept | Mapping[str, Any]],
    *,
    sibling_margin: float = SIBLING_MARGIN,
) -> dict[str, Any]:
    """Build target rows from one page without labels or result-level outcomes.

    Target mapping keys are opaque output identifiers. Assignment is performed
    only at canonical target-identity level. Duplicate canonical identities are
    retained as unresolved rows and never separated by their mapping keys.
    """

    if not isinstance(page, Mapping):
        raise ValueError("layout page must be a mapping")
    if not math.isfinite(float(sibling_margin)) or sibling_margin < 0.0:
        raise ValueError("sibling_margin must be a finite nonnegative number")
    _validate_label_blind(page)
    targets, by_canonical = _targets(target_profiles)
    frozen_concepts = _concepts(concepts)
    atoms = _layout_atoms(page)
    canonical_profiles = {
        canonical: [target.profile for target in members]
        for canonical, members in by_canonical.items()
    }
    accumulators = {canonical: _Accumulator() for canonical in by_canonical}

    anchor_kind_by_atom: dict[str, dict[str, str]] = {}
    anchor_strength_by_atom: dict[str, dict[str, float]] = {}
    block_strength: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for atom in atoms:
        kinds: dict[str, str] = {}
        strengths: dict[str, float] = {}
        for canonical, profiles in canonical_profiles.items():
            kind, strength = _anchor_kind(atom.text, profiles)
            kinds[canonical] = kind
            strengths[canonical] = strength
            if kind != "none":
                accumulators[canonical].anchors[kind] += 1
                accumulators[canonical].anchors["all"] += 1
            if atom.block_id is not None:
                block_strength[atom.block_id][canonical] = max(
                    block_strength[atom.block_id][canonical], strength
                )
        anchor_kind_by_atom[atom.source_id] = kinds
        anchor_strength_by_atom[atom.source_id] = strengths

    assignments: list[dict[str, Any]] = []
    event_number = 0
    audit_counts: Counter[str] = Counter()
    for atom in atoms:
        concept_hits: list[tuple[_FrozenConcept, int, int]] = []
        for concept in frozen_concepts:
            spans = {
                (match.start(), match.end())
                for pattern in concept.patterns
                for match in pattern.finditer(atom.text)
            }
            concept_hits.extend((concept, start, end) for start, end in sorted(spans))
        for concept, start, end in concept_hits:
            event_number += 1
            candidates: list[tuple[float, str, str]] = []
            for canonical in sorted(by_canonical):
                direct = anchor_strength_by_atom[atom.source_id][canonical]
                relation = ""
                strength = 0.0
                if atom.kind == "line" and direct > 0.0:
                    relation, strength = "same_line", direct
                elif atom.kind == "table_row" and direct > 0.0:
                    relation, strength = "same_table_row", direct
                elif atom.kind == "caption" and direct > 0.0:
                    relation, strength = "same_caption", direct
                elif atom.kind == "block" and direct > 0.0:
                    relation, strength = "same_block", direct
                elif atom.kind == "line" and atom.block_id is not None:
                    inherited = block_strength[atom.block_id][canonical]
                    if inherited > 0.0:
                        relation, strength = "same_block", inherited
                if relation:
                    candidates.append(
                        (RELATION_STRENGTH[relation] * strength, canonical, relation)
                    )
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            assignment: dict[str, Any] = {
                "event_id": event_number,
                "source_id": atom.source_id,
                "source_kind": atom.kind,
                "concept_id": concept.concept_id,
                "category": concept.category,
                "span": [start, end],
                "status": "unassigned_no_target",
                "target_keys": [],
                "canonical_targets": [],
                "sibling_margin": 0.0,
                "layout_confidence": atom.confidence if atom.confidence_available else None,
            }
            if not candidates:
                audit_counts["unassigned_no_target"] += 1
                assignments.append(assignment)
                continue
            top_score = candidates[0][0]
            second_score = candidates[1][0] if len(candidates) > 1 else 0.0
            margin = top_score - second_score
            close = [item for item in candidates if top_score - item[0] <= sibling_margin]
            shared = (
                len(close) > 1
                and close[0][2] in {"same_line", "same_table_row", "same_caption"}
                and all(item[2] == close[0][2] for item in close)
            )
            assignment["sibling_margin"] = round(float(margin), 8)
            assignment["canonical_targets"] = [item[1] for item in close]
            if len(close) > 1 and not shared:
                assignment["status"] = "ambiguous"
                entropy = _normalized_entropy([1.0] * len(close), len(close))
                for _, canonical, _ in close:
                    part = top_score / len(close)
                    accumulator = accumulators[canonical]
                    accumulator.margins.append(margin)
                    if len(by_canonical[canonical]) > 1:
                        accumulator.unresolved_mass += part
                    else:
                        accumulator.ambiguous_mass += part
                        accumulator.ambiguous_category_mass[concept.category] += part
                        accumulator.ambiguous_count += 1
                        accumulator.assignment_entropy_mass += part * entropy
                audit_counts["ambiguous"] += 1
                assignments.append(assignment)
                continue

            selected = close if shared else [candidates[0]]
            status = "shared" if shared else "exclusive"
            assignment["status"] = status
            entropy = _normalized_entropy([1.0] * len(selected), len(selected))
            unresolved_selected = False
            for _, canonical, relation in selected:
                part = top_score / len(selected)
                if len(by_canonical[canonical]) > 1:
                    accumulators[canonical].unresolved_mass += part
                    unresolved_selected = True
                    continue
                _add_resolved(
                    accumulators[canonical],
                    category=concept.category,
                    relation=relation,
                    mass=part,
                    status=status,
                    margin=margin,
                    entropy=entropy,
                    atom=atom,
                )
                assignment["target_keys"].append(by_canonical[canonical][0].key)
            if unresolved_selected:
                assignment["status"] = (
                    "shared_with_unresolved_duplicate_identity"
                    if shared and assignment["target_keys"]
                    else "unresolved_duplicate_identity"
                )
            audit_counts[assignment["status"]] += 1
            assignments.append(assignment)

    total_resolved_mass = sum(
        accumulator.evidence_mass
        for canonical, accumulator in accumulators.items()
        if len(by_canonical[canonical]) == 1
    )
    resolved_masses = {
        canonical: accumulator.evidence_mass
        for canonical, accumulator in accumulators.items()
        if len(by_canonical[canonical]) == 1
    }
    target_features: dict[str, dict[str, float]] = {}
    target_metadata: dict[str, dict[str, Any]] = {}
    for target in targets:
        members = by_canonical[target.canonical]
        duplicate = len(members) > 1
        accumulator = accumulators[target.canonical]
        values = _base_feature_map()
        values["layout_resolved_identity"] = 0.0 if duplicate else 1.0
        values["layout_unresolved_duplicate_identity"] = 1.0 if duplicate else 0.0
        values["layout_duplicate_identity_size_log"] = _log1p(len(members) - 1)
        values["layout_anchor_exact_scope_count_log"] = _log1p(
            accumulator.anchors["exact"]
        )
        values["layout_anchor_cluster_scope_count_log"] = _log1p(
            accumulator.anchors["cluster"]
        )
        values["layout_anchor_scope_count_log"] = _log1p(accumulator.anchors["all"])
        values["layout_unresolved_evidence_mass"] = accumulator.unresolved_mass
        if not duplicate:
            for category in CATEGORIES:
                for relation in RELATIONS:
                    mass = accumulator.relation_mass[(category, relation)]
                    values[f"layout_{category}_{relation}_available"] = float(mass > 0.0)
                    values[f"layout_{category}_{relation}_mass"] = mass
                values[f"layout_{category}_evidence_mass"] = accumulator.category_mass[category]
                values[f"layout_{category}_exclusive_mass"] = accumulator.exclusive_category_mass[category]
                values[f"layout_{category}_shared_mass"] = accumulator.shared_category_mass[category]
                values[f"layout_{category}_ambiguous_mass"] = accumulator.ambiguous_category_mass[category]
            chain = [accumulator.category_mass[item] for item in ("method", "carrier", "bridge")]
            exclusive_chain = [
                accumulator.exclusive_category_mass[item]
                for item in ("method", "carrier", "bridge")
            ]
            values["layout_complete_chain"] = float(all(value > 0.0 for value in chain))
            values["layout_exclusive_complete_chain"] = float(
                all(value > 0.0 for value in exclusive_chain)
            )
            values["layout_exclusive_complete_chain_with_diagnostic"] = float(
                all(value > 0.0 for value in exclusive_chain)
                and accumulator.exclusive_category_mass["diagnostic"] > 0.0
            )
            values["layout_exclusive_chain_category_fraction"] = sum(
                value > 0.0 for value in exclusive_chain
            ) / 3.0
            values["layout_exclusive_chain_min_mass"] = min(exclusive_chain)
            values["layout_evidence_mass"] = accumulator.evidence_mass
            values["layout_evidence_share"] = (
                accumulator.evidence_mass / total_resolved_mass
                if total_resolved_mass > 0.0
                else 0.0
            )
            sibling_max = max(
                (mass for canonical, mass in resolved_masses.items() if canonical != target.canonical),
                default=0.0,
            )
            values["layout_evidence_mass_minus_sibling_max"] = (
                accumulator.evidence_mass - sibling_max
            )
            values["layout_exclusive_evidence_mass"] = accumulator.exclusive_mass
            values["layout_shared_evidence_mass"] = accumulator.shared_mass
            values["layout_ambiguous_evidence_mass"] = accumulator.ambiguous_mass
            values["layout_unique_assignment_count_log"] = _log1p(accumulator.unique_count)
            values["layout_shared_assignment_count_log"] = _log1p(accumulator.shared_count)
            values["layout_ambiguous_assignment_count_log"] = _log1p(accumulator.ambiguous_count)
            values["layout_has_shared_evidence"] = float(accumulator.shared_mass > 0.0)
            values["layout_has_ambiguous_evidence"] = float(accumulator.ambiguous_mass > 0.0)
            if accumulator.margins:
                values["layout_unique_sibling_margin_min"] = min(accumulator.margins)
                values["layout_unique_sibling_margin_mean"] = sum(accumulator.margins) / len(accumulator.margins)
                values["layout_unique_sibling_margin_max"] = max(accumulator.margins)
            values["layout_evidence_category_entropy"] = _normalized_entropy(
                [accumulator.category_mass[category] for category in CATEGORIES],
                len(CATEGORIES),
            )
            values["layout_assignment_entropy_mean"] = (
                accumulator.assignment_entropy_mass
                / (accumulator.evidence_mass + accumulator.ambiguous_mass)
                if accumulator.evidence_mass + accumulator.ambiguous_mass > 0.0
                else 0.0
            )
            values["layout_ambiguity_ratio"] = (
                accumulator.ambiguous_mass
                / (accumulator.evidence_mass + accumulator.ambiguous_mass)
                if accumulator.evidence_mass + accumulator.ambiguous_mass > 0.0
                else 0.0
            )
            values["layout_confidence_available_ratio"] = (
                accumulator.confidence_available_mass / accumulator.evidence_mass
                if accumulator.evidence_mass > 0.0
                else 0.0
            )
            values["layout_confidence_mean"] = (
                accumulator.confidence_weighted_mass / accumulator.confidence_available_mass
                if accumulator.confidence_available_mass > 0.0
                else 0.0
            )
            values["layout_confidence_min"] = min(accumulator.confidence_values, default=0.0)
            values["layout_confidence"] = (
                values["layout_confidence_mean"]
                * values["layout_confidence_available_ratio"]
            )
            values["layout_confidence_weighted_evidence_mass"] = accumulator.confidence_weighted_mass
        target_features[target.key] = values
        target_metadata[target.key] = {
            "canonical_identity": target.canonical,
            "resolution_status": "unresolved_duplicate_identity" if duplicate else "resolved",
            "duplicate_target_keys": [member.key for member in members] if duplicate else [],
        }

    matrix = np.asarray(
        [[target_features[target.key][name] for name in FEATURE_NAMES] for target in targets],
        dtype=np.float32,
    )
    if matrix.shape != (len(targets), len(FEATURE_NAMES)) or not np.isfinite(matrix).all():
        raise AssertionError("layout feature matrix is invalid")
    return {
        "feature_names": list(FEATURE_NAMES),
        "matrix": matrix,
        "target_keys": [target.key for target in targets],
        "canonical_identities": [target.canonical for target in targets],
        "target_features": target_features,
        "target_metadata": target_metadata,
        "assignments": assignments,
        "audit": {
            "page": page.get("page"),
            "layout_atoms": len(atoms),
            "concept_events": event_number,
            "assignment_status_counts": dict(sorted(audit_counts.items())),
            "duplicate_canonical_identities": {
                canonical: [target.key for target in members]
                for canonical, members in sorted(by_canonical.items())
                if len(members) > 1
            },
            "labels_read": False,
            "result_keys_used_for_assignment": False,
        },
    }


__all__ = [
    "CATEGORIES",
    "FEATURE_NAMES",
    "RELATIONS",
    "SIBLING_MARGIN",
    "build_layout_graph_target_features",
]
