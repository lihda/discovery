#!/usr/bin/env python3
"""Freeze label-blind event bags for single-target R3 multiple-instance learning.

The command accepts only the three-column identity, original-paper page text,
local-PDF membership, and label-free frozen knowledge/alignment artifacts. It
does not expose an input for outcomes, source-table evidence, predictions, or
evaluation reports.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Pattern, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
from experiment_r3_own_study_events import event_signals as derive_event_signals  # noqa: E402
from r3_scientific_evidence_chunks import (  # noqa: E402
    CATEGORIES,
    SCALES,
    EvidenceChunk,
    build_scientific_evidence_chunks,
    compile_category_patterns,
)


DEFAULT_IDENTITY = trace_freeze.DEFAULT_IDENTITY
DEFAULT_PAGE_CACHE = trace_freeze.DEFAULT_PAGE_CACHE
DEFAULT_PDF_DIR = trace_freeze.DEFAULT_PDF_DIR
DEFAULT_CONTEXT = (
    ROOT
    / "tmp/r3_all_labelled_pdf_grouped_oof_v1/trace_target_context_documents_v4.jsonl"
)
DEFAULT_FOLD_MANIFEST = (
    ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1/trace_refno_fold_manifest_v4.json"
)
DEFAULT_CONCEPT_SNAPSHOT = (
    ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1/trace_page_graph_concepts_v4.json"
)
DEFAULT_MIL_KB = ROOT / "data/knowledge/r3_evidence_mil_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_evidence_mil_single872_v1"

FEATURE_NAME = "r3_evidence_mil_bags_v1.npz"
RECEIPT_NAME = "r3_evidence_mil_receipt_v1.jsonl"
CANDIDATE_NAME = "r3_evidence_mil_candidate_freeze_v1.json"
MANIFEST_NAME = "r3_evidence_mil_bags_v1.manifest.json"

KB_SCHEMA = "r3_evidence_mil_v1"
FEATURE_SCHEMA = "r3_evidence_mil_bags_v1"
CANDIDATE_SCHEMA = "r3_evidence_mil_candidate_freeze_v1"
MANIFEST_SCHEMA = "r3_evidence_mil_bags_manifest_v1"
EXPECTED_SINGLE_TARGET_ROWS = 872
ALLOWED_SCALES = ("sentence_window", "paragraph")
ROUTES = (
    "lexical_anchor",
    "own_result_context",
    "counter_evidence",
    "uniform_fallback",
)
EXPECTED_ROUTE_QUOTAS = {
    "lexical_anchor": 32,
    "own_result_context": 16,
    "counter_evidence": 8,
    "uniform_fallback": 8,
}
COUNTER_GROUPS = (
    "demagnetization_only",
    "external_attribution",
    "future_recommendation",
)
SECTION_CLASSES = (
    "abstract",
    "introduction",
    "geology",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "other",
)
EVENT_BOOLEAN_FEATURES = (
    "first_person",
    "passive_execution",
    "procedure",
    "result_verb",
    "quantitative",
    "figure_table",
    "sample",
    "citation",
    "external_attribution",
    "definition",
    "future",
    "method_negation",
    "own_strong",
    "own_likely",
    "external",
)


class MilBagError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateInstance:
    chunk: EvidenceChunk
    section_class: str
    counter_counts: tuple[int, int, int]
    event_signals: Mapping[str, Any]
    trusted_categories: tuple[bool, bool, bool, bool]
    exclusion_codes: tuple[str, ...]


@dataclass(frozen=True)
class SelectedInstance:
    candidate: CandidateInstance
    primary_route: str
    eligible_routes: tuple[str, ...]


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def acquire_output_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".r3_evidence_mil_builder.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise MilBagError(f"another MIL bag builder owns {output_dir}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MilBagError(f"{label} must be a JSON object")
    return value


def load_mil_kb(path: Path) -> dict[str, Any]:
    payload = load_json(path, "MIL knowledge")
    if (
        payload.get("schema_version") != KB_SCHEMA
        or payload.get("status") != "FROZEN_LABEL_FREE_KNOWLEDGE"
    ):
        raise MilBagError("MIL knowledge has an invalid schema or status")
    boundary = payload.get("runtime_boundary") or {}
    if any(
        boundary.get(name) is not False
        for name in (
            "labels_read",
            "pmagresult_or_rockmag_read",
            "prior_predictions_read",
            "evaluation_results_read",
        )
    ):
        raise MilBagError("MIL knowledge violates the label-free boundary")
    if (
        boundary.get("target_context_used_only_for_alignment") is not True
        or boundary.get("resultno_used_only_for_alignment") is not True
    ):
        raise MilBagError("MIL knowledge changed an alignment-only input boundary")
    selection = payload.get("candidate_selection") or {}
    quotas = selection.get("quotas") or {}
    if (
        tuple(selection.get("allowed_scales") or ()) != ALLOWED_SCALES
        or tuple(selection.get("selection_order") or ()) != ROUTES
        or set(quotas) != set(ROUTES)
        or {name: int(quotas[name]) for name in ROUTES} != EXPECTED_ROUTE_QUOTAS
        or sum(int(quotas[name]) for name in ROUTES)
        != int(selection.get("max_instances_per_bag", -1))
        or int(selection.get("max_instances_per_bag", -1)) != 64
        or tuple(selection.get("lexical_category_round_robin") or ()) != CATEGORIES
        or selection.get("deduplication_key") != "normalized_text_sha256"
        or not set(map(clean, selection.get("own_result_sections") or ()))
        or not set(map(clean, selection.get("own_result_sections") or ())).issubset(
            set(SECTION_CLASSES)
        )
    ):
        raise MilBagError("MIL candidate-selection contract changed")
    chunking = payload.get("chunking") or {}
    scale_quotas = chunking.get("scale_quotas") or {}
    if (
        tuple(chunking.get("scales") or ()) != SCALES
        or set(scale_quotas) != set(SCALES)
        or any(int(scale_quotas[scale]) < 1 for scale in SCALES)
        or sum(int(scale_quotas[scale]) for scale in SCALES)
        > int(chunking.get("max_chunks_per_document", -1))
    ):
        raise MilBagError("MIL chunking contract changed")
    counter_groups = payload.get("counter_pattern_groups") or {}
    if set(counter_groups) != set(COUNTER_GROUPS) or any(
        not isinstance(counter_groups[name], list) or not counter_groups[name]
        for name in COUNTER_GROUPS
    ):
        raise MilBagError("MIL counter patterns are incomplete")
    for name in COUNTER_GROUPS:
        for expression in counter_groups[name]:
            re.compile(clean(expression), re.I)
    features = payload.get("instance_features") or {}
    if (
        tuple(features.get("section_classes") or ()) != SECTION_CLASSES
        or int(features.get("lexical_count_clip", 0)) < 1
        or int(features.get("counter_count_clip", 0)) < 1
        or int(features.get("word_count_log_reference", 0)) < 8
        or features.get("include_primary_route_one_hot") is not True
        or features.get("include_eligible_route_one_hot") is not True
    ):
        raise MilBagError("MIL instance-feature contract changed")
    return payload


def load_concept_snapshot(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path, "concept snapshot")
    if (
        payload.get("schema_version") != trace_freeze.CONCEPT_SCHEMA_VERSION
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise MilBagError("concept snapshot is not frozen V4 knowledge")
    concepts = payload.get("concepts") or []
    if not isinstance(concepts, list) or not concepts:
        raise MilBagError("concept snapshot contains no concepts")
    compile_category_patterns(concepts)
    return [dict(item) for item in concepts]


def validate_fold_context_alignment(
    records: Sequence[Mapping[str, str]], fold_path: Path, context_path: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = load_json(fold_path, "fold manifest")
    alignment = trace_freeze._record_alignment_sha256(records)
    if (
        payload.get("schema_version") != trace_freeze.FOLD_SCHEMA_VERSION
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or payload.get("labels_read") is not False
        or clean(payload.get("alignment_sha256")) != alignment
    ):
        raise MilBagError("fold manifest does not align with frozen V4 identity")
    rows = payload.get("rows") or []
    if len(rows) != len(records):
        raise MilBagError("fold rows do not align")
    folds = np.zeros(len(records), dtype=np.int8)
    document_groups = np.empty(len(records), dtype="U64")
    refno_folds: dict[str, set[int]] = {}
    group_folds: dict[str, set[int]] = {}
    for index, (record, row) in enumerate(zip(records, rows)):
        if (
            int(row.get("row_index", -1)) != index
            or clean(row.get("REFNO")) != clean(record.get("REFNO"))
            or clean(row.get("RESULTNO")) != clean(record.get("RESULTNO"))
        ):
            raise MilBagError("fold identity changed")
        fold = int(row.get("fold", 0))
        group = clean(row.get("document_sha256"))
        if fold < 1 or not group:
            raise MilBagError("fold row lacks a valid fold or document group")
        folds[index] = fold
        document_groups[index] = group
        refno_folds.setdefault(clean(record.get("REFNO")), set()).add(fold)
        group_folds.setdefault(group, set()).add(fold)
    if any(len(values) != 1 for values in refno_folds.values()):
        raise MilBagError("outer folds split a REFNO")
    if any(len(values) != 1 for values in group_folds.values()):
        raise MilBagError("outer folds split normalized document content")
    if hashlib.sha256(folds.tobytes()).hexdigest() != clean(
        payload.get("fold_id_sha256")
    ):
        raise MilBagError("fold-id hash changed")
    context_rows = 0
    with context_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, str) or not value.strip():
                raise MilBagError(
                    f"target context is invalid at line {line_number}"
                )
            context_rows += 1
    if context_rows != len(records):
        raise MilBagError("target context rows do not align")
    return folds, document_groups, {
        "alignment_sha256": alignment,
        "fold_rows": len(rows),
        "context_rows": context_rows,
        "fold_id_sha256": trace_freeze.array_sha256(folds),
        "document_group_sha256": trace_freeze.array_sha256(document_groups),
    }


def compile_counter_patterns(
    kb: Mapping[str, Any],
) -> dict[str, tuple[Pattern[str], ...]]:
    raw = kb.get("counter_pattern_groups") or {}
    return {
        name: tuple(re.compile(clean(value), re.I) for value in raw[name])
        for name in COUNTER_GROUPS
    }


def _section_class(value: str) -> str:
    folded = clean(value).casefold().replace("-", "_").replace(" ", "_")
    if folded == "abstract":
        return "abstract"
    if folded in {"introduction", "background", "summary"}:
        return "introduction"
    if folded in {"geology", "geological_setting"}:
        return "geology"
    if folded in {
        "method",
        "methods",
        "sampling",
        "laboratory_procedure",
        "rock_magnetism",
        "magnetic_mineralogy",
        "demagnetization",
    }:
        return "methods"
    if folded in {"result", "results", "results_and_discussion"}:
        return "results"
    if folded in {"discussion", "interpretation"}:
        return "discussion"
    if folded in {"conclusion", "conclusions"}:
        return "conclusion"
    return "other"


def _counter_counts(
    text: str, patterns: Mapping[str, Sequence[Pattern[str]]]
) -> tuple[int, int, int]:
    return tuple(
        min(99, sum(bool(pattern.search(text)) for pattern in patterns[name]))
        for name in COUNTER_GROUPS
    )  # type: ignore[return-value]


def _candidate_preference(candidate: CandidateInstance) -> tuple[Any, ...]:
    chunk = candidate.chunk
    return (
        -sum(candidate.trusted_categories),
        -sum(chunk.lexical_counts),
        -sum(value > 0 for value in chunk.lexical_counts),
        -int(candidate.event_signals["own_strong"]),
        -int(candidate.event_signals["own_likely"]),
        -int(chunk.own_study),
        -int(chunk.result_language),
        ALLOWED_SCALES.index(chunk.scale),
        chunk.page,
        chunk.chunk_id,
    )


def _trusted_categories(
    chunk: EvidenceChunk,
    section_class: str,
    counter_counts: tuple[int, int, int],
    signals: Mapping[str, Any],
) -> tuple[tuple[bool, bool, bool, bool], tuple[str, ...]]:
    blocked = bool(
        section_class == "references"
        or signals["external"]
        or signals["future"]
        or signals["method_negation"]
    )
    execution = bool(
        signals["first_person"]
        or signals["passive_execution"]
        or signals["procedure"]
    )
    assertion = bool(signals["result_verb"] or signals["quantitative"])
    owned = bool(signals["own_likely"] or execution)
    method, diagnostic, carrier, bridge = chunk.lexical_counts
    trusted = (
        bool(method and (owned or assertion) and not blocked),
        bool(diagnostic and assertion and not blocked),
        bool(carrier and (owned or assertion) and not blocked),
        bool(bridge and (owned or assertion) and not blocked),
    )
    exclusions = []
    if signals["external"]:
        exclusions.append("EXTERNAL")
    if signals["future"]:
        exclusions.append("FUTURE")
    if signals["method_negation"]:
        exclusions.append("METHOD_NEGATION")
    if counter_counts[0] and not any(chunk.lexical_counts):
        exclusions.append("DEMAG_ONLY")
    if signals["citation"] and not signals["own_likely"]:
        exclusions.append("CITATION_WITHOUT_OWNERSHIP")
    if not any(chunk.lexical_counts):
        exclusions.append("NO_SUPPORT_ANCHOR")
    return trusted, tuple(exclusions)


def prepare_candidate_pool(
    chunks: Sequence[EvidenceChunk],
    counter_patterns: Mapping[str, Sequence[Pattern[str]]],
) -> list[CandidateInstance]:
    by_text: dict[str, CandidateInstance] = {}
    for chunk in chunks:
        if chunk.scale not in ALLOWED_SCALES:
            continue
        section_class = _section_class(chunk.section)
        counter_counts = _counter_counts(chunk.text, counter_patterns)
        signals = derive_event_signals(chunk.text, section_class, False)
        trusted, exclusions = _trusted_categories(
            chunk, section_class, counter_counts, signals
        )
        candidate = CandidateInstance(
            chunk=chunk,
            section_class=section_class,
            counter_counts=counter_counts,
            event_signals=signals,
            trusted_categories=trusted,
            exclusion_codes=exclusions,
        )
        prior = by_text.get(chunk.text_sha256)
        if prior is None or _candidate_preference(candidate) < _candidate_preference(prior):
            by_text[chunk.text_sha256] = candidate
    return sorted(
        by_text.values(),
        key=lambda item: (
            item.chunk.page,
            ALLOWED_SCALES.index(item.chunk.scale),
            item.chunk.chunk_id,
        ),
    )


def _eligible_routes(
    candidate: CandidateInstance, own_result_sections: set[str]
) -> tuple[str, ...]:
    chunk = candidate.chunk
    routes = []
    if sum(chunk.lexical_counts) > 0:
        routes.append("lexical_anchor")
    if (
        candidate.event_signals["own_likely"]
        or candidate.event_signals["result_verb"]
        or candidate.section_class in own_result_sections
    ):
        routes.append("own_result_context")
    if sum(candidate.counter_counts) > 0:
        routes.append("counter_evidence")
    if not routes:
        routes.append("uniform_fallback")
    return tuple(routes)


def _even_indices(length: int, count: int) -> list[int]:
    if length <= 0 or count <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [length // 2]
    return sorted(
        {round(offset * (length - 1) / (count - 1)) for offset in range(count)}
    )


def select_mil_candidates(
    chunks: Sequence[EvidenceChunk], kb: Mapping[str, Any]
) -> tuple[list[SelectedInstance], dict[str, Any]]:
    selection = kb["candidate_selection"]
    quotas = {name: int(selection["quotas"][name]) for name in ROUTES}
    own_result_sections = set(map(clean, selection["own_result_sections"]))
    counter_patterns = compile_counter_patterns(kb)
    pool = prepare_candidate_pool(chunks, counter_patterns)
    selected: list[SelectedInstance] = []
    selected_ids: set[str] = set()

    def append_route(route: str, values: Sequence[CandidateInstance]) -> None:
        for candidate in values:
            if len([item for item in selected if item.primary_route == route]) >= quotas[route]:
                break
            if candidate.chunk.chunk_id in selected_ids:
                continue
            selected.append(
                SelectedInstance(
                    candidate=candidate,
                    primary_route=route,
                    eligible_routes=_eligible_routes(candidate, own_result_sections),
                )
            )
            selected_ids.add(candidate.chunk.chunk_id)

    lexical = [item for item in pool if sum(item.chunk.lexical_counts) > 0]
    category_queues = {
        category: sorted(
            [item for item in lexical if item.chunk.lexical(category) > 0],
            key=lambda item, category=category: (
                -item.chunk.lexical(category),
                -sum(value > 0 for value in item.chunk.lexical_counts),
                -sum(item.chunk.lexical_counts),
                -sum(item.trusted_categories),
                -int(item.event_signals["own_strong"]),
                -int(item.chunk.own_study),
                -int(item.chunk.result_language),
                int(item.chunk.citation_language),
                int(item.chunk.adverse_language),
                item.chunk.page,
                item.chunk.chunk_id,
            ),
        )
        for category in CATEGORIES
    }
    pointers = {category: 0 for category in CATEGORIES}
    round_robin: list[CandidateInstance] = []
    round_robin_ids: set[str] = set()
    while len(round_robin) < quotas["lexical_anchor"]:
        progressed = False
        for category in CATEGORIES:
            queue = category_queues[category]
            while (
                pointers[category] < len(queue)
                and queue[pointers[category]].chunk.chunk_id in round_robin_ids
            ):
                pointers[category] += 1
            if pointers[category] >= len(queue):
                continue
            candidate = queue[pointers[category]]
            pointers[category] += 1
            round_robin.append(candidate)
            round_robin_ids.add(candidate.chunk.chunk_id)
            progressed = True
            if len(round_robin) >= quotas["lexical_anchor"]:
                break
        if not progressed:
            break
    lexical_global = sorted(lexical, key=_candidate_preference)
    append_route("lexical_anchor", [*round_robin, *lexical_global])

    own_result = sorted(
        [
            item
            for item in pool
            if (
                item.event_signals["own_likely"]
                or item.event_signals["result_verb"]
                or item.section_class in own_result_sections
            )
        ],
        key=lambda item: (
            -int(item.event_signals["own_strong"]),
            -int(item.event_signals["own_likely"]),
            -int(item.chunk.own_study and item.chunk.result_language),
            -int(item.chunk.own_study),
            -int(item.chunk.result_language),
            -int(item.section_class in own_result_sections),
            -sum(item.chunk.lexical_counts),
            int(item.chunk.citation_language),
            int(item.chunk.adverse_language),
            item.chunk.page,
            item.chunk.chunk_id,
        ),
    )
    append_route("own_result_context", own_result)

    counter = sorted(
        [item for item in pool if sum(item.counter_counts) > 0],
        key=lambda item: (
            -sum(item.counter_counts),
            -int(item.chunk.adverse_language),
            -int(item.chunk.citation_language),
            sum(item.chunk.lexical_counts),
            item.chunk.page,
            item.chunk.chunk_id,
        ),
    )
    append_route("counter_evidence", counter)

    remaining = [item for item in pool if item.chunk.chunk_id not in selected_ids]
    fallback = [remaining[index] for index in _even_indices(len(remaining), quotas["uniform_fallback"])]
    append_route("uniform_fallback", fallback)
    maximum = int(selection["max_instances_per_bag"])
    if len(selected) > maximum or len(selected_ids) != len(selected):
        raise AssertionError("MIL candidate cap or deduplication failed")
    route_counts = Counter(item.primary_route for item in selected)
    return selected, {
        "eligible_chunks_before_cross_scale_dedup": sum(
            chunk.scale in ALLOWED_SCALES for chunk in chunks
        ),
        "candidate_pool_after_cross_scale_dedup": len(pool),
        "selected_instances": len(selected),
        "selected_route_counts": {
            name: int(route_counts.get(name, 0)) for name in ROUTES
        },
        "underfilled_route_quotas": {
            name: quotas[name] - int(route_counts.get(name, 0))
            for name in ROUTES
            if int(route_counts.get(name, 0)) < quotas[name]
        },
    }


def instance_feature_names() -> list[str]:
    return [
        *(f"lexical_{name}_count_clip" for name in CATEGORIES),
        "lexical_total_log",
        "lexical_category_coverage",
        *(f"scale_{name}" for name in ALLOWED_SCALES),
        *(f"section_{name}" for name in SECTION_CLASSES),
        "own_study_language",
        "result_language",
        "citation_language",
        "adverse_language",
        "own_and_result_language",
        "own_or_result_section",
        "support_without_citation",
        *(f"event_{name}" for name in EVENT_BOOLEAN_FEATURES),
        "event_score_scaled",
        *(f"trusted_{name}" for name in CATEGORIES),
        "trusted_category_fraction",
        "trusted_any",
        "support_blocked",
        *(f"counter_{name}_count_clip" for name in COUNTER_GROUPS),
        "counter_total_log",
        "counter_without_normative_support",
        "word_count_log_scaled",
        "page_position",
        *(f"primary_route_{name}" for name in ROUTES),
        *(f"eligible_route_{name}" for name in ROUTES),
    ]


def instance_feature_vector(
    selected: SelectedInstance, kb: Mapping[str, Any], maximum_page: int
) -> np.ndarray:
    candidate = selected.candidate
    chunk = candidate.chunk
    config = kb["instance_features"]
    lexical_clip = int(config["lexical_count_clip"])
    counter_clip = int(config["counter_count_clip"])
    word_reference = int(config["word_count_log_reference"])
    lexical_total = sum(chunk.lexical_counts)
    counter_total = sum(candidate.counter_counts)
    own_result_sections = set(kb["candidate_selection"]["own_result_sections"])
    values = [
        *(min(value, lexical_clip) / lexical_clip for value in chunk.lexical_counts),
        math.log1p(lexical_total) / math.log1p(99 * len(CATEGORIES)),
        sum(value > 0 for value in chunk.lexical_counts) / len(CATEGORIES),
        *(float(chunk.scale == name) for name in ALLOWED_SCALES),
        *(float(candidate.section_class == name) for name in SECTION_CLASSES),
        float(chunk.own_study),
        float(chunk.result_language),
        float(chunk.citation_language),
        float(chunk.adverse_language),
        float(chunk.own_study and chunk.result_language),
        float(
            chunk.own_study
            or chunk.result_language
            or candidate.section_class in own_result_sections
        ),
        float(lexical_total > 0 and not chunk.citation_language),
        *(float(bool(candidate.event_signals[name])) for name in EVENT_BOOLEAN_FEATURES),
        float(np.clip(float(candidate.event_signals["score"]) / 12.0, -1.0, 1.0)),
        *(float(value) for value in candidate.trusted_categories),
        sum(candidate.trusted_categories) / len(CATEGORIES),
        float(any(candidate.trusted_categories)),
        float(lexical_total > 0 and not any(candidate.trusted_categories)),
        *(min(value, counter_clip) / counter_clip for value in candidate.counter_counts),
        math.log1p(counter_total) / math.log1p(99 * len(COUNTER_GROUPS)),
        float(counter_total > 0 and lexical_total == 0),
        min(1.0, math.log1p(chunk.word_count) / math.log1p(word_reference)),
        min(1.0, max(0.0, chunk.page / max(1, maximum_page))),
        *(float(selected.primary_route == name) for name in ROUTES),
        *(float(name in selected.eligible_routes) for name in ROUTES),
    ]
    output = np.asarray(values, dtype=np.float32)
    if output.shape != (len(instance_feature_names()),) or not np.isfinite(output).all():
        raise AssertionError("MIL instance feature vector is invalid")
    return output


def build_document_bag(
    *,
    refno: str,
    pages: Sequence[Mapping[str, Any]],
    concept_patterns: Mapping[str, Sequence[Pattern[str]]],
    kb: Mapping[str, Any],
) -> tuple[list[SelectedInstance], np.ndarray, dict[str, Any]]:
    chunks, chunk_audit = build_scientific_evidence_chunks(
        refno, pages, concept_patterns, kb["chunking"]
    )
    selected, selection_audit = select_mil_candidates(chunks, kb)
    maximum_page = max(
        [int(page.get("page") or 0) for page in pages if isinstance(page, Mapping)]
        or [1]
    )
    names = instance_feature_names()
    matrix = (
        np.vstack(
            [instance_feature_vector(item, kb, maximum_page) for item in selected]
        ).astype(np.float32, copy=False)
        if selected
        else np.empty((0, len(names)), dtype=np.float32)
    )
    return selected, matrix, {
        "chunking": chunk_audit,
        "selection": selection_audit,
        "maximum_page": maximum_page,
    }


def _receipt_instance(
    selected: SelectedInstance, global_instance_index: int, bag_offset: int
) -> dict[str, Any]:
    candidate = selected.candidate
    chunk = candidate.chunk
    return {
        "global_instance_index": global_instance_index,
        "bag_instance_index": bag_offset,
        "chunk_id": chunk.chunk_id,
        "text_sha256": chunk.text_sha256,
        "page": chunk.page,
        "scale": chunk.scale,
        "section": chunk.section,
        "section_class": candidate.section_class,
        "word_count": chunk.word_count,
        "lexical_counts": {
            name: chunk.lexical(name) for name in CATEGORIES
        },
        "counter_counts": {
            name: candidate.counter_counts[index]
            for index, name in enumerate(COUNTER_GROUPS)
        },
        "language_flags": {
            "own_study": chunk.own_study,
            "result": chunk.result_language,
            "citation": chunk.citation_language,
            "adverse": chunk.adverse_language,
        },
        "event_signals": dict(candidate.event_signals),
        "trusted_categories": {
            name: bool(candidate.trusted_categories[index])
            for index, name in enumerate(CATEGORIES)
        },
        "exclusion_codes": list(candidate.exclusion_codes),
        "primary_route": selected.primary_route,
        "eligible_routes": list(selected.eligible_routes),
        "text": chunk.text,
    }


def build_frozen_mil_bags(
    *,
    identity_path: Path,
    page_cache_path: Path,
    pdf_dir: Path,
    context_path: Path,
    fold_path: Path,
    concept_path: Path,
    knowledge_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    paths = {
        "identity": identity_path.resolve(),
        "page_cache": page_cache_path.resolve(),
        "pdf_dir": pdf_dir.resolve(),
        "context": context_path.resolve(),
        "fold_manifest": fold_path.resolve(),
        "concept_snapshot": concept_path.resolve(),
        "knowledge": knowledge_path.resolve(),
        "output_dir": output_dir.resolve(),
    }
    output_lock = acquire_output_lock(paths["output_dir"])
    try:
        all_records = trace_freeze.load_three_column_identity(paths["identity"])
        records, coverage, pdf_membership_sha256 = trace_freeze.select_pdf_backed_records(
            all_records, paths["pdf_dir"]
        )
        folds, document_groups, alignment_audit = validate_fold_context_alignment(
            records, paths["fold_manifest"], paths["context"]
        )
        multiplicity = Counter(record["REFNO"] for record in all_records)
        single_positions = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if multiplicity[record["REFNO"]] == 1
            ],
            dtype=np.int32,
        )
        if len(single_positions) != EXPECTED_SINGLE_TARGET_ROWS:
            raise MilBagError(
                "single-target scope changed: expected "
                f"{EXPECTED_SINGLE_TARGET_ROWS}, got {len(single_positions)}"
            )
        single_records = [records[int(index)] for index in single_positions]
        single_folds = folds[single_positions]
        single_groups = document_groups[single_positions]
        wanted_refnos = {record["REFNO"] for record in single_records}
        page_rows = trace_freeze.load_sanitized_page_cache(
            paths["page_cache"], wanted_refnos
        )
        kb = load_mil_kb(paths["knowledge"])
        concepts = load_concept_snapshot(paths["concept_snapshot"])
        concept_patterns = compile_category_patterns(concepts)
        feature_names = instance_feature_names()
        code_paths = [
            Path(__file__).resolve(),
            (SCRIPTS / "r3_scientific_evidence_chunks.py").resolve(),
            (SCRIPTS / "build_r3_trace_page_graph_features.py").resolve(),
            (SCRIPTS / "paper_text_retrieval.py").resolve(),
            (SCRIPTS / "experiment_r3_own_study_events.py").resolve(),
        ]
        candidate_path = paths["output_dir"] / CANDIDATE_NAME
        candidate = {
            "schema_version": CANDIDATE_SCHEMA,
            "status": "FROZEN_BEFORE_LABEL_ACCESS",
            "primary_scope": "all 872 PDF-backed single-target papers at full coverage",
            "complete_alignment_sha256": alignment_audit["alignment_sha256"],
            "single_target_rows": len(single_records),
            "single_source_row_indices_sha256": trace_freeze.array_sha256(
                single_positions
            ),
            "single_fold_ids_sha256": trace_freeze.array_sha256(single_folds),
            "single_document_groups_sha256": trace_freeze.array_sha256(single_groups),
            "candidate_selection": kb["candidate_selection"],
            "instance_feature_names": feature_names,
            "instance_feature_names_sha256": json_sha256(feature_names),
            "inputs": {
                "identity": {
                    "path": str(paths["identity"]),
                    "sha256": trace_freeze.file_sha256(paths["identity"]),
                },
                "page_cache": {
                    "path": str(paths["page_cache"]),
                    "sha256": trace_freeze.file_sha256(paths["page_cache"]),
                },
                "pdf_membership": {
                    "path": str(paths["pdf_dir"]),
                    "sha256": pdf_membership_sha256,
                },
                "target_context": {
                    "path": str(paths["context"]),
                    "sha256": trace_freeze.file_sha256(paths["context"]),
                    "usage": "alignment_only",
                },
                "fold_manifest": {
                    "path": str(paths["fold_manifest"]),
                    "sha256": trace_freeze.file_sha256(paths["fold_manifest"]),
                },
                "concept_snapshot": {
                    "path": str(paths["concept_snapshot"]),
                    "sha256": trace_freeze.file_sha256(paths["concept_snapshot"]),
                },
                "knowledge": {
                    "path": str(paths["knowledge"]),
                    "sha256": trace_freeze.file_sha256(paths["knowledge"]),
                },
                "code": [
                    {
                        "path": str(path),
                        "sha256": trace_freeze.file_sha256(path),
                    }
                    for path in code_paths
                ],
            },
            "runtime_boundary": kb["runtime_boundary"],
        }
        atomic_json(candidate_path, candidate)

        bag_indptr = [0]
        matrices: list[np.ndarray] = []
        receipts: list[dict[str, Any]] = []
        instance_bag_indices: list[int] = []
        instance_pages: list[int] = []
        instance_scale_ids: list[int] = []
        instance_route_ids: list[int] = []
        instance_chunk_ids: list[str] = []
        instance_text_hashes: list[str] = []
        total_route_counts: Counter[str] = Counter()
        papers_without_instances = 0
        for bag_index, (source_index, record) in enumerate(
            zip(single_positions, single_records)
        ):
            selected, matrix, audit = build_document_bag(
                refno=record["REFNO"],
                pages=page_rows[record["REFNO"]]["pages"],
                concept_patterns=concept_patterns,
                kb=kb,
            )
            start = bag_indptr[-1]
            matrices.append(matrix)
            bag_indptr.append(start + len(selected))
            papers_without_instances += int(not selected)
            for offset, item in enumerate(selected):
                chunk = item.candidate.chunk
                instance_bag_indices.append(bag_index)
                instance_pages.append(chunk.page)
                instance_scale_ids.append(ALLOWED_SCALES.index(chunk.scale))
                instance_route_ids.append(ROUTES.index(item.primary_route))
                instance_chunk_ids.append(chunk.chunk_id)
                instance_text_hashes.append(chunk.text_sha256)
                total_route_counts[item.primary_route] += 1
            receipts.append(
                {
                    "schema_version": "r3_evidence_mil_bag_receipt_v1",
                    "bag_index": bag_index,
                    "source_row_index": int(source_index),
                    "REFNO": record["REFNO"],
                    "RESULTNO": record["RESULTNO"],
                    "ROCKNAME": record["ROCKNAME"],
                    "fold": int(single_folds[bag_index]),
                    "document_group": clean(single_groups[bag_index]),
                    "bag_start": start,
                    "bag_end": bag_indptr[-1],
                    "audit": audit,
                    "instances": [
                        _receipt_instance(item, start + offset, offset)
                        for offset, item in enumerate(selected)
                    ],
                }
            )
            if (bag_index + 1) % 100 == 0:
                print(
                    f"MIL bag freeze: papers={bag_index + 1}/{len(single_records)}",
                    flush=True,
                )
        instance_features = (
            np.vstack(matrices).astype(np.float32, copy=False)
            if matrices
            else np.empty((0, len(feature_names)), dtype=np.float32)
        )
        indptr = np.asarray(bag_indptr, dtype=np.int64)
        if (
            indptr.shape != (len(single_records) + 1,)
            or int(indptr[-1]) != len(instance_features)
            or not np.isfinite(instance_features).all()
            or np.any(np.diff(indptr) > 64)
        ):
            raise AssertionError("frozen MIL bag arrays are invalid")
        arrays = {
            "bag_indptr": indptr,
            "instance_features": instance_features,
            "fold_ids": single_folds.astype(np.int8, copy=False),
            "source_row_indices": single_positions,
            "refnos": np.asarray([row["REFNO"] for row in single_records], dtype="U32"),
            "resultnos": np.asarray(
                [row["RESULTNO"] for row in single_records], dtype="U32"
            ),
            "rocknames": np.asarray(
                [row["ROCKNAME"] for row in single_records], dtype="U512"
            ),
            "document_groups": single_groups.astype("U64", copy=False),
            "instance_bag_indices": np.asarray(instance_bag_indices, dtype=np.int32),
            "instance_pages": np.asarray(instance_pages, dtype=np.int32),
            "instance_scale_ids": np.asarray(instance_scale_ids, dtype=np.int8),
            "instance_route_ids": np.asarray(instance_route_ids, dtype=np.int8),
            "instance_chunk_ids": np.asarray(instance_chunk_ids, dtype="U64"),
            "instance_text_sha256": np.asarray(instance_text_hashes, dtype="U64"),
        }
        feature_path = paths["output_dir"] / FEATURE_NAME
        receipt_path = paths["output_dir"] / RECEIPT_NAME
        atomic_npz(feature_path, **arrays)
        atomic_jsonl(receipt_path, receipts)
        array_hashes = {
            name: trace_freeze.array_sha256(value) for name, value in arrays.items()
        }
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "feature_schema_version": FEATURE_SCHEMA,
            "status": "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS",
            "bags": len(single_records),
            "instances": len(instance_features),
            "papers_without_instances": papers_without_instances,
            "maximum_instances_in_bag": int(np.diff(indptr).max(initial=0)),
            "instance_feature_count": len(feature_names),
            "instance_feature_names": feature_names,
            "instance_feature_names_sha256": json_sha256(feature_names),
            "selection_route_counts": {
                name: int(total_route_counts.get(name, 0)) for name in ROUTES
            },
            "coverage": coverage,
            "alignment": alignment_audit,
            "array_sha256": array_hashes,
            "outputs": {
                "bags": {
                    "path": str(feature_path),
                    "sha256": trace_freeze.file_sha256(feature_path),
                },
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": trace_freeze.file_sha256(receipt_path),
                },
                "candidate_freeze": {
                    "path": str(candidate_path),
                    "sha256": trace_freeze.file_sha256(candidate_path),
                },
            },
            "inputs": candidate["inputs"],
            "candidate_freeze_sha256": trace_freeze.file_sha256(candidate_path),
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
            },
            "runtime_boundary": candidate["runtime_boundary"],
        }
        manifest_path = paths["output_dir"] / MANIFEST_NAME
        atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "bags": len(single_records),
                    "instances": len(instance_features),
                    "instance_features": list(instance_features.shape),
                },
                indent=2,
            ),
            flush=True,
        )
        return {
            "features": feature_path,
            "receipt": receipt_path,
            "candidate_freeze": candidate_path,
            "manifest": manifest_path,
        }
    finally:
        output_lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--target-context", type=Path, default=DEFAULT_CONTEXT)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--concept-snapshot", type=Path, default=DEFAULT_CONCEPT_SNAPSHOT)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_MIL_KB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_frozen_mil_bags(
        identity_path=args.identity,
        page_cache_path=args.page_cache,
        pdf_dir=args.pdf_dir,
        context_path=args.target_context,
        fold_path=args.fold_manifest,
        concept_path=args.concept_snapshot,
        knowledge_path=args.knowledge,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ALLOWED_SCALES",
    "COUNTER_GROUPS",
    "MilBagError",
    "ROUTES",
    "SelectedInstance",
    "build_document_bag",
    "build_frozen_mil_bags",
    "build_parser",
    "compile_counter_patterns",
    "instance_feature_names",
    "instance_feature_vector",
    "load_mil_kb",
    "prepare_candidate_pool",
    "select_mil_candidates",
    "validate_fold_context_alignment",
]
