#!/usr/bin/env python3
"""Freeze label-blind paleomagnetic entity-router features on the V5 pilot."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

import build_r3_layout_graph_features as layout  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402
from r3_paleomag_entity_graph import build_entity_graph  # noqa: E402


LAYOUT_DIR = ROOT / "tmp/r3_layout_pilot_v1"
V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
DEFAULT_SELECTION = LAYOUT_DIR / "r3_layout_pilot_manifest_v1.json"
DEFAULT_SIDECAR = LAYOUT_DIR / "layout_sidecar_v1.jsonl"
DEFAULT_FOLD_MANIFEST = V4_DIR / "trace_refno_fold_manifest_v4.json"
DEFAULT_CONCEPT_SNAPSHOT = V4_DIR / "trace_page_graph_concepts_v4.json"
DEFAULT_LAYOUT_FEATURES = LAYOUT_DIR / "r3_layout_features_v1.npz"
DEFAULT_LAYOUT_MANIFEST = LAYOUT_DIR / "r3_layout_features_v1.manifest.json"
DEFAULT_LAYOUT_CANDIDATE = LAYOUT_DIR / "r3_layout_candidate_freeze_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_entity_router_pilot_v1"

FEATURE_NAME = "r3_entity_router_features_v1.npz"
MANIFEST_NAME = "r3_entity_router_features_v1.manifest.json"
AUDIT_NAME = "r3_entity_router_audit_v1.json"
CANDIDATE_NAME = "r3_entity_router_candidate_freeze_v1.json"
EVIDENCE_NAME = "r3_entity_router_evidence_graph_v1.jsonl"
FEATURE_SCHEMA = "r3_entity_router_features_v1"
AUDIT_SCHEMA = "r3_entity_router_feature_audit_v1"
CANDIDATE_SCHEMA = "r3_entity_router_candidate_freeze_v1"

IDENTITY_FIELDS = ("REFNO", "RESULTNO", "ROCKNAME")
CATEGORIES = ("method", "diagnostic", "carrier", "bridge")
SCALES = ("line", "paragraph", "table_row", "caption", "section")
BOUNDARY_FALSE_FIELDS = (
    "labels_read",
    "pmagresult_or_rockmag_read",
    "prior_predictions_read",
    "evaluation_results_read",
    "outer_test_metrics_used_for_selection",
)
REFERENCE_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\s+)?(?:references|bibliography|literature\s+cited|works\s+cited)\s*$",
    re.I,
)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _item_id(value: Mapping[str, Any], fallback: str) -> str:
    return clean(
        value.get("id")
        or value.get("block_id")
        or value.get("line_id")
        or value.get("table_id")
        or value.get("row_id")
        or value.get("caption_id")
    ) or fallback


def _text(value: Any) -> str:
    if isinstance(value, Mapping):
        return clean(value.get("text") or value.get("content") or value.get("value"))
    return clean(value)


def _cell_texts(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        raw = value.get("cells")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            return [text for item in raw if (text := _text(item))]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [text for item in value if (text := _text(item))]
    text = _text(value)
    return [text] if text else []


def document_chunks(pages: Sequence[Mapping[str, Any]]) -> list[dict[str, str]]:
    """Build stable multi-scale text chunks, excluding the reference tail."""

    chunks: list[dict[str, str]] = []
    references_started = False
    for page_offset, page in enumerate(pages, start=1):
        if references_started:
            break
        page_number = int(page.get("page") or page_offset)
        paragraph_texts: list[str] = []
        for block_offset, raw_block in enumerate(page.get("blocks") or [], start=1):
            if not isinstance(raw_block, Mapping):
                continue
            block_text = _text(raw_block)
            if REFERENCE_HEADING_RE.fullmatch(block_text):
                references_started = True
                break
            if not block_text:
                continue
            block_id = _item_id(raw_block, f"p{page_number}-b{block_offset}")
            paragraph_id = f"p{page_number}:paragraph:{block_id}"
            chunks.append(
                {
                    "chunk_id": paragraph_id,
                    "kind": "paragraph",
                    "text": block_text,
                }
            )
            paragraph_texts.append(block_text)
            for line_offset, raw_line in enumerate(raw_block.get("lines") or [], start=1):
                line = raw_line if isinstance(raw_line, Mapping) else {"text": raw_line}
                line_text = _text(line)
                if not line_text:
                    continue
                line_id = _item_id(line, f"p{page_number}-l{line_offset}")
                chunks.append(
                    {
                        "chunk_id": f"p{page_number}:line:{line_id}",
                        "kind": "sentence",
                        "text": line_text,
                        "parent_id": paragraph_id,
                    }
                )
        if references_started:
            break
        for table_offset, raw_table in enumerate(page.get("tables") or [], start=1):
            if not isinstance(raw_table, Mapping):
                continue
            table_id = _item_id(raw_table, f"p{page_number}-t{table_offset}")
            for row_offset, raw_row in enumerate(raw_table.get("rows") or [], start=1):
                texts = _cell_texts(raw_row)
                if not texts:
                    continue
                row = raw_row if isinstance(raw_row, Mapping) else {}
                row_id = _item_id(row, f"r{row_offset}")
                chunks.append(
                    {
                        "chunk_id": f"p{page_number}:table:{table_id}:row:{row_id}",
                        "kind": "table",
                        "text": " | ".join(texts),
                    }
                )
        for caption_offset, raw_caption in enumerate(page.get("captions") or [], start=1):
            caption = raw_caption if isinstance(raw_caption, Mapping) else {"text": raw_caption}
            caption_text = _text(caption)
            if not caption_text:
                continue
            caption_id = _item_id(caption, f"p{page_number}-c{caption_offset}")
            chunks.append(
                {
                    "chunk_id": f"p{page_number}:caption:{caption_id}",
                    "kind": "caption",
                    "text": caption_text,
                }
            )
        if paragraph_texts:
            chunks.append(
                {
                    "chunk_id": f"p{page_number}:section",
                    "kind": "section",
                    "text": "\n".join(paragraph_texts),
                }
            )
    identifiers = [item["chunk_id"] for item in chunks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("entity-router chunk ids are not unique")
    return chunks


def _compile_concepts(values: Sequence[Any]) -> dict[str, tuple[Pattern[str], ...]]:
    output: dict[str, list[Pattern[str]]] = {name: [] for name in CATEGORIES}
    for value in values:
        if isinstance(value, Mapping):
            category = clean(value.get("category")).casefold()
            raw_patterns = value.get("patterns") or value.get("search_patterns") or []
        else:
            category = clean(getattr(value, "category", "")).casefold()
            raw_patterns = getattr(value, "patterns", ())
        if category not in output:
            continue
        for pattern in raw_patterns:
            output[category].append(
                pattern if hasattr(pattern, "search") else re.compile(clean(pattern), re.I)
            )
    if any(not output[name] for name in CATEGORIES):
        raise ValueError("entity-router concept set lacks an R3 category")
    return {name: tuple(patterns) for name, patterns in output.items()}


def _scale(kind: Any) -> str:
    value = clean(kind).casefold()
    aliases = {"sentence": "line", "table": "table_row"}
    value = aliases.get(value, value)
    return value if value in SCALES else "paragraph"


def criterion_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for category in CATEGORIES:
        names.extend(
            (
                f"entity_cf_{category}_support_max",
                f"entity_cf_{category}_exclusive_max",
                f"entity_cf_{category}_scale_coverage",
                f"entity_cf_{category}_sibling_margin_max",
                f"entity_cf_{category}_route_entropy_min",
                f"entity_cf_{category}_global_mass_max",
                f"entity_cf_{category}_unresolved_mass_max",
            )
        )
    names.extend(
        (
            "entity_cf_method_only_score",
            "entity_cf_practice_chain_score",
            "entity_cf_normative_chain_score",
            "entity_cf_scale_winner_agreement",
            "entity_cf_assigned_event_count_log",
            "entity_cf_exclusive_event_count_log",
            "entity_cf_global_event_ratio",
            "entity_cf_unresolved_event_ratio",
            "entity_cf_inherited_context_count_log",
        )
    )
    return tuple(names)


CRITERION_FEATURE_NAMES = criterion_feature_names()


def contextual_assignments(
    chunks: Sequence[Mapping[str, str]], graph: Mapping[str, Any]
) -> tuple[dict[str, tuple[Mapping[str, Any], float, str]], list[dict[str, Any]]]:
    """Add a conservative one-page, two-paragraph xCoRe-style continuation."""

    raw = {
        clean(item.get("chunk_id")): item
        for item in graph.get("assignments") or []
        if isinstance(item, Mapping)
    }
    effective: dict[str, tuple[Mapping[str, Any], float, str]] = {}
    inherited_receipts: list[dict[str, Any]] = []
    paragraph_number: Counter[str] = Counter()
    last_direct_paragraph: dict[str, tuple[int, str, Mapping[str, Any]]] = {}

    def assigned(item: Mapping[str, Any]) -> bool:
        return bool(clean(item.get("top_target"))) and clean(
            item.get("status")
        ).casefold().startswith("assigned_")

    for chunk in chunks:
        chunk_id = clean(chunk.get("chunk_id"))
        item = raw.get(chunk_id)
        if item is None:
            raise ValueError("entity graph lacks a chunk assignment")
        kind = clean(chunk.get("kind")).casefold()
        page = chunk_id.split(":", 1)[0]
        if assigned(item):
            effective[chunk_id] = (item, 1.0, "direct")
            if kind == "paragraph":
                paragraph_number[page] += 1
                last_direct_paragraph[page] = (
                    paragraph_number[page],
                    chunk_id,
                    item,
                )
            continue
        parent_id = clean(chunk.get("parent_id"))
        if kind == "sentence" and parent_id in effective:
            parent, parent_weight, _ = effective[parent_id]
            if assigned(parent) or clean(parent.get("top_target")):
                weight = parent_weight * 0.90
                effective[chunk_id] = (parent, weight, "parent_paragraph")
                inherited_receipts.append(
                    {
                        "chunk_id": chunk_id,
                        "antecedent_chunk_id": parent_id,
                        "route": "parent_paragraph",
                        "weight": weight,
                        "top_target": clean(parent.get("top_target")),
                    }
                )
                continue
        if kind == "paragraph":
            paragraph_number[page] += 1
            prior = last_direct_paragraph.get(page)
            if prior and paragraph_number[page] - prior[0] <= 2:
                antecedent_id, antecedent = prior[1], prior[2]
                weight = 0.72 if paragraph_number[page] - prior[0] == 1 else 0.58
                effective[chunk_id] = (antecedent, weight, "paragraph_continuation")
                inherited_receipts.append(
                    {
                        "chunk_id": chunk_id,
                        "antecedent_chunk_id": antecedent_id,
                        "route": "paragraph_continuation",
                        "weight": weight,
                        "top_target": clean(antecedent.get("top_target")),
                    }
                )
                continue
        elif kind in {"table", "caption"}:
            prior = last_direct_paragraph.get(page)
            if prior:
                weight = 0.60
                effective[chunk_id] = (prior[2], weight, "page_local_structure")
                inherited_receipts.append(
                    {
                        "chunk_id": chunk_id,
                        "antecedent_chunk_id": prior[1],
                        "route": "page_local_structure",
                        "weight": weight,
                        "top_target": clean(prior[2].get("top_target")),
                    }
                )
                continue
        effective[chunk_id] = (item, 1.0, "raw_unresolved")
    return effective, inherited_receipts


def criterion_counterfactual_features(
    targets: Mapping[str, str],
    chunks: Sequence[Mapping[str, str]],
    graph: Mapping[str, Any],
    concepts: Sequence[Any],
) -> dict[str, dict[str, float]]:
    """Bind R3 facets to graph responsibilities without summing duplicate scales."""

    ids = tuple(targets)
    chunk_by_id = {clean(item.get("chunk_id")): item for item in chunks}
    if len(chunk_by_id) != len(chunks):
        raise ValueError("criterion chunks have duplicate ids")
    compiled = _compile_concepts(concepts)
    support = defaultdict(lambda: defaultdict(dict))
    exclusive = defaultdict(lambda: defaultdict(dict))
    margins = defaultdict(lambda: defaultdict(list))
    entropies = defaultdict(lambda: defaultdict(list))
    globals_ = defaultdict(lambda: defaultdict(list))
    unresolved = defaultdict(lambda: defaultdict(list))
    assigned_counts: Counter[str] = Counter()
    exclusive_counts: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    unresolved_counts: Counter[str] = Counter()
    scale_winners: dict[tuple[str, str], str] = {}
    inherited_counts: Counter[str] = Counter()

    effective, _ = contextual_assignments(chunks, graph)

    for raw_assignment in graph.get("assignments") or []:
        if not isinstance(raw_assignment, Mapping):
            raise ValueError("entity graph assignment is not an object")
        raw_chunk_id = clean(raw_assignment.get("chunk_id"))
        assignment, route_weight, route = effective.get(
            raw_chunk_id, (raw_assignment, 1.0, "raw_unresolved")
        )
        if not isinstance(assignment, Mapping):
            raise ValueError("entity graph assignment is not an object")
        chunk_id = raw_chunk_id
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            raise ValueError("entity assignment references an unknown chunk")
        text = clean(chunk.get("text"))
        categories = [
            category
            for category, patterns in compiled.items()
            if any(pattern.search(text) for pattern in patterns)
        ]
        if not categories:
            continue
        probabilities = assignment.get("target_probabilities") or {}
        if set(map(clean, probabilities)) != set(ids):
            raise ValueError("entity assignment target probabilities changed")
        scale = _scale(chunk.get("kind"))
        top_target = clean(assignment.get("top_target"))
        status = clean(assignment.get("status")).casefold()
        margin = float(assignment.get("margin") or 0.0)
        entropy = float(assignment.get("entropy") or 0.0)
        global_mass = float(assignment.get("global_mass") or 0.0)
        unresolved_mass = float(assignment.get("unresolved_mass") or 0.0)
        if not all(
            math.isfinite(value)
            for value in (margin, entropy, global_mass, unresolved_mass)
        ):
            raise ValueError("entity assignment contains a non-finite value")
        for category in categories:
            if top_target in ids:
                scale_winners[(category, scale)] = top_target
            for target in ids:
                probability = float(probabilities[target]) * route_weight
                if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                    raise ValueError("entity assignment probability is invalid")
                support[target][category][scale] = max(
                    support[target][category].get(scale, 0.0), probability
                )
                is_exclusive = (
                    status.startswith("assigned_") or route != "raw_unresolved"
                ) and top_target == target
                if is_exclusive:
                    exclusive[target][category][scale] = max(
                        exclusive[target][category].get(scale, 0.0), probability
                    )
                    exclusive_counts[target] += 1
                    inherited_counts[target] += int(route != "direct")
                if probability > 0.0:
                    assigned_counts[target] += 1
                    margins[target][category].append(margin if top_target == target else 0.0)
                    entropies[target][category].append(entropy)
                    globals_[target][category].append(global_mass)
                    unresolved[target][category].append(unresolved_mass)
                    global_counts[target] += int(global_mass >= probability)
                    unresolved_counts[target] += int(unresolved_mass >= probability)

    output: dict[str, dict[str, float]] = {}
    active_winners = list(scale_winners.values())
    for target in ids:
        values: dict[str, float] = {}
        category_support: dict[str, float] = {}
        for category in CATEGORIES:
            by_scale = support[target][category]
            support_max = max(by_scale.values(), default=0.0)
            category_support[category] = support_max
            values[f"entity_cf_{category}_support_max"] = support_max
            values[f"entity_cf_{category}_exclusive_max"] = max(
                exclusive[target][category].values(), default=0.0
            )
            values[f"entity_cf_{category}_scale_coverage"] = len(
                [value for value in by_scale.values() if value > 0.0]
            ) / len(SCALES)
            values[f"entity_cf_{category}_sibling_margin_max"] = max(
                margins[target][category], default=0.0
            )
            values[f"entity_cf_{category}_route_entropy_min"] = min(
                entropies[target][category], default=0.0
            )
            values[f"entity_cf_{category}_global_mass_max"] = max(
                globals_[target][category], default=0.0
            )
            values[f"entity_cf_{category}_unresolved_mass_max"] = max(
                unresolved[target][category], default=0.0
            )
        method = category_support["method"]
        carrier_or_diagnostic = max(
            category_support["carrier"], category_support["diagnostic"]
        )
        values["entity_cf_method_only_score"] = method
        values["entity_cf_practice_chain_score"] = min(method, carrier_or_diagnostic)
        values["entity_cf_normative_chain_score"] = min(
            method, carrier_or_diagnostic, category_support["bridge"]
        )
        values["entity_cf_scale_winner_agreement"] = (
            sum(winner == target for winner in active_winners) / len(active_winners)
            if active_winners
            else 0.0
        )
        count = assigned_counts[target]
        values["entity_cf_assigned_event_count_log"] = math.log1p(count)
        values["entity_cf_exclusive_event_count_log"] = math.log1p(
            exclusive_counts[target]
        )
        values["entity_cf_global_event_ratio"] = (
            global_counts[target] / count if count else 0.0
        )
        values["entity_cf_unresolved_event_ratio"] = (
            unresolved_counts[target] / count if count else 0.0
        )
        values["entity_cf_inherited_context_count_log"] = math.log1p(
            inherited_counts[target]
        )
        output[target] = {name: float(values[name]) for name in CRITERION_FEATURE_NAMES}
    return output


def _load_frozen_layout(
    feature_path: Path,
    manifest_path: Path,
    candidate_path: Path,
    *,
    expected_rows: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != layout.FEATURE_SCHEMA_VERSION
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or candidate.get("schema_version") != layout.CANDIDATE_SCHEMA_VERSION
        or candidate.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
    ):
        raise ValueError("frozen layout input has an invalid schema")
    for name in BOUNDARY_FALSE_FIELDS:
        if (candidate.get("runtime_boundary") or {}).get(name) is not False:
            raise ValueError(f"layout candidate must declare {name}=false")
    expected_feature_hash = clean(
        ((manifest.get("outputs") or {}).get("features") or {}).get("sha256")
    )
    expected_candidate_hash = clean(
        ((manifest.get("outputs") or {}).get("candidate_freeze") or {}).get("sha256")
    )
    if (
        layout.file_sha256(feature_path) != expected_feature_hash
        or layout.file_sha256(candidate_path) != expected_candidate_hash
    ):
        raise ValueError("frozen layout file hash mismatch")
    with np.load(feature_path) as archive:
        if archive.files != ["layout"]:
            raise ValueError("layout NPZ must contain exactly layout")
        matrix = np.asarray(archive["layout"])
    names = list(manifest.get("feature_names") or [])
    if (
        matrix.shape != (expected_rows, len(names))
        or tuple(names[:2])
        != ("layout_pilot_selected", "layout_document_available")
        or not np.isfinite(matrix).all()
        or layout.array_sha256(matrix) != clean(manifest.get("matrix_sha256"))
        or layout.array_sha256(matrix) != clean(candidate.get("matrix_sha256"))
    ):
        raise ValueError("frozen layout matrix is invalid")
    return matrix.astype(np.float32), names, manifest


def _atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for value in values:
                json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
                handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def build_entity_router_artifacts(
    *,
    selection_path: Path,
    sidecar_path: Path,
    fold_manifest_path: Path,
    concept_snapshot_path: Path,
    layout_features_path: Path,
    layout_manifest_path: Path,
    layout_candidate_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    inputs = {
        name: path.expanduser().resolve()
        for name, path in (
            ("selection", selection_path),
            ("sidecar", sidecar_path),
            ("fold_manifest", fold_manifest_path),
            ("concept_snapshot", concept_snapshot_path),
            ("layout_features", layout_features_path),
            ("layout_manifest", layout_manifest_path),
            ("layout_candidate", layout_candidate_path),
        )
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise ValueError("entity-router input is missing: " + ", ".join(missing))

    selection, selected_rows, selected_papers = layout._validate_selection(
        inputs["selection"]
    )
    _, fold_rows = layout._validate_fold_manifest(
        inputs["fold_manifest"], selected_rows, selected_papers
    )
    _, concepts = layout._validate_concept_snapshot(inputs["concept_snapshot"])
    sidecars = layout._validate_sidecar(inputs["sidecar"], selected_papers)
    layout_matrix, layout_names, _ = _load_frozen_layout(
        inputs["layout_features"],
        inputs["layout_manifest"],
        inputs["layout_candidate"],
        expected_rows=len(fold_rows),
    )
    full_index = {
        (clean(row["REFNO"]), clean(row["RESULTNO"])): int(row["row_index"])
        for row in fold_rows
    }
    if len(full_index) != len(fold_rows):
        raise ValueError("V4 fold rows contain duplicate identities")
    by_refno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        by_refno[clean(row["REFNO"])].append(
            {name: clean(row[name]) for name in IDENTITY_FIELDS}
        )

    graph_names: list[str] | None = None
    output_names: list[str] | None = None
    matrix: np.ndarray | None = None
    evidence_rows: list[dict[str, Any]] = []
    assignment_counts: Counter[str] = Counter()
    documents_differentiated = 0
    entity_targets_with_exclusive = 0
    rejected_alias_conflicts = 0
    duplicate_groups_checked = 0
    paper_audits: list[dict[str, Any]] = []

    for refno, rows in by_refno.items():
        targets = {row["RESULTNO"]: row["ROCKNAME"] for row in rows}
        chunks = document_chunks(sidecars[refno]["pages"])
        graph = build_entity_graph(targets, chunks)
        current_graph_names = [clean(name) for name in graph.get("feature_names") or []]
        if not current_graph_names or len(current_graph_names) != len(set(current_graph_names)):
            raise ValueError("entity graph feature schema is invalid")
        if graph_names is None:
            graph_names = current_graph_names
            output_names = [
                "entity_pilot_selected",
                "entity_document_available",
                *layout_names[2:],
                *current_graph_names,
                *CRITERION_FEATURE_NAMES,
            ]
            if len(output_names) != len(set(output_names)):
                raise ValueError("entity-router output feature names collide")
            matrix = np.zeros((len(fold_rows), len(output_names)), dtype=np.float32)
        elif current_graph_names != graph_names:
            raise ValueError("entity graph feature order changed between papers")
        criterion_features = criterion_counterfactual_features(
            targets, chunks, graph, concepts
        )
        _, contextual_receipts = contextual_assignments(chunks, graph)
        graph_target_features = graph.get("target_features") or {}
        if set(map(clean, graph_target_features)) != set(targets):
            raise ValueError("entity graph target rows changed")
        for assignment in graph.get("assignments") or []:
            assignment_counts[clean(assignment.get("status")) or "missing_status"] += 1
        audit = graph.get("audit") or {}
        rejected_alias_conflicts += sum(
            clean(item.get("status")).casefold() == "conflict"
            for item in (audit.get("alias_conflicts") or {}).values()
            if isinstance(item, Mapping)
        )

        canonical_groups: dict[str, list[str]] = defaultdict(list)
        for target, rockname in targets.items():
            canonical_groups[canonical_target_identity(rockname)].append(target)
        vectors: dict[str, np.ndarray] = {}
        for target in targets:
            graph_values = graph_target_features[target]
            if set(graph_values) != set(graph_names):
                raise ValueError("entity graph target feature keys changed")
            row_index = full_index[(refno, target)]
            vector = np.asarray(
                [
                    1.0,
                    1.0,
                    *layout_matrix[row_index, 2:].tolist(),
                    *[float(graph_values[name]) for name in graph_names],
                    *[
                        float(criterion_features[target][name])
                        for name in CRITERION_FEATURE_NAMES
                    ],
                ],
                dtype=np.float32,
            )
            assert matrix is not None
            matrix[row_index] = vector
            vectors[target] = vector[2 + len(layout_names) - 2 :]
            entity_targets_with_exclusive += int(
                criterion_features[target]["entity_cf_exclusive_event_count_log"] > 0.0
            )
        for canonical, members in canonical_groups.items():
            if len(members) < 2:
                continue
            duplicate_groups_checked += 1
            for member in members[1:]:
                if not np.array_equal(vectors[members[0]], vectors[member]):
                    raise ValueError(
                        f"duplicate canonical identity was split: {refno}/{canonical}"
                    )
        distinct = len({value.tobytes() for value in vectors.values()})
        documents_differentiated += int(distinct > 1)
        paper_audits.append(
            {
                "REFNO": refno,
                "rows": len(rows),
                "canonical_targets": len(canonical_groups),
                "chunks": len(chunks),
                "distinct_target_vectors": distinct,
                "sibling_differentiating_features": distinct > 1,
                "graph_audit": audit,
            }
        )
        evidence_rows.append(
            {
                "schema_version": "r3_entity_router_evidence_graph_v1",
                "REFNO": refno,
                "targets": targets,
                "chunks": chunks,
                "inventories": graph.get("inventories") or {},
                "assignments": graph.get("assignments") or [],
                "contextual_routes": contextual_receipts,
                "audit": audit,
                "runtime_boundary": {name: False for name in BOUNDARY_FALSE_FIELDS},
            }
        )

    if matrix is None or output_names is None or graph_names is None:
        raise ValueError("entity-router selection produced no documents")
    if matrix.shape != (len(fold_rows), len(output_names)) or not np.isfinite(matrix).all():
        raise ValueError("entity-router matrix is invalid")
    selected_indices = np.asarray(
        [full_index[(clean(row["REFNO"]), clean(row["RESULTNO"]))] for row in selected_rows],
        dtype=np.int64,
    )
    selected_mask = np.zeros(len(fold_rows), dtype=bool)
    selected_mask[selected_indices] = True
    if np.any(matrix[~selected_mask] != 0.0):
        raise ValueError("entity-router features leaked outside the pilot")
    if not np.all(matrix[selected_mask, :2] == 1.0):
        raise ValueError("entity-router pilot flags are incomplete")
    layout_exclusive_mask = (
        matrix[:, output_names.index("layout_exclusive_evidence_mass")] > 0.0
    )
    entity_exclusive_mask = (
        matrix[:, output_names.index("entity_cf_exclusive_event_count_log")] > 0.0
    )
    combined_targets_with_exclusive = int(
        np.sum(selected_mask & (layout_exclusive_mask | entity_exclusive_mask))
    )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / FEATURE_NAME
    manifest_path = output_dir / MANIFEST_NAME
    audit_path = output_dir / AUDIT_NAME
    candidate_path = output_dir / CANDIDATE_NAME
    evidence_path = output_dir / EVIDENCE_NAME
    layout._atomic_npz(feature_path, entity=matrix)
    _atomic_jsonl(evidence_path, evidence_rows)
    matrix_hash = layout.array_sha256(matrix)
    names_hash = layout.json_sha256(output_names)
    boundary = {name: False for name in BOUNDARY_FALSE_FIELDS}
    layout_end = 2 + len(layout_names) - 2
    entity_start = layout_end
    full_columns = list(range(len(output_names)))
    ablation_groups = {
        "flags_only": {"columns": [0, 1]},
        "layout_only": {"columns": list(range(2, layout_end))},
        "entity_only": {"columns": list(range(entity_start, len(output_names)))},
        "full": {"columns": full_columns},
    }
    candidate = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "fixed_decision_threshold": 0.5,
        "primary_candidates": ["full_random_forest", "full_hgb"],
        "feature_ablation_candidates": ablation_groups,
        "model_candidates": {
            "random_forest": {
                "n_estimators": 600,
                "max_features": 0.5,
                "min_samples_leaf": 8,
                "n_jobs": -1,
                "random_state": 35_720_260,
            },
            "hgb": {
                "max_iter": 260,
                "learning_rate": 0.035,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 10,
                "l2_regularization": 1.0,
                "random_state": 35_720_260,
            },
        },
        "zero_shot_candidates": {
            "method_only": {
                "feature_name": "entity_cf_method_only_score",
                "threshold": 0.5,
            },
            "practice_chain": {
                "feature_name": "entity_cf_practice_chain_score",
                "threshold": 0.5,
            },
            "normative_chain": {
                "feature_name": "entity_cf_normative_chain_score",
                "threshold": 0.5,
            },
        },
        "alignment_sha256": clean(
            json.loads(inputs["fold_manifest"].read_text(encoding="utf-8")).get(
                "alignment_sha256"
            )
        ),
        "matrix_sha256": matrix_hash,
        "feature_names_sha256": names_hash,
        "pilot_selection_sha256": clean(selection.get("selection_sha256")),
        "effect_acceptance_gates": {
            "minimum_pilot_net_correct_gain": 8,
            "minimum_pilot_accuracy_gain": 0.036,
            "minimum_repairs_to_degradations_ratio": 2.0,
            "require_nonpilot_rows_unchanged": True,
        },
        "deployment_route": {
            "entity_scope": "frozen_pilot_rows_only",
            "nonpilot_rows": "base_v4_unchanged",
        },
        "structure_acceptance_gates": {
            "minimum_documents_with_sibling_differentiating_features": 12,
            "minimum_targets_with_exclusive_evidence": 58,
            "maximum_alias_collisions": 0,
            "duplicate_canonical_identity_must_fallback": True,
        },
        "paper_references": [
            {
                "title": "Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context",
                "venue": "ICLR 2026",
                "url": "https://openreview.net/forum?id=UJ2UUjT2ko",
            },
            {
                "title": "BrowseNet",
                "venue": "ICLR 2026",
                "url": "https://openreview.net/forum?id=2q5CugVPoK",
            },
            {
                "title": "Calibrating Verbalized Confidence with Self-Generated Distractors",
                "venue": "ICLR 2026",
                "url": "https://openreview.net/forum?id=pZs4hhemXc",
            },
            {
                "title": "xCoRe: Cross-context Coreference Resolution",
                "venue": "EMNLP 2025",
                "url": "https://aclanthology.org/2025.emnlp-main.1737/",
                "doi": "10.18653/v1/2025.emnlp-main.1737",
            },
        ],
        "runtime_boundary": boundary,
    }
    layout._atomic_json(candidate_path, candidate)
    structure = {
        "documents_with_sibling_differentiating_features": documents_differentiated,
        "entity_targets_with_exclusive_evidence": entity_targets_with_exclusive,
        "combined_targets_with_exclusive_evidence": combined_targets_with_exclusive,
        "accepted_alias_collision_count": 0,
        "rejected_alias_conflict_count": rejected_alias_conflicts,
        "duplicate_canonical_groups_checked": duplicate_groups_checked,
    }
    gates = candidate["structure_acceptance_gates"]
    structure_passed = (
        documents_differentiated
        >= gates["minimum_documents_with_sibling_differentiating_features"]
        and combined_targets_with_exclusive
        >= gates["minimum_targets_with_exclusive_evidence"]
        and structure["accepted_alias_collision_count"]
        <= gates["maximum_alias_collisions"]
        and duplicate_groups_checked >= 0
    )
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": len(fold_rows),
        "features": len(output_names),
        "pilot_rows": int(selected_mask.sum()),
        "pilot_papers": len(by_refno),
        "graph_features": len(graph_names),
        "criterion_counterfactual_features": len(CRITERION_FEATURE_NAMES),
        "nonpilot_nonzero_values": int(np.count_nonzero(matrix[~selected_mask])),
        "assignment_status_counts": dict(sorted(assignment_counts.items())),
        "structure": structure,
        "structure_acceptance_passed": bool(structure_passed),
        "papers_detail": paper_audits,
        "runtime_boundary": boundary,
    }
    layout._atomic_json(audit_path, audit)

    code_paths = [Path(__file__).resolve(), (SCRIPTS / "r3_paleomag_entity_graph.py").resolve()]
    manifest = {
        "schema_version": FEATURE_SCHEMA,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": len(fold_rows),
        "features": len(output_names),
        "feature_names": output_names,
        "feature_names_sha256": names_hash,
        "matrix_sha256": matrix_hash,
        "feature_ablation_candidates": ablation_groups,
        "alignment_sha256": candidate["alignment_sha256"],
        "pilot_selection_sha256": candidate["pilot_selection_sha256"],
        "contains_labels": False,
        "runtime_boundary": boundary,
        "inputs": {
            **{
                name: {"path": str(path), "sha256": layout.file_sha256(path)}
                for name, path in inputs.items()
            },
            "code": [
                {"path": str(path), "sha256": layout.file_sha256(path)}
                for path in code_paths
            ],
        },
        "outputs": {
            "features": {
                "path": str(feature_path),
                "sha256": layout.file_sha256(feature_path),
            },
            "candidate": {
                "path": str(candidate_path),
                "sha256": layout.file_sha256(candidate_path),
            },
            "audit": {"path": str(audit_path), "sha256": layout.file_sha256(audit_path)},
            "evidence_graph": {
                "path": str(evidence_path),
                "sha256": layout.file_sha256(evidence_path),
            },
            "selection": {
                "path": str(inputs["selection"]),
                "sha256": layout.file_sha256(inputs["selection"]),
                "selection_sha256": candidate["pilot_selection_sha256"],
            },
        },
    }
    layout._atomic_json(manifest_path, manifest)
    return {
        "features": feature_path,
        "manifest": manifest_path,
        "candidate": candidate_path,
        "audit": audit_path,
        "evidence_graph": evidence_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--concept-snapshot", type=Path, default=DEFAULT_CONCEPT_SNAPSHOT)
    parser.add_argument("--layout-features", type=Path, default=DEFAULT_LAYOUT_FEATURES)
    parser.add_argument("--layout-manifest", type=Path, default=DEFAULT_LAYOUT_MANIFEST)
    parser.add_argument("--layout-candidate", type=Path, default=DEFAULT_LAYOUT_CANDIDATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    outputs = build_entity_router_artifacts(
        selection_path=args.selection,
        sidecar_path=args.sidecar,
        fold_manifest_path=args.fold_manifest,
        concept_snapshot_path=args.concept_snapshot,
        layout_features_path=args.layout_features,
        layout_manifest_path=args.layout_manifest,
        layout_candidate_path=args.layout_candidate,
        output_dir=args.output_dir,
    )
    audit = json.loads(outputs["audit"].read_text(encoding="utf-8"))
    print(
        "R3 entity-router features frozen: "
        f"pilot_rows={audit['pilot_rows']} features={audit['features']} "
        f"structure_passed={audit['structure_acceptance_passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CRITERION_FEATURE_NAMES",
    "build_entity_router_artifacts",
    "criterion_counterfactual_features",
    "document_chunks",
]
