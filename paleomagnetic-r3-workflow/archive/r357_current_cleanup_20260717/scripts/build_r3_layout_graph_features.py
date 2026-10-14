#!/usr/bin/env python3
"""Freeze pilot layout-graph features in the complete V4 row alignment."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any, Mapping, Optional, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from paper_text_retrieval import canonical_target_identity  # noqa: E402
from r3_layout_graph_features import (  # noqa: E402
    FEATURE_NAMES as PAGE_FEATURE_NAMES,
    build_layout_graph_target_features,
)


DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_layout_pilot_v1"
DEFAULT_SELECTION = DEFAULT_OUTPUT_DIR / "r3_layout_pilot_manifest_v1.json"
DEFAULT_SIDECAR = DEFAULT_OUTPUT_DIR / "layout_sidecar_v1.jsonl"
DEFAULT_FOLD_MANIFEST = (
    ROOT
    / "tmp/r3_all_labelled_pdf_grouped_oof_v1/trace_refno_fold_manifest_v4.json"
)
DEFAULT_CONCEPT_SNAPSHOT = (
    ROOT
    / "tmp/r3_all_labelled_pdf_grouped_oof_v1/trace_page_graph_concepts_v4.json"
)

FEATURE_NAME = "r3_layout_features_v1.npz"
MANIFEST_NAME = "r3_layout_features_v1.manifest.json"
AUDIT_NAME = "r3_layout_features_v1.audit.json"
CANDIDATE_NAME = "r3_layout_candidate_freeze_v1.json"
FEATURE_SCHEMA_VERSION = "r3_layout_features_v1"
AUDIT_SCHEMA_VERSION = "r3_layout_feature_audit_v1"
CANDIDATE_SCHEMA_VERSION = "r3_layout_candidate_freeze_v1"
SELECTION_SCHEMA_VERSION = "r3_layout_pilot_manifest_v1"
SIDECAR_SCHEMA_VERSION = "r3_layout_sidecar_v1"
FOLD_SCHEMA_VERSION = "r3_trace_refno_fold_manifest_v4"
CONCEPT_SCHEMA_VERSION = "r3_trace_page_graph_concepts_v4"
IDENTITY_FIELDS = ("REFNO", "RESULTNO", "ROCKNAME")
OUTPUT_FEATURE_NAMES = (
    "layout_pilot_selected",
    "layout_document_available",
    *PAGE_FEATURE_NAMES,
)
RUNTIME_BOUNDARY_FIELDS = (
    "labels_read",
    "pmagresult_or_rockmag_read",
    "prior_predictions_read",
    "evaluation_results_read",
)
FORBIDDEN_FIELD_RE = re.compile(
    r"^(?:r[357]|pmagresult|rockmag|gold|ground_truth|supervised_label|predictions?|"
    r"oof|evaluation|metrics?|outcomes?)$",
    re.I,
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RANDOM_STATE = 35_720_260


class LayoutGraphFeatureBuilderError(ValueError):
    """Raised when frozen V5 inputs do not satisfy the source boundary."""


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _require_sha256(value: Any, description: str) -> str:
    digest = clean(value).casefold()
    if not SHA256_RE.fullmatch(digest):
        raise LayoutGraphFeatureBuilderError(f"{description} is not a SHA-256 digest")
    return digest


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayoutGraphFeatureBuilderError(
            f"invalid {description} JSON {path}: {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise LayoutGraphFeatureBuilderError(f"{description} must be a JSON object")
    return dict(payload)


def _reject_forbidden_fields(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = clean(key)
            child_path = f"{path}.{name}"
            structural_reference_label = name.casefold() == "label" and (
                ".captions[" in path or ".crossrefs[" in path
            )
            if FORBIDDEN_FIELD_RE.fullmatch(name) and not structural_reference_label:
                raise LayoutGraphFeatureBuilderError(
                    f"forbidden supervised field at {child_path}"
                )
            _reject_forbidden_fields(child, child_path)
    elif isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _validate_runtime_boundary(payload: Mapping[str, Any], description: str) -> None:
    boundary = payload.get("runtime_boundary") or {}
    if not isinstance(boundary, Mapping):
        raise LayoutGraphFeatureBuilderError(f"{description} runtime boundary is invalid")
    for name in RUNTIME_BOUNDARY_FIELDS:
        if boundary.get(name) is not False:
            raise LayoutGraphFeatureBuilderError(
                f"{description} must declare {name}=false"
            )


def _validate_selection(path: Path) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, dict[str, Any]]]:
    payload = _load_json(path, "pilot selection manifest")
    if (
        clean(payload.get("schema_version")) != SELECTION_SCHEMA_VERSION
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise LayoutGraphFeatureBuilderError("pilot selection manifest schema is invalid")
    _validate_runtime_boundary(payload, "pilot selection manifest")
    if tuple(payload.get("identity_fields") or ()) != IDENTITY_FIELDS:
        raise LayoutGraphFeatureBuilderError("pilot identity fields changed")
    rule = payload.get("selection_rule")
    rows = payload.get("rows")
    papers = payload.get("papers")
    if not isinstance(rule, Mapping) or rule.get("label_free") is not True:
        raise LayoutGraphFeatureBuilderError("pilot selection rule is not label-free")
    if not isinstance(rows, list) or not rows:
        raise LayoutGraphFeatureBuilderError("pilot selection rows are empty")
    normalized_rows: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or tuple(row.keys()) != IDENTITY_FIELDS:
            raise LayoutGraphFeatureBuilderError(
                f"pilot row {index} must contain exactly REFNO, RESULTNO, ROCKNAME"
            )
        normalized = {field: clean(row[field]) for field in IDENTITY_FIELDS}
        if not all(normalized.values()):
            raise LayoutGraphFeatureBuilderError(f"pilot row {index} is incomplete")
        normalized_rows.append(normalized)
    resultnos = [row["RESULTNO"] for row in normalized_rows]
    if len(resultnos) != len(set(resultnos)):
        raise LayoutGraphFeatureBuilderError("pilot selection contains duplicate RESULTNO")
    if json_sha256(rows) != _require_sha256(
        payload.get("selected_rows_sha256"), "pilot selected_rows_sha256"
    ):
        raise LayoutGraphFeatureBuilderError("pilot selected-row hash mismatch")
    if json_sha256(rule) != _require_sha256(
        payload.get("selection_rule_sha256"), "pilot selection_rule_sha256"
    ):
        raise LayoutGraphFeatureBuilderError("pilot selection-rule hash mismatch")
    if not isinstance(papers, list) or not papers:
        raise LayoutGraphFeatureBuilderError("pilot selection papers are empty")
    paper_map: dict[str, dict[str, Any]] = {}
    frozen_pdfs: list[dict[str, str]] = []
    for index, paper in enumerate(papers):
        if not isinstance(paper, Mapping) or not isinstance(paper.get("pdf"), Mapping):
            raise LayoutGraphFeatureBuilderError(f"pilot paper {index} is invalid")
        refno = clean(paper.get("REFNO"))
        if not refno or refno in paper_map:
            raise LayoutGraphFeatureBuilderError("pilot papers contain duplicate REFNO")
        pdf = dict(paper["pdf"])
        pdf_path = Path(clean(pdf.get("path"))).expanduser().resolve()
        declared = _require_sha256(pdf.get("sha256"), f"pilot PDF {refno} hash")
        if not pdf_path.is_file() or file_sha256(pdf_path) != declared:
            raise LayoutGraphFeatureBuilderError(
                f"pilot PDF is missing or changed for REFNO {refno}"
            )
        expected_rows = sum(row["REFNO"] == refno for row in normalized_rows)
        if int(paper.get("identity_row_count") or -1) != expected_rows:
            raise LayoutGraphFeatureBuilderError(
                f"pilot identity_row_count changed for REFNO {refno}"
            )
        paper_map[refno] = {"REFNO": refno, "pdf": pdf, "path": pdf_path}
        frozen_pdfs.append(
            {"REFNO": refno, "name": clean(pdf.get("name")), "sha256": declared}
        )
    row_refnos = {row["REFNO"] for row in normalized_rows}
    if row_refnos != set(paper_map):
        raise LayoutGraphFeatureBuilderError("pilot paper and row REFNO sets differ")
    if json_sha256(frozen_pdfs) != _require_sha256(
        payload.get("selected_pdf_set_sha256"), "pilot selected_pdf_set_sha256"
    ):
        raise LayoutGraphFeatureBuilderError("pilot selected-PDF hash mismatch")
    selection_payload = {"selection_rule": rule, "rows": rows, "pdfs": frozen_pdfs}
    if json_sha256(selection_payload) != _require_sha256(
        payload.get("selection_sha256"), "pilot selection_sha256"
    ):
        raise LayoutGraphFeatureBuilderError("pilot combined selection hash mismatch")
    return payload, normalized_rows, paper_map


def _validate_fold_manifest(
    path: Path,
    selection_rows: Sequence[Mapping[str, str]],
    selection_papers: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path, "V4 fold manifest")
    if (
        clean(payload.get("schema_version")) != FOLD_SCHEMA_VERSION
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or payload.get("labels_read") is not False
    ):
        raise LayoutGraphFeatureBuilderError("V4 fold manifest schema is invalid")
    _require_sha256(payload.get("alignment_sha256"), "V4 alignment_sha256")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise LayoutGraphFeatureBuilderError("V4 fold manifest rows are empty")
    resultnos: set[str] = set()
    refno_folds: dict[str, set[int]] = defaultdict(set)
    document_folds: dict[str, set[int]] = defaultdict(set)
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        if not isinstance(raw, Mapping) or raw.get("row_index") != index:
            raise LayoutGraphFeatureBuilderError("V4 fold row order is not contiguous")
        refno = clean(raw.get("REFNO"))
        resultno = clean(raw.get("RESULTNO"))
        fold = raw.get("fold")
        if not refno or not resultno or isinstance(fold, bool) or not isinstance(fold, int):
            raise LayoutGraphFeatureBuilderError(f"V4 fold row {index} is invalid")
        if resultno in resultnos:
            raise LayoutGraphFeatureBuilderError("V4 fold manifest has duplicate RESULTNO")
        resultnos.add(resultno)
        if fold < 1:
            raise LayoutGraphFeatureBuilderError("V4 fold ids must be positive")
        refno_folds[refno].add(fold)
        document = _require_sha256(raw.get("document_sha256"), "document_sha256")
        document_folds[document].add(fold)
        normalized.append(dict(raw))
    if any(len(values) != 1 for values in refno_folds.values()):
        raise LayoutGraphFeatureBuilderError("V4 fold manifest splits a REFNO")
    if any(len(values) != 1 for values in document_folds.values()):
        raise LayoutGraphFeatureBuilderError("V4 fold manifest splits a document hash")
    fold_ids = np.asarray([row["fold"] for row in normalized], dtype=np.int8)
    if hashlib.sha256(fold_ids.tobytes()).hexdigest() != _require_sha256(
        payload.get("fold_id_sha256"), "V4 fold_id_sha256"
    ):
        raise LayoutGraphFeatureBuilderError("V4 fold-id hash mismatch")
    expected_counts = {
        str(fold): int(np.sum(fold_ids == fold)) for fold in sorted(set(fold_ids))
    }
    if payload.get("fold_counts") != expected_counts:
        raise LayoutGraphFeatureBuilderError("V4 fold counts changed")
    by_resultno = {clean(row["RESULTNO"]): row for row in normalized}
    for row in selection_rows:
        fold_row = by_resultno.get(row["RESULTNO"])
        if fold_row is None or clean(fold_row.get("REFNO")) != row["REFNO"]:
            raise LayoutGraphFeatureBuilderError(
                f"pilot row is absent from V4 alignment: {row['RESULTNO']}"
            )
        selected_pdf = _require_sha256(
            selection_papers[row["REFNO"]]["pdf"].get("sha256"),
            f"pilot PDF {row['REFNO']} hash",
        )
        if _require_sha256(fold_row.get("pdf_sha256"), "V4 row PDF hash") != selected_pdf:
            raise LayoutGraphFeatureBuilderError(
                f"V4 and pilot PDF hashes differ for REFNO {row['REFNO']}"
            )
    return payload, normalized


def _validate_concept_snapshot(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _load_json(path, "V4 concept snapshot")
    if (
        clean(payload.get("schema_version")) != CONCEPT_SCHEMA_VERSION
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise LayoutGraphFeatureBuilderError("V4 concept snapshot schema is invalid")
    _reject_forbidden_fields(payload.get("concepts") or [], "concepts")
    for name in ("criterion", "normative_concepts"):
        source = payload.get(name)
        if not isinstance(source, Mapping):
            raise LayoutGraphFeatureBuilderError(f"concept snapshot lacks {name}")
        source_path = Path(clean(source.get("path"))).expanduser().resolve()
        source_hash = _require_sha256(source.get("sha256"), f"{name} source hash")
        if not source_path.is_file() or file_sha256(source_path) != source_hash:
            raise LayoutGraphFeatureBuilderError(
                f"concept snapshot source is missing or changed: {source_path}"
            )
    concepts = payload.get("concepts")
    if not isinstance(concepts, list) or not concepts:
        raise LayoutGraphFeatureBuilderError("V4 concept snapshot has no concepts")
    categories: set[str] = set()
    identifiers: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for concept in concepts:
        if not isinstance(concept, Mapping):
            raise LayoutGraphFeatureBuilderError("V4 concept must be an object")
        concept_id = clean(concept.get("concept_id"))
        category = clean(concept.get("category")).casefold()
        patterns = concept.get("search_patterns")
        if (
            not concept_id
            or concept_id in identifiers
            or category not in {"method", "diagnostic", "carrier", "bridge"}
            or not isinstance(patterns, list)
            or not all(clean(pattern) for pattern in patterns)
        ):
            raise LayoutGraphFeatureBuilderError(f"invalid V4 concept: {concept_id}")
        for pattern in patterns:
            try:
                re.compile(clean(pattern), re.I)
            except re.error as exc:
                raise LayoutGraphFeatureBuilderError(
                    f"invalid concept pattern for {concept_id}: {exc}"
                ) from exc
        categories.add(category)
        identifiers.add(concept_id)
        normalized.append(dict(concept))
    if categories != {"method", "diagnostic", "carrier", "bridge"}:
        raise LayoutGraphFeatureBuilderError("V4 concepts do not cover all R3 categories")
    return payload, normalized


def _validate_sidecar(
    path: Path,
    selection_papers: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LayoutGraphFeatureBuilderError(
                    f"invalid sidecar JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise LayoutGraphFeatureBuilderError(
                    f"sidecar line {line_number} must be an object"
                )
            _reject_forbidden_fields(raw, f"sidecar[{line_number}]")
            if clean(raw.get("schema_version")) != SIDECAR_SCHEMA_VERSION:
                raise LayoutGraphFeatureBuilderError("sidecar schema changed")
            refno = clean(raw.get("REFNO"))
            if refno not in selection_papers or refno in records:
                raise LayoutGraphFeatureBuilderError(
                    f"sidecar has an unexpected or duplicate REFNO: {refno}"
                )
            pdf = raw.get("pdf")
            pages = raw.get("pages")
            if not isinstance(pdf, Mapping) or not isinstance(pages, list):
                raise LayoutGraphFeatureBuilderError(f"sidecar REFNO {refno} is incomplete")
            declared = _require_sha256(pdf.get("sha256"), f"sidecar PDF {refno} hash")
            expected = _require_sha256(
                selection_papers[refno]["pdf"].get("sha256"),
                f"pilot PDF {refno} hash",
            )
            if declared != expected:
                raise LayoutGraphFeatureBuilderError(
                    f"sidecar and pilot PDF hashes differ for REFNO {refno}"
                )
            if int(pdf.get("page_count") or -1) != len(pages):
                raise LayoutGraphFeatureBuilderError(
                    f"sidecar page_count mismatch for REFNO {refno}"
                )
            page_numbers = [page.get("page") for page in pages if isinstance(page, Mapping)]
            if len(page_numbers) != len(pages) or len(page_numbers) != len(set(page_numbers)):
                raise LayoutGraphFeatureBuilderError(
                    f"sidecar page numbers are invalid for REFNO {refno}"
                )
            records[refno] = dict(raw)
    if set(records) != set(selection_papers):
        missing = sorted(set(selection_papers) - set(records))
        raise LayoutGraphFeatureBuilderError(
            "sidecar does not exactly cover pilot papers: " + ", ".join(missing[:20])
        )
    return records


def _signed_log1p(value: float) -> float:
    return math.copysign(math.log1p(abs(float(value))), float(value))


def _aggregation_kind(name: str) -> str:
    binary = {
        "layout_resolved_identity",
        "layout_unresolved_duplicate_identity",
        "layout_complete_chain",
        "layout_exclusive_complete_chain",
        "layout_exclusive_complete_chain_with_diagnostic",
        "layout_has_shared_evidence",
        "layout_has_ambiguous_evidence",
    }
    if name in binary or name.endswith("_available"):
        return "binary_max"
    if name == "layout_duplicate_identity_size_log":
        return "max"
    if name.endswith("_count_log") or name.endswith("_scope_count_log"):
        return "log_count_sum"
    if name == "layout_evidence_mass_minus_sibling_max":
        return "signed_log_sum"
    if name.endswith("_mass"):
        return "log_mass_sum"
    if name in {
        "layout_exclusive_chain_category_fraction",
        "layout_unique_sibling_margin_max",
    }:
        return "max"
    if name in {"layout_unique_sibling_margin_min", "layout_confidence_min"}:
        return "supported_min"
    return "evidence_weighted_mean"


AGGREGATION_POLICY = {
    "binary_and_availability": "maximum across physical pages",
    "count_log": "sum expm1(page value), then log1p",
    "mass": "sum page mass, then log1p; signed contrast uses signed log1p",
    "ratios_entropy_and_confidence": "mean weighted by raw page evidence mass; maximum if no evidence mass",
    "minimum": "minimum over pages with supporting evidence or confidence",
    "maximum": "maximum across physical pages",
}


def aggregate_target_page_features(
    page_values: Sequence[Mapping[str, float]],
    feature_names: Sequence[str] = PAGE_FEATURE_NAMES,
) -> dict[str, float]:
    if not page_values:
        return {name: 0.0 for name in feature_names}
    for values in page_values:
        if set(values) != set(feature_names):
            raise LayoutGraphFeatureBuilderError("page layout feature names changed")
    raw_weights = [
        max(
            0.0,
            float(values.get("layout_evidence_mass", 0.0))
            + float(values.get("layout_ambiguous_evidence_mass", 0.0))
            + float(values.get("layout_unresolved_evidence_mass", 0.0)),
        )
        for values in page_values
    ]
    output: dict[str, float] = {}
    for name in feature_names:
        values = [float(item[name]) for item in page_values]
        kind = _aggregation_kind(name)
        if kind in {"binary_max", "max"}:
            aggregated = max(values, default=0.0)
        elif kind == "log_count_sum":
            aggregated = math.log1p(sum(max(0.0, math.expm1(value)) for value in values))
        elif kind == "log_mass_sum":
            aggregated = math.log1p(sum(max(0.0, value) for value in values))
        elif kind == "signed_log_sum":
            aggregated = _signed_log1p(sum(values))
        elif kind == "supported_min":
            supported = [
                value
                for value, weight in zip(values, raw_weights)
                if weight > 0.0
            ]
            aggregated = min(supported) if supported else 0.0
        else:
            denominator = sum(raw_weights)
            aggregated = (
                sum(value * weight for value, weight in zip(values, raw_weights))
                / denominator
                if denominator > 0.0
                else max(values, default=0.0)
            )
        if not math.isfinite(aggregated):
            raise LayoutGraphFeatureBuilderError(
                f"non-finite aggregate for layout feature {name}"
            )
        output[name] = float(aggregated)
    return output


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_layout_graph_feature_artifacts(
    *,
    selection_path: Path,
    sidecar_path: Path,
    fold_manifest_path: Path,
    concept_snapshot_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    inputs = {
        "selection": selection_path.expanduser().resolve(),
        "sidecar": sidecar_path.expanduser().resolve(),
        "fold_manifest": fold_manifest_path.expanduser().resolve(),
        "concept_snapshot": concept_snapshot_path.expanduser().resolve(),
    }
    if any(not path.is_file() for path in inputs.values()):
        missing = [str(path) for path in inputs.values() if not path.is_file()]
        raise LayoutGraphFeatureBuilderError(
            "layout graph input does not exist: " + ", ".join(missing)
        )
    pilot, selected_rows, selection_papers = _validate_selection(
        inputs["selection"]
    )
    folds, fold_rows = _validate_fold_manifest(
        inputs["fold_manifest"], selected_rows, selection_papers
    )
    snapshot, concepts = _validate_concept_snapshot(inputs["concept_snapshot"])
    sidecars = _validate_sidecar(inputs["sidecar"], selection_papers)

    full_index = {
        (clean(row["REFNO"]), clean(row["RESULTNO"])): int(row["row_index"])
        for row in fold_rows
    }
    if len(full_index) != len(fold_rows):
        raise LayoutGraphFeatureBuilderError("V4 fold alignment has duplicate row keys")
    selected_by_refno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in selected_rows:
        selected_by_refno[row["REFNO"]].append(dict(row))

    feature_names = list(OUTPUT_FEATURE_NAMES)
    if len(feature_names) != len(set(feature_names)) or any(
        not name.startswith("layout_") for name in feature_names
    ):
        raise LayoutGraphFeatureBuilderError("layout feature names are invalid")
    matrix = np.zeros((len(fold_rows), len(feature_names)), dtype=np.float32)
    assignment_counts: Counter[str] = Counter()
    duplicate_groups_checked = 0
    paper_audits: list[dict[str, Any]] = []

    for refno, rows in selected_by_refno.items():
        profiles = {row["RESULTNO"]: row["ROCKNAME"] for row in rows}
        page_maps: dict[str, list[Mapping[str, float]]] = {
            resultno: [] for resultno in profiles
        }
        page_count = concept_events = layout_atoms = 0
        for page in sidecars[refno]["pages"]:
            result = build_layout_graph_target_features(
                page,
                profiles,
                concepts,
            )
            if set(result["target_keys"]) != set(profiles):
                raise LayoutGraphFeatureBuilderError(
                    f"layout target output changed for REFNO {refno}"
                )
            page_count += 1
            concept_events += int(result["audit"]["concept_events"])
            layout_atoms += int(result["audit"]["layout_atoms"])
            assignment_counts.update(result["audit"]["assignment_status_counts"])
            for resultno in profiles:
                page_maps[resultno].append(result["target_features"][resultno])
        aggregated = {
            resultno: aggregate_target_page_features(values)
            for resultno, values in page_maps.items()
        }
        canonical_groups: dict[str, list[str]] = defaultdict(list)
        for resultno, rockname in profiles.items():
            canonical_groups[canonical_target_identity(rockname)].append(resultno)
        for canonical, resultnos in canonical_groups.items():
            if len(resultnos) < 2:
                continue
            duplicate_groups_checked += 1
            reference = np.asarray(
                [aggregated[resultnos[0]][name] for name in PAGE_FEATURE_NAMES]
            )
            for resultno in resultnos[1:]:
                candidate = np.asarray(
                    [aggregated[resultno][name] for name in PAGE_FEATURE_NAMES]
                )
                if not np.array_equal(reference, candidate):
                    raise LayoutGraphFeatureBuilderError(
                        f"duplicate canonical identity split by RESULTNO: {refno}/{canonical}"
                    )
        distinct_vectors: set[bytes] = set()
        exclusive_targets = 0
        for row in rows:
            resultno = clean(row["RESULTNO"])
            row_index = full_index[(refno, resultno)]
            values = aggregated[resultno]
            vector = np.asarray(
                [1.0, 1.0, *[values[name] for name in PAGE_FEATURE_NAMES]],
                dtype=np.float32,
            )
            matrix[row_index] = vector
            distinct_vectors.add(vector[2:].tobytes())
            exclusive_targets += int(values["layout_exclusive_evidence_mass"] > 0.0)
        paper_audits.append(
            {
                "REFNO": refno,
                "rows": len(rows),
                "canonical_targets": len(canonical_groups),
                "pages": page_count,
                "layout_atoms": layout_atoms,
                "concept_events": concept_events,
                "targets_with_exclusive_evidence": exclusive_targets,
                "distinct_target_feature_vectors": len(distinct_vectors),
                "sibling_differentiating_features": len(distinct_vectors) > 1,
            }
        )

    if matrix.shape != (len(fold_rows), len(feature_names)) or not np.isfinite(matrix).all():
        raise LayoutGraphFeatureBuilderError("aligned layout feature matrix is invalid")
    selected_indices = np.asarray(
        [
            full_index[(clean(row["REFNO"]), clean(row["RESULTNO"]))]
            for row in selected_rows
        ],
        dtype=np.int64,
    )
    selected_mask = np.zeros(len(fold_rows), dtype=bool)
    selected_mask[selected_indices] = True
    if np.any(matrix[~selected_mask] != 0.0):
        raise LayoutGraphFeatureBuilderError("layout features leaked outside pilot rows")
    if not np.all(matrix[selected_mask, 0] == 1.0) or not np.all(
        matrix[selected_mask, 1] == 1.0
    ):
        raise LayoutGraphFeatureBuilderError(
            "pilot selection/availability flags are incomplete"
        )

    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / FEATURE_NAME
    manifest_path = output_dir / MANIFEST_NAME
    audit_path = output_dir / AUDIT_NAME
    candidate_path = output_dir / CANDIDATE_NAME
    _atomic_npz(feature_path, layout=matrix)
    matrix_hash = array_sha256(matrix)
    selection_hash = clean(pilot["selection_sha256"])
    boundary = {
        "labels_read": False,
        "pmagresult_or_rockmag_read": False,
        "prior_predictions_read": False,
        "evaluation_results_read": False,
        "resultno_used_only_for_alignment": True,
        "outer_test_metrics_used_for_selection": False,
    }
    aggregation_policy = {
        "binary_and_availability": "maximum_over_pages",
        "count_logs": "sum_expm1_then_log1p",
        "nonnegative_mass": "sum_then_log1p",
        "signed_mass_contrast": "sum_then_signed_log1p",
        "ratios_entropy_and_confidence": "raw_evidence_mass_weighted_mean_or_max_without_mass",
        "minimum_and_maximum": "minimum_or_maximum_over_pages_with_evidence",
        "duplicate_canonical_identity": "identical feature vector for every RESULTNO member",
    }
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": len(fold_rows),
        "papers": len({clean(row["REFNO"]) for row in fold_rows}),
        "features": len(feature_names),
        "pilot_rows": len(selected_rows),
        "pilot_papers": len(selected_by_refno),
        "nonpilot_nonzero_values": int(np.count_nonzero(matrix[~selected_mask])),
        "duplicate_canonical_groups_checked": duplicate_groups_checked,
        "documents_with_sibling_differentiating_features": sum(
            item["sibling_differentiating_features"] for item in paper_audits
        ),
        "targets_with_exclusive_evidence": sum(
            item["targets_with_exclusive_evidence"] for item in paper_audits
        ),
        "assignment_status_counts": dict(sorted(assignment_counts.items())),
        "aggregation_policy": aggregation_policy,
        "papers_detail": paper_audits,
        "runtime_boundary": boundary,
    }
    _atomic_json(audit_path, audit)

    random_state = RANDOM_STATE
    candidate = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "primary_candidates": ["augmented_random_forest", "augmented_hgb"],
        "feature_ablation_candidates": {
            "base_v4": {
                "layout_columns": [],
                "purpose": "unchanged V4 structured-feature control",
            },
            "pilot_flags_only": {
                "layout_columns": [0, 1],
                "feature_names": [
                    "layout_pilot_selected",
                    "layout_document_available",
                ],
                "purpose": "separate pilot membership and availability from layout evidence",
            },
            "full_layout": {
                "layout_columns": "all",
                "purpose": "pilot flags plus all frozen layout-graph evidence features",
            },
        },
        "model_candidates": {
            "random_forest": {
                "n_estimators": 600,
                "max_features": 0.5,
                "min_samples_leaf": 8,
                "n_jobs": -1,
                "random_state": random_state,
            },
            "hgb": {
                "max_iter": 260,
                "learning_rate": 0.035,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 10,
                "l2_regularization": 1.0,
                "random_state": random_state,
            },
        },
        "fixed_decision_threshold": 0.5,
        "alignment_sha256": clean(folds["alignment_sha256"]),
        "matrix_sha256": matrix_hash,
        "feature_names_sha256": json_sha256(feature_names),
        "pilot_selection_sha256": selection_hash,
        "structure_acceptance_gates": {
            "minimum_documents_with_sibling_differentiating_features": 6,
            "minimum_targets_with_exclusive_evidence": 30,
            "manual_target_evidence_edge_precision": 0.90,
            "manual_target_evidence_edge_recall": 0.75,
            "maximum_sibling_contamination_rate": 0.05,
        },
        "effect_acceptance_gates": {
            "minimum_pilot_net_correct_gain": 8,
            "minimum_pilot_accuracy_gain": 0.036,
            "require_non_decreasing_macro_f1": True,
            "require_nonpilot_rows_unchanged": True,
            "mixed_subset_is_posthoc_only": True,
            "deployment_route": "layout candidate may replace base only on frozen pilot rows",
            "nonpilot_route": "use unchanged base V4 prediction",
        },
        "deployment_route": {
            "layout_scope": "frozen_pilot_rows_only",
            "nonpilot_rows": "base_v4_unchanged",
            "membership_feature_is_not_evidence": True,
        },
        "runtime_boundary": boundary,
    }
    _atomic_json(candidate_path, candidate)

    code_paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/r3_layout_graph_features.py",
        ROOT / "scripts/paper_text_retrieval.py",
        ROOT / "scripts/r3_paleomag_scope_features.py",
    ]
    manifest = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": len(fold_rows),
        "papers": len({clean(row["REFNO"]) for row in fold_rows}),
        "features": len(feature_names),
        "feature_names": feature_names,
        "feature_names_sha256": json_sha256(feature_names),
        "alignment_sha256": clean(folds["alignment_sha256"]),
        "matrix_sha256": matrix_hash,
        "pilot_selection_sha256": selection_hash,
        "contains_labels": False,
        "aggregation_policy": aggregation_policy,
        "runtime_boundary": boundary,
        "allowed_sources": [
            "frozen REFNO/RESULTNO/ROCKNAME pilot identity",
            "original-PDF layout sidecar",
            "V4 REFNO/PDF grouped fold alignment",
            "V4 frozen normative R3 concept snapshot",
        ],
        "forbidden_sources": [
            "PMAGRESULT and ROCKMAG",
            "R3/R5/R7 labels or assessment results",
            "prior predictions, OOF arrays, evaluations, and metrics",
        ],
        "inputs": {
            "selection_manifest": {
                "path": str(inputs["selection"]),
                "sha256": file_sha256(inputs["selection"]),
                "selection_sha256": selection_hash,
                "selected_rows_sha256": clean(pilot["selected_rows_sha256"]),
                "selected_pdf_set_sha256": clean(pilot["selected_pdf_set_sha256"]),
            },
            "sidecar": {
                "path": str(inputs["sidecar"]),
                "sha256": file_sha256(inputs["sidecar"]),
                "schema_version": "r3_layout_sidecar_v1",
            },
            "fold_manifest": {
                "path": str(inputs["fold_manifest"]),
                "sha256": file_sha256(inputs["fold_manifest"]),
                "fold_id_sha256": clean(folds["fold_id_sha256"]),
                "alignment_sha256": clean(folds["alignment_sha256"]),
            },
            "concept_snapshot": {
                "path": str(inputs["concept_snapshot"]),
                "sha256": file_sha256(inputs["concept_snapshot"]),
                "schema_version": clean(snapshot["schema_version"]),
            },
            "code": [
                {"path": str(path), "sha256": file_sha256(path)}
                for path in code_paths
            ],
        },
        "outputs": {
            "features": {"path": str(feature_path), "sha256": file_sha256(feature_path)},
            "audit": {"path": str(audit_path), "sha256": file_sha256(audit_path)},
            "candidate_freeze": {
                "path": str(candidate_path),
                "sha256": file_sha256(candidate_path),
            },
        },
    }
    _atomic_json(manifest_path, manifest)
    return {
        "features": feature_path,
        "manifest": manifest_path,
        "audit": audit_path,
        "candidate_freeze": candidate_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--concept-snapshot", type=Path, default=DEFAULT_CONCEPT_SNAPSHOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    try:
        outputs = build_layout_graph_feature_artifacts(
            selection_path=args.selection,
            sidecar_path=args.sidecar,
            fold_manifest_path=args.fold_manifest,
            concept_snapshot_path=args.concept_snapshot,
            output_dir=args.output_dir,
        )
    except (LayoutGraphFeatureBuilderError, OSError, ValueError) as exc:
        print(f"R3 layout graph builder error: {exc}", file=sys.stderr)
        return 2
    print(
        "R3 layout graph features frozen: "
        f"features={outputs['features']} manifest={outputs['manifest']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUDIT_NAME",
    "CANDIDATE_NAME",
    "FEATURE_NAME",
    "MANIFEST_NAME",
    "LayoutGraphFeatureBuilderError",
    "aggregate_target_page_features",
    "build_layout_graph_feature_artifacts",
    "main",
]
