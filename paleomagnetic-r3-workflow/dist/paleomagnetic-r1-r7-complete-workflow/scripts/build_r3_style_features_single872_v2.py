#!/usr/bin/env python3
"""Freeze V2 label-blind curation/style and expert-knowledge features for 872 papers."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import experiment_r3_pdf_only_strategies as knowledge_source  # noqa: E402
import experiment_r3_small_screen_curation_style as legacy  # noqa: E402


V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
DEFAULT_V4_MANIFEST = V4_DIR / trace_freeze.MANIFEST_NAME
DEFAULT_FOLD_MANIFEST = V4_DIR / trace_freeze.FOLD_MANIFEST_NAME
DEFAULT_PAGE_CACHE = trace_freeze.DEFAULT_PAGE_CACHE
DEFAULT_PDF_DIR = trace_freeze.DEFAULT_PDF_DIR
DEFAULT_TAXONOMY = ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json"
DEFAULT_POSITIVE_KB = ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_style_features_single872_v2"

FEATURE_NAME = "r3_style_features_single872_v2.npz"
MANIFEST_NAME = "r3_style_features_single872_v2.manifest.json"
FEATURE_SCHEMA = "r3_style_features_single872_v2"
MANIFEST_SCHEMA = "r3_style_features_single872_v2_manifest"
EXPECTED_V4_ROWS = 2751
EXPECTED_SINGLE_ROWS = 872
EXPECTED_STYLE_FEATURES = 105
EXPECTED_KNOWLEDGE_FEATURES = 84
EXPECTED_ALL_FEATURES = 189
EXPECTED_TAXONOMY_SHA256 = "149df539a937678c8204aa6d6e60e77bd901e4a77975d26caa5362749aa18ad4"
EXPECTED_POSITIVE_KB_SHA256 = "b0fc28866fe6702938a7cc02fcf7b87b68abc9e3b4e391a53a53de1615b5e9ef"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class StyleFeatureError(ValueError):
    pass


def clean(value: Any) -> str:
    return trace_freeze.clean(value)


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StyleFeatureError(f"{label} must be a JSON object")
    return value


def require_sha256(value: Any, label: str) -> str:
    output = clean(value).casefold()
    if not SHA256_RE.fullmatch(output):
        raise StyleFeatureError(f"{label} is not a SHA-256 digest")
    return output


def pdf_membership_sha256(pdf_dir: Path) -> str:
    names = (
        sorted(
            path.name
            for path in pdf_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        )
        if pdf_dir.is_dir()
        else []
    )
    return trace_freeze.json_sha256(names)


def acquire_output_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".r3_style_features_single872_v2.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise StyleFeatureError(f"another V2 style builder owns {output_dir}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def load_v4_single_alignment(
    v4_manifest_path: Path,
    fold_path: Path,
    *,
    expected_v4_rows: int = EXPECTED_V4_ROWS,
    expected_single_rows: int = EXPECTED_SINGLE_ROWS,
) -> dict[str, Any]:
    v4 = load_json(v4_manifest_path, "V4 manifest")
    if (
        v4.get("schema_version") != trace_freeze.FEATURE_SCHEMA_VERSION
        or v4.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or v4.get("contains_labels") is not False
        or v4.get("feature_archive_contains_identifiers") is not False
        or int(v4.get("rows", -1)) != expected_v4_rows
    ):
        raise StyleFeatureError("V4 manifest is not the frozen label-blind source")
    boundary = v4.get("runtime_boundary") or {}
    if any(
        boundary.get(name) is not False
        for name in (
            "labels_read",
            "pmagresult_or_rockmag_read",
            "prior_predictions_read",
            "evaluation_results_read",
        )
    ):
        raise StyleFeatureError("V4 manifest violates the label-blind boundary")

    fold_output = (v4.get("outputs") or {}).get("fold_manifest") or {}
    if trace_freeze.file_sha256(fold_path) != require_sha256(
        fold_output.get("sha256"), "V4 fold-manifest hash"
    ):
        raise StyleFeatureError("V4 fold manifest hash changed")
    fold = load_json(fold_path, "V4 fold manifest")
    rows = fold.get("rows") or []
    if (
        fold.get("schema_version") != trace_freeze.FOLD_SCHEMA_VERSION
        or fold.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or fold.get("labels_read") is not False
        or not isinstance(rows, list)
        or len(rows) != expected_v4_rows
        or clean(fold.get("alignment_sha256")) != clean(v4.get("alignment_sha256"))
    ):
        raise StyleFeatureError("V4 fold manifest contract changed")

    identity_input = (v4.get("inputs") or {}).get("identity") or {}
    identity_path = Path(clean(identity_input.get("path"))).resolve()
    if not identity_path.is_file() or trace_freeze.file_sha256(identity_path) != require_sha256(
        identity_input.get("sha256"), "V4 identity hash"
    ):
        raise StyleFeatureError("V4 identity source changed")
    identity = trace_freeze.load_three_column_identity(identity_path)
    by_resultno = {record["RESULTNO"]: record for record in identity}
    records: list[dict[str, str]] = []
    fold_ids = np.zeros(len(rows), dtype=np.int8)
    document_groups = np.empty(len(rows), dtype="U64")
    pdf_hashes = np.empty(len(rows), dtype="U64")
    text_hashes = np.empty(len(rows), dtype="U64")
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise StyleFeatureError(f"V4 fold row {index} is not an object")
        resultno = clean(row.get("RESULTNO"))
        record = by_resultno.get(resultno)
        fold_id = int(row.get("fold", 0))
        if (
            int(row.get("row_index", -1)) != index
            or record is None
            or clean(row.get("REFNO")) != record["REFNO"]
            or fold_id not in range(1, 6)
        ):
            raise StyleFeatureError(f"V4 fold identity drift at row {index}")
        records.append(dict(record))
        fold_ids[index] = fold_id
        document_groups[index] = require_sha256(row.get("document_sha256"), "document hash")
        pdf_hashes[index] = require_sha256(row.get("pdf_sha256"), "PDF hash")
        text_hashes[index] = require_sha256(
            row.get("normalized_text_sha256"), "normalized-text hash"
        )
    if trace_freeze._record_alignment_sha256(records) != clean(v4.get("alignment_sha256")):
        raise StyleFeatureError("V4 record alignment hash changed")
    if hashlib.sha256(fold_ids.tobytes()).hexdigest() != require_sha256(
        fold.get("fold_id_sha256"), "V4 fold-id hash"
    ):
        raise StyleFeatureError("V4 fold-id hash changed")

    multiplicity = Counter(record["REFNO"] for record in records)
    positions = np.asarray(
        [index for index, record in enumerate(records) if multiplicity[record["REFNO"]] == 1],
        dtype=np.int32,
    )
    if len(positions) != expected_single_rows:
        raise StyleFeatureError(
            f"single-target scope changed: expected {expected_single_rows}, got {len(positions)}"
        )
    selected_records = [records[int(index)] for index in positions]
    if len({record["REFNO"] for record in selected_records}) != expected_single_rows:
        raise StyleFeatureError("single-target V4 rows are not one paper per row")
    return {
        "v4_manifest": v4,
        "fold_manifest": fold,
        "identity_path": identity_path,
        "records": selected_records,
        "source_row_indices": positions,
        "fold_ids": fold_ids[positions],
        "document_groups": document_groups[positions],
        "pdf_sha256": pdf_hashes[positions],
        "normalized_text_sha256": text_hashes[positions],
    }


def load_style_pages(
    path: Path, wanted_refnos: set[str]
) -> tuple[dict[str, list[dict[str, Any]]], set[str]]:
    pages_by_refno: dict[str, list[dict[str, Any]]] = {}
    present: set[str] = set()
    if not path.is_file():
        return pages_by_refno, present
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                raise StyleFeatureError(f"page-cache line {line_number} is not an object")
            refno = clean(raw.get("REFNO"))
            if refno not in wanted_refnos:
                continue
            forbidden = sorted(
                clean(name)
                for name in raw
                if trace_freeze.FORBIDDEN_INPUT_FIELD_RE.search(clean(name))
            )
            if forbidden:
                raise StyleFeatureError(
                    f"page-cache REFNO {refno} contains forbidden fields: {forbidden}"
                )
            if refno in present:
                raise StyleFeatureError(f"duplicate page-cache REFNO {refno}")
            present.add(refno)
            pages: list[dict[str, Any]] = []
            for page_number, raw_page in enumerate(raw.get("pages") or [], start=1):
                if not isinstance(raw_page, Mapping):
                    continue
                forbidden_page = sorted(
                    clean(name)
                    for name in raw_page
                    if trace_freeze.FORBIDDEN_INPUT_FIELD_RE.search(clean(name))
                )
                if forbidden_page:
                    raise StyleFeatureError(
                        f"page-cache REFNO {refno} page {page_number} contains forbidden "
                        f"fields: {forbidden_page}"
                    )
                text = str(raw_page.get("text") or "")
                if not text.strip():
                    continue
                pages.append(
                    {
                        "page": clean(raw_page.get("page")) or str(page_number),
                        "text": text,
                        "source": clean(raw_page.get("source")) or "unknown",
                        "section_hints": [
                            clean(item)
                            for item in raw_page.get("section_hints") or []
                            if clean(item)
                        ],
                    }
                )
            pages_by_refno[refno] = pages
    return pages_by_refno, present


def build_matrices(
    records: Sequence[dict[str, str]],
    pages_by_refno: Mapping[str, Sequence[Mapping[str, Any]]],
    page_rows_present: set[str],
    concepts: Sequence[Mapping[str, Any]],
    pdf_dir: Path,
) -> dict[str, Any]:
    pdf_paths = (
        {
            path.stem: path
            for path in pdf_dir.iterdir()
            if path.is_file() and path.suffix.casefold() == ".pdf"
        }
        if pdf_dir.is_dir()
        else {}
    )
    page_row_missing = np.asarray(
        [record["REFNO"] not in page_rows_present for record in records], dtype=np.bool_
    )
    page_text_missing = np.asarray(
        [not pages_by_refno.get(record["REFNO"]) for record in records], dtype=np.bool_
    )
    pdf_missing = np.asarray(
        [record["REFNO"] not in pdf_paths for record in records], dtype=np.bool_
    )
    available = ~(page_text_missing | pdf_missing)
    available_indices = np.flatnonzero(available)
    if not len(available_indices):
        raise StyleFeatureError("no V4 single-target row has both page text and a local PDF")
    available_records = [records[int(index)] for index in available_indices]
    features = legacy.build_features(
        available_records,
        {record["REFNO"]: pages_by_refno[record["REFNO"]] for record in available_records},
        concepts,
        pdf_dir,
    )
    style_names = list(features["style_feature_names"])
    legacy_knowledge_names = list(features["knowledge_feature_names"])
    concept_keys = [
        f"{clean(concept['source'])}_{legacy.concept_category(concept)}_"
        f"{clean(concept['concept_id'])}"
        for concept in concepts
    ]
    knowledge_names = [f"article_concept_{key}_log" for key in concept_keys]
    knowledge_names.extend(
        f"article_category_{category}_{kind}_log"
        for kind in ("total", "distinct")
        for category in ("method", "diagnostic", "carrier", "other")
    )
    knowledge_names.extend(f"target_concept_{key}_log" for key in concept_keys)
    all_names = [*style_names, *knowledge_names]
    if (
        features["style"].shape != (len(available_indices), EXPECTED_STYLE_FEATURES)
        or features["knowledge"].shape
        != (len(available_indices), EXPECTED_KNOWLEDGE_FEATURES)
        or features["all"].shape != (len(available_indices), EXPECTED_ALL_FEATURES)
        or len(style_names) != EXPECTED_STYLE_FEATURES
        or len(knowledge_names) != EXPECTED_KNOWLEDGE_FEATURES
        or len(set(knowledge_names)) != EXPECTED_KNOWLEDGE_FEATURES
        or len(legacy_knowledge_names) != EXPECTED_KNOWLEDGE_FEATURES
        or len(all_names) != EXPECTED_ALL_FEATURES
        or len(set(all_names)) != EXPECTED_ALL_FEATURES
    ):
        raise StyleFeatureError("legacy style/knowledge feature contract changed")
    style = np.zeros((len(records), EXPECTED_STYLE_FEATURES), dtype=np.float64)
    knowledge = np.zeros((len(records), EXPECTED_KNOWLEDGE_FEATURES), dtype=np.float64)
    style[available_indices] = np.asarray(features["style"], dtype=np.float64)
    knowledge[available_indices] = np.asarray(features["knowledge"], dtype=np.float64)
    combined = np.hstack((style, knowledge))
    publication_years = np.full(len(records), -1, dtype=np.int16)
    for local_index, global_index in enumerate(available_indices):
        year = features["row_meta"][local_index].get("year")
        if year is not None:
            publication_years[global_index] = int(year)
    page_counts = np.asarray(
        [len(pages_by_refno.get(record["REFNO"]) or ()) for record in records],
        dtype=np.int32,
    )
    pdf_bytes = np.asarray(
        [
            pdf_paths[record["REFNO"]].stat().st_size
            if record["REFNO"] in pdf_paths
            else 0
            for record in records
        ],
        dtype=np.int64,
    )
    if not np.isfinite(combined).all() or combined.shape != (
        len(records), EXPECTED_ALL_FEATURES
    ):
        raise StyleFeatureError("V2 style feature matrices are invalid")
    return {
        "style": style,
        "knowledge": knowledge,
        "all": combined,
        "style_feature_names": style_names,
        "knowledge_feature_names": knowledge_names,
        "legacy_knowledge_feature_names": legacy_knowledge_names,
        "all_feature_names": all_names,
        "available": available,
        "page_cache_row_missing": page_row_missing,
        "page_text_missing": page_text_missing,
        "pdf_missing": pdf_missing,
        "page_counts": page_counts,
        "pdf_bytes": pdf_bytes,
        "publication_years": publication_years,
    }


def build_style_features(
    *,
    v4_manifest_path: Path,
    fold_path: Path,
    page_cache_path: Path,
    pdf_dir: Path,
    taxonomy_path: Path,
    positive_kb_path: Path,
    output_dir: Path,
    expected_v4_rows: int = EXPECTED_V4_ROWS,
    expected_single_rows: int = EXPECTED_SINGLE_ROWS,
) -> dict[str, Path]:
    paths = {
        "v4_manifest": v4_manifest_path.resolve(),
        "fold_manifest": fold_path.resolve(),
        "page_cache": page_cache_path.resolve(),
        "pdf_dir": pdf_dir.resolve(),
        "taxonomy": taxonomy_path.resolve(),
        "positive_kb": positive_kb_path.resolve(),
        "output_dir": output_dir.resolve(),
    }
    for name in ("v4_manifest", "fold_manifest", "taxonomy", "positive_kb"):
        if not paths[name].is_file():
            raise StyleFeatureError(f"missing V2 style input {name}: {paths[name]}")
    if trace_freeze.file_sha256(paths["taxonomy"]) != EXPECTED_TAXONOMY_SHA256:
        raise StyleFeatureError("the fixed V2 taxonomy hash changed")
    if trace_freeze.file_sha256(paths["positive_kb"]) != EXPECTED_POSITIVE_KB_SHA256:
        raise StyleFeatureError("the fixed V2 positive-KB hash changed")
    output_lock = acquire_output_lock(paths["output_dir"])
    try:
        alignment = load_v4_single_alignment(
            paths["v4_manifest"],
            paths["fold_manifest"],
            expected_v4_rows=expected_v4_rows,
            expected_single_rows=expected_single_rows,
        )
        records = alignment["records"]
        wanted_refnos = {record["REFNO"] for record in records}
        pages, page_rows_present = load_style_pages(paths["page_cache"], wanted_refnos)
        concepts, knowledge_audit = knowledge_source.load_knowledge_patterns(
            paths["taxonomy"], paths["positive_kb"]
        )
        if len(concepts) != 38:
            raise StyleFeatureError(f"knowledge concept contract changed: {len(concepts)}")

        pdf_paths = (
            {
                path.stem: path
                for path in paths["pdf_dir"].iterdir()
                if path.is_file()
                and path.suffix.casefold() == ".pdf"
                and path.stem in wanted_refnos
            }
            if paths["pdf_dir"].is_dir()
            else {}
        )
        for index, record in enumerate(records):
            refno = record["REFNO"]
            pdf_path = pdf_paths.get(refno)
            if pdf_path is not None and trace_freeze.file_sha256(pdf_path) != clean(
                alignment["pdf_sha256"][index]
            ):
                raise StyleFeatureError(f"V4 PDF content changed for REFNO {refno}")
            if pages.get(refno):
                _, text_hashes, _ = trace_freeze.normalized_text_document_groups(
                    {refno: {"REFNO": refno, "pages": pages[refno]}},
                    {refno: clean(alignment["pdf_sha256"][index])},
                )
                if text_hashes[refno] != clean(alignment["normalized_text_sha256"][index]):
                    raise StyleFeatureError(
                        f"V4 normalized page text changed for REFNO {refno}"
                    )

        built = build_matrices(
            records, pages, page_rows_present, concepts, paths["pdf_dir"]
        )
        arrays: dict[str, np.ndarray] = {
            "style": built["style"],
            "knowledge": built["knowledge"],
            "all": built["all"],
            "style_feature_names": np.asarray(
                built["style_feature_names"], dtype="U128"
            ),
            "knowledge_feature_names": np.asarray(
                built["knowledge_feature_names"], dtype="U192"
            ),
            "legacy_knowledge_feature_names": np.asarray(
                built["legacy_knowledge_feature_names"], dtype="U192"
            ),
            "all_feature_names": np.asarray(built["all_feature_names"], dtype="U192"),
            "available": built["available"],
            "page_cache_row_missing": built["page_cache_row_missing"],
            "page_text_missing": built["page_text_missing"],
            "pdf_missing": built["pdf_missing"],
            "page_counts": built["page_counts"],
            "pdf_bytes": built["pdf_bytes"],
            "publication_years": built["publication_years"],
            "source_row_indices": alignment["source_row_indices"],
            "fold_ids": alignment["fold_ids"].astype(np.int8, copy=False),
            "refnos": np.asarray([row["REFNO"] for row in records], dtype="U32"),
            "resultnos": np.asarray([row["RESULTNO"] for row in records], dtype="U32"),
            "rocknames": np.asarray([row["ROCKNAME"] for row in records], dtype="U512"),
            "document_groups": alignment["document_groups"],
            "pdf_sha256": alignment["pdf_sha256"],
            "normalized_text_sha256": alignment["normalized_text_sha256"],
        }
        feature_path = paths["output_dir"] / FEATURE_NAME
        manifest_path = paths["output_dir"] / MANIFEST_NAME
        trace_freeze.atomic_npz(feature_path, **arrays)
        v4 = alignment["v4_manifest"]
        v4_inputs = v4.get("inputs") or {}
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "feature_schema_version": FEATURE_SCHEMA,
            "status": "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS",
            "rows": len(records),
            "coverage": 1.0,
            "source_v4_rows": expected_v4_rows,
            "style_feature_count": EXPECTED_STYLE_FEATURES,
            "knowledge_feature_count": EXPECTED_KNOWLEDGE_FEATURES,
            "all_feature_count": EXPECTED_ALL_FEATURES,
            "style_feature_names": built["style_feature_names"],
            "knowledge_feature_names": built["knowledge_feature_names"],
            "all_feature_names": built["all_feature_names"],
            "style_feature_names_sha256": trace_freeze.json_sha256(
                built["style_feature_names"]
            ),
            "knowledge_feature_names_sha256": trace_freeze.json_sha256(
                built["knowledge_feature_names"]
            ),
            "all_feature_names_sha256": trace_freeze.json_sha256(
                built["all_feature_names"]
            ),
            "legacy_knowledge_feature_names_sha256": trace_freeze.json_sha256(
                built["legacy_knowledge_feature_names"]
            ),
            "legacy_duplicate_knowledge_feature_names": (
                EXPECTED_KNOWLEDGE_FEATURES
                - len(set(built["legacy_knowledge_feature_names"]))
            ),
            "available_rows": int(built["available"].sum()),
            "missing": {
                "page_cache_rows": int(built["page_cache_row_missing"].sum()),
                "page_text_rows": int(built["page_text_missing"].sum()),
                "pdf_rows": int(built["pdf_missing"].sum()),
                "publication_year_rows": int((built["publication_years"] < 0).sum()),
            },
            "fold_counts": {
                str(fold): int(np.sum(arrays["fold_ids"] == fold))
                for fold in range(1, 6)
            },
            "single_alignment_sha256": trace_freeze._record_alignment_sha256(records),
            "knowledge_audit": knowledge_audit,
            "array_sha256": {
                name: trace_freeze.array_sha256(value) for name, value in arrays.items()
            },
            "inputs": {
                "v4_manifest": {
                    "path": str(paths["v4_manifest"]),
                    "sha256": trace_freeze.file_sha256(paths["v4_manifest"]),
                },
                "fold_manifest": {
                    "path": str(paths["fold_manifest"]),
                    "sha256": trace_freeze.file_sha256(paths["fold_manifest"]),
                },
                "identity": {
                    "path": str(alignment["identity_path"]),
                    "sha256": trace_freeze.file_sha256(alignment["identity_path"]),
                },
                "page_cache": {
                    "path": str(paths["page_cache"]),
                    "sha256": (
                        trace_freeze.file_sha256(paths["page_cache"])
                        if paths["page_cache"].is_file()
                        else None
                    ),
                    "frozen_v4_sha256": clean(
                        (v4_inputs.get("page_cache") or {}).get("sha256")
                    ),
                },
                "pdf_membership": {
                    "path": str(paths["pdf_dir"]),
                    "sha256": pdf_membership_sha256(paths["pdf_dir"]),
                    "frozen_v4_sha256": clean(
                        (v4_inputs.get("pdf_membership") or {}).get("sha256")
                    ),
                },
                "taxonomy": {
                    "path": str(paths["taxonomy"]),
                    "sha256": trace_freeze.file_sha256(paths["taxonomy"]),
                },
                "positive_kb": {
                    "path": str(paths["positive_kb"]),
                    "sha256": trace_freeze.file_sha256(paths["positive_kb"]),
                },
                "code": [
                    {
                        "path": str(path),
                        "sha256": trace_freeze.file_sha256(path),
                    }
                    for path in (
                        Path(__file__).resolve(),
                        Path(legacy.__file__).resolve(),
                        Path(knowledge_source.__file__).resolve(),
                    )
                ],
            },
            "outputs": {
                "features": {
                    "path": str(feature_path),
                    "sha256": trace_freeze.file_sha256(feature_path),
                }
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
            },
            "runtime_boundary": {
                "labels_read": False,
                "pmagresult_or_rockmag_read": False,
                "prior_predictions_read": False,
                "evaluation_results_read": False,
                "identity_values_used_as_numeric_features": False,
                "knowledge_runtime_uses_only_original_text_patterns": True,
            },
        }
        trace_freeze.atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "rows": len(records),
                    "style": list(arrays["style"].shape),
                    "knowledge": list(arrays["knowledge"].shape),
                    "all": list(arrays["all"].shape),
                    "available": int(arrays["available"].sum()),
                },
                indent=2,
            ),
            flush=True,
        )
        return {"features": feature_path, "manifest": manifest_path}
    finally:
        output_lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-manifest", type=Path, default=DEFAULT_V4_MANIFEST)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=DEFAULT_POSITIVE_KB)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_style_features(
        v4_manifest_path=args.v4_manifest,
        fold_path=args.fold_manifest,
        page_cache_path=args.page_cache,
        pdf_dir=args.pdf_dir,
        taxonomy_path=args.taxonomy,
        positive_kb_path=args.positive_kb,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "StyleFeatureError",
    "build_matrices",
    "build_parser",
    "build_style_features",
    "load_style_pages",
    "load_v4_single_alignment",
]
