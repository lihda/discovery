#!/usr/bin/env python3
"""Build a label-free numeric feature cache for exact300 auxiliary training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/auxiliary_numeric_cache_v1"
DEFAULT_CACHE = OUTPUT_DIR / "features.npz"
DEFAULT_MANIFEST = OUTPUT_DIR / "features.manifest.json"
EXPECTED_ARRAYS = {
    "base_engineered": (auxiliary.EXPECTED_FULL_PDF_ROWS, 162),
    "curation_style": (auxiliary.EXPECTED_FULL_PDF_ROWS, 105),
    "curation_knowledge": (auxiliary.EXPECTED_FULL_PDF_ROWS, 84),
    "curation_all": (auxiliary.EXPECTED_FULL_PDF_ROWS, 189),
    "combined": (auxiliary.EXPECTED_FULL_PDF_ROWS, 351),
}
FORBIDDEN_NAME_RE = re.compile(
    r"(?i)^(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)$"
)


def source_inputs(
    exact_identity: Path,
    full_identity: Path,
    protocol: Path,
    fold_manifest: Path,
    page_cache: Path,
    taxonomy: Path,
    positive_kb: Path,
) -> dict[str, tuple[Path, str]]:
    return {
        "exact_identity": (exact_identity, auxiliary.EXACT_IDENTITY_SHA256),
        "full_identity": (full_identity, auxiliary.FULL_IDENTITY_SHA256),
        "screening_protocol": (protocol, auxiliary.SCREENING_PROTOCOL_SHA256),
        "fold_manifest": (fold_manifest, auxiliary.FOLD_MANIFEST_SHA256),
        "page_cache": (page_cache, auxiliary.PAGE_CACHE_SHA256),
        "taxonomy": (taxonomy, auxiliary.TAXONOMY_SHA256),
        "positive_kb": (positive_kb, auxiliary.POSITIVE_KB_SHA256),
    }


def validate_sources(inputs: Mapping[str, tuple[Path, str]]) -> dict[str, Any]:
    audit: dict[str, Any] = {}
    for name, (path, expected_hash) in inputs.items():
        if not path.is_file():
            raise ValueError(f"missing frozen input {name}: {path}")
        observed = base.file_sha256(path)
        if observed != expected_hash:
            raise ValueError(f"frozen input hash mismatch for {name}")
        audit[name] = {"path": str(path), "sha256": observed}
    return audit


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    if set(arrays) != set(EXPECTED_ARRAYS):
        raise ValueError(
            "numeric cache arrays differ from the frozen schema: "
            f"missing={sorted(set(EXPECTED_ARRAYS) - set(arrays))}, "
            f"extra={sorted(set(arrays) - set(EXPECTED_ARRAYS))}"
        )
    if any(FORBIDDEN_NAME_RE.fullmatch(name) for name in arrays):
        raise ValueError("numeric cache contains a forbidden label or identifier array")
    audit: dict[str, Any] = {}
    for name in sorted(arrays):
        matrix = np.asarray(arrays[name])
        if matrix.shape != EXPECTED_ARRAYS[name]:
            raise ValueError(f"unexpected shape for {name}: {matrix.shape}")
        if matrix.dtype != np.float32:
            raise ValueError(f"numeric cache matrix {name} is not float32")
        if not np.isfinite(matrix).all():
            raise ValueError(f"numeric cache matrix {name} contains non-finite values")
        audit[name] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "sha256": auxiliary.sha256_array(matrix),
            "finite": True,
        }
    return audit


def write_cache(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    validate_arrays(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{name: np.asarray(arrays[name], dtype=np.float32) for name in sorted(arrays)},
        )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument(
        "--screening-protocol", type=Path, default=auxiliary.SCREENING_PROTOCOL
    )
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    protocol = args.screening_protocol.resolve()
    fold_manifest = args.fold_manifest.resolve()
    page_cache = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy = args.taxonomy.resolve()
    positive_kb = args.positive_kb.resolve()
    cache_path = args.cache.resolve()
    manifest_path = args.manifest.resolve()
    inputs = validate_sources(
        source_inputs(
            exact_identity,
            full_identity,
            protocol,
            fold_manifest,
            page_cache,
            taxonomy,
            positive_kb,
        )
    )
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    auxiliary.load_fold_manifest(fold_manifest, exact_records)

    refnos = {record["REFNO"] for record in full_records}
    paper_text = base.load_page_text(page_cache, refnos)
    pages = curation.load_pages(page_cache, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(taxonomy, positive_kb)
    base_features = base.build_features(full_records, paper_text, concepts)
    curation_features = curation.build_features(
        full_records, pages, concepts, pdf_dir
    )
    arrays = {
        "base_engineered": np.asarray(
            base_features["engineered_rows"], dtype=np.float32
        ),
        "curation_style": np.asarray(curation_features["style"], dtype=np.float32),
        "curation_knowledge": np.asarray(
            curation_features["knowledge"], dtype=np.float32
        ),
        "curation_all": np.asarray(curation_features["all"], dtype=np.float32),
    }
    arrays["combined"] = np.concatenate(
        (arrays["base_engineered"], arrays["curation_all"]), axis=1
    ).astype(np.float32, copy=False)
    matrix_audit = validate_arrays(arrays)
    write_cache(cache_path, arrays)
    manifest = {
        "schema_version": "r3_exact300_auxiliary_numeric_cache_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "scope": {
            "exact_evaluation": exact_coverage,
            "auxiliary_candidate_pool": full_coverage,
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": auxiliary.sha256_records(full_records),
        },
        "runtime_boundary": {
            "labels_read": False,
            "targets_workbook_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "numeric_cache_contains_identifiers": False,
        },
        "allowed_sources": [
            "ROCKNAME from frozen identity",
            "original-paper page-cache text/layout",
            "local PDF byte size",
            "frozen aggregate R3 taxonomy",
            "frozen positive ROCKMAG expert vocabulary",
        ],
        "inputs": inputs,
        "knowledge": knowledge_audit,
        "matrices": matrix_audit,
        "output": {
            "path": str(cache_path),
            "sha256": base.file_sha256(cache_path),
            "contains_labels": False,
            "contains_identifiers": False,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    base.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "cache": manifest["output"],
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": base.file_sha256(manifest_path),
                },
                "matrices": matrix_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
