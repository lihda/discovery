#!/usr/bin/env python3
"""Freeze label-free own-study attribution features for exact300 R3 evaluation."""

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

import build_r3_exact300_auxiliary_numeric_cache as numeric_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_own_study_events as own_study  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/own_study_cache_v1"
DEFAULT_CACHE = OUTPUT_DIR / "features.npz"
DEFAULT_MANIFEST = OUTPUT_DIR / "features.manifest.json"
EXPECTED_ARRAY_NAMES = {
    "own_article",
    "own_target",
    "own_all",
    "curation_all_plus_own_article",
    "curation_all_plus_own_target",
    "curation_all_plus_own_all",
    "combined_plus_own_article",
    "combined_plus_own_target",
    "combined_plus_own_all",
}
FORBIDDEN_NAME_RE = re.compile(
    r"(?i)^(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)$"
)


def validate_arrays(
    arrays: Mapping[str, np.ndarray], expected_rows: int
) -> dict[str, Any]:
    if set(arrays) != EXPECTED_ARRAY_NAMES:
        raise ValueError(
            "own-study cache arrays differ from the frozen schema: "
            f"missing={sorted(EXPECTED_ARRAY_NAMES - set(arrays))}, "
            f"extra={sorted(set(arrays) - EXPECTED_ARRAY_NAMES)}"
        )
    if any(FORBIDDEN_NAME_RE.fullmatch(name) for name in arrays):
        raise ValueError("own-study cache contains a forbidden label or identifier array")
    audit: dict[str, Any] = {}
    for name in sorted(arrays):
        matrix = np.asarray(arrays[name])
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] < 1:
            raise ValueError(f"unexpected shape for {name}: {matrix.shape}")
        if matrix.dtype != np.float32:
            raise ValueError(f"own-study matrix {name} is not float32")
        if not np.isfinite(matrix).all():
            raise ValueError(f"own-study matrix {name} contains non-finite values")
        audit[name] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "sha256": auxiliary.sha256_array(matrix),
            "finite": True,
        }
    return audit


def write_cache(path: Path, arrays: Mapping[str, np.ndarray], expected_rows: int) -> None:
    validate_arrays(arrays, expected_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(arrays[name], dtype=np.float32)
                for name in sorted(arrays)
            },
        )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--screening-protocol", type=Path, default=auxiliary.SCREENING_PROTOCOL)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--numeric-cache", type=Path, default=numeric_builder.DEFAULT_CACHE)
    parser.add_argument(
        "--numeric-cache-manifest", type=Path, default=numeric_builder.DEFAULT_MANIFEST
    )
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
    numeric_cache_path = args.numeric_cache.resolve()
    numeric_manifest_path = args.numeric_cache_manifest.resolve()
    cache_path = args.cache.resolve()
    manifest_path = args.manifest.resolve()

    inputs = numeric_builder.validate_sources(
        numeric_builder.source_inputs(
            exact_identity,
            full_identity,
            protocol,
            fold_manifest,
            page_cache,
            taxonomy,
            positive_kb,
        )
    )
    numeric_arrays, numeric_manifest = numeric_sweep.validate_cache(
        numeric_cache_path, numeric_manifest_path
    )
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    auxiliary.load_fold_manifest(fold_manifest, exact_records)
    full_alignment = auxiliary.sha256_records(full_records)
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from auxiliary identity")

    refnos = {record["REFNO"] for record in full_records}
    structured_pages, _paper_text = own_study.load_structured_pages(page_cache, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(taxonomy, positive_kb)
    own = own_study.build_own_study_features(full_records, structured_pages, concepts)
    article = np.asarray(own["row_article"], dtype=np.float32)
    target = np.asarray(own["row_target"], dtype=np.float32)
    own_all = np.concatenate((article, target), axis=1).astype(np.float32, copy=False)
    curation_all = numeric_arrays["curation_all"]
    combined = numeric_arrays["combined"]
    arrays = {
        "own_article": article,
        "own_target": target,
        "own_all": own_all,
        "curation_all_plus_own_article": np.concatenate((curation_all, article), axis=1),
        "curation_all_plus_own_target": np.concatenate((curation_all, target), axis=1),
        "curation_all_plus_own_all": np.concatenate((curation_all, own_all), axis=1),
        "combined_plus_own_article": np.concatenate((combined, article), axis=1),
        "combined_plus_own_target": np.concatenate((combined, target), axis=1),
        "combined_plus_own_all": np.concatenate((combined, own_all), axis=1),
    }
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}
    matrix_audit = validate_arrays(arrays, len(full_records))
    write_cache(cache_path, arrays, len(full_records))

    row_meta = own["row_meta"]
    metadata_summary = {
        "paper_event_count": int(own["paper_event_count"]),
        "paper_episode_count": int(own["paper_episode_count"]),
        "rows_with_target_mention": int(
            sum(int(item["target_mention_count"]) > 0 for item in row_meta)
        ),
        "rows_with_target_local_strong": int(
            sum(int(item["target_local_strong"]) > 0 for item in row_meta)
        ),
        "rows_with_target_assigned_strong": int(
            sum(int(item["target_assigned_strong"]) > 0 for item in row_meta)
        ),
        "rows_with_article_strong_chain": int(
            sum(int(item["article_strong_chain_episodes"]) > 0 for item in row_meta)
        ),
    }
    manifest = {
        "schema_version": "r3_exact300_own_study_cache_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "scope": {
            "exact_evaluation": exact_coverage,
            "auxiliary_candidate_pool": full_coverage,
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": full_alignment,
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
            "frozen aggregate R3 taxonomy",
            "frozen positive ROCKMAG expert vocabulary",
            "frozen label-free auxiliary numeric cache",
        ],
        "inputs": {
            **inputs,
            "numeric_cache": {
                "path": str(numeric_cache_path),
                "sha256": base.file_sha256(numeric_cache_path),
            },
            "numeric_cache_manifest": {
                "path": str(numeric_manifest_path),
                "sha256": base.file_sha256(numeric_manifest_path),
            },
        },
        "knowledge": knowledge_audit,
        "feature_names": {
            "own_article": list(own["article_names"]),
            "own_target": list(own["target_names"]),
        },
        "event_summary": metadata_summary,
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
                "event_summary": metadata_summary,
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
