#!/usr/bin/env python3
"""Freeze label-free paleomagnetic document-structure features for R3 OOF."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_all_labelled_pdf_grouped_oof as all_oof  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
from r3_grep_entity_pair_features import load_page_cache_rows  # noqa: E402
from r3_paleomag_scope_features import build_paleomag_scope_features  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
FEATURE_OUTPUT = OUTPUT_DIR / "paleomag_scope_features_v1.npz"
FEATURE_MANIFEST = OUTPUT_DIR / "paleomag_scope_features_v1.manifest.json"
FEATURE_AUDIT = OUTPUT_DIR / "paleomag_scope_features_v1.audit.json"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=all_oof.IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=all_oof.PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / FEATURE_OUTPUT.name
    manifest_path = output_dir / FEATURE_MANIFEST.name
    audit_path = output_dir / FEATURE_AUDIT.name

    labelled_records = all_oof.load_identity(args.identity.resolve())
    records, coverage = all_oof.pdf_only_records(labelled_records, args.pdf_dir.resolve())
    alignment_sha256 = all_oof.record_sha256(records)
    page_rows = load_page_cache_rows(
        args.page_cache.resolve(), {record["REFNO"] for record in records}
    )
    features = build_paleomag_scope_features(records, page_rows)
    matrix = np.asarray(features["matrix"], dtype=np.float32)
    if matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
        raise ValueError("paleomagnetic feature matrix is invalid")

    temporary = feature_path.with_name(feature_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, paleomag=matrix)
    temporary.replace(feature_path)

    anchor_counts = Counter(
        int(item["anchor_occurrences"])
        for item in features["row_metadata"]
        if item is not None
    )
    paper_audit = features["paper_audit"]
    audit = {
        "schema_version": "r3_paleomag_scope_feature_audit_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": len(records),
        "papers": features["paper_count"],
        "papers_with_at_least_one_anchorless_target": sum(
            int(item["targets_without_anchor"] > 0) for item in paper_audit
        ),
        "anchorless_canonical_targets": sum(
            int(item["targets_without_anchor"]) for item in paper_audit
        ),
        "anchor_occurrence_distribution_by_row": {
            str(key): value for key, value in sorted(anchor_counts.items())
        },
        "papers_detail": paper_audit,
    }
    base.atomic_json(audit_path, audit)

    manifest = {
        "schema_version": "r3_paleomag_scope_features_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": len(records),
        "papers": features["paper_count"],
        "features": matrix.shape[1],
        "feature_names": features["feature_names"],
        "alignment_sha256": alignment_sha256,
        "matrix_sha256": auxiliary.sha256_array(matrix),
        "contains_labels": False,
        "contains_identifiers": False,
        "allowed_sources": [
            "REFNO/RESULTNO/ROCKNAME identity",
            "original-paper page cache",
            "frozen paleomagnetic lexical and document-structure rules",
        ],
        "forbidden_sources": [
            "PMAGRESULT",
            "row ROCKMAG",
            "R3 labels",
            "prior predictions or evaluations",
        ],
        "coverage": coverage,
        "inputs": {
            "identity": {
                "path": str(args.identity.resolve()),
                "sha256": base.file_sha256(args.identity.resolve()),
            },
            "page_cache": {
                "path": str(args.page_cache.resolve()),
                "sha256": base.file_sha256(args.page_cache.resolve()),
            },
        },
        "output": {
            "path": str(feature_path),
            "sha256": base.file_sha256(feature_path),
        },
        "audit": {
            "path": str(audit_path),
            "sha256": base.file_sha256(audit_path),
        },
    }
    base.atomic_json(manifest_path, manifest)
    print(
        f"frozen paleomagnetic features: rows={len(records)}, papers={features['paper_count']}, "
        f"features={matrix.shape[1]}, output={feature_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
