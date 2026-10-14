#!/usr/bin/env python3
"""Freeze the final 300-row R3 cohort disjoint from three opened benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import freeze_r3_random300_extension_cohort as random300  # noqa: E402
import freeze_r3_third300_validation_cohort as third300  # noqa: E402


OUTPUT = ROOT / "tmp/r3_fourth300_validation_v1/cohort_manifest.json"
THIRD300_MANIFEST = third300.OUTPUT
THIRD300_MANIFEST_SHA256 = (
    "a275dd97b234a746ce3ba564157de3c7cf9766f37a445f4a8ae1b30307d2e3e1"
)
RANDOM_SEED = 35_720_260_715
SAMPLE_ROWS = 300


def clean_record(record: Mapping[str, str]) -> dict[str, str]:
    return random300.clean_record(record)


def build_manifest(
    full_identity: Path,
    exact_identity: Path,
    random_manifest_path: Path,
    third_manifest_path: Path,
    pdf_dir: Path,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    expected = {
        full_identity: auxiliary.FULL_IDENTITY_SHA256,
        exact_identity: auxiliary.EXACT_IDENTITY_SHA256,
        random_manifest_path: third300.RANDOM300_MANIFEST_SHA256,
        third_manifest_path: THIRD300_MANIFEST_SHA256,
    }
    for path, digest in expected.items():
        if base.file_sha256(path) != digest:
            raise ValueError(f"frozen input hash mismatch: {path}")
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, _ = base.load_pdf_only_identity(exact_identity, pdf_dir)
    random_manifest = json.loads(random_manifest_path.read_text(encoding="utf-8"))
    random_records, _ = random300.validate_manifest(
        random_manifest, full_records, exact_records
    )
    third_manifest = json.loads(third_manifest_path.read_text(encoding="utf-8"))
    third_records, _ = third300.validate_manifest(
        third_manifest, full_records, exact_records, random_records
    )
    opened_refnos = {
        record["REFNO"]
        for record in [*exact_records, *random_records, *third_records]
    }
    eligible_indices = np.asarray(
        [
            index
            for index, record in enumerate(full_records)
            if record["REFNO"] not in opened_refnos
        ],
        dtype=np.int64,
    )
    if len(eligible_indices) < SAMPLE_ROWS:
        raise ValueError("fewer than 300 rows remain outside all opened cohorts")
    rng = np.random.default_rng(seed)
    sampled_positions = np.sort(
        rng.choice(len(eligible_indices), size=SAMPLE_ROWS, replace=False)
    )
    sampled_full_indices = eligible_indices[sampled_positions]
    sampled_records = [clean_record(full_records[int(index)]) for index in sampled_full_indices]
    sampled_refnos = {record["REFNO"] for record in sampled_records}
    excluded_indices = [
        index
        for index, record in enumerate(full_records)
        if record["REFNO"] in sampled_refnos
    ]
    train_records = [
        record for record in full_records if record["REFNO"] not in sampled_refnos
    ]
    rows = [
        {"sample_index": order, "full_index": int(index), **record}
        for order, (index, record) in enumerate(
            zip(sampled_full_indices, sampled_records)
        )
    ]
    return {
        "schema_version": "r3_fourth300_validation_cohort_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "selection": {
            "seed": seed,
            "generator": "numpy.random.default_rng(PCG64).choice",
            "sampling_unit": "rows without replacement",
            "eligibility": "local PDF and REFNO absent from all three opened cohorts",
            "sampled_index_sha256": hashlib.sha256(
                np.asarray(sampled_full_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "scope": {
            "sample_rows": len(sampled_records),
            "sample_paper_groups": len(sampled_refnos),
            "eligible_rows": len(eligible_indices),
            "eligible_paper_groups": len(
                {full_records[int(index)]["REFNO"] for index in eligible_indices}
            ),
            "all_sample_paper_rows_excluded_from_fit": len(excluded_indices),
            "additional_sibling_rows_excluded_from_fit": (
                len(excluded_indices) - len(sampled_records)
            ),
            "remaining_training_rows": len(train_records),
            "remaining_training_paper_groups": len(
                {record["REFNO"] for record in train_records}
            ),
        },
        "opened_cohort_exclusions": {
            "canonical_exact300_papers": len(
                {record["REFNO"] for record in exact_records}
            ),
            "random300_papers": len({record["REFNO"] for record in random_records}),
            "third300_papers": len({record["REFNO"] for record in third_records}),
            "fourth_cohort_refno_overlap": 0,
        },
        "inputs": {
            "full_identity": {
                "path": str(full_identity),
                "sha256": base.file_sha256(full_identity),
                "pdf_coverage": full_coverage,
            },
            "exact_identity": {
                "path": str(exact_identity),
                "sha256": base.file_sha256(exact_identity),
            },
            "random300_manifest": {
                "path": str(random_manifest_path),
                "sha256": base.file_sha256(random_manifest_path),
            },
            "third300_manifest": {
                "path": str(third_manifest_path),
                "sha256": base.file_sha256(third_manifest_path),
            },
            "full_pdf_alignment_sha256": auxiliary.sha256_records(full_records),
        },
        "sample_alignment_sha256": auxiliary.sha256_records(sampled_records),
        "runtime_boundary": {
            "labels_read": False,
            "targets_workbook_read": False,
            "pdf_text_read": False,
            "pmagresult_or_row_rockmag_read": False,
            "prior_predictions_or_evaluations_read_for_sampling": False,
            "opened_manifests_used_only_for_refno_exclusion": True,
        },
        "evaluation_protocol": {
            "outer_design": "single sealed 300-row paper-isolated holdout",
            "exclude_every_row_of_each_sample_refno_from_all_fits": True,
            "candidate_family_frozen_before_test_label_access": True,
            "prediction_receipt_written_before_test_label_scoring": True,
        },
        "rows": rows,
    }


def validate_manifest(
    manifest: Mapping[str, Any],
    full_records: Sequence[Mapping[str, str]],
    opened_records: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], np.ndarray]:
    boundary = manifest.get("runtime_boundary", {})
    if (
        manifest.get("schema_version") != "r3_fourth300_validation_cohort_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("selection", {}).get("seed") != RANDOM_SEED
        or boundary.get("labels_read") is not False
        or boundary.get("targets_workbook_read") is not False
        or boundary.get("pmagresult_or_row_rockmag_read") is not False
    ):
        raise ValueError("fourth300 violates its sealed cohort contract")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != SAMPLE_ROWS:
        raise ValueError("fourth300 must contain exactly 300 rows")
    records: list[dict[str, str]] = []
    indices = np.zeros(SAMPLE_ROWS, dtype=np.int64)
    for order, entry in enumerate(rows):
        if not isinstance(entry, Mapping) or entry.get("sample_index") != order:
            raise ValueError("fourth300 sample order is invalid")
        index = int(entry.get("full_index", -1))
        if index < 0 or index >= len(full_records):
            raise ValueError("fourth300 full index is invalid")
        observed = clean_record(entry)
        if observed != clean_record(full_records[index]):
            raise ValueError("fourth300 identity alignment mismatch")
        records.append(observed)
        indices[order] = index
    if len(set(int(index) for index in indices)) != SAMPLE_ROWS:
        raise ValueError("fourth300 contains duplicate rows")
    if auxiliary.sha256_records(records) != manifest.get("sample_alignment_sha256"):
        raise ValueError("fourth300 sample digest mismatch")
    opened_refnos = {record["REFNO"] for record in opened_records}
    if opened_refnos & {record["REFNO"] for record in records}:
        raise ValueError("fourth300 overlaps an opened paper")
    return records, indices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--random300-manifest", type=Path, default=random300.OUTPUT)
    parser.add_argument("--third300-manifest", type=Path, default=THIRD300_MANIFEST)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.seed != RANDOM_SEED:
        raise ValueError(f"fourth300-v1 seed is frozen as {RANDOM_SEED}")
    manifest = build_manifest(
        args.full_identity.resolve(),
        args.exact_identity.resolve(),
        args.random300_manifest.resolve(),
        args.third300_manifest.resolve(),
        args.pdf_dir.resolve(),
        seed=args.seed,
    )
    base.atomic_json(args.output.resolve(), manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scope": manifest["scope"],
                "sample_alignment_sha256": manifest["sample_alignment_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
