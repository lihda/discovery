#!/usr/bin/env python3
"""Freeze a third label-free R3 cohort disjoint from both opened benchmarks."""

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


OUTPUT = ROOT / "tmp/r3_third300_validation_v1/cohort_manifest.json"
RANDOM300_MANIFEST = random300.OUTPUT
RANDOM300_MANIFEST_SHA256 = (
    "f3692dbf1908fbd00e274491822955cbad27d1d6c9db79b02c01108a56efb5c3"
)
RANDOM_SEED = 35_720_260_714
SAMPLE_ROWS = 300


def clean_record(record: Mapping[str, str]) -> dict[str, str]:
    return random300.clean_record(record)


def build_manifest(
    full_identity: Path,
    exact_identity: Path,
    random300_manifest_path: Path,
    pdf_dir: Path,
    *,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    if base.file_sha256(full_identity) != auxiliary.FULL_IDENTITY_SHA256:
        raise ValueError("full identity hash mismatch")
    if base.file_sha256(exact_identity) != auxiliary.EXACT_IDENTITY_SHA256:
        raise ValueError("canonical exact300 identity hash mismatch")
    if base.file_sha256(random300_manifest_path) != RANDOM300_MANIFEST_SHA256:
        raise ValueError("opened random300 cohort hash mismatch")

    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, _ = base.load_pdf_only_identity(exact_identity, pdf_dir)
    random_manifest = json.loads(random300_manifest_path.read_text(encoding="utf-8"))
    random_records, _ = random300.validate_manifest(
        random_manifest, full_records, exact_records
    )
    opened_refnos = {
        record["REFNO"] for record in [*exact_records, *random_records]
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
        raise ValueError("fewer than 300 rows remain outside both opened cohorts")

    rng = np.random.default_rng(seed)
    sampled_positions = np.sort(
        rng.choice(len(eligible_indices), size=SAMPLE_ROWS, replace=False)
    )
    sampled_full_indices = eligible_indices[sampled_positions]
    sampled_records = [clean_record(full_records[int(index)]) for index in sampled_full_indices]
    sampled_refnos = {record["REFNO"] for record in sampled_records}
    excluded_full_indices = [
        index
        for index, record in enumerate(full_records)
        if record["REFNO"] in sampled_refnos
    ]
    train_records = [
        record for record in full_records if record["REFNO"] not in sampled_refnos
    ]
    rows = [
        {
            "sample_index": sample_index,
            "full_index": int(full_index),
            **record,
        }
        for sample_index, (full_index, record) in enumerate(
            zip(sampled_full_indices, sampled_records)
        )
    ]
    return {
        "schema_version": "r3_third300_validation_cohort_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "selection": {
            "seed": seed,
            "generator": "numpy.random.default_rng(PCG64).choice",
            "sampling_unit": "rows without replacement",
            "source_order": "frozen full PDF identity order",
            "eligibility": (
                "local PDF and REFNO absent from canonical exact300 and random300"
            ),
            "sampled_index_sha256": hashlib.sha256(
                np.asarray(sampled_full_indices, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
        "scope": {
            "sample_rows": len(sampled_records),
            "sample_paper_groups": len(sampled_refnos),
            "eligible_rows": int(len(eligible_indices)),
            "eligible_paper_groups": len(
                {full_records[int(index)]["REFNO"] for index in eligible_indices}
            ),
            "all_sample_paper_rows_excluded_from_fit": len(excluded_full_indices),
            "additional_sibling_rows_excluded_from_fit": (
                len(excluded_full_indices) - len(sampled_records)
            ),
            "remaining_training_rows": len(train_records),
            "remaining_training_paper_groups": len(
                {record["REFNO"] for record in train_records}
            ),
        },
        "opened_cohort_exclusions": {
            "canonical_exact300": {
                "identity_sha256": auxiliary.EXACT_IDENTITY_SHA256,
                "rows": len(exact_records),
                "paper_groups": len({record["REFNO"] for record in exact_records}),
            },
            "random300_extension_v1": {
                "manifest_sha256": RANDOM300_MANIFEST_SHA256,
                "rows": len(random_records),
                "paper_groups": len({record["REFNO"] for record in random_records}),
            },
            "third_cohort_refno_overlap": 0,
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
                "path": str(random300_manifest_path),
                "sha256": base.file_sha256(random300_manifest_path),
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
            "opened_cohort_manifests_used_only_for_refno_exclusion": True,
        },
        "evaluation_protocol": {
            "outer_design": "single frozen 300-row paper-isolated holdout",
            "exclude_every_row_of_each_sample_refno_from_all_fits": True,
            "prediction_receipt_written_before_test_label_scoring": True,
            "candidate_family_frozen_before_test_label_access": True,
        },
        "rows": rows,
    }


def validate_manifest(
    manifest: Mapping[str, Any],
    full_records: Sequence[Mapping[str, str]],
    exact_records: Sequence[Mapping[str, str]],
    random_records: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, str]], np.ndarray]:
    boundary = manifest.get("runtime_boundary", {})
    if (
        manifest.get("schema_version") != "r3_third300_validation_cohort_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("selection", {}).get("seed") != RANDOM_SEED
        or boundary.get("labels_read") is not False
        or boundary.get("targets_workbook_read") is not False
        or boundary.get("pmagresult_or_row_rockmag_read") is not False
    ):
        raise ValueError("third300 manifest violates its label-free contract")
    if manifest.get("inputs", {}).get("full_pdf_alignment_sha256") != auxiliary.sha256_records(
        full_records
    ):
        raise ValueError("third300 full identity alignment mismatch")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != SAMPLE_ROWS:
        raise ValueError("third300 must contain exactly 300 rows")
    records: list[dict[str, str]] = []
    full_indices = np.zeros(SAMPLE_ROWS, dtype=np.int64)
    for sample_index, entry in enumerate(rows):
        if not isinstance(entry, Mapping) or entry.get("sample_index") != sample_index:
            raise ValueError("third300 sample ordering is invalid")
        full_index = int(entry.get("full_index", -1))
        if full_index < 0 or full_index >= len(full_records):
            raise ValueError("third300 contains an invalid full index")
        observed = clean_record(entry)
        if observed != clean_record(full_records[full_index]):
            raise ValueError("third300 row differs from frozen full identity")
        records.append(observed)
        full_indices[sample_index] = full_index
    if len(set(int(index) for index in full_indices)) != SAMPLE_ROWS:
        raise ValueError("third300 contains duplicate rows")
    if auxiliary.sha256_records(records) != manifest.get("sample_alignment_sha256"):
        raise ValueError("third300 sample alignment digest mismatch")
    opened_refnos = {
        record["REFNO"] for record in [*exact_records, *random_records]
    }
    if opened_refnos & {record["REFNO"] for record in records}:
        raise ValueError("third300 overlaps an opened benchmark paper")
    return records, full_indices


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--random300-manifest", type=Path, default=RANDOM300_MANIFEST)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.seed != RANDOM_SEED:
        raise ValueError(f"third300-v1 seed is frozen as {RANDOM_SEED}")
    manifest = build_manifest(
        args.full_identity.resolve(),
        args.exact_identity.resolve(),
        args.random300_manifest.resolve(),
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
