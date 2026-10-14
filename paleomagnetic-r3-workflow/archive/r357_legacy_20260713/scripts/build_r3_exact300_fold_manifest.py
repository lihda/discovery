#!/usr/bin/env python3
"""Freeze label-free exact300 REFNO folds under the canonical sklearn runtime."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402


IDENTITY = ROOT / "tmp/r3_exact_300_resultno_v2/identity.xlsx"
OUTPUT = ROOT / "tmp/r3_exact_300_resultno_v2/refno_fold_manifest.json"
IDENTITY_SHA256 = (
    "93ea5dfba0a32501a564d06eb020386fd86b11f59ded0982043f90c064ab960b"
)
CANONICAL_SKLEARN_VERSION = "1.9.0"
EXPECTED_ROWS = 300
EXPECTED_GROUPS = 258
N_SPLITS = 5


def clean_record(record: Mapping[str, str]) -> dict[str, str]:
    return {
        "REFNO": str(record["REFNO"]),
        "RESULTNO": str(record["RESULTNO"]),
        "ROCKNAME": str(record["ROCKNAME"]),
    }


def alignment_sha256(records: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        payload = "\0".join(
            (record["REFNO"], record["RESULTNO"], record["ROCKNAME"])
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def assign_folds(records: Sequence[Mapping[str, str]]) -> np.ndarray:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    fold_ids = np.zeros(len(records), dtype=np.int8)
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (_, test) in enumerate(
        splitter.split(np.zeros(len(records)), np.zeros(len(records)), groups), start=1
    ):
        fold_ids[test] = fold
    if set(int(value) for value in fold_ids) != set(range(1, N_SPLITS + 1)):
        raise AssertionError("fold assignment is incomplete")
    group_folds: dict[str, set[int]] = {}
    for group, fold in zip(groups, fold_ids):
        group_folds.setdefault(str(group), set()).add(int(fold))
    if any(len(values) != 1 for values in group_folds.values()):
        raise AssertionError("one REFNO was assigned to multiple folds")
    return fold_ids


def build_manifest(identity: Path) -> dict[str, Any]:
    if sklearn.__version__ != CANONICAL_SKLEARN_VERSION:
        raise ValueError(
            "canonical fold generation requires sklearn "
            f"{CANONICAL_SKLEARN_VERSION}; observed {sklearn.__version__}"
        )
    if base.file_sha256(identity) != IDENTITY_SHA256:
        raise ValueError("identity hash differs from frozen exact300 membership")
    records = [clean_record(record) for record in base.load_training_identity(identity)]
    if len(records) != EXPECTED_ROWS or len(
        {record["REFNO"] for record in records}
    ) != EXPECTED_GROUPS:
        raise ValueError("identity is not the frozen 300-row/258-paper screen")
    fold_ids = assign_folds(records)
    counts = Counter(int(value) for value in fold_ids)
    if counts != Counter({fold: 60 for fold in range(1, N_SPLITS + 1)}):
        raise AssertionError("canonical exact300 folds are not 60 rows each")
    rows = [
        {
            "row_index": index,
            "REFNO": record["REFNO"],
            "RESULTNO": record["RESULTNO"],
            "fold": int(fold_ids[index]),
        }
        for index, record in enumerate(records)
    ]
    fold_digest = hashlib.sha256(
        bytes(int(value) for value in fold_ids)
    ).hexdigest()
    return {
        "schema_version": "r3_exact300_label_free_refno_folds_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "scope": {
            "rows": len(records),
            "paper_groups": len({record["REFNO"] for record in records}),
            "folds": N_SPLITS,
            "rows_per_fold": {
                str(fold): int(counts[fold]) for fold in range(1, N_SPLITS + 1)
            },
        },
        "generator": {
            "method": "sklearn.model_selection.GroupKFold over exact identity REFNO",
            "sklearn_version": sklearn.__version__,
            "labels_argument": "all-zero placeholder; R3 was not opened",
            "shuffle": False,
        },
        "runtime_boundary": {
            "labels_read": False,
            "targets_workbook_read": False,
            "pdf_text_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifiers_used_only_for_alignment_and_grouping": True,
        },
        "input": {
            "identity": str(identity),
            "identity_sha256": base.file_sha256(identity),
            "alignment_sha256": alignment_sha256(records),
        },
        "fold_id_sha256": fold_digest,
        "rows": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    manifest = build_manifest(args.identity.resolve())
    base.atomic_json(args.output.resolve(), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
