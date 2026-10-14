#!/usr/bin/env python3
"""Freeze exact-300 DeBERTa features before any R3 label access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_deberta_target_nli_embeddings as builder  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/deberta_target_nli_v2"
DEFAULT_FEATURES = OUTPUT_DIR / "features.npz"
DEFAULT_MANIFEST = OUTPUT_DIR / "features.manifest.json"
DEFAULT_OUTPUT = OUTPUT_DIR / "feature_freeze.json"
SCREENING_PROTOCOL = ROOT / "tmp/r3_exact_300_resultno_v2/screening_protocol.json"
SCREENING_PROTOCOL_SHA256 = (
    "6412244a5d268e6dcabfaed63f6f529d4976dd17ef6c4313f70bdb6748c04b4f"
)
PRIMARY_STRATEGY = (
    "repair__joint_ensemble__curation_deberta_taxonomy_relative__paper_style_25"
)
FORBIDDEN_ARRAY_RE = re.compile(
    r"(?i)(?:labels?|answers?|refnos?|resultnos?|rocknames?|identifiers?|ids?)"
)


def validate_numeric_cache(path: Path) -> dict[str, list[int]]:
    with np.load(path, allow_pickle=False) as payload:
        names = set(payload.files)
        expected = {
            "structured",
            *{
                f"{hypothesis}__{pool}"
                for hypothesis in builder.HYPOTHESIS_NAMES
                for pool in ("mean", "cls", "top")
            },
        }
        if names != expected:
            raise ValueError(
                f"unexpected feature arrays: missing={sorted(expected - names)}, "
                f"extra={sorted(names - expected)}"
            )
        if any(FORBIDDEN_ARRAY_RE.search(name) for name in names):
            raise ValueError("numeric cache contains a forbidden label or identifier array")
        shapes = {name: list(payload[name].shape) for name in sorted(names)}
        if any(shape[0] != builder.FROZEN_ROWS or len(shape) != 2 for shape in shapes.values()):
            raise ValueError("numeric cache does not cover exactly 300 aligned rows")
        if any(not np.isfinite(payload[name]).all() for name in names):
            raise ValueError("numeric cache contains non-finite values")
    return shapes


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    features_path = args.features.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    if manifest.get("schema_version") != "r3_exact300_deberta_target_nli_embeddings_v2":
        raise ValueError("manifest schema is not the exact-300 DeBERTa v2 schema")
    if manifest.get("status") != "PASS" or any(
        boundary.get(name) is not False
        for name in (
            "labels_read",
            "pmagresult_or_rockmag_read",
            "prior_predictions_or_evaluations_read",
            "identifier_features",
            "numeric_cache_contains_identifiers",
        )
    ):
        raise ValueError("manifest does not prove a label-free feature build")
    if (
        manifest.get("inputs", {}).get("identity", {}).get("sha256")
        != builder.FROZEN_IDENTITY_SHA256
        or manifest.get("inputs", {}).get("alignment_sha256")
        != builder.FROZEN_ALIGNMENT_SHA256
        or manifest.get("scope", {}).get("embedded_rows") != builder.FROZEN_ROWS
        or manifest.get("scope", {}).get("embedded_papers") != builder.FROZEN_PAPERS
        or manifest.get("scope", {}).get("target_limit") is not None
    ):
        raise ValueError("manifest scope is not the frozen exact-300 membership")
    if manifest.get("output", {}).get("sha256") != base.file_sha256(features_path):
        raise ValueError("manifest does not bind the requested numerical cache")
    if base.file_sha256(SCREENING_PROTOCOL) != SCREENING_PROTOCOL_SHA256:
        raise ValueError("exact-300 screening protocol changed")

    shapes = validate_numeric_cache(features_path)
    freeze = {
        "schema_version": "r3_exact300_deberta_feature_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": {
            "identity_sha256": builder.FROZEN_IDENTITY_SHA256,
            "alignment_sha256": builder.FROZEN_ALIGNMENT_SHA256,
            "rows": builder.FROZEN_ROWS,
            "papers": builder.FROZEN_PAPERS,
        },
        "screening_protocol_sha256": SCREENING_PROTOCOL_SHA256,
        "features_sha256": base.file_sha256(features_path),
        "manifest_sha256": base.file_sha256(manifest_path),
        "builder_sha256": manifest["generator"]["sha256"],
        "primary_strategy": PRIMARY_STRATEGY,
        "array_shapes": shapes,
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "numeric_cache_contains_labels_or_identifiers": False,
        },
    }
    base.atomic_json(args.output.resolve(), freeze)
    print(json.dumps(freeze, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
