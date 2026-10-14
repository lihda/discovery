#!/usr/bin/env python3
"""Build a label-free, target-conditioned BGE cache for the R3 auxiliary pool."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = (
    ROOT
    / "tmp/r3_exact_300_resultno_v2/scincl_auxiliary_pool_v1/full_pool_inputs.npz"
)
DEFAULT_INPUT_MANIFEST = DEFAULT_INPUTS.with_suffix(".manifest.json")
DEFAULT_MODEL = ROOT / "tmp/models/bge-base-en-v1.5"
DEFAULT_OUTPUT = (
    ROOT / "tmp/r3_exact_300_resultno_v2/bge_semantic_cache_v1/features.npz"
)
DEFAULT_MANIFEST = DEFAULT_OUTPUT.with_suffix(".manifest.json")
EXPECTED_INPUT_SHA256 = (
    "f6d3b64bf14fc8200fa3555d132bb92d68d98d98aecb569c1ecb67fbb272ec62"
)
EXPECTED_ROWS = 2_172
EXPECTED_COMPONENTS = {"target", "abstract", "local_evidence", "paper_evidence"}
MAX_SEQUENCE_LENGTH = 384


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"model directory is empty: {path}")
    for item in files:
        digest.update(str(item.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def array_record(matrix: np.ndarray) -> dict[str, Any]:
    values = np.ascontiguousarray(matrix)
    return {
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        "finite": bool(np.isfinite(values).all()),
    }


def load_inputs(path: Path, manifest_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    if (
        manifest.get("schema_version") != "r3_scientific_finetune_inputs_v2"
        or manifest.get("status") != "PASS"
        or manifest.get("output", {}).get("sha256") != sha256_file(path)
        or sha256_file(path) != EXPECTED_INPUT_SHA256
        or boundary.get("labels_read") is not False
        or boundary.get("pmagresult_or_rockmag_read") is not False
        or boundary.get("prior_predictions_or_evaluations_read") is not False
        or boundary.get("cache_contains_record_identifiers") is not False
    ):
        raise ValueError("input component cache does not prove a label-free boundary")
    with np.load(path, allow_pickle=False) as payload:
        if set(payload.files) != EXPECTED_COMPONENTS:
            raise ValueError("unexpected text component arrays")
        arrays = {name: np.asarray(payload[name], dtype=str) for name in payload.files}
    if any(values.shape != (EXPECTED_ROWS,) for values in arrays.values()):
        raise ValueError("text component cache is not aligned to 2,172 rows")
    if any(any(not str(value).strip() for value in values) for values in arrays.values()):
        raise ValueError("text component cache contains empty text")
    return arrays, manifest


def encode_unique(model: Any, documents: Sequence[str], batch_size: int) -> np.ndarray:
    unique: list[str] = []
    lookup: dict[str, int] = {}
    inverse = np.empty(len(documents), dtype=np.int64)
    for row, document in enumerate(documents):
        value = str(document)
        position = lookup.get(value)
        if position is None:
            position = len(unique)
            lookup[value] = position
            unique.append(value)
        inverse[row] = position
    encoded = model.encode(
        unique,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(encoded, dtype=np.float32)[inverse]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="mps")
    args = parser.parse_args(argv)

    inputs_path = args.inputs.resolve()
    input_manifest_path = args.input_manifest.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    components, input_manifest = load_inputs(inputs_path, input_manifest_path)
    model_sha256 = sha256_tree(model_path)

    # Importing the encoder remains isolated from the project's lightweight
    # evaluation environment. This executable never accepts a labels path.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(model_path), device=args.device)
    model.max_seq_length = MAX_SEQUENCE_LENGTH
    target = components["target"]
    documents = {
        "target_query": [
            "query: assess rock-magnetic mineral or remanence carrier evidence for "
            + value
            for value in target
        ],
        "abstract": ["passage: " + value for value in components["abstract"]],
        "target_local": [
            "query target: " + name + ". passage: " + evidence
            for name, evidence in zip(target, components["local_evidence"])
        ],
        "target_paper": [
            "query target: " + name + ". passage: " + evidence
            for name, evidence in zip(target, components["paper_evidence"])
        ],
    }
    arrays = {
        name: encode_unique(model, values, args.batch_size)
        for name, values in documents.items()
    }
    relation = np.column_stack(
        (
            np.sum(arrays["target_query"] * arrays["abstract"], axis=1),
            np.sum(arrays["target_query"] * arrays["target_local"], axis=1),
            np.sum(arrays["target_query"] * arrays["target_paper"], axis=1),
            np.sum(arrays["target_local"] * arrays["target_paper"], axis=1),
        )
    ).astype(np.float32)
    arrays["relation"] = relation
    if any(matrix.shape[0] != EXPECTED_ROWS for matrix in arrays.values()):
        raise AssertionError("BGE cache lost row alignment")
    if any(not np.isfinite(matrix).all() for matrix in arrays.values()):
        raise ValueError("BGE encoder produced non-finite features")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output_path)
    manifest = {
        "schema_version": "r3_exact300_bge_semantic_cache_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "runtime_boundary": {
            "labels_read": False,
            "targets_workbook_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "cache_contains_identifiers": False,
            "target_names_are_semantic_text": True,
        },
        "scope": {
            "rows": EXPECTED_ROWS,
            "alignment_sha256": input_manifest.get("inputs", {}).get(
                "alignment_sha256"
            ),
        },
        "encoder": {
            "model": "BAAI/bge-base-en-v1.5",
            "local_path": str(model_path),
            "tree_sha256": model_sha256,
            "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
            "normalized_embeddings": True,
            "device_is_not_part_of_runtime_features": args.device,
        },
        "inputs": {
            "component_cache": {
                "path": str(inputs_path),
                "sha256": sha256_file(inputs_path),
            },
            "component_manifest": {
                "path": str(input_manifest_path),
                "sha256": sha256_file(input_manifest_path),
            },
        },
        "matrices": {name: array_record(value) for name, value in arrays.items()},
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "contains_labels_or_identifiers": False,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
