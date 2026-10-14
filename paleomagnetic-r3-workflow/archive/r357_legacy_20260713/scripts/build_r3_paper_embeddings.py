#!/usr/bin/env python3
"""Build label-free R3 paper embeddings from local original-paper text."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
import sentence_transformers
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402
from experiment_r3_sparse_paper_models import trim_reference_section  # noqa: E402


DEFAULT_OUTPUT = ROOT / "tmp/r3_paper_embeddings/minilm_l6_v2.npz"
DEFAULT_MANIFEST = ROOT / "tmp/r3_paper_embeddings/minilm_l6_v2.manifest.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
BOUNDARY_RE = re.compile(r"\n\s*\n+|(?<=[.!?])\s+(?=[A-Z0-9])")
SPACE_RE = re.compile(r"\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_chunks(
    text: str,
    *,
    target_chars: int = 1100,
    maximum_chunks: int,
) -> list[str]:
    pieces = [SPACE_RE.sub(" ", item).strip() for item in BOUNDARY_RE.split(text)]
    pieces = [item for item in pieces if item]
    chunks: list[str] = []
    current: list[str] = []
    characters = 0
    for piece in pieces:
        if current and characters + len(piece) + 1 > target_chars:
            chunks.append(" ".join(current))
            current = []
            characters = 0
        if len(piece) > target_chars:
            if current:
                chunks.append(" ".join(current))
                current = []
                characters = 0
            for start in range(0, len(piece), target_chars):
                chunks.append(piece[start : start + target_chars])
            continue
        current.append(piece)
        characters += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    if not chunks and text.strip():
        chunks = [SPACE_RE.sub(" ", text).strip()[:target_chars]]
    if len(chunks) <= maximum_chunks:
        return chunks
    indices = np.linspace(0, len(chunks) - 1, maximum_chunks, dtype=int)
    return [chunks[int(index)] for index in indices]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def pool_embeddings(
    embeddings: np.ndarray,
    owners: Sequence[int],
    paper_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = embeddings.shape[1]
    mean = np.zeros((paper_count, dimension), dtype=np.float32)
    maximum = np.full((paper_count, dimension), -np.inf, dtype=np.float32)
    counts = np.zeros(paper_count, dtype=np.int32)
    for vector, owner in zip(embeddings, owners):
        mean[owner] += vector
        maximum[owner] = np.maximum(maximum[owner], vector)
        counts[owner] += 1
    if np.any(counts == 0):
        raise ValueError("every paper must contribute at least one embedding chunk")
    mean /= counts[:, None]
    maximum[~np.isfinite(maximum)] = 0.0
    return l2_normalize(mean), l2_normalize(maximum)


def encode_grouped_chunks(
    model: SentenceTransformer,
    grouped: Sequence[Sequence[str]],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    flat: list[str] = []
    owners: list[int] = []
    for owner, chunks in enumerate(grouped):
        for chunk in chunks:
            flat.append(chunk)
            owners.append(owner)
    encoded = np.asarray(
        model.encode(
            flat,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ),
        dtype=np.float32,
    )
    mean, maximum = pool_embeddings(encoded, owners, len(grouped))
    return mean, maximum, len(flat)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=base.DEFAULT_IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    identity_path = args.identity.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy_path = args.taxonomy.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")

    # This phase deliberately never accepts a labels path.
    records, coverage = base.load_pdf_only_identity(identity_path, pdf_dir)
    refnos = sorted({record["REFNO"] for record in records})
    paper_text = base.load_page_text(page_cache_path, set(refnos))
    concepts, knowledge_audit = base.load_knowledge_patterns(taxonomy_path, None)
    features = base.build_features(records, paper_text, concepts)
    article_documents: Mapping[str, str] = features["article_documents"]

    target_names: dict[str, list[str]] = defaultdict(list)
    for record in records:
        name = record["ROCKNAME"]
        if name not in target_names[record["REFNO"]]:
            target_names[record["REFNO"]].append(name)

    front_chunks: list[list[str]] = []
    knowledge_chunks: list[list[str]] = []
    for refno in refnos:
        targets = " ; ".join(target_names[refno])
        body = trim_reference_section(paper_text[refno], cap=60_000)
        front = f"Targets: {targets}.\n{body[:12_000]}"
        knowledge = f"Targets: {targets}.\n{article_documents[refno]}"
        front_chunks.append(text_chunks(front, maximum_chunks=5))
        knowledge_chunks.append(text_chunks(knowledge, maximum_chunks=18))

    model = SentenceTransformer(args.model, device=args.device)
    front_mean, front_max, front_count = encode_grouped_chunks(
        model, front_chunks, batch_size=args.batch_size
    )
    knowledge_mean, knowledge_max, knowledge_count = encode_grouped_chunks(
        model, knowledge_chunks, batch_size=args.batch_size
    )
    combined = np.concatenate(
        (front_mean, front_max, knowledge_mean, knowledge_max), axis=1
    ).astype(np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        refnos=np.asarray(refnos, dtype=str),
        front_mean=front_mean,
        front_max=front_max,
        knowledge_mean=knowledge_mean,
        knowledge_max=knowledge_max,
        combined=combined,
    )
    manifest = {
        "schema_version": "r3_label_free_paper_embeddings_v1",
        "status": "PASS",
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "allowed_sources": [
                "identity manifest REFNO/RESULTNO/ROCKNAME",
                "original-paper page cache",
                "frozen aggregate taxonomy",
                "pretrained embedding model",
            ],
        },
        "inputs": {
            "identity": {"path": str(identity_path), "sha256": sha256_file(identity_path)},
            "page_cache": {"path": str(page_cache_path), "sha256": sha256_file(page_cache_path)},
            "taxonomy": {"path": str(taxonomy_path), "sha256": sha256_file(taxonomy_path)},
            "pdf_dir": str(pdf_dir),
        },
        "scope": {**coverage, "embedded_papers": len(refnos)},
        "knowledge": knowledge_audit,
        "model": {
            "name": args.model,
            "device": args.device,
            "sentence_transformers_version": sentence_transformers.__version__,
            "torch_version": torch.__version__,
            "embedding_dimension": int(front_mean.shape[1]),
            "combined_dimension": int(combined.shape[1]),
        },
        "chunks": {
            "front": front_count,
            "knowledge": knowledge_count,
            "front_maximum_per_paper": 5,
            "knowledge_maximum_per_paper": 18,
        },
        "output": {"path": str(output_path), "sha256": sha256_file(output_path)},
    }
    base.atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
