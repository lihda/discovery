#!/usr/bin/env python3
"""Build label-free, row-aligned SciNCL features for PDF-backed R3 rows."""

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


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_MODEL = ROOT / "tmp/models/scincl"
DEFAULT_OUTPUT = ROOT / "tmp/r3_scientific_embeddings/scincl_row_features.npz"
DEFAULT_MANIFEST = ROOT / "tmp/r3_scientific_embeddings/scincl_row_features.manifest.json"
SCINCL_UPSTREAM = "malteos/scincl"
SCINCL_REVISION = "ebc5348d184ba2fc9beee69b4e394263fce57b2e"
PILOT_SELECTION_SALT = "r3-scincl-source-stratified-pilot-v1"
ABSTRACT_HEADING_RE = re.compile(
    r"(?im)^\s*(?:abstract|summary)(?:\s*[:.-]|\s*$)"
)
INTRODUCTION_HEADING_RE = re.compile(
    r"(?im)^\s*(?:(?:1|i)[.\s]+)?introduction\s*$"
)
PAGE_MARKER_RE = re.compile(r"(?m)^\[\[PAGE\s+\d+\]\]\s*$")
SPACE_RE = re.compile(r"\s+")
BOUNDARY_RE = re.compile(r"\n\s*\n+|(?<=[.!?])\s+(?=[A-Z0-9])")

# The query is fixed expert knowledge, not learned from R3 labels. It is used only
# to select the most criterion-relevant original-text chunk before classification.
R3_EXPERT_QUERY = (
    "Rock magnetic experiments identify the magnetic mineral or remanence carrier "
    "using IRM acquisition or backfield curves, hysteresis, thermomagnetic or Curie "
    "temperature measurements, coercivity spectra, FORC diagrams, low-temperature "
    "magnetic behavior, microscopy, or mineral-specific magnetic tests."
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alignment_sha256(records: Sequence[Mapping[str, str]]) -> str:
    """Hash alignment keys without exposing them in the numerical cache."""

    digest = hashlib.sha256()
    for record in records:
        payload = "\0".join(
            (record["REFNO"], record["RESULTNO"], record["ROCKNAME"])
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def source_stratified_pilot(
    records: Sequence[dict[str, str]],
    paper_count: int,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Select a deterministic identity-only pilot without consulting labels."""

    refno_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    for record in records:
        refno_records[record["REFNO"]].append(record)
    total_papers = len(refno_records)
    if paper_count <= 0 or paper_count >= total_papers:
        return list(records), {
            "mode": "full",
            "salt": PILOT_SELECTION_SALT,
            "requested_papers": paper_count,
            "selected_papers": total_papers,
            "selected_rows": len(records),
        }

    strata: dict[str, list[str]] = defaultdict(list)
    for refno, rows in refno_records.items():
        canonical_targets = {
            base.canonical_target_identity(record["ROCKNAME"]) for record in rows
        }
        target_band = "single_target" if len(canonical_targets) == 1 else "multi_target"
        row_band = "rows_1" if len(rows) == 1 else "rows_2" if len(rows) == 2 else "rows_3plus"
        strata[f"{target_band}__{row_band}"].append(refno)

    exact = {
        name: paper_count * len(refnos) / total_papers
        for name, refnos in strata.items()
    }
    quota = {name: int(value) for name, value in exact.items()}
    remaining = paper_count - sum(quota.values())
    remainder_order = sorted(
        strata,
        key=lambda name: (exact[name] - quota[name], len(strata[name]), name),
        reverse=True,
    )
    for name in remainder_order[:remaining]:
        quota[name] += 1

    def score(refno: str) -> tuple[bytes, str]:
        payload = f"{PILOT_SELECTION_SALT}\0{refno}".encode("utf-8")
        return hashlib.sha256(payload).digest(), refno

    selected_refnos: set[str] = set()
    selected_by_stratum: dict[str, int] = {}
    for name, refnos in sorted(strata.items()):
        chosen = sorted(refnos, key=score)[: quota[name]]
        selected_refnos.update(chosen)
        selected_by_stratum[name] = len(chosen)
    selected = [record for record in records if record["REFNO"] in selected_refnos]
    if len(selected_refnos) != paper_count:
        raise AssertionError("source-stratified pilot did not select its requested paper count")
    return selected, {
        "mode": "source_stratified_refno_hash_pilot",
        "salt": PILOT_SELECTION_SALT,
        "requested_papers": paper_count,
        "selected_papers": len(selected_refnos),
        "selected_rows": len(selected),
        "population_strata": {name: len(refnos) for name, refnos in sorted(strata.items())},
        "selected_strata": dict(sorted(selected_by_stratum.items())),
    }


def extract_abstract_proxy(text: str, targets: Sequence[str]) -> str:
    """Return a bounded title/abstract proxy robust to older-paper OCR layouts."""

    front = PAGE_MARKER_RE.sub("", text[:20_000]).strip()
    abstract_match = ABSTRACT_HEADING_RE.search(front)
    if abstract_match:
        start = abstract_match.end()
        introduction = INTRODUCTION_HEADING_RE.search(front, start)
        end = introduction.start() if introduction else min(len(front), start + 8_000)
        abstract = front[start:end].strip()
        title_context = front[: abstract_match.start()].strip()[-2_500:]
        body = f"{title_context} [SEP] {abstract}"
    else:
        introduction = INTRODUCTION_HEADING_RE.search(front)
        body = front[: introduction.start() if introduction else 6_000]
    target_text = " ; ".join(targets)
    return SPACE_RE.sub(" ", f"{target_text} [SEP] {body}").strip()[:12_000]


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def text_chunks(
    text: str,
    *,
    target_chars: int,
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
            chunks.extend(
                piece[start : start + target_chars]
                for start in range(0, len(piece), target_chars)
            )
            continue
        current.append(piece)
        characters += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    if not chunks and text.strip():
        chunks = [SPACE_RE.sub(" ", text).strip()[:target_chars]]
    if len(chunks) <= maximum_chunks:
        return chunks
    positions = np.linspace(0, len(chunks) - 1, maximum_chunks, dtype=int)
    return [chunks[int(position)] for position in positions]


def select_expert_chunks(
    text: str,
    concepts: Sequence[Mapping[str, Any]],
    *,
    target_prefix: str,
    target_chars: int = 1_500,
    maximum_chunks: int = 6,
) -> list[str]:
    """Use frozen taxonomy hits to shortlist local chunks before encoding."""

    candidates = text_chunks(
        text,
        target_chars=target_chars,
        maximum_chunks=10_000,
    )
    if len(candidates) > maximum_chunks:
        scored: list[tuple[int, int, int]] = []
        for index, chunk in enumerate(candidates):
            counts = base.concept_counts(chunk, concepts)
            scored.append((sum(value > 0 for value in counts), sum(counts), index))
        if max((distinct, total) for distinct, total, _ in scored) == (0, 0):
            selected = np.linspace(
                0, len(candidates) - 1, maximum_chunks, dtype=int
            ).tolist()
        else:
            ranked = sorted(
                scored,
                key=lambda item: (item[0], item[1], -item[2]),
                reverse=True,
            )[:maximum_chunks]
            selected = sorted(index for _, _, index in ranked)
        candidates = [candidates[int(index)] for index in selected]
    return [f"Target: {target_prefix}. [SEP] {chunk}" for chunk in candidates]


def encode_documents(
    model: Any,
    documents: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    encoded = model.encode(
        list(documents),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(encoded, dtype=np.float32)


def criterion_pool(
    embeddings: np.ndarray,
    owners: Sequence[int],
    owner_count: int,
    query: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool chunks by mean, maximum, top-1, and query-ranked top-3 mean."""

    buckets: list[list[np.ndarray]] = [[] for _ in range(owner_count)]
    for embedding, owner in zip(embeddings, owners):
        buckets[owner].append(embedding)
    if any(not bucket for bucket in buckets):
        raise ValueError("every document owner must contribute at least one chunk")

    means: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    tops: list[np.ndarray] = []
    top3_means: list[np.ndarray] = []
    for bucket in buckets:
        matrix = np.asarray(bucket, dtype=np.float32)
        scores = matrix @ query
        ranking = np.argsort(scores)[::-1]
        means.append(matrix.mean(axis=0))
        maxima.append(matrix.max(axis=0))
        tops.append(matrix[int(ranking[0])])
        top3_means.append(matrix[ranking[: min(3, len(ranking))]].mean(axis=0))
    return (
        l2_normalize(np.asarray(means, dtype=np.float32)),
        l2_normalize(np.asarray(maxima, dtype=np.float32)),
        l2_normalize(np.asarray(tops, dtype=np.float32)),
        l2_normalize(np.asarray(top3_means, dtype=np.float32)),
    )


def encode_chunk_groups(
    model: Any,
    grouped: Sequence[Sequence[str]],
    query: np.ndarray,
    *,
    batch_size: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]:
    flat: list[str] = []
    owners: list[int] = []
    for owner, chunks in enumerate(grouped):
        for chunk in chunks:
            flat.append(chunk)
            owners.append(owner)
    embeddings = encode_documents(model, flat, batch_size=batch_size)
    return criterion_pool(embeddings, owners, len(grouped), query), len(flat)


def row_take(
    matrix: np.ndarray,
    positions: Sequence[int],
) -> np.ndarray:
    return np.asarray(matrix[np.asarray(positions, dtype=np.int64)], dtype=np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=base.DEFAULT_IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--pilot-paper-count",
        type=int,
        default=0,
        help="identity-only source-stratified pilot size; 0 encodes all PDF papers",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    try:
        import sentence_transformers
        from sentence_transformers import SentenceTransformer
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "SciNCL feature building requires sentence-transformers and torch"
        ) from exc
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")

    identity_path = args.identity.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy_path = args.taxonomy.resolve()
    model_path = args.model.resolve()
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()
    required_model_files = (
        "config.json",
        "modules.json",
        "model.safetensors",
        "vocab.txt",
        "1_Pooling/config.json",
    )
    missing_model_files = [
        name for name in required_model_files if not (model_path / name).is_file()
    ]
    if missing_model_files:
        raise SystemExit(f"incomplete local SciNCL snapshot: {missing_model_files}")

    # This phase has no labels argument and never opens targets.xlsx.
    all_records, coverage = base.load_pdf_only_identity(identity_path, pdf_dir)
    records, selection_audit = source_stratified_pilot(
        all_records, args.pilot_paper_count
    )
    refnos = sorted({record["REFNO"] for record in records})
    paper_position = {refno: index for index, refno in enumerate(refnos)}
    paper_text = base.load_page_text(page_cache_path, set(refnos))
    concepts, knowledge_audit = base.load_knowledge_patterns(taxonomy_path, None)
    source_features = base.build_features(records, paper_text, concepts)

    targets_by_paper: dict[str, list[str]] = defaultdict(list)
    for record in records:
        if record["ROCKNAME"] not in targets_by_paper[record["REFNO"]]:
            targets_by_paper[record["REFNO"]].append(record["ROCKNAME"])

    abstracts = [
        extract_abstract_proxy(paper_text[refno], targets_by_paper[refno])
        for refno in refnos
    ]
    paper_chunk_groups = [
        text_chunks(
            source_features["article_documents"][refno],
            target_chars=1_500,
            maximum_chunks=12,
        )
        for refno in refnos
    ]

    target_position: dict[tuple[str, str], int] = {}
    local_chunk_groups: list[list[str]] = []
    local_positions: list[int] = []
    for index, (record, meta) in enumerate(zip(records, source_features["row_meta"])):
        key = (record["REFNO"], meta["canonical_target"])
        if key not in target_position:
            target_position[key] = len(local_chunk_groups)
            local_chunk_groups.append(
                select_expert_chunks(
                    source_features["local_documents"][index],
                    concepts,
                    target_prefix=record["ROCKNAME"],
                )
            )
        local_positions.append(target_position[key])

    model = SentenceTransformer(str(model_path), device=args.device)
    query = encode_documents(
        model, [R3_EXPERT_QUERY], batch_size=args.batch_size
    )[0]
    abstract_embeddings = encode_documents(model, abstracts, batch_size=args.batch_size)
    paper_pools, paper_chunk_count = encode_chunk_groups(
        model,
        paper_chunk_groups,
        query,
        batch_size=args.batch_size,
    )
    local_pools, local_chunk_count = encode_chunk_groups(
        model,
        local_chunk_groups,
        query,
        batch_size=args.batch_size,
    )

    paper_positions = [paper_position[record["REFNO"]] for record in records]
    paper_mean, paper_max, paper_top1, paper_top3 = paper_pools
    local_mean, local_max, local_top1, local_top3 = local_pools
    paper_query_scores = np.column_stack(
        (paper_mean @ query, paper_top1 @ query, paper_top3 @ query)
    ).astype(np.float32)
    local_query_scores = np.column_stack(
        (local_mean @ query, local_top1 @ query, local_top3 @ query)
    ).astype(np.float32)
    arrays = {
        "abstract": row_take(abstract_embeddings, paper_positions),
        "paper_mean": row_take(paper_mean, paper_positions),
        "paper_max": row_take(paper_max, paper_positions),
        "paper_top1": row_take(paper_top1, paper_positions),
        "paper_top3": row_take(paper_top3, paper_positions),
        "local_mean": row_take(local_mean, local_positions),
        "local_max": row_take(local_max, local_positions),
        "local_top1": row_take(local_top1, local_positions),
        "local_top3": row_take(local_top3, local_positions),
        "retrieval_scores": np.concatenate(
            (
                row_take(paper_query_scores, paper_positions),
                row_take(local_query_scores, local_positions),
            ),
            axis=1,
        ).astype(np.float32),
        "engineered": np.asarray(source_features["engineered_rows"], dtype=np.float32),
    }
    if any(matrix.shape[0] != len(records) for matrix in arrays.values()):
        raise AssertionError("one or more SciNCL matrices lost row alignment")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    model_hashes = {
        name: sha256_file(model_path / name) for name in required_model_files
    }
    manifest = {
        "schema_version": "r3_label_free_scientific_embeddings_v1",
        "status": "PASS",
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifier_features": False,
            "numeric_cache_contains_identifiers": False,
            "allowed_sources": [
                "identity manifest REFNO/RESULTNO/ROCKNAME for alignment and target text",
                "local original-paper page cache",
                "frozen aggregate taxonomy",
                "fixed pretrained SciNCL weights",
                "fixed R3 expert retrieval query",
            ],
        },
        "inputs": {
            "identity": {"path": str(identity_path), "sha256": sha256_file(identity_path)},
            "page_cache": {"path": str(page_cache_path), "sha256": sha256_file(page_cache_path)},
            "taxonomy": {"path": str(taxonomy_path), "sha256": sha256_file(taxonomy_path)},
            "pdf_dir": str(pdf_dir),
            "alignment_sha256": alignment_sha256(records),
        },
        "scope": {
            **coverage,
            "selection": selection_audit,
            "embedded_rows": len(records),
            "embedded_papers": len(refnos),
            "embedded_canonical_targets": len(target_position),
        },
        "knowledge": knowledge_audit,
        "model": {
            "upstream_name": SCINCL_UPSTREAM,
            "upstream_revision": SCINCL_REVISION,
            "local_path": str(model_path),
            "files": model_hashes,
            "device": args.device,
            "sentence_transformers_version": sentence_transformers.__version__,
            "torch_version": torch.__version__,
            "embedding_dimension": int(abstract_embeddings.shape[1]),
            "maximum_sequence_length": int(model.max_seq_length),
        },
        "pooling": {
            "method": "mean, elementwise maximum, expert-query top-1, expert-query top-3 mean",
            "expert_query_sha256": hashlib.sha256(R3_EXPERT_QUERY.encode("utf-8")).hexdigest(),
            "paper_chunks": paper_chunk_count,
            "local_target_chunks": local_chunk_count,
            "paper_maximum_chunks": 12,
            "local_target_maximum_chunks": 6,
            "local_shortlisting": "frozen-taxonomy distinct/count ranking; uniform fallback when no hit",
        },
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "arrays": {name: list(matrix.shape) for name, matrix in arrays.items()},
        },
    }
    base.atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
