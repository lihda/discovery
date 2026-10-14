#!/usr/bin/env python3
"""Freeze label-blind query-adaptive BGE retrieval features for R3."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
from r3_scientific_evidence_chunks import (  # noqa: E402
    CATEGORIES,
    SCALES,
    EvidenceChunk,
    build_scientific_evidence_chunks,
    compile_category_patterns,
    reciprocal_rank_fusion,
    robust_distractor_statistics,
    select_distractor_chunks,
)


DEFAULT_IDENTITY = trace_freeze.DEFAULT_IDENTITY
DEFAULT_PAGE_CACHE = trace_freeze.DEFAULT_PAGE_CACHE
DEFAULT_PDF_DIR = trace_freeze.DEFAULT_PDF_DIR
DEFAULT_FOLD_MANIFEST = (
    ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1/trace_refno_fold_manifest_v4.json"
)
DEFAULT_CONCEPT_SNAPSHOT = (
    ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1/trace_page_graph_concepts_v4.json"
)
DEFAULT_QUERY_KB = ROOT / "data/knowledge/r3_query_adaptive_retrieval_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_query_adaptive_single872_v1"

FEATURE_NAME = "r3_query_adaptive_features_v1.npz"
MANIFEST_NAME = "r3_query_adaptive_features_v1.manifest.json"
CANDIDATE_NAME = "r3_query_adaptive_candidate_freeze_v1.json"
PREINFERENCE_NAME = "r3_query_adaptive_preinference_freeze_v1.json"
DISTRACTOR_NAME = "r3_query_adaptive_distractor_stats_v1.json"
EVIDENCE_NAME = "r3_query_adaptive_evidence_v1.jsonl"
AUDIT_NAME = "r3_query_adaptive_audit_v1.json"

FEATURE_SCHEMA = "r3_query_adaptive_features_v1"
CANDIDATE_SCHEMA = "r3_query_adaptive_candidate_freeze_v1"
PREINFERENCE_SCHEMA = "r3_query_adaptive_preinference_freeze_v1"
DISTRACTOR_SCHEMA = "r3_query_adaptive_distractor_stats_v1"
AUDIT_SCHEMA = "r3_query_adaptive_audit_v1"
QUERY_SCHEMA = "r3_query_adaptive_retrieval_v1"
MODEL_ID = "BAAI/bge-base-en-v1.5"
MODEL_REVISION = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SUPPORT_CATEGORIES = CATEGORIES
POOL_CATEGORIES = (*SUPPORT_CATEGORIES, "counter")
SECTION_CLASSES = (
    "abstract",
    "introduction",
    "geology",
    "methods",
    "results",
    "discussion",
    "conclusion",
    "other",
)
MODEL_RUNTIME_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "modules.json",
    "sentence_bert_config.json",
    "config_sentence_transformers.json",
    "1_Pooling/config.json",
)


class RetrievalFeatureError(ValueError):
    pass


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def invalid_embedding_rows(value: np.ndarray, tolerance: float = 0.02) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise RetrievalFeatureError("embedding matrix must be two-dimensional")
    finite = np.isfinite(array).all(axis=1)
    invalid = ~finite
    if finite.any():
        selected = array[finite].astype(np.float64, copy=False)
        norms = np.sqrt(np.sum(selected * selected, axis=1))
        amplitudes = np.max(np.abs(selected), axis=1)
        invalid[np.flatnonzero(finite)] = (
            (norms < 1.0 - tolerance)
            | (norms > 1.0 + tolerance)
            | (amplitudes > 1.0 + tolerance)
        )
    return invalid


def cosine_similarity_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float32)
    right_array = np.asarray(right, dtype=np.float32)
    if (
        left_array.ndim != 2
        or right_array.ndim != 2
        or left_array.shape[1] != right_array.shape[1]
    ):
        raise RetrievalFeatureError("cosine embedding matrices do not align")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        values = np.matmul(left_array, right_array.T)
    if not np.isfinite(values).all():
        raise RetrievalFeatureError("cosine similarity produced non-finite values")
    if values.size and (float(values.min()) < -1.02 or float(values.max()) > 1.02):
        raise RetrievalFeatureError("cosine similarity escaped its numeric range")
    return values.astype(np.float32, copy=False)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for value in values:
                handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def acquire_output_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".r3_query_adaptive_builder.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RetrievalFeatureError(
            f"another query-adaptive builder owns the output directory: {output_dir}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RetrievalFeatureError(f"{label} must be a JSON object")
    return value


def default_model_snapshot(revision: str = MODEL_REVISION) -> Path:
    return (
        Path.home()
        / ".cache/huggingface/hub"
        / "models--BAAI--bge-base-en-v1.5"
        / "snapshots"
        / revision
    )


def inspect_model_snapshot(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if path.name != MODEL_REVISION:
        raise RetrievalFeatureError(
            f"BGE snapshot revision changed: expected {MODEL_REVISION}, got {path.name}"
        )
    files = []
    for name in MODEL_RUNTIME_FILES:
        current = path / name
        if not current.is_file():
            raise RetrievalFeatureError(f"BGE runtime file is missing: {current}")
        files.append(
            {
                "name": name,
                "size_bytes": current.stat().st_size,
                "sha256": trace_freeze.file_sha256(current),
            }
        )
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "path": str(path),
        "files": files,
        "snapshot_sha256": json_sha256(files),
    }


def load_query_kb(path: Path) -> dict[str, Any]:
    payload = load_json(path, "query KB")
    if (
        payload.get("schema_version") != QUERY_SCHEMA
        or payload.get("status") != "FROZEN_LABEL_FREE_KNOWLEDGE"
    ):
        raise RetrievalFeatureError("query KB has an invalid schema or status")
    boundary = payload.get("runtime_boundary") or {}
    if any(
        boundary.get(name) is not False
        for name in (
            "labels_read",
            "pmagresult_or_rockmag_read",
            "prior_predictions_read",
            "evaluation_results_read",
        )
    ):
        raise RetrievalFeatureError("query KB violates the label-free boundary")
    if payload.get("retrieval_instruction") != QUERY_PREFIX:
        raise RetrievalFeatureError("BGE query instruction changed")
    encoder = payload.get("encoder") or {}
    if (
        encoder.get("model_id") != MODEL_ID
        or encoder.get("revision") != MODEL_REVISION
        or encoder.get("pooling") != "last_hidden_state_cls"
        or encoder.get("normalize_embeddings") is not True
    ):
        raise RetrievalFeatureError("query KB encoder contract changed")
    queries = payload.get("queries") or []
    query_ids = [clean(item.get("query_id")) for item in queries if isinstance(item, Mapping)]
    if len(queries) != len(query_ids) or len(query_ids) != len(set(query_ids)):
        raise RetrievalFeatureError("query KB contains invalid query identifiers")
    if not queries or any(
        clean(item.get("category")) not in POOL_CATEGORIES
        or clean(item.get("mode")) not in {"support", "counter"}
        or not clean(item.get("text"))
        for item in queries
    ):
        raise RetrievalFeatureError("query KB contains an invalid query")
    templates = payload.get("target_query_templates") or {}
    if set(templates) != set(SUPPORT_CATEGORIES) or any(
        "{target}" not in clean(templates[category]) for category in SUPPORT_CATEGORIES
    ):
        raise RetrievalFeatureError("target query templates changed")
    return payload


def load_concept_snapshot(path: Path) -> list[dict[str, Any]]:
    payload = load_json(path, "concept snapshot")
    if (
        payload.get("schema_version") != trace_freeze.CONCEPT_SCHEMA_VERSION
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise RetrievalFeatureError("concept snapshot is not frozen V4 knowledge")
    concepts = payload.get("concepts") or []
    if not isinstance(concepts, list) or not concepts:
        raise RetrievalFeatureError("concept snapshot contains no concepts")
    return [dict(item) for item in concepts]


def validate_fold_alignment(
    path: Path, records: Sequence[Mapping[str, str]]
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = load_json(path, "fold manifest")
    if (
        payload.get("schema_version") != trace_freeze.FOLD_SCHEMA_VERSION
        or payload.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or payload.get("labels_read") is not False
        or payload.get("alignment_sha256")
        != trace_freeze._record_alignment_sha256(records)
    ):
        raise RetrievalFeatureError("fold manifest does not align with V4")
    rows = payload.get("rows") or []
    if len(rows) != len(records):
        raise RetrievalFeatureError("fold rows do not align")
    folds = np.zeros(len(records), dtype=np.int8)
    for index, (record, row) in enumerate(zip(records, rows)):
        if (
            int(row.get("row_index", -1)) != index
            or clean(row.get("REFNO")) != record["REFNO"]
            or clean(row.get("RESULTNO")) != record["RESULTNO"]
        ):
            raise RetrievalFeatureError("fold identity changed")
        folds[index] = int(row["fold"])
    if hashlib.sha256(folds.tobytes()).hexdigest() != clean(
        payload.get("fold_id_sha256")
    ):
        raise RetrievalFeatureError("fold-id hash changed")
    return folds, payload


def _section_class(value: str) -> str:
    folded = clean(value).casefold().replace("-", "_").replace(" ", "_")
    if folded == "abstract":
        return "abstract"
    if folded in {"introduction", "background", "summary"}:
        return "introduction"
    if folded in {"geology", "geological_setting"}:
        return "geology"
    if folded in {
        "method",
        "methods",
        "sampling",
        "laboratory_procedure",
        "rock_magnetism",
        "magnetic_mineralogy",
        "demagnetization",
    }:
        return "methods"
    if folded in {"result", "results", "results_and_discussion"}:
        return "results"
    if folded in {"discussion", "interpretation"}:
        return "discussion"
    if folded in {"conclusion", "conclusions"}:
        return "conclusion"
    return "other"


class BGEEncoder:
    def __init__(self, snapshot: Path, config: Mapping[str, Any], device: str) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.batch_size = int(config["batch_size"])
        self.max_length = int(config["max_length"])
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(snapshot), local_files_only=True
        )
        self.model = AutoModel.from_pretrained(
            str(snapshot), local_files_only=True, use_safetensors=True
        ).to(device)
        self.model.eval()
        self.dimension = int(self.model.config.hidden_size)
        self.snapshot = snapshot
        self.cpu_model: Any | None = None
        self.fallback_batches = 0
        self.fallback_rows = 0
        self.fallback_exception_batches = 0

    def _encode_cpu(self, batch: Sequence[str]) -> np.ndarray:
        from transformers import AutoModel

        if self.cpu_model is None:
            self.cpu_model = AutoModel.from_pretrained(
                str(self.snapshot), local_files_only=True, use_safetensors=True
            )
            self.cpu_model.eval()
        tokens = self.tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with self.torch.inference_mode():
            hidden = self.cpu_model(**tokens).last_hidden_state[:, 0]
            hidden = self.torch.nn.functional.normalize(hidden, p=2, dim=1)
        values = hidden.detach().to(self.torch.float32).numpy()
        if invalid_embedding_rows(values).any():
            raise RetrievalFeatureError("BGE CPU fallback produced invalid embeddings")
        return values.astype(np.float32, copy=False)

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        output = []
        values = [QUERY_PREFIX + text if query else text for text in texts]
        with self.torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                batch = values[start : start + self.batch_size]
                tokens = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {name: value.to(self.device) for name, value in tokens.items()}
                try:
                    hidden = self.model(**tokens).last_hidden_state[:, 0]
                    hidden = self.torch.nn.functional.normalize(hidden, p=2, dim=1)
                    current = hidden.detach().cpu().to(self.torch.float32).numpy()
                except RuntimeError:
                    if self.device != "mps":
                        raise
                    current = self._encode_cpu(batch)
                    self.fallback_batches += 1
                    self.fallback_exception_batches += 1
                    self.fallback_rows += len(batch)
                    self.torch.mps.empty_cache()
                invalid = invalid_embedding_rows(current)
                if invalid.any():
                    fallback = self._encode_cpu(batch)
                    current[invalid] = fallback[invalid]
                    self.fallback_batches += 1
                    self.fallback_rows += int(invalid.sum())
                    if self.device == "mps":
                        self.torch.mps.empty_cache()
                if invalid_embedding_rows(current).any():
                    raise RetrievalFeatureError("BGE encoder produced invalid embeddings")
                output.append(current)
        combined = np.concatenate(output, axis=0).astype(np.float32, copy=False)
        if invalid_embedding_rows(combined).any():
            raise RetrievalFeatureError("combined BGE embedding matrix is invalid")
        return combined


def _top_mean(values: np.ndarray, count: int) -> float:
    if not len(values):
        return 0.0
    return float(np.mean(np.sort(values)[-min(count, len(values)) :]))


def _normalized_pool(embeddings: np.ndarray, scores: np.ndarray, top_k: int) -> np.ndarray:
    if not len(embeddings):
        return np.zeros(embeddings.shape[1], dtype=np.float32)
    indices = np.argsort(scores)[-min(top_k, len(scores)) :]
    selected_scores = scores[indices].astype(np.float64)
    weights = np.exp(np.clip((selected_scores - selected_scores.max()) * 20.0, -40, 0))
    weights /= weights.sum()
    pooled = np.sum(embeddings[indices] * weights[:, None], axis=0)
    norm = float(np.linalg.norm(pooled))
    return (pooled / norm if norm > 0 else pooled).astype(np.float32)


def _append(values: list[float], names: list[str], name: str, value: Any) -> None:
    names.append(name)
    values.append(float(value))


def summarize_row(
    *,
    chunks: Sequence[EvidenceChunk],
    embeddings: np.ndarray,
    query_specs: Sequence[Mapping[str, Any]],
    query_embeddings: np.ndarray,
    target_embeddings: Mapping[str, np.ndarray],
    distractor_stats: Mapping[str, Mapping[str, float]],
    chunk_audit: Mapping[str, Any],
    top_k: int,
    rrf_k: int,
) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, Any]]:
    if embeddings.shape != (len(chunks), query_embeddings.shape[1]):
        raise RetrievalFeatureError("chunk embeddings do not align")
    similarity = cosine_similarity_matrix(embeddings, query_embeddings) if len(chunks) else np.empty(
        (0, len(query_specs)), dtype=np.float32
    )
    values: list[float] = []
    names: list[str] = []
    _append(values, names, "retrieval_available", bool(chunks))
    _append(values, names, "document_pages_log", math.log1p(int(chunk_audit.get("pages", 0))))
    _append(
        values,
        names,
        "document_body_characters_log",
        math.log1p(int(chunk_audit.get("body_characters", 0))),
    )
    reference_characters = int(chunk_audit.get("reference_characters_excluded", 0))
    body_characters = int(chunk_audit.get("body_characters", 0))
    _append(values, names, "reference_characters_log", math.log1p(reference_characters))
    _append(
        values,
        names,
        "reference_character_fraction",
        reference_characters / max(1, reference_characters + body_characters),
    )
    _append(values, names, "selected_chunks_log", math.log1p(len(chunks)))
    _append(
        values,
        names,
        "lexical_chunks_log",
        math.log1p(sum(bool(sum(chunk.lexical_counts)) for chunk in chunks)),
    )
    for scale in SCALES:
        count = sum(chunk.scale == scale for chunk in chunks)
        _append(values, names, f"scale_{scale}_count_log", math.log1p(count))
        _append(values, names, f"scale_{scale}_ratio", count / max(1, len(chunks)))

    query_by_category: dict[str, list[int]] = defaultdict(list)
    query_top_receipt: dict[str, Any] = {}
    for query_index, query in enumerate(query_specs):
        query_id = clean(query["query_id"])
        category = clean(query["category"])
        query_by_category[category].append(query_index)
        scores = similarity[:, query_index]
        stats = distractor_stats[f"{query_id}::all"]
        maximum = float(scores.max()) if len(scores) else 0.0
        order = np.argsort(scores)[::-1] if len(scores) else np.empty(0, dtype=int)
        top_index = int(order[0]) if len(order) else None
        gap = maximum - float(scores[order[1]]) if len(order) > 1 else maximum
        prefix = f"query_{query_id}"
        _append(values, names, prefix + "_max", maximum)
        _append(values, names, prefix + "_top3_mean", _top_mean(scores, 3))
        _append(values, names, prefix + "_top5_mean", _top_mean(scores, 5))
        _append(values, names, prefix + "_mean", float(scores.mean()) if len(scores) else 0.0)
        _append(values, names, prefix + "_std", float(scores.std()) if len(scores) else 0.0)
        _append(values, names, prefix + "_top_gap", gap)
        denominator = 1.4826 * float(stats["mad"]) + 1e-6
        _append(values, names, prefix + "_robust_z", (maximum - float(stats["median"])) / denominator)
        _append(values, names, prefix + "_minus_q90", maximum - float(stats["q90"]))
        _append(values, names, prefix + "_minus_q95", maximum - float(stats["q95"]))
        top = chunks[top_index] if top_index is not None else None
        for field in ("own_study", "result_language", "citation_language", "adverse_language"):
            _append(values, names, prefix + "_top_" + field, bool(top and getattr(top, field)))
        lexical_top = (
            int(top.adverse_language) + int(top.citation_language)
            if top and category == "counter"
            else top.lexical(category)
            if top
            else 0
        )
        _append(values, names, prefix + "_top_lexical_log", math.log1p(lexical_top))
        for scale in SCALES:
            _append(values, names, prefix + f"_top_scale_{scale}", bool(top and top.scale == scale))
        section = _section_class(top.section) if top else "other"
        for section_class in SECTION_CLASSES:
            _append(
                values,
                names,
                prefix + f"_top_section_{section_class}",
                section == section_class,
            )
        for scale in SCALES:
            selected = np.asarray(
                [scores[index] for index, chunk in enumerate(chunks) if chunk.scale == scale],
                dtype=np.float32,
            )
            _append(
                values,
                names,
                prefix + f"_scale_{scale}_max",
                float(selected.max()) if len(selected) else 0.0,
            )
            scale_maximum = float(selected.max()) if len(selected) else 0.0
            scale_stats = distractor_stats[f"{query_id}::{scale}"]
            scale_denominator = 1.4826 * float(scale_stats["mad"]) + 1e-6
            _append(
                values,
                names,
                prefix + f"_scale_{scale}_robust_z",
                (scale_maximum - float(scale_stats["median"])) / scale_denominator,
            )
            _append(
                values,
                names,
                prefix + f"_scale_{scale}_minus_q90",
                scale_maximum - float(scale_stats["q90"]),
            )
            _append(
                values,
                names,
                prefix + f"_scale_{scale}_minus_q95",
                scale_maximum - float(scale_stats["q95"]),
            )
        query_top_receipt[query_id] = (
            None
            if top is None
            else {
                "chunk_id": top.chunk_id,
                "page": top.page,
                "scale": top.scale,
                "section": top.section,
                "score": round(maximum, 7),
                "text": top.text[:1200],
                "text_sha256": top.text_sha256,
            }
        )

    category_scores: dict[str, np.ndarray] = {}
    category_generic: dict[str, np.ndarray] = {}
    category_target: dict[str, np.ndarray] = {}
    category_lexical: dict[str, np.ndarray] = {}
    category_rrf: dict[str, np.ndarray] = {}
    for category in POOL_CATEGORIES:
        generic_indices = query_by_category[category]
        generic = (
            similarity[:, generic_indices].max(axis=1)
            if len(chunks) and generic_indices
            else np.zeros(len(chunks), dtype=np.float32)
        )
        target = (
            cosine_similarity_matrix(
                embeddings, np.asarray(target_embeddings[category])[None, :]
            )[:, 0]
            if category in target_embeddings and len(chunks)
            else np.zeros(len(chunks), dtype=np.float32)
        )
        combined = np.maximum(generic, target) if category in target_embeddings else generic
        lexical = np.asarray(
            [
                int(chunk.adverse_language) + int(chunk.citation_language)
                if category == "counter"
                else chunk.lexical(category)
                for chunk in chunks
            ],
            dtype=np.float32,
        )
        category_generic[category] = generic
        category_target[category] = target
        category_scores[category] = combined
        category_lexical[category] = lexical
        category_rrf[category] = np.asarray(
            reciprocal_rank_fusion(combined, lexical, k=rrf_k), dtype=np.float32
        )

    category_top: dict[str, int | None] = {}
    category_receipt: dict[str, Any] = {}
    for category in POOL_CATEGORIES:
        generic = category_generic[category]
        target = category_target[category]
        combined = category_scores[category]
        lexical = category_lexical[category]
        rrf = category_rrf[category]
        top_index = int(np.argmax(combined)) if len(combined) else None
        rrf_index = int(np.argmax(rrf)) if len(rrf) else None
        category_top[category] = top_index
        prefix = f"category_{category}"
        _append(values, names, prefix + "_max", float(combined.max()) if len(combined) else 0.0)
        _append(values, names, prefix + "_top3_mean", _top_mean(combined, 3))
        _append(values, names, prefix + "_top5_mean", _top_mean(combined, 5))
        _append(values, names, prefix + "_mean", float(combined.mean()) if len(combined) else 0.0)
        _append(values, names, prefix + "_std", float(combined.std()) if len(combined) else 0.0)
        _append(
            values,
            names,
            prefix + "_target_gain",
            (float(combined.max()) - float(generic.max())) if len(combined) else 0.0,
        )
        _append(values, names, prefix + "_lexical_total_log", math.log1p(float(lexical.sum())))
        _append(values, names, prefix + "_lexical_max_log", math.log1p(float(lexical.max())) if len(lexical) else 0.0)
        _append(values, names, prefix + "_lexical_chunks_log", math.log1p(int((lexical > 0).sum())))
        _append(values, names, prefix + "_rrf_max", float(rrf.max()) if len(rrf) else 0.0)
        rrf_top = chunks[rrf_index] if rrf_index is not None else None
        _append(values, names, prefix + "_rrf_dense", float(combined[rrf_index]) if rrf_index is not None else 0.0)
        _append(values, names, prefix + "_rrf_lexical_log", math.log1p(float(lexical[rrf_index])) if rrf_index is not None else 0.0)
        for field in ("own_study", "result_language", "citation_language", "adverse_language"):
            _append(values, names, prefix + "_rrf_" + field, bool(rrf_top and getattr(rrf_top, field)))
        if top_index is None:
            foil_margin = 0.0
        else:
            foil = max(
                (
                    float(category_scores[other][top_index])
                    for other in category_scores
                    if other != category
                ),
                default=0.0,
            )
            foil_margin = float(combined[top_index]) - foil
        _append(values, names, prefix + "_foil_margin", foil_margin)
        for scale in SCALES:
            selected = np.asarray(
                [combined[index] for index, chunk in enumerate(chunks) if chunk.scale == scale],
                dtype=np.float32,
            )
            _append(
                values,
                names,
                prefix + f"_scale_{scale}_max",
                float(selected.max()) if len(selected) else 0.0,
            )
        top = chunks[top_index] if top_index is not None else None
        category_receipt[category] = (
            None
            if top is None
            else {
                "chunk_id": top.chunk_id,
                "page": top.page,
                "scale": top.scale,
                "section": top.section,
                "score": round(float(combined[top_index]), 7),
                "generic_score": round(float(generic[top_index]), 7),
                "target_score": round(float(target[top_index]), 7),
                "lexical": int(lexical[top_index]),
                "text": top.text[:1600],
                "text_sha256": top.text_sha256,
            }
        )

    for left_index, left in enumerate(SUPPORT_CATEGORIES):
        for right in SUPPORT_CATEGORIES[left_index + 1 :]:
            left_scores = category_scores[left]
            right_scores = category_scores[right]
            prefix = f"chain_{left}_{right}"
            _append(
                values,
                names,
                prefix + "_same_chunk_max_min",
                float(np.minimum(left_scores, right_scores).max()) if len(chunks) else 0.0,
            )
            left_top = category_top[left]
            right_top = category_top[right]
            if left_top is None or right_top is None:
                page_distance = 0.0
                same_page = 0.0
                score_min = 0.0
            else:
                difference = abs(chunks[left_top].page - chunks[right_top].page)
                page_distance = 1.0 / (1.0 + difference)
                same_page = float(difference == 0)
                score_min = min(float(left_scores[left_top]), float(right_scores[right_top]))
            _append(values, names, prefix + "_top_page_proximity", page_distance)
            _append(values, names, prefix + "_top_same_page", same_page)
            _append(values, names, prefix + "_top_score_min", score_min)

    if len(chunks):
        support_matrix = np.column_stack(
            [category_scores[category] for category in SUPPORT_CATEGORIES]
        )
        _append(values, names, "chain_all_four_same_chunk", float(support_matrix.min(axis=1).max()))
        _append(
            values,
            names,
            "chain_method_diagnostic_carrier_same_chunk",
            float(support_matrix[:, :3].min(axis=1).max()),
        )
        _append(
            values,
            names,
            "chain_method_carrier_bridge_same_chunk",
            float(support_matrix[:, [0, 2, 3]].min(axis=1).max()),
        )
    else:
        for name in (
            "chain_all_four_same_chunk",
            "chain_method_diagnostic_carrier_same_chunk",
            "chain_method_carrier_bridge_same_chunk",
        ):
            _append(values, names, name, 0.0)
    support_maxima = [
        float(category_scores[category].max()) if len(chunks) else 0.0
        for category in SUPPORT_CATEGORIES
    ]
    _append(values, names, "practice_any_support_max", max(support_maxima, default=0.0))
    _append(values, names, "practice_support_mean", float(np.mean(support_maxima)))
    _append(
        values,
        names,
        "practice_support_minus_counter",
        max(support_maxima, default=0.0)
        - (float(category_scores["counter"].max()) if len(chunks) else 0.0),
    )

    dimension = embeddings.shape[1]
    pooled = []
    if len(embeddings):
        centroid = embeddings.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        pooled.append((centroid / norm if norm > 0 else centroid).astype(np.float32))
    else:
        pooled.append(np.zeros(dimension, dtype=np.float32))
    for category in POOL_CATEGORIES:
        pooled.append(_normalized_pool(embeddings, category_scores[category], top_k))
    receipt = {
        "top_by_query": query_top_receipt,
        "top_by_category": category_receipt,
        "chunk_count": len(chunks),
    }
    return (
        np.asarray(values, dtype=np.float32),
        names,
        np.concatenate(pooled).astype(np.float32),
        receipt,
    )


def build_chunk_inventory(
    records: Sequence[Mapping[str, str]],
    page_rows: Mapping[str, Mapping[str, Any]],
    concepts: Sequence[Mapping[str, Any]],
    chunk_config: Mapping[str, Any],
) -> tuple[
    dict[str, list[EvidenceChunk]], dict[str, dict[str, Any]], dict[str, Any]
]:
    patterns = compile_category_patterns(concepts)
    refnos = sorted({record["REFNO"] for record in records}, key=lambda value: (int(value), value))
    inventory: dict[str, list[EvidenceChunk]] = {}
    audits: dict[str, dict[str, Any]] = {}
    digest_rows = []
    for number, refno in enumerate(refnos, start=1):
        chunks, audit = build_scientific_evidence_chunks(
            refno, page_rows[refno]["pages"], patterns, chunk_config
        )
        inventory[refno] = chunks
        audits[refno] = audit
        digest_rows.append(
            {
                "REFNO": refno,
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
                "audit": audit,
            }
        )
        if number % 250 == 0:
            print(f"retrieval chunks: papers={number}/{len(refnos)}", flush=True)
    return inventory, audits, {
        "papers": len(refnos),
        "chunks": sum(len(chunks) for chunks in inventory.values()),
        "papers_without_chunks": sum(not chunks for chunks in inventory.values()),
        "inventory_sha256": json_sha256(digest_rows),
        "scale_counts": {
            scale: sum(chunk.scale == scale for chunks in inventory.values() for chunk in chunks)
            for scale in SCALES
        },
    }


def build_distractor_statistics(
    *,
    bank_refnos: Sequence[str],
    inventory: Mapping[str, Sequence[EvidenceChunk]],
    encoder: BGEEncoder,
    query_specs: Sequence[Mapping[str, Any]],
    query_embeddings: np.ndarray,
    chunks_per_document: int,
) -> dict[str, Any]:
    selected_by_refno = {
        refno: select_distractor_chunks(inventory[refno], chunks_per_document)
        for refno in bank_refnos
        if inventory.get(refno)
    }
    flat = [chunk for refno in bank_refnos for chunk in selected_by_refno.get(refno, ())]
    print(
        f"retrieval distractor encoding: papers={len(selected_by_refno)}, chunks={len(flat)}",
        flush=True,
    )
    embeddings = encoder.encode([chunk.text for chunk in flat])
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    for refno in bank_refnos:
        count = len(selected_by_refno.get(refno, ()))
        offsets[refno] = (cursor, cursor + count)
        cursor += count
    values: dict[str, list[float]] = defaultdict(list)
    for refno in bank_refnos:
        start, end = offsets[refno]
        chunks = selected_by_refno.get(refno, ())
        if start == end:
            continue
        scores = cosine_similarity_matrix(embeddings[start:end], query_embeddings)
        for query_index, query in enumerate(query_specs):
            query_id = clean(query["query_id"])
            values[f"{query_id}::all"].append(float(scores[:, query_index].max()))
            for scale in SCALES:
                current = [
                    float(scores[index, query_index])
                    for index, chunk in enumerate(chunks)
                    if chunk.scale == scale
                ]
                if current:
                    values[f"{query_id}::{scale}"].append(max(current))
    statistics = {
        key: robust_distractor_statistics(current) for key, current in sorted(values.items())
    }
    required = {
        f"{clean(query['query_id'])}::{scale}"
        for query in query_specs
        for scale in ("all", *SCALES)
    }
    if required - set(statistics):
        raise RetrievalFeatureError(
            f"distractor bank lacks query-scale statistics: {sorted(required - set(statistics))}"
        )
    return {
        "schema_version": DISTRACTOR_SCHEMA,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "bank_definition": "PDF-backed REFNO values with multiplicity greater than one in the complete three-column identity manifest",
        "bank_papers": len(selected_by_refno),
        "sampled_chunks": len(flat),
        "chunks_per_document": chunks_per_document,
        "statistics": statistics,
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
        },
    }


def _encode_target_queries(
    records: Sequence[Mapping[str, str]],
    templates: Mapping[str, str],
    encoder: BGEEncoder,
    enabled_mask: np.ndarray,
) -> dict[tuple[int, str], np.ndarray]:
    if enabled_mask.shape != (len(records),):
        raise RetrievalFeatureError("target-query enablement mask does not align")
    unique: dict[str, int] = {}
    requests: list[tuple[int, str, str]] = []
    texts: list[str] = []
    for row_index, record in enumerate(records):
        if not bool(enabled_mask[row_index]):
            continue
        target = clean(record["ROCKNAME"]).replace("{", "(").replace("}", ")")
        for category in SUPPORT_CATEGORIES:
            text = clean(templates[category]).format(target=target)
            if text not in unique:
                unique[text] = len(texts)
                texts.append(text)
            requests.append((row_index, category, text))
    print(f"retrieval target-query encoding: unique_queries={len(texts)}", flush=True)
    matrix = encoder.encode(texts, query=True)
    return {
        (row_index, category): matrix[unique[text]]
        for row_index, category, text in requests
    }


def build_features(
    *,
    records: Sequence[Mapping[str, str]],
    inventory: Mapping[str, Sequence[EvidenceChunk]],
    chunk_audits: Mapping[str, Mapping[str, Any]],
    query_kb: Mapping[str, Any],
    distractor: Mapping[str, Any],
    encoder: BGEEncoder,
    target_enabled_mask: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray, list[dict[str, Any]]]:
    query_specs = [dict(item) for item in query_kb["queries"]]
    query_embeddings = encoder.encode(
        [clean(item["text"]) for item in query_specs], query=True
    )
    target_queries = _encode_target_queries(
        records, query_kb["target_query_templates"], encoder, target_enabled_mask
    )
    row_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        row_indices[record["REFNO"]].append(index)
    compact_rows: list[np.ndarray | None] = [None] * len(records)
    pooled_rows: list[np.ndarray | None] = [None] * len(records)
    receipts: list[dict[str, Any] | None] = [None] * len(records)
    names: list[str] | None = None
    refnos = sorted(row_indices, key=lambda value: (int(value), value))
    pending: list[str] = []
    pending_chunks = 0
    processed = 0

    def flush() -> None:
        nonlocal pending, pending_chunks, processed, names
        if not pending:
            return
        flat = [chunk for refno in pending for chunk in inventory[refno]]
        embeddings = encoder.encode([chunk.text for chunk in flat])
        cursor = 0
        for refno in pending:
            chunks = inventory[refno]
            current = embeddings[cursor : cursor + len(chunks)]
            cursor += len(chunks)
            for row_index in row_indices[refno]:
                target = (
                    {
                        category: target_queries[(row_index, category)]
                        for category in SUPPORT_CATEGORIES
                    }
                    if bool(target_enabled_mask[row_index])
                    else {}
                )
                compact, current_names, pooled, receipt = summarize_row(
                    chunks=chunks,
                    embeddings=current,
                    query_specs=query_specs,
                    query_embeddings=query_embeddings,
                    target_embeddings=target,
                    distractor_stats=distractor["statistics"],
                    chunk_audit=chunk_audits[refno],
                    top_k=int(query_kb["ranking"]["pooled_top_k"]),
                    rrf_k=int(query_kb["ranking"]["rrf_k"]),
                )
                if names is None:
                    names = current_names
                elif names != current_names:
                    raise AssertionError("query-adaptive feature order changed")
                compact_rows[row_index] = compact
                pooled_rows[row_index] = pooled
                receipts[row_index] = {
                    "row_index": row_index,
                    "REFNO": records[row_index]["REFNO"],
                    "RESULTNO": records[row_index]["RESULTNO"],
                    "ROCKNAME": records[row_index]["ROCKNAME"],
                    **receipt,
                }
            processed += 1
            if processed % 100 == 0:
                print(f"retrieval feature encoding: papers={processed}/{len(refnos)}", flush=True)
        pending = []
        pending_chunks = 0

    for refno in refnos:
        count = len(inventory[refno])
        if pending and pending_chunks + count > 720:
            flush()
        pending.append(refno)
        pending_chunks += count
    flush()
    if any(value is None for value in compact_rows + pooled_rows + receipts):
        raise AssertionError("query-adaptive feature rows are incomplete")
    compact_matrix = np.vstack(compact_rows).astype(np.float32, copy=False)  # type: ignore[arg-type]
    pooled_matrix = np.vstack(pooled_rows).astype(np.float32, copy=False)  # type: ignore[arg-type]
    if not np.isfinite(compact_matrix).all() or not np.isfinite(pooled_matrix).all():
        raise AssertionError("query-adaptive features contain non-finite values")
    return compact_matrix, names or [], pooled_matrix, list(receipts)  # type: ignore[arg-type]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument("--concept-snapshot", type=Path, default=DEFAULT_CONCEPT_SNAPSHOT)
    parser.add_argument("--query-kb", type=Path, default=DEFAULT_QUERY_KB)
    parser.add_argument("--model-snapshot", type=Path, default=default_model_snapshot())
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    args = parser.parse_args(argv)

    identity_path = args.identity.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    fold_path = args.fold_manifest.resolve()
    concept_path = args.concept_snapshot.resolve()
    query_path = args.query_kb.resolve()
    model_path = args.model_snapshot.resolve()
    output_dir = args.output_dir.resolve()
    output_lock = acquire_output_lock(output_dir)

    all_records = trace_freeze.load_three_column_identity(identity_path)
    records, coverage, pdf_membership_sha256 = trace_freeze.select_pdf_backed_records(
        all_records, pdf_dir
    )
    fold_ids, fold_manifest = validate_fold_alignment(fold_path, records)
    wanted_refnos = {record["REFNO"] for record in records}
    page_rows = trace_freeze.load_sanitized_page_cache(page_cache_path, wanted_refnos)
    query_kb = load_query_kb(query_path)
    concepts = load_concept_snapshot(concept_path)
    model_snapshot = inspect_model_snapshot(model_path)
    multiplicity = Counter(record["REFNO"] for record in all_records)
    single_mask = np.asarray(
        [multiplicity[record["REFNO"]] == 1 for record in records], dtype=bool
    )
    if int(single_mask.sum()) != 872:
        raise RetrievalFeatureError(
            f"single-target scope changed: expected 872, got {int(single_mask.sum())}"
        )
    bank_refnos = sorted(
        {record["REFNO"] for record in records if multiplicity[record["REFNO"]] > 1},
        key=lambda value: (int(value), value),
    )
    if len(bank_refnos) != 606:
        raise RetrievalFeatureError(
            f"multi-target distractor bank changed: expected 606, got {len(bank_refnos)}"
        )
    inventory, chunk_audits, inventory_audit = build_chunk_inventory(
        records, page_rows, concepts, query_kb["chunking"]
    )
    alignment_sha256 = trace_freeze._record_alignment_sha256(records)
    code_paths = [
        Path(__file__).resolve(),
        (SCRIPTS / "r3_scientific_evidence_chunks.py").resolve(),
        (SCRIPTS / "build_r3_trace_page_graph_features.py").resolve(),
        (SCRIPTS / "paper_text_retrieval.py").resolve(),
        (SCRIPTS / "experiment_r3_own_study_events.py").resolve(),
    ]
    requirements_path = (ROOT / "requirements-r3-retrieval-model.txt").resolve()
    import torch
    import transformers

    runtime_dependencies = {
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    candidate_path = output_dir / CANDIDATE_NAME
    preinference_path = output_dir / PREINFERENCE_NAME
    candidate = {
        "schema_version": CANDIDATE_SCHEMA,
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "primary_scope": "all 872 PDF-backed single-target papers at full coverage",
        "alignment_sha256": alignment_sha256,
        "single_target_rows": int(single_mask.sum()),
        "single_mask_sha256": array_sha256(single_mask),
        "outer_fold_ids_sha256": array_sha256(fold_ids),
        "evaluation_candidates": {
            "training_scopes": ["all_pdf_training", "single_target_training"],
            "feature_families": [
                "v4",
                "retrieval_compact",
                "v4_plus_retrieval_compact",
                "retrieval_pooled",
                "retrieval_compact_plus_pooled",
                "v4_plus_retrieval_compact_plus_pooled",
            ],
            "random_forest": {
                "n_estimators": 600,
                "max_features": 0.5,
                "min_samples_leaf": 8,
                "n_jobs": -1,
                "random_state": 35720260,
            },
            "extra_trees": {
                "n_estimators": 600,
                "max_features": 0.5,
                "min_samples_leaf": 6,
                "n_jobs": -1,
                "random_state": 35720260,
            },
            "hist_gradient_boosting": {
                "max_iter": 180,
                "learning_rate": 0.04,
                "max_leaf_nodes": 7,
                "min_samples_leaf": 16,
                "l2_regularization": 2.0,
                "random_state": 35720260,
            },
            "logistic_c_values": [0.001, 0.01, 0.1, 1.0],
            "pca_dimensions": [32, 64, 128],
            "fusion_alphas": [0.2, 0.35, 0.5, 0.65, 0.8],
            "inner_folds": 3,
            "inner_seed": 35720260761,
            "threshold": 0.5,
        },
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
            "combined_supervised_features_read": False,
            "resultno_used_only_for_alignment": True,
            "refno_used_for_grouping_and_multiplicity": True,
        },
    }
    atomic_json(candidate_path, candidate)
    preinference = {
        "schema_version": PREINFERENCE_SCHEMA,
        "status": "FROZEN_BEFORE_MODEL_INFERENCE_AND_LABEL_ACCESS",
        "alignment_sha256": alignment_sha256,
        "coverage": coverage,
        "single_target_rows": int(single_mask.sum()),
        "distractor_bank_papers": len(bank_refnos),
        "chunk_inventory": inventory_audit,
        "model_snapshot": model_snapshot,
        "inputs": {
            "identity": {"path": str(identity_path), "sha256": trace_freeze.file_sha256(identity_path)},
            "page_cache": {"path": str(page_cache_path), "sha256": trace_freeze.file_sha256(page_cache_path)},
            "pdf_membership": {"path": str(pdf_dir), "sha256": pdf_membership_sha256},
            "fold_manifest": {"path": str(fold_path), "sha256": trace_freeze.file_sha256(fold_path)},
            "concept_snapshot": {"path": str(concept_path), "sha256": trace_freeze.file_sha256(concept_path)},
            "query_kb": {"path": str(query_path), "sha256": trace_freeze.file_sha256(query_path)},
            "candidate_freeze": {"path": str(candidate_path), "sha256": trace_freeze.file_sha256(candidate_path)},
            "code": [
                {"path": str(path), "sha256": trace_freeze.file_sha256(path)} for path in code_paths
            ],
            "requirements": {
                "path": str(requirements_path),
                "sha256": trace_freeze.file_sha256(requirements_path),
            },
        },
        "runtime_dependencies": runtime_dependencies,
        "runtime_boundary": candidate["runtime_boundary"],
    }
    atomic_json(preinference_path, preinference)

    encoder = BGEEncoder(model_path, query_kb["encoder"], args.device)
    if encoder.dimension != 768:
        raise RetrievalFeatureError(f"BGE embedding dimension changed: {encoder.dimension}")
    query_specs = [dict(item) for item in query_kb["queries"]]
    global_query_embeddings = encoder.encode(
        [clean(item["text"]) for item in query_specs], query=True
    )
    distractor = build_distractor_statistics(
        bank_refnos=bank_refnos,
        inventory=inventory,
        encoder=encoder,
        query_specs=query_specs,
        query_embeddings=global_query_embeddings,
        chunks_per_document=int(query_kb["chunking"]["distractor_chunks_per_document"]),
    )
    distractor["inputs"] = {
        "preinference_freeze_sha256": trace_freeze.file_sha256(preinference_path),
        "model_snapshot_sha256": model_snapshot["snapshot_sha256"],
        "query_kb_sha256": trace_freeze.file_sha256(query_path),
        "chunk_inventory_sha256": inventory_audit["inventory_sha256"],
    }
    distractor_path = output_dir / DISTRACTOR_NAME
    atomic_json(distractor_path, distractor)
    compact, feature_names, pooled, receipts = build_features(
        records=records,
        inventory=inventory,
        chunk_audits=chunk_audits,
        query_kb=query_kb,
        distractor=distractor,
        encoder=encoder,
        target_enabled_mask=single_mask,
    )
    feature_path = output_dir / FEATURE_NAME
    atomic_npz(
        feature_path,
        compact=compact,
        pooled=pooled,
        single_mask=single_mask,
        fold_ids=fold_ids,
    )
    evidence_path = output_dir / EVIDENCE_NAME
    atomic_jsonl(evidence_path, receipts)
    audit = {
        "schema_version": AUDIT_SCHEMA,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "rows": len(records),
        "papers": len(inventory),
        "single_target_rows": int(single_mask.sum()),
        "multi_target_rows": int((~single_mask).sum()),
        "compact_shape": list(compact.shape),
        "pooled_shape": list(pooled.shape),
        "embedding_dimension": encoder.dimension,
        "mps_cpu_fallback_batches": encoder.fallback_batches,
        "mps_cpu_fallback_rows": encoder.fallback_rows,
        "mps_cpu_fallback_exception_batches": encoder.fallback_exception_batches,
        "pool_groups": ["document", *POOL_CATEGORIES],
        "chunk_inventory": inventory_audit,
        "chunk_audit_totals": {
            "body_characters": sum(int(item["body_characters"]) for item in chunk_audits.values()),
            "reference_characters_excluded": sum(
                int(item["reference_characters_excluded"]) for item in chunk_audits.values()
            ),
            "lexical_chunks": sum(int(item["lexical_chunks"]) for item in chunk_audits.values()),
        },
        "runtime_boundary": candidate["runtime_boundary"],
    }
    audit_path = output_dir / AUDIT_NAME
    atomic_json(audit_path, audit)
    manifest = {
        "schema_version": FEATURE_SCHEMA,
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "alignment_sha256": alignment_sha256,
        "rows": len(records),
        "single_target_rows": int(single_mask.sum()),
        "feature_names": feature_names,
        "pool_groups": ["document", *POOL_CATEGORIES],
        "embedding_dimension": encoder.dimension,
        "matrix_hashes": {
            "compact": array_sha256(compact),
            "pooled": array_sha256(pooled),
            "single_mask": array_sha256(single_mask),
            "fold_ids": array_sha256(fold_ids),
        },
        "inputs": preinference["inputs"],
        "model_snapshot": model_snapshot,
        "outputs": {
            "features": {"path": str(feature_path), "sha256": trace_freeze.file_sha256(feature_path)},
            "candidate_freeze": {"path": str(candidate_path), "sha256": trace_freeze.file_sha256(candidate_path)},
            "preinference_freeze": {"path": str(preinference_path), "sha256": trace_freeze.file_sha256(preinference_path)},
            "distractor_stats": {"path": str(distractor_path), "sha256": trace_freeze.file_sha256(distractor_path)},
            "evidence": {"path": str(evidence_path), "sha256": trace_freeze.file_sha256(evidence_path)},
            "audit": {"path": str(audit_path), "sha256": trace_freeze.file_sha256(audit_path)},
        },
        "runtime": {
            "python": sys.version.split()[0],
            **runtime_dependencies,
            "device": args.device,
            "mps_cpu_fallback_batches": encoder.fallback_batches,
            "mps_cpu_fallback_rows": encoder.fallback_rows,
            "mps_cpu_fallback_exception_batches": encoder.fallback_exception_batches,
        },
        "runtime_boundary": candidate["runtime_boundary"],
    }
    manifest_path = output_dir / MANIFEST_NAME
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "rows": len(records),
                "single_target_rows": int(single_mask.sum()),
                "compact_shape": list(compact.shape),
                "pooled_shape": list(pooled.shape),
                "chunks": inventory_audit["chunks"],
            },
            indent=2,
        ),
        flush=True,
    )
    output_lock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BGEEncoder",
    "RetrievalFeatureError",
    "acquire_output_lock",
    "build_chunk_inventory",
    "build_distractor_statistics",
    "build_features",
    "inspect_model_snapshot",
    "cosine_similarity_matrix",
    "invalid_embedding_rows",
    "load_query_kb",
    "summarize_row",
    "validate_fold_alignment",
]
