#!/usr/bin/env python3
"""Build label-free text components for supervised SciNCL R3 experiments."""

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

import build_r3_scientific_embeddings as scientific  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_IDENTITY = ROOT / "tmp/r3_small_screen_v1/identity.xlsx"
DEFAULT_OUTPUT = ROOT / "tmp/r3_scientific_finetune/scincl_small_screen_inputs.npz"
DEFAULT_MANIFEST = (
    ROOT / "tmp/r3_scientific_finetune/scincl_small_screen_inputs.manifest.json"
)
SPACE_RE = re.compile(r"\s+")


def compact_abstract(text: str) -> str:
    """Keep title context and abstract content within a compact token budget."""

    front = scientific.PAGE_MARKER_RE.sub("", text[:20_000]).strip()
    abstract_match = scientific.ABSTRACT_HEADING_RE.search(front)
    if abstract_match:
        start = abstract_match.end()
        introduction = scientific.INTRODUCTION_HEADING_RE.search(front, start)
        end = introduction.start() if introduction else min(len(front), start + 5_000)
        title = front[: abstract_match.start()].strip()[-800:]
        abstract = front[start:end].strip()[:3_500]
        value = f"{title} [SEP] {abstract}"
    else:
        introduction = scientific.INTRODUCTION_HEADING_RE.search(front)
        value = front[: introduction.start() if introduction else 4_000]
    compact = SPACE_RE.sub(" ", value).strip()
    return compact or SPACE_RE.sub(" ", front[:4_000]).strip()


def highest_expert_chunk(
    text: str,
    concepts: Sequence[Mapping[str, Any]],
    *,
    target_chars: int = 1_500,
) -> str:
    """Choose one source chunk using only frozen taxonomy match density."""

    candidates = scientific.text_chunks(
        text,
        target_chars=target_chars,
        maximum_chunks=10_000,
    )
    if not candidates:
        return SPACE_RE.sub(" ", text[:target_chars]).strip()
    scored: list[tuple[int, int, int, str]] = []
    for index, chunk in enumerate(candidates):
        counts = base.concept_counts(chunk, concepts)
        scored.append(
            (sum(value > 0 for value in counts), sum(counts), -index, chunk)
        )
    return max(scored, key=lambda item: item[:3])[3]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    identity_path = args.identity.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy_path = args.taxonomy.resolve()
    positive_kb_path = args.positive_kb.resolve() if args.positive_kb else None
    output_path = args.output.resolve()
    manifest_path = args.manifest.resolve()

    # This executable intentionally has no labels/targets argument.
    records, coverage = base.load_pdf_only_identity(identity_path, pdf_dir)
    refnos = {record["REFNO"] for record in records}
    paper_text = base.load_page_text(page_cache_path, refnos)
    if positive_kb_path is not None and not positive_kb_path.is_file():
        raise ValueError(f"positive expert KB does not exist: {positive_kb_path}")
    concepts, knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )
    source_features = base.build_features(records, paper_text, concepts)

    abstract_by_refno = {
        refno: compact_abstract(text) for refno, text in paper_text.items()
    }
    paper_evidence_by_refno = {
        refno: highest_expert_chunk(
            source_features["article_documents"][refno], concepts
        )
        for refno in refnos
    }
    local_evidence: list[str] = []
    local_cache: dict[tuple[str, str], str] = {}
    for index, (record, metadata) in enumerate(
        zip(records, source_features["row_meta"])
    ):
        key = (record["REFNO"], metadata["canonical_target"])
        if key not in local_cache:
            local_cache[key] = highest_expert_chunk(
                source_features["local_documents"][index], concepts
            )
        local_evidence.append(local_cache[key])

    arrays = {
        "target": np.asarray([record["ROCKNAME"] for record in records], dtype=str),
        "abstract": np.asarray(
            [abstract_by_refno[record["REFNO"]] for record in records], dtype=str
        ),
        "local_evidence": np.asarray(local_evidence, dtype=str),
        "paper_evidence": np.asarray(
            [paper_evidence_by_refno[record["REFNO"]] for record in records],
            dtype=str,
        ),
    }
    if any(len(values) != len(records) for values in arrays.values()):
        raise AssertionError("fine-tuning text components lost row alignment")
    if any(any(not str(value).strip() for value in values) for values in arrays.values()):
        raise ValueError("every fine-tuning text component must be nonempty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output_path)
    input_artifacts = {
        "identity": {
            "path": str(identity_path),
            "sha256": scientific.sha256_file(identity_path),
        },
        "page_cache": {
            "path": str(page_cache_path),
            "sha256": scientific.sha256_file(page_cache_path),
        },
        "taxonomy": {
            "path": str(taxonomy_path),
            "sha256": scientific.sha256_file(taxonomy_path),
        },
        "pdf_dir": str(pdf_dir),
        "alignment_sha256": scientific.alignment_sha256(records),
    }
    if positive_kb_path is not None:
        input_artifacts["positive_kb"] = {
            "path": str(positive_kb_path),
            "sha256": scientific.sha256_file(positive_kb_path),
        }
    manifest = {
        "schema_version": (
            "r3_scientific_finetune_inputs_v2"
            if positive_kb_path is not None
            else "r3_scientific_finetune_inputs_v1"
        ),
        "status": "PASS",
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": scientific.sha256_file(Path(__file__).resolve()),
        },
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "cache_contains_record_identifiers": False,
            "target_names_are_model_text": True,
        },
        "inputs": input_artifacts,
        "scope": {
            **coverage,
            "component_rows": len(records),
            "canonical_target_groups": len(local_cache),
        },
        "knowledge": knowledge_audit,
        "retrieval": {
            "paper_evidence": "highest frozen expert-pattern distinct/count chunk",
            "local_evidence": "highest frozen expert-pattern distinct/count target-local chunk",
            "labels_used": False,
        },
        "output": {
            "path": str(output_path),
            "sha256": scientific.sha256_file(output_path),
            "arrays": {name: [len(values)] for name, values in arrays.items()},
        },
    }
    base.atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
