#!/usr/bin/env python3
"""Build label-free, target-bound DeBERTa NLI embeddings for R3.

The numerical cache is row-aligned but deliberately contains no labels or
record identifiers. Candidate passages come only from original-paper text and
frozen R3 knowledge. PMAGRESULT/ROCKMAG are never opened at runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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
from paper_text_retrieval import canonical_target_identity  # noqa: E402


DEFAULT_MODEL = (
    Path.home()
    / ".cache/huggingface/hub/models--cross-encoder--nli-deberta-v3-base"
    / "snapshots/6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
)
DEFAULT_REUSED_PASSAGES = ROOT / "tmp/r3_nli_evidence/retrieved_passages.jsonl"
DEFAULT_POSITIVE_KB = ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json"
DEFAULT_IDENTITY = ROOT / "tmp/r3_exact_300_resultno_v2/identity.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/deberta_target_nli_v2"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "features.npz"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "features.manifest.json"
MODEL_REVISION = "6c749ce3425cd33b46d187e45b92bbf96ee12ec7"
FROZEN_IDENTITY_SHA256 = (
    "93ea5dfba0a32501a564d06eb020386fd86b11f59ded0982043f90c064ab960b"
)
FROZEN_ALIGNMENT_SHA256 = (
    "16fba124ba12920ed31aed22e93a4b1ace5db6ca35b7a63724a6e933e73f0a48"
)
FROZEN_ROWS = 300
FROZEN_PAPERS = 258
FROZEN_MODEL_SHA256 = {
    "config.json": "897e756eb59d3183adb505952e7910e7cbc7750a43f3b3747a96b688d2b02a47",
    "model.safetensors": "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
    "tokenizer.json": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    "spm.model": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
}
SPACE_RE = re.compile(r"\s+")
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]{2,}")
NONWORD_RE = re.compile(r"[^a-z0-9]+")
OWN_STUDY_RE = re.compile(
    r"\b(?:we|our|this\s+study|this\s+work|samples?\s+(?:were|was)|"
    r"specimens?\s+(?:were|was)|we\s+(?:measured|performed|used|applied|examined)|"
    r"(?:measurements?|analys(?:is|es)|experiments?|tests?)\s+(?:were|was)\s+"
    r"(?:made|performed|carried\s+out|conducted))\b",
    re.I,
)
RESULT_CUE_RE = re.compile(
    r"\b(?:indicat(?:e|es|ed|ing)|reveal(?:s|ed)?|show(?:s|ed)?|suggest(?:s|ed)?|"
    r"consistent\s+with|dominated\s+by|identified\s+as|"
    r"carrier(?:s)?\s+(?:is|are|was|were))\b",
    re.I,
)
CITATION_HEAVY_RE = re.compile(
    r"(?:\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)){2,}"
)

TARGET_STOPWORDS = {
    "and",
    "basalt",
    "basalts",
    "bed",
    "beds",
    "complex",
    "dyke",
    "dykes",
    "formation",
    "formations",
    "group",
    "granite",
    "intrusion",
    "lava",
    "lavas",
    "member",
    "province",
    "rock",
    "rocks",
    "sediment",
    "sediments",
    "series",
    "site",
    "sites",
    "suite",
    "unit",
    "volcanics",
}

HYPOTHESIS_NAMES = (
    "paper_criterion",
    "target_criterion",
    "target_method",
    "target_carrier",
    "concept_method",
)
HYPOTHESIS_TEMPLATES = {
    "paper_criterion": (
        "This study uses rock-magnetic, mineralogical, or petrographic evidence "
        "to characterize magnetic minerals or remanence carriers."
    ),
    "target_criterion": (
        "For {target}, this study uses rock-magnetic, mineralogical, or "
        "petrographic evidence to characterize magnetic minerals or remanence carriers."
    ),
    "target_method": (
        "The authors performed rock-magnetic, mineralogical, or petrographic "
        "measurements on {target}."
    ),
    "target_carrier": (
        "The authors identify or interpret the magnetic mineral or remanence "
        "carrier of {target}."
    ),
}

CONCEPT_METHOD_LABELS = {
    "irm_acquisition": "isothermal remanent magnetization measurements",
    "thermomagnetic_curve": "thermomagnetic measurements",
    "temperature_dependent_susceptibility": (
        "temperature-dependent magnetic susceptibility measurements"
    ),
    "three_axis_irm": "three-axis IRM measurements",
    "low_temperature_irm": "low-temperature IRM measurements",
    "remanence_temperature_experiment": "remanence-versus-temperature measurements",
    "hysteresis_forc": "magnetic hysteresis or FORC measurements",
    "coercivity_backfield": "magnetic coercivity or backfield measurements",
    "unblocking_spectrum": "magnetic unblocking-temperature measurements",
    "magnetic_fabric": "magnetic fabric measurements",
    "petrographic_microscopy": "microscopic examination",
    "mineralogical_phase_analysis": "mineralogical phase measurements",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alignment_sha256(records: Sequence[Mapping[str, str]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(
            "\0".join(
                (record["REFNO"], record["RESULTNO"], record["ROCKNAME"])
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def normalize_phrase(value: str) -> str:
    return NONWORD_RE.sub(" ", value.casefold()).strip()


def informative_target_tokens(rockname: str) -> tuple[str, ...]:
    tokens = {
        token.casefold()
        for token in TOKEN_RE.findall(rockname)
        if token.casefold() not in TARGET_STOPWORDS
    }
    return tuple(sorted(tokens, key=lambda token: (-len(token), token)))


def target_match_features(text: str, rockname: str) -> tuple[int, int, float]:
    normalized = normalize_phrase(text)
    target = normalize_phrase(rockname)
    exact = int(len(target) >= 6 and target in normalized)
    tokens = informative_target_tokens(rockname)
    matches = sum(bool(re.search(rf"\b{re.escape(token)}\b", normalized)) for token in tokens)
    coverage = matches / max(1, len(tokens))
    return exact, matches, coverage


def passage_rank_key(row: Mapping[str, Any], rockname: str) -> tuple[float, ...]:
    exact, token_hits, coverage = target_match_features(str(row["text"]), rockname)
    strong_chain = int(
        int(row.get("method_hits", 0)) > 0
        and (
            int(row.get("diagnostic_hits", 0)) > 0
            or int(row.get("carrier_hits", 0)) > 0
        )
    )
    score = (
        float(row.get("retrieval_score", 0))
        + 7.0 * exact
        + 3.0 * min(token_hits, 3)
        + 4.0 * strong_chain
        + 1.5 * min(int(row.get("own_study_hits", 0)), 2)
        + min(int(row.get("result_hits", 0)), 2)
        - 1.5 * int(bool(row.get("citation_heavy", False)))
    )
    return (
        score,
        exact,
        coverage,
        strong_chain,
        float(row.get("retrieval_score", 0)),
        -float(row.get("start", 0)),
    )


def select_target_passages(
    passages: Sequence[Mapping[str, Any]],
    rockname: str,
    maximum: int,
) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("maximum passages must be positive")
    ranked = sorted(
        (dict(row) for row in passages if str(row.get("text", "")).strip()),
        key=lambda row: passage_rank_key(row, rockname),
        reverse=True,
    )
    if not ranked:
        raise ValueError("each paper must supply at least one nonempty passage")

    selected: list[dict[str, Any]] = []
    expert_top = max(
        ranked,
        key=lambda row: (
            float(row.get("retrieval_score", 0)),
            int(row.get("method_hits", 0)),
            int(row.get("diagnostic_hits", 0)),
            int(row.get("carrier_hits", 0)),
        ),
    )
    selected.append(expert_top)
    target_candidates = [
        row for row in ranked if target_match_features(str(row["text"]), rockname)[1] > 0
    ]
    if target_candidates and target_candidates[0] is not expert_top:
        selected.append(target_candidates[0])
    for row in ranked:
        if all(row is not old for old in selected):
            selected.append(row)
        if len(selected) >= maximum:
            break
    return selected[:maximum]


def merge_passage_sets(
    reused: Sequence[Mapping[str, Any]],
    fresh: Sequence[Mapping[str, Any]],
    target_local: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Merge duplicate contexts while preserving richer expert metadata."""

    output: list[dict[str, Any]] = []
    by_span: dict[tuple[int, int], int] = {}
    by_text: dict[str, int] = {}
    for source_name, rows in (
        ("reused", reused),
        ("fresh_expert_retrieval", fresh),
        ("target_local", target_local),
    ):
        for raw in rows:
            row = dict(raw)
            key = (int(row.get("start", -1)), int(row.get("end", -1)))
            text_key = normalize_phrase(str(row.get("text", "")))[:800]
            existing_index = by_span.get(key)
            if existing_index is None and text_key:
                existing_index = by_text.get(text_key)
            if existing_index is not None:
                existing = output[existing_index]
                existing["concepts"] = sorted(
                    set(existing.get("concepts", [])) | set(row.get("concepts", []))
                )
                existing["matched_sources"] = sorted(
                    set(existing.get("matched_sources", []))
                    | set(row.get("matched_sources", []))
                )
                existing["retrieval_origins"] = sorted(
                    set(existing.get("retrieval_origins", [])) | {source_name}
                )
                for name in (
                    "method_hits",
                    "diagnostic_hits",
                    "carrier_hits",
                    "own_study_hits",
                    "result_hits",
                    "retrieval_score",
                ):
                    existing[name] = max(
                        float(existing.get(name, 0)), float(row.get(name, 0))
                    )
                existing["citation_heavy"] = bool(
                    existing.get("citation_heavy", False)
                    and row.get("citation_heavy", False)
                )
                continue
            row["retrieval_origins"] = [source_name]
            output.append(row)
            position = len(output) - 1
            by_span[key] = position
            if text_key:
                by_text[text_key] = position
    return output


def enrich_passage(
    raw: Mapping[str, Any],
    concepts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute source-aware concept metadata from the passage text."""

    row = dict(raw)
    text = str(row.get("text", ""))
    concept_ids: set[str] = set()
    matched_sources: set[str] = set()
    category_hits: Counter[str] = Counter()
    for concept in concepts:
        count = 0
        for pattern in concept["patterns"]:
            count += sum(1 for _ in pattern.finditer(text))
            if count >= 5:
                count = 5
                break
        if not count:
            continue
        concept_ids.add(str(concept["concept_id"]))
        matched_sources.add(str(concept["source"]))
        category = str(concept["category"])
        if "method" in category:
            category_hits["method"] += count
        elif "diagnostic" in category:
            category_hits["diagnostic"] += count
        else:
            category_hits["carrier"] += count
    own_study_hits = len(OWN_STUDY_RE.findall(text))
    result_hits = len(RESULT_CUE_RE.findall(text))
    citation_heavy = bool(CITATION_HEAVY_RE.search(text))
    score = (
        5 * int(category_hits["method"] > 0)
        + 3 * int(category_hits["diagnostic"] > 0)
        + 2 * int(category_hits["carrier"] > 0)
        + min(3, len(concept_ids))
        + 2 * min(2, own_study_hits)
        + min(2, result_hits)
        - 2 * int(citation_heavy)
    )
    row.update(
        {
            "concepts": sorted(concept_ids),
            "matched_sources": sorted(matched_sources),
            "method_hits": int(category_hits["method"]),
            "diagnostic_hits": int(category_hits["diagnostic"]),
            "carrier_hits": int(category_hits["carrier"]),
            "own_study_hits": own_study_hits,
            "result_hits": result_hits,
            "citation_heavy": citation_heavy,
            "retrieval_score": score,
        }
    )
    return row


def target_local_passages(
    text: str,
    rockname: str,
    concepts: Sequence[Mapping[str, Any]],
    *,
    maximum: int = 4,
    radius: int = 700,
) -> list[dict[str, Any]]:
    """Extract target-name windows directly from the full original paper."""

    hits = base.target_hit_spans(text, rockname)
    spans = base.merge_spans(
        (
            (max(0, start - radius), min(len(text), end + radius))
            for start, end in hits
        ),
        gap=120,
    )
    if len(spans) > 16:
        positions = np.linspace(0, len(spans) - 1, 16, dtype=int)
        spans = [spans[int(position)] for position in positions]
    rows = [
        enrich_passage(
            {
                "start": start,
                "end": end,
                "text": (
                    f"Target anchor: {rockname}. Context: "
                    + SPACE_RE.sub(" ", text[start:end]).strip()
                )[:2_200],
            },
            concepts,
        )
        for start, end in spans
        if text[start:end].strip()
    ]
    rows.sort(key=lambda row: passage_rank_key(row, rockname), reverse=True)
    return rows[:maximum]


def concept_method_hypothesis(row: Mapping[str, Any], target: str) -> str:
    concepts = {str(value) for value in row.get("concepts", [])}
    label = next(
        (value for concept, value in CONCEPT_METHOD_LABELS.items() if concept in concepts),
        "rock-magnetic or mineralogical measurements",
    )
    return f"The authors performed {label} on {target} in this study."


def hypotheses_for_passage(row: Mapping[str, Any], target: str) -> tuple[str, ...]:
    return (
        HYPOTHESIS_TEMPLATES["paper_criterion"],
        HYPOTHESIS_TEMPLATES["target_criterion"].format(target=target),
        HYPOTHESIS_TEMPLATES["target_method"].format(target=target),
        HYPOTHESIS_TEMPLATES["target_carrier"].format(target=target),
        concept_method_hypothesis(row, target),
    )


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def aggregate_target_pairs(
    mean_embeddings: np.ndarray,
    cls_embeddings: np.ndarray,
    probabilities: np.ndarray,
    owners: Sequence[int],
    hypothesis_indices: Sequence[int],
    target_count: int,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    width = mean_embeddings.shape[1]
    for hypothesis_index, hypothesis_name in enumerate(HYPOTHESIS_NAMES):
        mean_pool = np.zeros((target_count, width), dtype=np.float32)
        cls_pool = np.zeros((target_count, width), dtype=np.float32)
        top_pool = np.zeros((target_count, width), dtype=np.float32)
        for owner in range(target_count):
            indices = np.asarray(
                [
                    index
                    for index, (pair_owner, pair_hypothesis) in enumerate(
                        zip(owners, hypothesis_indices)
                    )
                    if pair_owner == owner and pair_hypothesis == hypothesis_index
                ],
                dtype=np.int64,
            )
            if not len(indices):
                raise ValueError("every target/hypothesis needs at least one pair")
            entailment = probabilities[indices, 1]
            top = indices[int(np.argmax(entailment))]
            weights = np.maximum(entailment, 0.05)
            weights = weights / weights.sum()
            mean_pool[owner] = np.sum(mean_embeddings[indices] * weights[:, None], axis=0)
            cls_pool[owner] = np.sum(cls_embeddings[indices] * weights[:, None], axis=0)
            top_pool[owner] = mean_embeddings[top]
        output[f"{hypothesis_name}__mean"] = l2_normalize(mean_pool)
        output[f"{hypothesis_name}__cls"] = l2_normalize(cls_pool)
        output[f"{hypothesis_name}__top"] = l2_normalize(top_pool)
    return output


def structured_target_features(
    probabilities: np.ndarray,
    owners: Sequence[int],
    hypothesis_indices: Sequence[int],
    target_rows: Sequence[Mapping[str, Any]],
    selected_by_target: Sequence[Sequence[Mapping[str, Any]]],
) -> tuple[np.ndarray, list[str]]:
    names: list[str] = []
    for hypothesis in HYPOTHESIS_NAMES:
        names.extend(
            [
                f"{hypothesis}__entailment_max",
                f"{hypothesis}__entailment_top2_mean",
                f"{hypothesis}__entailment_mean",
                f"{hypothesis}__contradiction_mean",
                f"{hypothesis}__neutral_mean",
            ]
        )
    names.extend(
        [
            "retrieval__passage_count",
            "retrieval__method_hits",
            "retrieval__diagnostic_hits",
            "retrieval__carrier_hits",
            "retrieval__own_study_hits",
            "retrieval__result_hits",
            "retrieval__max_score",
            "retrieval__strong_chain_passages",
            "retrieval__positive_kb_passages",
            "retrieval__target_local_passages",
            "target__exact_passages",
            "target__token_hit_passages",
            "target__maximum_token_coverage",
        ]
    )
    matrix = np.zeros((len(target_rows), len(names)), dtype=np.float32)
    for owner, (target, passages) in enumerate(zip(target_rows, selected_by_target)):
        cursor = 0
        for hypothesis_index in range(len(HYPOTHESIS_NAMES)):
            indices = np.asarray(
                [
                    index
                    for index, (pair_owner, pair_hypothesis) in enumerate(
                        zip(owners, hypothesis_indices)
                    )
                    if pair_owner == owner and pair_hypothesis == hypothesis_index
                ],
                dtype=np.int64,
            )
            entailment = np.sort(probabilities[indices, 1])[::-1]
            matrix[owner, cursor : cursor + 5] = (
                entailment[0],
                entailment[:2].mean(),
                entailment.mean(),
                probabilities[indices, 0].mean(),
                probabilities[indices, 2].mean(),
            )
            cursor += 5
        match_values = [
            target_match_features(str(row["text"]), str(target["ROCKNAME"]))
            for row in passages
        ]
        matrix[owner, cursor:] = np.asarray(
            [
                len(passages),
                sum(int(row.get("method_hits", 0)) for row in passages),
                sum(int(row.get("diagnostic_hits", 0)) for row in passages),
                sum(int(row.get("carrier_hits", 0)) for row in passages),
                sum(int(row.get("own_study_hits", 0)) for row in passages),
                sum(int(row.get("result_hits", 0)) for row in passages),
                max(float(row.get("retrieval_score", 0)) for row in passages),
                sum(
                    int(row.get("method_hits", 0)) > 0
                    and (
                        int(row.get("diagnostic_hits", 0)) > 0
                        or int(row.get("carrier_hits", 0)) > 0
                    )
                    for row in passages
                ),
                sum(
                    "positive_rockmag_expert_kb" in row.get("matched_sources", [])
                    for row in passages
                ),
                sum(
                    "target_local" in row.get("retrieval_origins", [])
                    for row in passages
                ),
                sum(value[0] for value in match_values),
                sum(value[1] > 0 for value in match_values),
                max(value[2] for value in match_values),
            ],
            dtype=np.float32,
        )
    return matrix, names


def encode_pairs(
    model_path: Path,
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    device: str,
    batch_size: int,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    labels = {int(index): str(value).casefold() for index, value in config.id2label.items()}
    label_order = [labels[index] for index in range(len(labels))]
    if label_order != ["contradiction", "entailment", "neutral"]:
        raise ValueError(f"unexpected NLI label order: {label_order}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path), local_files_only=True
    )
    model.eval()
    model.to(device)
    width = int(config.hidden_size)
    mean_output = np.empty((len(premises), width), dtype=np.float32)
    cls_output = np.empty((len(premises), width), dtype=np.float32)
    probability_output = np.empty((len(premises), 3), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(premises), batch_size):
            end = min(len(premises), start + batch_size)
            encoded = tokenizer(
                list(premises[start:end]),
                list(hypotheses[start:end]),
                padding=True,
                truncation="only_first",
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            encoder_result = model.deberta(**encoded, return_dict=True)
            hidden = encoder_result.last_hidden_state
            classification_pool = model.pooler(hidden)
            logits = model.classifier(model.dropout(classification_pool))
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            mean_output[start:end] = pooled.detach().cpu().float().numpy()
            cls_output[start:end] = hidden[:, 0].detach().cpu().float().numpy()
            probability_output[start:end] = (
                torch.softmax(logits, dim=1).detach().cpu().float().numpy()
            )
            if start == 0 or end == len(premises) or end % (batch_size * 25) == 0:
                print(f"DeBERTa pairs {end}/{len(premises)}", flush=True)
    return (
        l2_normalize(mean_output),
        l2_normalize(cls_output),
        probability_output,
        {
            "architecture": config.architectures,
            "hidden_size": width,
            "label_order": label_order,
            "max_position_embeddings": int(config.max_position_embeddings),
        },
    )


def load_reused_passages(path: Path, refnos: set[str]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            refno = str(row.get("REFNO", "")).strip()
            if refno in refnos:
                grouped[refno].append(row)
    missing = sorted(refnos - set(grouped))
    if missing:
        raise ValueError(f"reused passage artifact misses {len(missing)} PDF papers")
    return grouped


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=DEFAULT_POSITIVE_KB)
    parser.add_argument("--reused-passages", type=Path, default=DEFAULT_REUSED_PASSAGES)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--maximum-passages", type=int, default=4)
    parser.add_argument("--fresh-passages", type=int, default=8)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    if args.batch_size < 1 or args.maximum_passages < 1 or args.fresh_passages < 1:
        raise SystemExit("batch and passage limits must be positive")
    if not 64 <= args.max_length <= 512:
        raise SystemExit("--max-length must be between 64 and 512")
    if args.target_limit is not None and args.target_limit < 1:
        raise SystemExit("--target-limit must be positive")

    identity_path = args.identity.resolve()
    if file_sha256(identity_path) != FROZEN_IDENTITY_SHA256:
        raise SystemExit("identity is not the frozen exact-300 RESULTNO manifest")
    pdf_dir = args.pdf_dir.resolve()
    page_cache_path = args.page_cache.resolve()
    taxonomy_path = args.taxonomy.resolve()
    positive_kb_path = args.positive_kb.resolve()
    reused_path = args.reused_passages.resolve()
    model_path = args.model.resolve()
    actual_model_hashes = {
        name: file_sha256(model_path / name) for name in FROZEN_MODEL_SHA256
    }
    if actual_model_hashes != FROZEN_MODEL_SHA256:
        raise SystemExit("--model does not match the frozen DeBERTa NLI snapshot")
    records, coverage = base.load_pdf_only_identity(identity_path, pdf_dir)
    if (
        coverage.get("pdf_only_rows") != FROZEN_ROWS
        or coverage.get("pdf_only_paper_groups") != FROZEN_PAPERS
        or alignment_sha256(records) != FROZEN_ALIGNMENT_SHA256
    ):
        raise SystemExit("PDF membership differs from the frozen exact-300 scope")
    refnos = {record["REFNO"] for record in records}
    paper_text = base.load_page_text(page_cache_path, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )
    reused = load_reused_passages(reused_path, refnos)

    # Imported here so unit tests for pure selection/alignment logic need no torch.
    import build_r3_nli_evidence_features as evidence

    fresh: dict[str, list[dict[str, Any]]] = {}
    retrieval_audit: Counter[str] = Counter()
    for index, refno in enumerate(sorted(refnos), start=1):
        passages, audit = evidence.retrieve_passages(
            paper_text[refno], concepts, maximum=args.fresh_passages
        )
        fresh[refno] = passages
        retrieval_audit.update(audit)
        if index % 100 == 0 or index == len(refnos):
            print(f"positive-KB retrieval {index}/{len(refnos)}", flush=True)

    target_rows: list[dict[str, str]] = []
    target_position: dict[tuple[str, str], int] = {}
    row_target_positions: list[int] = []
    for record in records:
        key = (record["REFNO"], canonical_target_identity(record["ROCKNAME"]))
        if key not in target_position:
            target_position[key] = len(target_rows)
            target_rows.append(record)
        row_target_positions.append(target_position[key])
    if args.target_limit is not None:
        target_rows = target_rows[: args.target_limit]
        allowed = set(range(len(target_rows)))
        retained_rows = [
            index for index, target_index in enumerate(row_target_positions) if target_index in allowed
        ]
        records = [records[index] for index in retained_rows]
        row_target_positions = [row_target_positions[index] for index in retained_rows]

    enriched_reused = {
        refno: [enrich_passage(row, concepts) for row in reused[refno]]
        for refno in refnos
    }
    enriched_fresh = {
        refno: [enrich_passage(row, concepts) for row in fresh[refno]]
        for refno in refnos
    }
    selected_by_target: list[list[dict[str, Any]]] = []
    for target in target_rows:
        local_rows = target_local_passages(
            paper_text[target["REFNO"]], target["ROCKNAME"], concepts
        )
        merged = merge_passage_sets(
            enriched_reused[target["REFNO"]],
            enriched_fresh[target["REFNO"]],
            local_rows,
        )
        selected_by_target.append(
            select_target_passages(merged, target["ROCKNAME"], args.maximum_passages)
        )

    premises: list[str] = []
    pair_hypotheses: list[str] = []
    owners: list[int] = []
    hypothesis_indices: list[int] = []
    for owner, (target, passages) in enumerate(zip(target_rows, selected_by_target)):
        for passage in passages:
            hypotheses = hypotheses_for_passage(passage, target["ROCKNAME"])
            for hypothesis_index, hypothesis in enumerate(hypotheses):
                premises.append(SPACE_RE.sub(" ", str(passage["text"])).strip())
                pair_hypotheses.append(hypothesis)
                owners.append(owner)
                hypothesis_indices.append(hypothesis_index)

    mean_embeddings, cls_embeddings, probabilities, model_audit = encode_pairs(
        model_path,
        premises,
        pair_hypotheses,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    target_arrays = aggregate_target_pairs(
        mean_embeddings,
        cls_embeddings,
        probabilities,
        owners,
        hypothesis_indices,
        len(target_rows),
    )
    structured, structured_names = structured_target_features(
        probabilities,
        owners,
        hypothesis_indices,
        target_rows,
        selected_by_target,
    )
    target_arrays["structured"] = structured
    row_positions = np.asarray(row_target_positions, dtype=np.int64)
    arrays = {
        name: np.asarray(matrix[row_positions], dtype=np.float32)
        for name, matrix in target_arrays.items()
    }
    if any(matrix.ndim != 2 or matrix.shape[0] != len(records) for matrix in arrays.values()):
        raise AssertionError("DeBERTa arrays lost row alignment")
    if any(not np.isfinite(matrix).all() for matrix in arrays.values()):
        raise AssertionError("DeBERTa arrays contain non-finite values")

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(output_path)

    model_files = actual_model_hashes
    origin_counts = Counter(
        origin
        for passages in selected_by_target
        for row in passages
        for origin in row.get("retrieval_origins", ["unknown"])
    )
    knowledge_source_counts = Counter(
        source
        for passages in selected_by_target
        for row in passages
        for source in row.get("matched_sources", [])
    )
    manifest = {
        "schema_version": "r3_exact300_deberta_target_nli_embeddings_v2",
        "status": "PASS",
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifier_features": False,
            "numeric_cache_contains_identifiers": False,
            "allowed_sources": [
                "identity manifest REFNO/RESULTNO/ROCKNAME for scope, alignment, and target text",
                "original-paper text from the frozen page cache",
                "frozen aggregate R3 taxonomy",
                "frozen positive ROCKMAG expert vocabulary",
                "fixed pretrained DeBERTa NLI weights",
            ],
        },
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": file_sha256(Path(__file__).resolve()),
        },
        "inputs": {
            "identity": {"path": str(identity_path), "sha256": file_sha256(identity_path)},
            "page_cache": {"path": str(page_cache_path), "sha256": file_sha256(page_cache_path)},
            "taxonomy": {"path": str(taxonomy_path), "sha256": file_sha256(taxonomy_path)},
            "positive_kb": {"path": str(positive_kb_path), "sha256": file_sha256(positive_kb_path)},
            "reused_passages": {"path": str(reused_path), "sha256": file_sha256(reused_path)},
            "pdf_dir": str(pdf_dir),
            "alignment_sha256": alignment_sha256(records),
        },
        "scope": {
            **coverage,
            "embedded_rows": len(records),
            "embedded_papers": len({record["REFNO"] for record in records}),
            "embedded_canonical_targets": len(target_rows),
            "target_limit": args.target_limit,
        },
        "knowledge": knowledge_audit,
        "retrieval": {
            "fresh_retrieval": dict(sorted(retrieval_audit.items())),
            "fresh_passages_per_paper": args.fresh_passages,
            "maximum_selected_passages_per_target": args.maximum_passages,
            "target_local": {
                "maximum_passages_per_target": 4,
                "radius_characters": 700,
                "target_anchor_prefixed": True,
            },
            "selected_passages_by_origin": dict(sorted(origin_counts.items())),
            "selected_passages_by_matched_knowledge_source": dict(
                sorted(knowledge_source_counts.items())
            ),
            "target_binding": "exact normalized target phrase plus informative target-token coverage",
            "method_only_is_retrieval_not_proof": True,
        },
        "nli": {
            "model_path": str(model_path),
            "upstream_name": "cross-encoder/nli-deberta-v3-base",
            "revision": MODEL_REVISION,
            "model_files": model_files,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "pair_count": len(premises),
            "hypothesis_names": list(HYPOTHESIS_NAMES),
            "hypothesis_templates": HYPOTHESIS_TEMPLATES,
            "model_audit": model_audit,
        },
        "output": {
            "path": str(output_path),
            "sha256": file_sha256(output_path),
            "array_names": sorted(arrays),
            "array_shapes": {name: list(matrix.shape) for name, matrix in sorted(arrays.items())},
            "structured_feature_names": structured_names,
        },
    }
    base.atomic_json(args.manifest.resolve(), manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "scope": manifest["scope"],
                "pair_count": len(premises),
                "output": manifest["output"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
