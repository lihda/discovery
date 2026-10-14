#!/usr/bin/env python3
"""Build label-free R3 own-study evidence features with scientific NLI.

The builder is deliberately separated from evaluation.  It accepts only the
strict identity manifest, original-paper text, and frozen aggregate knowledge;
there is no labels argument and no source-table access.  Candidate passages
are retrieved before NLI so that bibliography/background mentions do not
dominate a whole-document representation.
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
try:
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
except ModuleNotFoundError as exc:  # Optional development-only inference stack.
    torch = None  # type: ignore[assignment]
    AutoConfig = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _OPTIONAL_DEPENDENCY_ERROR: ModuleNotFoundError | None = exc
else:
    _OPTIONAL_DEPENDENCY_ERROR = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as base  # noqa: E402
from experiment_r3_sparse_paper_models import trim_reference_section  # noqa: E402


DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"
DEFAULT_OUTPUT = ROOT / "tmp/r3_nli_evidence/features.npz"
DEFAULT_MANIFEST = ROOT / "tmp/r3_nli_evidence/manifest.json"
DEFAULT_EVIDENCE = ROOT / "tmp/r3_nli_evidence/retrieved_passages.jsonl"

SPACE_RE = re.compile(r"\s+")
OWN_STUDY_RE = re.compile(
    r"\b(?:we|our|this\s+study|this\s+work|samples?\s+(?:were|was)|"
    r"specimens?\s+(?:were|was)|we\s+(?:measured|performed|used|applied|examined)|"
    r"(?:measurements?|analys(?:is|es)|experiments?|tests?)\s+(?:were|was)\s+"
    r"(?:made|performed|carried\s+out|conducted))\b",
    re.I,
)
RESULT_CUE_RE = re.compile(
    r"\b(?:indicat(?:e|es|ed|ing)|reveal(?:s|ed)?|show(?:s|ed)?|suggest(?:s|ed)?|"
    r"consistent\s+with|dominated\s+by|identified\s+as|carrier(?:s)?\s+(?:is|are|was|were))\b",
    re.I,
)
CITATION_HEAVY_RE = re.compile(r"(?:\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,?\s*\d{4}[a-z]?\)){2,}")

HYPOTHESIS_NAMES = (
    "own_study_generic",
    "concept_specific_method",
    "carrier_identification",
)

METHOD_HYPOTHESES = {
    "irm_acquisition": (
        "Isothermal remanent magnetization measurements were performed in this study."
    ),
    "thermomagnetic_curve": "Thermomagnetic measurements were performed in this study.",
    "temperature_dependent_susceptibility": (
        "Temperature-dependent magnetic susceptibility measurements were performed "
        "in this study."
    ),
    "three_axis_irm": "Three-axis IRM measurements were performed in this study.",
    "low_temperature_irm": "Low-temperature IRM measurements were performed in this study.",
    "remanence_temperature_experiment": (
        "Remanence-versus-temperature measurements were performed in this study."
    ),
    "hysteresis_forc": "Magnetic hysteresis measurements were performed in this study.",
    "coercivity_backfield": "Magnetic coercivity measurements were performed in this study.",
    "unblocking_spectrum": (
        "A magnetic unblocking-temperature spectrum was measured in this study."
    ),
    "magnetic_fabric": "Magnetic fabric measurements were performed in this study.",
    "petrographic_microscopy": (
        "Microscopic examination of the samples was performed in this study."
    ),
    "mineralogical_phase_analysis": (
        "Mineralogical phase measurements were performed in this study."
    ),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def passage_bounds(text: str, start: int, end: int, radius: int = 620) -> tuple[int, int]:
    left_limit = max(0, start - radius)
    right_limit = min(len(text), end + radius)
    left_candidates = (
        text.rfind("\n\n", left_limit, start),
        text.rfind(". ", left_limit, start),
    )
    left = max(left_candidates)
    left = left + (2 if left >= 0 and text[left : left + 2] in {"\n\n", ". "} else 1)
    if left <= 0:
        left = left_limit

    right_candidates = [
        value
        for value in (
            text.find("\n\n", end, right_limit),
            text.find(". ", end, right_limit),
        )
        if value >= 0
    ]
    right = min(right_candidates) + 1 if right_candidates else right_limit
    return left, right


def overlap_fraction(left: tuple[int, int], right: tuple[int, int]) -> float:
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    denominator = max(1, min(left[1] - left[0], right[1] - right[0]))
    return intersection / denominator


def retrieve_passages(
    text: str,
    concepts: Sequence[Mapping[str, Any]],
    *,
    maximum: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    body = trim_reference_section(text, cap=max(len(text) + 1, 180_000))
    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    raw_match_count = 0
    for concept in concepts:
        category = str(concept["category"])
        if "method" in category:
            category_key = "method"
        elif "diagnostic" in category:
            category_key = "diagnostic"
        else:
            category_key = "carrier"
        for pattern in concept["patterns"]:
            for match in pattern.finditer(body):
                raw_match_count += 1
                bounds = passage_bounds(body, match.start(), match.end())
                item = candidates.setdefault(
                    bounds,
                    {
                        "start": bounds[0],
                        "end": bounds[1],
                        "concepts": set(),
                        "categories": Counter(),
                    },
                )
                item["concepts"].add(str(concept["concept_id"]))
                item["categories"][category_key] += 1

    ranked: list[dict[str, Any]] = []
    for item in candidates.values():
        passage = SPACE_RE.sub(" ", body[item["start"] : item["end"]]).strip()
        if not passage:
            continue
        categories: Counter[str] = item["categories"]
        own_study_hits = len(OWN_STUDY_RE.findall(passage))
        result_hits = len(RESULT_CUE_RE.findall(passage))
        citation_heavy = bool(CITATION_HEAVY_RE.search(passage))
        score = (
            5 * int(categories["method"] > 0)
            + 3 * int(categories["diagnostic"] > 0)
            + 2 * int(categories["carrier"] > 0)
            + min(3, len(item["concepts"]))
            + 2 * min(2, own_study_hits)
            + min(2, result_hits)
            - 2 * int(citation_heavy)
        )
        ranked.append(
            {
                "start": item["start"],
                "end": item["end"],
                "text": passage[:2600],
                "concepts": sorted(item["concepts"]),
                "method_hits": int(categories["method"]),
                "diagnostic_hits": int(categories["diagnostic"]),
                "carrier_hits": int(categories["carrier"]),
                "own_study_hits": own_study_hits,
                "result_hits": result_hits,
                "citation_heavy": citation_heavy,
                "retrieval_score": score,
            }
        )
    ranked.sort(
        key=lambda item: (
            item["retrieval_score"],
            item["method_hits"],
            item["diagnostic_hits"],
            item["carrier_hits"],
            -item["start"],
        ),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for item in ranked:
        bounds = (int(item["start"]), int(item["end"]))
        if any(
            overlap_fraction(bounds, (int(old["start"]), int(old["end"]))) >= 0.72
            for old in selected
        ):
            continue
        selected.append(item)
        if len(selected) >= maximum:
            break
    if not selected:
        fallback = SPACE_RE.sub(" ", body[:1800]).strip() or "No extractable passage."
        selected = [
            {
                "start": 0,
                "end": min(len(body), 1800),
                "text": fallback,
                "concepts": [],
                "method_hits": 0,
                "diagnostic_hits": 0,
                "carrier_hits": 0,
                "own_study_hits": len(OWN_STUDY_RE.findall(fallback)),
                "result_hits": len(RESULT_CUE_RE.findall(fallback)),
                "citation_heavy": False,
                "retrieval_score": 0,
            }
        ]
    return selected, {
        "raw_pattern_matches": raw_match_count,
        "candidate_passages": len(ranked),
        "selected_passages": len(selected),
    }


def passage_hypotheses(row: Mapping[str, Any]) -> tuple[str, str, str]:
    concepts = {str(value) for value in row["concepts"]}
    method_hypothesis = next(
        (
            hypothesis
            for concept, hypothesis in METHOD_HYPOTHESES.items()
            if concept in concepts
        ),
        "Measurements of magnetic properties were performed in this study.",
    )
    return (
        "Measurements were performed in this study.",
        method_hypothesis,
        "The passage identifies a remanence carrier.",
    )


def entailment_index(model_name: str) -> int:
    if AutoConfig is None:
        raise RuntimeError("transformers is required to score NLI evidence")
    config = AutoConfig.from_pretrained(model_name)
    for index, label in config.id2label.items():
        if str(label).casefold() == "entailment":
            return int(index)
    raise ValueError(f"NLI model {model_name!r} has no entailment label")


def score_pairs(
    model_name: str,
    passages: Sequence[str],
    hypotheses: Sequence[str],
    *,
    batch_size: int,
    device: str,
) -> np.ndarray:
    if AutoTokenizer is None or AutoModelForSequenceClassification is None:
        raise RuntimeError("transformers is required to score NLI evidence")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    entailment = entailment_index(model_name)
    output = np.empty(len(passages), dtype=np.float32)
    for start in range(0, len(passages), batch_size):
        end = min(len(passages), start + batch_size)
        encoded = tokenizer(
            list(passages[start:end]),
            list(hypotheses[start:end]),
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1)[:, entailment]
        output[start:end] = probabilities.detach().cpu().numpy().astype(np.float32)
        if start == 0 or end == len(passages) or end % (batch_size * 25) == 0:
            print(f"NLI pairs {end}/{len(passages)}", flush=True)
    return output


def aggregate_features(
    passage_scores: np.ndarray,
    owners: Sequence[int],
    passage_rows: Sequence[Mapping[str, Any]],
    paper_count: int,
) -> tuple[np.ndarray, list[str]]:
    feature_names: list[str] = []
    for hypothesis_name in HYPOTHESIS_NAMES:
        feature_names.extend(
            [
                f"{hypothesis_name}__max",
                f"{hypothesis_name}__top2_mean",
                f"{hypothesis_name}__top3_mean",
                f"{hypothesis_name}__mean",
            ]
        )
    feature_names.extend(
        [
            "retrieval__passage_count",
            "retrieval__method_hits",
            "retrieval__diagnostic_hits",
            "retrieval__carrier_hits",
            "retrieval__own_study_hits",
            "retrieval__result_hits",
            "retrieval__max_score",
            "retrieval__citation_heavy_count",
        ]
    )
    output = np.zeros((paper_count, len(feature_names)), dtype=np.float32)
    members: dict[int, list[int]] = defaultdict(list)
    for index, owner in enumerate(owners):
        members[int(owner)].append(index)
    hypothesis_count = len(HYPOTHESIS_NAMES)
    for owner in range(paper_count):
        indices = members[owner]
        cursor = 0
        for hypothesis_index in range(hypothesis_count):
            values = np.sort(passage_scores[indices, hypothesis_index])[::-1]
            output[owner, cursor : cursor + 4] = (
                values[0],
                values[:2].mean(),
                values[:3].mean(),
                values.mean(),
            )
            cursor += 4
        rows = [passage_rows[index] for index in indices]
        output[owner, cursor:] = np.asarray(
            [
                len(rows),
                sum(int(row["method_hits"]) for row in rows),
                sum(int(row["diagnostic_hits"]) for row in rows),
                sum(int(row["carrier_hits"]) for row in rows),
                sum(int(row["own_study_hits"]) for row in rows),
                sum(int(row["result_hits"]) for row in rows),
                max(float(row["retrieval_score"]) for row in rows),
                sum(bool(row["citation_heavy"]) for row in rows),
            ],
            dtype=np.float32,
        )
    return output, feature_names


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=base.DEFAULT_IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--maximum-passages", type=int, default=8)
    parser.add_argument("--paper-limit", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    args = parser.parse_args(argv)

    if _OPTIONAL_DEPENDENCY_ERROR is not None or torch is None:
        raise SystemExit(
            "NLI feature construction requires the isolated torch/transformers "
            f"environment; missing {_OPTIONAL_DEPENDENCY_ERROR.name if _OPTIONAL_DEPENDENCY_ERROR else 'torch'}"
        )
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    if args.batch_size < 1 or args.maximum_passages < 1:
        raise SystemExit("batch size and maximum passages must be positive")

    identity_path = args.identity.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy_path = args.taxonomy.resolve()
    positive_kb_path = args.positive_kb.resolve() if args.positive_kb else None
    records, coverage = base.load_pdf_only_identity(identity_path, pdf_dir)
    refnos = sorted({record["REFNO"] for record in records})
    if args.paper_limit is not None:
        if args.paper_limit < 1:
            raise SystemExit("--paper-limit must be positive")
        refnos = refnos[: args.paper_limit]
    paper_text = base.load_page_text(page_cache_path, set(refnos))
    concepts, knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )

    grouped_passages: list[list[dict[str, Any]]] = []
    retrieval_audit: Counter[str] = Counter()
    for position, refno in enumerate(refnos, start=1):
        passages, audit = retrieve_passages(
            paper_text[refno], concepts, maximum=args.maximum_passages
        )
        grouped_passages.append(passages)
        retrieval_audit.update(audit)
        if position % 200 == 0 or position == len(refnos):
            print(f"retrieved papers {position}/{len(refnos)}", flush=True)

    flat_rows: list[dict[str, Any]] = []
    owners: list[int] = []
    for owner, passages in enumerate(grouped_passages):
        for passage in passages:
            flat_rows.append(passage)
            owners.append(owner)

    premises: list[str] = []
    pair_hypotheses: list[str] = []
    pair_locations: list[tuple[int, int]] = []
    for passage_index, row in enumerate(flat_rows):
        for hypothesis_index, hypothesis in enumerate(passage_hypotheses(row)):
            premises.append(str(row["text"]))
            pair_hypotheses.append(hypothesis)
            pair_locations.append((passage_index, hypothesis_index))
    flat_scores = score_pairs(
        args.model,
        premises,
        pair_hypotheses,
        batch_size=args.batch_size,
        device=args.device,
    )
    passage_scores = np.zeros(
        (len(flat_rows), len(HYPOTHESIS_NAMES)), dtype=np.float32
    )
    for score, (passage_index, hypothesis_index) in zip(flat_scores, pair_locations):
        passage_scores[passage_index, hypothesis_index] = score
    features, feature_names = aggregate_features(
        passage_scores, owners, flat_rows, len(refnos)
    )

    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        refnos=np.asarray(refnos, dtype=str),
        features=features,
        feature_names=np.asarray(feature_names, dtype=str),
        passage_scores=passage_scores,
        passage_owners=np.asarray(owners, dtype=np.int32),
    )
    evidence_path = args.evidence.resolve()
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    with evidence_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(flat_rows):
            record = {
                "REFNO": refnos[owners[index]],
                **row,
                "nli": {
                    HYPOTHESIS_NAMES[hypothesis]: round(
                        float(passage_scores[index, hypothesis]), 8
                    )
                    for hypothesis in range(len(HYPOTHESIS_NAMES))
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest_path = args.manifest.resolve()
    manifest = {
        "schema_version": "r3_label_free_nli_evidence_features_v1",
        "status": "PASS",
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "allowed_sources": [
                "strict identity manifest",
                "original-paper page cache",
                "frozen aggregate R3 knowledge",
                "pretrained NLI model",
            ],
        },
        "inputs": {
            "identity": {"path": str(identity_path), "sha256": file_sha256(identity_path)},
            "page_cache": {"path": str(page_cache_path), "sha256": file_sha256(page_cache_path)},
            "taxonomy": {"path": str(taxonomy_path), "sha256": file_sha256(taxonomy_path)},
            "positive_kb": (
                {"path": str(positive_kb_path), "sha256": file_sha256(positive_kb_path)}
                if positive_kb_path is not None
                else None
            ),
            "pdf_dir": str(pdf_dir),
        },
        "scope": {
            **coverage,
            "processed_papers": len(refnos),
            "paper_limit": args.paper_limit,
        },
        "knowledge": knowledge_audit,
        "retrieval": {
            "maximum_passages_per_paper": args.maximum_passages,
            **dict(sorted(retrieval_audit.items())),
        },
        "nli": {
            "model": args.model,
            "device": args.device,
            "batch_size": args.batch_size,
            "hypotheses": {
                "names": list(HYPOTHESIS_NAMES),
                "own_study_generic": "Measurements were performed in this study.",
                "concept_specific_method": METHOD_HYPOTHESES,
                "concept_specific_fallback": (
                    "Measurements of magnetic properties were performed in this study."
                ),
                "carrier_identification": "The passage identifies a remanence carrier.",
            },
            "pair_count": len(premises),
        },
        "output": {
            "features": {"path": str(output_path), "sha256": file_sha256(output_path)},
            "evidence": {"path": str(evidence_path), "sha256": file_sha256(evidence_path)},
            "feature_names": feature_names,
        },
    }
    base.atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
