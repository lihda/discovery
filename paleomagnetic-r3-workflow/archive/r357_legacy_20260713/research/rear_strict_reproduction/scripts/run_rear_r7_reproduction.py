#!/usr/bin/env python3
"""Run a REAR-style strict reproduction shell on paleomagnetic R7.

This script follows the REAR paper's inference objects and equations as closely
as possible without training a 7B LLaMA rank head locally:

1. Build query-document pairs from R7 RESULTNO records.
2. Estimate a document relevance score s_ce with a local relevance scorer.
3. Construct REAR training labels with Eq. 10: s_rel = 0.5 * (s_ce + y).
4. Run the two REAR verification strategies:
   - source reliability: trust the answer from the highest-s_rel document.
   - knowledge consistency: combine source relevance with a baseline-consistency
     proxy, matching the paper's role for inverse perplexity under s_rel = 0.

The output is isolated under research/rear_strict_reproduction/outputs and does
not modify the production R3/R5/R7 workflow.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "research/rear_strict_reproduction/outputs"


NEGATIVE_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("younger_pole_or_apwp", r"younger .{0,120}(?:pole|palaeopole|paleopole|APWP)|(?:pole|palaeopole|paleopole|APWP).{0,120}younger", 1.0),
    ("remagnetization", r"remagneti[sz](?:ed|ation)|secondary (?:component|magneti[sz]ation|remanence)|overprint(?:ed|ing)?", 1.0),
    ("present_day_or_viscous", r"present[- ]day|recent overprint|viscous remanent magneti[sz]ation|VRM", 0.85),
    ("age_reset_event", r"impact|pseudotachylite|hydrothermal|metamorphic resetting|thermal resetting|reheating", 0.8),
    ("age_ambiguity", r"age spread|multi[- ]phase|multiple intrusive episodes|older and younger ages", 0.75),
    ("author_rejection", r"not reliable|unreliable|rejected|questionable|dubious|should not be used|cannot be considered", 0.9),
)

RESCUE_PATTERNS: tuple[tuple[str, str, float], ...] = (
    ("primary_origin", r"primary (?:remanence|magneti[sz]ation|origin)|primary ChRM|acquired (?:during|at) (?:cooling|emplacement|deposition)", 0.95),
    ("positive_field_test", r"positive (?:fold|reversal|baked contact|contact|conglomerate) test|fold test.{0,80}positive|reversal test.{0,80}positive", 1.0),
    ("independent_age", r"U[- ]Pb|baddeleyite|zircon|40Ar/39Ar|Ar[- ]Ar|K[- ]Ar|radiometric age|well dated|independent age", 0.7),
    ("stable_chrm", r"characteristic remanent magneti[sz]ation|ChRM|stable component|high[- ]temperature component", 0.65),
    ("contemporaneous_match", r"(?:consistent|agrees?|matches?|similar).{0,140}(?:contemporaneous|coeval|same age).{0,80}(?:pole|poles|APWP)", 0.8),
)

GENERIC_TARGET_TOKENS = {
    "formation",
    "group",
    "suite",
    "complex",
    "dyke",
    "dykes",
    "dike",
    "dikes",
    "swarm",
    "basalt",
    "basalts",
    "granite",
    "granites",
    "sediment",
    "sediments",
    "sandstone",
    "redbeds",
    "intrusive",
    "intrusives",
    "volcanic",
    "volcanics",
    "component",
}


@dataclass
class CandidateDoc:
    doc_id: str
    title: str
    text: str
    polarity: str
    candidate_answer: int
    s_ce: float
    y: int
    s_rel_label: float
    matched: list[str]
    features: list[float]
    trained_score: float = 0.0


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def norm_score(value: Any) -> int | None:
    text = as_text(value)
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def compact(text: str, limit: int = 1400) -> str:
    text = re.sub(r"\s+", " ", as_text(text)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def parse_json_cell(value: Any) -> dict[str, Any]:
    text = as_text(value)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def target_tokens(target: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", target)
    }
    return {token for token in tokens if token.lower() not in GENERIC_TARGET_TOKENS}


def pattern_hits(text: str, patterns: tuple[tuple[str, str, float], ...]) -> list[tuple[str, float]]:
    hits: list[tuple[str, float]] = []
    for name, pattern, weight in patterns:
        if re.search(pattern, text, flags=re.I | re.S):
            hits.append((name, weight))
    return hits


def relevance_score(query: str, doc: str, target: str, polarity: str) -> tuple[float, list[str]]:
    lower_doc = doc.lower()
    lower_query = query.lower()
    t_hits = [tok for tok in sorted(target_tokens(target)) if tok in lower_doc]
    q_tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9'-]{3,}", lower_query)
        if token not in GENERIC_TARGET_TOKENS
    }
    doc_tokens = set(re.findall(r"[a-z][a-z0-9'-]{3,}", lower_doc))
    overlap = len(q_tokens & doc_tokens) / max(1, len(q_tokens))
    neg_hits = pattern_hits(doc, NEGATIVE_PATTERNS)
    rescue_hits = pattern_hits(doc, RESCUE_PATTERNS)
    selected = neg_hits if polarity == "negative" else rescue_hits if polarity == "rescue" else neg_hits + rescue_hits
    pattern_signal = sum(weight for _, weight in selected)
    raw = -1.15 + 2.2 * min(1.0, overlap) + 0.85 * min(2.0, pattern_signal)
    if t_hits:
        raw += 0.55
    if polarity == "metadata" and pattern_signal == 0:
        raw += 0.15
    score = sigmoid(raw)
    matched = [name for name, _ in selected]
    if t_hits:
        matched.append("target:" + ",".join(t_hits[:4]))
    return round(score, 4), matched


def feature_vector(query: str, doc: str, target: str, polarity: str, baseline: int, candidate_answer: int) -> list[float]:
    lower_doc = doc.lower()
    lower_query = query.lower()
    q_tokens = {
        token
        for token in re.findall(r"[a-z][a-z0-9'-]{3,}", lower_query)
        if token not in GENERIC_TARGET_TOKENS
    }
    doc_tokens = set(re.findall(r"[a-z][a-z0-9'-]{3,}", lower_doc))
    overlap = len(q_tokens & doc_tokens) / max(1, len(q_tokens))
    target_hit_count = sum(1 for tok in target_tokens(target) if tok in lower_doc)
    neg_signal = sum(weight for _, weight in pattern_hits(doc, NEGATIVE_PATTERNS))
    rescue_signal = sum(weight for _, weight in pattern_hits(doc, RESCUE_PATTERNS))
    return [
        1.0,
        overlap,
        min(3.0, neg_signal),
        min(3.0, rescue_signal),
        min(5.0, target_hit_count) / 5.0,
        1.0 if polarity == "metadata" else 0.0,
        1.0 if polarity == "negative" else 0.0,
        1.0 if polarity == "rescue" else 0.0,
        1.0 if candidate_answer == baseline else 0.0,
        min(1.0, math.log1p(len(doc)) / 8.0),
    ]


def train_linear_assessor(samples: list[dict[str, Any]], epochs: int = 600, lr: float = 0.08) -> dict[str, Any]:
    rows: list[list[float]] = []
    labels: list[float] = []
    for sample in samples:
        if sample["split"] != "train":
            continue
        for doc in sample["docs"]:
            rows.append(doc.features)
            labels.append(doc.s_rel_label)
    if not rows:
        return {"weights": [], "bias": 0.0, "mean": [], "std": []}
    x = np.nan_to_num(np.asarray(rows, dtype=float), nan=0.0, posinf=8.0, neginf=-8.0)
    y = np.nan_to_num(np.asarray(labels, dtype=float), nan=0.0, posinf=1.0, neginf=0.0)
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-3] = 1.0
    x = np.nan_to_num(np.clip((x - mean) / std, -8.0, 8.0), nan=0.0, posinf=8.0, neginf=-8.0)
    y = np.clip(y, 0.02, 0.98)
    target_logit = np.log(y / (1.0 - y))
    design = np.concatenate([x, np.ones((x.shape[0], 1), dtype=float)], axis=1)
    reg = 0.15 * np.eye(design.shape[1], dtype=float)
    reg[-1, -1] = 0.0
    with np.errstate(all="ignore"):
        solved = np.linalg.pinv(design.T @ design + reg) @ design.T @ target_logit
    weights = np.nan_to_num(np.clip(solved[:-1], -6.0, 6.0), nan=0.0, posinf=6.0, neginf=-6.0)
    bias = max(-6.0, min(6.0, float(np.nan_to_num(solved[-1], nan=0.0, posinf=6.0, neginf=-6.0))))
    return {
        "weights": weights.tolist(),
        "bias": float(bias),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "train_docs": len(rows),
        "train_label_mean": float(y.mean()),
    }


def predict_linear_assessor(model: dict[str, Any], features: list[float]) -> float:
    if not model.get("weights"):
        return 0.0
    x = np.nan_to_num(np.asarray(features, dtype=float), nan=0.0, posinf=8.0, neginf=-8.0)
    mean = np.asarray(model["mean"], dtype=float)
    std = np.asarray(model["std"], dtype=float)
    weights = np.asarray(model["weights"], dtype=float)
    logit = float(np.clip((x - mean) / std, -8.0, 8.0) @ weights + model["bias"])
    return round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, logit)))), 4)


def iter_nested_quotes(value: Any) -> list[str]:
    quotes: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"quote", "reasoning", "summary", "final_decision_reasoning", "manual_review_reason"}:
                text = compact(child, 1000)
                if text:
                    quotes.append(text)
            else:
                quotes.extend(iter_nested_quotes(child))
    elif isinstance(value, list):
        for child in value:
            quotes.extend(iter_nested_quotes(child))
    return quotes


def load_page_cache(path: Path) -> dict[str, str]:
    cache: dict[str, str] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            refno = as_text(row.get("REFNO") or row.get("refno") or (row.get("source") or {}).get("refno"))
            chunks: list[str] = []
            for page in row.get("pages") or []:
                page_no = as_text(page.get("page") or page.get("page_no"))
                text = as_text(page.get("text"))
                if text:
                    chunks.append(f"[page {page_no}] {text}")
            if refno and chunks:
                cache[refno] = "\n".join(chunks)
    return cache


def snippet_windows(text: str, patterns: tuple[tuple[str, str, float], ...], target: str, limit: int = 4) -> list[tuple[str, str]]:
    snippets: list[tuple[str, str]] = []
    seen: set[tuple[str, int]] = set()
    for name, pattern, _ in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            bucket = match.start() // 500
            key = (name, bucket)
            if key in seen:
                continue
            seen.add(key)
            left = max(0, match.start() - 420)
            right = min(len(text), match.end() + 420)
            snippet = compact(text[left:right], 1200)
            snippets.append((name, snippet))
            if len(snippets) >= limit:
                return snippets
    if not snippets and target:
        for token in target_tokens(target):
            match = re.search(re.escape(token), text, flags=re.I)
            if match:
                left = max(0, match.start() - 420)
                right = min(len(text), match.end() + 420)
                snippets.append(("target_context", compact(text[left:right], 1200)))
                break
    return snippets


def metadata_text(payload: dict[str, Any]) -> str:
    meta = payload.get("source_metadata")
    if not isinstance(meta, dict):
        return ""
    keep = [
        "PMAGRESULT.COMPONENT",
        "PMAGRESULT.LOMAGAGE",
        "PMAGRESULT.HIMAGAGE",
        "PMAGRESULT.TESTS",
        "PMAGRESULT.COMMENTS",
        "PMAGRESULT.STATUS",
        "PMAGRESULT.ROCKMAG",
        "ROCKUNIT.ROCKTYPE",
        "ROCKUNIT.METHOD",
        "ROCKUNIT.STRUCTURE",
        "ROCKUNIT.PLACE",
        "ROCKUNIT.STRATA",
        "ROCKUNIT.STRATAGE",
        "ROCKUNIT.LOWAGE",
        "ROCKUNIT.HIGHAGE",
    ]
    parts = [f"{key}={as_text(meta.get(key))}" for key in keep if as_text(meta.get(key))]
    return "; ".join(parts)


def build_query(row: pd.Series, payload: dict[str, Any]) -> str:
    meta = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
    return compact(
        "Assess R7 for this paleomagnetic RESULTNO. "
        "R7=1 means the magnetization age/paleopole is reliable for the target age; "
        "R7=0 means younger pole similarity, remagnetization, overprint, age-reset, "
        "or age ambiguity makes the result unreliable. "
        f"RESULTNO={as_text(row.get('RESULTNO'))}; REFNO={as_text(row.get('REFNO'))}; "
        f"target={as_text(row.get('TARGET'))}; component={as_text(meta.get('PMAGRESULT.COMPONENT'))}; "
        f"mag_age={as_text(meta.get('PMAGRESULT.LOMAGAGE'))}-{as_text(meta.get('PMAGRESULT.HIMAGAGE'))}; "
        f"tests={as_text(meta.get('PMAGRESULT.TESTS'))}; comments={as_text(meta.get('PMAGRESULT.COMMENTS'))}; "
        f"rockunit_age={as_text(meta.get('ROCKUNIT.LOWAGE'))}-{as_text(meta.get('ROCKUNIT.HIGHAGE'))}; "
        f"method={as_text(meta.get('ROCKUNIT.METHOD'))}.",
        1400,
    )


def candidate_docs(row: pd.Series, payload: dict[str, Any], source_text: str, gold: int) -> list[CandidateDoc]:
    target = as_text(row.get("TARGET"))
    query = build_query(row, payload)
    docs: list[tuple[str, str, str, int, str]] = []
    meta = metadata_text(payload)
    baseline_answer = norm_score(row.get("R7")) or 1
    if meta:
        docs.append(("metadata", "PMAGRESULT/ROCKUNIT metadata", meta, baseline_answer, "metadata"))

    quotes = iter_nested_quotes(payload)
    for index, quote in enumerate(quotes[:5], start=1):
        neg = bool(pattern_hits(quote, NEGATIVE_PATTERNS))
        rescue = bool(pattern_hits(quote, RESCUE_PATTERNS))
        if neg and not rescue:
            polarity, answer = "negative", 0
        elif rescue and not neg:
            polarity, answer = "rescue", 1
        elif neg and rescue:
            polarity, answer = "negative", 0
        else:
            polarity, answer = "metadata", baseline_answer
        docs.append((f"payload_quote_{index}", f"R7 payload quote {index}", quote, answer, polarity))

    for index, (name, snippet) in enumerate(snippet_windows(source_text, NEGATIVE_PATTERNS, target, limit=4), start=1):
        docs.append((f"source_negative_{index}", f"Source negative evidence: {name}", snippet, 0, "negative"))
    for index, (name, snippet) in enumerate(snippet_windows(source_text, RESCUE_PATTERNS, target, limit=3), start=1):
        docs.append((f"source_rescue_{index}", f"Source rescue evidence: {name}", snippet, 1, "rescue"))

    out: list[CandidateDoc] = []
    seen_text: set[str] = set()
    for doc_id, title, text, answer, polarity in docs:
        text = compact(text, 1600)
        if not text or text in seen_text:
            continue
        seen_text.add(text)
        s_ce, matched = relevance_score(query, text, target, polarity)
        y = 1 if answer == gold and polarity != "metadata" else 0
        if polarity == "metadata" and answer == gold and matched:
            y = 1
        s_rel_label = round(0.5 * (s_ce + y), 4)
        features = feature_vector(query, text, target, polarity, baseline_answer, answer)
        out.append(CandidateDoc(doc_id, title, text, polarity, answer, s_ce, y, s_rel_label, matched, features))
    return sorted(out, key=lambda doc: doc.s_ce, reverse=True)


def metrics(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    gold = [row["gold"] for row in rows]
    pred = [row[pred_key] for row in rows]
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)

    def f1_for(label: int) -> float:
        local_tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        local_fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        local_fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = local_tp / (local_tp + local_fp) if local_tp + local_fp else 0.0
        recall = local_tp / (local_tp + local_fn) if local_tp + local_fn else 0.0
        return 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "rows": len(rows),
        "correct": tp + tn,
        "accuracy": (tp + tn) / len(rows) if rows else 0.0,
        "macro_f1": (f1_for(0) + f1_for(1)) / 2,
        "f1_0": f1_for(0),
        "f1_1": f1_for(1),
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "pred_counts": dict(Counter(pred)),
        "gold_counts": dict(Counter(gold)),
    }


def split_name(resultno: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{resultno}".encode("utf-8")).hexdigest()
    value = int(digest[:12], 16) / float(16**12 - 1)
    if value < 0.2:
        return "test"
    if value < 0.4:
        return "dev"
    return "train"


def run(args: argparse.Namespace) -> Path:
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_excel(args.targets, dtype=str)
    labels = targets[["RESULTNO", "R7"]].rename(columns={"R7": "R7_gold"})
    frame = pd.read_excel(args.r7_workbook, dtype=str)
    frame = frame.merge(labels, on="RESULTNO", how="left")
    if args.max_rows:
        frame = frame.head(args.max_rows).copy()
    cache = load_page_cache(args.page_cache)

    records: list[dict[str, Any]] = []
    dataset: list[dict[str, Any]] = []
    pairwise_examples: list[dict[str, Any]] = []
    sample_refs: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        gold = norm_score(row.get("R7_gold"))
        baseline = norm_score(row.get("R7"))
        if gold is None or baseline is None:
            continue
        payload = parse_json_cell(row.get("R7_RESULT_JSON"))
        refno = as_text(row.get("REFNO"))
        source_text = cache.get(refno, "")
        query = build_query(row, payload)
        docs = candidate_docs(row, payload, source_text, gold)
        if not docs:
            continue

        source_doc = max(docs, key=lambda doc: doc.s_ce)
        source_reliability_pred = source_doc.candidate_answer if source_doc.s_ce >= args.relevance_threshold else baseline

        def consistency_score(doc: CandidateDoc) -> float:
            c = 1.0 if doc.candidate_answer == baseline else 0.0
            return args.alpha * doc.s_ce + (1.0 - args.alpha) * c

        consistency_doc = max(docs, key=consistency_score)
        consistency_pred = consistency_doc.candidate_answer if consistency_score(consistency_doc) >= args.relevance_threshold else baseline

        split = split_name(as_text(row.get("RESULTNO")), args.seed)
        record = {
            "split": split,
            "refno": refno,
            "resultno": as_text(row.get("RESULTNO")),
            "target": as_text(row.get("TARGET")),
            "gold": gold,
            "baseline": baseline,
            "rear_source_reliability": source_reliability_pred,
            "rear_knowledge_consistency_proxy": consistency_pred,
            "top_doc_id": source_doc.doc_id,
            "top_doc_title": source_doc.title,
            "top_doc_polarity": source_doc.polarity,
            "top_doc_answer": source_doc.candidate_answer,
            "top_doc_s_ce": source_doc.s_ce,
            "top_doc_s_rel_label": source_doc.s_rel_label,
            "top_doc_y": source_doc.y,
            "top_doc_matched": ";".join(source_doc.matched),
            "consistency_doc_id": consistency_doc.doc_id,
            "consistency_doc_score": round(consistency_score(consistency_doc), 4),
            "changed_source_reliability": int(source_reliability_pred != baseline),
            "changed_knowledge_consistency": int(consistency_pred != baseline),
            "rear_trained_source_reliability": baseline,
            "rear_trained_knowledge_consistency": baseline,
            "trained_top_doc_id": "",
            "trained_top_doc_score": 0.0,
            "changed_trained_source_reliability": 0,
            "changed_trained_knowledge_consistency": 0,
        }
        records.append(record)
        sample_refs.append(
            {
                "record": record,
                "docs": docs,
                "baseline": baseline,
                "gold": gold,
                "split": split,
            }
        )

        dataset.append(
            {
                "question": query,
                "reference": [f"R7={gold}"],
                "ctxs": [
                    {
                        "id": doc.doc_id,
                        "title": doc.title,
                        "text": doc.text,
                        "polarity": doc.polarity,
                        "candidate_answer": f"R7={doc.candidate_answer}",
                        "s_ce_proxy": doc.s_ce,
                        "binary_y": doc.y,
                        "s_rel_eq10_label": doc.s_rel_label,
                        "matched": doc.matched,
                    }
                    for doc in docs[: args.max_docs_per_record]
                ],
                "meta": {
                    "split": split,
                    "refno": refno,
                    "resultno": as_text(row.get("RESULTNO")),
                    "target": as_text(row.get("TARGET")),
                    "baseline_r7": baseline,
                    "gold_r7": gold,
                },
            }
        )

        for doc in docs[: args.max_docs_per_record]:
            pairwise_examples.append(
                {
                    "prompt": query + "\nDocument:\n" + doc.text + "\nJudge whether the document is relevant before answering R7.",
                    "response": f"<RELEVANT>\nAnswer: R7={doc.candidate_answer}" if doc.s_rel_label >= 0.5 else f"<IRRELEVANT>\nAnswer: R7={baseline}",
                    "classes": [doc.s_rel_label],
                    "ans_score": None,
                    "meta": {
                        "refno": refno,
                        "resultno": as_text(row.get("RESULTNO")),
                        "doc_id": doc.doc_id,
                        "s_ce_proxy": doc.s_ce,
                        "binary_y": doc.y,
                        "equation": "s_rel = 0.5 * (s_ce + y), REAR Eq. 10",
                    },
                }
            )

    assessor = train_linear_assessor(sample_refs, epochs=args.assessor_epochs, lr=args.assessor_lr)
    for sample in sample_refs:
        baseline = sample["baseline"]
        docs = sample["docs"]
        record = sample["record"]
        for doc in docs:
            doc.trained_score = predict_linear_assessor(assessor, doc.features)

        trained_doc = max(docs, key=lambda doc: doc.trained_score)
        trained_source_pred = trained_doc.candidate_answer if trained_doc.trained_score >= args.relevance_threshold else baseline

        def trained_consistency_score(doc: CandidateDoc) -> float:
            c = 1.0 if doc.candidate_answer == baseline else 0.0
            return args.alpha * doc.trained_score + (1.0 - args.alpha) * c

        trained_consistency_doc = max(docs, key=trained_consistency_score)
        trained_consistency_pred = (
            trained_consistency_doc.candidate_answer
            if trained_consistency_score(trained_consistency_doc) >= args.relevance_threshold
            else baseline
        )
        record["rear_trained_source_reliability"] = trained_source_pred
        record["rear_trained_knowledge_consistency"] = trained_consistency_pred
        record["trained_top_doc_id"] = trained_doc.doc_id
        record["trained_top_doc_score"] = trained_doc.trained_score
        record["changed_trained_source_reliability"] = int(trained_source_pred != baseline)
        record["changed_trained_knowledge_consistency"] = int(trained_consistency_pred != baseline)

    for sample, data_row in zip(sample_refs, dataset):
        for doc_payload, doc in zip(data_row["ctxs"], sample["docs"][: args.max_docs_per_record]):
            doc_payload["trained_assessor_score"] = doc.trained_score

    summary = {
        "paper": "REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering",
        "conference": "EMNLP 2024 main conference",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "inputs": {
            "targets": str(args.targets),
            "r7_workbook": str(args.r7_workbook),
            "page_cache": str(args.page_cache),
        },
        "strict_alignment": {
            "query_document_pairs": "question/reference/ctxs objects mirror the official REAR test data format.",
            "relevance_assessment": "s_ce is a local cross-encoder substitute; full 7B linear rank head training is not run locally.",
            "eq10": "s_rel labels are computed exactly as 0.5 * (s_ce + y).",
            "source_reliability": "highest relevance document chooses final answer if above threshold.",
            "knowledge_consistency": "uses the REAR linear-combination structure with deterministic baseline as the s_rel=0 consistency proxy.",
            "trained_assessor": "a local linear module is trained on Eq. 10 labels from the paleomagnetic train split as a small-scale proxy for the paper's LLaMA hidden-state rank head.",
        },
        "parameters": {
            "relevance_threshold": args.relevance_threshold,
            "alpha": args.alpha,
            "seed": args.seed,
            "max_rows": args.max_rows,
            "assessor_epochs": args.assessor_epochs,
            "assessor_lr": args.assessor_lr,
        },
        "trained_assessor": assessor,
        "metrics_all": {
            "baseline": metrics(records, "baseline"),
            "rear_source_reliability": metrics(records, "rear_source_reliability"),
            "rear_knowledge_consistency_proxy": metrics(records, "rear_knowledge_consistency_proxy"),
            "rear_trained_source_reliability": metrics(records, "rear_trained_source_reliability"),
            "rear_trained_knowledge_consistency": metrics(records, "rear_trained_knowledge_consistency"),
        },
        "metrics_by_split": {},
    }
    for split in ("train", "dev", "test"):
        split_rows = [row for row in records if row["split"] == split]
        summary["metrics_by_split"][split] = {
            "baseline": metrics(split_rows, "baseline"),
            "rear_source_reliability": metrics(split_rows, "rear_source_reliability"),
            "rear_knowledge_consistency_proxy": metrics(split_rows, "rear_knowledge_consistency_proxy"),
            "rear_trained_source_reliability": metrics(split_rows, "rear_trained_source_reliability"),
            "rear_trained_knowledge_consistency": metrics(split_rows, "rear_trained_knowledge_consistency"),
        }

    (out_dir / "rear_r7_dataset.json").write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "rear_rank_sft_examples.jsonl").open("w", encoding="utf-8") as handle:
        for example in pairwise_examples:
            handle.write(json.dumps(example, ensure_ascii=False) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with (out_dir / "cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()) if records else [])
        if records:
            writer.writeheader()
            writer.writerows(records)

    report_lines = [
        "# REAR R7 Strict Reproduction Shell",
        "",
        f"Generated: {summary['created_at']}",
        "",
        "## Alignment Boundary",
        "",
        "- Official REAR code/data were downloaded under `external/`.",
        "- This run reproduces the REAR data objects, Eq. 10 relevance label construction, source-reliability verification, and knowledge-consistency selection shape.",
        "- It does not train LLaMA2-7B with a learned linear rank head; `s_ce` is a deterministic local scorer used as the small-reranker substitute.",
        "- A local linear relevance assessor is trained on Eq. 10 labels as a CPU-feasible proxy for the paper's explicit assessment module.",
        "",
        "## Metrics All Rows",
        "",
        "| Method | Accuracy | Macro-F1 | R7=0 F1 | FP | FN | Changed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, pred_key in [
        ("baseline", "baseline"),
        ("rear_source_reliability", "rear_source_reliability"),
        ("rear_knowledge_consistency_proxy", "rear_knowledge_consistency_proxy"),
        ("rear_trained_source_reliability", "rear_trained_source_reliability"),
        ("rear_trained_knowledge_consistency", "rear_trained_knowledge_consistency"),
    ]:
        m = summary["metrics_all"][name]
        changed = sum(row[pred_key] != row["baseline"] for row in records)
        report_lines.append(
            f"| {name} | {m['accuracy']:.4f} | {m['macro_f1']:.4f} | {m['f1_0']:.4f} | "
            f"{m['false_positive']} | {m['false_negative']} | {changed} |"
        )
    report_lines.extend(
        [
            "",
            "## Outputs",
            "",
            "- `rear_r7_dataset.json`: REAR-style question/reference/ctxs records.",
            "- `rear_rank_sft_examples.jsonl`: rank-SFT pair examples with Eq. 10 labels.",
            "- `cases.csv`: per-RESULTNO decisions and top relevance evidence.",
            "- `summary.json`: full metrics and alignment metadata.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps({"output_dir": str(out_dir), "summary": summary["metrics_all"]}, ensure_ascii=False, indent=2))
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "data/targets.xlsx")
    parser.add_argument("--r7-workbook", type=Path, default=ROOT / "outputs/r7_07081817_8966.xlsx")
    parser.add_argument("--page-cache", type=Path, default=ROOT / "outputs/r357_targets_07081817/r3_page_source_cache.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--relevance-threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for s_rel in knowledge-consistency selection.")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-docs-per-record", type=int, default=10)
    parser.add_argument("--assessor-epochs", type=int, default=600)
    parser.add_argument("--assessor-lr", type=float, default=0.01)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
