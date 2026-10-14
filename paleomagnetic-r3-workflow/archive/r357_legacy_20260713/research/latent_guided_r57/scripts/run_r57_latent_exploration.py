#!/usr/bin/env python3
"""Run isolated latent-guided R5/R7 research experiments.

This script deliberately lives under research/ and never edits the production
R3/R5/R7 workflow. It recovers gold labels from the existing HTML review
payloads, joins workbook JSON payloads and page-cache text, then evaluates a
few lightweight reproductions of recent RAG/knowledge-conflict ideas:

1. Prototype latent guidance: score samples against expert-rule prototypes.
2. CRAG-style correction: keep the baseline unless retrieved evidence is strong.
3. Self-RAG-style abstention: mark ambiguous evidence for manual review.
4. Conflict-aware guidance: expose negative-vs-rescue evidence conflicts.

The embedding backend defaults to a local hash embedder so the experiment is
reproducible offline. A Qwen/OpenAI-compatible endpoint can be enabled with
--embedding-provider qwen or auto and EMBEDDING_URL/EMBEDDING_TOKEN env vars.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "research/latent_guided_r57/outputs"


@dataclass(frozen=True)
class Prototype:
    indicator: str
    proto_id: str
    polarity: str
    title: str
    text: str
    patterns: tuple[str, ...]


R5_NEGATIVE_PATTERNS = (
    r"no palaeohorizontal control|no paleohorizontal control",
    r"without (?:any )?structural correction|not structurally corrected|no structural correction",
    r"vertical[- ]axis rotation|block rotation|rotated clockwise|rotated counter[- ]?clockwise",
    r"allochthonous|translated block|displaced block|noticeably shifted",
    r"inclination shallowing|flattening correction|corrected inclination",
    r"attitudes? (?:are )?not known|positions? are not known",
    r"suture|orogenic belt|terrane|rift block|mobile belt",
    r"admin(?:istrator)? calculated|combined result|data included|restricted data",
)

R5_RESCUE_PATTERNS = (
    r"bedding|strike|dip|tilt correction|tilt[- ]corrected|structural correction",
    r"fold test|positive fold test|pre[- ]folding|syn[- ]folding|post[- ]folding",
    r"paleohorizontal|palaeohorizontal|horizontal bedding|flat[- ]lying|sub[- ]horizontal",
    r"contact test|baked contact|field test|conglomerate test",
    r"no significant flattening|E/I method|volcanic interbed|intercalated volcanic",
)

R7_NEGATIVE_PATTERNS = (
    r"younger (?:pole|poles|palaeopole|paleopole|magneti[sz]ation|remanence)",
    r"remagneti[sz](?:ed|ation)|overprint|secondary component|secondary magneti[sz]ation",
    r"present[- ]day overprint|recent overprint|viscous remanent magneti[sz]ation",
    r"does not match contemporaneous poles|mismatch(?:es)? contemporaneous",
    r"similar(?:ity)? .{0,120}(?:Bushveld|younger|reference pole|APWP)",
    r"apparent polar wander|APWP|polar wander path",
    r"age spread|older and younger ages|multi[- ]phase|multiple intrusive episodes",
    r"impact|pseudotachylite|hydrothermal|metamorphic resetting|resetting",
)

R7_RESCUE_PATTERNS = (
    r"primary remanence|primary magneti[sz]ation|primary origin",
    r"positive fold test|pre[- ]folding|positive baked contact|baked contact test",
    r"reversal test|positive reversal|conglomerate test|field test",
    r"well dated|U[- ]Pb|Ar[- ]Ar|40Ar/39Ar|independent age|radiometric age",
    r"single component|stable component|characteristic remanent magneti[sz]ation|ChRM",
    r"consistent with contemporaneous poles|agrees? with .* poles|matches? .* APWP",
)


PROTOTYPES: list[Prototype] = [
    Prototype(
        "R5",
        "R5_L1_no_paleohorizontal_control",
        "negative",
        "Missing paleohorizontal or structural datum",
        "R5 should be questioned when a target lacks paleohorizontal control, bedding attitude, structural correction, or reliable tilt datum.",
        R5_NEGATIVE_PATTERNS[:2],
    ),
    Prototype(
        "R5",
        "R5_L2_mobile_block_rotation",
        "negative",
        "Mobile block or vertical-axis rotation",
        "R5 flips to unreliable when a target is in a rotated, allochthonous, translated, displaced, active terrane, rift block, or mobile orogenic setting without correction.",
        R5_NEGATIVE_PATTERNS[2:4] + R5_NEGATIVE_PATTERNS[6:7],
    ),
    Prototype(
        "R5",
        "R5_L3_unvalidated_flattening_correction",
        "negative",
        "Unvalidated inclination shallowing correction",
        "Inclination shallowing, flattening correction, or corrected inclination requires independent validation before structural control is accepted.",
        R5_NEGATIVE_PATTERNS[4:5],
    ),
    Prototype(
        "R5",
        "R5_L4_metadata_structural_risk",
        "negative",
        "Metadata-level structural risk",
        "Administrator calculated, combined, restricted, included, or transferred paleomagnetic results can indicate structural-control risk.",
        R5_NEGATIVE_PATTERNS[7:8],
    ),
    Prototype(
        "R5",
        "R5_R1_structural_rescue",
        "rescue",
        "Explicit structural control rescue",
        "R5 can be preserved when the source provides bedding, strike, dip, tilt correction, structural correction, paleohorizontal markers, or positive fold/contact tests.",
        R5_RESCUE_PATTERNS,
    ),
    Prototype(
        "R7",
        "R7_L1_younger_pole_similarity",
        "negative",
        "Younger pole or APWP similarity",
        "R7 should be questioned when the target paleopole is closer to a younger reference pole or APWP segment than to contemporaneous poles.",
        R7_NEGATIVE_PATTERNS[0:1] + R7_NEGATIVE_PATTERNS[4:6],
    ),
    Prototype(
        "R7",
        "R7_L2_remagnetization_overprint",
        "negative",
        "Remagnetization or overprint",
        "R7 flips to unreliable when a component is interpreted as remagnetized, secondary, overprinted, present-day, or a younger magnetization.",
        R7_NEGATIVE_PATTERNS[1:4],
    ),
    Prototype(
        "R7",
        "R7_L3_age_ambiguity_resetting",
        "negative",
        "Age ambiguity or resetting",
        "R7 should be questioned when dyke swarms, impact rocks, hydrothermal alteration, metamorphism, or multiple intrusive phases create younger age ambiguity.",
        R7_NEGATIVE_PATTERNS[6:8],
    ),
    Prototype(
        "R7",
        "R7_R1_primary_age_rescue",
        "rescue",
        "Primary age and field-test rescue",
        "R7 can be preserved when primary origin is supported by independent age constraints, positive fold, reversal, baked contact, or conglomerate tests.",
        R7_RESCUE_PATTERNS,
    ),
]


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_score(value: Any) -> int | None:
    text = as_text(value)
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def extract_html_payload(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r"<script[^>]+id=[\"']r3-payload[\"'][^>]*>(.*?)</script>",
        text,
        flags=re.I | re.S,
    )
    if not match:
        raise SystemExit(f"Could not find r3-payload JSON in {path}")
    payload_text = html.unescape(match.group(1).strip())
    return json.loads(payload_text)


def parse_json_cell(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = as_text(value)
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def iter_quotes(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"quote", "reasoning", "summary", "final_decision_reasoning", "label", "evidence"}:
                text = as_text(child)
                if text:
                    yield text
            else:
                yield from iter_quotes(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_quotes(child)


def load_workbook_payloads(path: Path, indicator: str) -> dict[str, dict[str, Any]]:
    frame = pd.read_excel(path, dtype=str)
    json_column = f"{indicator}_RESULT_JSON"
    payloads: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        resultno = as_text(row.get("RESULTNO"))
        payload = parse_json_cell(row.get(json_column))
        payloads[resultno] = payload
    return payloads


def load_source_cache(path: Path, refnos: set[str]) -> dict[str, str]:
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
            if refno not in refnos:
                continue
            pages = row.get("pages") or []
            chunks = []
            for page in pages:
                page_no = as_text(page.get("page"))
                page_text = as_text(page.get("text"))
                if page_text:
                    chunks.append(f"[page {page_no}] {page_text}")
            cache[refno] = "\n".join(chunks)
    return cache


def metadata_text(payload: dict[str, Any]) -> str:
    metadata = payload.get("source_metadata") or {}
    if not isinstance(metadata, dict):
        return ""
    keep = [
        "ROCKNAME",
        "PMAGRESULT.COMPONENT",
        "PMAGRESULT.TESTS",
        "PMAGRESULT.TILT",
        "PMAGRESULT.COMMENTS",
        "ROCKUNIT.ROCKTYPE",
        "ROCKUNIT.METHOD",
        "ROCKUNIT.STRUCTURE",
        "ROCKUNIT.PLACE",
        "ROCKUNIT.TERRANE",
        "ROCKUNIT.STRATA",
        "ROCKUNIT.STRATAGE",
        "ROCKUNIT.STATUS",
    ]
    return "; ".join(f"{key}={as_text(metadata.get(key))}" for key in keep if as_text(metadata.get(key)))


def keyword_excerpts(text: str, patterns: tuple[str, ...], radius: int = 420, limit: int = 4500) -> str:
    snippets: list[str] = []
    seen: set[tuple[int, int]] = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I | re.S):
            start = max(0, match.start() - radius)
            end = min(len(text), match.end() + radius)
            key = (start // 200, end // 200)
            if key in seen:
                continue
            seen.add(key)
            snippet = re.sub(r"\s+", " ", text[start:end]).strip()
            snippets.append(snippet)
            if sum(len(item) for item in snippets) >= limit:
                return "\n".join(snippets)[:limit]
    return "\n".join(snippets)[:limit]


def build_records(
    indicator: str,
    html_path: Path,
    workbook_path: Path,
    source_cache: dict[str, str],
) -> list[dict[str, Any]]:
    html_payload = extract_html_payload(html_path)
    workbook_payloads = load_workbook_payloads(workbook_path, indicator)
    patterns = R5_NEGATIVE_PATTERNS + R5_RESCUE_PATTERNS if indicator == "R5" else R7_NEGATIVE_PATTERNS + R7_RESCUE_PATTERNS
    records: list[dict[str, Any]] = []
    for row in html_payload.get("rows") or []:
        resultno = as_text(row.get("resultno"))
        refno = as_text(row.get("refno"))
        target = as_text(row.get("target"))
        payload = workbook_payloads.get(resultno, {})
        baseline = normalize_score(row.get("score"))
        gold = normalize_score(row.get("standard_score"))
        if baseline is None or gold is None:
            continue
        source_text = source_cache.get(refno, "")
        evidence_quotes = " ".join(list(iter_quotes(payload))[:80])
        evidence_excerpt = keyword_excerpts(source_text, patterns)
        compact_text = "\n".join(
            part
            for part in [
                f"indicator={indicator}",
                f"target={target}",
                metadata_text(payload),
                evidence_quotes,
                evidence_excerpt or source_text[:2200],
            ]
            if part
        )
        records.append(
            {
                "indicator": indicator,
                "refno": refno,
                "resultno": resultno,
                "target": target,
                "baseline": baseline,
                "gold": gold,
                "mismatch": bool(row.get("mismatch")),
                "error_type": as_text(row.get("error_type")),
                "manual_review": bool(payload.get("manual_review")),
                "source_available": bool(source_text.strip()),
                "source_chars": len(source_text),
                "metadata": metadata_text(payload),
                "baseline_summary": as_text(payload.get("summary") or row.get("summary")),
                "compact_text": compact_text[:12000],
            }
        )
    return records


TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+\-/]{1,}", re.I)


def tokenize(text: str) -> list[str]:
    raw = [token.lower() for token in TOKEN_RE.findall(text)]
    filtered = [token for token in raw if len(token) > 1]
    bigrams = [f"{a}_{b}" for a, b in zip(filtered, filtered[1:])]
    return filtered + bigrams


def hash_embed_texts(texts: list[str], dims: int = 4096) -> np.ndarray:
    vectors = np.zeros((len(texts), dims), dtype=np.float32)
    document_frequency: dict[str, int] = {}
    tokenized = [tokenize(text) for text in texts]
    for tokens in tokenized:
        for token in set(tokens):
            document_frequency[token] = document_frequency.get(token, 0) + 1
    n_docs = max(len(texts), 1)
    for row, tokens in enumerate(tokenized):
        counts: dict[str, int] = {}
        for token in tokens:
            counts[token] = counts.get(token, 0) + 1
        for token, count in counts.items():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            idf = math.log((1 + n_docs) / (1 + document_frequency.get(token, 1))) + 1.0
            vectors[row, index] += sign * (1.0 + math.log(count)) * idf
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def qwen_embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    url = os.environ.get("EMBEDDING_URL", "").strip()
    token = os.environ.get("EMBEDDING_TOKEN", "").strip()
    model = os.environ.get("EMBEDDING_MODEL", "qwen3-embedding").strip()
    if not url:
        raise RuntimeError("EMBEDDING_URL is not set")
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        body = json.dumps({"model": model, "input": batch}).encode("utf-8")
        request = urllib.request.Request(url, data=body, method="POST")
        request.add_header("Content-Type", "application/json")
        if token:
            request.add_header("Authorization", token)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"Qwen embedding request failed: {exc}") from exc
        data = payload.get("data") or []
        if len(data) != len(batch):
            raise RuntimeError(f"Embedding endpoint returned {len(data)} vectors for {len(batch)} inputs")
        data = sorted(data, key=lambda item: item.get("index", 0))
        vectors.extend(item["embedding"] for item in data)
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def embed_texts(texts: list[str], provider: str) -> tuple[np.ndarray, str]:
    if provider == "hash":
        return hash_embed_texts(texts), "hash"
    if provider == "qwen":
        return qwen_embed_texts(texts), "qwen"
    if provider == "auto":
        try:
            return qwen_embed_texts(texts), "qwen"
        except Exception as exc:  # noqa: BLE001 - report fallback in output metadata
            print(f"[warn] Qwen embeddings unavailable; falling back to hash embeddings: {exc}", file=sys.stderr)
            return hash_embed_texts(texts), "hash"
    raise ValueError(provider)


def split_name(refno: str) -> str:
    digest = hashlib.md5(refno.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 10 < 7 else "test"


def pattern_score(text: str, patterns: tuple[str, ...]) -> float:
    if not patterns:
        return 0.0
    hits = 0
    for pattern in patterns:
        if re.search(pattern, text, flags=re.I | re.S):
            hits += 1
    return hits / len(patterns)


STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "not",
    "but",
    "source",
    "metadata",
    "indicator",
    "target",
    "rockunit",
    "pmagresult",
    "result",
    "results",
    "profile",
    "matched",
    "rules",
    "none",
    "default",
    "calibrated",
    "evidence",
    "recovered",
    "present",
    "absent",
    "criterion",
}


def row_mask_tokens(row: dict[str, Any]) -> set[str]:
    """Tokens that are too sample-specific for contrastive rule discovery."""

    mask_text_parts = [row.get("target", "")]
    metadata = row.get("metadata", "")
    for key in ("ROCKNAME", "ROCKUNIT.PLACE", "ROCKUNIT.TERRANE", "ROCKUNIT.STRATA"):
        match = re.search(rf"{re.escape(key)}=([^;]+)", metadata)
        if match:
            mask_text_parts.append(match.group(1))
    return {
        token
        for token in tokenize(" ".join(mask_text_parts))
        if len(token) > 2
    }


def contrastive_tokens(row: dict[str, Any], masked: bool) -> set[str]:
    tokens = {
        token
        for token in tokenize(row["compact_text"])
        if token not in STOPWORDS and not token.isdigit() and len(token) > 2
    }
    if masked:
        tokens -= row_mask_tokens(row)
    return tokens


def learn_contrastive_scores(rows: list[dict[str, Any]], masked: bool = False) -> dict[str, float]:
    """Learn dev-only log-odds tokens associated with gold=0 vs gold=1."""

    zero_counts: dict[str, int] = {}
    one_counts: dict[str, int] = {}
    document_frequency: dict[str, int] = {}
    zero_total = 0
    one_total = 0
    for row in rows:
        if row["split"] != "dev":
            continue
        tokens = contrastive_tokens(row, masked)
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1
        if row["gold"] == 0:
            for token in tokens:
                zero_counts[token] = zero_counts.get(token, 0) + 1
            zero_total += len(tokens)
        else:
            for token in tokens:
                one_counts[token] = one_counts.get(token, 0) + 1
            one_total += len(tokens)
    vocab = {
        token
        for token, df in document_frequency.items()
        if df >= 4 and not re.fullmatch(r"\d+(?:\.\d+)?", token)
    }
    alpha = 0.5
    vocab_size = max(len(vocab), 1)
    scores: dict[str, float] = {}
    for token in vocab:
        p_zero = (zero_counts.get(token, 0) + alpha) / (zero_total + alpha * vocab_size)
        p_one = (one_counts.get(token, 0) + alpha) / (one_total + alpha * vocab_size)
        scores[token] = math.log(p_zero / p_one)
    return scores


def apply_contrastive_scores(rows: list[dict[str, Any]]) -> dict[str, list[tuple[str, float]]]:
    """Attach contrastive risk/rescue scores and return top discovered tokens."""

    top_tokens: dict[str, list[tuple[str, float]]] = {}
    for indicator in ("R5", "R7"):
        subset = [row for row in rows if row["indicator"] == indicator]
        scores = learn_contrastive_scores(subset, masked=False)
        masked_scores = learn_contrastive_scores(subset, masked=True)
        top_tokens[indicator] = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:40]
        top_tokens[f"{indicator}_masked"] = sorted(masked_scores.items(), key=lambda item: item[1], reverse=True)[:40]
        for row in subset:
            tokens = contrastive_tokens(row, masked=False)
            denom = math.sqrt(max(len(tokens), 1))
            risk = sum(max(scores.get(token, 0.0), 0.0) for token in tokens) / denom
            rescue = sum(max(-scores.get(token, 0.0), 0.0) for token in tokens) / denom
            row["contrastive_negative_score"] = float(risk)
            row["contrastive_rescue_score"] = float(rescue)
            masked_tokens = contrastive_tokens(row, masked=True)
            masked_denom = math.sqrt(max(len(masked_tokens), 1))
            masked_risk = sum(max(masked_scores.get(token, 0.0), 0.0) for token in masked_tokens) / masked_denom
            masked_rescue = sum(max(-masked_scores.get(token, 0.0), 0.0) for token in masked_tokens) / masked_denom
            row["masked_contrastive_negative_score"] = float(masked_risk)
            row["masked_contrastive_rescue_score"] = float(masked_rescue)
    return top_tokens


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    def safe(num: float, den: float) -> float:
        return num / den if den else 0.0

    precision_1 = safe(tp, tp + fp)
    recall_1 = safe(tp, tp + fn)
    f1_1 = safe(2 * precision_1 * recall_1, precision_1 + recall_1)
    precision_0 = safe(tn, tn + fn)
    recall_0 = safe(tn, tn + fp)
    f1_0 = safe(2 * precision_0 * recall_0, precision_0 + recall_0)
    return {
        "n": len(y_true),
        "accuracy": safe(tp + tn, len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_1": f1_1,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_0": f1_0,
        "macro_f1": (f1_0 + f1_1) / 2,
    }


def objective(metrics: dict[str, Any]) -> float:
    return 0.45 * metrics["macro_f1"] + 0.35 * metrics["f1_0"] + 0.20 * metrics["accuracy"]


def decision_mode(method: str) -> str:
    if method == "prototype_only":
        return "prototype_only"
    if method == "selfrag_abstain":
        return "selfrag_abstain"
    return "crag_correction"


def score_fields(method: str) -> tuple[str, str]:
    if method in {"prototype_only", "crag_correction", "selfrag_abstain"}:
        return "negative_score", "rescue_score"
    if method == "pattern_crag":
        return "pattern_negative_score", "pattern_rescue_score"
    if method == "hybrid_crag":
        return "hybrid_negative_score", "hybrid_rescue_score"
    if method == "contrastive_crag":
        return "contrastive_negative_score", "contrastive_rescue_score"
    if method == "masked_contrastive_crag":
        return "masked_contrastive_negative_score", "masked_contrastive_rescue_score"
    raise ValueError(method)


def apply_decision(
    baseline: int,
    negative_score: float,
    rescue_score: float,
    params: dict[str, float],
    method: str,
) -> tuple[int, bool, str]:
    threshold = params["threshold"]
    margin = params["margin"]
    rescue_threshold = params["rescue_threshold"]
    uncertainty = params.get("uncertainty", 0.025)
    negative_active = negative_score >= threshold and (negative_score - rescue_score) >= margin
    rescue_active = rescue_score >= rescue_threshold and (rescue_score - negative_score) >= margin
    conflict = negative_score >= threshold and rescue_score >= rescue_threshold and abs(negative_score - rescue_score) < uncertainty
    mode = decision_mode(method)
    if mode == "prototype_only":
        if negative_active:
            return 0, conflict, "negative_prototype"
        return 1, conflict, "default_positive"
    if mode == "crag_correction":
        if baseline == 1 and negative_active:
            return 0, conflict, "corrective_flip_to_zero"
        if baseline == 0 and rescue_active and not negative_active:
            return 1, conflict, "rescue_flip_to_one"
        return baseline, conflict, "baseline_preserved"
    if mode == "selfrag_abstain":
        if conflict:
            return baseline, True, "manual_review_conflict"
        if baseline == 1 and negative_active:
            return 0, False, "selfrag_retrieved_counterevidence"
        if baseline == 0 and rescue_active and not negative_active:
            return 1, False, "selfrag_retrieved_rescue"
        return baseline, False, "selfrag_no_change"
    raise ValueError(method)


def tune_method(rows: list[dict[str, Any]], method: str) -> dict[str, float]:
    dev = [row for row in rows if row["split"] == "dev"]
    negative_field, rescue_field = score_fields(method)
    negative_values = np.asarray([row[negative_field] for row in dev], dtype=np.float32)
    rescue_values = np.asarray([row[rescue_field] for row in dev], dtype=np.float32)
    diff_values = negative_values - rescue_values

    def quantile_grid(values: np.ndarray, fallback: tuple[float, float]) -> list[float]:
        if values.size == 0 or float(values.max()) == float(values.min()):
            return list(np.linspace(fallback[0], fallback[1], 9))
        grid = np.quantile(values, np.linspace(0.05, 0.95, 19))
        grid = np.unique(np.concatenate([grid, np.asarray([0.0, float(values.mean())])]))
        return [float(item) for item in grid]

    best_score = -1.0
    best_params: dict[str, float] = {}
    thresholds = quantile_grid(negative_values, (0.03, 0.34))
    margins = quantile_grid(diff_values, (-0.04, 0.12))
    rescue_thresholds = quantile_grid(rescue_values, (0.04, 0.30))
    for threshold in thresholds:
        for margin in margins:
            for rescue_threshold in rescue_thresholds:
                params = {
                    "threshold": float(threshold),
                    "margin": float(margin),
                    "rescue_threshold": float(rescue_threshold),
                    "uncertainty": 0.03,
                }
                y_true: list[int] = []
                y_pred: list[int] = []
                for row in dev:
                    pred, _, _ = apply_decision(
                        row["baseline"],
                        row[negative_field],
                        row[rescue_field],
                        params,
                        method,
                    )
                    y_true.append(row["gold"])
                    y_pred.append(pred)
                metrics = confusion(y_true, y_pred)
                score = objective(metrics)
                if score > best_score:
                    best_score = score
                    best_params = params
    return best_params


def evaluate_method(rows: list[dict[str, Any]], method: str, params: dict[str, float]) -> dict[str, Any]:
    output: dict[str, Any] = {"params": params, "splits": {}, "changed_cases": [], "manual_review_cases": []}
    negative_field, rescue_field = score_fields(method)
    for split in ("dev", "test", "all"):
        selected = rows if split == "all" else [row for row in rows if row["split"] == split]
        y_true: list[int] = []
        y_pred: list[int] = []
        changed = 0
        manual_review = 0
        for row in selected:
            pred, needs_review, reason = apply_decision(
                row["baseline"],
                row[negative_field],
                row[rescue_field],
                params,
                method,
            )
            y_true.append(row["gold"])
            y_pred.append(pred)
            changed += int(pred != row["baseline"])
            manual_review += int(needs_review)
            if split == "all" and (pred != row["baseline"] or needs_review):
                case = {
                    "indicator": row["indicator"],
                    "refno": row["refno"],
                    "resultno": row["resultno"],
                    "target": row["target"],
                    "baseline": row["baseline"],
                    "pred": pred,
                    "gold": row["gold"],
                    "baseline_error_type": row["error_type"],
                    "negative_score": round(row[negative_field], 4),
                    "rescue_score": round(row[rescue_field], 4),
                    "top_negative": row["top_negative"],
                    "top_rescue": row["top_rescue"],
                    "reason": reason,
                    "needs_review": needs_review,
                    "summary": row["baseline_summary"][:300],
                }
                output["changed_cases"].append(case)
                if needs_review:
                    output["manual_review_cases"].append(case)
        metrics = confusion(y_true, y_pred)
        metrics["changed"] = changed
        metrics["manual_review"] = manual_review
        output["splits"][split] = metrics
    output["changed_cases"] = sorted(
        output["changed_cases"],
        key=lambda row: (row["needs_review"], abs(row["negative_score"] - row["rescue_score"]), row["negative_score"]),
        reverse=True,
    )[:200]
    output["manual_review_cases"] = output["manual_review_cases"][:200]
    return output


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_cases_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = [
        "indicator",
        "refno",
        "resultno",
        "target",
        "baseline",
        "pred",
        "gold",
        "baseline_error_type",
        "negative_score",
        "rescue_score",
        "top_negative",
        "top_rescue",
        "reason",
        "needs_review",
        "summary",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def build_report(
    output_dir: Path,
    embedding_provider: str,
    records: list[dict[str, Any]],
    results: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# R5/R7 Latent-Guided Exploration Report")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Embedding backend: `{embedding_provider}`")
    lines.append(f"Records: {len(records)}")
    lines.append("")
    lines.append("## Baseline")
    for indicator in ("R5", "R7"):
        subset = [row for row in records if row["indicator"] == indicator]
        metrics = confusion([row["gold"] for row in subset], [row["baseline"] for row in subset])
        lines.append(
            f"- {indicator}: accuracy={metrics['accuracy']:.4f}, macro_f1={metrics['macro_f1']:.4f}, "
            f"f1_0={metrics['f1_0']:.4f}, fp={metrics['fp']}, fn={metrics['fn']}"
        )
    lines.append("")
    lines.append("## Methods")
    for indicator, methods in results["by_indicator"].items():
        lines.append(f"### {indicator}")
        for method, payload in methods.items():
            if method == "baseline":
                all_metrics = payload["splits"]["all"]
                lines.append(
                    f"- `baseline` all: accuracy={all_metrics['accuracy']:.4f}, "
                    f"macro_f1={all_metrics['macro_f1']:.4f}, f1_0={all_metrics['f1_0']:.4f}, "
                    f"fp={all_metrics['fp']}, fn={all_metrics['fn']}"
                )
                continue
            test = payload["splits"]["test"]
            all_metrics = payload["splits"]["all"]
            lines.append(
                f"- `{method}` test: accuracy={test['accuracy']:.4f}, macro_f1={test['macro_f1']:.4f}, "
                f"f1_0={test['f1_0']:.4f}, fp={test['fp']}, fn={test['fn']}, changed={test['changed']}, "
                f"manual_review={test['manual_review']}"
            )
            lines.append(
                f"  all: accuracy={all_metrics['accuracy']:.4f}, macro_f1={all_metrics['macro_f1']:.4f}, "
                f"f1_0={all_metrics['f1_0']:.4f}, fp={all_metrics['fp']}, fn={all_metrics['fn']}, "
                f"changed={all_metrics['changed']}, manual_review={all_metrics['manual_review']}"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "The experiments are lightweight reproductions of retrieval correction and conflict-aware reasoning ideas. "
        "They should be treated as research probes, not replacements for the calibrated production rules."
    )
    lines.append(
        "A promising paper/patent direction is to use the generated changed/manual-review cases as expert-feedback seeds, "
        "then iteratively convert validated cases into rule cards and latent prototypes."
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r5-html", type=Path, default=ROOT / "outputs/r5_07050954_9559.html")
    parser.add_argument("--r7-html", type=Path, default=ROOT / "outputs/r7_07050954_8966.html")
    parser.add_argument("--r5-workbook", type=Path, default=ROOT / "outputs/r5_07050954_9559.xlsx")
    parser.add_argument("--r7-workbook", type=Path, default=ROOT / "outputs/r7_07050954_8966.xlsx")
    parser.add_argument("--page-cache", type=Path, default=ROOT / "outputs/r357_latest_full_07050954/r3_page_source_cache.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--embedding-provider", choices=["hash", "qwen", "auto"], default="hash")
    parser.add_argument("--max-records", type=int, default=0, help="Debug limit; 0 means all records.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or DEFAULT_OUTPUT / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    r5_payload = extract_html_payload(args.r5_html)
    r7_payload = extract_html_payload(args.r7_html)
    refnos = {
        as_text(row.get("refno"))
        for payload in (r5_payload, r7_payload)
        for row in (payload.get("rows") or [])
        if as_text(row.get("refno"))
    }
    source_cache = load_source_cache(args.page_cache, refnos)
    records = []
    records.extend(build_records("R5", args.r5_html, args.r5_workbook, source_cache))
    records.extend(build_records("R7", args.r7_html, args.r7_workbook, source_cache))
    if args.max_records:
        records = records[: args.max_records]
    for row in records:
        row["split"] = split_name(row["refno"])

    prototype_texts = [f"{proto.title}. {proto.text}. Patterns: {'; '.join(proto.patterns)}" for proto in PROTOTYPES]
    record_texts = [row["compact_text"] for row in records]
    embeddings, provider_used = embed_texts(prototype_texts + record_texts, args.embedding_provider)
    proto_vectors = embeddings[: len(PROTOTYPES)]
    record_vectors = embeddings[len(PROTOTYPES) :]

    for row_index, row in enumerate(records):
        candidates = [
            (proto, float(np.dot(record_vectors[row_index], proto_vectors[proto_index])))
            for proto_index, proto in enumerate(PROTOTYPES)
            if proto.indicator == row["indicator"]
        ]
        negative = sorted((item for item in candidates if item[0].polarity == "negative"), key=lambda item: item[1], reverse=True)
        rescue = sorted((item for item in candidates if item[0].polarity == "rescue"), key=lambda item: item[1], reverse=True)
        row["negative_score"] = negative[0][1] if negative else 0.0
        row["rescue_score"] = rescue[0][1] if rescue else 0.0
        row["top_negative"] = negative[0][0].proto_id if negative else ""
        row["top_rescue"] = rescue[0][0].proto_id if rescue else ""
        if row["indicator"] == "R5":
            pattern_negative = pattern_score(row["compact_text"], R5_NEGATIVE_PATTERNS)
            pattern_rescue = pattern_score(row["compact_text"], R5_RESCUE_PATTERNS)
        else:
            pattern_negative = pattern_score(row["compact_text"], R7_NEGATIVE_PATTERNS)
            pattern_rescue = pattern_score(row["compact_text"], R7_RESCUE_PATTERNS)
        row["pattern_negative_score"] = pattern_negative
        row["pattern_rescue_score"] = pattern_rescue
        row["hybrid_negative_score"] = max(row["negative_score"], pattern_negative)
        row["hybrid_rescue_score"] = max(row["rescue_score"], pattern_rescue)

    contrastive_tokens = apply_contrastive_scores(records)

    result_payload: dict[str, Any] = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "embedding_provider": provider_used,
        "inputs": {
            "r5_html": str(args.r5_html),
            "r7_html": str(args.r7_html),
            "r5_workbook": str(args.r5_workbook),
            "r7_workbook": str(args.r7_workbook),
            "page_cache": str(args.page_cache),
        },
        "contrastive_top_tokens": {
            indicator: [{"token": token, "score": score} for token, score in tokens]
            for indicator, tokens in contrastive_tokens.items()
        },
        "by_indicator": {},
    }
    for indicator in ("R5", "R7"):
        rows = [row for row in records if row["indicator"] == indicator]
        result_payload["by_indicator"][indicator] = {}
        baseline_metrics = {}
        for split in ("dev", "test", "all"):
            selected = rows if split == "all" else [row for row in rows if row["split"] == split]
            baseline_metrics[split] = confusion([row["gold"] for row in selected], [row["baseline"] for row in selected])
        result_payload["by_indicator"][indicator]["baseline"] = {"splits": baseline_metrics}
        for method in (
            "prototype_only",
            "crag_correction",
            "selfrag_abstain",
            "pattern_crag",
            "hybrid_crag",
            "contrastive_crag",
            "masked_contrastive_crag",
        ):
            params = tune_method(rows, method)
            result_payload["by_indicator"][indicator][method] = evaluate_method(rows, method, params)

    compact_records = [
        {key: value for key, value in row.items() if key != "compact_text"}
        for row in records
    ]
    write_jsonl(output_dir / "dataset_compact.jsonl", compact_records)
    (output_dir / "experiment_results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    all_cases: list[dict[str, Any]] = []
    for methods in result_payload["by_indicator"].values():
        for method_name, payload in methods.items():
            if method_name == "baseline":
                continue
            for case in payload.get("changed_cases") or []:
                case = dict(case)
                case["method"] = method_name
                all_cases.append(case)
    write_cases_csv(output_dir / "changed_or_review_cases.csv", all_cases)
    build_report(output_dir, provider_used, records, result_payload)
    print(f"Wrote latent-guided R5/R7 exploration to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
