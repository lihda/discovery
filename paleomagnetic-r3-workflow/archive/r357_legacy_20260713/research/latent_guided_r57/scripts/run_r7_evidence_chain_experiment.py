#!/usr/bin/env python3
"""R7 evidence-chain experiment inspired by Self-RAG, CRAG, and GraphRAG.

This is a deeper research probe than run_r57_latent_exploration.py. It keeps
the existing deterministic R7 output as a baseline, then constructs an
explicit evidence graph for each RESULTNO:

source snippets -> evidence nodes -> supporting/conflicting edges
-> retrieval-quality gate -> score/manual-review decision.

The goal is not to replace the production workflow, but to test whether the
paper ideas can be adapted to paleomagnetic R7 judgement.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import importlib.util
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
BASE_SCRIPT = ROOT / "research/latent_guided_r57/scripts/run_r57_latent_exploration.py"
DEFAULT_OUTPUT = ROOT / "research/latent_guided_r57/outputs"


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("r57_latent_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Unable to import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


@dataclass(frozen=True)
class EvidencePattern:
    kind: str
    polarity: str
    label: str
    pattern: str
    weight: float
    selfrag_reflection: str


PATTERNS: tuple[EvidencePattern, ...] = (
    EvidencePattern(
        "younger_pole_similarity",
        "negative",
        "pole matches younger reference or APWP segment",
        r"(?:paleopole|palaeopole|pole|poles).{0,220}(?:younger|Bushveld|APWP|apparent polar wander|reference)|"
        r"(?:younger|Bushveld|APWP|apparent polar wander).{0,220}(?:paleopole|palaeopole|pole|poles)|"
        r"does not match contemporaneous poles|do not correspond to the actual paleogeography",
        2.7,
        "retrieved evidence directly challenges the age-consistency of the pole",
    ),
    EvidencePattern(
        "remagnetization_overprint",
        "negative",
        "explicit remagnetization, overprint, or secondary component",
        r"remagneti[sz](?:ed|ation)|overprint(?:ed|ing)?|secondary (?:component|magneti[sz]ation|remanence)|"
        r"present[- ]day overprint|recent overprint|viscous remanent magneti[sz]ation",
        2.5,
        "retrieved evidence directly states a younger or secondary magnetic component",
    ),
    EvidencePattern(
        "age_reset_event",
        "negative",
        "impact, hydrothermal, metamorphic, or thermal resetting",
        r"impact (?:event|structure|melt|dykes?|dikes?)|pseudotachylite|hydrothermal|metamorphic resetting|"
        r"therm(?:al)? resetting|reset(?:ting)? of (?:the )?magneti[sz]ation|reheating",
        2.1,
        "retrieved evidence indicates a plausible remanence reset mechanism",
    ),
    EvidencePattern(
        "age_ambiguity",
        "negative",
        "multi-phase age ambiguity or younger component ambiguity",
        r"age spread|older and younger ages|multi[- ]phase|multiple intrusive episodes|"
        r"(?:component|components) [A-Z](?: and [A-Z])?.{0,160}(?:younger|overprint|secondary)",
        1.8,
        "retrieved evidence creates ambiguity in the component age",
    ),
    EvidencePattern(
        "author_rejection",
        "negative",
        "authors reject or caution against using the pole",
        r"should not be used|not reliable|unreliable|rejected|questionable|dubious|"
        r"do not use|cannot be considered|must be considered with caution",
        1.7,
        "retrieved evidence includes author-level rejection or caution",
    ),
    EvidencePattern(
        "primary_origin_rescue",
        "rescue",
        "explicit primary origin or primary remanence",
        r"primary (?:origin|remanence|magneti[sz]ation)|acquired (?:during|at) (?:cooling|emplacement|deposition)|"
        r"syn[- ]?depositional|primary ChRM",
        1.8,
        "retrieved evidence supports primary origin",
    ),
    EvidencePattern(
        "field_test_rescue",
        "rescue",
        "positive field test supports primary magnetization",
        r"positive (?:fold|reversal|baked contact|contact|conglomerate) test|"
        r"fold test.{0,100}(?:positive|passed)|reversal test.{0,100}(?:positive|passed)|"
        r"baked contact test",
        2.2,
        "retrieved evidence supports the pole through a field test",
    ),
    EvidencePattern(
        "independent_age_rescue",
        "rescue",
        "independent age constraint supports the target component",
        r"U[- ]Pb|baddeleyite|zircon|40Ar/39Ar|Ar[- ]Ar|K[- ]Ar|radiometric age|"
        r"well dated|independent age|precise age",
        1.2,
        "retrieved evidence provides independent age constraints",
    ),
    EvidencePattern(
        "contemporaneous_match_rescue",
        "rescue",
        "pole agrees with contemporaneous reference poles",
        r"(?:consistent|agrees?|matches?|similar) .{0,160}(?:contemporaneous|coeval|same age|reference) .{0,80}(?:pole|poles|APWP)|"
        r"(?:pole|poles).{0,120}(?:consistent|agree|match).{0,120}(?:contemporaneous|coeval|same age)",
        1.8,
        "retrieved evidence supports age consistency with contemporaneous poles",
    ),
)

ANCHOR_PATTERNS = (
    r"\bChRM\b|characteristic remanent magneti[sz]ation|component [A-Z]\b|high[- ]temperature component",
    r"\bpaleopole\b|\bpalaeopole\b|\bpole position\b|\bAPWP\b|apparent polar wander",
    r"magneti[sz]ation age|remanence age|age of magneti[sz]ation",
)

GENERIC_TARGET_TOKENS = {
    "formation",
    "group",
    "suite",
    "complex",
    "dykes",
    "dikes",
    "dyke",
    "dike",
    "swarm",
    "basalts",
    "basalt",
    "sediments",
    "redbeds",
    "red",
    "beds",
    "granite",
    "granites",
    "gabbro",
    "limestone",
    "sandstone",
    "volcanics",
    "volcanic",
    "intrusives",
    "combined",
}


def split_pages(source_text: str) -> list[dict[str, Any]]:
    pages: list[dict[str, Any]] = []
    matches = list(re.finditer(r"\[page\s+(\d+)\]", source_text, flags=re.I))
    if not matches:
        return [{"page": 0, "text": source_text}]
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source_text)
        pages.append({"page": int(match.group(1)), "text": source_text[start:end]})
    return pages


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def target_tokens(target: str) -> set[str]:
    tokens = {
        token.lower()
        for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]{2,}", target)
    }
    return {token for token in tokens if token.lower() not in GENERIC_TARGET_TOKENS}


def has_anchor(text: str, target: str) -> tuple[bool, list[str]]:
    labels: list[str] = []
    lower = text.lower()
    target_hits = [token for token in sorted(target_tokens(target)) if token in lower]
    if target_hits:
        labels.append("target:" + ",".join(target_hits[:3]))
    for index, pattern in enumerate(ANCHOR_PATTERNS, start=1):
        if re.search(pattern, text, flags=re.I | re.S):
            labels.append(f"chain_anchor:{index}")
    return bool(labels), labels


def snippet_window(text: str, start: int, end: int, radius: int = 420) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    return clean(text[left:right])[:1200]


def extract_evidence_nodes(record: dict[str, Any], source_text: str) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    pages = split_pages(source_text)
    for page in pages:
        page_no = page["page"]
        text = page["text"]
        for pattern in PATTERNS:
            for match in re.finditer(pattern.pattern, text, flags=re.I | re.S):
                key = (pattern.kind, page_no, match.start() // 180)
                if key in seen:
                    continue
                seen.add(key)
                quote = snippet_window(text, match.start(), match.end())
                anchored, anchor_labels = has_anchor(quote, record["target"])
                anchor_bonus = 0.7 if anchored else 0.0
                node_score = pattern.weight + anchor_bonus
                if record.get("component_aware_context") and pattern.kind == "remagnetization_overprint":
                    node_score *= 0.35
                    anchor_labels = [*anchor_labels, "component_context:overprint_target"]
                nodes.append(
                    {
                        "id": f"n{len(nodes):03d}",
                        "kind": pattern.kind,
                        "polarity": pattern.polarity,
                        "label": pattern.label,
                        "page": page_no,
                        "score": round(node_score, 3),
                        "base_weight": pattern.weight,
                        "anchored": anchored,
                        "anchor_labels": anchor_labels,
                        "selfrag_reflection": pattern.selfrag_reflection,
                        "quote": quote,
                    }
                )
    return sorted(nodes, key=lambda node: node["score"], reverse=True)


def build_edges(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    for i, left in enumerate(nodes):
        for right in nodes[i + 1 :]:
            if left["page"] != right["page"]:
                continue
            relation = ""
            if left["polarity"] == right["polarity"]:
                relation = "co_supports"
            else:
                relation = "conflicts_with"
            if left["anchored"] or right["anchored"]:
                relation += "_anchored"
            edges.append(
                {
                    "source": left["id"],
                    "target": right["id"],
                    "relation": relation,
                    "page": left["page"],
                    "weight": round((left["score"] + right["score"]) / 2, 3),
                }
            )
    return edges


def graph_scores(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], source_available: bool) -> dict[str, Any]:
    negative = [node for node in nodes if node["polarity"] == "negative"]
    rescue = [node for node in nodes if node["polarity"] == "rescue"]
    negative_kinds = {node["kind"] for node in negative}
    rescue_kinds = {node["kind"] for node in rescue}
    anchored_negative = [node for node in negative if node["anchored"]]
    anchored_rescue = [node for node in rescue if node["anchored"]]
    conflict_edges = [edge for edge in edges if edge["relation"].startswith("conflicts_with")]
    leakage_nodes = [
        node
        for node in negative
        if not node["anchored"]
        and re.search(
            r"reference pole|published poles|other poles|compilation|table \d|APWP|apparent polar wander path|"
            r"poles from|pole positions from|for comparison",
            node["quote"],
            flags=re.I,
        )
    ]

    negative_score = sum(node["score"] for node in negative[:5])
    rescue_score = sum(node["score"] for node in rescue[:5])
    anchored_negative_score = sum(node["score"] for node in anchored_negative[:5])
    anchored_rescue_score = sum(node["score"] for node in anchored_rescue[:5])
    explicit_unanchored = [
        node
        for node in negative
        if not node["anchored"] and node["kind"] in {"remagnetization_overprint", "age_reset_event"}
    ]
    graph_bonus = 0.35 * len({edge["page"] for edge in edges if edge["relation"].startswith("co_supports")})
    anchor_bonus = 0.8 if anchored_negative else 0.0
    retrieval_quality = 0.0
    if source_available:
        retrieval_quality = min(1.0, 0.18 * len(nodes) + 0.20 * len({node["page"] for node in nodes}) + 0.18 * len(conflict_edges))
        if anchored_negative or anchored_rescue:
            retrieval_quality += 0.18
        retrieval_quality = min(1.0, retrieval_quality)

    chain_score = negative_score + graph_bonus + anchor_bonus - 0.55 * rescue_score
    targeted_chain_score = (
        anchored_negative_score
        + 0.35 * sum(node["score"] for node in explicit_unanchored[:2])
        + 0.20 * len([edge for edge in edges if "anchored" in edge["relation"] and edge["relation"].startswith("co_supports")])
        - 0.65 * anchored_rescue_score
        - 1.10 * len(leakage_nodes)
    )
    conflict_score = min(1.0, 0.18 * len(conflict_edges) + 0.12 * min(len(negative_kinds), 3) + 0.12 * min(len(rescue_kinds), 3))
    reflection = {
        "retrieve": source_available,
        "relevant": retrieval_quality >= 0.25,
        "supported_negative": bool(negative and (anchored_negative or len(negative_kinds) >= 2)),
        "supported_rescue": bool(rescue and (anchored_rescue or len(rescue_kinds) >= 2)),
        "conflict": bool(conflict_edges and negative and rescue),
    }
    return {
        "negative_score": round(negative_score, 4),
        "rescue_score": round(rescue_score, 4),
        "chain_score": round(chain_score, 4),
        "targeted_chain_score": round(targeted_chain_score, 4),
        "retrieval_quality": round(retrieval_quality, 4),
        "conflict_score": round(conflict_score, 4),
        "negative_kinds": sorted(negative_kinds),
        "rescue_kinds": sorted(rescue_kinds),
        "anchored_negative": len(anchored_negative),
        "anchored_rescue": len(anchored_rescue),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "leakage_nodes": len(leakage_nodes),
        "reflection": reflection,
    }


def decision_for(record: dict[str, Any], scores: dict[str, Any], params: dict[str, float], method: str) -> tuple[int, bool, str]:
    baseline = int(record["baseline"])
    retrieval_quality = scores["retrieval_quality"]
    chain_score = scores["chain_score"]
    conflict_score = scores["conflict_score"]
    reflection = scores["reflection"]
    targeted_chain_score = scores.get("targeted_chain_score", chain_score)

    if method == "graph_chain":
        if retrieval_quality >= params["quality"] and chain_score >= params["chain"]:
            return 0, False, "graph_chain_negative"
        return baseline, False, "graph_chain_preserve_baseline"

    if method == "targeted_graph_chain":
        if retrieval_quality >= params["quality"] and targeted_chain_score >= params["chain"]:
            return 0, False, "targeted_graph_chain_negative"
        if chain_score >= params["chain"] and targeted_chain_score < params["chain"] and scores["leakage_nodes"]:
            return baseline, True, "targeted_graph_possible_context_leakage"
        return baseline, False, "targeted_graph_preserve_baseline"

    if method == "crag_graph":
        if retrieval_quality < params["quality"]:
            return baseline, True, "crag_low_quality_retrieval"
        if baseline == 1 and chain_score >= params["chain"]:
            return 0, False, "crag_corrected_to_zero"
        if baseline == 0 and scores["rescue_score"] >= params["rescue"] and chain_score < params["chain"]:
            return 1, False, "crag_rescue_to_one"
        return baseline, False, "crag_preserve_baseline"

    if method in {"targeted_crag_graph", "targeted_conservative_crag"}:
        if retrieval_quality < params["quality"]:
            return baseline, True, "targeted_crag_low_quality_retrieval"
        if baseline == 1 and targeted_chain_score >= params["chain"]:
            return 0, False, "targeted_crag_corrected_to_zero"
        if baseline == 1 and chain_score >= params["chain"] and scores["leakage_nodes"]:
            return baseline, True, "targeted_crag_unanchored_counterevidence"
        if baseline == 0 and scores["rescue_score"] >= params["rescue"] and targeted_chain_score < params["chain"]:
            return 1, False, "targeted_crag_rescue_to_one"
        return baseline, False, "targeted_crag_preserve_baseline"

    if method == "selfrag_graph":
        if retrieval_quality < params["quality"]:
            return baseline, True, "selfrag_retrieve_more"
        if reflection["conflict"] and conflict_score >= params["conflict"]:
            return baseline, True, "selfrag_conflict_review"
        if reflection["supported_negative"] and chain_score >= params["chain"]:
            return 0, False, "selfrag_supported_negative"
        if reflection["supported_rescue"] and baseline == 0 and chain_score < params["chain"]:
            return 1, False, "selfrag_supported_rescue"
        return baseline, False, "selfrag_no_change"

    raise ValueError(method)


def confusion(y_true: list[int], y_pred: list[int]) -> dict[str, Any]:
    return BASE.confusion(y_true, y_pred)


def objective(metrics: dict[str, Any]) -> float:
    return 0.50 * metrics["macro_f1"] + 0.35 * metrics["f1_0"] + 0.15 * metrics["accuracy"]


def quantile_grid(values: list[float], fallback: tuple[float, float], steps: int = 16) -> list[float]:
    if not values:
        return list(np.linspace(fallback[0], fallback[1], steps))
    array = np.asarray(values, dtype=np.float32)
    if float(array.min()) == float(array.max()):
        return [float(array.min())]
    grid = np.quantile(array, np.linspace(0.05, 0.95, steps))
    grid = np.unique(np.concatenate([grid, np.asarray([0.0, float(array.mean())])]))
    return [float(item) for item in grid]


def tune(records: list[dict[str, Any]], method: str) -> dict[str, float]:
    dev = [record for record in records if record["split"] == "dev"]
    chain_key = "targeted_chain_score" if method.startswith("targeted_") else "chain_score"
    baseline_dev = confusion([record["gold"] for record in dev], [record["baseline"] for record in dev])
    chain_grid = quantile_grid([record["scores"][chain_key] for record in dev], (-1.0, 8.0), 18)
    quality_grid = quantile_grid([record["scores"]["retrieval_quality"] for record in dev], (0.05, 0.85), 12)
    rescue_grid = quantile_grid([record["scores"]["rescue_score"] for record in dev], (0.0, 8.0), 10)
    conflict_grid = quantile_grid([record["scores"]["conflict_score"] for record in dev], (0.1, 0.9), 8)
    best_score = -1.0
    best_params: dict[str, float] = {}
    for chain in chain_grid:
        for quality in quality_grid:
            for rescue in rescue_grid:
                for conflict in conflict_grid:
                    params = {
                        "chain": float(chain),
                        "quality": float(quality),
                        "rescue": float(rescue),
                        "conflict": float(conflict),
                    }
                    y_true: list[int] = []
                    y_pred: list[int] = []
                    for record in dev:
                        pred, _, _ = decision_for(record, record["scores"], params, method)
                        y_true.append(record["gold"])
                        y_pred.append(pred)
                    metrics = confusion(y_true, y_pred)
                    score = objective(metrics)
                    if method == "targeted_conservative_crag":
                        accuracy_floor = baseline_dev["accuracy"] - 0.02
                        if metrics["accuracy"] < accuracy_floor:
                            score -= 2.0 * (accuracy_floor - metrics["accuracy"])
                        if metrics["fn"] > baseline_dev["fn"] + 25:
                            score -= 0.015 * (metrics["fn"] - baseline_dev["fn"] - 25)
                    if score > best_score:
                        best_score = score
                        best_params = params
    return best_params


def evaluate(records: list[dict[str, Any]], method: str, params: dict[str, float]) -> dict[str, Any]:
    payload: dict[str, Any] = {"params": params, "splits": {}, "cases": []}
    for split in ("dev", "test", "all"):
        selected = records if split == "all" else [record for record in records if record["split"] == split]
        y_true: list[int] = []
        y_pred: list[int] = []
        changed = 0
        review = 0
        for record in selected:
            pred, needs_review, reason = decision_for(record, record["scores"], params, method)
            y_true.append(record["gold"])
            y_pred.append(pred)
            changed += int(pred != record["baseline"])
            review += int(needs_review)
            if split == "all" and (pred != record["baseline"] or needs_review):
                top_nodes = record["nodes"][:4]
                payload["cases"].append(
                    {
                        "refno": record["refno"],
                        "resultno": record["resultno"],
                        "target": record["target"],
                        "baseline": record["baseline"],
                        "pred": pred,
                        "gold": record["gold"],
                        "error_type": record["error_type"],
                        "reason": reason,
                        "needs_review": needs_review,
                        "chain_score": record["scores"]["chain_score"],
                        "targeted_chain_score": record["scores"]["targeted_chain_score"],
                        "retrieval_quality": record["scores"]["retrieval_quality"],
                        "conflict_score": record["scores"]["conflict_score"],
                        "leakage_nodes": record["scores"]["leakage_nodes"],
                        "negative_kinds": "|".join(record["scores"]["negative_kinds"]),
                        "rescue_kinds": "|".join(record["scores"]["rescue_kinds"]),
                        "top_node": top_nodes[0]["label"] if top_nodes else "",
                        "top_quote": top_nodes[0]["quote"][:500] if top_nodes else "",
                    }
                )
        metrics = confusion(y_true, y_pred)
        metrics["changed"] = changed
        metrics["manual_review"] = review
        payload["splits"][split] = metrics
    payload["cases"] = sorted(
        payload["cases"],
        key=lambda case: (case["needs_review"], case["chain_score"], case["conflict_score"]),
        reverse=True,
    )[:250]
    return payload


def load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    html_payload = BASE.extract_html_payload(args.r7_html)
    workbook_payloads = BASE.load_workbook_payloads(args.r7_workbook, "R7")
    refnos = {BASE.as_text(row.get("refno")) for row in html_payload.get("rows") or [] if BASE.as_text(row.get("refno"))}
    source_cache = BASE.load_source_cache(args.page_cache, refnos)
    records: list[dict[str, Any]] = []
    for row in html_payload.get("rows") or []:
        resultno = BASE.as_text(row.get("resultno"))
        refno = BASE.as_text(row.get("refno"))
        target = BASE.as_text(row.get("target"))
        baseline = BASE.normalize_score(row.get("score"))
        gold = BASE.normalize_score(row.get("standard_score"))
        if baseline is None or gold is None:
            continue
        payload = workbook_payloads.get(resultno, {})
        metadata = payload.get("source_metadata") if isinstance(payload.get("source_metadata"), dict) else {}
        component_text = " ".join(
            [
                target,
                BASE.as_text(metadata.get("PMAGRESULT.COMPONENT")),
                BASE.as_text(metadata.get("PMAGRESULT.COMMENTS")),
            ]
        )
        component_aware_context = bool(
            re.search(r"\boverprint\b|secondary component|post[- ]?folding|present[- ]day", component_text, flags=re.I)
        )
        source_text = source_cache.get(refno, "")
        nodes = extract_evidence_nodes(
            {
                "target": target,
                "baseline": baseline,
                "gold": gold,
                "component_aware_context": component_aware_context,
            },
            source_text,
        )
        edges = build_edges(nodes)
        scores = graph_scores(nodes, edges, bool(source_text.strip()))
        records.append(
            {
                "indicator": "R7",
                "refno": refno,
                "resultno": resultno,
                "target": target,
                "baseline": int(baseline),
                "gold": int(gold),
                "error_type": BASE.as_text(row.get("error_type")),
                "split": BASE.split_name(refno),
                "source_available": bool(source_text.strip()),
                "source_chars": len(source_text),
                "baseline_summary": BASE.as_text(payload.get("summary") or row.get("summary")),
                "component_aware_context": component_aware_context,
                "nodes": nodes,
                "edges": edges,
                "scores": scores,
            }
        )
    if args.max_records:
        records = records[: args.max_records]
    return records


def write_cases(path: Path, methods: dict[str, Any]) -> None:
    fields = [
        "method",
        "refno",
        "resultno",
        "target",
        "baseline",
        "pred",
        "gold",
        "error_type",
        "reason",
        "needs_review",
        "chain_score",
        "targeted_chain_score",
        "retrieval_quality",
        "conflict_score",
        "leakage_nodes",
        "negative_kinds",
        "rescue_kinds",
        "top_node",
        "top_quote",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, payload in methods.items():
            if method == "baseline":
                continue
            for case in payload["cases"]:
                row = dict(case)
                row["method"] = method
                writer.writerow({field: row.get(field, "") for field in fields})


def write_graph_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    {
                        "refno": record["refno"],
                        "resultno": record["resultno"],
                        "target": record["target"],
                        "baseline": record["baseline"],
                        "gold": record["gold"],
                        "error_type": record["error_type"],
                        "scores": record["scores"],
                        "nodes": record["nodes"][:20],
                        "edges": record["edges"][:80],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def write_report(path: Path, results: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# R7 Evidence-Chain Experiment Report")
    lines.append("")
    lines.append(f"Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    baseline = results["methods"]["baseline"]["splits"]["all"]
    lines.append("## Baseline")
    lines.append(
        f"- R7 baseline: accuracy={baseline['accuracy']:.4f}, macro_f1={baseline['macro_f1']:.4f}, "
        f"f1_0={baseline['f1_0']:.4f}, fp={baseline['fp']}, fn={baseline['fn']}"
    )
    lines.append("")
    lines.append("## Paper-Inspired Variants")
    for method, payload in results["methods"].items():
        if method == "baseline":
            continue
        test = payload["splits"]["test"]
        all_metrics = payload["splits"]["all"]
        lines.append(
            f"- `{method}` test: accuracy={test['accuracy']:.4f}, macro_f1={test['macro_f1']:.4f}, "
            f"f1_0={test['f1_0']:.4f}, fp={test['fp']}, fn={test['fn']}, "
            f"changed={test['changed']}, review={test['manual_review']}"
        )
        lines.append(
            f"  all: accuracy={all_metrics['accuracy']:.4f}, macro_f1={all_metrics['macro_f1']:.4f}, "
            f"f1_0={all_metrics['f1_0']:.4f}, fp={all_metrics['fp']}, fn={all_metrics['fn']}, "
            f"changed={all_metrics['changed']}, review={all_metrics['manual_review']}"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "This run operationalizes GraphRAG as explicit R7 evidence nodes/edges, CRAG as a retrieval-quality gate, "
        "and Self-RAG as reflection flags for retrieve/relevant/supported/conflict."
    )
    lines.append(
        "If test-set f1_0 improves while accuracy remains acceptable, the evidence-chain idea is a candidate method. "
        "If it only improves all-set metrics, expert-reviewed graph nodes should be used to build stronger rule cards before claiming generalization."
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r7-html", type=Path, default=ROOT / "outputs/r7_07050954_8966.html")
    parser.add_argument("--r7-workbook", type=Path, default=ROOT / "outputs/r7_07050954_8966.xlsx")
    parser.add_argument("--page-cache", type=Path, default=ROOT / "outputs/r357_latest_full_07050954/r3_page_source_cache.jsonl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-records", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or DEFAULT_OUTPUT / f"r7_chain_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_records(args)

    methods: dict[str, Any] = {}
    methods["baseline"] = {
        "splits": {
            split: confusion(
                [record["gold"] for record in (records if split == "all" else [r for r in records if r["split"] == split])],
                [record["baseline"] for record in (records if split == "all" else [r for r in records if r["split"] == split])],
            )
            for split in ("dev", "test", "all")
        }
    }
    for method in (
        "graph_chain",
        "targeted_graph_chain",
        "crag_graph",
        "targeted_crag_graph",
        "targeted_conservative_crag",
        "selfrag_graph",
    ):
        params = tune(records, method)
        methods[method] = evaluate(records, method, params)

    results = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "records": len(records),
        "inputs": {
            "r7_html": str(args.r7_html),
            "r7_workbook": str(args.r7_workbook),
            "page_cache": str(args.page_cache),
        },
        "methods": methods,
    }
    (output_dir / "experiment_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_cases(output_dir / "cases.csv", methods)
    write_graph_jsonl(output_dir / "evidence_graphs.jsonl", records)
    write_report(output_dir / "REPORT.md", results)
    print(f"Wrote R7 evidence-chain experiment to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
