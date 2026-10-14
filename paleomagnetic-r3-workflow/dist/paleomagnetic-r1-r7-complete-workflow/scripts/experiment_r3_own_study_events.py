#!/usr/bin/env python3
"""Grouped PDF-only R3 ablation for source-derived own-study episodes.

The feature candidate is built and hashed before training R3 labels are opened.
Runtime-recomputable inputs are restricted to frozen-training identity fields,
original-paper page-cache text, and frozen taxonomy/positive-KB patterns.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as baseline  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


DEFAULT_IDENTITY = ROOT / "tmp/r3_distilled_training_preflight/identity.xlsx"
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_PAGE_CACHE = ROOT / "tmp/mineru_cache_all_hard_risk.jsonl"
DEFAULT_PDF_DIR = ROOT / "data/raw_pdfs"
DEFAULT_TAXONOMY = ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json"
DEFAULT_POSITIVE_KB = ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json"
DEFAULT_OUTPUT = ROOT / "tmp/r3_own_study_events/report.json"
DEFAULT_FREEZE = ROOT / "tmp/r3_own_study_events/feature_candidate_freeze.json"
DEFAULT_OOF = ROOT / "tmp/r3_own_study_events/oof_probabilities.npz"

N_SPLITS = 5
RANDOM_STATE = 357
SPACE_RE = re.compile(r"\s+")
YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b", re.I)
CITATION_RE = re.compile(
    r"\([^)]{0,100}\b(?:18|19|20)\d{2}[a-z]?[^)]*\)|"
    r"\b[A-Z][A-Za-z'-]+\s+(?:et\s+al\.?|&\s+[A-Z][A-Za-z'-]+)\s*[,;(]?\s*(?:18|19|20)\d{2}|"
    r"\[[0-9,; -]{1,30}\]",
    re.I,
)
FIRST_PERSON_RE = re.compile(
    r"\b(?:we|our|in\s+this\s+(?:study|paper|work)|the\s+present\s+study)\b.{0,180}"
    r"\b(?:measur|perform|conduct|carry|carried|obtain|determin|analys|analyz|investigat|"
    r"subject|heat|cool|apply|acquir|demagneti[sz]|show|indicat|reveal|yield)",
    re.I | re.S,
)
PASSIVE_EXECUTION_RE = re.compile(
    r"\b(?:samples?|specimens?|measurements?|experiments?|analyses?|curves?|loops?|"
    r"susceptibilit(?:y|ies)|remanence|magneti[sz]ation)\b.{0,100}"
    r"\b(?:were|was|have\s+been|has\s+been)\s+"
    r"(?:measured|performed|conducted|carried\s+out|obtained|determined|analysed|analyzed|"
    r"investigated|subjected|heated|cooled|applied|acquired|demagneti[sz]ed|calculated)",
    re.I | re.S,
)
PROCEDURE_RE = re.compile(
    r"\b(?:measur(?:e|ed|ement|ements|ing)|perform(?:ed|ing)?|conduct(?:ed|ing)?|"
    r"carried\s+out|subjected\s+to|heated|cooled|applied\s+(?:field|fields)|"
    r"acquisition\s+(?:curve|curves|experiment|experiments)|using\s+(?:a|an|the)|"
    r"analys(?:ed|is|es)|analyz(?:ed|ing)|determined)\b",
    re.I,
)
RESULT_RE = re.compile(
    r"\b(?:show(?:s|ed)?|indicat(?:e|es|ed)|reveal(?:s|ed)?|yield(?:s|ed)?|display(?:s|ed)?|"
    r"suggest(?:s|ed)?|demonstrat(?:e|es|ed)|characteri[sz](?:e|es|ed)|is\s+consistent|"
    r"are\s+consistent|predominant|dominant|carrier|carried\s+by)\b",
    re.I,
)
QUANTITATIVE_RE = re.compile(
    r"(?:\b\d+(?:\.\d+)?\s*(?:mT|T|kOe|Oe|A\s*/\s*m|emu|K|°?\s*C)\b|"
    r"\b(?:Tc|Hc|Hcr|Bc|Bcr|Mrs|Ms|Mr|Js|Jrs)\s*[=:/]\s*\d)",
    re.I,
)
FIGURE_TABLE_RE = re.compile(r"\b(?:fig(?:ure)?\.?|table)\s*\d+[a-z]?\b", re.I)
SAMPLE_RE = re.compile(
    r"\b(?:samples?|specimens?|sites?|cores?|powders?|chips?|whole[- ]rock|separate|fraction)\b",
    re.I,
)
EXTERNAL_RE = re.compile(
    r"\b(?:according\s+to|reported\s+by|published\s+by|described\s+by|data\s+(?:of|from)|"
    r"previous(?:ly)?\s+(?:study|studies|work|reported)|earlier\s+(?:study|studies|work)|"
    r"as\s+(?:shown|found|noted|suggested)\s+by|literature\s+(?:data|values?))\b",
    re.I,
)
DEFINITION_RE = re.compile(
    r"\b(?:is\s+defined\s+as|are\s+defined\s+as|is\s+known\s+as|generally|typically|"
    r"commonly|in\s+general|can\s+be\s+used\s+to|may\s+be\s+used\s+to)\b",
    re.I,
)
FUTURE_RE = re.compile(
    r"\b(?:future\s+(?:work|study|studies)|will\s+be\s+(?:measured|performed|investigated|analysed|analyzed)|"
    r"remains?\s+to\s+be|should\s+be\s+(?:measured|investigated|tested)|is\s+planned)\b",
    re.I,
)
METHOD_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|lack(?:ing|ed|s)?)\b.{0,80}"
    r"\b(?:measurement|experiment|analysis|analyses|test|investigation|data|curve|curves)\b|"
    r"\b(?:measurement|experiment|analysis|analyses|test|investigation)\b.{0,50}"
    r"\b(?:was|were)\s+not\b",
    re.I | re.S,
)

SECTIONS = ("abstract", "introduction", "methods", "results", "discussion", "conclusion", "references", "unknown")
CATEGORIES = ("method", "diagnostic", "carrier", "other")
HEADING_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("references", re.compile(r"^(?:references|bibliography|literature\s+cited)\b", re.I)),
    ("conclusion", re.compile(r"^(?:\d+[. ]*)?(?:conclusions?|summary\s+and\s+conclusions?)\b", re.I)),
    ("discussion", re.compile(r"^(?:\d+[. ]*)?(?:discussion|interpretation|implications?)\b", re.I)),
    ("results", re.compile(r"^(?:\d+[. ]*)?(?:(?:rock[- ]?|palaeo|paleo)?magnetic\s+(?:results?|mineralogy|properties)|results?|observations?)\b", re.I)),
    ("methods", re.compile(r"^(?:\d+[. ]*)?(?:materials?\s+and\s+methods?|methods?|experimental(?:\s+methods?)?|laboratory\s+(?:methods?|procedure|techniques?)|analytical\s+(?:methods?|procedure)|rock[- ]magnetic\s+(?:methods?|measurements?|experiments?)|palaeo(?:magnetic)?\s+(?:methods?|techniques?)|sampling(?:\s+and\s+methods?)?)\b", re.I)),
    ("introduction", re.compile(r"^(?:\d+[. ]*)?(?:introduction|background|geological\s+setting)\b", re.I)),
    ("abstract", re.compile(r"^(?:abstract|summary)\b", re.I)),
)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(value)
    return sha256_bytes(contiguous.view(np.uint8).tobytes())


def load_structured_pages(path: Path, refnos: set[str]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str]]:
    structured: dict[str, list[dict[str, Any]]] = {}
    paper_text: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            refno = clean(row.get("REFNO"))
            if refno not in refnos:
                continue
            if refno in structured:
                raise ValueError(f"duplicate page-cache REFNO {refno} at line {line_number}")
            pages: list[dict[str, Any]] = []
            flattened: list[str] = []
            for index, page in enumerate(row.get("pages", []), start=1):
                if not isinstance(page, Mapping):
                    continue
                text = clean(page.get("text"))
                if not text:
                    continue
                number = clean(page.get("page")) or str(index)
                hints = [clean(item) for item in page.get("section_hints", []) if clean(item)]
                pages.append({"page": number, "text": text, "section_hints": hints})
                flattened.append(f"[[PAGE {number}]]\n{text}")
            structured[refno] = pages
            paper_text[refno] = "\n\n".join(flattened)
    missing = sorted(refnos - structured.keys())
    empty = sorted(refno for refno in refnos if not structured.get(refno))
    if missing or empty:
        raise ValueError(f"missing/empty page cache: missing={missing[:10]}, empty={empty[:10]}")
    return structured, paper_text


def classify_heading(line: str) -> str | None:
    candidate = SPACE_RE.sub(" ", line).strip(" .:;-\t")
    if not candidate or len(candidate) > 130 or len(candidate.split()) > 16:
        return None
    for section, pattern in HEADING_RULES:
        if pattern.search(candidate):
            return section
    return None


def sectionize(pages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current = "abstract"
    segment_id = 0
    for page_index, page in enumerate(pages):
        if current == "abstract" and page_index > 0:
            current = "unknown"
        if current != "references":
            for hint in page.get("section_hints", []):
                detected = classify_heading(clean(hint))
                if detected and detected not in {"abstract", "references"}:
                    current = detected
                    break
        buffer: list[str] = []

        def flush() -> None:
            nonlocal segment_id
            text = SPACE_RE.sub(" ", " ".join(buffer)).strip()
            if text:
                segments.append({
                    "segment_id": segment_id,
                    "page": clean(page.get("page")),
                    "section": current,
                    "text": text,
                })
                segment_id += 1
            buffer.clear()

        for raw_line in clean(page.get("text")).splitlines():
            line = SPACE_RE.sub(" ", raw_line).strip()
            if not line:
                continue
            detected = classify_heading(line)
            if detected:
                flush()
                current = detected
            buffer.append(line)
        flush()
    return segments


def concept_category(concept: Mapping[str, Any]) -> str:
    value = f"{concept.get('category', '')}:{concept.get('concept_id', '')}".casefold()
    if "method" in value:
        return "method"
    if "diagnostic" in value:
        return "diagnostic"
    if "carrier" in value:
        return "carrier"
    return "other"


def target_spans(text: str, rockname: str) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for pattern in baseline.target_patterns(rockname):
        for match in pattern.finditer(text):
            output.append((match.start(), match.end()))
            if len(output) >= 80:
                return output
    return output


def span_distance(start: int, end: int, spans: Sequence[tuple[int, int]]) -> int | None:
    if not spans:
        return None
    return min(max(span_start - end, start - span_end, 0) for span_start, span_end in spans)


def event_signals(context: str, section: str, any_target: bool) -> dict[str, Any]:
    flags = {
        "first_person": bool(FIRST_PERSON_RE.search(context)),
        "passive_execution": bool(PASSIVE_EXECUTION_RE.search(context)),
        "procedure": bool(PROCEDURE_RE.search(context)),
        "result_verb": bool(RESULT_RE.search(context)),
        "quantitative": bool(QUANTITATIVE_RE.search(context)),
        "figure_table": bool(FIGURE_TABLE_RE.search(context)),
        "sample": bool(SAMPLE_RE.search(context)),
        "target": any_target,
        "citation": bool(CITATION_RE.search(context) or YEAR_RE.search(context)),
        "external_attribution": bool(EXTERNAL_RE.search(context)),
        "definition": bool(DEFINITION_RE.search(context)),
        "future": bool(FUTURE_RE.search(context)),
        "method_negation": bool(METHOD_NEGATION_RE.search(context)),
    }
    score = 0
    score += 4 * int(flags["first_person"])
    score += 4 * int(flags["passive_execution"])
    score += 2 * int(flags["procedure"])
    score += 2 * int(flags["result_verb"])
    score += int(flags["quantitative"])
    score += int(flags["figure_table"])
    score += int(flags["sample"])
    score += int(flags["target"])
    score += int(section in {"methods", "results", "conclusion"})
    score += int(section == "abstract" and (flags["procedure"] or flags["result_verb"]))
    score -= 7 * int(section == "references")
    score -= 3 * int(flags["external_attribution"])
    score -= 2 * int(section == "introduction" and not (flags["first_person"] or flags["passive_execution"]))
    score -= int(flags["citation"] and not (flags["first_person"] or flags["passive_execution"]))
    score -= 2 * int(flags["definition"])
    score -= 2 * int(flags["future"])
    score -= 5 * int(flags["method_negation"])
    external = bool(
        section == "references"
        or flags["external_attribution"]
        or (section == "introduction" and flags["citation"] and not flags["first_person"])
        or ((flags["definition"] or flags["future"]) and not (flags["first_person"] or flags["passive_execution"]))
    )
    return {
        **flags,
        "score": score,
        "own_strong": score >= 6 and section != "references" and not flags["method_negation"],
        "own_likely": score >= 3 and section != "references" and not flags["method_negation"],
        "external": external,
    }


def extract_paper_events(
    pages: Sequence[Mapping[str, Any]],
    targets: Mapping[str, str],
    concepts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    segments = sectionize(pages)
    hits: dict[str, dict[int, list[tuple[int, int]]]] = defaultdict(dict)
    for target_key, rockname in targets.items():
        for segment in segments:
            hits[target_key][segment["segment_id"]] = target_spans(segment["text"], rockname)

    events: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    for segment in segments:
        text = segment["text"]
        target_hits = {
            target_key: target_segments.get(segment["segment_id"], [])
            for target_key, target_segments in hits.items()
        }
        for concept in concepts:
            concept_id = clean(concept.get("concept_id"))
            category = concept_category(concept)
            matched = 0
            for pattern in concept["patterns"]:
                for match in pattern.finditer(text):
                    key = (segment["segment_id"], concept_id, match.start() // 8)
                    if key in seen:
                        continue
                    seen.add(key)
                    matched += 1
                    left = max(0, match.start() - 320)
                    right = min(len(text), match.end() + 320)
                    context = text[left:right]
                    distances = {
                        target_key: span_distance(match.start(), match.end(), spans)
                        for target_key, spans in target_hits.items()
                    }
                    nearest = sorted(
                        ((distance, target_key) for target_key, distance in distances.items() if distance is not None)
                    )
                    assigned_target = ""
                    if nearest and nearest[0][0] <= 1200 and (
                        len(nearest) == 1 or nearest[1][0] - nearest[0][0] >= 80
                    ):
                        assigned_target = nearest[0][1]
                    signals = event_signals(context, segment["section"], bool(nearest and nearest[0][0] <= 500))
                    events.append({
                        "segment_id": segment["segment_id"],
                        "page": segment["page"],
                        "section": segment["section"],
                        "start": match.start(),
                        "end": match.end(),
                        "concept_id": concept_id,
                        "category": category,
                        "distances": distances,
                        "assigned_target": assigned_target,
                        **signals,
                    })
                    if matched >= 80:
                        break
                if matched >= 80:
                    break

    episodes: list[dict[str, Any]] = []
    by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        by_segment[event["segment_id"]].append(event)
    for segment_events in by_segment.values():
        ordered = sorted(segment_events, key=lambda item: item["start"])
        clusters: list[list[dict[str, Any]]] = []
        for event in ordered:
            if not clusters or event["start"] - clusters[-1][-1]["end"] > 800:
                clusters.append([event])
            else:
                clusters[-1].append(event)
        for cluster in clusters:
            categories = {event["category"] for event in cluster}
            assigned = Counter(event["assigned_target"] for event in cluster if event["assigned_target"])
            episodes.append({
                "section": cluster[0]["section"],
                "event_count": len(cluster),
                "categories": categories,
                "own_strong": any(event["own_strong"] for event in cluster),
                "own_likely": any(event["own_likely"] for event in cluster),
                "external_only": all(event["external"] for event in cluster),
                "method_chain": "method" in categories and bool(categories & {"diagnostic", "carrier"}),
                "assigned_target": assigned.most_common(1)[0][0] if assigned else "",
            })
    return {"segments": segments, "events": events, "episodes": episodes, "target_hits": hits}


def log_count(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def aggregate_article(events: Sequence[Mapping[str, Any]], episodes: Sequence[Mapping[str, Any]], char_count: int) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(float(value))

    total = len(events)
    strong = sum(bool(event["own_strong"]) for event in events)
    likely = sum(bool(event["own_likely"]) for event in events)
    external = sum(bool(event["external"]) for event in events)
    add("article_event_count_log", log_count(total))
    add("article_own_strong_count_log", log_count(strong))
    add("article_own_likely_count_log", log_count(likely))
    add("article_external_count_log", log_count(external))
    add("article_own_strong_ratio", strong / max(1, total))
    add("article_external_ratio", external / max(1, total))
    add("article_event_density_10k", min(20.0, total * 10000.0 / max(1, char_count)))
    add("article_strong_density_10k", min(20.0, strong * 10000.0 / max(1, char_count)))
    add("article_distinct_concepts_log", log_count(len({event["concept_id"] for event in events})))
    add("article_distinct_own_concepts_log", log_count(len({event["concept_id"] for event in events if event["own_strong"]})))
    add("article_max_event_score", max((event["score"] for event in events), default=-7) / 10.0)
    add("article_positive_score_sum_log", log_count(sum(max(0, int(event["score"])) for event in events)))
    for flag in ("first_person", "passive_execution", "procedure", "result_verb", "quantitative", "figure_table", "sample", "citation", "external_attribution", "definition", "future", "method_negation"):
        count = sum(bool(event[flag]) for event in events)
        add(f"article_{flag}_count_log", log_count(count))
        add(f"article_{flag}_ratio", count / max(1, total))
    for section in SECTIONS:
        selected = [event for event in events if event["section"] == section]
        add(f"article_section_{section}_count_log", log_count(len(selected)))
        add(f"article_section_{section}_strong_log", log_count(sum(bool(event["own_strong"]) for event in selected)))
        add(f"article_section_{section}_external_log", log_count(sum(bool(event["external"]) for event in selected)))
    for category in CATEGORIES:
        selected = [event for event in events if event["category"] == category]
        add(f"article_category_{category}_count_log", log_count(len(selected)))
        add(f"article_category_{category}_strong_log", log_count(sum(bool(event["own_strong"]) for event in selected)))
        add(f"article_category_{category}_external_log", log_count(sum(bool(event["external"]) for event in selected)))
    add("article_episode_count_log", log_count(len(episodes)))
    add("article_strong_episode_count_log", log_count(sum(bool(item["own_strong"]) for item in episodes)))
    add("article_likely_episode_count_log", log_count(sum(bool(item["own_likely"]) for item in episodes)))
    add("article_external_only_episode_count_log", log_count(sum(bool(item["external_only"]) for item in episodes)))
    add("article_method_chain_episode_count_log", log_count(sum(bool(item["method_chain"]) for item in episodes)))
    add("article_strong_method_chain_episode_count_log", log_count(sum(bool(item["method_chain"] and item["own_strong"]) for item in episodes)))
    return values, names


def aggregate_target(
    target_key: str,
    events: Sequence[Mapping[str, Any]],
    episodes: Sequence[Mapping[str, Any]],
    hit_map: Mapping[int, Sequence[tuple[int, int]]],
) -> tuple[list[float], list[str], dict[str, Any]]:
    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(float(value))

    mention_count = sum(len(spans) for spans in hit_map.values())
    add("target_mention_count_log", log_count(mention_count))
    selected_1200 = [event for event in events if event["distances"].get(target_key) is not None and event["distances"][target_key] <= 1200]
    for radius in (180, 500, 1200):
        selected = [event for event in selected_1200 if event["distances"][target_key] <= radius]
        add(f"target_r{radius}_event_count_log", log_count(len(selected)))
        add(f"target_r{radius}_strong_count_log", log_count(sum(bool(event["own_strong"]) for event in selected)))
        add(f"target_r{radius}_likely_count_log", log_count(sum(bool(event["own_likely"]) for event in selected)))
        add(f"target_r{radius}_external_count_log", log_count(sum(bool(event["external"]) for event in selected)))
        for category in CATEGORIES:
            add(f"target_r{radius}_{category}_count_log", log_count(sum(event["category"] == category for event in selected)))
            add(f"target_r{radius}_{category}_strong_log", log_count(sum(event["category"] == category and event["own_strong"] for event in selected)))
    assigned = [event for event in events if event["assigned_target"] == target_key]
    add("target_assigned_event_count_log", log_count(len(assigned)))
    add("target_assigned_strong_count_log", log_count(sum(bool(event["own_strong"]) for event in assigned)))
    add("target_assigned_external_count_log", log_count(sum(bool(event["external"]) for event in assigned)))
    add("target_local_strong_ratio", sum(bool(event["own_strong"]) for event in selected_1200) / max(1, len(selected_1200)))
    add("target_local_external_ratio", sum(bool(event["external"]) for event in selected_1200) / max(1, len(selected_1200)))
    add("target_local_max_score", max((event["score"] for event in selected_1200), default=-7) / 10.0)
    assigned_episodes = [item for item in episodes if item["assigned_target"] == target_key]
    add("target_assigned_episode_count_log", log_count(len(assigned_episodes)))
    add("target_assigned_strong_episode_count_log", log_count(sum(bool(item["own_strong"]) for item in assigned_episodes)))
    add("target_assigned_chain_episode_count_log", log_count(sum(bool(item["method_chain"]) for item in assigned_episodes)))
    meta = {
        "target_mention_count": mention_count,
        "target_local_events": len(selected_1200),
        "target_local_strong": sum(bool(event["own_strong"]) for event in selected_1200),
        "target_local_external": sum(bool(event["external"]) for event in selected_1200),
        "target_assigned_events": len(assigned),
        "target_assigned_strong": sum(bool(event["own_strong"]) for event in assigned),
        "target_assigned_strong_episodes": sum(bool(item["own_strong"]) for item in assigned_episodes),
    }
    return values, names, meta


def build_own_study_features(
    records: Sequence[dict[str, str]],
    structured_pages: Mapping[str, Sequence[Mapping[str, Any]]],
    concepts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    targets_by_refno: dict[str, dict[str, str]] = defaultdict(dict)
    for record in records:
        key = canonical_target_identity(record["ROCKNAME"])
        targets_by_refno[record["REFNO"]].setdefault(key, record["ROCKNAME"])
    papers: dict[str, dict[str, Any]] = {}
    article_vectors: dict[str, list[float]] = {}
    target_vectors: dict[tuple[str, str], list[float]] = {}
    target_meta: dict[tuple[str, str], dict[str, Any]] = {}
    article_names: list[str] | None = None
    target_names: list[str] | None = None
    for refno in sorted(targets_by_refno):
        paper = extract_paper_events(structured_pages[refno], targets_by_refno[refno], concepts)
        papers[refno] = paper
        char_count = sum(len(clean(page.get("text"))) for page in structured_pages[refno])
        vector, names = aggregate_article(paper["events"], paper["episodes"], char_count)
        article_vectors[refno] = vector
        article_names = article_names or names
        if article_names != names:
            raise AssertionError("article feature names changed between papers")
        for target_key in targets_by_refno[refno]:
            vector, names, meta = aggregate_target(
                target_key,
                paper["events"],
                paper["episodes"],
                paper["target_hits"][target_key],
            )
            target_vectors[(refno, target_key)] = vector
            target_meta[(refno, target_key)] = meta
            target_names = target_names or names
            if target_names != names:
                raise AssertionError("target feature names changed between targets")
    row_article = np.asarray([article_vectors[record["REFNO"]] for record in records], dtype=np.float64)
    row_target = np.asarray([
        target_vectors[(record["REFNO"], canonical_target_identity(record["ROCKNAME"]))]
        for record in records
    ], dtype=np.float64)
    row_meta = [
        {
            **target_meta[(record["REFNO"], canonical_target_identity(record["ROCKNAME"]))],
            "article_events": len(papers[record["REFNO"]]["events"]),
            "article_strong": sum(bool(item["own_strong"]) for item in papers[record["REFNO"]]["events"]),
            "article_external": sum(bool(item["external"]) for item in papers[record["REFNO"]]["events"]),
            "article_strong_episodes": sum(bool(item["own_strong"]) for item in papers[record["REFNO"]]["episodes"]),
            "article_strong_chain_episodes": sum(bool(item["own_strong"] and item["method_chain"]) for item in papers[record["REFNO"]]["episodes"]),
            "article_reference_events": sum(item["section"] == "references" for item in papers[record["REFNO"]]["events"]),
        }
        for record in records
    ]
    return {
        "row_article": row_article,
        "row_target": row_target,
        "article_vectors": article_vectors,
        "article_names": article_names or [],
        "target_names": target_names or [],
        "row_meta": row_meta,
        "paper_event_count": sum(len(item["events"]) for item in papers.values()),
        "paper_episode_count": sum(len(item["episodes"]) for item in papers.values()),
    }


def multiplicity_weights(indices: np.ndarray, records: Sequence[dict[str, str]]) -> np.ndarray:
    keys = [(records[index]["REFNO"], canonical_target_identity(records[index]["ROCKNAME"])) for index in indices]
    counts = Counter(keys)
    return np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)


def make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=240,
        learning_rate=0.04,
        max_leaf_nodes=7,
        min_samples_leaf=16,
        l2_regularization=0.5,
        random_state=RANDOM_STATE,
    )


def make_extra() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=350,
        max_features="sqrt",
        min_samples_leaf=5,
        n_jobs=1,
        random_state=RANDOM_STATE,
    )


def metric(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "rows": int(len(labels)),
        "coverage": 1.0,
        "accuracy": round(float(accuracy_score(labels, prediction)), 8),
        "macro_f1": round(float(f1_score(labels, prediction, average="macro")), 8),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, prediction)), 8),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_positive": int(prediction.sum()),
        "threshold": 0.5,
    }


def pool_target(probability: np.ndarray, records: Sequence[dict[str, str]]) -> np.ndarray:
    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        buckets[(record["REFNO"], canonical_target_identity(record["ROCKNAME"]))].append(index)
    output = probability.copy()
    for indices in buckets.values():
        output[indices] = float(probability[indices].mean())
    return output


def run_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    matrices: Mapping[str, np.ndarray],
    base_features: Mapping[str, Any],
    own: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[np.ndarray]]:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    probability = {
        f"{name}_{model_name}": np.full(len(records), np.nan, dtype=np.float64)
        for name in matrices
        for model_name in ("hgb", "extra")
    }
    probability.update({
        "article_baseline_hgb": np.full(len(records), np.nan, dtype=np.float64),
        "article_plus_own_hgb": np.full(len(records), np.nan, dtype=np.float64),
    })
    fold_indices: list[np.ndarray] = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(records)), labels, groups), start=1):
        fold_indices.append(test)
        weights = multiplicity_weights(train, records)
        for matrix_name, matrix in matrices.items():
            for model_name, model in (("hgb", make_hgb()), ("extra", make_extra())):
                model.fit(matrix[train], labels[train], sample_weight=weights)
                probability[f"{matrix_name}_{model_name}"][test] = model.predict_proba(matrix[test])[:, 1]

        train_groups = sorted(set(str(groups[index]) for index in train))
        test_groups = sorted(set(str(groups[index]) for index in test))
        members: dict[str, list[int]] = defaultdict(list)
        for index in train:
            members[str(groups[index])].append(int(index))
        train_group_labels = np.asarray([int(labels[members[group]].mean() >= 0.5) for group in train_groups], dtype=np.int8)
        article_base_train = np.asarray([base_features["article_features"][group] for group in train_groups], dtype=np.float64)
        article_base_test = np.asarray([base_features["article_features"][group] for group in test_groups], dtype=np.float64)
        article_own_train = np.asarray([own["article_vectors"][group] for group in train_groups], dtype=np.float64)
        article_own_test = np.asarray([own["article_vectors"][group] for group in test_groups], dtype=np.float64)
        for name, fit_matrix, test_matrix in (
            ("article_baseline_hgb", article_base_train, article_base_test),
            ("article_plus_own_hgb", np.hstack([article_base_train, article_own_train]), np.hstack([article_base_test, article_own_test])),
        ):
            model = make_hgb()
            model.fit(fit_matrix, train_group_labels)
            group_probability = dict(zip(test_groups, model.predict_proba(test_matrix)[:, 1]))
            probability[name][test] = np.asarray([group_probability[str(groups[index])] for index in test])
        print(f"fold {fold}/{N_SPLITS}: baseline={metric(labels[test], probability['base_hgb'][test])['accuracy']:.4f}, plus_own={metric(labels[test], probability['base_plus_all_hgb'][test])['accuracy']:.4f}", flush=True)
    for name, values in probability.items():
        if not np.isfinite(values).all():
            raise AssertionError(f"incomplete OOF probabilities: {name}")
    return probability, fold_indices


def fold_metrics(labels: np.ndarray, probability: np.ndarray, folds: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    return [{"fold": index, **metric(labels[fold], probability[fold])} for index, fold in enumerate(folds, start=1)]


def error_attribution(labels: np.ndarray, base_probability: np.ndarray, own_probability: np.ndarray, metadata: Sequence[Mapping[str, Any]], records: Sequence[dict[str, str]]) -> dict[str, Any]:
    base_pred = base_probability >= 0.5
    own_pred = own_probability >= 0.5
    paper_target_counts = Counter(record["REFNO"] for record in records)
    masks = {
        "zero_article_strong_events": np.asarray([item["article_strong"] == 0 for item in metadata]),
        "article_strong_but_no_target_local_strong": np.asarray([item["article_strong"] > 0 and item["target_local_strong"] == 0 for item in metadata]),
        "target_local_strong_event": np.asarray([item["target_local_strong"] > 0 for item in metadata]),
        "target_assigned_strong_event": np.asarray([item["target_assigned_strong"] > 0 for item in metadata]),
        "strong_method_chain_episode": np.asarray([item["article_strong_chain_episodes"] > 0 for item in metadata]),
        "external_dominated_article": np.asarray([item["article_external"] > item["article_strong"] for item in metadata]),
        "target_not_mentioned": np.asarray([item["target_mention_count"] == 0 for item in metadata]),
        "multi_row_paper": np.asarray([paper_target_counts[record["REFNO"]] > 1 for record in records]),
    }
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        base_error = base_pred != labels
        own_error = own_pred != labels
        output[name] = {
            "rows": int(mask.sum()),
            "positive_label_rate_diagnostic_only": round(float(labels[mask].mean()), 8),
            "baseline_errors": int((base_error & mask).sum()),
            "plus_own_errors": int((own_error & mask).sum()),
            "errors_resolved": int((base_error & ~own_error & mask).sum()),
            "errors_introduced": int((~base_error & own_error & mask).sum()),
            "baseline_false_positive": int((base_pred & (labels == 0) & mask).sum()),
            "baseline_false_negative": int((~base_pred & (labels == 1) & mask).sum()),
            "plus_own_false_positive": int((own_pred & (labels == 0) & mask).sum()),
            "plus_own_false_negative": int((~own_pred & (labels == 1) & mask).sum()),
        }
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=DEFAULT_POSITIVE_KB)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--freeze-output", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--oof-output", type=Path, default=DEFAULT_OOF)
    args = parser.parse_args(argv)

    records, coverage = baseline.load_pdf_only_identity(args.identity.resolve(), args.pdf_dir.resolve())
    refnos = {record["REFNO"] for record in records}
    structured_pages, paper_text = load_structured_pages(args.page_cache.resolve(), refnos)
    concepts, knowledge_audit = baseline.load_knowledge_patterns(args.taxonomy.resolve(), args.positive_kb.resolve())
    base_features = baseline.build_features(records, paper_text, concepts)
    own = build_own_study_features(records, structured_pages, concepts)
    base_matrix = np.asarray(base_features["engineered_rows"], dtype=np.float64)
    matrices = {
        "base": base_matrix,
        "own_only": np.hstack([own["row_target"], own["row_article"]]),
        "base_plus_target": np.hstack([base_matrix, own["row_target"]]),
        "base_plus_article": np.hstack([base_matrix, own["row_article"]]),
        "base_plus_all": np.hstack([base_matrix, own["row_target"], own["row_article"]]),
    }

    # This artifact is durably written before any R3 cell is dereferenced.
    freeze = {
        "schema_version": "r3_own_study_feature_candidate_freeze_v1",
        "status": "frozen_before_label_access",
        "allowed_inputs": ["REFNO/RESULTNO/ROCKNAME identity", "original-paper page cache", "frozen taxonomy", "frozen positive expert KB"],
        "forbidden_inputs_not_read": ["R3 labels", "PMAGRESULT", "ROCKMAG cells", "prior predictions", "prior evaluations"],
        "rows": len(records),
        "paper_groups": len(refnos),
        "candidate_spec": {
            "sections": list(SECTIONS),
            "categories": list(CATEGORIES),
            "target_radii_characters": [180, 500, 1200],
            "episode_gap_characters": 800,
            "strong_score_threshold": 6,
            "likely_score_threshold": 3,
            "article_feature_names": own["article_names"],
            "target_feature_names": own["target_names"],
            "model_threshold": 0.5,
            "cross_validation": "5-fold GroupKFold by REFNO",
        },
        "matrix_hashes": {name: {"shape": list(matrix.shape), "sha256": array_sha256(matrix)} for name, matrix in matrices.items()},
        "input_hashes": {
            "identity": baseline.file_sha256(args.identity.resolve()),
            "page_cache": baseline.file_sha256(args.page_cache.resolve()),
            "taxonomy": baseline.file_sha256(args.taxonomy.resolve()),
            "positive_kb": baseline.file_sha256(args.positive_kb.resolve()),
        },
    }
    baseline.atomic_json(args.freeze_output.resolve(), freeze)
    freeze_hash = baseline.file_sha256(args.freeze_output.resolve())

    labels_by_resultno = baseline.load_training_labels(args.targets.resolve(), {record["RESULTNO"] for record in records})
    labels = np.asarray([labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8)
    probabilities, folds = run_oof(records, labels, matrices, base_features, own)
    strategies: dict[str, np.ndarray] = dict(probabilities)
    for name, probability in list(probabilities.items()):
        if name.startswith(("base_", "own_")) and not name.startswith("article_"):
            strategies[f"{name}_target_pool"] = pool_target(probability, records)
    local = pool_target(probabilities["base_plus_all_hgb"], records)
    article = probabilities["article_plus_own_hgb"]
    strategies["predefined_own_local_article_25"] = 0.75 * local + 0.25 * article
    strategies["predefined_own_local_article_50"] = 0.50 * local + 0.50 * article
    strategy_results = {
        name: {"aggregate": metric(labels, value), "folds": fold_metrics(labels, value, folds)}
        for name, value in strategies.items()
    }
    ranked = sorted(strategy_results, key=lambda name: (strategy_results[name]["aggregate"]["accuracy"], strategy_results[name]["aggregate"]["macro_f1"]), reverse=True)
    predefined_best = max(
        ("base_plus_all_hgb_target_pool", "predefined_own_local_article_25", "predefined_own_local_article_50"),
        key=lambda name: strategy_results[name]["aggregate"]["accuracy"],
    )
    args.oof_output.resolve().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.oof_output.resolve(),
        base_hgb=probabilities["base_hgb"],
        base_plus_all_hgb=probabilities["base_plus_all_hgb"],
        article_baseline_hgb=probabilities["article_baseline_hgb"],
        article_plus_own_hgb=probabilities["article_plus_own_hgb"],
        base_plus_all_hgb_target_pool=strategies["base_plus_all_hgb_target_pool"],
        predefined_own_local_article_25=strategies["predefined_own_local_article_25"],
        predefined_own_local_article_50=strategies["predefined_own_local_article_50"],
        best_predefined=strategies[predefined_best],
        fold_id=np.concatenate([np.full(len(index), fold, dtype=np.int8) for fold, index in enumerate(folds, start=1)])[np.argsort(np.concatenate(folds))],
    )
    report = {
        "schema_version": "r3_own_study_events_grouped_oof_v1",
        "status": "complete",
        "scope": "development-only frozen training partition; rows without local PDF excluded",
        "guardrails": {
            "feature_candidate_frozen_before_label_access": True,
            "feature_freeze_sha256": freeze_hash,
            "cross_validation": "5-fold GroupKFold by REFNO",
            "labels": "R3 dereferenced only after fixed training/PDF membership and feature freeze",
            "forbidden_sources_not_read": ["held-out labels", "PMAGRESULT", "ROCKMAG source cells", "prior predictions/evaluations"],
            "identifiers_not_model_features": True,
        },
        "data": {
            **coverage,
            "label_counts": {str(key): value for key, value in sorted(Counter(int(item) for item in labels).items())},
            "base_feature_count": int(base_matrix.shape[1]),
            "own_article_feature_count": int(own["row_article"].shape[1]),
            "own_target_feature_count": int(own["row_target"].shape[1]),
            "knowledge_event_count": int(own["paper_event_count"]),
            "own_study_episode_count": int(own["paper_episode_count"]),
        },
        "knowledge": knowledge_audit,
        "same_config_ablation": {
            "row_hgb": {
                "baseline": strategy_results["base_hgb"],
                "plus_target_events": strategy_results["base_plus_target_hgb"],
                "plus_article_events": strategy_results["base_plus_article_hgb"],
                "plus_all_own_study_events": strategy_results["base_plus_all_hgb"],
            },
            "row_extra_trees": {
                "baseline": strategy_results["base_extra"],
                "plus_all_own_study_events": strategy_results["base_plus_all_extra"],
            },
            "article_majority_hgb": {
                "baseline": strategy_results["article_baseline_hgb"],
                "plus_own_study_events": strategy_results["article_plus_own_hgb"],
            },
        },
        "strategies": strategy_results,
        "ranking_exploratory": ranked,
        "best_predefined_fusion": {"name": predefined_best, **strategy_results[predefined_best]},
        "error_attribution": error_attribution(labels, probabilities["base_hgb"], probabilities["base_plus_all_hgb"], own["row_meta"], records),
        "artifacts": {
            "feature_freeze": {"path": str(args.freeze_output.resolve()), "sha256": freeze_hash},
            "oof_probabilities": {"path": str(args.oof_output.resolve()), "sha256": baseline.file_sha256(args.oof_output.resolve()), "contains_labels": False, "contains_record_ids": False, "alignment": "same row order as frozen PDF-only identity; fold_id included"},
            "script": {"path": str(Path(__file__).resolve()), "sha256": baseline.file_sha256(Path(__file__).resolve())},
        },
        "interpretation": {
            "full_coverage_90_reached": strategy_results[predefined_best]["aggregate"]["accuracy"] >= 0.9,
            "same_config_delta_hgb": round(strategy_results["base_plus_all_hgb"]["aggregate"]["accuracy"] - strategy_results["base_hgb"]["aggregate"]["accuracy"], 8),
            "warning": "All rankings are development grouped-OOF results; no held-out claim is made.",
        },
    }
    baseline.atomic_json(args.output.resolve(), report)
    print(json.dumps({"output": str(args.output.resolve()), "baseline": strategy_results["base_hgb"]["aggregate"], "plus_own": strategy_results["base_plus_all_hgb"]["aggregate"], "best_predefined": report["best_predefined_fusion"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
