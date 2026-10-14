#!/usr/bin/env python3
"""Small-screen R3 experiment for PDF-derived curation/style metadata.

This branch is deliberately locked to ``tmp/r3_small_screen_v1/identity.xlsx``.
It freezes all features before opening R3 labels, evaluates only by REFNO
GroupKFold, and never reads prior predictions or evaluations.
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
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as common  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


IDENTITY = ROOT / "tmp/r3_small_screen_v1/identity.xlsx"
TARGETS = ROOT / "data/targets.xlsx"
PAGE_CACHE = ROOT / "tmp/mineru_cache_all_hard_risk.jsonl"
PDF_DIR = ROOT / "data/raw_pdfs"
TAXONOMY = ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json"
POSITIVE_KB = ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json"
OUTPUT_DIR = ROOT / "tmp/r3_small_screen_curation_style"
FREEZE_OUTPUT = OUTPUT_DIR / "feature_freeze.json"
SMOKE_OUTPUT = OUTPUT_DIR / "smoke_report.json"
REPORT_OUTPUT = OUTPUT_DIR / "report.json"
OOF_OUTPUT = OUTPUT_DIR / "oof_probabilities.npz"
SCREENING_PROTOCOL = ROOT / "tmp/r3_small_screen_v1/screening_protocol.json"
SCREENING_PROTOCOL_SHA256 = "2488bb8dc2167ae93bfedffa315f44f7968049c7895ef803a9e598aa066e291d"

EXPECTED_ROWS = 640
EXPECTED_PAPERS = 330
PROMOTION_THRESHOLD = 0.81375
N_SPLITS = 5
RANDOM_STATE = 357

SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")
YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
AFFILIATION_RE = re.compile(
    r"\b(?:department|university|universit[a-z]+|institute|institut[a-z]+|"
    r"geological survey|academy|laboratory|observatory|college|school|centre|center)\b",
    re.I,
)
FRONT_STOP_RE = re.compile(
    r"^\s*(?:received|accepted|submitted|abstract|summary|introduction|keywords?)\b",
    re.I,
)
AUTHOR_RE = re.compile(
    r"\b(?:[A-Z]\.?\s*){1,4}([A-Z][A-Za-z'-]{2,})\b|"
    r"\b([A-Z][a-z'-]{2,})\s+(?:[A-Z]\.?\s*){1,3}[A-Z][A-Za-z'-]{2,}\b"
)
REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:references|bibliography|literature\s+cited|works\s+cited)\s*[:.]?\s*$"
)
SECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"(?im)^\s*(?:abstract|summary)\b")),
    ("introduction", re.compile(r"(?im)^\s*(?:\d+[. ]*)?(?:introduction|background)\b")),
    ("methods", re.compile(r"(?im)^\s*(?:\d+[. ]*)?(?:materials?\s+and\s+methods?|methods?|experimental|laboratory\s+(?:procedure|methods?|techniques?)|sampling)\b")),
    ("results", re.compile(r"(?im)^\s*(?:\d+[. ]*)?(?:results?|observations?|magnetic\s+(?:properties|mineralogy))\b")),
    ("discussion", re.compile(r"(?im)^\s*(?:\d+[. ]*)?(?:discussion|interpretation|implications?)\b")),
    ("conclusion", re.compile(r"(?im)^\s*(?:\d+[. ]*)?(?:conclusions?)\b")),
    ("references", REFERENCE_HEADING_RE),
)
FIGURE_RE = re.compile(r"(?im)^\s*(?:fig(?:ure)?\.?)[ ]*\d+[a-z]?\b")
TABLE_RE = re.compile(r"(?im)^\s*table[ ]*\d+[a-z]?\b")
TITLE_KEYWORDS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("paleomagnetic", re.compile(r"\bpal(?:ae|eo)magnet\w*\b", re.I)),
    ("rock_magnetic", re.compile(r"\brock[- ]magnet\w*\b", re.I)),
    ("magnetic_mineralogy", re.compile(r"\bmagnetic\s+mineralog\w*\b", re.I)),
    ("magnetostratigraphy", re.compile(r"\bmagnetostrat\w*\b", re.I)),
    ("remanence", re.compile(r"\breman\w+\b", re.I)),
    ("new_results", re.compile(r"\b(?:new|results?|study|investigation)\b", re.I)),
    ("age_test", re.compile(r"\b(?:age|dating|test|reliability)\b", re.I)),
)
JOURNAL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("gji", re.compile(r"geophys(?:ical)?\.?\s*j(?:ournal)?|geophys\.\s*j\.\s*r\.\s*astr", re.I)),
    ("jgr", re.compile(r"journal\s+of\s+geophysical\s+research|j\.\s*geophys\.\s*res", re.I)),
    ("epsl", re.compile(r"earth\s+and\s+planetary\s+science\s+letters", re.I)),
    ("tectonophysics", re.compile(r"tectonophysics", re.I)),
    ("grl", re.compile(r"geophysical\s+research\s+letters", re.I)),
    ("pepi", re.compile(r"physics\s+of\s+the\s+earth\s+and\s+planetary\s+interiors", re.I)),
    ("palaeo3", re.compile(r"palaeogeography|paleogeography", re.I)),
    ("canadian", re.compile(r"canadian\s+journal|can\.\s*j\.", re.I)),
    ("nature_science", re.compile(r"\bnature\b|\bscience\b", re.I)),
    ("agu_other", re.compile(r"american\s+geophysical\s+union|agu", re.I)),
)
JOURNAL_FAMILIES = tuple(name for name, _ in JOURNAL_RULES) + ("other",)
DECADES = ("pre1970", "1970s", "1980s", "1990s", "2000s", "2010s", "2020s", "missing")
AUTHOR_STOPWORDS = {
    "Abstract", "Academy", "Address", "American", "Canada", "Centre", "College",
    "Department", "Earth", "Formation", "Geological", "Geophysics", "Institute",
    "Journal", "Laboratory", "London", "Magnetic", "National", "Physics", "Research",
    "Received", "School", "Science", "Sciences", "Summary", "Survey", "University",
}


def clean(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").replace("\x00", " ")).strip()


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8).tobytes()).hexdigest()


def log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def decade_name(year: int | None) -> str:
    if year is None:
        return "missing"
    if year < 1970:
        return "pre1970"
    if year < 1980:
        return "1970s"
    if year < 1990:
        return "1980s"
    if year < 2000:
        return "1990s"
    if year < 2010:
        return "2000s"
    if year < 2020:
        return "2010s"
    return "2020s"


def publication_year(first_page: str) -> tuple[int | None, str]:
    front = first_page[:2400]
    priority = (
        ("journal_parenthetical", re.compile(r"\(((?:18|19|20)\d{2})\)\s*\d{1,4}(?:\s*[,:(]|\s*$)", re.I)),
        ("copyright", re.compile(r"(?:copyright|©|published\s+by).{0,50}((?:18|19|20)\d{2})", re.I | re.S)),
        ("publication_line", re.compile(r"(?:published|publication|volume|vol\.).{0,80}\b((?:18|19|20)\d{2})\b", re.I | re.S)),
        ("received", re.compile(r"(?:received|accepted).{0,50}\b((?:18|19|20)\d{2})\b", re.I | re.S)),
    )
    for source, pattern in priority:
        match = pattern.search(front)
        if match:
            return int(match.group(1)), source
    match = YEAR_RE.search(front[:900])
    return (int(match.group(1)), "generic_front") if match else (None, "missing")


def journal_family(front: str) -> str:
    for name, pattern in JOURNAL_RULES:
        if pattern.search(front[:1800]):
            return name
    return "other"


def title_author_metadata(first_page: str) -> dict[str, Any]:
    lines = [clean(line) for line in first_page.splitlines() if clean(line)]
    stop = next((index for index, line in enumerate(lines) if FRONT_STOP_RE.search(line)), min(24, len(lines)))
    front_lines = lines[:stop]
    topical = next(
        (
            index
            for index, line in enumerate(front_lines)
            if any(pattern.search(line) for _, pattern in TITLE_KEYWORDS)
            and not re.search(r"\bjournal\b|geophys\.?\s*j\.?", line, re.I)
        ),
        1 if len(front_lines) > 1 else 0,
    )
    title_lines: list[str] = []
    author_start = len(front_lines)
    for index in range(topical, min(len(front_lines), topical + 7)):
        line = front_lines[index]
        if title_lines and (AFFILIATION_RE.search(line) or AUTHOR_RE.search(line)):
            author_start = index
            break
        title_lines.append(line)
        author_start = index + 1
        if len(" ".join(title_lines)) >= 260:
            break
    title = clean(" ".join(title_lines))[:600]
    author_block = clean(" ".join(front_lines[author_start:]))[:2200]
    authors: set[str] = set()
    for match in AUTHOR_RE.finditer(author_block):
        raw = match.group(1) or match.group(2) or ""
        if raw and raw not in AUTHOR_STOPWORDS:
            authors.add(raw.casefold())
    return {
        "title": title,
        "title_line_count": len(title_lines),
        "author_block": author_block,
        "author_count": len(authors),
        "affiliation_count": len(AFFILIATION_RE.findall(author_block)),
        "email_count": len(EMAIL_RE.findall(author_block)),
    }


def concept_category(concept: Mapping[str, Any]) -> str:
    value = f"{concept.get('category', '')}:{concept.get('concept_id', '')}".casefold()
    for category in ("method", "diagnostic", "carrier"):
        if category in value:
            return category
    return "other"


def count_concepts(text: str, concepts: Sequence[Mapping[str, Any]]) -> list[int]:
    counts: list[int] = []
    for concept in concepts:
        total = 0
        for pattern in concept["patterns"]:
            for _ in pattern.finditer(text):
                total += 1
                if total >= 50:
                    break
            if total >= 50:
                break
        counts.append(total)
    return counts


def target_document(text: str, rockname: str) -> tuple[str, int]:
    spans: list[tuple[int, int]] = []
    for pattern in common.target_patterns(rockname):
        for match in pattern.finditer(text):
            spans.append((max(0, match.start() - 1000), min(len(text), match.end() + 1000)))
            if len(spans) >= 80:
                break
        if len(spans) >= 80:
            break
    merged = common.merge_spans(spans)
    document = "\n".join(clean(text[start:end]) for start, end in merged)
    return document, len(spans)


def load_pages(path: Path, refnos: set[str]) -> dict[str, list[dict[str, Any]]]:
    papers: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            refno = clean(row.get("REFNO"))
            if refno not in refnos:
                continue
            if refno in papers:
                raise ValueError(f"duplicate REFNO {refno} at cache line {line_number}")
            pages = []
            for index, page in enumerate(row.get("pages", []), start=1):
                if not isinstance(page, Mapping) or not clean(page.get("text")):
                    continue
                pages.append({
                    "page": clean(page.get("page")) or str(index),
                    "text": str(page.get("text")),
                    "source": clean(page.get("source")) or "unknown",
                    "section_hints": [clean(item) for item in page.get("section_hints", []) if clean(item)],
                })
            papers[refno] = pages
    missing = sorted(refnos - papers.keys())
    empty = sorted(refno for refno, pages in papers.items() if not pages)
    if missing or empty:
        raise ValueError(f"page-cache mismatch: missing={missing[:10]}, empty={empty[:10]}")
    return papers


def load_small_identity(
    path: Path,
    pdf_dir: Path,
    protocol: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    selection = protocol["selection"]
    if common.file_sha256(path) != selection.get("identity_sha256"):
        raise ValueError("identity does not match the frozen screening protocol")
    raw = common.load_training_identity(path)
    pdf_refnos = common.local_pdf_refnos(pdf_dir)
    selected = [record for record in raw if record["REFNO"] in pdf_refnos]
    expected_rows = int(selection["pdf_only_rows"])
    expected_papers = int(selection["pdf_only_paper_groups"])
    if len(selected) != expected_rows or len({record["REFNO"] for record in selected}) != expected_papers:
        raise ValueError("small-screen PDF membership changed; refusing an expanded or incomplete experiment")
    return selected, {
        "identity_rows_before_pdf_gate": len(raw),
        "identity_papers_before_pdf_gate": len({record["REFNO"] for record in raw}),
        "pdf_only_rows": len(selected),
        "pdf_only_papers": len({record["REFNO"] for record in selected}),
        "rows_skipped_without_pdf": len(raw) - len(selected),
    }


def paper_style_features(
    pages: Sequence[Mapping[str, Any]],
    pdf_path: Path,
) -> tuple[list[float], list[str], dict[str, Any]]:
    texts = [str(page["text"]) for page in pages]
    text = "\n\n".join(texts)
    first = texts[0]
    front = first[:3000]
    year, year_source = publication_year(first)
    decade = decade_name(year)
    journal = journal_family(front)
    title_meta = title_author_metadata(first)
    title = title_meta["title"]
    page_lengths = np.asarray([len(item) for item in texts], dtype=np.float64)
    line_count = sum(item.count("\n") + 1 for item in texts)
    words = WORD_RE.findall(text)
    letters = [character for character in text if character.isalpha()]
    sources = Counter(clean(page.get("source")) or "unknown" for page in pages)
    section_counts = {name: len(pattern.findall(text)) for name, pattern in SECTION_RULES}
    reference_match = REFERENCE_HEADING_RE.search(text)
    reference_text = text[reference_match.end():] if reference_match else ""

    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        values.append(float(value))

    normalized_year = ((year or 1985) - 1950) / 50.0
    add("publication_year_normalized", normalized_year)
    add("publication_year_missing", float(year is None))
    for name in DECADES:
        add(f"decade_{name}", float(decade == name))
    for source_name in ("journal_parenthetical", "copyright", "publication_line", "received", "generic_front", "missing"):
        add(f"year_source_{source_name}", float(year_source == source_name))
    for family in JOURNAL_FAMILIES:
        add(f"journal_{family}", float(journal == family))
    add("page_count_log", log1p(len(pages)))
    add("pdf_bytes_log", log1p(pdf_path.stat().st_size))
    add("text_characters_log", log1p(len(text)))
    add("text_words_log", log1p(len(words)))
    add("text_lines_log", log1p(line_count))
    add("mean_page_characters_log", log1p(float(page_lengths.mean())))
    add("std_page_characters_log", log1p(float(page_lengths.std())))
    add("min_page_characters_log", log1p(float(page_lengths.min())))
    add("max_page_characters_log", log1p(float(page_lengths.max())))
    add("first_page_characters_log", log1p(len(first)))
    add("pdf_bytes_per_text_character_log", log1p(pdf_path.stat().st_size / max(1, len(text))))
    add("newline_density_1k", min(100.0, line_count * 1000.0 / max(1, len(text))))
    add("digit_ratio", sum(character.isdigit() for character in text) / max(1, len(text)))
    add("alpha_ratio", len(letters) / max(1, len(text)))
    add("uppercase_letter_ratio", sum(character.isupper() for character in letters) / max(1, len(letters)))
    add("replacement_character_count_log", log1p(text.count("�")))
    add("hyphen_linebreak_count_log", log1p(len(re.findall(r"-\s*\n\s*[a-z]", text))))
    add("download_watermark_count_log", log1p(len(re.findall(r"Downloaded from https?://", text, re.I))))
    add("doi_count_log", log1p(len(DOI_RE.findall(front))))
    add("section_hint_page_count_log", log1p(sum(bool(page.get("section_hints")) for page in pages)))
    add("section_hint_total_log", log1p(sum(len(page.get("section_hints", [])) for page in pages)))
    for source_name in ("pypdf", "mineru", "ocr", "unknown"):
        count = sum(value for key, value in sources.items() if source_name in key.casefold())
        add(f"page_source_{source_name}_ratio", count / max(1, len(pages)))
    add("figure_caption_count_log", log1p(len(FIGURE_RE.findall(text))))
    add("table_caption_count_log", log1p(len(TABLE_RE.findall(text))))
    add("pipe_table_line_count_log", log1p(sum(line.count("|") >= 2 for line in text.splitlines())))
    for section_name in (name for name, _ in SECTION_RULES):
        add(f"section_{section_name}_count_log", log1p(section_counts[section_name]))
        add(f"section_{section_name}_present", float(section_counts[section_name] > 0))
    add("reference_year_count_log", log1p(len(YEAR_RE.findall(reference_text))))
    add("reference_fraction", len(reference_text) / max(1, len(text)))
    add("title_characters_log", log1p(len(title)))
    add("title_words_log", log1p(len(WORD_RE.findall(title))))
    add("title_line_count_log", log1p(title_meta["title_line_count"]))
    add("title_has_colon", float(":" in title))
    add("title_uppercase_ratio", sum(character.isupper() for character in title) / max(1, sum(character.isalpha() for character in title)))
    for keyword, pattern in TITLE_KEYWORDS:
        add(f"title_keyword_{keyword}", float(bool(pattern.search(title))))
    add("author_count_log", log1p(title_meta["author_count"]))
    add("author_block_characters_log", log1p(len(title_meta["author_block"])))
    add("affiliation_count_log", log1p(title_meta["affiliation_count"]))
    add("email_count_log", log1p(title_meta["email_count"]))
    add("front_received_present", float(bool(re.search(r"\breceived\b", front, re.I))))
    add("front_accepted_present", float(bool(re.search(r"\baccepted\b", front, re.I))))
    add("front_keywords_present", float(bool(re.search(r"\bkeywords?\b", front, re.I))))
    meta = {
        "year": year,
        "year_source": year_source,
        "decade": decade,
        "journal": journal,
        "page_count": len(pages),
        "source_family": max(sources, key=sources.get),
        "has_structured_sections": sum(section_counts.values()) >= 3,
        "has_figures_or_tables": bool(FIGURE_RE.search(text) or TABLE_RE.search(text)),
    }
    return values, names, meta


def build_features(
    records: Sequence[dict[str, str]],
    pages_by_refno: Mapping[str, Sequence[Mapping[str, Any]]],
    concepts: Sequence[Mapping[str, Any]],
    pdf_dir: Path,
) -> dict[str, Any]:
    raw_targets: dict[str, dict[str, str]] = defaultdict(dict)
    rows_per_target: Counter[tuple[str, str]] = Counter()
    for record in records:
        target = canonical_target_identity(record["ROCKNAME"])
        raw_targets[record["REFNO"]].setdefault(target, record["ROCKNAME"])
        rows_per_target[(record["REFNO"], target)] += 1

    style_by_paper: dict[str, list[float]] = {}
    knowledge_by_paper: dict[str, list[float]] = {}
    paper_meta: dict[str, dict[str, Any]] = {}
    text_by_paper: dict[str, str] = {}
    style_names: list[str] | None = None
    knowledge_names = [f"article_concept_{concept_category(concept)}_{concept['concept_id']}_log" for concept in concepts]
    knowledge_names += [
        f"article_category_{category}_total_log" for category in ("method", "diagnostic", "carrier", "other")
    ] + [
        f"article_category_{category}_distinct_log" for category in ("method", "diagnostic", "carrier", "other")
    ]
    for refno in sorted(raw_targets):
        pages = pages_by_refno[refno]
        text = "\n\n".join(str(page["text"]) for page in pages)
        text_by_paper[refno] = text
        style, names, meta = paper_style_features(pages, pdf_dir / f"{refno}.pdf")
        style_names = style_names or names
        if style_names != names:
            raise AssertionError("paper style feature order changed")
        counts = count_concepts(text, concepts)
        by_category: dict[str, list[int]] = defaultdict(list)
        for concept, count in zip(concepts, counts):
            by_category[concept_category(concept)].append(count)
        knowledge = [log1p(value) for value in counts]
        knowledge += [log1p(sum(by_category[category])) for category in ("method", "diagnostic", "carrier", "other")]
        knowledge += [log1p(sum(value > 0 for value in by_category[category])) for category in ("method", "diagnostic", "carrier", "other")]
        style_by_paper[refno] = style
        knowledge_by_paper[refno] = knowledge
        paper_meta[refno] = meta

    row_style: list[list[float]] = []
    row_knowledge: list[list[float]] = []
    row_meta: list[dict[str, Any]] = []
    target_names = [
        "target_count_log", "rows_in_paper_log", "rows_per_target_log",
        "rockname_characters_log", "rockname_words_log", "target_mention_count_log",
        "title_target_mention", "year_x_target_count", "year_x_rows_in_paper",
    ]
    local_knowledge_names = [f"target_concept_{concept_category(concept)}_{concept['concept_id']}_log" for concept in concepts]
    row_counts = Counter(record["REFNO"] for record in records)
    for record in records:
        refno = record["REFNO"]
        target = canonical_target_identity(record["ROCKNAME"])
        local_document, target_mentions = target_document(text_by_paper[refno], record["ROCKNAME"])
        local_counts = count_concepts(local_document, concepts)
        year = paper_meta[refno]["year"]
        year_norm = ((year or 1985) - 1950) / 50.0
        target_count = len(raw_targets[refno])
        title_front = str(pages_by_refno[refno][0]["text"])[:1600]
        title_target = int(any(pattern.search(title_front) for pattern in common.target_patterns(record["ROCKNAME"])))
        target_style = [
            log1p(target_count), log1p(row_counts[refno]), log1p(rows_per_target[(refno, target)]),
            log1p(len(record["ROCKNAME"])), log1p(len(WORD_RE.findall(record["ROCKNAME"]))),
            log1p(target_mentions), float(title_target), year_norm * log1p(target_count),
            year_norm * log1p(row_counts[refno]),
        ]
        row_style.append([*style_by_paper[refno], *target_style])
        row_knowledge.append([*knowledge_by_paper[refno], *[log1p(value) for value in local_counts]])
        row_meta.append({
            **paper_meta[refno],
            "target_count": target_count,
            "target_mentions": target_mentions,
            "article_method_events": sum(
                count for concept, count in zip(concepts, [int(round(math.expm1(value))) for value in knowledge_by_paper[refno][:len(concepts)]])
                if concept_category(concept) == "method"
            ),
        })

    style_matrix = np.asarray(row_style, dtype=np.float64)
    knowledge_matrix = np.asarray(row_knowledge, dtype=np.float64)
    # Explicit year interactions are frozen, coarse and independent of labels.
    year_column = style_matrix[:, (style_names or []).index("publication_year_normalized")]
    method_indices = [index for index, concept in enumerate(concepts) if concept_category(concept) == "method"]
    article_method_total = np.asarray([
        sum(math.expm1(knowledge_by_paper[record["REFNO"]][index]) for index in method_indices)
        for record in records
    ])
    article_method_distinct = np.asarray([
        sum(knowledge_by_paper[record["REFNO"]][index] > 0 for index in method_indices)
        for record in records
    ])
    interaction = np.column_stack([
        year_column * np.log1p(article_method_total),
        year_column * np.log1p(article_method_distinct),
        year_column * style_matrix[:, (style_names or []).index("page_count_log")],
        year_column * style_matrix[:, (style_names or []).index("figure_caption_count_log")],
        year_column * style_matrix[:, (style_names or []).index("table_caption_count_log")],
        year_column * style_matrix[:, (style_names or []).index("section_hint_page_count_log")],
    ])
    interaction_names = [
        "year_x_article_method_total", "year_x_article_method_distinct", "year_x_page_count",
        "year_x_figure_count", "year_x_table_count", "year_x_section_hint_pages",
    ]
    style_matrix = np.hstack([style_matrix, interaction])
    return {
        "style": style_matrix,
        "knowledge": knowledge_matrix,
        "all": np.hstack([style_matrix, knowledge_matrix]),
        "style_feature_names": [*(style_names or []), *target_names, *interaction_names],
        "knowledge_feature_names": [*knowledge_names, *local_knowledge_names],
        "paper_style": style_by_paper,
        "paper_knowledge": knowledge_by_paper,
        "paper_meta": paper_meta,
        "row_meta": row_meta,
    }


def make_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=260, learning_rate=0.035, max_leaf_nodes=7,
        min_samples_leaf=10, l2_regularization=1.0, random_state=RANDOM_STATE,
    )


def make_extra() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500, max_features="sqrt", min_samples_leaf=3,
        n_jobs=1, random_state=RANDOM_STATE,
    )


def make_gradient() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=220, learning_rate=0.035, max_depth=1,
        min_samples_leaf=8, random_state=RANDOM_STATE,
    )


def sample_weights(indices: np.ndarray, records: Sequence[dict[str, str]]) -> np.ndarray:
    keys = [(records[index]["REFNO"], canonical_target_identity(records[index]["ROCKNAME"])) for index in indices]
    counts = Counter(keys)
    return np.asarray([1.0 / counts[key] for key in keys], dtype=np.float64)


def metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    prediction = (probability >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, prediction, labels=[0, 1]).ravel()
    return {
        "rows": int(len(labels)), "coverage": 1.0,
        "accuracy": round(float(accuracy_score(labels, prediction)), 8),
        "macro_f1": round(float(f1_score(labels, prediction, average="macro")), 8),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, prediction)), 8),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "predicted_positive": int(prediction.sum()), "threshold": 0.5,
    }


def group_folds(records: Sequence[dict[str, str]], labels: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    return list(GroupKFold(n_splits=N_SPLITS).split(np.zeros(len(records)), labels, groups))


def paper_training_data(
    train: np.ndarray,
    test: np.ndarray,
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    features: Mapping[str, Any],
    include_knowledge: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    train_members: dict[str, list[int]] = defaultdict(list)
    for index in train:
        train_members[records[index]["REFNO"]].append(int(index))
    train_groups = sorted(train_members)
    test_groups = sorted({records[index]["REFNO"] for index in test})
    train_labels = np.asarray([int(labels[train_members[refno]].mean() >= 0.5) for refno in train_groups], dtype=np.int8)
    def vector(refno: str) -> list[float]:
        value = list(features["paper_style"][refno])
        if include_knowledge:
            value.extend(features["paper_knowledge"][refno])
        return value
    return (
        np.asarray([vector(refno) for refno in train_groups], dtype=np.float64),
        train_labels,
        np.asarray([vector(refno) for refno in test_groups], dtype=np.float64),
        test_groups,
    )


def run_smoke(
    records: Sequence[dict[str, str]], labels: np.ndarray,
    features: Mapping[str, Any], fold: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    train, test = fold
    weights = sample_weights(train, records)
    model = make_hgb()
    model.fit(features["all"][train], labels[train], sample_weight=weights)
    probability = model.predict_proba(features["all"][test])[:, 1]
    paper_train, paper_labels, paper_test, test_groups = paper_training_data(
        train, test, records, labels, features, include_knowledge=True
    )
    paper_model = make_hgb()
    paper_model.fit(paper_train, paper_labels)
    mapping = dict(zip(test_groups, paper_model.predict_proba(paper_test)[:, 1]))
    paper_probability = np.asarray([mapping[records[index]["REFNO"]] for index in test])
    return {
        "fold": 1,
        "train_rows": int(len(train)), "test_rows": int(len(test)),
        "train_papers": len({records[index]["REFNO"] for index in train}),
        "test_papers": len({records[index]["REFNO"] for index in test}),
        "row_all_hgb": metrics(labels[test], probability),
        "paper_all_hgb": metrics(labels[test], paper_probability),
        "finite_probabilities": bool(np.isfinite(probability).all() and np.isfinite(paper_probability).all()),
    }


def run_full_oof(
    records: Sequence[dict[str, str]], labels: np.ndarray,
    features: Mapping[str, Any], folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> dict[str, np.ndarray]:
    names = (
        "row_style_hgb", "row_knowledge_hgb", "row_all_hgb",
        "row_all_extra", "row_all_gradient", "paper_style_hgb", "paper_all_hgb",
    )
    output = {name: np.full(len(records), np.nan, dtype=np.float64) for name in names}
    for fold_number, (train, test) in enumerate(folds, start=1):
        weights = sample_weights(train, records)
        specs = (
            ("row_style_hgb", make_hgb(), features["style"]),
            ("row_knowledge_hgb", make_hgb(), features["knowledge"]),
            ("row_all_hgb", make_hgb(), features["all"]),
            ("row_all_extra", make_extra(), features["all"]),
            ("row_all_gradient", make_gradient(), features["all"]),
        )
        for name, model, matrix in specs:
            model.fit(matrix[train], labels[train], sample_weight=weights)
            output[name][test] = model.predict_proba(matrix[test])[:, 1]
        for name, include_knowledge in (("paper_style_hgb", False), ("paper_all_hgb", True)):
            paper_train, paper_labels, paper_test, test_groups = paper_training_data(
                train, test, records, labels, features, include_knowledge
            )
            model = make_hgb()
            model.fit(paper_train, paper_labels)
            mapping = dict(zip(test_groups, model.predict_proba(paper_test)[:, 1]))
            output[name][test] = np.asarray([mapping[records[index]["REFNO"]] for index in test])
        print(
            f"fold {fold_number}/{N_SPLITS}: style={metrics(labels[test], output['row_style_hgb'][test])['accuracy']:.4f}, "
            f"all={metrics(labels[test], output['row_all_hgb'][test])['accuracy']:.4f}, "
            f"paper={metrics(labels[test], output['paper_all_hgb'][test])['accuracy']:.4f}",
            flush=True,
        )
    if any(not np.isfinite(value).all() for value in output.values()):
        raise AssertionError("incomplete OOF probability")
    output["predefined_row_model_ensemble"] = 0.50 * output["row_all_hgb"] + 0.30 * output["row_all_extra"] + 0.20 * output["row_all_gradient"]
    for weight in (0.25, 0.50, 0.75):
        output[f"predefined_row_paper_all_{int(weight * 100):02d}"] = (1.0 - weight) * output["predefined_row_model_ensemble"] + weight * output["paper_all_hgb"]
        output[f"predefined_row_paper_style_{int(weight * 100):02d}"] = (1.0 - weight) * output["predefined_row_model_ensemble"] + weight * output["paper_style_hgb"]
    return output


def fold_records(labels: np.ndarray, probability: np.ndarray, folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> list[dict[str, Any]]:
    return [{"fold": number, **metrics(labels[test], probability[test])} for number, (_, test) in enumerate(folds, start=1)]


def subset_attribution(
    labels: np.ndarray, probability: np.ndarray,
    records: Sequence[dict[str, str]], metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    prediction = probability >= 0.5
    masks: dict[str, np.ndarray] = {}
    for decade in DECADES:
        masks[f"decade_{decade}"] = np.asarray([item["decade"] == decade for item in metadata])
    for journal in JOURNAL_FAMILIES:
        masks[f"journal_{journal}"] = np.asarray([item["journal"] == journal for item in metadata])
    masks["single_target_paper"] = np.asarray([item["target_count"] == 1 for item in metadata])
    masks["multi_target_paper"] = np.asarray([item["target_count"] > 1 for item in metadata])
    masks["structured_sections"] = np.asarray([bool(item["has_structured_sections"]) for item in metadata])
    masks["figures_or_tables"] = np.asarray([bool(item["has_figures_or_tables"]) for item in metadata])
    output: dict[str, Any] = {}
    for name, mask in masks.items():
        if not mask.any():
            continue
        output[name] = {
            "rows": int(mask.sum()),
            "paper_groups": len({records[index]["REFNO"] for index in np.flatnonzero(mask)}),
            "positive_label_rate_diagnostic_only": round(float(labels[mask].mean()), 8),
            "accuracy": round(float((prediction[mask] == labels[mask]).mean()), 8),
            "false_positive": int((prediction & (labels == 0) & mask).sum()),
            "false_negative": int((~prediction & (labels == 1) & mask).sum()),
        }
    return output


def load_screening_protocol(path: Path, identity_path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_CANDIDATE_RESULTS":
        raise ValueError("small-screen protocol is not frozen before candidate results")
    selection = protocol.get("selection", {})
    gate = protocol.get("promotion_gate", {})
    if selection.get("identity_sha256") != common.file_sha256(identity_path):
        raise ValueError("screening protocol identity hash mismatch")
    if int(selection.get("pdf_only_rows", -1)) < 1 or int(
        selection.get("pdf_only_paper_groups", -1)
    ) < 1:
        raise ValueError("screening protocol membership counts mismatch")
    required_gate_keys = {
        "minimum_aggregate_accuracy",
        "minimum_absolute_gain_over_strict_baseline",
        "minimum_folds_not_worse_than_strict_baseline",
        "maximum_allowed_accuracy_regression_in_any_fold",
    }
    if not required_gate_keys.issubset(gate):
        raise ValueError("screening protocol promotion gate is incomplete")
    return protocol


def promotion_record(
    candidate: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> dict[str, Any]:
    gate = protocol["promotion_gate"]
    baseline = protocol["baseline"]
    candidate_accuracy = float(candidate["aggregate"]["accuracy"])
    fold_accuracy = [float(item["accuracy"]) for item in candidate["folds"]]
    baseline_folds = [float(item) for item in baseline["strict_fold_accuracy"]]
    folds_not_worse = sum(value >= reference for value, reference in zip(fold_accuracy, baseline_folds))
    maximum_regression = max(reference - value for value, reference in zip(fold_accuracy, baseline_folds))
    aggregate_gain = candidate_accuracy - float(baseline["strict_accuracy"])
    checks = {
        "minimum_aggregate_accuracy": candidate_accuracy >= float(gate["minimum_aggregate_accuracy"]),
        "minimum_absolute_gain_over_strict_baseline": aggregate_gain >= float(gate["minimum_absolute_gain_over_strict_baseline"]),
        "minimum_folds_not_worse_than_strict_baseline": folds_not_worse >= int(gate["minimum_folds_not_worse_than_strict_baseline"]),
        "maximum_allowed_accuracy_regression_in_any_fold": maximum_regression <= float(gate["maximum_allowed_accuracy_regression_in_any_fold"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "candidate_accuracy": round(candidate_accuracy, 8),
        "absolute_gain_over_strict_baseline": round(aggregate_gain, 8),
        "folds_not_worse_than_strict_baseline": folds_not_worse,
        "maximum_fold_regression": round(maximum_regression, 8),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=IDENTITY)
    parser.add_argument(
        "--screening-protocol", type=Path, default=SCREENING_PROTOCOL
    )
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    freeze_path = output_dir / FREEZE_OUTPUT.name
    smoke_path = output_dir / SMOKE_OUTPUT.name
    report_path = output_dir / REPORT_OUTPUT.name
    oof_path = output_dir / OOF_OUTPUT.name

    identity_path = args.identity.resolve()
    protocol_path = args.screening_protocol.resolve()
    protocol = load_screening_protocol(protocol_path, identity_path)
    protocol_sha256 = common.file_sha256(protocol_path)
    selection = protocol["selection"]

    records, coverage = load_small_identity(
        identity_path, PDF_DIR.resolve(), protocol
    )
    refnos = {record["REFNO"] for record in records}
    pages = load_pages(PAGE_CACHE.resolve(), refnos)
    concepts, knowledge_audit = common.load_knowledge_patterns(TAXONOMY.resolve(), POSITIVE_KB.resolve())
    features = build_features(records, pages, concepts, PDF_DIR.resolve())
    matrices = {name: features[name] for name in ("style", "knowledge", "all")}
    freeze = {
        "schema_version": "r3_small_screen_curation_style_feature_freeze_v1",
        "status": "frozen_before_label_access",
        "scope_lock": {
            "identity_path": str(identity_path),
            "identity_sha256": common.file_sha256(identity_path),
            "expected_pdf_only_rows": int(selection["pdf_only_rows"]),
            "expected_pdf_only_papers": int(selection["pdf_only_paper_groups"]),
            "identity_must_match_frozen_protocol": True,
        },
        "allowed_sources": [
            "REFNO/RESULTNO/ROCKNAME fixed identity", "local PDF file size",
            "original-paper page-cache text/layout metadata", "frozen taxonomy",
            "frozen positive expert KB",
        ],
        "forbidden_sources_not_read": [
            "R3 labels before this freeze", "PMAGRESULT", "ROCKMAG source cells",
            "prior predictions", "prior evaluations", "numeric REFNO/RESULTNO model features",
        ],
        "candidate": {
            "promotion_threshold": float(
                protocol["promotion_gate"]["minimum_aggregate_accuracy"]
            ),
            "screening_protocol_sha256": protocol_sha256,
            "cross_validation": "5-fold GroupKFold by REFNO",
            "model_threshold": 0.5,
            "style_feature_names": features["style_feature_names"],
            "knowledge_feature_names": features["knowledge_feature_names"],
            "publication_year_interactions_explicit": True,
            "catboost_used": False,
            "catboost_reason": "not installed in isolated workspace runtime; sklearn boosting used",
        },
        "matrices": {name: {"shape": list(value.shape), "sha256": sha256_array(value)} for name, value in matrices.items()},
        "inputs": {
            "page_cache": {"path": str(PAGE_CACHE.resolve()), "sha256": common.file_sha256(PAGE_CACHE.resolve())},
            "taxonomy": {"path": str(TAXONOMY.resolve()), "sha256": common.file_sha256(TAXONOMY.resolve())},
            "positive_kb": {"path": str(POSITIVE_KB.resolve()), "sha256": common.file_sha256(POSITIVE_KB.resolve())},
            "screening_protocol": {
                "path": str(protocol_path), "sha256": protocol_sha256
            },
        },
    }
    common.atomic_json(freeze_path, freeze)
    freeze_sha256 = common.file_sha256(freeze_path)

    labels_by_resultno = common.load_training_labels(TARGETS.resolve(), {record["RESULTNO"] for record in records})
    labels = np.asarray([labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8)
    folds = group_folds(records, labels)
    smoke = run_smoke(records, labels, features, folds[0])
    smoke_report = {
        "schema_version": "r3_small_screen_curation_style_smoke_v1",
        "status": "pass" if smoke["finite_probabilities"] else "fail",
        "feature_freeze_sha256": freeze_sha256,
        "scope": coverage,
        "result": smoke,
        "labels_used": "only outer-fold training labels fit each smoke model; test labels used after prediction for metrics",
    }
    common.atomic_json(smoke_path, smoke_report)
    print(json.dumps(smoke_report, ensure_ascii=False), flush=True)
    if args.smoke_only:
        return 0

    probabilities = run_full_oof(records, labels, features, folds)
    results = {
        name: {"aggregate": metrics(labels, value), "folds": fold_records(labels, value, folds)}
        for name, value in probabilities.items()
    }
    ranked = sorted(results, key=lambda name: (results[name]["aggregate"]["accuracy"], results[name]["aggregate"]["macro_f1"]), reverse=True)
    best = ranked[0]
    fold_id = np.empty(len(records), dtype=np.int8)
    for fold_number, (_, test) in enumerate(folds, start=1):
        fold_id[test] = fold_number
    np.savez_compressed(oof_path, **probabilities, fold_id=fold_id)
    candidate_gates = {
        name: promotion_record(result, protocol) for name, result in results.items()
    }
    passing_candidates = [name for name in ranked if candidate_gates[name]["passed"]]
    promoted = bool(passing_candidates)
    report = {
        "schema_version": "r3_small_screen_curation_style_grouped_oof_v1",
        "status": "complete",
        "scope": {**coverage, "full_dataset_rows_evaluated": 0, "small_screen_only": True},
        "guardrails": {
            "feature_frozen_before_label_access": True,
            "feature_freeze_sha256": freeze_sha256,
            "labels": "R3 opened only after fixed membership and feature freeze; every model uses outer-fold training labels only",
            "cross_validation": "5-fold GroupKFold by REFNO",
            "numeric_identifiers_are_not_features": True,
            "old_predictions_or_evaluations_read": False,
            "PMAGRESULT_or_ROCKMAG_read": False,
        },
        "data": {
            "rows": len(records), "paper_groups": len(refnos),
            "label_counts": {str(key): value for key, value in sorted(Counter(int(item) for item in labels).items())},
            "style_feature_count": int(features["style"].shape[1]),
            "knowledge_feature_count": int(features["knowledge"].shape[1]),
        },
        "knowledge": knowledge_audit,
        "smoke": smoke_report,
        "strategies": results,
        "ranking": ranked,
        "best": {"name": best, **results[best]},
        "promotion_gate": {
            "protocol": {"path": str(protocol_path), "sha256": protocol_sha256},
            "requirements": protocol["promotion_gate"],
            "strict_baseline": protocol["baseline"],
            "observed_accuracy": results[best]["aggregate"]["accuracy"],
            "passed": promoted,
            "passing_candidates": passing_candidates,
            "candidate_checks": candidate_gates,
            "decision": "target_reached_on_sample" if promoted else "target_not_reached_on_sample",
            "larger_dataset_was_not_run": True,
        },
        "error_attribution": subset_attribution(labels, probabilities[best], records, features["row_meta"]),
        "artifacts": {
            "feature_freeze": {"path": str(freeze_path), "sha256": freeze_sha256},
            "smoke_report": {"path": str(smoke_path), "sha256": common.file_sha256(smoke_path)},
            "oof_probabilities": {
                "path": str(oof_path), "sha256": common.file_sha256(oof_path),
                "contains_labels": False, "contains_record_ids": False,
                "alignment": f"same row order as the fixed {len(records)}-row PDF-only identity",
            },
            "script": {"path": str(Path(__file__).resolve()), "sha256": common.file_sha256(Path(__file__).resolve())},
        },
        "interpretation": {
            "development_only": True,
            "ninety_percent_reached": results[best]["aggregate"]["accuracy"] >= 0.90,
            "warning": "Best-of-frozen-candidate ranking is small-screen development evidence, not an independent test claim.",
        },
    }
    common.atomic_json(report_path, report)
    print(json.dumps({"output": str(report_path), "best": report["best"], "promotion_gate": report["promotion_gate"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
