#!/usr/bin/env python3
"""Leakage-controlled PDF-derived paper-relation experiments for R3.

The relation branch builds one node per paper directly from the page cache.
Its node and edge attributes are limited to title/front matter, authors, year,
reference strings, DOI/citation keys, research-object phrases, and magnetic
method events.  R3 labels are opened only after the frozen-training and local-
PDF gates.  Every test-paper prediction is computed from labels belonging to
the outer-fold training papers; a paper can never be its own neighbour.

The optional comparison branch recomputes the existing taxonomy-only
original-text model from source inputs.  It does not read an old prediction or
evaluation artifact, and it deliberately excludes the ROCKMAG-derived
positive expert KB.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_pdf_only_strategies as original_experiment  # noqa: E402


DEFAULT_IDENTITY = original_experiment.DEFAULT_IDENTITY
DEFAULT_TARGETS = original_experiment.DEFAULT_TARGETS
DEFAULT_PAGE_CACHE = original_experiment.DEFAULT_PAGE_CACHE
DEFAULT_PDF_DIR = original_experiment.DEFAULT_PDF_DIR
DEFAULT_TAXONOMY = original_experiment.DEFAULT_TAXONOMY
DEFAULT_OUTPUT = ROOT / "tmp/r3_citation_graph/aggregate_result.json"
DEFAULT_OOF_OUTPUT = ROOT / "tmp/r3_citation_graph/oof_probabilities.npz"

N_SPLITS = 5
RANDOM_STATE = 357
SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
YEAR_RE = re.compile(r"(?<!\d)((?:18|19|20)\d{2})(?!\d)")
DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.I)
REFERENCE_HEADING_RE = re.compile(
    r"(?im)^\s*(?:references|bibliography|literature\s+cited|works\s+cited)\s*[:.]?\s*$"
)
FRONT_STOP_RE = re.compile(
    r"(?i)^\s*(?:received|accepted|submitted|abstract|summary|introduction|keywords?)\b"
)
AFFILIATION_RE = re.compile(
    r"(?i)\b(?:department|university|universit[a-z]+|institute|institut[a-z]+|"
    r"geological survey|academy|laboratory|observatory|college|school of|"
    r"address|present address|research centre|research center)\b"
)
TITLE_SIGNAL_RE = re.compile(
    r"(?i)\b(?:palaeo|paleo|archaeo|magneti[csz]|remanence|remanent|"
    r"magnetization|magnetisation|geomagnetic|polar wander|magnetostrat)"
)

GENERIC_WORDS = {
    "about", "after", "also", "analysis", "been", "between", "both", "data",
    "earth", "from", "geological", "geology", "into", "magnetic", "magnetism",
    "magnetization", "magnetisation", "more", "north", "palaeomagnetic",
    "paleomagnetic", "paper", "results", "rock", "rocks", "south", "study",
    "their", "these", "this", "using", "were", "which", "with",
}
AUTHOR_STOPWORDS = {
    "Abstract", "Academy", "Address", "American", "Applied", "Australia",
    "Canada", "Centre", "College", "Department", "Earth", "Editorial",
    "England", "Formation", "Geological", "Geophysics", "Institute", "Journal",
    "Laboratory", "London", "Magnetic", "National", "Physics", "Research",
    "Received", "School", "Science", "Sciences", "Summary", "Survey",
    "University", "Western",
}
OBJECT_SUFFIX = (
    r"Formation|Group|Member|Complex|Province|Basin|Craton|Shield|Terrane|"
    r"Supergroup|Series|Plateau|Peninsula|Island|Islands|Lake|River|Mountain|"
    r"Mountains|Range|Dykes?|Dikes?|Sills?|Basalts?|Granites?|Gabbros?|"
    r"Limestones?|Sandstones?|Red\s+Beds?"
)
OBJECT_RE = re.compile(
    rf"\b((?:[A-Z][A-Za-z'\-]+\s+){{1,4}}(?:{OBJECT_SUFFIX}))\b"
)

METHOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("irm", re.compile(r"(?i)\b(?:isothermal remanent magneti[sz]ation|IRM|SIRM)\b")),
    ("hysteresis", re.compile(r"(?i)\b(?:hystere\w+|back[- ]field|coerciv\w+)\b")),
    ("thermomagnetic", re.compile(r"(?i)\b(?:thermomagnetic|Curie\s+(?:point|temperature)|J[sr]?[- ]?T)\b")),
    ("susceptibility_temperature", re.compile(r"(?i)\b(?:susceptibility[- ]temperature|k[- ]?T|chi[- ]?T)\b")),
    ("microscopy", re.compile(r"(?i)\b(?:reflected[- ]light|polished section|microscop\w+|SEM|electron microprobe)\b")),
    ("lowrie_three_axis", re.compile(r"(?i)\b(?:Lowrie|three[- ]axis|3[- ]axis)\b")),
    ("magnetic_carrier", re.compile(r"(?i)\b(?:carrier|carried)\s+(?:of|by)\b")),
    ("magnetite", re.compile(r"(?i)\b(?:titano)?magnetite\b")),
    ("hematite", re.compile(r"(?i)\bhematite\b")),
    ("maghemite", re.compile(r"(?i)\bmaghemite\b")),
    ("pyrrhotite", re.compile(r"(?i)\bpyrrhotite\b")),
    ("goethite", re.compile(r"(?i)\bgoethite\b")),
    ("domain_state", re.compile(r"(?i)\b(?:single|pseudo[- ]single|multi)[- ]domain\b|\b(?:SD|PSD|MD)\b")),
    ("saturation", re.compile(r"(?i)\b(?:saturat\w+|remanent coerciv\w+)\b")),
    ("unmixing", re.compile(r"(?i)\b(?:component unmix\w+|coercivity spectrum|end[- ]member)\b")),
    ("rock_magnetic", re.compile(r"(?i)\b(?:rock[- ]magnetic|magnetic mineralog\w+)\b")),
    ("forc", re.compile(r"(?i)\b(?:FORC|first[- ]order reversal curve)\b")),
    ("day_plot", re.compile(r"(?i)\b(?:Day plot|Day diagram|Mrs/Ms|Bcr/Bc)\b")),
    ("arm", re.compile(r"(?i)\b(?:anhysteretic remanent magneti[sz]ation|ARM)\b")),
    ("susceptibility", re.compile(r"(?i)\bmagnetic susceptibility\b")),
    ("magnetometer", re.compile(r"(?i)\b(?:VSM|AGM|vibrating sample magnetometer|alternating gradient magnetometer)\b")),
    ("xrd_mossbauer", re.compile(r"(?i)\b(?:X[- ]?ray diffraction|XRD|M(?:o|\u00f6)ssbauer)\b")),
    ("oxide_petrology", re.compile(r"(?i)\b(?:opaque mineral|iron[- ]oxide|oxide petrograph\w+)\b")),
    ("af_demagnetization", re.compile(r"(?i)\b(?:alternating[- ]field|a\.?f\.?)\s+(?:clean\w+|demagneti[sz]\w+)\b")),
    ("thermal_demagnetization", re.compile(r"(?i)\b(?:thermal|thermally)\s+(?:clean\w+|demagneti[sz]\w+)\b")),
    ("chemical_demagnetization", re.compile(r"(?i)\b(?:chemical|acid[- ]leach\w+)\s+(?:clean\w+|demagneti[sz]\w+|treatment)\b")),
    ("unblocking_temperature", re.compile(r"(?i)\b(?:unblocking|blocking) temperatures?\b")),
    ("anisotropy", re.compile(r"(?i)\b(?:anisotropy of magnetic susceptibility|AMS|AARM|ATRM)\b")),
    ("nrm", re.compile(r"(?i)\b(?:natural remanent magneti[sz]ation|NRM)\b")),
)


@dataclass(frozen=True)
class PaperProfile:
    refno: str
    title: str
    authors: frozenset[str]
    year: int | None
    source_dois: frozenset[str]
    citation_dois: frozenset[str]
    citation_keys: frozenset[str]
    citation_fingerprints: frozenset[str]
    cited_surnames: frozenset[str]
    research_objects: frozenset[str]
    series_terms: frozenset[str]
    method_events: frozenset[str]
    method_event_counts: tuple[int, ...]
    relation_document: str
    reference_count: int


def clean_text(value: Any) -> str:
    return SPACE_RE.sub(" ", str(value or "").replace("\x00", " ")).strip()


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def normalize_doi(value: str) -> str:
    return value.casefold().rstrip(".,;:)]}")


def split_references(text: str) -> tuple[str, str]:
    matches = list(REFERENCE_HEADING_RE.finditer(text))
    eligible = [match for match in matches if match.start() >= int(len(text) * 0.30)]
    if not eligible:
        return text, ""
    boundary = eligible[0].start()
    return text[:boundary], text[eligible[0].end():]


def publication_year(first_page: str) -> int | None:
    for raw in YEAR_RE.findall(first_page[:1800]):
        year = int(raw)
        if 1800 <= year <= 2026:
            return year
    return None


def title_and_author_block(first_page: str) -> tuple[str, str]:
    lines = [clean_text(line) for line in first_page.splitlines()]
    lines = [line for line in lines if line]
    stop = next(
        (index for index, line in enumerate(lines) if FRONT_STOP_RE.search(line)),
        min(len(lines), 24),
    )
    front = lines[:stop]
    start = next(
        (
            index
            for index, line in enumerate(front)
            if TITLE_SIGNAL_RE.search(line)
            and "journal" not in line.casefold()
            and "geophys." not in line.casefold()
        ),
        1 if len(front) > 1 else 0,
    )
    title_lines: list[str] = []
    author_start = len(front)
    for index in range(start, min(len(front), start + 7)):
        line = front[index]
        if title_lines and AFFILIATION_RE.search(line):
            author_start = index
            break
        if title_lines and re.search(r"(?:\b[A-Z]\.?\s*){1,4}[A-Z][a-z]{2,}", line):
            author_start = index
            break
        title_lines.append(line)
        author_start = index + 1
        if len(" ".join(title_lines)) >= 240:
            break
    title = clean_text(" ".join(title_lines))[:500]
    author_block = clean_text(" ".join(front[author_start:]))[:1800]
    return title, author_block


def extract_authors(author_block: str) -> frozenset[str]:
    candidates: set[str] = set()
    patterns = (
        re.compile(r"\b(?:[A-Z]\.?\s*){1,4}([A-Z][A-Za-z'\-]{2,})\b"),
        re.compile(r"\b[A-Z][a-z'\-]{2,}\s+(?:[A-Z]\.?\s*){0,3}([A-Z][A-Za-z'\-]{2,})\b"),
    )
    for pattern in patterns:
        for match in pattern.finditer(author_block):
            raw = match.group(1)
            if raw in AUTHOR_STOPWORDS:
                continue
            token = normalize_token(raw)
            if len(token) >= 3:
                candidates.add(token)
    return frozenset(sorted(candidates))


def reference_entries(reference_text: str) -> list[str]:
    lines = [clean_text(line) for line in reference_text.splitlines()]
    entries: list[str] = []
    current: list[str] = []
    for line in lines:
        if not line:
            continue
        begins = bool(
            re.match(r"^(?:\[?\d+[.)\]]?\s+)?[A-Z][A-Za-z'\-]{2,}", line)
            and YEAR_RE.search(line)
        )
        if begins and current:
            entries.append(clean_text(" ".join(current)))
            current = [line]
        else:
            current.append(line)
    if current:
        entries.append(clean_text(" ".join(current)))
    return [entry for entry in entries if YEAR_RE.search(entry)]


def citation_fields(reference_text: str) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    keys: set[str] = set()
    fingerprints: set[str] = set()
    surnames: set[str] = set()
    for entry in reference_entries(reference_text):
        year_match = YEAR_RE.search(entry)
        if year_match is None:
            continue
        year = year_match.group(1)
        prefix = entry[:year_match.start()]
        names = [
            normalize_token(name)
            for name in re.findall(r"\b[A-Z][A-Za-z'\-]{2,}\b", prefix)
            if name not in AUTHOR_STOPWORDS
        ]
        names = [name for name in names if len(name) >= 3]
        for name in names[:8]:
            keys.add(f"{name}:{year}")
            surnames.add(name)
        words = [
            normalize_token(word)
            for word in WORD_RE.findall(entry[:240])
            if word.casefold() not in GENERIC_WORDS
        ]
        words = [word for word in words if len(word) >= 4]
        if words:
            fingerprints.add(f"{year}:" + ":".join(words[:6]))
    return (
        frozenset(sorted(keys)),
        frozenset(sorted(fingerprints)),
        frozenset(sorted(surnames)),
    )


def extract_research_objects(text: str, title: str) -> frozenset[str]:
    values: set[str] = set()
    for match in OBJECT_RE.finditer(text[:80000]):
        phrase = " ".join(normalize_token(token) for token in WORD_RE.findall(match.group(1)))
        phrase = clean_text(phrase)
        if len(phrase) >= 6:
            values.add(phrase)
        if len(values) >= 160:
            break
    title_tokens = {
        normalize_token(token)
        for token in WORD_RE.findall(title)
        if token.casefold() not in GENERIC_WORDS and len(token) >= 4
    }
    values.update(token for token in title_tokens if token)
    return frozenset(sorted(values))


def extract_method_events(body_text: str) -> tuple[frozenset[str], tuple[int, ...], str]:
    events: set[str] = set()
    counts: list[int] = []
    snippets: list[str] = []
    for name, pattern in METHOD_PATTERNS:
        matches = list(pattern.finditer(body_text))[:50]
        counts.append(len(matches))
        if not matches:
            continue
        events.add(name)
        for match in matches[:3]:
            snippets.append(
                clean_text(body_text[max(0, match.start() - 150):match.end() + 210])
            )
    return frozenset(sorted(events)), tuple(counts), " ".join(snippets)[:18000]


def build_profile(refno: str, pages: Sequence[str]) -> PaperProfile:
    first_page = pages[0] if pages else ""
    full_text = "\n".join(pages)
    body_text, reference_text = split_references(full_text)
    title, author_block = title_and_author_block(first_page)
    authors = extract_authors(author_block)
    year = publication_year(first_page)
    body_dois = frozenset(normalize_doi(value) for value in DOI_RE.findall(body_text[:12000]))
    citation_dois = frozenset(normalize_doi(value) for value in DOI_RE.findall(reference_text))
    citation_keys, fingerprints, cited_surnames = citation_fields(reference_text)
    objects = extract_research_objects(body_text, title)
    method_events, method_event_counts, method_snippets = extract_method_events(body_text)
    series_terms = frozenset(
        normalize_token(token)
        for token in WORD_RE.findall(title)
        if token.casefold() not in GENERIC_WORDS and len(token) >= 4
    )
    pseudo_tokens = [
        *(f"author_{value}" for value in sorted(authors)),
        *(f"year_{year // 5 * 5}" for _ in [0] if year is not None),
        *(f"object_{value.replace(' ', '_')}" for value in sorted(objects)),
        *(f"method_{value}" for value in sorted(method_events)),
        *(f"cite_{value.replace(':', '_')}" for value in sorted(citation_keys)[:400]),
    ]
    relation_document = clean_text(
        f"{title} {title} {' '.join(pseudo_tokens)} {body_text[:6500]} {method_snippets}"
    )[:50000]
    return PaperProfile(
        refno=refno,
        title=title,
        authors=authors,
        year=year,
        source_dois=body_dois,
        citation_dois=citation_dois,
        citation_keys=citation_keys,
        citation_fingerprints=fingerprints,
        cited_surnames=cited_surnames,
        research_objects=objects,
        series_terms=series_terms,
        method_events=method_events,
        method_event_counts=method_event_counts,
        relation_document=relation_document,
        reference_count=len(reference_entries(reference_text)),
    )


def load_page_profiles(path: Path, refnos: set[str]) -> tuple[dict[str, PaperProfile], dict[str, str]]:
    profiles: dict[str, PaperProfile] = {}
    paper_text: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            refno = original_experiment.clean(row.get("REFNO"))
            if refno not in refnos:
                continue
            if refno in profiles:
                raise ValueError(f"duplicate page-cache REFNO {refno} at line {line_number}")
            pages = [
                str(page.get("text") or "")
                for page in row.get("pages", [])
                if isinstance(page, Mapping) and clean_text(page.get("text"))
            ]
            if not pages:
                raise ValueError(f"PDF-only REFNO {refno} has no cached page text")
            profiles[refno] = build_profile(refno, pages)
            paper_text[refno] = "\n\n".join(
                f"[[PAGE {index}]]\n{text}" for index, text in enumerate(pages, start=1)
            )
    missing = sorted(refnos - profiles.keys())
    if missing:
        raise ValueError(f"page cache is missing PDF-only REFNOs: {missing[:10]}")
    return profiles, paper_text


def paper_members(records: Sequence[dict[str, str]]) -> tuple[list[str], dict[str, np.ndarray]]:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    refnos = sorted(set(str(value) for value in groups))
    return refnos, {refno: np.flatnonzero(groups == refno) for refno in refnos}


def paper_labels_and_weights(
    refnos: Sequence[str], members: Mapping[str, np.ndarray], row_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(
        [int(row_labels[members[refno]].mean() >= 0.5) for refno in refnos], dtype=np.int8
    )
    weights = np.asarray([len(members[refno]) for refno in refnos], dtype=np.float64)
    return labels, weights


def weighted_set_similarity(
    left: Sequence[PaperProfile],
    right: Sequence[PaperProfile],
    fields: Sequence[tuple[str, float]],
) -> np.ndarray:
    output = np.zeros((len(left), len(right)), dtype=np.float32)
    inverted: dict[tuple[str, str], list[int]] = defaultdict(list)
    for right_index, profile in enumerate(right):
        for field, _ in fields:
            for value in getattr(profile, field):
                inverted[(field, value)].append(right_index)
    normalizers = np.zeros((len(left), len(right)), dtype=np.float32)
    for field, weight in fields:
        right_sizes = np.asarray(
            [max(1, len(getattr(profile, field))) for profile in right], dtype=np.float32
        )
        for left_index, profile in enumerate(left):
            values = getattr(profile, field)
            if not values:
                continue
            left_size = float(len(values))
            normalizers[left_index] += weight * np.sqrt(left_size * right_sizes)
            for value in values:
                for right_index in inverted.get((field, value), ()):
                    output[left_index, right_index] += weight
    return np.divide(output, normalizers, out=np.zeros_like(output), where=normalizers > 0)


def author_series_similarity(
    left: Sequence[PaperProfile], right: Sequence[PaperProfile]
) -> np.ndarray:
    output = weighted_set_similarity(
        left,
        right,
        (
            ("authors", 4.0),
            ("research_objects", 2.0),
            ("series_terms", 1.5),
            ("method_events", 0.75),
        ),
    )
    for left_index, lhs in enumerate(left):
        if lhs.year is None:
            continue
        for right_index, rhs in enumerate(right):
            if rhs.year is None:
                continue
            gap = abs(lhs.year - rhs.year)
            if gap <= 2:
                output[left_index, right_index] += 0.06 * (3 - gap) / 3.0
    return np.clip(output, 0.0, 1.0)


def citation_similarity(
    left: Sequence[PaperProfile], right: Sequence[PaperProfile]
) -> np.ndarray:
    overlap = weighted_set_similarity(
        left,
        right,
        (
            ("citation_dois", 6.0),
            ("citation_fingerprints", 3.0),
            ("citation_keys", 2.0),
            ("cited_surnames", 0.35),
        ),
    )
    # Explicit directed citation edges: a test paper cites a training paper's
    # DOI or first-author/year identity.  All endpoint attributes are textual.
    for left_index, lhs in enumerate(left):
        for right_index, rhs in enumerate(right):
            linked = bool(lhs.citation_dois & rhs.source_dois)
            if not linked and rhs.year is not None:
                linked = any(
                    f"{author}:{rhs.year}" in lhs.citation_keys for author in rhs.authors
                )
            if linked:
                overlap[left_index, right_index] = max(
                    overlap[left_index, right_index], 0.90
                )
    return overlap


def text_similarity(
    train: Sequence[PaperProfile], test: Sequence[PaperProfile]
) -> tuple[np.ndarray, int]:
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.997,
        max_features=120000,
        sublinear_tf=True,
        strip_accents="unicode",
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform([profile.relation_document for profile in train])
    test_matrix = vectorizer.transform([profile.relation_document for profile in test])
    similarity = (test_matrix @ train_matrix.T).toarray().astype(np.float32)
    return similarity, len(vectorizer.vocabulary_)


def relation_matrices(
    train: Sequence[PaperProfile], test: Sequence[PaperProfile]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    text, vocabulary_size = text_similarity(train, test)
    citation = citation_similarity(test, train)
    author_series = author_series_similarity(test, train)
    combined = np.clip(0.48 * text + 0.29 * citation + 0.23 * author_series, 0.0, 1.0)
    return {
        "text": text,
        "citation": citation,
        "author_series": author_series,
        "combined": combined,
    }, {
        "vocabulary_size": vocabulary_size,
        "citation_connected_test_papers": int((citation.max(axis=1) > 0).sum()),
        "author_series_connected_test_papers": int((author_series.max(axis=1) > 0).sum()),
        "explicit_citation_strength_edges": int((citation >= 0.90).sum()),
    }


def knn_probability(
    similarity: np.ndarray,
    train_labels: np.ndarray,
    *,
    k: int,
    prior: float,
    prior_strength: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    if similarity.shape[1] == 0:
        return np.full(similarity.shape[0], prior), np.zeros(similarity.shape[0])
    k = min(k, similarity.shape[1])
    selected = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
    values = np.take_along_axis(similarity, selected, axis=1)
    labels = train_labels[selected]
    weights = np.square(np.maximum(values, 0.0))
    totals = weights.sum(axis=1)
    probabilities = (weights * labels).sum(axis=1) + prior_strength * prior
    probabilities /= totals + prior_strength
    probabilities[totals <= 1e-10] = prior
    return probabilities, similarity.max(axis=1)


def relation_count_features(profiles: Sequence[PaperProfile]) -> np.ndarray:
    values: list[list[float]] = []
    for profile in profiles:
        values.append(
            [
                math.log1p(len(profile.authors)),
                math.log1p(profile.reference_count),
                math.log1p(len(profile.citation_keys)),
                math.log1p(len(profile.citation_dois)),
                math.log1p(len(profile.research_objects)),
                math.log1p(len(profile.method_events)),
                float(profile.year or 0) / 2100.0,
                *(float(name in profile.method_events) for name, _ in METHOD_PATTERNS),
                *(math.log1p(value) for value in profile.method_event_counts),
                *(
                    float(profile.year is not None and lower <= profile.year < lower + 5)
                    for lower in range(1940, 2025, 5)
                ),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def leave_one_out_relation_features(
    profiles: Sequence[PaperProfile], labels: np.ndarray, prior: float
) -> tuple[np.ndarray, dict[str, Any]]:
    matrices, audit = relation_matrices(profiles, profiles)
    columns: list[np.ndarray] = []
    for name in ("text", "citation", "author_series", "combined"):
        matrix = matrices[name].copy()
        np.fill_diagonal(matrix, 0.0)
        probability, maximum = knn_probability(
            matrix, labels, k=15 if name != "citation" else 25, prior=prior
        )
        columns.extend((probability, maximum))
    columns.append(relation_count_features(profiles).T)
    flat: list[np.ndarray] = []
    for value in columns:
        if value.ndim == 1:
            flat.append(value[:, None])
        else:
            flat.extend(value[index][:, None] for index in range(value.shape[0]))
    return np.hstack(flat), audit


def outer_relation_predictions(
    train: Sequence[PaperProfile],
    test: Sequence[PaperProfile],
    train_labels: np.ndarray,
    train_weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    prior = float(np.average(train_labels, weights=train_weights))
    matrices, audit = relation_matrices(train, test)
    outputs: dict[str, np.ndarray] = {}
    maxima: dict[str, np.ndarray] = {}
    for name in ("text", "citation", "author_series", "combined"):
        outputs[f"{name}_knn"] , maxima[name] = knn_probability(
            matrices[name],
            train_labels,
            k=15 if name != "citation" else 25,
            prior=prior,
        )

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform([profile.relation_document for profile in train])
    test_matrix = vectorizer.transform([profile.relation_document for profile in test])
    text_model = LogisticRegression(
        C=1.5, solver="liblinear", max_iter=1200, random_state=RANDOM_STATE
    )
    text_model.fit(train_matrix, train_labels, sample_weight=train_weights)
    outputs["metadata_text_logistic"] = text_model.predict_proba(test_matrix)[:, 1]

    structured_train = relation_count_features(train)
    structured_test = relation_count_features(test)
    method_model = GradientBoostingClassifier(
        n_estimators=240,
        learning_rate=0.03,
        max_depth=2,
        min_samples_leaf=10,
        random_state=RANDOM_STATE,
    )
    method_model.fit(structured_train, train_labels, sample_weight=train_weights)
    outputs["method_event_gradient_boosting"] = method_model.predict_proba(
        structured_test
    )[:, 1]
    method_hist = HistGradientBoostingClassifier(
        max_iter=220,
        learning_rate=0.04,
        max_leaf_nodes=7,
        min_samples_leaf=14,
        l2_regularization=0.8,
        random_state=RANDOM_STATE,
    )
    method_hist.fit(structured_train, train_labels, sample_weight=train_weights)
    outputs["method_event_hist_gradient_boosting"] = method_hist.predict_proba(
        structured_test
    )[:, 1]

    loo_features, loo_audit = leave_one_out_relation_features(train, train_labels, prior)
    outer_columns: list[np.ndarray] = []
    for name in ("text", "citation", "author_series", "combined"):
        outer_columns.extend((outputs[f"{name}_knn"][:, None], maxima[name][:, None]))
    outer_columns.append(relation_count_features(test))
    outer_features = np.hstack(outer_columns)
    stacker = LogisticRegression(
        C=0.5, solver="liblinear", max_iter=1000, random_state=RANDOM_STATE
    )
    stacker.fit(loo_features, train_labels, sample_weight=train_weights)
    outputs["relation_stacker"] = stacker.predict_proba(outer_features)[:, 1]
    outputs["relation_predefined_fusion"] = (
        0.10 * outputs["text_knn"]
        + 0.05 * outputs["citation_knn"]
        + 0.05 * outputs["author_series_knn"]
        + 0.10 * outputs["combined_knn"]
        + 0.15 * outputs["metadata_text_logistic"]
        + 0.20 * outputs["relation_stacker"]
        + 0.20 * outputs["method_event_gradient_boosting"]
        + 0.15 * outputs["method_event_hist_gradient_boosting"]
    )
    audit.update(
        {
            "char_vocabulary_size": len(vectorizer.vocabulary_),
            "outer_train_label_count": int(len(train_labels)),
            "leave_one_out_training_self_edges_removed": True,
            "leave_one_out_stacker_audit": loo_audit,
        }
    )
    return outputs, audit


def metric_record(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predicted = (probabilities >= 0.5).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    return {
        "rows": int(len(labels)),
        "coverage": 1.0,
        "accuracy": round(float(accuracy_score(labels, predicted)), 8),
        "macro_f1": round(float(f1_score(labels, predicted, average="macro")), 8),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predicted)), 8),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def row_probabilities(
    paper_probability: Mapping[str, float], records: Sequence[dict[str, str]]
) -> np.ndarray:
    return np.asarray([paper_probability[record["REFNO"]] for record in records])


def paper_metric(
    refnos: Sequence[str], labels: np.ndarray, probabilities: Mapping[str, float]
) -> dict[str, Any]:
    return metric_record(labels, np.asarray([probabilities[refno] for refno in refnos]))


def subset_metric(labels: np.ndarray, probabilities: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    return metric_record(labels[mask], probabilities[mask])


def paper_majority_oracle_accuracy(
    refnos: Sequence[str], members: Mapping[str, np.ndarray], row_labels: np.ndarray
) -> float:
    correct = 0
    for refno in refnos:
        counts = np.bincount(row_labels[members[refno]], minlength=2)
        correct += int(counts.max())
    return float(correct / len(row_labels))


def write_probability_npz(path: Path, probabilities: Mapping[str, np.ndarray]) -> None:
    forbidden = re.compile(r"(?i)(?:label|answer|refno|resultno|target)")
    invalid = sorted(name for name in probabilities if forbidden.search(name))
    if invalid:
        raise ValueError(f"OOF probability array names contain forbidden fields: {invalid}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(value, dtype=np.float32)
                for name, value in sorted(probabilities.items())
            },
        )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--without-original-baseline", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--oof-output", type=Path, default=DEFAULT_OOF_OUTPUT)
    args = parser.parse_args(argv)
    original_experiment.require_experiment_dependencies()

    required = {
        "training_identity": args.identity.resolve(),
        "targets": args.targets.resolve(),
        "page_cache": args.page_cache.resolve(),
        "taxonomy": args.taxonomy.resolve(),
    }
    for label, path in required.items():
        if not path.is_file():
            raise SystemExit(f"missing {label}: {path}")
    records, coverage = original_experiment.load_pdf_only_identity(
        required["training_identity"], args.pdf_dir.resolve()
    )
    labels_by_resultno = original_experiment.load_training_labels(
        required["targets"], {record["RESULTNO"] for record in records}
    )
    row_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    row_groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    refnos, members = paper_members(records)
    profiles, paper_text = load_page_profiles(required["page_cache"], set(refnos))
    paper_labels, paper_weights = paper_labels_and_weights(refnos, members, row_labels)
    paper_position = {refno: index for index, refno in enumerate(refnos)}

    strategy_paper_probability: dict[str, dict[str, float]] = defaultdict(dict)
    fold_test_rows: list[np.ndarray] = []
    fold_audits: list[dict[str, Any]] = []
    splitter = GroupKFold(n_splits=N_SPLITS)
    for fold, (train_rows, test_rows) in enumerate(
        splitter.split(np.zeros(len(records)), row_labels, row_groups), start=1
    ):
        train_refnos = sorted(set(str(row_groups[index]) for index in train_rows))
        test_refnos = sorted(set(str(row_groups[index]) for index in test_rows))
        overlap = set(train_refnos) & set(test_refnos)
        if overlap:
            raise AssertionError(f"outer fold {fold} REFNO leakage: {sorted(overlap)[:5]}")
        train_profiles = [profiles[refno] for refno in train_refnos]
        test_profiles = [profiles[refno] for refno in test_refnos]
        train_indices = np.asarray([paper_position[refno] for refno in train_refnos])
        outputs, graph_audit = outer_relation_predictions(
            train_profiles,
            test_profiles,
            paper_labels[train_indices],
            paper_weights[train_indices],
        )
        for name, values in outputs.items():
            strategy_paper_probability[name].update(zip(test_refnos, map(float, values)))
        fold_test_rows.append(test_rows)
        fold_audits.append(
            {
                "fold": fold,
                "train_papers": len(train_refnos),
                "test_papers": len(test_refnos),
                "train_rows": int(len(train_rows)),
                "test_rows": int(len(test_rows)),
                "train_test_refno_overlap": 0,
                "test_labels_available_to_predictor": False,
                "graph": graph_audit,
            }
        )
        print(
            f"relation fold {fold}/{N_SPLITS}: papers={len(test_refnos)}, "
            f"connected={graph_audit['citation_connected_test_papers']}",
            flush=True,
        )

    for name, values in strategy_paper_probability.items():
        missing = set(refnos) - values.keys()
        if missing:
            raise AssertionError(f"incomplete relation OOF strategy {name}: {sorted(missing)[:5]}")

    original_probabilities: dict[str, np.ndarray] = {}
    if not args.without_original_baseline:
        concepts, _ = original_experiment.load_knowledge_patterns(required["taxonomy"], None)
        original_features = original_experiment.build_features(records, paper_text, concepts)
        base_probability, original_folds, _, _ = original_experiment.run_grouped_oof(
            records, row_labels, original_features
        )
        if any(not np.array_equal(left, right) for left, right in zip(fold_test_rows, original_folds)):
            raise AssertionError("relation and original-model outer folds differ")
        original_strategies = original_experiment.build_strategies(
            records, original_features, base_probability
        )
        original_probabilities = {
            "original_taxonomy_unscoped_current_best": original_strategies[
                "multi_all_article_100"
            ]["probabilities"],
            "original_taxonomy_scope_gated": original_strategies[
                "scope_gated_article_50"
            ]["probabilities"],
        }

    row_strategy_probabilities: dict[str, np.ndarray] = {
        name: row_probabilities(values, records)
        for name, values in strategy_paper_probability.items()
    }
    row_strategy_probabilities.update(original_probabilities)
    for baseline_name, baseline_probability in original_probabilities.items():
        for relation_name in (
            "relation_stacker",
            "method_event_gradient_boosting",
            "relation_predefined_fusion",
        ):
            relation_probability = row_strategy_probabilities[relation_name]
            short_name = relation_name.replace("method_event_gradient_boosting", "method_gb")
            for relation_weight in (0.15, 0.25, 0.35, 0.50):
                name = (
                    f"{baseline_name}_plus_{short_name}_{int(relation_weight * 100):02d}"
                )
                row_strategy_probabilities[name] = (
                    (1.0 - relation_weight) * baseline_probability
                    + relation_weight * relation_probability
                )
            confidence = np.abs(relation_probability - 0.5) * 2.0
            gate = confidence >= 0.70
            gated = baseline_probability.copy()
            gated[gate] = (
                0.65 * baseline_probability[gate] + 0.35 * relation_probability[gate]
            )
            row_strategy_probabilities[
                f"{baseline_name}_{short_name}_confidence_gate"
            ] = gated

    homogeneous = np.zeros(len(records), dtype=bool)
    mixed = np.zeros(len(records), dtype=bool)
    for refno in refnos:
        target = homogeneous if len(set(row_labels[members[refno]])) == 1 else mixed
        target[members[refno]] = True
    strategy_results: dict[str, Any] = {}
    for name, probability in row_strategy_probabilities.items():
        strategy_results[name] = {
            "family": (
                "pdf_relation"
                if name in strategy_paper_probability
                else "recomputed_original_text" if name in original_probabilities else "fusion"
            ),
            "aggregate": metric_record(row_labels, probability),
            "folds": [
                {"fold": fold, **metric_record(row_labels[index], probability[index])}
                for fold, index in enumerate(fold_test_rows, start=1)
            ],
            "diagnostic_subsets": {
                "homogeneous_paper_rows": subset_metric(row_labels, probability, homogeneous),
                "mixed_paper_rows": subset_metric(row_labels, probability, mixed),
            },
        }
    ranking = sorted(
        strategy_results,
        key=lambda name: (
            strategy_results[name]["aggregate"]["accuracy"],
            strategy_results[name]["aggregate"]["macro_f1"],
        ),
        reverse=True,
    )
    best_relation = max(
        strategy_paper_probability,
        key=lambda name: strategy_results[name]["aggregate"]["accuracy"],
    )
    profile_audit = {
        "papers": len(profiles),
        "title_nonempty": sum(bool(profile.title) for profile in profiles.values()),
        "author_nonempty": sum(bool(profile.authors) for profile in profiles.values()),
        "year_nonempty": sum(profile.year is not None for profile in profiles.values()),
        "references_nonempty": sum(profile.reference_count > 0 for profile in profiles.values()),
        "citation_doi_nonempty": sum(bool(profile.citation_dois) for profile in profiles.values()),
        "citation_key_nonempty": sum(bool(profile.citation_keys) for profile in profiles.values()),
        "research_object_nonempty": sum(bool(profile.research_objects) for profile in profiles.values()),
        "method_event_nonempty": sum(bool(profile.method_events) for profile in profiles.values()),
    }
    oof_output = args.oof_output.resolve()
    write_probability_npz(oof_output, row_strategy_probabilities)
    oracle_accuracy = paper_majority_oracle_accuracy(refnos, members, row_labels)
    result = {
        "schema_version": "r3_pdf_citation_relation_grouped_oof_v1",
        "status": "complete",
        "scope": "development-only frozen training partition; local PDF required",
        "guardrails": {
            "cross_validation": "5-fold GroupKFold by REFNO",
            "relation_node_source": "page cache only",
            "relation_attributes": [
                "title/front matter", "authors", "publication year", "reference strings",
                "DOI and author-year citation keys", "research-object phrases", "method events",
            ],
            "relation_branch_does_not_use_rockname": True,
            "test_labels_used_for_prediction": False,
            "train_test_refno_overlap_every_fold": 0,
            "forbidden_sources_not_read": [
                "PMAGRESULT", "ROCKMAG source cells", "positive ROCKMAG expert KB",
                "prior predictions", "prior evaluations", "held-out R3 labels",
            ],
            "original_comparison": (
                "recomputed from page text, ROCKNAME, and taxonomy; no stored prediction read"
                if original_probabilities else "disabled"
            ),
            "strategy_ranking_warning": "development OOF strategy ranking is post-hoc and requires external validation",
        },
        "inputs": {
            **{
                label: {"path": str(path), "sha256": original_experiment.file_sha256(path)}
                for label, path in required.items()
            },
            "pdf_dir": str(args.pdf_dir.resolve()),
        },
        "data": {
            **coverage,
            "row_label_counts": dict(sorted(Counter(map(int, row_labels)).items())),
            "paper_majority_label_counts": dict(sorted(Counter(map(int, paper_labels)).items())),
            "homogeneous_papers": sum(len(set(row_labels[members[refno]])) == 1 for refno in refnos),
            "mixed_papers": sum(len(set(row_labels[members[refno]])) > 1 for refno in refnos),
            "profile_extraction": profile_audit,
        },
        "fold_leakage_audit": fold_audits,
        "strategies": strategy_results,
        "ranking_by_full_coverage_row_accuracy": ranking,
        "best_relation_only": {
            "strategy": best_relation,
            **strategy_results[best_relation]["aggregate"],
        },
        "best_full_coverage": {
            "strategy": ranking[0],
            **strategy_results[ranking[0]]["aggregate"],
        },
        "paper_majority_metrics": {
            name: paper_metric(refnos, paper_labels, values)
            for name, values in strategy_paper_probability.items()
        },
        "interpretation": {
            "full_coverage_90_percent_reached": bool(
                strategy_results[ranking[0]]["aggregate"]["accuracy"] >= 0.90
            ),
            "relation_only_90_percent_reached": bool(
                strategy_results[best_relation]["aggregate"]["accuracy"] >= 0.90
            ),
            "paper_majority_oracle_row_accuracy": round(oracle_accuracy, 8),
            "paper_majority_oracle_definition": (
                "sum of the larger R3 class count within each REFNO divided by evaluated rows"
            ),
            "mixed_paper_target_attribution_remains_unresolved": True,
        },
        "artifacts": {
            "script": str(Path(__file__).resolve()),
            "script_sha256": original_experiment.file_sha256(Path(__file__).resolve()),
            "output": str(args.output.resolve()),
            "oof_probability_npz": {
                "path": str(oof_output),
                "sha256": original_experiment.file_sha256(oof_output),
                "row_count_per_array": len(records),
                "array_names": sorted(row_strategy_probabilities),
                "contains_labels": False,
                "contains_refno": False,
                "contains_resultno": False,
            },
        },
    }
    original_experiment.atomic_json(args.output.resolve(), result)
    print(json.dumps(result["best_relation_only"], ensure_ascii=False, indent=2))
    print(json.dumps(result["best_full_coverage"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
