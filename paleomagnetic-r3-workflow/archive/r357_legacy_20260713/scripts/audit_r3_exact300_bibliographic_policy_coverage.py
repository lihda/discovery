#!/usr/bin/env python3
"""Audit label-free bibliographic-policy coverage on canonical exact300 R3.

Every bibliographic value is extracted from the frozen original-paper page
cache.  The audit uses the canonical REFNO fold manifest and, for each outer
fold, removes every auxiliary-pool row sharing a test REFNO before measuring
whether test-paper entities recur in the remaining papers.  It never opens R3
or any prior prediction/evaluation artifact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
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

import experiment_r3_citation_graph as citation  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402


EXACT_IDENTITY = auxiliary.EXACT_IDENTITY
FULL_IDENTITY = auxiliary.FULL_IDENTITY
SCREENING_PROTOCOL = auxiliary.SCREENING_PROTOCOL
FOLD_MANIFEST = auxiliary.FOLD_MANIFEST
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/bibliographic_policy_v2"
OUTPUT = OUTPUT_DIR / "coverage_audit.json"
FREEZE = OUTPUT_DIR / "coverage_protocol.json"

CANONICAL_FOLD_MANIFEST_SHA256 = auxiliary.FOLD_MANIFEST_SHA256
EXPECTED_EXACT_ROWS = auxiliary.EXPECTED_EXACT_ROWS
EXPECTED_EXACT_PAPERS = auxiliary.EXPECTED_EXACT_PAPERS
EXPECTED_FULL_ROWS = auxiliary.EXPECTED_FULL_PDF_ROWS
EXPECTED_FULL_PAPERS = auxiliary.EXPECTED_FULL_PDF_PAPERS

AUTHOR_STABLE_SUPPORT = 2
JOURNAL_DECADE_STABLE_SUPPORT = 15
PUBLISHER_DECADE_STABLE_SUPPORT = 15
DOI_PREFIX_STABLE_SUPPORT = 10
SHARED_CITATION_STABLE_SUPPORT = 3

MIN_TITLE_EXTRACTION_COVERAGE = 0.90
MIN_YEAR_EXTRACTION_COVERAGE = 0.75
MIN_STABLE_POLICY_ROW_COVERAGE = 0.75
MIN_SPECIFIC_RELATION_ROW_COVERAGE = 0.35
MIN_FOLD_STABLE_POLICY_COVERAGE = 0.65

DOI_PREFIX_PUBLISHER = {
    "10.1016": "elsevier",
    "10.1029": "agu",
    "10.1038": "nature",
    "10.1046": "wiley",
    "10.1080": "taylor_francis",
    "10.1093": "oxford_university_press",
    "10.1111": "wiley",
    "10.1139": "canadian_science_publishing",
    "10.1144": "geological_society_london",
    "10.1186": "springer_nature",
    "10.1007": "springer",
}
PUBLISHER_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("agu", re.compile(r"American Geophysical Union|AGU Publications?|agupubs", re.I)),
    ("elsevier", re.compile(r"\bElsevier\b|ScienceDirect", re.I)),
    ("wiley", re.compile(r"John Wiley|Wiley(?:-Blackwell)?|Blackwell Publishing", re.I)),
    ("oxford_university_press", re.compile(r"Oxford University Press|academic\.oup\.com", re.I)),
    ("springer", re.compile(r"\bSpringer(?:-Verlag)?\b", re.I)),
    ("nature", re.compile(r"Nature Publishing|Springer Nature", re.I)),
    ("geological_society_london", re.compile(r"Geological Society(?: of)? London", re.I)),
    ("taylor_francis", re.compile(r"Taylor\s*&\s*Francis", re.I)),
    ("cambridge_university_press", re.compile(r"Cambridge University Press", re.I)),
    ("canadian_science_publishing", re.compile(r"NRC Research Press|Canadian Science Publishing", re.I)),
)
AUTHOR_BOUNDARY_RE = re.compile(
    r"(?i)\b(?:sub[- ]?department|department|university|universit[a-z]+|"
    r"institute|institut[a-z]+|geological survey|academy|laboratory|"
    r"observatory|college|school|research cent(?:re|er)|P\.?\s*O\.?\s*Box|"
    r"postal address|present address|corresponding author|e-?mail|"
    r"street|road|avenue|boulevard)\b"
)
AUTHOR_TOKEN_STOPWORDS = {
    "abstract", "academy", "address", "africa", "american", "applied",
    "australia", "box", "branch", "canada", "cedex", "center", "centre",
    "china", "college", "department", "earth", "editorial", "email",
    "england", "evolution", "federal", "formation", "geological", "geology",
    "geophysical", "geophysics", "hall", "institute", "journal", "key",
    "laboratory", "lane", "lodge", "magnetic", "nacional", "national",
    "normal", "park", "petroleum", "physics", "plateau", "research",
    "resources", "road", "school", "science", "sciences", "section",
    "special", "state", "street", "subdepartment", "summary", "survey",
    "the", "university", "van", "western",
}


@dataclass(frozen=True)
class BibliographicProfile:
    refno: str
    title: str
    authors: frozenset[str]
    year: int | None
    journal: str
    publisher: str | None
    doi_prefixes: frozenset[str]
    source_dois: frozenset[str]
    citation_dois: frozenset[str]
    citation_keys: frozenset[str]
    source_author_year_keys: frozenset[str]


def doi_prefix(value: str) -> str:
    return value.split("/", 1)[0].casefold() if "/" in value else ""


def decade(year: int | None) -> str:
    if year is None:
        return "missing"
    return f"{year // 10 * 10}s"


def extract_publisher(front: str, prefixes: Sequence[str]) -> str | None:
    for prefix in prefixes:
        if prefix in DOI_PREFIX_PUBLISHER:
            return DOI_PREFIX_PUBLISHER[prefix]
    for name, pattern in PUBLISHER_RULES:
        if pattern.search(front[:5000]):
            return name
    return None


def extract_clean_authors(first_page: str) -> frozenset[str]:
    """Extract surnames only from the pre-affiliation part of the author block."""

    _, raw_block = citation.title_and_author_block(first_page)
    boundary = AUTHOR_BOUNDARY_RE.search(raw_block)
    block = raw_block[: boundary.start()] if boundary else raw_block
    block = re.sub(r"[*†‡§\\]+|\b\d{1,2}\b", " ", block)
    candidates: set[str] = set()
    patterns = (
        re.compile(r"\b(?:[A-Z]\.?\s*){1,4}([A-Z][A-Za-z'\-]{2,})\b"),
        re.compile(r"\b([A-Z][A-Za-z'\-]{2,})\s*,\s*(?:[A-Z]\.?\s*){1,4}\b"),
        re.compile(
            r"\b(?:[A-Z][a-z'\-]{2,}\s+){1,3}([A-Z][A-Za-z'\-]{2,})\b"
        ),
    )
    for pattern in patterns:
        for match in pattern.finditer(block):
            token = citation.normalize_token(match.group(1))
            if (
                len(token) >= 3
                and token not in AUTHOR_TOKEN_STOPWORDS
                and not token.isdigit()
            ):
                candidates.add(token)
    return frozenset(sorted(candidates))


def build_bibliographic_profile(refno: str, pages: Sequence[str]) -> BibliographicProfile:
    profile = citation.build_profile(refno, pages)
    front = pages[0] if pages else ""
    prefixes = frozenset(
        prefix
        for prefix in (doi_prefix(value) for value in profile.source_dois)
        if prefix
    )
    publisher = extract_publisher(front, sorted(prefixes))
    authors = extract_clean_authors(front)
    source_keys = frozenset(
        f"{author}:{profile.year}"
        for author in authors
        if profile.year is not None
    )
    return BibliographicProfile(
        refno=refno,
        title=profile.title,
        authors=authors,
        year=profile.year,
        journal=curation.journal_family(front),
        publisher=publisher,
        doi_prefixes=prefixes,
        source_dois=profile.source_dois,
        citation_dois=profile.citation_dois,
        citation_keys=profile.citation_keys,
        source_author_year_keys=source_keys,
    )


def load_profiles(
    page_cache: Path,
    refnos: set[str],
) -> dict[str, BibliographicProfile]:
    profiles: dict[str, BibliographicProfile] = {}
    with page_cache.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            refno = base.clean(row.get("REFNO"))
            if refno not in refnos:
                continue
            if refno in profiles:
                raise ValueError(f"duplicate page-cache REFNO {refno} at line {line_number}")
            pages = [
                str(page.get("text") or "")
                for page in row.get("pages", [])
                if isinstance(page, Mapping) and str(page.get("text") or "").strip()
            ]
            if not pages:
                raise ValueError(f"PDF-backed REFNO {refno} has no page text")
            profiles[refno] = build_bibliographic_profile(refno, pages)
    missing = sorted(refnos - profiles.keys())
    if missing:
        raise ValueError(f"page cache missing PDF-backed profiles: {missing[:10]}")
    return profiles


def support_counter(
    profiles: Sequence[BibliographicProfile],
    field: str,
) -> Counter[str]:
    counter: Counter[str] = Counter()
    for profile in profiles:
        values = getattr(profile, field)
        for value in set(values):
            counter[str(value)] += 1
    return counter


def max_support(values: Sequence[str], counts: Mapping[str, int]) -> int:
    return max((int(counts.get(value, 0)) for value in values), default=0)


def profile_coverage_record(
    profile: BibliographicProfile,
    *,
    author_counts: Mapping[str, int],
    journal_decade_counts: Mapping[str, int],
    publisher_decade_counts: Mapping[str, int],
    doi_prefix_counts: Mapping[str, int],
    citation_doi_counts: Mapping[str, int],
    citation_key_counts: Mapping[str, int],
    train_source_dois: set[str],
    train_source_keys: set[str],
    train_citation_dois: set[str],
    train_citation_keys: set[str],
) -> dict[str, Any]:
    author_support = max_support(sorted(profile.authors), author_counts)
    journal_decade = f"{profile.journal}:{decade(profile.year)}"
    journal_support = int(journal_decade_counts.get(journal_decade, 0))
    publisher_decade = (
        f"{profile.publisher}:{decade(profile.year)}" if profile.publisher else ""
    )
    publisher_support = int(publisher_decade_counts.get(publisher_decade, 0))
    prefix_support = max_support(sorted(profile.doi_prefixes), doi_prefix_counts)
    shared_citation_support = max(
        max_support(sorted(profile.citation_dois), citation_doi_counts),
        max_support(sorted(profile.citation_keys), citation_key_counts),
    )
    outgoing_direct = bool(
        profile.citation_dois & train_source_dois
        or profile.citation_keys & train_source_keys
    )
    incoming_direct = bool(
        profile.source_dois & train_citation_dois
        or profile.source_author_year_keys & train_citation_keys
    )
    direct = outgoing_direct or incoming_direct
    stable_policy = bool(
        author_support >= AUTHOR_STABLE_SUPPORT
        or journal_support >= JOURNAL_DECADE_STABLE_SUPPORT
        or publisher_support >= PUBLISHER_DECADE_STABLE_SUPPORT
        or prefix_support >= DOI_PREFIX_STABLE_SUPPORT
        or shared_citation_support >= SHARED_CITATION_STABLE_SUPPORT
    )
    specific_relation = bool(
        author_support >= AUTHOR_STABLE_SUPPORT
        or direct
        or shared_citation_support >= SHARED_CITATION_STABLE_SUPPORT
    )
    return {
        "title_extracted": bool(profile.title),
        "authors_extracted": bool(profile.authors),
        "year_extracted": profile.year is not None,
        "journal_recognized": profile.journal != "other",
        "publisher_recognized": profile.publisher is not None,
        "source_doi_extracted": bool(profile.source_dois),
        "citation_entities_extracted": bool(profile.citation_dois or profile.citation_keys),
        "recurring_author_support_1": author_support >= 1,
        "recurring_author_support_2": author_support >= 2,
        "recurring_author_support_3": author_support >= 3,
        "journal_decade_support": journal_support,
        "publisher_decade_support": publisher_support,
        "doi_prefix_support": prefix_support,
        "shared_citation_support": shared_citation_support,
        "outgoing_direct_citation": outgoing_direct,
        "incoming_direct_citation": incoming_direct,
        "any_direct_citation": direct,
        "stable_policy_support": stable_policy,
        "specific_relation_support": specific_relation,
    }


def fraction(records: Sequence[Mapping[str, Any]], field: str) -> float:
    return round(sum(bool(record[field]) for record in records) / max(1, len(records)), 8)


def summarize(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    binary_fields = (
        "title_extracted",
        "authors_extracted",
        "year_extracted",
        "journal_recognized",
        "publisher_recognized",
        "source_doi_extracted",
        "citation_entities_extracted",
        "recurring_author_support_1",
        "recurring_author_support_2",
        "recurring_author_support_3",
        "outgoing_direct_citation",
        "incoming_direct_citation",
        "any_direct_citation",
        "stable_policy_support",
        "specific_relation_support",
    )
    return {
        "units": len(records),
        "coverage": {field: fraction(records, field) for field in binary_fields},
        "support_distribution": {
            field: dict(
                sorted(Counter(str(int(record[field])) for record in records).items())
            )
            for field in (
                "journal_decade_support",
                "publisher_decade_support",
                "doi_prefix_support",
                "shared_citation_support",
            )
        },
    }


def fold_coverage(
    fold: int,
    *,
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    fold_ids: np.ndarray,
    profiles: Mapping[str, BibliographicProfile],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    exact_test = np.flatnonzero(fold_ids == fold).astype(np.int64)
    full_lookup = {
        auxiliary.row_key(record): index for index, record in enumerate(full_records)
    }
    train_full, _, test_refnos = auxiliary.auxiliary_split(
        full_records, exact_records, exact_test, full_lookup
    )
    train_refnos = sorted({full_records[int(index)]["REFNO"] for index in train_full})
    train_profiles = [profiles[refno] for refno in train_refnos]
    test_profiles = [profiles[refno] for refno in sorted(test_refnos)]

    author_counts = support_counter(train_profiles, "authors")
    doi_prefix_counts = support_counter(train_profiles, "doi_prefixes")
    citation_doi_counts = support_counter(train_profiles, "citation_dois")
    citation_key_counts = support_counter(train_profiles, "citation_keys")
    journal_decade_counts = Counter(
        f"{profile.journal}:{decade(profile.year)}" for profile in train_profiles
    )
    publisher_decade_counts = Counter(
        f"{profile.publisher}:{decade(profile.year)}"
        for profile in train_profiles
        if profile.publisher is not None
    )
    train_source_dois = set().union(
        *(profile.source_dois for profile in train_profiles)
    )
    train_source_keys = set().union(
        *(profile.source_author_year_keys for profile in train_profiles)
    )
    train_citation_dois = set().union(
        *(profile.citation_dois for profile in train_profiles)
    )
    train_citation_keys = set().union(
        *(profile.citation_keys for profile in train_profiles)
    )
    paper_records = [
        profile_coverage_record(
            profile,
            author_counts=author_counts,
            journal_decade_counts=journal_decade_counts,
            publisher_decade_counts=publisher_decade_counts,
            doi_prefix_counts=doi_prefix_counts,
            citation_doi_counts=citation_doi_counts,
            citation_key_counts=citation_key_counts,
            train_source_dois=train_source_dois,
            train_source_keys=train_source_keys,
            train_citation_dois=train_citation_dois,
            train_citation_keys=train_citation_keys,
        )
        for profile in test_profiles
    ]
    by_refno = dict(zip((profile.refno for profile in test_profiles), paper_records))
    row_records = [
        by_refno[exact_records[int(index)]["REFNO"]] for index in exact_test
    ]
    train_refno_set = set(train_refnos)
    audit = {
        "fold": fold,
        "exact_test_rows": int(len(exact_test)),
        "exact_test_papers": len(test_refnos),
        "outer_train_rows": int(len(train_full)),
        "outer_train_papers": len(train_refno_set),
        "outer_train_test_refno_overlap": len(train_refno_set & test_refnos),
        "paper_level": summarize(paper_records),
        "row_level": summarize(row_records),
    }
    return audit, paper_records, row_records


def protocol() -> dict[str, Any]:
    return {
        "schema_version": "r3_exact300_bibliographic_policy_coverage_protocol_v2",
        "status": "FROZEN_BEFORE_COVERAGE_OBSERVATION",
        "canonical_fold_manifest_sha256": CANONICAL_FOLD_MANIFEST_SHA256,
        "support_thresholds": {
            "recurring_author_training_papers": AUTHOR_STABLE_SUPPORT,
            "journal_decade_training_papers": JOURNAL_DECADE_STABLE_SUPPORT,
            "publisher_decade_training_papers": PUBLISHER_DECADE_STABLE_SUPPORT,
            "doi_prefix_training_papers": DOI_PREFIX_STABLE_SUPPORT,
            "shared_citation_training_papers": SHARED_CITATION_STABLE_SUPPORT,
        },
        "continuation_gate": {
            "minimum_title_extraction_coverage": MIN_TITLE_EXTRACTION_COVERAGE,
            "minimum_year_extraction_coverage": MIN_YEAR_EXTRACTION_COVERAGE,
            "minimum_stable_policy_row_coverage": MIN_STABLE_POLICY_ROW_COVERAGE,
            "minimum_specific_relation_row_coverage": MIN_SPECIFIC_RELATION_ROW_COVERAGE,
            "minimum_each_fold_stable_policy_row_coverage": MIN_FOLD_STABLE_POLICY_COVERAGE,
            "decision": (
                "train only if every threshold passes; otherwise stop before R3 access"
            ),
        },
        "runtime_boundary": {
            "labels_read": False,
            "targets_workbook_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifiers_used_for_alignment_and_refno_isolation_only": True,
        },
        "entity_sources": [
            "title/front matter from original-paper page cache",
            "author surnames from original-paper page cache",
            "publication year from original-paper page cache",
            "journal family from original-paper page cache",
            "publisher or source DOI prefix from original-paper page cache",
            "reference DOI and author-year keys from original-paper page cache",
        ],
        "author_quality_rule": (
            "surname patterns are applied only before the first affiliation/address "
            "boundary; institution and address tokens are filtered"
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=FULL_IDENTITY)
    parser.add_argument("--screening-protocol", type=Path, default=SCREENING_PROTOCOL)
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    screening_protocol = args.screening_protocol.resolve()
    fold_manifest = args.fold_manifest.resolve()
    targets = args.targets.resolve()
    page_cache = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy = args.taxonomy.resolve()
    positive_kb = args.positive_kb.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    frozen_protocol = protocol()
    freeze_path = output_dir / FREEZE.name
    base.atomic_json(freeze_path, frozen_protocol)
    freeze_sha = base.file_sha256(freeze_path)

    input_audit = auxiliary.validate_inputs(
        exact_identity,
        full_identity,
        screening_protocol,
        fold_manifest,
        targets,
        page_cache,
        taxonomy,
        positive_kb,
    )
    if base.file_sha256(fold_manifest) != CANONICAL_FOLD_MANIFEST_SHA256:
        raise ValueError("coverage audit requires canonical exact300 folds")
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(fold_manifest, exact_records)
    refnos = {record["REFNO"] for record in full_records}
    profiles = load_profiles(page_cache, refnos)

    fold_audits: list[dict[str, Any]] = []
    all_paper_records: list[dict[str, Any]] = []
    all_row_records: list[dict[str, Any]] = []
    for fold in range(1, auxiliary.N_SPLITS + 1):
        audit, paper_records, row_records = fold_coverage(
            fold,
            full_records=full_records,
            exact_records=exact_records,
            fold_ids=fold_ids,
            profiles=profiles,
        )
        fold_audits.append(audit)
        all_paper_records.extend(paper_records)
        all_row_records.extend(row_records)

    overall_paper = summarize(all_paper_records)
    overall_row = summarize(all_row_records)
    coverage = overall_row["coverage"]
    gate_checks = {
        "title_extraction": coverage["title_extracted"] >= MIN_TITLE_EXTRACTION_COVERAGE,
        "year_extraction": coverage["year_extracted"] >= MIN_YEAR_EXTRACTION_COVERAGE,
        "stable_policy_row_coverage": (
            coverage["stable_policy_support"] >= MIN_STABLE_POLICY_ROW_COVERAGE
        ),
        "specific_relation_row_coverage": (
            coverage["specific_relation_support"]
            >= MIN_SPECIFIC_RELATION_ROW_COVERAGE
        ),
        "each_fold_stable_policy_row_coverage": all(
            audit["row_level"]["coverage"]["stable_policy_support"]
            >= MIN_FOLD_STABLE_POLICY_COVERAGE
            for audit in fold_audits
        ),
        "all_outer_refno_overlaps_zero": all(
            audit["outer_train_test_refno_overlap"] == 0 for audit in fold_audits
        ),
    }
    passed = all(gate_checks.values())
    extraction = {
        "papers": len(profiles),
        "title_nonempty": sum(bool(profile.title) for profile in profiles.values()),
        "authors_nonempty": sum(bool(profile.authors) for profile in profiles.values()),
        "year_nonempty": sum(profile.year is not None for profile in profiles.values()),
        "recognized_journal": sum(profile.journal != "other" for profile in profiles.values()),
        "recognized_publisher": sum(profile.publisher is not None for profile in profiles.values()),
        "source_doi_nonempty": sum(bool(profile.source_dois) for profile in profiles.values()),
        "citation_entity_nonempty": sum(
            bool(profile.citation_dois or profile.citation_keys)
            for profile in profiles.values()
        ),
        "maximum_authors_one_paper": max(
            len(profile.authors) for profile in profiles.values()
        ),
        "papers_with_more_than_12_authors": sum(
            len(profile.authors) > 12 for profile in profiles.values()
        ),
    }
    report = {
        "schema_version": "r3_exact300_bibliographic_policy_coverage_audit_v2",
        "status": "complete",
        "scope": {
            "evaluation_rows": len(exact_records),
            "evaluation_papers": len({record["REFNO"] for record in exact_records}),
            "auxiliary_rows": len(full_records),
            "auxiliary_papers": len(refnos),
            "full_2172_rows_evaluated": False,
            "full_pool_coverage": full_coverage,
            "exact_coverage": exact_coverage,
        },
        "guardrails": {
            **frozen_protocol["runtime_boundary"],
            "canonical_fold_manifest_sha256": base.file_sha256(fold_manifest),
            "every_outer_test_refno_fully_excluded": True,
            "coverage_protocol_frozen_before_profile_observation": True,
            "coverage_protocol_sha256": freeze_sha,
        },
        "inputs": input_audit,
        "profile_extraction": extraction,
        "folds": fold_audits,
        "overall_paper_level": overall_paper,
        "overall_row_level": overall_row,
        "continuation_gate": {
            "checks": gate_checks,
            "passed": passed,
            "decision": (
                "ENTITY_COVERAGE_SUPPORTS_NESTED_POLICY_EXPERIMENT"
                if passed
                else "STOP_BIBLIOGRAPHIC_POLICY_BRANCH_BEFORE_LABEL_ACCESS"
            ),
            "canonical_accuracy_to_improve": 0.87333333,
            "labels_opened_for_this_audit": False,
        },
        "artifacts": {
            "coverage_protocol": {
                "path": str(freeze_path),
                "sha256": freeze_sha,
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
    }
    output_path = output_dir / OUTPUT.name
    base.atomic_json(output_path, report)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "profile_extraction": extraction,
                "overall_row_coverage": coverage,
                "continuation_gate": report["continuation_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
