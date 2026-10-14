#!/usr/bin/env python3
"""Induce a de-identified R3 terminology graph from ROCKMAG values.

This is an offline knowledge-construction utility, not a scoring component.
It reads exactly one spreadsheet column, aggregates concept frequencies and
co-occurrences, and never emits source rows or record identifiers. Runtime R3
scoring consumes only the frozen aggregate artifact produced by this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/source_metadata/PMAGRESULT.xlsx"
DEFAULT_OUTPUT = ROOT / "data/knowledge/r3_rockmag_lexicon_v1.json"
SOURCE_COLUMN = "ROCKMAG"


# Aliases are intentionally record-independent domain terms. More specific
# expressions precede broad abbreviations so each concept is counted once per
# ROCKMAG value even when several aliases match.
CONCEPTS: tuple[dict[str, Any], ...] = (
    {
        "concept_id": "thin_section_petrography",
        "category": "method",
        "aliases": (r"\bTS\b", r"thin\s+sections?", r"petrograph(?:y|ic)"),
    },
    {
        "concept_id": "optical_microscopy",
        "category": "method",
        "aliases": (r"\bOP\b", r"optical\s+microscop", r"reflected[- ]light"),
    },
    {
        "concept_id": "electron_microscopy_microanalysis",
        "category": "method",
        "aliases": (r"\bSEM\b", r"\bTEM\b", r"\bEDS\b", r"\bEPMA\b", r"microprobe"),
    },
    {
        "concept_id": "xray_diffraction",
        "category": "method",
        "aliases": (r"\bX[- ]?RAY\b", r"\bXRD\b", r"x[- ]ray\s+diffraction"),
    },
    {
        "concept_id": "chemical_analysis",
        "category": "method",
        "aliases": (r"\bCHEM\b", r"chemical\s+analys"),
    },
    {
        "concept_id": "irm_acquisition",
        "category": "method",
        "aliases": (r"\bS?IRM\b", r"isothermal\s+remanent\s+magneti[sz]ation"),
    },
    {
        "concept_id": "three_axis_thermal_demagnetization",
        "category": "method",
        "aliases": (r"3[- ]axis\s+Jrs[- ]?T", r"three[- ]axis", r"\bLowrie\b"),
    },
    {
        "concept_id": "thermomagnetic_curve",
        "category": "method",
        "aliases": (
            r"\bJ(?:s|rs)[- ]?T\b",
            r"\bSUSC[- ]?T\b",
            r"\bHcs[- ]?T\b",
            r"thermo[- ]?magnetic",
            r"Curie\s+(?:point|temperature)",
        ),
    },
    {
        "concept_id": "magnetic_susceptibility",
        "category": "method",
        "aliases": (r"\bSUSC\b", r"magnetic\s+susceptibility"),
    },
    {
        "concept_id": "hysteresis",
        "category": "method",
        "aliases": (r"\bHYST\b", r"hysteresis", r"\bFORC\b"),
    },
    {
        "concept_id": "coercivity_measurement",
        "category": "method",
        "aliases": (r"\bHc(?:r|s)?\s*=", r"coerciv(?:e|ity)", r"back[- ]field"),
    },
    {
        "concept_id": "magnetic_anisotropy",
        "category": "method",
        "aliases": (r"\bAN\b", r"anisotrop(?:y|ic)"),
    },
    {
        "concept_id": "redeposition_test",
        "category": "method",
        "aliases": (r"\bREDEP\b", r"redeposition"),
    },
    {
        "concept_id": "magnetite",
        "category": "carrier",
        "aliases": (r"\bMAGN\b", r"\bmagnetite\b"),
    },
    {
        "concept_id": "titanomagnetite",
        "category": "carrier",
        "aliases": (r"\bTM\b", r"titano[- ]?magnetite"),
    },
    {
        "concept_id": "hematite",
        "category": "carrier",
        "aliases": (r"\bHEM\b", r"ha?ematite"),
    },
    {
        "concept_id": "maghemite",
        "category": "carrier",
        "aliases": (r"\bMAGH\b", r"maghemite"),
    },
    {
        "concept_id": "ilmenite",
        "category": "carrier",
        "aliases": (r"\bILM\b", r"ilmenite"),
    },
    {
        "concept_id": "goethite",
        "category": "carrier",
        "aliases": (r"\bGOETH\b", r"goethite"),
    },
    {
        "concept_id": "pyrrhotite",
        "category": "carrier",
        "aliases": (r"\bPYRR\b", r"pyrrhotite"),
    },
    {
        "concept_id": "greigite",
        "category": "carrier",
        "aliases": (r"\bGREIG\b", r"greigite"),
    },
    {
        "concept_id": "saturation_behavior",
        "category": "diagnostic",
        "aliases": (r"\bno\s+sat\b", r"\bsat\.?\s*[0-9]", r"saturat(?:e|ed|ion)"),
    },
    {
        "concept_id": "curie_temperature",
        "category": "diagnostic",
        "aliases": (r"\bCT\s*[0-9]", r"Curie\s+(?:point|temperature)"),
    },
    {
        "concept_id": "domain_state",
        "category": "diagnostic",
        "aliases": (r"\b(?:SD|PSD|MD)\b", r"domain\s+state"),
    },
    {
        "concept_id": "koenigsberger_ratio",
        "category": "diagnostic",
        "aliases": (r"\bQ\s*=", r"K[oö]nigsberger"),
    },
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compiled_concepts() -> tuple[tuple[Mapping[str, Any], tuple[re.Pattern[str], ...]], ...]:
    return tuple(
        (
            concept,
            tuple(re.compile(alias, re.I) for alias in concept["aliases"]),
        )
        for concept in CONCEPTS
    )


def iter_column_values(path: Path, column: str = SOURCE_COLUMN) -> Iterable[str]:
    """Yield non-empty values while loading no other spreadsheet data cells."""

    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[workbook.sheetnames[0]]
        header = next(worksheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
        matches = [index for index, value in enumerate(header) if str(value or "").strip() == column]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {column} column in {path}; found {len(matches)}")
        column_index = matches[0] + 1
        for (value,) in worksheet.iter_rows(
            min_row=2,
            min_col=column_index,
            max_col=column_index,
            values_only=True,
        ):
            text = str(value or "").strip()
            if text:
                yield text
    finally:
        workbook.close()


def induce(values: Iterable[str], *, minimum_count: int = 1) -> dict[str, Any]:
    compiled = _compiled_concepts()
    concept_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()
    nonempty_rows = 0
    matched_rows = 0

    for value in values:
        nonempty_rows += 1
        matched = {
            str(concept["concept_id"])
            for concept, patterns in compiled
            if any(pattern.search(value) for pattern in patterns)
        }
        if not matched:
            continue
        matched_rows += 1
        concept_counts.update(matched)
        categories = {
            str(concept["category"])
            for concept, _ in compiled
            if concept["concept_id"] in matched
        }
        category_counts.update(categories)
        ordered = sorted(matched)
        edge_counts.update(
            (ordered[left], ordered[right])
            for left in range(len(ordered))
            for right in range(left + 1, len(ordered))
        )

    concepts = []
    for concept, _ in compiled:
        count = concept_counts[str(concept["concept_id"])]
        if count < minimum_count:
            continue
        concepts.append(
            {
                "concept_id": concept["concept_id"],
                "category": concept["category"],
                "document_frequency": count,
                "normalized_frequency": round(count / nonempty_rows, 8) if nonempty_rows else 0.0,
                "search_patterns": list(concept["aliases"]),
            }
        )

    edges = [
        {
            "source_concept": source,
            "target_concept": target,
            "cooccurrence_count": count,
            "normalized_frequency": round(count / nonempty_rows, 8) if nonempty_rows else 0.0,
        }
        for (source, target), count in sorted(
            edge_counts.items(), key=lambda item: (-item[1], item[0])
        )
        if count >= minimum_count
    ]
    return {
        "counts": {
            "nonempty_source_values": nonempty_rows,
            "values_matching_known_concepts": matched_rows,
            "coverage": round(matched_rows / nonempty_rows, 8) if nonempty_rows else 0.0,
            "category_document_frequency": dict(sorted(category_counts.items())),
        },
        "concepts": concepts,
        "cooccurrence_edges": edges,
    }


def build_artifact(input_path: Path, *, minimum_count: int = 1) -> dict[str, Any]:
    induced = induce(iter_column_values(input_path), minimum_count=minimum_count)
    return {
        "schema_version": "r3_deidentified_terminology_graph_v1",
        "knowledge_version": "r3_rockmag_lexicon_v1",
        "provenance": {
            "phase": "offline_knowledge_induction",
            "source_field": "PMAGRESULT.ROCKMAG",
            "source_sha256": file_sha256(input_path),
            "columns_read": [SOURCE_COLUMN],
            "label_supervision": False,
            "evaluation_results_used": False,
            "record_identifiers_read": False,
            "raw_values_retained": False,
        },
        "runtime_policy": {
            "scoring_may_read_source_workbook": False,
            "scoring_uses_frozen_aggregate_only": True,
            "contains_record_answer_mapping": False,
        },
        "minimum_document_frequency": minimum_count,
        **induced,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--minimum-count", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_count < 1:
        raise SystemExit("--minimum-count must be at least 1")
    artifact = build_artifact(args.input, minimum_count=args.minimum_count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "concepts": len(artifact["concepts"]),
        "cooccurrence_edges": len(artifact["cooccurrence_edges"]),
        **artifact["counts"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
