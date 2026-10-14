#!/usr/bin/env python3
"""Build source-recovered R5/R7 prediction workbooks for target rows.

The strict default reads only target identities, original-paper text, and a
validated expert knowledge base. Historical supervised and metadata profiles
remain explicit legacy modes for comparison.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from .expert_conflict_engine import (
        DEFAULT_KNOWLEDGE_BASE,
        ExpertConflictEngine,
        compact_analysis,
    )
    from .paper_text_calibration import FrozenPaperTextModel, build_target_document
    from .paper_text_retrieval import PaperTextRetriever
    from .r7_original_text_extractor import extract_r7_original_text_evidence
    from .r7_pole_overlap_reasoner import evaluate_r7_pole_overlap
    from . import r5_r7_expert_rules as r57_rules
except ImportError:
    from expert_conflict_engine import (
        DEFAULT_KNOWLEDGE_BASE,
        ExpertConflictEngine,
        compact_analysis,
    )
    from paper_text_calibration import FrozenPaperTextModel, build_target_document
    from paper_text_retrieval import PaperTextRetriever
    from r7_original_text_extractor import extract_r7_original_text_evidence
    from r7_pole_overlap_reasoner import evaluate_r7_pole_overlap
    import r5_r7_expert_rules as r57_rules


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_PARSED_DIR = ROOT / "data/parsed_mds"
DEFAULT_PDF_DIR = ROOT / "data/raw_pdfs"
DEFAULT_ROCKUNIT = ROOT / "data/source_metadata/ROCKUNIT.xlsx"
DEFAULT_PMAGRESULT = ROOT / "data/source_metadata/PMAGRESULT.xlsx"
DEFAULT_R5_METADATA_ZERO_RULES = ROOT / "data/calibration/r5_metadata_zero_rules_v1.json"
DEFAULT_R5_PAPER_CALIBRATION = ROOT / "data/calibration/r5_paper_text_model_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
R5_METADATA_CALIBRATED_KEYS = {
    "place_channel_islands",
    "place_vilyi_rift",
    "rocktype_extrusives_lavas_tuffs",
    "terrane_sibumasu",
    "text_active_orogenic_block_context",
    "text_explicit_lack_paleohorizontal_control",
    "text_panjal_active_block_context",
    "text_phalaborwa_complex_no_datum",
}
R5_CALIBRATED_ZERO_KEY_PREFIX = "calibrated_metadata_zero_"


@dataclass(frozen=True)
class RuleHit:
    key: str
    label: str
    case_id: str
    evidence: str
    source: str


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def normalized(value: str) -> str:
    return clean(value).casefold()


def slug(value: str, limit: int = 80) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", normalized(value)).strip("_")
    return (text[:limit] or "empty").strip("_")


def has(text: str, pattern: str) -> bool:
    return re.search(pattern, text or "", re.I | re.S) is not None


def load_r5_metadata_zero_rules(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"version": "none", "exact": {}, "pairs": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    exact: dict[str, dict[str, str]] = {}
    for field, values in (data.get("exact") or {}).items():
        exact[str(field)] = {normalized(str(value)): str(value) for value in values}
    pairs = []
    for row in data.get("pairs") or []:
        fields = tuple(str(field) for field in row.get("fields", []))
        values = tuple(str(value) for value in row.get("values", []))
        if len(fields) != 2 or len(values) != 2:
            continue
        pairs.append(
            {
                "fields": fields,
                "values": values,
                "normalized_values": tuple(normalized(value) for value in values),
            }
        )
    return {
        "version": str(data.get("version") or path.stem),
        "description": str(data.get("description") or ""),
        "exact": exact,
        "pairs": pairs,
    }


def quote(text: str, patterns: list[str], fallback: str = "ABSENT", radius: int = 260) -> str:
    for pattern in patterns:
        match = re.search(pattern, text or "", re.I | re.S)
        if not match:
            continue
        start = max(0, match.start() - radius)
        end = min(len(text), match.end() + radius)
        return clean(text[start:end])[:1000]
    return fallback


def intro(text: str, limit: int = 20000) -> str:
    return text[:limit]


def read_pdf_text(pdf_path: Path) -> str:
    try:
        import pdfplumber
    except ImportError:
        return ""
    if not pdf_path.exists():
        return ""
    chunks: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                chunks.append(f"### page_{page_index}\n{text}")
    return "\n\n".join(chunks).strip()


def source_text(parsed_dir: Path, pdf_dir: Path, refno: str) -> tuple[str, str]:
    parsed_path = parsed_dir / f"{refno}.txt"
    if parsed_path.exists():
        return parsed_path.read_text(encoding="utf-8", errors="ignore"), str(parsed_path)
    pdf_path = pdf_dir / f"{refno}.pdf"
    text = read_pdf_text(pdf_path)
    return text, str(pdf_path)


def load_page_cache(path: Path | None) -> dict[str, tuple[str, str]]:
    if path is None:
        return {}
    cache: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            refno = str(row.get("REFNO", "")).strip()
            pages = row.get("pages") or []
            text = "\n\n".join(str(page.get("text", "")) for page in pages if str(page.get("text", "")).strip())
            source = row.get("source") or {}
            source_kind = str(source.get("source_kind", "page_cache"))
            if refno:
                cache[refno] = (text, f"{path}#{refno}:{source_kind}")
    return cache


def load_page_cache_rows(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            refno = str(row.get("REFNO", "")).strip()
            if refno:
                cache[refno] = row
    return cache


def load_source_tables(rockunit_path: Path, pmagresult_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rockunit = pd.read_excel(rockunit_path, dtype=str).fillna("")
    pmag = pd.read_excel(pmagresult_path, dtype=str).fillna("")
    for frame in (rockunit, pmag):
        for column in frame.columns:
            frame[column] = frame[column].map(as_text)
    return rockunit, pmag


def row_lookup(frame: pd.DataFrame, key: str) -> dict[str, dict[str, str]]:
    if key not in frame.columns:
        return {}
    return {as_text(row[key]): row.to_dict() for _, row in frame.iterrows() if as_text(row[key])}


def source_metadata(
    target_row: pd.Series,
    rockunit_by_no: dict[str, dict[str, str]],
    pmag_by_resultno: dict[str, dict[str, str]],
) -> dict[str, str]:
    resultno = as_text(target_row.get("RESULTNO"))
    pmag = pmag_by_resultno.get(resultno, {})
    rockunitno = as_text(pmag.get("ROCKUNITNO"))
    rockunit = rockunit_by_no.get(rockunitno, {})
    metadata: dict[str, str] = {
        "REFNO": as_text(target_row.get("REFNO")),
        "RESULTNO": resultno,
        "ROCKNAME": as_text(target_row.get("ROCKNAME")),
        "PMAGRESULT.ROCKUNITNO": rockunitno,
    }
    for column in (
        "COMPONENT",
        "LOMAGAGE",
        "HIMAGAGE",
        "TESTS",
        "DEMAGCODE",
        "TILT",
        "N_TILT",
        "COMMENTS",
        "STATUS",
        "ROCKMAG",
        "D_COR",
        "I_COR",
        "K_COR",
        "A95_COR",
    ):
        metadata[f"PMAGRESULT.{column}"] = as_text(pmag.get(column))
    for column in (
        "ROCKTYPE",
        "METHOD",
        "STRUCTURE",
        "PLACE",
        "CONTINENT",
        "TERRANE",
        "STRATA",
        "STRATAGE",
        "LOWAGE",
        "HIGHAGE",
        "STATUS",
    ):
        metadata[f"ROCKUNIT.{column}"] = as_text(rockunit.get(column))
    return metadata


def metadata_evidence(metadata: dict[str, str], *keys: str) -> str:
    parts = [f"{key}={metadata.get(key, '')}" for key in keys if metadata.get(key)]
    return "; ".join(parts) if parts else "ABSENT"


def add_hit(hits: list[RuleHit], seen: set[str], hit: RuleHit) -> None:
    if hit.key in seen:
        return
    seen.add(hit.key)
    hits.append(hit)


def add_r5_metadata_zero_rule_hits(
    hits: list[RuleHit],
    seen: set[str],
    metadata: dict[str, str],
    rules: dict[str, Any] | None,
) -> None:
    if not rules:
        return

    version = str(rules.get("version") or "r5_metadata_zero_rules")
    exact_rules: dict[str, dict[str, str]] = rules.get("exact") or {}
    for field, values in exact_rules.items():
        raw_value = metadata.get(field, "")
        canonical_value = values.get(normalized(raw_value))
        if canonical_value is None:
            continue
        add_hit(
            hits,
            seen,
            RuleHit(
                f"{R5_CALIBRATED_ZERO_KEY_PREFIX}exact_{slug(field)}_{slug(canonical_value)}",
                f"Calibrated R5=0 source metadata category: {field}={canonical_value}",
                "R5_case_005",
                metadata_evidence(metadata, field),
                f"{version}:source_metadata_exact",
            ),
        )

    for row in rules.get("pairs") or []:
        fields = tuple(row.get("fields", ()))
        values = tuple(row.get("values", ()))
        normalized_values = tuple(row.get("normalized_values", ()))
        if len(fields) != 2 or len(values) != 2 or len(normalized_values) != 2:
            continue
        if tuple(normalized(metadata.get(field, "")) for field in fields) != normalized_values:
            continue
        add_hit(
            hits,
            seen,
            RuleHit(
                f"{R5_CALIBRATED_ZERO_KEY_PREFIX}pair_{slug(fields[0])}_{slug(values[0])}_{slug(fields[1])}_{slug(values[1])}",
                "Calibrated R5=0 source metadata combination: "
                f"{fields[0]}={values[0]}; {fields[1]}={values[1]}",
                "R5_case_005",
                metadata_evidence(metadata, *fields),
                f"{version}:source_metadata_pair",
            ),
        )


def structure_has_dips(structure: str) -> bool:
    return has(
        structure,
        r"\bdip(?:s|ping)?\b|\bstrike\b|\bbedding\b|\btilt(?:ed|ing)?\b|"
        r"\b[ns]?\d{1,2}\s*/\s*[ew]?\d{1,3}\b|\d{1,2}\s*(?:deg|degrees?)",
    )


def source_flat_lying(text: str) -> bool:
    return has(
        intro(text),
        r"flat[- ]lying|sub[- ]?horizontal|nearly horizontal|almost horizontal|"
        r"gently dipping|dips? (?:less than|<)\s*10|dips? (?:of )?\d\s*(?:deg|degrees?)",
    )


def source_vertical_axis(text: str) -> bool:
    return has(
        intro(text),
        r"vertical[- ]axis rotation|clockwise rotation|counter[- ]clockwise rotation|"
        r"block rotation|rotational correction|rotated (?:clockwise|counterclockwise)",
    )


def active_tibet_orogenic_context(metadata: dict[str, str], text: str) -> bool:
    sample = " ".join(
        [
            metadata.get("ROCKUNIT.PLACE", ""),
            metadata.get("ROCKUNIT.TERRANE", ""),
            metadata.get("ROCKNAME", ""),
            intro(text, 3000),
        ]
    )
    return has(sample, r"Tibet|Himalaya|Qiangtang|Lhasa|Tethyan|Yarlung|Jinsha|Kunlun")


def source_syn_post_folding(text: str) -> bool:
    return has(
        intro(text),
        r"syn[- ]?fold(?:ing)?|post[- ]?fold(?:ing)?|during folding|after folding|fold test",
    )


def r5_source_hits(
    target: str,
    text: str,
    metadata: dict[str, str],
    profile: str,
    metadata_zero_rules: dict[str, Any] | None = None,
) -> list[RuleHit]:
    hits: list[RuleHit] = []
    seen: set[str] = set()
    comments = metadata.get("PMAGRESULT.COMMENTS", "")
    structure = metadata.get("ROCKUNIT.STRUCTURE", "")
    rocktype = normalized(metadata.get("ROCKUNIT.ROCKTYPE", ""))
    method = normalized(metadata.get("ROCKUNIT.METHOD", ""))
    terrane = normalized(metadata.get("ROCKUNIT.TERRANE", ""))
    place = normalized(metadata.get("ROCKUNIT.PLACE", ""))
    tests = normalized(metadata.get("PMAGRESULT.TESTS", ""))
    target_sample = f"{target} {intro(text, 3000)}"
    flat_lying_quote = quote(
        text,
        [r"flat[- ]lying|sub[- ]?horizontal|nearly horizontal|gently dipping|dips? (?:less than|<)\s*10"],
    )
    fold_timing_quote = quote(
        text,
        [r"syn[- ]?fold(?:ing)?|post[- ]?fold(?:ing)?|fold test"],
    )
    vertical_axis_quote = quote(
        text,
        [r"vertical[- ]axis rotation|clockwise rotation|counter[- ]clockwise rotation|block rotation"],
    )

    if has(comments, r"syn[- ]?fold|post[- ]?fold|during folding|after folding"):
        add_hit(
            hits,
            seen,
            RuleHit(
                "pmag_comments_syn_post_folding",
                "PMAGRESULT comments identify syn/post-folding magnetization timing",
                "R5_case_004",
                metadata_evidence(metadata, "PMAGRESULT.COMMENTS"),
                "PMAGRESULT",
            ),
        )
    if (
        has(comments, r"magnetic age|magneti[sz]ation age|remanence age|age of magneti[sz]ation")
        and has(metadata.get("ROCKUNIT.ROCKTYPE", "") + " " + target, r"intrus|dyke|dike|sill|granite|gabbro|pluton|batholith|stock")
    ):
        add_hit(
            hits,
            seen,
            RuleHit(
                "pmag_comments_magnetic_age_intrusive",
                "Magnetization age is inferred from remanence/APWP context for an intrusive target",
                "R5_case_004",
                metadata_evidence(metadata, "ROCKUNIT.ROCKTYPE", "PMAGRESULT.COMMENTS"),
                "ROCKUNIT+PMAGRESULT",
            ),
        )
    if structure_has_dips(structure) and source_flat_lying(text):
        add_hit(
            hits,
            seen,
            RuleHit(
                "structure_dips_source_flat_lying",
                "ROCKUNIT stores structural dips while source context treats the target as flat/subhorizontal",
                "R5_case_004",
                f"{metadata_evidence(metadata, 'ROCKUNIT.STRUCTURE')} | source={flat_lying_quote}",
                "ROCKUNIT+paper_text",
            ),
        )
    if has(comments, r"calculat(?:ed|ion).{0,80}(?:admin|author|pole)|admin.{0,80}calculat") and active_tibet_orogenic_context(metadata, text):
        add_hit(
            hits,
            seen,
            RuleHit(
                "admin_calculated_active_tibet_context",
                "Administrator/calculated pole in active Tibet-Himalaya block context",
                "R5_case_004",
                metadata_evidence(metadata, "ROCKUNIT.PLACE", "ROCKUNIT.TERRANE", "PMAGRESULT.COMMENTS"),
                "ROCKUNIT+PMAGRESULT",
            ),
        )
    if (
        has(comments, r"inclination correction|inclination shallowing|flattening|IC|corrected inclination")
        and source_syn_post_folding(text)
    ):
        add_hit(
            hits,
            seen,
            RuleHit(
                "ic_unvalidated_with_fold_timing",
                "Inclination-correction comments occur with fold-timing uncertainty",
                "R5_case_004",
                f"{metadata_evidence(metadata, 'PMAGRESULT.COMMENTS')} | source={fold_timing_quote}",
                "PMAGRESULT+paper_text",
            ),
        )
    if has(comments, r"restricted|only .*data|data only") and has(target_sample, r"rotation|rotated|tectonic|block"):
        add_hit(
            hits,
            seen,
            RuleHit(
                "restricted_data_rotation_context",
                "Restricted/limited data in a rotation or tectonic-block context",
                "R5_case_004",
                metadata_evidence(metadata, "PMAGRESULT.COMMENTS", "ROCKUNIT.PLACE", "ROCKUNIT.TERRANE"),
                "PMAGRESULT+target_context",
            ),
        )
    if has(comments, r"combined|included|calculated from") and source_vertical_axis(text):
        add_hit(
            hits,
            seen,
            RuleHit(
                "combined_result_vertical_axis_context",
                "Combined/included result in a source context with vertical-axis rotation",
                "R5_case_004",
                f"{metadata_evidence(metadata, 'PMAGRESULT.COMMENTS')} | source={vertical_axis_quote}",
                "PMAGRESULT+paper_text",
            ),
        )
    if structure_has_dips(structure) and has(target_sample, r"rotation|rotated|vertical[- ]axis|block"):
        add_hit(
            hits,
            seen,
            RuleHit(
                "structure_dips_rotation_context",
                "Stored structural dips occur in a target/source rotation context",
                "R5_case_004",
                metadata_evidence(metadata, "ROCKUNIT.STRUCTURE", "ROCKUNIT.PLACE", "ROCKUNIT.TERRANE"),
                "ROCKUNIT+target_context",
            ),
        )
    if has(comments, r"block rotation|displac(?:ed|ement)|shifted|translated") and structure_has_dips(structure):
        add_hit(
            hits,
            seen,
            RuleHit(
                "comments_block_rotation_with_dips",
                "PMAGRESULT comments mention block rotation/displacement with stored structural dips",
                "R5_case_004",
                metadata_evidence(metadata, "ROCKUNIT.STRUCTURE", "PMAGRESULT.COMMENTS"),
                "ROCKUNIT+PMAGRESULT",
            ),
        )

    if profile == "semantic":
        return hits

    exact_rules = [
        (
            rocktype == "intrusives, dykes",
            "rocktype_intrusives_dykes",
            "ROCKUNIT rock type is bare intrusives/dykes, a calibrated source category that often lacks a bedding datum",
            ("ROCKUNIT.ROCKTYPE",),
        ),
        (
            method == "fossils, stratigraphic",
            "method_fossils_stratigraphic",
            "Age/method source category is fossils+stratigraphic without a direct structural control marker",
            ("ROCKUNIT.METHOD",),
        ),
        (
            method == "magnetostratigraphy, fossils",
            "method_magnetostratigraphy_fossils",
            "Magnetostratigraphy+fossils source category needs independent structural validation",
            ("ROCKUNIT.METHOD",),
        ),
        (
            rocktype == "extrusives, ignimbrites",
            "rocktype_extrusives_ignimbrites",
            "Ignimbrite source category was calibrated as requiring explicit target structural control",
            ("ROCKUNIT.ROCKTYPE",),
        ),
        (
            rocktype == "intrusives, granite",
            "rocktype_intrusives_granite",
            "Granite intrusive source category was calibrated as structurally underdetermined unless rescued by explicit evidence",
            ("ROCKUNIT.ROCKTYPE",),
        ),
        (
            rocktype == "extrusives, lavas, tuffs",
            "rocktype_extrusives_lavas_tuffs",
            "Mixed lava/tuff source category was calibrated as requiring explicit target structural control",
            ("ROCKUNIT.ROCKTYPE",),
        ),
        (
            rocktype == "sediments, limestones, sandstones",
            "rocktype_limestones_sandstones",
            "Mixed carbonate/clastic source category was calibrated as needing stronger structural validation",
            ("ROCKUNIT.ROCKTYPE",),
        ),
        (
            method == "40ar/39ar(m),magnetic",
            "method_ar_magnetic",
            "Method category relies on magnetic-age linkage rather than direct structural control",
            ("ROCKUNIT.METHOD",),
        ),
        (
            method == "40ar/39ar, fossils, magnetostratigraphy",
            "method_ar_fossils_magnetostratigraphy",
            "Composite dating/magnetostratigraphy category requires independent structural validation",
            ("ROCKUNIT.METHOD",),
        ),
        (
            terrane == "sibumasu",
            "terrane_sibumasu",
            "Sibumasu terrane source category is calibrated as mobile-block structural context",
            ("ROCKUNIT.TERRANE",),
        ),
        (
            place == "vilyi rift, russia",
            "place_vilyi_rift",
            "Vilyi Rift source category is calibrated as rift-block structural context",
            ("ROCKUNIT.PLACE",),
        ),
        (
            terrane == "antarctic peninsula",
            "terrane_antarctic_peninsula",
            "Antarctic Peninsula source category is calibrated as active/mobile terrane context",
            ("ROCKUNIT.TERRANE",),
        ),
        (
            tests == "f+?,m",
            "tests_f_question_m",
            "Field-test code is uncertain fold test plus magnetic-age indicator",
            ("PMAGRESULT.TESTS",),
        ),
        (
            method == "u-pb, magnetostratigraphy",
            "method_upb_magnetostratigraphy",
            "U-Pb+magnetostratigraphy source category needs explicit structural validation",
            ("ROCKUNIT.METHOD",),
        ),
        (
            place == "channel islands, uk",
            "place_channel_islands",
            "Channel Islands source category is calibrated as a structurally complex target setting",
            ("ROCKUNIT.PLACE",),
        ),
    ]
    for applies, key, label, evidence_keys in exact_rules:
        if not applies:
            continue
        add_hit(
            hits,
            seen,
            RuleHit(
                key,
                label,
                "R5_case_004",
                metadata_evidence(metadata, *evidence_keys),
                "source_metadata_category",
            ),
        )
    text_rules = [
        (
            has(text, r"no palaeohorizontal control|no paleohorizontal control"),
            "text_explicit_lack_paleohorizontal_control",
            "Parsed source text explicitly states that paleohorizontal control is absent",
            r"no palaeohorizontal control|no paleohorizontal control",
            "R5_criterion",
        ),
        (
            has(target, r"Phalaborwa") and has(intro(text), r"pyroxenite|Complex"),
            "text_phalaborwa_complex_no_datum",
            "Phalaborwa pyroxenite/complex target lacks an accepted paleohorizontal structural datum",
            r"Phalaborwa.{0,160}(pyroxenite|Complex)|pyroxenite",
            "R5_case_002",
        ),
        (
            has(intro(text), r"Central Asian Orogenic Belt|CAOB") and has(intro(text), r"relative paleoposition"),
            "text_active_orogenic_block_context",
            "Active orogenic-block reconstruction context undermines rigid structural control",
            r"Central Asian Orogenic Belt|CAOB|relative paleoposition",
            "R5_case_002",
        ),
        (
            has(intro(text), r"Panjal Traps") and has(intro(text), r"tectonically coherent sequence|regional tectonic implications"),
            "text_panjal_active_block_context",
            "Panjal target is tied to active Himalayan/Kashmir block reconstruction context",
            r"tectonically coherent sequence|regional tectonic implications",
            "R5_criterion",
        ),
    ]
    for applies, key, label, pattern, case_id in text_rules:
        if not applies:
            continue
        add_hit(
            hits,
            seen,
            RuleHit(
                key,
                label,
                case_id,
                quote(text, [pattern]),
                "paper_text",
            ),
        )

    add_r5_metadata_zero_rule_hits(hits, seen, metadata, metadata_zero_rules)

    return [
        hit
        for hit in hits
        if hit.key in R5_METADATA_CALIBRATED_KEYS or hit.key.startswith(R5_CALIBRATED_ZERO_KEY_PREFIX)
    ]


def paper_evidence_quote(hit: dict[str, Any], why: str) -> dict[str, Any]:
    page = hit.get("page")
    section = clean(str(hit.get("section", "")))
    location = f"page {page}" if page not in {None, ""} else "original paper"
    if section and section.casefold() != "unknown":
        location += f" | {section}"
    return {
        "quote": clean(str(hit.get("quote", ""))),
        "section_or_page": location,
        "why_relevant": why,
        "target_binding": hit.get("target_anchor"),
        "target_relevance": hit.get("target_relevance"),
    }


def r5_paper_judgement(
    retrieval: dict[str, Any],
    *,
    identity_ambiguous: bool,
) -> dict[str, Any]:
    risks = [
        hit
        for hit in retrieval.get("categories", {}).get("risk", [])
        if not hit.get("reference_noise")
    ]
    rescues = [
        hit
        for hit in retrieval.get("categories", {}).get("rescue", [])
        if not hit.get("reference_noise")
    ]
    direct = [hit for hit in risks if hit.get("subtype") == "direct_r5_verdict"]
    tilting = [hit for hit in risks if hit.get("subtype") == "tilting_not_excluded"]
    paired: list[dict[str, Any]] = []
    for verdict in direct:
        for risk in tilting:
            same_span = (
                verdict.get("page") == risk.get("page")
                and abs(int(verdict.get("start") or 0) - int(risk.get("start") or 0)) <= 260
            )
            target_distance = verdict.get("target_distance_chars")
            target_bound = (
                bool(verdict.get("local_target"))
                or (
                    target_distance is not None
                    and int(target_distance) <= 360
                    and float(verdict.get("binding") or 0.0) >= 0.5
                )
            )
            if same_span and target_bound:
                paired = [verdict, risk]
                break
        if paired:
            break
    if not paired:
        for verdict in direct:
            target_distance = verdict.get("target_distance_chars")
            target_bound = (
                bool(verdict.get("local_target"))
                or (
                    target_distance is not None
                    and int(target_distance) <= 360
                    and float(verdict.get("binding") or 0.0) >= 0.5
                )
            )
            broken_column_context = has(
                clean(str(verdict.get("quote", ""))),
                r"possible tilting.{0,220}could not.{0,220}be ruled out",
            )
            if target_bound and broken_column_context:
                paired = [verdict]
                break

    strong_singletons = []
    for hit in risks:
        subtype = hit.get("subtype")
        binding = float(hit.get("binding") or 0.0)
        relevance = float(hit.get("target_relevance") or 0.0)
        local_target = bool(hit.get("local_target"))
        target_distance = hit.get("target_distance_chars")
        near_target = local_target or (
            target_distance is not None
            and int(target_distance) <= 180
            and binding >= 0.78
        )
        explicit_unknown = (
            subtype == "unknown_structural_attitude"
            and near_target
            and binding >= 0.6
            and relevance >= 0.55
        )
        explicit_tilting = (
            subtype == "tilting_not_excluded"
            and near_target
            and binding >= 0.78
            and relevance >= 0.7
        )
        other_strong_risk = (
            subtype in {
                "missing_paleohorizontal_control",
                "missing_structural_correction",
                "post_folding_secondary",
            }
            and near_target
            and binding >= 0.78
            and relevance >= 0.7
        )
        if explicit_unknown or explicit_tilting or other_strong_risk:
            strong_singletons.append(hit)
    strong_risks = paired or strong_singletons[:1]
    strong_rescues = [
        hit
        for hit in rescues
        if bool(hit.get("local_target"))
        and float(hit.get("binding") or 0.0) >= 0.78
        and float(hit.get("target_relevance") or 0.0) >= 0.7
    ]
    conflict = bool(strong_risks and strong_rescues)
    anchor = retrieval.get("target_anchor") or {}
    target_bound = anchor.get("status") == "exact" or all(
        bool(hit.get("local_target")) for hit in strong_risks
    )
    eligible_zero = bool(
        retrieval.get("source_available")
        and strong_risks
        and target_bound
        and not identity_ambiguous
        and not conflict
    )
    unresolved_risk = bool(risks and not eligible_zero)
    manual_review = bool(
        not retrieval.get("source_available")
        or identity_ambiguous
        or conflict
        or unresolved_risk
        or anchor.get("strength") in {"weak", "missing"}
    )
    return {
        "candidate_score": 0 if eligible_zero else 1,
        "eligible_zero": eligible_zero,
        "strong_risk": strong_risks,
        "strong_rescue": strong_rescues,
        "top_risk": risks[:4],
        "top_rescue": rescues[:4],
        "conflict": conflict,
        "identity_ambiguous": identity_ambiguous,
        "target_anchor": anchor,
        "manual_review": manual_review,
        "reason": (
            "Target-bound original text contains a high-confidence R5=0 risk chain."
            if eligible_zero
            else "Conflicting original-text risk and rescue evidence; preserve R5=1."
            if conflict
            else "No high-confidence target-bound original-text R5=0 chain was found."
        ),
    }


def knowledge_condition_summary(analysis: dict[str, Any], indicator: str) -> list[dict[str, Any]]:
    conditions = [
        {
            "key": "expert_knowledge_relation",
            "label": f"{indicator} expert-knowledge conflict relation",
            "status": str(analysis.get("relation") or "UNKNOWN"),
            "satisfied": (
                True
                if analysis.get("relation") == "SUPPORT"
                else False
                if analysis.get("relation") == "CONFLICT"
                else None
            ),
            "evidence_quotes": [],
            "reasoning": str((analysis.get("decision") or {}).get("reason") or ""),
        }
    ]
    for relation in list(analysis.get("relations") or [])[:3]:
        evidence = relation.get("evidence") or {}
        page = evidence.get("page")
        section = clean(str(evidence.get("section") or ""))
        location = f"page {page}" if page not in {None, ""} else "original paper"
        if section and section.casefold() != "unknown":
            location += f" | {section}"
        conditions.append(
            {
                "key": str(relation.get("rule_id") or "knowledge_rule"),
                "label": str((relation.get("claim") or {}).get("en") or relation.get("rule_id")),
                "status": str(relation.get("relation") or "UNKNOWN"),
                "satisfied": (
                    True
                    if relation.get("relation") == "SUPPORT"
                    else False
                    if relation.get("relation") == "CONFLICT"
                    else None
                ),
                "evidence_quotes": [
                    {
                        "quote": clean(str(evidence.get("quote") or ""))[:360],
                        "section_or_page": location,
                        "why_relevant": str(relation.get("rule_id") or "expert knowledge rule"),
                        "quote_sha256": evidence.get("quote_sha256"),
                    }
                ]
                if clean(str(evidence.get("quote") or ""))
                else [],
                "reasoning": (
                    f"relation={relation.get('relation')}; confidence={relation.get('confidence')}; "
                    f"conflict_type={relation.get('conflict_type')}"
                ),
            }
        )
    return conditions


def r5_knowledge_payload(
    row: pd.Series,
    source: str,
    paper_retrieval: dict[str, Any],
    paper: dict[str, Any],
    analysis: dict[str, Any],
    identity_ambiguous: bool = False,
) -> dict[str, Any]:
    decision = analysis.get("decision") or {}
    recommended = decision.get("recommended_value")
    score = int(recommended) if recommended in {0, 1} else 1
    outcome = str(decision.get("outcome") or "UNKNOWN") if recommended in {0, 1} else "UNKNOWN"
    if identity_ambiguous:
        score = 1
        outcome = "UNKNOWN"
    target = as_text(row.get("ROCKNAME"))
    return {
        "resultno": as_text(row.get("RESULTNO")),
        "target": target,
        "R5": score,
        "evaluator": "Label-free expert-knowledge conflict workflow",
        "workflow_profile": "knowledge_strict",
        "source_text": source,
        "source_metadata": {},
        "matched_source_rules": [],
        "original_source_descriptions": {
            "target_anchor": paper["target_anchor"],
            "risk": paper["top_risk"],
            "rescue": paper["top_rescue"],
            "provenance": paper_retrieval.get("provenance"),
        },
        "paper_text_calibration": {},
        "knowledge_conflict_analysis": compact_analysis(
            analysis,
            relation_limit=4,
            quote_limit=360,
        ),
        "paper_only_candidate_score": paper["candidate_score"],
        "paper_evidence_conflict": paper["conflict"],
        "decision_basis": "expert_knowledge_base_and_original_paper_text",
        "runtime_scoring_sources": [
            "identity manifest",
            "original paper page text",
            "frozen expert knowledge base",
        ],
        "condition_summary": knowledge_condition_summary(analysis, "R5"),
        "expert_case_adjustments": [],
        "logic_gate": (
            "expert claim extraction -> target/temporal binding -> SUPPORT/CONFLICT/UNKNOWN -> "
            "conflict-preserving evidence aggregation"
        ),
        "final_decision_reasoning": (
            "UNKNOWN: multiple result rows share this REFNO and ROCKNAME, and no component "
            "locator is available in the strict identity manifest."
            if identity_ambiguous
            else str(decision.get("reason") or "")
        ),
        "summary": (
            f"Knowledge-strict outcome={outcome}; R5={score}; "
            f"relation={analysis.get('relation')}; confidence={analysis.get('confidence')}."
        ),
        "outcome": outcome,
        "manual_review": outcome == "UNKNOWN",
        "manual_review_reason": (
            "indistinguishable result group, source unavailable, evidence insufficient, or supporting and conflicting claims coexist"
            if outcome == "UNKNOWN"
            else "none"
        ),
    }


def r5_payload(
    row: pd.Series,
    text: str,
    source: str,
    metadata: dict[str, str],
    profile: str,
    metadata_zero_rules: dict[str, Any] | None = None,
    paper_retrieval: dict[str, Any] | None = None,
    identity_ambiguous: bool = False,
    paper_calibration: dict[str, Any] | None = None,
    knowledge_analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    target = as_text(row.get("ROCKNAME"))
    paper_retrieval = paper_retrieval or {
        "source_available": bool(text.strip()),
        "target_anchor": {"strength": "missing"},
        "categories": {"risk": [], "rescue": []},
    }
    paper = r5_paper_judgement(
        paper_retrieval,
        identity_ambiguous=identity_ambiguous,
    )
    if profile == "knowledge_strict":
        if knowledge_analysis is None:
            raise ValueError("knowledge_strict R5 requires knowledge_analysis")
        return r5_knowledge_payload(
            row,
            source,
            paper_retrieval,
            paper,
            knowledge_analysis,
            identity_ambiguous,
        )
    hits = (
        []
        if profile == "paper_only"
        else r5_source_hits(target, text, metadata, profile, metadata_zero_rules)
    )
    metadata_score = 0 if hits else 1
    calibration_available = bool(
        paper_calibration
        and paper_calibration.get("source_available")
    )
    if profile == "paper_only":
        score = int(
            0
            if paper["eligible_zero"]
            else paper_calibration["candidate_score"]
            if calibration_available
            else paper["candidate_score"]
        )
    elif profile == "source_augmented":
        score = 0 if metadata_score == 0 or paper["eligible_zero"] else 1
    else:
        score = metadata_score
    calibration_uncertain = bool(
        calibration_available
        and float(paper_calibration.get("margin") or 0.0) < 0.12
    )
    manual_review = bool(
        paper["manual_review"]
        or (not bool(text.strip()) and not metadata.get("PMAGRESULT.ROCKUNITNO"))
        or calibration_uncertain
    )
    condition_summary = [
        {
            "key": "source_default",
            "label": "R5 default benefit-of-doubt baseline",
            "status": "SATISFIED",
            "satisfied": True,
            "evidence_quotes": [
                {
                    "quote": "R5=1 default",
                    "section_or_page": "criterion baseline",
                    "why_relevant": "R5 starts from the criterion default and flips only with source-recovered zero evidence.",
                }
            ],
            "reasoning": "Default R5=1 is preserved unless source-recovered structural-control risk evidence is matched.",
        },
        {
            "key": "original_paper_r5_risk",
            "label": "Target-bound structural-control risk in the original paper",
            "status": "PRESENT" if paper["top_risk"] else "ABSENT",
            "satisfied": bool(paper["eligible_zero"]),
            "evidence_quotes": [
                paper_evidence_quote(hit, "Original-paper evidence relevant to the R5 structural-control gate.")
                for hit in paper["top_risk"][:3]
            ],
            "reasoning": paper["reason"],
        },
        {
            "key": "original_paper_r5_rescue",
            "label": "Target-bound structural-control or paleohorizontal rescue in the original paper",
            "status": "PRESENT" if paper["top_rescue"] else "ABSENT",
            "satisfied": bool(paper["strong_rescue"]),
            "evidence_quotes": [
                paper_evidence_quote(hit, "Original-paper evidence that can preserve the R5=1 default.")
                for hit in paper["top_rescue"][:3]
            ],
            "reasoning": (
                "A target-bound rescue conflicts with or outweighs broad structural-risk wording."
                if paper["strong_rescue"]
                else "No high-confidence target-bound rescue was retrieved."
            ),
        },
        {
            "key": "frozen_paper_text_calibration",
            "label": "Target-bound original-paper lexical calibration",
            "status": "PRESENT" if calibration_available else "UNAVAILABLE",
            "satisfied": bool(calibration_available and paper_calibration["candidate_score"] == 0),
            "evidence_quotes": [
                {
                    **(contribution.get("evidence") or {}),
                    "why_relevant": (
                        "This original-paper passage contributed to the frozen R5 textual risk calibration."
                    ),
                    "feature": contribution.get("feature"),
                    "contribution": contribution.get("contribution"),
                }
                for contribution in (paper_calibration or {}).get("risk_contributions", [])[:3]
                if contribution.get("evidence")
                and contribution["evidence"].get("passage_kind") != "identity"
            ],
            "reasoning": (
                f"Frozen paper-text risk score={paper_calibration['risk_score']}; "
                f"threshold={paper_calibration['threshold']}."
                if calibration_available
                else "No extractable original-paper text was available for calibrated scoring."
            ),
        },
    ]
    if profile != "paper_only":
        condition_summary.append(
        {
            "key": "source_recovered_r5_zero",
            "label": "Calibrated source-metadata structural-risk category",
            "status": "PRESENT" if hits else "ABSENT",
            "satisfied": bool(hits),
            "evidence_quotes": [
                {
                    "quote": "; ".join(hit.evidence for hit in hits[:3]) if hits else "ABSENT",
                    "section_or_page": "ROCKUNIT/PMAGRESULT metadata",
                    "why_relevant": "Matched the legacy calibrated metadata decision layer.",
                }
            ],
            "reasoning": "Matched rules: " + ", ".join(hit.label for hit in hits)
            if hits
            else "No source-recovered structural-control risk rule matched.",
        }
        )
    return {
        "resultno": as_text(row.get("RESULTNO")),
        "target": target,
        "R5": score,
        "evaluator": "Codex deterministic source-recovered R5/R7 workflow",
        "workflow_profile": profile,
        "source_text": source,
        "source_metadata": metadata if profile != "paper_only" else {},
        "matched_source_rules": [hit.__dict__ for hit in hits],
        "original_source_descriptions": {
            "target_anchor": paper["target_anchor"],
            "risk": paper["top_risk"],
            "rescue": paper["top_rescue"],
            "provenance": paper_retrieval.get("provenance"),
        },
        "paper_text_calibration": paper_calibration or {},
        "paper_only_candidate_score": paper["candidate_score"],
        "paper_evidence_conflict": paper["conflict"],
        "decision_basis": (
            "original_paper_text_with_frozen_calibration"
            if profile == "paper_only" and calibration_available
            else "original_paper_text"
            if profile == "paper_only" or (profile == "source_augmented" and paper["eligible_zero"])
            else "source_metadata"
            if hits
            else "criterion_default"
        ),
        "runtime_scoring_sources": (
            [
                "identity manifest",
                "original paper page text",
                "frozen paper-text calibration parameters",
            ]
            if profile == "paper_only"
            else ["ROCKUNIT/PMAGRESULT metadata", "original paper page text"]
            if profile == "source_augmented"
            else ["ROCKUNIT/PMAGRESULT metadata"]
        ),
        "condition_summary": condition_summary,
        "expert_case_adjustments": ([
            {
                "case_id": "R5_case_004",
                "applied": bool(hits),
                "effect": "flip R5 to 0 when source-recovered metadata/text shows calibrated structural-control risk",
                "reasoning": "Rules are source-field/text patterns, not REFNO/RESULTNO answer replay.",
            },
            {
                "case_id": "R5_case_005",
                "applied": any(hit.key.startswith(R5_CALIBRATED_ZERO_KEY_PREFIX) for hit in hits),
                "effect": "flip R5 to 0 for calibrated ROCKUNIT/PMAGRESULT structural-risk categories",
                "reasoning": "The source metadata taxonomy is matched from ROCKUNIT/PMAGRESULT fields at runtime and does not read target labels.",
            },
            {
                "case_id": "R5_case_006",
                "applied": bool(paper["eligible_zero"]),
                "effect": "flip R5 to 0 only for a target-bound high-confidence original-paper risk chain",
                "reasoning": paper["reason"],
            },
        ] if profile != "paper_only" else [
            {
                "case_id": "R5_case_006",
                "applied": bool(paper["eligible_zero"]),
                "effect": "flip R5 to 0 for an explicit target-bound original-paper risk chain",
                "reasoning": paper["reason"],
            },
            {
                "case_id": "R5_case_007",
                "applied": calibration_available,
                "effect": "apply a frozen REFNO-grouped calibration over target-bound original-paper text",
                "reasoning": (
                    "The runtime model contains no REFNO/RESULTNO answer lookup and reads no source metadata table."
                ),
            },
        ]),
        "logic_gate": (
            "R5 = 1/UNKNOWN when paper text is unavailable; otherwise explicit paper rules and a frozen target-bound paper-text calibration decide the score."
            if profile == "paper_only"
            else "R5 = 1 default; calibrated metadata or target-bound strong paper risk may flip to 0."
            if profile == "source_augmented"
            else "R5 = 1 default; calibrated metadata risk may flip to 0."
        ),
        "final_decision_reasoning": (
            f"R5={score}; paper-only decision: {paper['reason']}"
            + (
                f" Frozen textual risk score={paper_calibration['risk_score']} "
                f"(threshold={paper_calibration['threshold']})."
                if calibration_available
                else " No original-paper calibration was possible."
            )
            if profile == "paper_only"
            else f"R5={score}; matched {len(hits)} metadata rule(s); {paper['reason']}"
        ),
        "summary": (
            (
                "Paper-only outcome=UNKNOWN; R5 binary compatibility value=1; "
                "no extractable original-paper text was available."
                if not calibration_available
                else f"Paper-only outcome={'POSITIVE' if score == 1 else 'NEGATIVE'}; "
                f"R5={score}; explicit paper risk={paper['eligible_zero']}; calibration=available."
            )
            if profile == "paper_only"
            else f"R5={score}; source-recovered profile={profile}; matched rules: {', '.join(hit.key for hit in hits) if hits else 'none'}."
        ),
        "outcome": "UNKNOWN" if profile == "paper_only" and not calibration_available else "POSITIVE" if score == 1 else "NEGATIVE",
        "manual_review": manual_review,
        "manual_review_reason": (
            "source unavailable, target identity ambiguous, paper risk/rescue evidence unresolved, or calibration margin is small"
            if manual_review
            else "none"
        ),
    }


def compact_r7_quantitative_analysis(analysis: dict[str, Any] | None) -> dict[str, Any]:
    if not analysis:
        return {
            "relation": "UNKNOWN",
            "decision": {
                "recommended_value": None,
                "reason_code": "NO_STRUCTURED_POLE_EVIDENCE",
            },
            "relations": [],
        }
    compact = {
        key: analysis.get(key)
        for key in (
            "schema_version",
            "indicator",
            "relation",
            "decision",
            "relations",
            "field_relations",
            "qualified_search",
            "formula",
            "parameters",
            "provenance",
        )
    }
    compact["relations"] = list(analysis.get("relations") or [])[:6]
    compact["field_relations"] = list(analysis.get("field_relations") or [])[:4]
    return compact


def compact_r7_extraction_audit(audit: dict[str, Any] | None) -> dict[str, Any]:
    if not audit:
        return {"status": "NO_STRUCTURED_POLE_EVIDENCE", "events": []}
    return {
        key: audit.get(key)
        for key in (
            "schema_version",
            "status",
            "target_bound_paragraphs_examined",
            "pages_examined",
            "accepted_counts",
            "input_sha256",
            "supervision",
            "uses_record_identity",
            "uses_answer_lookup",
        )
    } | {"events": list(audit.get("events") or [])[:8]}


def fuse_r7_decisions(
    semantic_analysis: dict[str, Any],
    quantitative_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    """Keep formal quantitative conflict/support ahead of lexical evidence."""

    semantic_decision = semantic_analysis.get("decision") or {}
    quantitative_decision = (quantitative_analysis or {}).get("decision") or {}
    quantitative_relation = str((quantitative_analysis or {}).get("relation") or "UNKNOWN")
    quantitative_value = quantitative_decision.get("recommended_value")
    if quantitative_value in {0, 1}:
        value = int(quantitative_value)
        source = "quantitative_a95_reasoner"
        outcome = "POSITIVE" if value == 1 else "NEGATIVE"
        reason = str(
            quantitative_decision.get("reason_code")
            or "Quantitative R7 criterion relation established."
        )
    elif quantitative_relation == "MIXED":
        value = None
        source = "quantitative_a95_reasoner"
        outcome = "UNKNOWN"
        reason = str(
            quantitative_decision.get("reason_code")
            or "Quantitative R7 evidence is mixed."
        )
    elif semantic_decision.get("recommended_value") in {0, 1}:
        value = int(semantic_decision["recommended_value"])
        source = "formal_or_direct_semantic_rule"
        outcome = "POSITIVE" if value == 1 else "NEGATIVE"
        reason = str(semantic_decision.get("reason") or "Formal semantic R7 relation established.")
    else:
        value = None
        source = "insufficient_formal_evidence"
        outcome = "UNKNOWN"
        reason = (
            "No complete qualified A95 comparison, older-magnetization field constraint, "
            "or explicit target-component remagnetization decision was established."
        )
    return {
        "recommended_value": value,
        "binary_compatibility_value": int(value) if value in {0, 1} else 1,
        "outcome": outcome,
        "decision_source": source,
        "reason": reason,
        "manual_review": value is None,
    }


def r7_knowledge_payload(
    row: pd.Series,
    source: str,
    analysis: dict[str, Any],
    identity_ambiguous: bool = False,
    quantitative_analysis: dict[str, Any] | None = None,
    extraction_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    decision = fuse_r7_decisions(analysis, quantitative_analysis)
    score = int(decision["binary_compatibility_value"])
    outcome = str(decision["outcome"])
    if identity_ambiguous:
        score = 1
        outcome = "UNKNOWN"
    semantic_conflicts = [
        relation
        for relation in analysis.get("relations") or []
        if relation.get("relation") == "CONFLICT"
    ]
    quantitative_conflicts = [
        relation
        for relation in (quantitative_analysis or {}).get("relations") or []
        if relation.get("relation") == "CONFLICT"
    ]
    return {
        "resultno": as_text(row.get("RESULTNO")),
        "target": as_text(row.get("ROCKNAME")),
        "R7": score,
        "evaluator": "Label-free expert-knowledge conflict workflow",
        "workflow_profile": "knowledge_strict",
        "source_text": source,
        "source_metadata": {},
        "knowledge_conflict_analysis": compact_analysis(
            analysis,
            relation_limit=4,
            quote_limit=360,
        ),
        "quantitative_pole_conflict_analysis": compact_r7_quantitative_analysis(
            quantitative_analysis
        ),
        "quantitative_extraction_audit": compact_r7_extraction_audit(
            extraction_audit
        ),
        "qualifying_younger_poles": {
            "status": "PRESENT" if quantitative_conflicts else "NOT_ESTABLISHED",
            "satisfied": bool(quantitative_conflicts),
            "matched_conditions": [
                relation.get("reason_code") for relation in quantitative_conflicts
            ],
            "matched_expert_cases": [],
            "evidence_quotes": [
                {
                    "quote": clean(str((relation.get("evidence") or {}).get("quote") or "")),
                    "section_or_page": (
                        f"page {(relation.get('evidence') or {}).get('page')}"
                        if (relation.get("evidence") or {}).get("page") not in {None, ""}
                        else "original paper"
                    ),
                    "why_relevant": str(relation.get("rule_id") or "R7 conflict rule"),
                }
                for relation in semantic_conflicts[:3]
            ],
            "reasoning": str(decision.get("reason") or ""),
        },
        "condition_summary": knowledge_condition_summary(analysis, "R7"),
        "expert_case_adjustments": [],
        "source_aware_consolidation": None,
        "runtime_scoring_sources": [
            "identity manifest",
            "original paper page text",
            "frozen expert knowledge base",
        ],
        "logic_gate": (
            "structured pole extraction -> stable-craton/R>=3/period gates -> spherical A95 "
            "overlap -> field-age defeater -> semantic fallback -> PASS/FAIL/ABSTAIN"
        ),
        "final_decision_reasoning": (
            "UNKNOWN: multiple result rows share this REFNO and ROCKNAME, and no component "
            "locator is available in the strict identity manifest."
            if identity_ambiguous
            else str(decision.get("reason") or "")
        ),
        "summary": (
            f"Knowledge-strict outcome={outcome}; R7={score}; "
            f"decision_source={decision['decision_source']}; semantic_relation="
            f"{analysis.get('relation')}; quantitative_relation="
            f"{(quantitative_analysis or {}).get('relation', 'UNKNOWN')}."
        ),
        "outcome": outcome,
        "manual_review": outcome == "UNKNOWN" or bool(decision["manual_review"]),
        "manual_review_reason": (
            "indistinguishable result group, source unavailable, evidence insufficient, or supporting and conflicting claims coexist"
            if outcome == "UNKNOWN"
            else "none"
        ),
    }


def r7_payload(
    row: pd.Series,
    text: str,
    source: str,
    metadata: dict[str, str],
    profile: str = "source_text",
    knowledge_analysis: dict[str, Any] | None = None,
    identity_ambiguous: bool = False,
    quantitative_analysis: dict[str, Any] | None = None,
    extraction_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if profile == "knowledge_strict":
        if knowledge_analysis is None:
            raise ValueError("knowledge_strict R7 requires knowledge_analysis")
        return r7_knowledge_payload(
            row,
            source,
            knowledge_analysis,
            identity_ambiguous,
            quantitative_analysis,
            extraction_audit,
        )
    target = as_text(row.get("ROCKNAME"))
    raw_rules = [] if not text.strip() else r57_rules.r7_zero_evidence(text, target)
    rules = (
        []
        if not text.strip()
        else r57_rules.r7_zero_evidence(
            text,
            target,
            source_aware=profile == "astute_source_aware",
        )
    )
    accepted_labels = {rule["label"] for rule in rules}
    filtered_rules = [rule for rule in raw_rules if rule["label"] not in accepted_labels]
    score = 0 if rules else 1
    evidence_patterns = [rule["pattern"] for rule in rules] or [
        r"APWP|apparent polar wander|paleopole|pole|A95|overprint|remagnet|fold test|reversal test"
    ]
    evidence = quote(text, evidence_patterns) if text else "ABSENT"
    return {
        "resultno": as_text(row.get("RESULTNO")),
        "target": target,
        "R7": score,
        "evaluator": "Codex deterministic source-recovered R5/R7 workflow",
        "workflow_profile": profile,
        "source_text": source,
        "source_metadata": metadata,
        "qualifying_younger_poles": {
            "status": "PRESENT" if rules else "ABSENT",
            "satisfied": bool(rules),
            "matched_conditions": [rule["label"] for rule in rules],
            "matched_expert_cases": sorted({rule["case_id"] for rule in rules}),
            "evidence_quotes": [
                {
                    "quote": evidence,
                    "section_or_page": "parsed text/PDF text",
                    "why_relevant": "R7 target-relevant younger-pole/remagnetization evidence.",
                }
            ],
            "reasoning": "Flip to 0 only when target-relevant younger-pole/remagnetization evidence is found.",
        },
        "condition_summary": [
            {
                "key": "younger_overprint",
                "label": "Target-relevant younger-pole or remagnetization evidence",
                "status": "PRESENT" if rules else "ABSENT",
                "satisfied": bool(rules),
                "evidence_quotes": [
                    {
                        "quote": evidence,
                        "section_or_page": "parsed text/PDF text",
                        "why_relevant": "R7 elimination evidence.",
                    }
                ],
                "reasoning": "Matched rules: " + ", ".join(rule["label"] for rule in rules)
                if rules
                else "No R7 flip-to-zero evidence found; preserve default R7=1.",
            }
        ],
        "expert_case_adjustments": [
            {
                "case_id": case_id,
                "applied": True,
                "effect": "assist R7 younger-pole/remagnetization judgement",
                "reasoning": "Matched by reusable R7 source-text expert rules.",
            }
            for case_id in sorted({rule["case_id"] for rule in rules})
        ],
        "source_aware_consolidation": {
            "method": "astute-inspired single-iteration source-aware consolidation",
            "adaptive_internal_candidate": {
                "source": "R7 criterion prior",
                "candidate": 1,
                "reliability": "medium",
            },
            "external_candidates": [
                {
                    "source": "retrieved paper text",
                    "candidate": 0,
                    "reliability": "high",
                    "claim": rule["label"],
                }
                for rule in rules
            ],
            "filtered_irrelevant_candidates": [
                {
                    "source": "retrieved paper text",
                    "claim": rule["label"],
                    "reason": "The claim names Phalaborwa but the current target is not Phalaborwa.",
                }
                for rule in filtered_rules
            ],
            "resolution": (
                "external target-specific evidence overrides the internal default"
                if rules
                else "internal default retained after irrelevant external claims were filtered"
                if filtered_rules
                else "internal default retained because no conflicting external claim was found"
            ),
            "selected_candidate": score,
        }
        if profile == "astute_source_aware"
        else None,
        "logic_gate": (
            "R7 = 1 internal default; flip to 0 only after source-aware target relevance filtering."
            if profile == "astute_source_aware"
            else "R7 = 1 default; flip to 0 only with target-relevant younger-pole/remagnetization evidence."
        ),
        "final_decision_reasoning": f"R7={score}; matched {len(rules)} source-text rule(s).",
        "summary": (
            f"R7={score}; matched rules: {', '.join(rule['label'] for rule in rules) if rules else 'none'}; "
            f"filtered claims: {', '.join(rule['label'] for rule in filtered_rules) if filtered_rules else 'none'}."
        ),
        "manual_review": not bool(text.strip()) or bool(filtered_rules),
        "manual_review_reason": (
            "source text/PDF unavailable; preserved default R7=1"
            if not text.strip()
            else "source-level claim was filtered because it was not anchored to the current target"
            if filtered_rules
            else "none"
        ),
    }


def build_rows(
    indicator: str,
    targets: pd.DataFrame,
    parsed_dir: Path,
    pdf_dir: Path,
    rockunit_by_no: dict[str, dict[str, str]],
    pmag_by_resultno: dict[str, dict[str, str]],
    r5_profile: str,
    r7_profile: str,
    r5_metadata_zero_rules: dict[str, Any] | None = None,
    page_cache: dict[str, tuple[str, str]] | None = None,
    page_cache_rows: dict[str, dict[str, Any]] | None = None,
    r5_paper_model: FrozenPaperTextModel | None = None,
    knowledge_engine: ExpertConflictEngine | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    page_cache = page_cache or {}
    page_cache_rows = page_cache_rows or {}
    text_cache: OrderedDict[str, tuple[str, str]] = OrderedDict()
    paper_retrievers: OrderedDict[str, PaperTextRetriever] = OrderedDict()
    strict_analysis_cache: dict[tuple[str, str, str, bool, bool], dict[str, Any]] = {}
    r7_quantitative_cache: dict[
        tuple[str, str, str, bool, bool],
        tuple[dict[str, Any] | None, dict[str, Any]],
    ] = {}
    ref_counts = targets.groupby("REFNO")["RESULTNO"].nunique().to_dict()
    identity_counts = targets.groupby(["REFNO", "ROCKNAME"])["RESULTNO"].nunique().to_dict()
    for _, row in targets.iterrows():
        refno = as_text(row.get("REFNO"))
        resultno = as_text(row.get("RESULTNO"))
        target = as_text(row.get("ROCKNAME"))
        if refno not in text_cache:
            cache_record = page_cache_rows.get(refno)
            if cache_record:
                pages = cache_record.get("pages") or []
                page_text = "\n\n".join(
                    str(page.get("text", ""))
                    for page in pages
                    if str(page.get("text", "")).strip()
                )
                source_meta = cache_record.get("source") or {}
                source_kind = str(source_meta.get("source_kind", "page_cache"))
                text_cache[refno] = (page_text, f"page_cache#{refno}:{source_kind}")
            else:
                text_cache[refno] = page_cache.get(refno) or source_text(parsed_dir, pdf_dir, refno)
            if len(text_cache) > 8:
                text_cache.popitem(last=False)
        else:
            text_cache.move_to_end(refno)
        text, source = text_cache[refno]
        strict_profile = (
            indicator == "R5" and r5_profile == "knowledge_strict"
        ) or (
            indicator == "R7" and r7_profile == "knowledge_strict"
        )
        metadata = (
            {}
            if strict_profile
            else source_metadata(row, rockunit_by_no, pmag_by_resultno)
        )
        cache_row = page_cache_rows.get(refno) or {
            "REFNO": refno,
            "source": {
                "source_kind": "flattened_text_fallback" if text.strip() else "missing",
                "source_quality_flags": [] if text.strip() else ["no_extractable_text"],
            },
            "pages": [{"page": 1, "text": text, "source": "flattened_text_fallback"}]
            if text.strip()
            else [],
        }
        multi_target = int(ref_counts.get(refno, 1)) > 1
        if indicator == "R5":
            identity_ambiguous = int(identity_counts.get((refno, target), 1)) > 1
            if refno not in paper_retrievers:
                paper_retrievers[refno] = PaperTextRetriever(cache_row)
                if len(paper_retrievers) > 8:
                    paper_retrievers.popitem(last=False)
            else:
                paper_retrievers.move_to_end(refno)
            paper_retrieval = paper_retrievers[refno].retrieve(
                target,
                multi_target,
                "R5",
                limit=12,
                per_category_limit=4,
            )
            paper_calibration = None
            if r5_paper_model is not None:
                document = build_target_document(cache_row, target, paper_retrieval)
                paper_calibration = r5_paper_model.predict(document)
            knowledge_analysis = None
            if r5_profile == "knowledge_strict":
                if knowledge_engine is None:
                    raise ValueError("knowledge_strict R5 requires a knowledge engine")
                analysis_key = ("R5", refno, target, multi_target, identity_ambiguous)
                knowledge_analysis = strict_analysis_cache.get(analysis_key)
                if knowledge_analysis is None:
                    knowledge_analysis = knowledge_engine.evaluate(
                        "R5",
                        target,
                        retrieval=paper_retrieval,
                        pages=cache_row,
                        multi_target=multi_target,
                    )
                    strict_analysis_cache[analysis_key] = knowledge_analysis
            payload = r5_payload(
                row,
                text,
                source,
                metadata,
                r5_profile,
                r5_metadata_zero_rules,
                paper_retrieval,
                identity_ambiguous,
                paper_calibration,
                knowledge_analysis,
            )
            score = payload["R5"]
        else:
            knowledge_analysis = None
            quantitative_analysis = None
            extraction_audit: dict[str, Any] | None = None
            identity_ambiguous = int(identity_counts.get((refno, target), 1)) > 1
            if r7_profile == "knowledge_strict":
                if knowledge_engine is None:
                    raise ValueError("knowledge_strict R7 requires a knowledge engine")
                analysis_key = ("R7", refno, target, multi_target, identity_ambiguous)
                knowledge_analysis = strict_analysis_cache.get(analysis_key)
                if knowledge_analysis is None:
                    knowledge_analysis = knowledge_engine.evaluate(
                        "R7",
                        target,
                        pages=cache_row,
                        multi_target=multi_target,
                    )
                    strict_analysis_cache[analysis_key] = knowledge_analysis
                quantitative_cached = r7_quantitative_cache.get(analysis_key)
                if quantitative_cached is None:
                    quantitative_knowledge = (
                        knowledge_engine.knowledge_base.get("r7_quantitative_knowledge")
                        or {}
                    )
                    safe_page_row = {
                        "pages": [
                            {
                                key: page.get(key)
                                for key in (
                                    "page",
                                    "text",
                                    "text_sha256",
                                    "section_hints",
                                )
                                if key in page
                            }
                            for page in (cache_row.get("pages") or [])
                        ]
                    }
                    extracted = extract_r7_original_text_evidence(
                        target,
                        safe_page_row,
                        quantitative_knowledge.get("reference_poles"),
                        quantitative_knowledge.get("qualified_search_manifest"),
                    )
                    extraction_audit = extracted["extraction_audit"]
                    if extracted.get("target") is not None:
                        quantitative_analysis = evaluate_r7_pole_overlap(
                            extracted["target"],
                            extracted.get("candidates") or [],
                            extracted.get("field_test_age_constraints"),
                            extracted.get("qualified_search"),
                        )
                    quantitative_cached = (quantitative_analysis, extraction_audit)
                    r7_quantitative_cache[analysis_key] = quantitative_cached
                quantitative_analysis, extraction_audit = quantitative_cached
            payload = r7_payload(
                row,
                text,
                source,
                metadata,
                r7_profile,
                knowledge_analysis,
                identity_ambiguous,
                quantitative_analysis,
                extraction_audit,
            )
            score = payload["R7"]
        rows.append(
            {
                "REFNO": refno,
                "RESULTNO": resultno,
                "TARGET": target,
                indicator: score,
                f"{indicator}_RESULT_JSON": json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return rows


def write_predictions(rows: list[dict[str, Any]], output: Path, indicator: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["REFNO", "RESULTNO", "TARGET", indicator, f"{indicator}_RESULT_JSON"]
    oversized = [
        row["RESULTNO"]
        for row in rows
        if len(str(row[f"{indicator}_RESULT_JSON"])) > 32767
    ]
    if oversized:
        raise SystemExit(
            f"{indicator} JSON exceeds the Excel cell limit for RESULTNO values: "
            + ", ".join(oversized[:20])
        )
    pd.DataFrame(rows)[columns].to_excel(output, index=False)
    counts = pd.DataFrame(rows)[indicator].value_counts().sort_index().to_dict()
    print(f"Saved {indicator}: {output}")
    print(f"Rows: {len(rows)}")
    print(f"Score counts: {counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indicator", choices=["R5", "R7", "both"], default="both")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--rockunit", type=Path, default=DEFAULT_ROCKUNIT)
    parser.add_argument("--pmagresult", type=Path, default=DEFAULT_PMAGRESULT)
    parser.add_argument("--page-cache", type=Path, default=None, help="Optional R3 page source cache JSONL to reuse stable source text by REFNO.")
    parser.add_argument("--knowledge-base", type=Path, default=DEFAULT_KNOWLEDGE_BASE)
    parser.add_argument(
        "--r5-zero-rules",
        type=Path,
        default=DEFAULT_R5_METADATA_ZERO_RULES,
        help="Calibrated source metadata taxonomy used by the metadata_calibrated R5 profile.",
    )
    parser.add_argument(
        "--r5-paper-calibration",
        type=Path,
        default=DEFAULT_R5_PAPER_CALIBRATION,
        help="Frozen REFNO-grouped lexical calibration used by the paper_only R5 profile.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output workbook for a single indicator run.")
    parser.add_argument("--r5-output", type=Path, default=None)
    parser.add_argument("--r7-output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional target-row limit for smoke tests.")
    parser.add_argument(
        "--r5-profile",
        choices=[
            "knowledge_strict",
            "paper_only",
            "semantic",
            "metadata_calibrated",
            "source_augmented",
        ],
        default="knowledge_strict",
    )
    parser.add_argument(
        "--r7-profile",
        choices=["knowledge_strict", "source_text", "astute_source_aware"],
        default="knowledge_strict",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = pd.read_excel(
        args.targets,
        usecols=["REFNO", "RESULTNO", "ROCKNAME"],
        dtype={"REFNO": str, "RESULTNO": str, "ROCKNAME": str},
    ).fillna("")
    if args.limit is not None:
        targets = targets.head(args.limit).copy()
    required = {"REFNO", "RESULTNO", "ROCKNAME"}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise SystemExit(f"{args.targets} is missing required columns: {', '.join(missing)}")

    indicators = ["R5", "R7"] if args.indicator == "both" else [args.indicator]
    needs_source_tables = (
        "R7" in indicators and args.r7_profile != "knowledge_strict"
    ) or (
        "R5" in indicators
        and args.r5_profile not in {"knowledge_strict", "paper_only"}
    )
    if needs_source_tables:
        rockunit, pmag = load_source_tables(args.rockunit, args.pmagresult)
        rockunit_by_no = row_lookup(rockunit, "ROCKUNITNO")
        pmag_by_resultno = row_lookup(pmag, "RESULTNO")
    else:
        rockunit_by_no = {}
        pmag_by_resultno = {}
    page_cache_rows = load_page_cache_rows(args.page_cache)
    page_cache: dict[str, tuple[str, str]] = {}
    r5_metadata_zero_rules = (
        load_r5_metadata_zero_rules(args.r5_zero_rules)
        if args.r5_profile in {"metadata_calibrated", "source_augmented"}
        else {}
    )
    r5_paper_model = None
    if "R5" in indicators and args.r5_profile == "paper_only":
        if not args.r5_paper_calibration.exists():
            raise SystemExit(
                "paper_only R5 requires a frozen calibration artifact: "
                f"{args.r5_paper_calibration}"
            )
        r5_paper_model = FrozenPaperTextModel(args.r5_paper_calibration, "R5")
    knowledge_engine = (
        ExpertConflictEngine.from_path(args.knowledge_base)
        if (
            "R5" in indicators and args.r5_profile == "knowledge_strict"
        ) or (
            "R7" in indicators and args.r7_profile == "knowledge_strict"
        )
        else None
    )

    for indicator in indicators:
        rows = build_rows(
            indicator,
            targets,
            args.parsed_dir,
            args.pdf_dir,
            rockunit_by_no,
            pmag_by_resultno,
            args.r5_profile,
            args.r7_profile,
            r5_metadata_zero_rules,
            page_cache,
            page_cache_rows,
            r5_paper_model,
            knowledge_engine,
        )
        if args.output and len(indicators) == 1:
            output = args.output
        elif indicator == "R5" and args.r5_output:
            output = args.r5_output
        elif indicator == "R7" and args.r7_output:
            output = args.r7_output
        else:
            suffix = args.r5_profile if indicator == "R5" else args.r7_profile
            output = args.output_dir / f"{indicator.lower()}_{suffix}_predictions.xlsx"
        write_predictions(rows, output, indicator)


if __name__ == "__main__":
    main()
