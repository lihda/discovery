#!/usr/bin/env python3
"""Build label-free R3 evidence packs for Codex direct review.

The script is intentionally not an evaluator. It reads only target identifiers
and local source text/PDFs, then packages target anchors, source-quality flags,
and candidate passages for a human/Codex reviewer to score later.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TargetRow:
    sequence: int
    refno: str
    resultno: str
    target: str


@dataclass
class RefContext:
    text: str
    source_meta: dict[str, Any]
    lower_text: str
    offsets: list[int]
    candidate_pools: dict[str, list[dict[str, Any]]]


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", text)


def token_sort_key(value: str) -> tuple[int, str]:
    return (0, f"{int(value):012d}") if value.isdigit() else (1, value.casefold())


def load_targets(path: Path, limit: int | None = None) -> list[TargetRow]:
    frame = pd.read_excel(
        path,
        usecols=["REFNO", "RESULTNO", "ROCKNAME"],
        dtype={"REFNO": str, "RESULTNO": str, "ROCKNAME": str},
    )
    if limit is not None:
        frame = frame.head(limit)
    rows: list[TargetRow] = []
    for index, row in enumerate(frame.itertuples(index=False), start=1):
        rows.append(
            TargetRow(
                sequence=index,
                refno=clean(row.REFNO),
                resultno=clean(row.RESULTNO),
                target=clean(row.ROCKNAME),
            )
        )
    return rows


def load_pmagresult_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None or not path.exists():
        return {}
    columns = [
        "RESULTNO",
        "ROCKMAG",
        "LABDETAILS",
        "TREATMENT",
        "DEMAGCODE",
        "COMMENTS",
        "COMPONENT",
    ]
    frame = pd.read_excel(
        path,
        usecols=lambda column: column in columns,
        dtype={"RESULTNO": str},
    )
    if "RESULTNO" not in frame.columns:
        return {}

    def clean_cell(value: object) -> str:
        if pd.isna(value):
            return ""
        return clean(value)

    metadata: dict[str, dict[str, str]] = {}
    for row in frame.to_dict(orient="records"):
        resultno = clean_cell(row.get("RESULTNO"))
        if not resultno:
            continue
        metadata[resultno] = {
            column: clean_cell(row.get(column))
            for column in columns
            if column != "RESULTNO" and column in frame.columns
        }
    return metadata


def load_canonical_ref_counts(pmagresult_path: Path, rockunit_path: Path) -> dict[str, int]:
    pmag = pd.read_excel(
        pmagresult_path,
        usecols=["RESULTNO", "ROCKUNITNO"],
        dtype={"RESULTNO": str, "ROCKUNITNO": str},
    ).fillna("")
    rockunit = pd.read_excel(
        rockunit_path,
        usecols=["ROCKUNITNO", "REFNO"],
        dtype={"ROCKUNITNO": str, "REFNO": str},
    ).fillna("")
    for frame, columns in ((pmag, ("RESULTNO", "ROCKUNITNO")), (rockunit, ("ROCKUNITNO", "REFNO"))):
        for column in columns:
            frame[column] = frame[column].map(clean)
    if pmag["RESULTNO"].eq("").any() or pmag["RESULTNO"].duplicated().any():
        raise SystemExit(f"{pmagresult_path} must contain unique, non-empty RESULTNO values")
    if rockunit["ROCKUNITNO"].eq("").any() or rockunit["ROCKUNITNO"].duplicated().any():
        raise SystemExit(f"{rockunit_path} must contain unique, non-empty ROCKUNITNO values")
    merged = pmag.merge(rockunit, on="ROCKUNITNO", how="left", validate="many_to_one")
    if merged["REFNO"].eq("").any():
        examples = ", ".join(merged.loc[merged["REFNO"].eq(""), "RESULTNO"].head(10))
        raise SystemExit(f"Missing ROCKUNIT REFNO for RESULTNO values: {examples}")
    return merged.groupby("REFNO")["RESULTNO"].nunique().astype(int).to_dict()


def target_tokens(target: str) -> list[str]:
    raw = re.split(r"[^A-Za-z0-9]+", target)
    stop = {
        "formation",
        "group",
        "member",
        "rocks",
        "rock",
        "dykes",
        "dikes",
        "beds",
        "series",
        "suite",
        "complex",
    }
    tokens = [part for part in raw if len(part) >= 4 and part.casefold() not in stop]
    return list(dict.fromkeys(tokens))


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    for match in re.finditer(r"\n", text):
        offsets.append(match.end())
    return offsets


def line_number(offsets: list[int], position: int) -> int:
    lo = 0
    hi = len(offsets)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if offsets[mid] <= position:
            lo = mid
        else:
            hi = mid
    return lo + 1


def excerpt(text: str, start: int, end: int, radius: int = 650) -> str:
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = clean(text[left:right])
    return snippet[:1600]


def load_page_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[str(row["REFNO"])] = row
    return cache


def page_cache_text(cache_row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    pages = cache_row.get("pages") or []
    chunks: list[str] = []
    offset = 0
    page_spans: list[dict[str, Any]] = []
    for page in pages:
        page_text = normalize_text(str(page.get("text", "")))
        start = offset
        chunks.append(page_text)
        offset += len(page_text)
        page_spans.append(
            {
                "start": start,
                "end": offset,
                "page": page.get("page"),
                "section_hints": page.get("section_hints") or [],
                "source": page.get("source"),
            }
        )
        chunks.append("\n")
        offset += 1
    source = dict(cache_row.get("source") or {})
    source["page_spans"] = page_spans
    return "".join(chunks), source


def page_context(source_meta: dict[str, Any], position: int) -> dict[str, Any]:
    for span in source_meta.get("page_spans", []) or []:
        if span["start"] <= position <= span["end"]:
            return {
                "page": span.get("page"),
                "section_hint": "; ".join(span.get("section_hints") or [])[:220],
                "page_source": span.get("source"),
                "page_offset": position - span["start"],
            }
    return {}


def compile_patterns(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.I | re.S) for pattern in patterns]


INVESTIGATION_PATTERNS = compile_patterns(
    [
        r"\bIRM\b|isothermal remanent magneti[sz]ation|back[- ]field|coerciv",
        r"Lowrie|three[- ]axis|3[- ]axis",
        r"thermomagnetic|Curie|N[eé]el|susceptibility.{0,80}temperature",
        r"hysteresis|Day plot|FORC|domain state|single[- ]domain|pseudo[- ]single[- ]domain|multi[- ]domain",
        r"unblocking temperature|blocking temperature|thermal demagneti[sz]ation|AF demagneti[sz]ation",
        r"AMS|anisotropy of magnetic susceptibility|AAR|AIR",
        r"reflected[- ]light|ore microscop|thin section|petrograph",
        r"\bSEM\b|\bBSE\b|\bTEM\b|\bEDS\b|\bWDS\b|electron microprobe|\bEPMA\b|micro[- ]imaging",
        r"mineral separation|magnetic extract|opaque mineral|Fe[- ]Ti oxide|titanomagnetite|ilmenite lamellae",
    ]
)

CARRIER_PATTERNS = compile_patterns(
    [
        r"magnetite|titanomagnetite|titano[- ]magnetite|hematite|haematite|maghemite|goethite|pyrrhotite|greigite",
        r"iron sulphide|iron sulfide|Fe[- ]Ti oxide|ferrimagnetic|antiferromagnetic",
        r"low[- ]coercivity|high[- ]coercivity|soft component|hard component",
    ]
)

SIGNIFICANCE_PATTERNS = compile_patterns(
    [
        r"carr(?:y|ies|ied|ier|iers).{0,180}(ChRM|characteristic|remanence|component|magneti[sz]ation)",
        r"(ChRM|characteristic|component|remanence|magneti[sz]ation).{0,180}carr(?:y|ies|ied|ier|iers)",
        r"(primary|secondary|overprint|remagneti[sz]|CRM|TRM|DRM|VRM|detrital|chemical|authigenic|alteration).{0,220}(magnetite|hematite|haematite|carrier|component|remanence)",
        r"(magnetite|hematite|haematite|carrier|component).{0,220}(primary|secondary|overprint|remagneti[sz]|CRM|TRM|DRM|VRM|stability|stable|reliab)",
        r"(domain state|coerciv|blocking temperature|unblocking temperature|single[- ]domain|pseudo[- ]single[- ]domain).{0,220}(stable|stability|ChRM|component|remanence|reliab)",
    ]
)

WEAK_DIRECTIONAL_PATTERNS = compile_patterns(
    [
        r"fold test|reversal test|contact test|baked contact|conglomerate test",
        r"APWP|apparent polar wander|pole.{0,80}agree|palaeopole|paleopole",
        r"stable component|well[- ]defined component|primary direction|primary magneti[sz]ation",
        r"polarity|normal polarity|reverse polarity|reversed polarity",
    ]
)

SECTION_PATTERNS = compile_patterns(
    [
        r"magnetic mineralogy|rock magneti[cs]|palaeomagnetic result|paleomagnetic result",
        r"demagneti[sz]ation|remanence|magneti[sz]ation component|discussion|conclusion",
        r"sample|site|formation|member|dyke|dike|basalt|sandstone|limestone|granite|gneiss",
    ]
)


def pdf_text_via_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        return ""
    try:
        raw = "\n".join((page.extract_text() or "") for page in PdfReader(str(pdf_path)).pages)
    except Exception:
        return ""
    if len(re.findall(r"/C\d{2,3}", raw)) >= 50:
        raw = re.sub(
            r"/C(\d{2,3})",
            lambda m: chr(int(m.group(1))) if 32 <= int(m.group(1)) <= 126 else " ",
            raw,
        )
    raw = re.sub(r"\(cid:\d+\)", " ", raw)
    return normalize_text(raw)


def glyph_noise(text: str) -> int:
    return text.count("(cid:") + len(re.findall(r"/C\d{2,3}", text))


def alpha_count(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def find_pdf(refno: str, pdf_dirs: list[Path]) -> Path | None:
    for directory in pdf_dirs:
        path = directory / f"{refno}.pdf"
        if path.exists():
            return path
    return None


def load_source(refno: str, parsed_dir: Path, pdf_dirs: list[Path]) -> tuple[str, dict[str, Any]]:
    parsed = parsed_dir / f"{refno}.txt"
    pdf_path = find_pdf(refno, pdf_dirs)
    text = ""
    source_kind = "missing"
    source_path = None
    if parsed.exists() and parsed.stat().st_size > 0:
        text = normalize_text(parsed.read_text(encoding="utf-8", errors="ignore"))
        source_kind = "parsed_text"
        source_path = str(parsed)
    if (not text.strip() or glyph_noise(text) > 50) and pdf_path:
        recovered = pdf_text_via_pypdf(pdf_path)
        recovered_is_better = (
            recovered.strip()
            and (
                not text.strip()
                or glyph_noise(recovered) < glyph_noise(text)
                or alpha_count(recovered) > alpha_count(text) * 1.2
            )
        )
        if recovered_is_better:
            text = recovered
            source_kind = "pdf_glyph_or_direct_extraction"
            source_path = str(pdf_path)
    flags: list[str] = []
    if not text.strip():
        flags.append("no_local_text")
    if not pdf_path:
        flags.append("no_local_pdf")
    if len(text.strip()) < 2000:
        flags.append("short_text")
    if pdf_path and not text.strip():
        flags.append("pdf_text_empty_or_image_only")
    if glyph_noise(text) > 20:
        flags.append("glyph_or_cid_text")
    lower = text.casefold()
    if text and not any(word in lower for word in ["abstract", "introduction", "discussion", "conclusion", "magnetic"]):
        flags.append("possibly_incomplete_extraction")
    meta = {
        "source_kind": source_kind,
        "source_path": source_path,
        "pdf_path": str(pdf_path) if pdf_path else None,
        "text_length": len(text),
        "line_count": text.count("\n") + 1 if text else 0,
        "source_quality_flags": flags,
    }
    return text, meta


def load_source_with_overrides(
    refno: str,
    parsed_dir: Path,
    pdf_dirs: list[Path],
    ocr_dir: Path | None = None,
) -> tuple[str, dict[str, Any]]:
    text, meta = load_source(refno, parsed_dir, pdf_dirs)
    if ocr_dir is None:
        return text, meta
    ocr_path = ocr_dir / f"{refno}_ocr.txt"
    if not ocr_path.exists() or ocr_path.stat().st_size == 0:
        return text, meta
    ocr_text = normalize_text(ocr_path.read_text(encoding="utf-8", errors="ignore"))
    if alpha_count(ocr_text) <= alpha_count(text):
        return text, meta
    meta = {
        **meta,
        "source_kind": "vision_ocr_override",
        "source_path": str(ocr_path),
        "text_length": len(ocr_text),
        "line_count": ocr_text.count("\n") + 1 if ocr_text else 0,
        "source_quality_flags": [
            flag
            for flag in meta.get("source_quality_flags", [])
            if flag not in {"no_local_text", "short_text", "pdf_text_empty_or_image_only"}
        ],
        "ocr_override_of": meta.get("source_kind"),
    }
    return ocr_text, meta


def collect_matches(
    text: str,
    patterns: list[re.Pattern[str]],
    label: str,
    target_token_patterns: list[re.Pattern[str]],
    source_meta: dict[str, Any] | None = None,
    max_items: int = 6,
) -> list[dict[str, Any]]:
    offsets = line_offsets(text)
    candidates: list[tuple[int, int, re.Match[str], bool]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            left = max(0, match.start() - 900)
            right = min(len(text), match.end() + 900)
            window = text[left:right]
            target_near = any(token.search(window) for token in target_token_patterns)
            candidates.append((0 if target_near else 1, match.start(), match.end(), match, target_near))
            if len(candidates) > max_items * 12:
                break
    candidates.sort(key=lambda item: (item[0], item[1]))
    seen: set[int] = set()
    results: list[dict[str, Any]] = []
    for _, start, end, match, target_near in candidates:
        bucket = start // 600
        if bucket in seen:
            continue
        seen.add(bucket)
        result = {
            "category": label,
            "matched_text": clean(match.group(0))[:220],
            "line": line_number(offsets, start),
            "target_token_nearby": target_near,
            "excerpt": excerpt(text, start, end),
        }
        if source_meta:
            result.update(page_context(source_meta, start))
        results.append(result)
        if len(results) >= max_items:
            break
    return results


def collect_match_pool(
    text: str,
    patterns: list[re.Pattern[str]],
    label: str,
    source_meta: dict[str, Any] | None = None,
    max_items: int = 6,
) -> list[dict[str, Any]]:
    offsets = line_offsets(text)
    pool: list[dict[str, Any]] = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            left = max(0, match.start() - 900)
            right = min(len(text), match.end() + 900)
            item = {
                "category": label,
                "matched_text": clean(match.group(0))[:220],
                "line": line_number(offsets, match.start()),
                "excerpt": excerpt(text, match.start(), match.end()),
                "_start": match.start(),
                "_end": match.end(),
                "_target_window": text[left:right],
            }
            if source_meta:
                item.update(page_context(source_meta, match.start()))
            pool.append(item)
            if len(pool) > max_items * 12:
                break
    return pool


def select_matches_from_pool(
    pool: list[dict[str, Any]],
    target_token_patterns: list[re.Pattern[str]],
    max_items: int = 6,
) -> list[dict[str, Any]]:
    candidates: list[tuple[int, int, dict[str, Any], bool]] = []
    for item in pool:
        window = str(item.get("_target_window", ""))
        target_near = any(token.search(window) for token in target_token_patterns)
        start = int(item.get("_start", 0))
        candidates.append((0 if target_near else 1, start, item, target_near))
    candidates.sort(key=lambda item: (item[0], item[1]))
    seen: set[int] = set()
    results: list[dict[str, Any]] = []
    for _, start, item, target_near in candidates:
        bucket = start // 600
        if bucket in seen:
            continue
        seen.add(bucket)
        result = {
            key: value
            for key, value in item.items()
            if key not in {"_start", "_end", "_target_window"}
        }
        result["target_token_nearby"] = target_near
        results.append(result)
        if len(results) >= max_items:
            break
    return results


def build_ref_candidate_pools(text: str, source_meta: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "carrier_investigation_candidates": collect_match_pool(
            text, INVESTIGATION_PATTERNS, "carrier_investigation", source_meta
        ),
        "carrier_identity_candidates": collect_match_pool(
            text, CARRIER_PATTERNS, "carrier_identity", source_meta
        ),
        "carrier_significance_candidates": collect_match_pool(
            text, SIGNIFICANCE_PATTERNS, "carrier_significance", source_meta
        ),
        "directional_or_weak_substitution_warnings": collect_match_pool(
            text, WEAK_DIRECTIONAL_PATTERNS, "directional_or_weak_substitution", source_meta, max_items=4
        ),
        "section_candidates": collect_match_pool(
            text, SECTION_PATTERNS, "section_or_context", source_meta, max_items=4
        ),
    }


def target_anchors(
    text: str,
    target: str,
    source_meta: dict[str, Any] | None = None,
    *,
    lower_text: str | None = None,
    offsets: list[int] | None = None,
) -> list[dict[str, Any]]:
    tokens = target_tokens(target)
    anchors: list[dict[str, Any]] = []
    lower_text = lower_text if lower_text is not None else text.casefold()
    offsets = offsets if offsets is not None else line_offsets(text)
    for token in tokens[:8]:
        start = lower_text.find(token.casefold())
        if start < 0:
            continue
        end = start + len(token)
        anchor = {
            "token": text[start:end],
            "line": line_number(offsets, start),
            "excerpt": excerpt(text, start, end, radius=360),
        }
        if source_meta:
            anchor.update(page_context(source_meta, start))
        anchors.append(anchor)
        if len(anchors) >= 5:
            break
    return anchors


def build_pack(
    row: TargetRow,
    text: str,
    source_meta: dict[str, Any],
    multi_target: bool,
    pmagresult_metadata: dict[str, str] | None = None,
    candidate_pools: dict[str, list[dict[str, Any]]] | None = None,
    lower_text: str | None = None,
    offsets: list[int] | None = None,
) -> dict[str, Any]:
    tokens = target_tokens(row.target)
    token_patterns = [re.compile(re.escape(token), re.I) for token in tokens]
    anchors = target_anchors(
        text,
        row.target,
        source_meta,
        lower_text=lower_text,
        offsets=offsets,
    )
    if candidate_pools is None:
        candidate_pools = build_ref_candidate_pools(text, source_meta)
    evidence = {
        "target_context": anchors,
        "carrier_investigation_candidates": select_matches_from_pool(
            candidate_pools["carrier_investigation_candidates"], token_patterns
        ),
        "carrier_identity_candidates": select_matches_from_pool(
            candidate_pools["carrier_identity_candidates"], token_patterns
        ),
        "carrier_significance_candidates": select_matches_from_pool(
            candidate_pools["carrier_significance_candidates"], token_patterns
        ),
        "directional_or_weak_substitution_warnings": select_matches_from_pool(
            candidate_pools["directional_or_weak_substitution_warnings"], token_patterns, max_items=4
        ),
        "section_candidates": select_matches_from_pool(
            candidate_pools["section_candidates"], token_patterns, max_items=4
        ),
    }
    reviewer_notes = [
        "This pack contains no standard labels, gold scores, or historical predictions.",
        "Score R3=1 only if Evidence A, B, and C are all satisfied for this RESULTNO/target.",
        "Directional tests, APWP agreement, and generic primary conclusions are warnings unless tied to carrier/component evidence.",
        "In multi-target papers, write a target anchor before assigning R3=1.",
    ]
    if source_meta["source_quality_flags"]:
        reviewer_notes.append(
            "Source quality is flagged; apply R3_case_010 before treating missing evidence as a stable negative."
        )
    if multi_target:
        reviewer_notes.append("This REFNO has multiple target rows; apply R3_case_009 target-specific anchoring.")
    metadata = pmagresult_metadata or {}
    rockmag = clean(metadata.get("ROCKMAG"))
    if rockmag:
        reviewer_notes.append(
            "PMAGRESULT.ROCKMAG is present and should be treated as RESULTNO-specific source metadata evidence for R3."
        )
    return {
        "sequence": row.sequence,
        "REFNO": row.refno,
        "RESULTNO": row.resultno,
        "TARGET": row.target,
        "multi_target_refno": multi_target,
        "target_tokens": tokens,
        "source": source_meta,
        "source_metadata": {
            "pmagresult": metadata,
            "metadata_notes": "PMAGRESULT metadata is joined by RESULTNO and contains no target label columns.",
        },
        "candidate_evidence": evidence,
        "reviewer_notes": reviewer_notes,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "data/targets.xlsx")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--parsed-dir", type=Path, default=ROOT / "data/parsed_mds")
    parser.add_argument("--pdf-dir", action="append", type=Path, default=[ROOT / "data/raw_pdfs"])
    parser.add_argument("--ocr-dir", type=Path, default=None)
    parser.add_argument("--page-cache", type=Path, default=None)
    parser.add_argument("--pmagresult", type=Path, default=ROOT / "data/source_metadata/PMAGRESULT.xlsx")
    parser.add_argument("--rockunit", type=Path, default=ROOT / "data/source_metadata/ROCKUNIT.xlsx")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()

    rows = load_targets(args.targets, args.limit)
    pmag_metadata = load_pmagresult_metadata(args.pmagresult)
    target_ref_counts: dict[str, int] = {}
    for row in rows:
        target_ref_counts[row.refno] = target_ref_counts.get(row.refno, 0) + 1
    canonical_ref_counts = load_canonical_ref_counts(args.pmagresult, args.rockunit)

    source_cache: dict[str, tuple[str, dict[str, Any]]] = {}
    ref_context_cache: dict[str, RefContext] = {}
    page_cache = load_page_cache(args.page_cache)
    packs: list[dict[str, Any]] = []
    for row in rows:
        if row.refno not in source_cache:
            if row.refno in page_cache:
                source_cache[row.refno] = page_cache_text(page_cache[row.refno])
            else:
                source_cache[row.refno] = load_source_with_overrides(
                    row.refno,
                    args.parsed_dir,
                    args.pdf_dir,
                    args.ocr_dir,
                )
        if row.refno not in ref_context_cache:
            text, source_meta = source_cache[row.refno]
            ref_context_cache[row.refno] = RefContext(
                text=text,
                source_meta=source_meta,
                lower_text=text.casefold(),
                offsets=line_offsets(text),
                candidate_pools=build_ref_candidate_pools(text, source_meta),
            )
        context = ref_context_cache[row.refno]
        packs.append(
            build_pack(
                row,
                context.text,
                context.source_meta,
                canonical_ref_counts.get(row.refno, target_ref_counts[row.refno]) > 1,
                pmag_metadata.get(row.resultno),
                context.candidate_pools,
                context.lower_text,
                context.offsets,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    packs_jsonl = args.output_dir / "r3_evidence_packs.jsonl"
    packs_json = args.output_dir / "r3_evidence_packs.json"
    write_jsonl(packs_jsonl, packs)
    packs_json.write_text(json.dumps(packs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    batch_paths: list[str] = []
    for index in range(0, len(packs), args.batch_size):
        batch = packs[index : index + args.batch_size]
        batch_number = index // args.batch_size + 1
        path = args.output_dir / f"r3_evidence_batch_{batch_number:02d}.json"
        path.write_text(json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        batch_paths.append(str(path))

    source_flags: dict[str, int] = {}
    for pack in packs:
        flags = pack["source"]["source_quality_flags"] or ["none"]
        for flag in flags:
            source_flags[flag] = source_flags.get(flag, 0) + 1
    summary = {
        "targets": str(args.targets),
        "target_rows": len(rows),
        "unique_refnos": len(target_ref_counts),
        "batch_size": args.batch_size,
        "batch_count": len(batch_paths),
        "packs_jsonl": str(packs_jsonl),
        "packs_json": str(packs_json),
        "batch_paths": batch_paths,
        "source_quality_flag_counts": dict(sorted(source_flags.items())),
        "page_cache": str(args.page_cache) if args.page_cache else None,
        "pmagresult": str(args.pmagresult) if args.pmagresult else None,
        "rockunit": str(args.rockunit) if args.rockunit else None,
        "pmagresult_rows_joined": sum(1 for row in rows if row.resultno in pmag_metadata),
        "multi_target_basis": "canonical PMAGRESULT.RESULTNO -> ROCKUNIT.REFNO universe",
        "multi_target_rows": sum(
            1
            for row in rows
            if canonical_ref_counts.get(row.refno, target_ref_counts[row.refno]) > 1
        ),
        "label_exclusion": "Only REFNO, RESULTNO, and ROCKNAME columns are read from the target workbook.",
    }
    (args.output_dir / "r3_evidence_pack_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
