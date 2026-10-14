#!/usr/bin/env python3
"""Build page-aware source text caches for R3 evidence extraction.

The cache is label-free. It reads local parsed text/PDFs and writes one JSONL
record per REFNO with page text, source-quality flags, and section hints.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


SECTION_HINT = re.compile(
    r"^(?:\\d+(?:\\.\\d+)*\\s+)?"
    r"(abstract|introduction|geological setting|geology|sampling|methods?|"
    r"palaeomagnetic methods?|paleomagnetic methods?|rock magneti(?:c|cs)|"
    r"magnetic mineralogy|results?|discussion|conclusions?|appendix|supplement)",
    re.I,
)


def clean(text: str | None) -> str:
    return re.sub(r"[ \\t]+", " ", text or "").strip()


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\(cid:\d+\)", " ", text)
    text = re.sub(
        r"/C(\d{2,3})",
        lambda match: chr(int(match.group(1))) if 32 <= int(match.group(1)) <= 126 else " ",
        text,
    )
    return text


def text_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text or "").strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def alpha_count(text: str) -> int:
    return sum(1 for char in text if char.isalpha())


def glyph_noise(text: str) -> int:
    return text.count("(cid:") + len(re.findall(r"/C\d{2,3}", text))


def parsed_pages(parsed_path: Path) -> list[dict[str, Any]]:
    text = normalize(parsed_path.read_text(encoding="utf-8", errors="ignore"))
    markers = list(re.finditer(r"(?m)^#{2,}\s*page[_ ](\d+)\b.*$", text))
    if not markers:
        return [{"page": None, "text": text, "source": "parsed_text"}] if text.strip() else []
    pages: list[dict[str, Any]] = []
    for index, marker in enumerate(markers):
        start = marker.end()
        end = markers[index + 1].start() if index + 1 < len(markers) else len(text)
        try:
            page_number = int(marker.group(1))
        except ValueError:
            page_number = index + 1
        page_text = text[start:end].strip()
        if page_text:
            pages.append({"page": page_number, "text": page_text, "source": "parsed_text"})
    return pages


def pdfplumber_pages(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        import pdfplumber
    except ModuleNotFoundError:
        return []
    pages: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for index, page in enumerate(pdf.pages, start=1):
                text, page_source = pdfplumber_page_text(page)
                if text.strip():
                    pages.append({"page": index, "text": text, "source": page_source})
    except Exception:
        return []
    return pages


def pdfplumber_page_text(page: Any) -> tuple[str, str]:
    """Extract a page in reading order, splitting clear two-column bodies."""

    normal = normalize(page.extract_text(x_tolerance=1, y_tolerance=3) or "")
    try:
        words = page.extract_words(
            x_tolerance=1,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=False,
        )
    except Exception:
        return normal, "pdfplumber"
    if not words or not page.width or not page.height:
        return normal, "pdfplumber"

    width = float(page.width)
    height = float(page.height)
    body = [
        word
        for word in words
        if height * 0.08 <= float(word.get("top", 0)) <= height * 0.92
    ]
    left = [word for word in body if float(word.get("x1", 0)) <= width * 0.52]
    right = [word for word in body if float(word.get("x0", width)) >= width * 0.48]
    center = [
        word
        for word in body
        if float(word.get("x0", 0)) < width * 0.48
        and float(word.get("x1", 0)) > width * 0.52
    ]
    two_columns = (
        len(left) >= 35
        and len(right) >= 35
        and len(center) <= max(12, int(len(body) * 0.08))
    )
    if not two_columns:
        return normal, "pdfplumber"

    split = width / 2
    try:
        left_text = page.crop((0, 0, split, height)).extract_text(
            x_tolerance=1,
            y_tolerance=3,
            layout=True,
        ) or ""
        right_text = page.crop((split, 0, width, height)).extract_text(
            x_tolerance=1,
            y_tolerance=3,
            layout=True,
        ) or ""
    except Exception:
        return normal, "pdfplumber"
    column_text = normalize(f"{left_text}\n{right_text}")
    if alpha_count(column_text) < alpha_count(normal) * 0.8:
        return normal, "pdfplumber"
    return column_text, "pdfplumber_columns"


def pypdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        return []
    pages: list[dict[str, Any]] = []
    try:
        for index, page in enumerate(PdfReader(str(pdf_path)).pages, start=1):
            text = normalize(page.extract_text() or "")
            if text.strip():
                pages.append({"page": index, "text": text, "source": "pypdf"})
    except Exception:
        return []
    return pages


def pdftotext_pages(pdf_path: Path) -> list[dict[str, Any]]:
    executable = shutil.which("pdftotext")
    if not executable:
        return []
    pages: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="r3-pdftotext-") as tmpdir:
        output_path = Path(tmpdir) / "out.txt"
        try:
            subprocess.run(
                [executable, "-layout", str(pdf_path), str(output_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=120,
            )
        except (OSError, subprocess.SubprocessError):
            return []
        text = normalize(output_path.read_text(encoding="utf-8", errors="ignore"))
    for index, page_text in enumerate(text.split("\f"), start=1):
        page_text = page_text.strip()
        if page_text:
            pages.append({"page": index, "text": page_text, "source": "pdftotext_layout"})
    return pages


def section_hints(page_text: str) -> list[str]:
    hints: list[str] = []
    for raw in page_text.splitlines()[:80]:
        line = clean(raw)
        if not line or len(line) > 120:
            continue
        if SECTION_HINT.search(line):
            hints.append(line[:120])
        if len(hints) >= 6:
            break
    return hints


def needs_pdfplumber(pages: list[dict[str, Any]]) -> bool:
    text = "\n".join(page["text"] for page in pages)
    if not text.strip():
        return True
    if len(text.strip()) < 2000:
        return True
    if glyph_noise(text) > 20:
        return True
    lower = text.casefold()
    return not any(word in lower for word in ["abstract", "introduction", "discussion", "conclusion", "magnetic"])


def needs_pdftotext(pages: list[dict[str, Any]]) -> bool:
    return needs_pdfplumber(pages)


def choose_pages(
    refno: str,
    parsed_dir: Path,
    pdf_dir: Path,
    *,
    pdfplumber_mode: str,
    pdftotext_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed_path = parsed_dir / f"{refno}.txt"
    pdf_path = pdf_dir / f"{refno}.pdf"
    candidates: list[tuple[str, list[dict[str, Any]]]] = []
    if parsed_path.exists() and parsed_path.stat().st_size > 0:
        candidates.append(("parsed_text", parsed_pages(parsed_path)))
    if pdf_path.exists():
        pypdf_candidate = pypdf_pages(pdf_path)
        candidates.append(("pypdf", pypdf_candidate))
        if pdftotext_mode == "always" or (pdftotext_mode == "fallback" and needs_pdftotext(pypdf_candidate)):
            candidates.append(("pdftotext_layout", pdftotext_pages(pdf_path)))
        if pdfplumber_mode == "always" or (pdfplumber_mode == "fallback" and needs_pdfplumber(pypdf_candidate)):
            pdfplumber_candidate = pdfplumber_pages(pdf_path)
            pdfplumber_name = (
                "pdfplumber_columns"
                if any(page.get("source") == "pdfplumber_columns" for page in pdfplumber_candidate)
                else "pdfplumber"
            )
            candidates.append((pdfplumber_name, pdfplumber_candidate))

    best_name = "missing"
    best_pages: list[dict[str, Any]] = []
    candidate_quality: list[tuple[str, list[dict[str, Any]], int, int, int]] = []
    for name, pages in candidates:
        text = "\n".join(page["text"] for page in pages)
        candidate_quality.append(
            (name, pages, alpha_count(text), glyph_noise(text), len(pages))
        )
    usable = [row for row in candidate_quality if row[1]]
    if usable:
        max_alpha = max(row[2] for row in usable)
        max_pages = max(row[4] for row in usable)
        near_complete = [
            row
            for row in usable
            if row[2] >= max_alpha * 0.88 and row[4] >= max(1, int(max_pages * 0.8))
        ]
        # Extractors can trade page count for text volume, leaving no candidate
        # in the intersection even though several candidates are usable.
        selection_pool = near_complete or usable
        source_priority = {
            "pdfplumber_columns": 5,
            "parsed_text": 4,
            "pdftotext_layout": 3,
            "pdfplumber": 2,
            "pypdf": 1,
        }
        best_name, best_pages, _, _, _ = max(
            selection_pool,
            key=lambda row: (
                source_priority.get(row[0], 0),
                -row[3],
                row[2],
                row[4],
            ),
        )

    flags: list[str] = []
    text = "\n".join(page["text"] for page in best_pages)
    if not parsed_path.exists() or parsed_path.stat().st_size == 0:
        flags.append("no_parsed_text")
    if not pdf_path.exists():
        flags.append("no_local_pdf")
    if not text.strip():
        flags.append("no_extractable_text")
    if len(text.strip()) < 2000:
        flags.append("short_text")
    if glyph_noise(text) > 20:
        flags.append("glyph_or_cid_text")
    lower = text.casefold()
    if text and not any(word in lower for word in ["abstract", "introduction", "discussion", "conclusion", "magnetic"]):
        flags.append("possibly_incomplete_extraction")

    normalized_pages = []
    page_hashes: list[str] = []
    for idx, page in enumerate(best_pages, start=1):
        digest = text_fingerprint(page["text"])
        page_hashes.append(digest)
        normalized_pages.append(
            {
                "page": page.get("page") or idx,
                "source": page.get("source") or best_name,
                "text": page["text"],
                "text_sha256": digest,
                "section_hints": section_hints(page["text"]),
            }
        )
    meta = {
        "refno": refno,
        "source_kind": best_name,
        "parsed_path": str(parsed_path) if parsed_path.exists() else None,
        "pdf_path": str(pdf_path) if pdf_path.exists() else None,
        "page_count": len(normalized_pages),
        "text_length": len(text),
        "cache_sha256": hashlib.sha256("".join(page_hashes).encode("ascii")).hexdigest(),
        "source_quality_flags": flags,
    }
    return normalized_pages, meta


def target_refnos(targets: Path) -> list[str]:
    frame = pd.read_excel(targets, usecols=["REFNO"], dtype={"REFNO": str})
    return list(dict.fromkeys(clean(value) for value in frame["REFNO"].tolist() if clean(value)))


def build_one(args: tuple[str, str, str, str, str]) -> dict[str, Any]:
    refno, parsed_dir, pdf_dir, pdfplumber_mode, pdftotext_mode = args
    pages, meta = choose_pages(
        refno,
        Path(parsed_dir),
        Path(pdf_dir),
        pdfplumber_mode=pdfplumber_mode,
        pdftotext_mode=pdftotext_mode,
    )
    return {"REFNO": refno, "source": meta, "pages": pages}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=ROOT / "data/targets.xlsx")
    parser.add_argument("--parsed-dir", type=Path, default=ROOT / "data/parsed_mds")
    parser.add_argument("--pdf-dir", type=Path, default=ROOT / "data/raw_pdfs")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--pdfplumber-mode",
        choices=["never", "fallback", "always"],
        default="fallback",
        help="Use pdfplumber never, only when pypdf/parsed text looks weak, or always.",
    )
    parser.add_argument(
        "--pdftotext-mode",
        choices=["never", "fallback", "always"],
        default="never",
        help="Optionally add poppler pdftotext -layout as a source candidate.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    refnos = target_refnos(args.targets)
    jobs = [
        (refno, str(args.parsed_dir), str(args.pdf_dir), args.pdfplumber_mode, args.pdftotext_mode)
        for refno in refnos
    ]
    if args.workers <= 1:
        records = [build_one(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(build_one, jobs))

    flag_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    total_pages = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            pages = record["pages"]
            meta = record["source"]
            total_pages += len(pages)
            source_counts[meta["source_kind"]] = source_counts.get(meta["source_kind"], 0) + 1
            for flag in meta["source_quality_flags"] or ["none"]:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "targets": str(args.targets),
        "unique_refnos": sum(source_counts.values()),
        "total_pages": total_pages,
        "source_kind_counts": dict(sorted(source_counts.items())),
        "source_quality_flag_counts": dict(sorted(flag_counts.items())),
        "output": str(args.output),
        "workers": args.workers,
        "pdfplumber_mode": args.pdfplumber_mode,
        "pdftotext_mode": args.pdftotext_mode,
        "pdftotext_available": bool(shutil.which("pdftotext")),
        "label_exclusion": "Only REFNO is read from the target workbook.",
    }
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
