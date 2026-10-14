#!/usr/bin/env python3
"""Build a label-free PDF layout sidecar without changing the R3 page cache."""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


SIDECAR_SCHEMA_VERSION = "r3_layout_sidecar_v1"
AUDIT_SCHEMA_VERSION = "r3_layout_sidecar_audit_v1"
SELECTION_REFNO_KEYS = ("selected_refnos", "refnos")
REFNO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
CAPTION_RE = re.compile(
    r"^\s*(?P<kind>fig(?:ure)?|table)\s*\.?\s*(?P<label>[A-Za-z]?\d+[A-Za-z]?)\b",
    re.I,
)
CROSSREF_RE = re.compile(
    r"\b(?P<kind>fig(?:ure)?|table)\s*\.?\s*(?P<label>[A-Za-z]?\d+[A-Za-z]?)\b",
    re.I,
)
FORBIDDEN_FIELD_RE = re.compile(
    r"(?:^|_)(?:pmagresult|rockmag|r3|labels?|predictions?|evaluations?|"
    r"metrics?|outcomes?)(?:$|_)",
    re.I,
)
NEGATIVE_BOUNDARY_SUFFIXES = ("_read", "_used", "_accessed", "_accepted")


class LayoutSidecarError(ValueError):
    """Raised when a sidecar input or extracted document violates the contract."""


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_sha256(value: Any) -> str:
    normalized = re.sub(r"\s+", " ", normalize_text(value)).strip()
    return hashlib.sha256(normalized.encode("utf-8", errors="ignore")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selection_sha256(refnos: Sequence[str]) -> str:
    payload = "".join(f"{refno}\n" for refno in sorted(refnos, key=_natural_key))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _natural_key(value: str) -> tuple[int, int | str, str]:
    return (0, int(value), value) if value.isdigit() else (1, value.casefold(), value)


def _boundary_declaration(key: str, value: Any) -> bool:
    folded = key.casefold()
    return (
        value is False
        and (
            folded.endswith(NEGATIVE_BOUNDARY_SUFFIXES)
            or folded.startswith("uses_")
        )
    ) or (
        value is True
        and folded in {"label_free", "labels_absent", "outcomes_absent"}
    )


def reject_supervised_fields(value: Any, path: str = "selection") -> None:
    """Reject answer-bearing fields while allowing explicit false boundary declarations."""

    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = clean(raw_key)
            child_path = f"{path}.{key}" if path else key
            if FORBIDDEN_FIELD_RE.search(key) and not _boundary_declaration(
                key, child
            ):
                raise LayoutSidecarError(
                    f"forbidden supervised field in sidecar input: {child_path}"
                )
            reject_supervised_fields(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_supervised_fields(child, f"{path}[{index}]")


def _manifest_refno_values(payload: Mapping[str, Any]) -> tuple[list[Any], list[Any]]:
    direct: list[Any] = []
    for key in SELECTION_REFNO_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            direct = value
            break
    selection = payload.get("selection")
    if not direct and isinstance(selection, Mapping):
        for key in SELECTION_REFNO_KEYS:
            value = selection.get(key)
            if isinstance(value, list):
                direct = value
                break
    papers = payload.get("papers")
    paper_rows = papers if isinstance(papers, list) else []
    return direct, paper_rows


def load_selection_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LayoutSidecarError(f"invalid selection manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise LayoutSidecarError("selection manifest must be a JSON object")
    reject_supervised_fields(payload)
    direct, paper_rows = _manifest_refno_values(payload)
    paper_refnos: list[Any] = []
    expected_pdf_hashes: dict[str, str] = {}
    for index, item in enumerate(paper_rows):
        if not isinstance(item, Mapping):
            raise LayoutSidecarError(f"selection papers[{index}] must be an object")
        refno = item.get("REFNO", item.get("refno"))
        paper_refnos.append(refno)
        pdf = item.get("pdf") if isinstance(item.get("pdf"), Mapping) else {}
        expected = clean(item.get("pdf_sha256") or pdf.get("sha256"))
        if expected:
            expected_pdf_hashes[clean(refno)] = expected.casefold()
    values = direct or paper_refnos
    if direct and paper_refnos:
        left = [clean(value) for value in direct]
        right = [clean(value) for value in paper_refnos]
        if set(left) != set(right):
            raise LayoutSidecarError(
                "selection selected_refnos and papers contain different REFNO sets"
            )
    if not values:
        raise LayoutSidecarError("selection manifest contains no selected REFNO values")
    refnos = [clean(value) for value in values]
    if any(not value or not REFNO_RE.fullmatch(value) for value in refnos):
        raise LayoutSidecarError("selection manifest contains an unsafe or empty REFNO")
    if len(refnos) != len(set(refnos)):
        raise LayoutSidecarError("selection manifest contains duplicate REFNO values")
    refnos = sorted(refnos, key=_natural_key)
    declared_hash = clean(payload.get("selection_sha256"))
    membership_hash = _selection_sha256(refnos)
    if clean(payload.get("schema_version")) == "r3_layout_pilot_manifest_v1":
        rows = payload.get("rows")
        rule = payload.get("selection_rule")
        if not isinstance(rows, list) or not isinstance(rule, Mapping):
            raise LayoutSidecarError("frozen pilot manifest lacks rows or selection_rule")
        frozen_pdfs = [
            {
                "REFNO": clean(item.get("REFNO", item.get("refno"))),
                "name": clean(
                    (item.get("pdf") or {}).get("name")
                    if isinstance(item.get("pdf"), Mapping)
                    else item.get("pdf_name")
                ),
                "sha256": clean(
                    (item.get("pdf") or {}).get("sha256")
                    if isinstance(item.get("pdf"), Mapping)
                    else item.get("pdf_sha256")
                ),
            }
            for item in paper_rows
        ]
        checks = {
            "selection_rule_sha256": _json_sha256(rule),
            "selected_rows_sha256": _json_sha256(rows),
            "selected_pdf_set_sha256": _json_sha256(frozen_pdfs),
        }
        for key, observed in checks.items():
            if clean(payload.get(key)).casefold() != observed:
                raise LayoutSidecarError(f"{key} does not match frozen manifest content")
        calculated_hash = _json_sha256(
            {"selection_rule": rule, "rows": rows, "pdfs": frozen_pdfs}
        )
        if not declared_hash or declared_hash.casefold() != calculated_hash:
            raise LayoutSidecarError(
                "selection_sha256 does not match frozen manifest content"
            )
    else:
        calculated_hash = membership_hash
        if declared_hash and declared_hash.casefold() != calculated_hash:
            raise LayoutSidecarError(
                "selection_sha256 does not match selected REFNO values"
            )
    return {
        "refnos": refnos,
        "selection_sha256": declared_hash.casefold() or calculated_hash,
        "refno_membership_sha256": membership_hash,
        "expected_pdf_hashes": expected_pdf_hashes,
        "schema_version": clean(payload.get("schema_version")),
        "status": clean(payload.get("status")),
    }


def load_page_cache(path: Path, wanted_refnos: set[str]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise LayoutSidecarError(f"could not open page cache {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LayoutSidecarError(
                    f"invalid page-cache JSON at line {line_number}: {exc}"
                ) from exc
            if not isinstance(raw, Mapping):
                raise LayoutSidecarError(
                    f"page-cache line {line_number} must be an object"
                )
            refno = clean(raw.get("REFNO"))
            if refno not in wanted_refnos:
                continue
            reject_supervised_fields(raw, f"page_cache[{refno}]")
            if refno in records:
                raise LayoutSidecarError(f"duplicate selected page-cache REFNO: {refno}")
            pages: list[dict[str, Any]] = []
            seen_pages: set[int] = set()
            for position, page in enumerate(raw.get("pages") or [], start=1):
                if not isinstance(page, Mapping):
                    continue
                number = page.get("page", position)
                if isinstance(number, bool):
                    number = position
                try:
                    number = int(number)
                except (TypeError, ValueError):
                    number = position
                if number < 1 or number in seen_pages:
                    raise LayoutSidecarError(
                        f"invalid or duplicate cache page for REFNO {refno}: {number}"
                    )
                seen_pages.add(number)
                text = normalize_text(page.get("text"))
                pages.append(
                    {
                        "page": number,
                        "text": text,
                        "text_sha256": clean(page.get("text_sha256"))
                        or text_sha256(text),
                    }
                )
            records[refno] = {"REFNO": refno, "pages": pages}
    missing = sorted(wanted_refnos - records.keys(), key=_natural_key)
    if missing:
        raise LayoutSidecarError(
            "page cache is missing selected REFNO values: " + ", ".join(missing[:20])
        )
    return records


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _valid_bbox(
    values: Optional[Sequence[Any]],
    width: float,
    height: float,
    *,
    tolerance: float = 2.0,
) -> tuple[Optional[list[float]], bool]:
    if not values or len(values) != 4:
        return None, False
    numbers = [_safe_float(value) for value in values]
    if any(value is None for value in numbers):
        return None, False
    x0, top, x1, bottom = (float(value) for value in numbers)
    if x1 <= x0 or bottom <= top:
        return None, False
    if x1 < -tolerance or bottom < -tolerance or x0 > width + tolerance or top > height + tolerance:
        return None, False
    bounded = [
        max(0.0, min(width, x0)),
        max(0.0, min(height, top)),
        max(0.0, min(width, x1)),
        max(0.0, min(height, bottom)),
    ]
    clamped = [round(value, 4) for value in bounded]
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        return None, False
    adjusted = any(
        abs(left - right) > 1e-6
        for left, right in zip(bounded, (x0, top, x1, bottom))
    )
    return clamped, adjusted


def _union_bbox(boxes: Iterable[Sequence[float]]) -> list[float]:
    values = list(boxes)
    return [
        round(min(box[0] for box in values), 4),
        round(min(box[1] for box in values), 4),
        round(max(box[2] for box in values), 4),
        round(max(box[3] for box in values), 4),
    ]


def extract_page_words(page: Any, page_number: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    width = float(page.width)
    height = float(page.height)
    kwargs = {
        "x_tolerance": 1,
        "y_tolerance": 3,
        "keep_blank_chars": False,
        "use_text_flow": False,
        "extra_attrs": ["fontname", "size"],
    }
    error = ""
    try:
        raw_words = page.extract_words(**kwargs)
    except TypeError:
        kwargs.pop("extra_attrs")
        try:
            raw_words = page.extract_words(**kwargs)
        except Exception as exc:  # pragma: no cover - exercised through generic failure test
            raw_words = []
            error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        raw_words = []
        error = f"{type(exc).__name__}: {exc}"
    words: list[dict[str, Any]] = []
    invalid = adjusted = 0
    for raw in raw_words or []:
        if not isinstance(raw, Mapping):
            invalid += 1
            continue
        text = normalize_text(raw.get("text"))
        bbox, was_adjusted = _valid_bbox(
            [raw.get("x0"), raw.get("top"), raw.get("x1"), raw.get("bottom")],
            width,
            height,
        )
        if not text or bbox is None:
            invalid += 1
            continue
        adjusted += int(was_adjusted)
        size = _safe_float(raw.get("size"))
        words.append(
            {
                "word_id": "",
                "text": text,
                "bbox": bbox,
                "fontname": clean(raw.get("fontname")) or None,
                "size": round(size, 4) if size is not None else None,
                "line_id": None,
                "column_id": None,
                "reading_order": None,
            }
        )
    words.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["bbox"][3]))
    for index, word in enumerate(words, start=1):
        word["word_id"] = f"p{page_number}-w{index}"
    return words, {
        "status": "degraded" if error else ("empty" if not words else "success"),
        "error": error or None,
        "raw_word_count": len(raw_words or []),
        "valid_word_count": len(words),
        "invalid_bbox_count": invalid,
        "adjusted_bbox_count": adjusted,
    }


def _split_vertical_group(words: Sequence[dict[str, Any]], width: float) -> list[list[dict[str, Any]]]:
    ordered = sorted(words, key=lambda item: item["bbox"][0])
    if len(ordered) < 2:
        return [ordered]
    threshold = max(36.0, width * 0.15)
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for word in ordered[1:]:
        gap = word["bbox"][0] - groups[-1][-1]["bbox"][2]
        if gap > threshold:
            groups.append([word])
        else:
            groups[-1].append(word)
    return groups


def build_lines_and_columns(
    words: Sequence[dict[str, Any]], page_number: int, width: float, height: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not words:
        return [], [], []
    heights = [word["bbox"][3] - word["bbox"][1] for word in words]
    y_tolerance = max(2.0, min(6.0, median(heights) * 0.45))
    vertical_groups: list[dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        center = (word["bbox"][1] + word["bbox"][3]) / 2.0
        candidate = next(
            (
                group
                for group in reversed(vertical_groups[-5:])
                if abs(center - group["center"]) <= y_tolerance
            ),
            None,
        )
        if candidate is None:
            vertical_groups.append({"center": center, "words": [word]})
        else:
            candidate["words"].append(word)
            candidate["center"] = sum(
                (item["bbox"][1] + item["bbox"][3]) / 2.0
                for item in candidate["words"]
            ) / len(candidate["words"])
    raw_lines: list[dict[str, Any]] = []
    for group in vertical_groups:
        for segment in _split_vertical_group(group["words"], width):
            if not segment:
                continue
            raw_lines.append(
                {
                    "text": " ".join(item["text"] for item in segment),
                    "bbox": _union_bbox(item["bbox"] for item in segment),
                    "word_ids": [item["word_id"] for item in segment],
                    "column_id": 1,
                }
            )
    raw_lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    body = [
        line
        for line in raw_lines
        if line["bbox"][1] >= height * 0.05 and line["bbox"][3] <= height * 0.95
    ]
    left = [line for line in body if line["bbox"][2] <= width * 0.58]
    right = [line for line in body if line["bbox"][0] >= width * 0.42]
    two_columns = len(left) >= 2 and len(right) >= 2 and len(raw_lines) >= 5
    if two_columns:
        for line in raw_lines:
            x0, _, x1, _ = line["bbox"]
            if x1 <= width * 0.58:
                line["column_id"] = 1
            elif x0 >= width * 0.42:
                line["column_id"] = 2
            else:
                line["column_id"] = 0
        column_lines = [line for line in raw_lines if line["column_id"] in {1, 2}]
        first_column_top = min((line["bbox"][1] for line in column_lines), default=height)
        headers = [
            line
            for line in raw_lines
            if line["column_id"] == 0 and line["bbox"][3] <= first_column_top + 1.0
        ]
        remaining_spans = [
            line for line in raw_lines if line["column_id"] == 0 and line not in headers
        ]
        ordered_lines = (
            sorted(headers, key=lambda item: (item["bbox"][1], item["bbox"][0]))
            + sorted(
                (line for line in raw_lines if line["column_id"] == 1),
                key=lambda item: (item["bbox"][1], item["bbox"][0]),
            )
            + sorted(
                (line for line in raw_lines if line["column_id"] == 2),
                key=lambda item: (item["bbox"][1], item["bbox"][0]),
            )
            + sorted(remaining_spans, key=lambda item: (item["bbox"][1], item["bbox"][0]))
        )
    else:
        ordered_lines = raw_lines
        for line in ordered_lines:
            line["column_id"] = 1
    word_by_id = {word["word_id"]: word for word in words}
    reading_words: list[dict[str, Any]] = []
    for line_index, line in enumerate(ordered_lines, start=1):
        line_id = f"p{page_number}-l{line_index}"
        line["line_id"] = line_id
        line["reading_order"] = line_index
        for word_id in line["word_ids"]:
            word = word_by_id[word_id]
            word["line_id"] = line_id
            word["column_id"] = line["column_id"]
            word["reading_order"] = len(reading_words) + 1
            reading_words.append(word)
    columns: list[dict[str, Any]] = []
    for column_id in sorted({line["column_id"] for line in ordered_lines}):
        selected = [line for line in ordered_lines if line["column_id"] == column_id]
        columns.append(
            {
                "column_id": column_id,
                "kind": "spanning" if column_id == 0 else "text",
                "bbox": _union_bbox(line["bbox"] for line in selected),
                "line_ids": [line["line_id"] for line in selected],
            }
        )
    return reading_words, ordered_lines, columns


def build_blocks(
    lines: Sequence[Mapping[str, Any]], page_number: int
) -> list[dict[str, Any]]:
    """Group adjacent same-column lines into conservative paragraph blocks."""

    if not lines:
        return []
    heights = [
        float(line["bbox"][3]) - float(line["bbox"][1])
        for line in lines
        if len(line.get("bbox") or []) == 4
    ]
    typical_height = median(heights) if heights else 10.0
    maximum_gap = max(4.0, typical_height * 0.70)
    groups: list[list[Mapping[str, Any]]] = []
    for line in sorted(lines, key=lambda item: int(item.get("reading_order") or 0)):
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1][-1]
        same_column = int(line.get("column_id") or 0) == int(
            previous.get("column_id") or 0
        )
        gap = float(line["bbox"][1]) - float(previous["bbox"][3])
        monotonic = float(line["bbox"][1]) >= float(previous["bbox"][1])
        if same_column and monotonic and gap <= maximum_gap:
            groups[-1].append(line)
        else:
            groups.append([line])
    blocks: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        block_id = f"p{page_number}-b{index}"
        block_lines = [
            {
                "line_id": clean(line.get("line_id")),
                "text": clean(line.get("text")),
                "bbox": list(line.get("bbox") or []),
                "column_id": int(line.get("column_id") or 0),
                "reading_order": int(line.get("reading_order") or 0),
                "layout_confidence": 1.0,
            }
            for line in group
        ]
        blocks.append(
            {
                "block_id": block_id,
                "kind": "paragraph",
                "column_id": int(group[0].get("column_id") or 0),
                "reading_order": min(
                    int(line.get("reading_order") or 0) for line in group
                ),
                "bbox": _union_bbox(line["bbox"] for line in group),
                "text": "\n".join(clean(line.get("text")) for line in group),
                "lines": block_lines,
                "layout_confidence": 1.0,
            }
        )
    return blocks


def _bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    x0 = max(left[0], right[0])
    top = max(left[1], right[1])
    x1 = min(left[2], right[2])
    bottom = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, bottom - top)
    if not intersection:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1e-9, left_area + right_area - intersection)


def _table_caption_cue(lines: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        (match := CAPTION_RE.search(clean(line.get("text"))))
        and match.group("kind").casefold().startswith("table")
        for line in lines
    )


def extract_native_tables(
    page: Any,
    page_number: int,
    lines: Sequence[Mapping[str, Any]],
    width: float,
    height: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    strategies: list[tuple[str, Optional[Mapping[str, Any]]]] = [("default", None)]
    if _table_caption_cue(lines):
        strategies.append(
            (
                "text",
                {
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3,
                },
            )
        )
    detected: list[tuple[str, Any, list[float]]] = []
    errors: list[dict[str, str]] = []
    adjusted = invalid = 0
    for strategy, settings in strategies:
        if strategy == "text" and detected:
            continue
        try:
            candidates = page.find_tables(table_settings=settings)
        except Exception as exc:
            errors.append(
                {
                    "strategy": strategy,
                    "error_type": type(exc).__name__,
                    "message": clean(exc)[:240],
                }
            )
            continue
        for candidate in candidates or []:
            bbox, was_adjusted = _valid_bbox(
                getattr(candidate, "bbox", None), width, height
            )
            if bbox is None:
                invalid += 1
                continue
            adjusted += int(was_adjusted)
            if any(_bbox_iou(bbox, item[2]) >= 0.90 for item in detected):
                continue
            detected.append((strategy, candidate, bbox))
    tables: list[dict[str, Any]] = []
    for table_index, (strategy, candidate, bbox) in enumerate(detected, start=1):
        table_id = f"p{page_number}-t{table_index}"
        matrix: list[list[Any]] = []
        try:
            extracted = candidate.extract()
            if isinstance(extracted, list):
                matrix = [list(row) if isinstance(row, list) else [] for row in extracted]
        except Exception as exc:
            errors.append(
                {
                    "strategy": f"{strategy}:extract",
                    "error_type": type(exc).__name__,
                    "message": clean(exc)[:240],
                }
            )
        cells: list[dict[str, Any]] = []
        rows = list(getattr(candidate, "rows", None) or [])
        for row_index, row in enumerate(rows):
            row_cells = list(getattr(row, "cells", None) or [])
            for column_index, raw_bbox in enumerate(row_cells):
                cell_bbox, was_adjusted = _valid_bbox(raw_bbox, width, height)
                if cell_bbox is None:
                    if raw_bbox is not None:
                        invalid += 1
                    continue
                adjusted += int(was_adjusted)
                value = ""
                if row_index < len(matrix) and column_index < len(matrix[row_index]):
                    value = normalize_text(matrix[row_index][column_index])
                cells.append(
                    {
                        "cell_id": f"{table_id}-r{row_index + 1}c{column_index + 1}",
                        "row": row_index + 1,
                        "column": column_index + 1,
                        "bbox": cell_bbox,
                        "text": value,
                    }
                )
        tables.append(
            {
                "table_id": table_id,
                "bbox": bbox,
                "detection_strategy": strategy,
                "row_count": max((cell["row"] for cell in cells), default=len(matrix)),
                "column_count": max(
                    (cell["column"] for cell in cells),
                    default=max((len(row) for row in matrix), default=0),
                ),
                "cells": cells,
                "rows": [
                    {
                        "row_id": f"{table_id}-r{row_number}",
                        "cells": [
                            cell for cell in cells if cell["row"] == row_number
                        ],
                        "layout_confidence": 1.0,
                    }
                    for row_number in sorted({cell["row"] for cell in cells})
                ],
                "layout_confidence": 1.0,
            }
        )
    return tables, {
        "status": "degraded" if errors else ("success" if tables else "no_tables"),
        "errors": errors,
        "invalid_bbox_count": invalid,
        "adjusted_bbox_count": adjusted,
    }


def _canonical_reference_kind(value: str) -> str:
    return "figure" if value.casefold().startswith("fig") else "table"


def _object_distance(caption_bbox: Sequence[float], object_bbox: Sequence[float]) -> float:
    if caption_bbox[3] <= object_bbox[1]:
        vertical = object_bbox[1] - caption_bbox[3]
    elif object_bbox[3] <= caption_bbox[1]:
        vertical = caption_bbox[1] - object_bbox[3]
    else:
        vertical = 0.0
    overlap = max(
        0.0,
        min(caption_bbox[2], object_bbox[2]) - max(caption_bbox[0], object_bbox[0]),
    )
    horizontal_penalty = 0.0 if overlap else abs(
        (caption_bbox[0] + caption_bbox[2]) / 2.0
        - (object_bbox[0] + object_bbox[2]) / 2.0
    )
    return vertical + horizontal_penalty * 0.25


def detect_captions_and_crossrefs(
    lines: Sequence[Mapping[str, Any]],
    tables: Sequence[Mapping[str, Any]],
    page_number: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    captions: list[dict[str, Any]] = []
    caption_line_ids: set[str] = set()
    for line in lines:
        text = clean(line.get("text"))
        match = CAPTION_RE.search(text)
        if not match:
            continue
        kind = _canonical_reference_kind(match.group("kind"))
        caption_id = f"p{page_number}-cap{len(captions) + 1}"
        linked_table_id = None
        if kind == "table" and tables:
            linked = min(
                tables,
                key=lambda item: _object_distance(line["bbox"], item["bbox"]),
            )
            linked_table_id = linked["table_id"]
        captions.append(
            {
                "caption_id": caption_id,
                "kind": kind,
                "label": match.group("label").casefold(),
                "text": text,
                "bbox": list(line["bbox"]),
                "line_ids": [line["line_id"]],
                "linked_table_id": linked_table_id,
                "layout_confidence": 1.0,
            }
        )
        caption_line_ids.add(str(line["line_id"]))
    crossrefs: list[dict[str, Any]] = []
    for line in lines:
        if str(line["line_id"]) in caption_line_ids:
            continue
        for match in CROSSREF_RE.finditer(clean(line.get("text"))):
            crossrefs.append(
                {
                    "crossref_id": f"p{page_number}-xref{len(crossrefs) + 1}",
                    "kind": _canonical_reference_kind(match.group("kind")),
                    "label": match.group("label").casefold(),
                    "text": clean(line.get("text")),
                    "bbox": list(line["bbox"]),
                    "line_id": line["line_id"],
                    "linked_caption_ids": [],
                    "linked_table_ids": [],
                }
            )
    return captions, crossrefs


def link_document_crossrefs(pages: Sequence[dict[str, Any]]) -> None:
    captions: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for page in pages:
        for caption in page["captions"]:
            captions.setdefault((caption["kind"], caption["label"]), []).append(caption)
    for page in pages:
        for crossref in page["crossrefs"]:
            matches = captions.get((crossref["kind"], crossref["label"]), [])
            crossref["linked_caption_ids"] = [item["caption_id"] for item in matches]
            crossref["linked_table_ids"] = sorted(
                {
                    item["linked_table_id"]
                    for item in matches
                    if item.get("linked_table_id")
                }
            )


def _tokens(value: str) -> Counter[str]:
    return Counter(TOKEN_RE.findall(normalize_text(value).casefold()))


def _token_alignment(cache_text: str, layout_text: str) -> dict[str, float]:
    cache = _tokens(cache_text)
    layout = _tokens(layout_text)
    cache_total = sum(cache.values())
    layout_total = sum(layout.values())
    overlap = sum((cache & layout).values())
    recall = overlap / cache_total if cache_total else float(not layout_total)
    precision = overlap / layout_total if layout_total else float(not cache_total)
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    union = sum((cache | layout).values())
    return {
        "token_recall": round(recall, 6),
        "token_precision": round(precision, 6),
        "token_f1": round(f1, 6),
        "token_jaccard": round(overlap / union if union else 1.0, 6),
    }


def align_pages_to_cache(
    pages: Sequence[dict[str, Any]], cache_pages: Sequence[Mapping[str, Any]]
) -> None:
    candidates: list[tuple[float, float, int, int, dict[str, float]]] = []
    for physical_index, page in enumerate(pages):
        layout_text = "\n".join(line["text"] for line in page["lines"])
        for cache_index, cache_page in enumerate(cache_pages):
            metrics = _token_alignment(clean(cache_page.get("text")), layout_text)
            same_page = int(page["page"] == int(cache_page.get("page") or -1))
            candidates.append(
                (
                    metrics["token_f1"] + 0.03 * same_page,
                    metrics["token_f1"],
                    physical_index,
                    cache_index,
                    metrics,
                )
            )
    assigned_physical: set[int] = set()
    assigned_cache: set[int] = set()
    mapping: dict[int, tuple[int, dict[str, float]]] = {}
    for _, score, physical_index, cache_index, metrics in sorted(
        candidates, key=lambda item: (-item[0], -item[1], item[2], item[3])
    ):
        if physical_index in assigned_physical or cache_index in assigned_cache:
            continue
        if score < 0.05:
            continue
        assigned_physical.add(physical_index)
        assigned_cache.add(cache_index)
        mapping[physical_index] = (cache_index, metrics)
    for physical_index, page in enumerate(pages):
        if physical_index not in mapping:
            page["cache_alignment"] = {
                "status": "unaligned",
                "cache_page": None,
                "cache_text_sha256": None,
                "token_recall": 0.0,
                "token_precision": 0.0,
                "token_f1": 0.0,
                "token_jaccard": 0.0,
                "method": "none",
            }
            continue
        cache_index, metrics = mapping[physical_index]
        cache_page = cache_pages[cache_index]
        cache_number = int(cache_page.get("page") or cache_index + 1)
        page["cache_alignment"] = {
            "status": "aligned",
            "cache_page": cache_number,
            "cache_text_sha256": clean(cache_page.get("text_sha256"))
            or text_sha256(cache_page.get("text")),
            **metrics,
            "method": "same_page" if cache_number == page["page"] else "token_overlap",
        }


def _default_pdf_opener(path: Path) -> Any:
    try:
        import pdfplumber
    except ModuleNotFoundError as exc:  # pragma: no cover - environment preflight
        raise LayoutSidecarError("pdfplumber is required for layout extraction") from exc
    return pdfplumber.open(str(path))


def extract_document_layout(
    refno: str,
    pdf_path: Path,
    cache_record: Mapping[str, Any],
    *,
    pdf_opener: Optional[Callable[[Path], Any]] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    opener = pdf_opener or _default_pdf_opener
    try:
        context = opener(pdf_path)
    except Exception as exc:
        raise LayoutSidecarError(f"could not open PDF for REFNO {refno}: {exc}") from exc
    pages: list[dict[str, Any]] = []
    table_errors: list[dict[str, Any]] = []
    word_errors: list[dict[str, Any]] = []
    invalid_bbox_count = adjusted_bbox_count = 0
    manager = context if hasattr(context, "__enter__") else nullcontext(context)
    try:
        with manager as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                width = _safe_float(getattr(page, "width", None))
                height = _safe_float(getattr(page, "height", None))
                if width is None or height is None or width <= 0 or height <= 0:
                    raise LayoutSidecarError(
                        f"invalid page dimensions for REFNO {refno} page {page_number}"
                    )
                words, word_audit = extract_page_words(page, page_number)
                words, lines, columns = build_lines_and_columns(
                    words, page_number, width, height
                )
                blocks = build_blocks(lines, page_number)
                tables, table_audit = extract_native_tables(
                    page, page_number, lines, width, height
                )
                captions, crossrefs = detect_captions_and_crossrefs(
                    lines, tables, page_number
                )
                invalid_bbox_count += int(word_audit["invalid_bbox_count"])
                invalid_bbox_count += int(table_audit["invalid_bbox_count"])
                adjusted_bbox_count += int(word_audit["adjusted_bbox_count"])
                adjusted_bbox_count += int(table_audit["adjusted_bbox_count"])
                if word_audit.get("error"):
                    word_errors.append(
                        {
                            "page": page_number,
                            "message": word_audit["error"],
                        }
                    )
                for error in table_audit["errors"]:
                    table_errors.append({"page": page_number, **error})
                rotation = getattr(page, "rotation", 0)
                rotation = int(rotation) if isinstance(rotation, (int, float)) else 0
                pages.append(
                    {
                        "page": page_number,
                        "width": round(width, 4),
                        "height": round(height, 4),
                        "rotation": rotation,
                        "words": words,
                        "lines": lines,
                        "blocks": blocks,
                        "columns": columns,
                        "tables": tables,
                        "captions": captions,
                        "crossrefs": crossrefs,
                        "cache_alignment": {},
                        "layout_confidence": 1.0 if words else 0.0,
                        "quality": {
                            "word_extraction_status": word_audit["status"],
                            "table_detection_status": table_audit["status"],
                            "table_detection_error_count": len(table_audit["errors"]),
                        },
                    }
                )
    except LayoutSidecarError:
        raise
    except Exception as exc:
        raise LayoutSidecarError(f"PDF extraction failed for REFNO {refno}: {exc}") from exc
    link_document_crossrefs(pages)
    align_pages_to_cache(pages, cache_record.get("pages") or [])
    pdf_hash = file_sha256(pdf_path)
    record = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "REFNO": refno,
        "pdf": {
            "path": str(pdf_path),
            "sha256": pdf_hash,
            "page_count": len(pages),
        },
        "page_cache": {
            "page_count": len(cache_record.get("pages") or []),
        },
        "pages": pages,
    }
    alignment_scores = [page["cache_alignment"]["token_f1"] for page in pages]
    document_audit = {
        "REFNO": refno,
        "pdf_sha256": pdf_hash,
        "physical_pages": len(pages),
        "cache_pages": len(cache_record.get("pages") or []),
        "words": sum(len(page["words"]) for page in pages),
        "lines": sum(len(page["lines"]) for page in pages),
        "blocks": sum(len(page["blocks"]) for page in pages),
        "tables": sum(len(page["tables"]) for page in pages),
        "cells": sum(
            len(table["cells"]) for page in pages for table in page["tables"]
        ),
        "captions": sum(len(page["captions"]) for page in pages),
        "crossrefs": sum(len(page["crossrefs"]) for page in pages),
        "pages_without_words": sum(not page["words"] for page in pages),
        "table_detection_error_count": len(table_errors),
        "table_detection_errors": table_errors,
        "word_extraction_error_count": len(word_errors),
        "word_extraction_errors": word_errors,
        "invalid_bbox_count": invalid_bbox_count,
        "adjusted_bbox_count": adjusted_bbox_count,
        "aligned_pages": sum(
            page["cache_alignment"]["status"] == "aligned" for page in pages
        ),
        "cache_alignment_token_f1_median": round(
            median(alignment_scores) if alignment_scores else 0.0, 6
        ),
        "cache_alignment_token_f1_minimum": round(
            min(alignment_scores) if alignment_scores else 0.0, 6
        ),
    }
    return record, document_audit


def _find_pdf(pdf_dir: Path, refno: str) -> Path:
    direct = pdf_dir / f"{refno}.pdf"
    if direct.is_file():
        return direct.resolve()
    matches = [
        path
        for path in pdf_dir.iterdir()
        if path.is_file()
        and path.suffix.casefold() == ".pdf"
        and path.stem.casefold() == refno.casefold()
    ]
    if len(matches) != 1:
        raise LayoutSidecarError(
            f"expected exactly one local PDF for REFNO {refno}, found {len(matches)}"
        )
    return matches[0].resolve()


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                )
                handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_layout_sidecar(
    selection_manifest: Path,
    pdf_dir: Path,
    page_cache: Path,
    output: Path,
    audit_json: Path,
    *,
    pdf_opener: Optional[Callable[[Path], Any]] = None,
) -> dict[str, Any]:
    selection_manifest = selection_manifest.expanduser().resolve()
    pdf_dir = pdf_dir.expanduser().resolve()
    page_cache = page_cache.expanduser().resolve()
    output = output.expanduser().resolve(strict=False)
    audit_json = audit_json.expanduser().resolve(strict=False)
    if not selection_manifest.is_file() or not page_cache.is_file() or not pdf_dir.is_dir():
        raise LayoutSidecarError(
            "selection manifest, PDF directory, and page cache must exist"
        )
    protected = {selection_manifest, page_cache, pdf_dir}
    if output in protected or audit_json in protected or output == audit_json:
        raise LayoutSidecarError("sidecar inputs and outputs must be distinct")
    if output.suffix.casefold() != ".jsonl" or audit_json.suffix.casefold() != ".json":
        raise LayoutSidecarError("sidecar output must be JSONL and audit output must be JSON")
    selection = load_selection_manifest(selection_manifest)
    refnos = selection["refnos"]
    cache_records = load_page_cache(page_cache, set(refnos))
    records: list[dict[str, Any]] = []
    documents: list[dict[str, Any]] = []
    for refno in refnos:
        pdf_path = _find_pdf(pdf_dir, refno)
        expected = selection["expected_pdf_hashes"].get(refno)
        actual_hash = file_sha256(pdf_path)
        if expected and expected != actual_hash:
            raise LayoutSidecarError(f"PDF SHA256 mismatch for REFNO {refno}")
        record, document_audit = extract_document_layout(
            refno,
            pdf_path,
            cache_records[refno],
            pdf_opener=pdf_opener,
        )
        records.append(record)
        documents.append(document_audit)
    _atomic_write_jsonl(output, records)
    table_errors = sum(item["table_detection_error_count"] for item in documents)
    word_errors = sum(item["word_extraction_error_count"] for item in documents)
    all_alignment_scores = [
        page["cache_alignment"]["token_f1"]
        for record in records
        for page in record["pages"]
    ]
    status = "PASS_WITH_DEGRADATION" if table_errors or word_errors else "PASS"
    audit: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": status,
        "runtime_boundary": {
            "identity_selection_only": True,
            "original_pdf_and_page_cache_only": True,
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_read": False,
            "evaluation_results_read": False,
        },
        "inputs": {
            "selection_manifest": {
                "path": str(selection_manifest),
                "sha256": file_sha256(selection_manifest),
                "selection_sha256": selection["selection_sha256"],
                "selected_refnos": len(refnos),
            },
            "pdf_dir": str(pdf_dir),
            "page_cache": {
                "path": str(page_cache),
                "sha256": file_sha256(page_cache),
            },
        },
        "counts": {
            "documents": len(records),
            "physical_pages": sum(item["physical_pages"] for item in documents),
            "cache_pages": sum(item["cache_pages"] for item in documents),
            "words": sum(item["words"] for item in documents),
            "lines": sum(item["lines"] for item in documents),
            "blocks": sum(item["blocks"] for item in documents),
            "tables": sum(item["tables"] for item in documents),
            "cells": sum(item["cells"] for item in documents),
            "captions": sum(item["captions"] for item in documents),
            "crossrefs": sum(item["crossrefs"] for item in documents),
            "pages_without_words": sum(item["pages_without_words"] for item in documents),
            "table_detection_errors": table_errors,
            "word_extraction_errors": word_errors,
            "invalid_bboxes_rejected": sum(item["invalid_bbox_count"] for item in documents),
            "bboxes_clamped": sum(item["adjusted_bbox_count"] for item in documents),
        },
        "cache_alignment": {
            "aligned_pages": sum(item["aligned_pages"] for item in documents),
            "token_f1_minimum": round(min(all_alignment_scores), 6)
            if all_alignment_scores
            else 0.0,
            "token_f1_median": round(median(all_alignment_scores), 6)
            if all_alignment_scores
            else 0.0,
        },
        "table_detection_degrades_per_page": True,
        "documents": documents,
        "output": {
            "path": str(output),
            "sha256": file_sha256(output),
            "records": len(records),
        },
    }
    _atomic_write_json(audit_json, audit)
    return audit


__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "LayoutSidecarError",
    "SIDECAR_SCHEMA_VERSION",
    "align_pages_to_cache",
    "build_layout_sidecar",
    "build_lines_and_columns",
    "detect_captions_and_crossrefs",
    "extract_document_layout",
    "extract_native_tables",
    "extract_page_words",
    "load_page_cache",
    "load_selection_manifest",
    "reject_supervised_fields",
]
