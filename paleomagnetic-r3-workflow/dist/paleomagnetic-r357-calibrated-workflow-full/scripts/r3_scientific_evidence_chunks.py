#!/usr/bin/env python3
"""Label-free multi-scale scientific evidence chunks for R3 retrieval."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Mapping, Pattern, Sequence

try:
    from .experiment_r3_own_study_events import classify_heading
    from .paper_text_retrieval import _reference_cutoffs, _sentence_spans
except ImportError:
    from experiment_r3_own_study_events import classify_heading
    from paper_text_retrieval import _reference_cutoffs, _sentence_spans


SCALES = ("sentence_window", "paragraph", "page_window", "section_window")
CATEGORIES = ("method", "diagnostic", "carrier", "bridge")
_SPACE_RE = re.compile(r"\s+")
_PARAGRAPH_BREAK_RE = re.compile(r"\n\s*\n+")
_WORD_RE = re.compile(r"\S+")
_OWN_STUDY_RE = re.compile(
    r"\b(?:we|our|this\s+study|this\s+work|samples?\s+(?:were|was)|"
    r"specimens?\s+(?:were|was)|measurements?\s+(?:were|was)|"
    r"(?:was|were)\s+(?:measured|analysed|analyzed|examined|heated|cooled))\b",
    re.I,
)
_RESULT_RE = re.compile(
    r"\b(?:show(?:s|ed)?|indicat(?:e|es|ed)|suggest(?:s|ed)?|reveal(?:s|ed)?|"
    r"demonstrat(?:e|es|ed)|identify|identified|consistent\s+with|"
    r"attribut(?:e|ed)|carried\s+by|due\s+to|saturat|Curie|coerciv|"
    r"transition|domain\s+state)\w*",
    re.I,
)
_CITATION_RE = re.compile(
    r"\b(?:19|20)\d{2}[a-z]?\b|\bet\s+al\.?\b|\b(?:doi|references?)\b",
    re.I,
)
_ADVERSE_RE = re.compile(
    r"\b(?:not|no|without|future|should\s+be|could\s+be|would\s+be|"
    r"recommended|remains?\s+to\s+be)\b",
    re.I,
)


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    scale: str
    page: int
    section: str
    text: str
    text_sha256: str
    word_count: int
    lexical_counts: tuple[int, int, int, int]
    own_study: bool
    result_language: bool
    citation_language: bool
    adverse_language: bool

    def lexical(self, category: str) -> int:
        return self.lexical_counts[CATEGORIES.index(category)]


def _clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _compact_text(value: str) -> str:
    return _SPACE_RE.sub(" ", value).strip()


def _safe_page(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def compile_category_patterns(
    concepts: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Pattern[str], ...]]:
    output: dict[str, list[Pattern[str]]] = {category: [] for category in CATEGORIES}
    for index, concept in enumerate(concepts):
        category = _clean(concept.get("category")).casefold()
        if category not in output:
            raise ValueError(f"unsupported concept category at index {index}: {category}")
        expressions = concept.get("search_patterns") or ()
        if isinstance(expressions, (str, bytes)) or not isinstance(expressions, Sequence):
            raise TypeError("concept search_patterns must be a sequence")
        for expression in expressions:
            if _clean(expression):
                output[category].append(re.compile(_clean(expression), re.I))
    if any(not values for values in output.values()):
        missing = sorted(category for category, values in output.items() if not values)
        raise ValueError(f"concept snapshot does not cover categories: {missing}")
    return {key: tuple(values) for key, values in output.items()}


def _section_name(value: str, fallback: str) -> str:
    detected = classify_heading(value)
    if detected and detected != "references":
        return detected
    return fallback


def _split_long_words(text: str, size: int, stride: int) -> list[str]:
    words = _WORD_RE.findall(text)
    if not words:
        return []
    if len(words) <= size:
        return [" ".join(words)]
    output = []
    for start in range(0, len(words), stride):
        current = words[start : start + size]
        if len(current) < max(24, size // 5) and output:
            break
        output.append(" ".join(current))
        if start + size >= len(words):
            break
    return output


def _split_section_words(
    parts: Sequence[tuple[int, str]], size: int, stride: int
) -> list[tuple[int, str]]:
    words = [
        (page, word)
        for page, text in parts
        for word in _WORD_RE.findall(text)
    ]
    if not words:
        return []
    if len(words) <= size:
        return [(words[len(words) // 2][0], " ".join(word for _, word in words))]
    output = []
    for start in range(0, len(words), stride):
        current = words[start : start + size]
        if len(current) < max(24, size // 5) and output:
            break
        page = current[len(current) // 2][0]
        output.append((page, " ".join(word for _, word in current)))
        if start + size >= len(words):
            break
    return output


def _page_segments(
    page: Mapping[str, Any],
    body: str,
    current_section: str,
) -> tuple[list[tuple[str, str]], str]:
    segments: list[tuple[str, str]] = []
    section = current_section
    start = 0
    for line in re.finditer(r"[^\n]*(?:\n|$)", body):
        if line.start() == line.end():
            continue
        detected = classify_heading(line.group(0).strip())
        if not detected:
            continue
        prior = body[start : line.start()]
        if prior.strip():
            segments.append((section, prior))
        if detected == "references":
            return segments, "references"
        section = detected
        start = line.end()
    tail = body[start:]
    if tail.strip():
        segments.append((section, tail))
    return segments, section


def _chunk(
    *,
    refno: str,
    scale: str,
    page: int,
    section: str,
    ordinal: int,
    text: str,
    patterns: Mapping[str, Sequence[Pattern[str]]],
) -> EvidenceChunk | None:
    compact = _compact_text(text)
    words = _WORD_RE.findall(compact)
    if len(words) < 8:
        return None
    lexical = tuple(
        min(99, sum(1 for pattern in patterns[category] if pattern.search(compact)))
        for category in CATEGORIES
    )
    digest = hashlib.sha256(compact.encode("utf-8")).hexdigest()
    identity = f"{refno}\0{scale}\0{page}\0{section}\0{ordinal}\0{digest}"
    return EvidenceChunk(
        chunk_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        scale=scale,
        page=page,
        section=section or "unknown",
        text=compact,
        text_sha256=digest,
        word_count=len(words),
        lexical_counts=lexical,
        own_study=bool(_OWN_STUDY_RE.search(compact)),
        result_language=bool(_RESULT_RE.search(compact)),
        citation_language=bool(_CITATION_RE.search(compact)),
        adverse_language=bool(_ADVERSE_RE.search(compact)),
    )


def _even_indices(length: int, count: int) -> list[int]:
    if count >= length:
        return list(range(length))
    if count <= 1:
        return [0]
    return sorted({round(offset * (length - 1) / (count - 1)) for offset in range(count)})


def _cap_scale(chunks: Sequence[EvidenceChunk], quota: int) -> list[EvidenceChunk]:
    if len(chunks) <= quota:
        return list(chunks)
    high_quota = max(1, quota // 2)
    ranked = sorted(
        range(len(chunks)),
        key=lambda index: (
            -sum(chunks[index].lexical_counts),
            -int(chunks[index].own_study),
            -int(chunks[index].result_language),
            chunks[index].page,
            chunks[index].chunk_id,
        ),
    )
    selected = set(ranked[:high_quota])
    remaining = [index for index in range(len(chunks)) if index not in selected]
    for position in _even_indices(len(remaining), quota - len(selected)):
        selected.add(remaining[position])
    return [chunks[index] for index in sorted(selected)]


def build_scientific_evidence_chunks(
    refno: str,
    pages: Sequence[Mapping[str, Any]],
    patterns: Mapping[str, Sequence[Pattern[str]]],
    config: Mapping[str, Any],
) -> tuple[list[EvidenceChunk], dict[str, Any]]:
    """Build bounded broad-retrieval chunks without labels or row outcomes."""

    scale_quotas = config.get("scale_quotas") or {}
    if tuple(config.get("scales") or ()) != SCALES:
        raise ValueError("retrieval chunk scales changed")
    if set(scale_quotas) != set(SCALES):
        raise ValueError("retrieval scale quotas are incomplete")
    raw_pages = [page for page in pages if isinstance(page, Mapping)]
    cutoffs = _reference_cutoffs(raw_pages)
    by_scale: dict[str, list[EvidenceChunk]] = defaultdict(list)
    section_parts: dict[str, list[tuple[int, str]]] = defaultdict(list)
    current_section = "abstract"
    reference_characters = 0
    reference_pages = 0
    body_characters = 0
    ordinal = 0
    for page_offset, (page, cutoff) in enumerate(zip(raw_pages, cutoffs), start=1):
        raw_text = str(page.get("text") or "")
        if cutoff == 0:
            reference_pages += 1
            reference_characters += len(raw_text)
            continue
        body = raw_text if cutoff is None else raw_text[:cutoff]
        reference_characters += len(raw_text) - len(body)
        reference_pages += int(cutoff is not None)
        body_characters += len(body)
        page_number = _safe_page(page.get("page"), page_offset)
        if page_offset > 1 and current_section == "abstract":
            current_section = "unknown"
        hints = tuple(_clean(value) for value in (page.get("section_hints") or ()) if _clean(value))
        for hint in hints:
            current_section = _section_name(hint, current_section)
        segments, current_section = _page_segments(page, body, current_section)
        for section, segment in segments:
            if section == "references":
                continue
            section_parts[section].append((page_number, segment))
            paragraphs = [part for part in _PARAGRAPH_BREAK_RE.split(segment) if part.strip()]
            for paragraph in paragraphs:
                paragraph_windows = _split_long_words(
                    paragraph, int(config["paragraph_max_words"]), int(config["paragraph_max_words"])
                )
                for value in paragraph_windows:
                    ordinal += 1
                    candidate = _chunk(
                        refno=refno,
                        scale="paragraph",
                        page=page_number,
                        section=section,
                        ordinal=ordinal,
                        text=value,
                        patterns=patterns,
                    )
                    if candidate:
                        by_scale["paragraph"].append(candidate)
                spans = _sentence_spans(paragraph)
                radius = int(config["sentence_radius"])
                for sentence_index in range(len(spans)):
                    left = max(0, sentence_index - radius)
                    right = min(len(spans) - 1, sentence_index + radius)
                    start = spans[left][0]
                    end = spans[right][1]
                    ordinal += 1
                    candidate = _chunk(
                        refno=refno,
                        scale="sentence_window",
                        page=page_number,
                        section=section,
                        ordinal=ordinal,
                        text=paragraph[start:end],
                        patterns=patterns,
                    )
                    if candidate:
                        by_scale["sentence_window"].append(candidate)
            for value in _split_long_words(
                segment,
                int(config["page_window_words"]),
                int(config["page_window_stride"]),
            ):
                ordinal += 1
                candidate = _chunk(
                    refno=refno,
                    scale="page_window",
                    page=page_number,
                    section=section,
                    ordinal=ordinal,
                    text=value,
                    patterns=patterns,
                )
                if candidate:
                    by_scale["page_window"].append(candidate)

    for section, parts in section_parts.items():
        for middle_page, value in _split_section_words(
            parts,
            int(config["section_window_words"]),
            int(config["section_window_stride"]),
        ):
            ordinal += 1
            candidate = _chunk(
                refno=refno,
                scale="section_window",
                page=middle_page,
                section=section,
                ordinal=ordinal,
                text=value,
                patterns=patterns,
            )
            if candidate:
                by_scale["section_window"].append(candidate)

    selected: list[EvidenceChunk] = []
    raw_counts = {scale: len(by_scale[scale]) for scale in SCALES}
    for scale in SCALES:
        deduplicated: list[EvidenceChunk] = []
        seen: set[str] = set()
        for chunk in by_scale[scale]:
            if chunk.text_sha256 in seen:
                continue
            seen.add(chunk.text_sha256)
            deduplicated.append(chunk)
        selected.extend(_cap_scale(deduplicated, int(scale_quotas[scale])))
    selected.sort(key=lambda item: (item.page, SCALES.index(item.scale), item.chunk_id))
    maximum = int(config["max_chunks_per_document"])
    if len(selected) > maximum:
        raise AssertionError("per-scale quotas exceed max_chunks_per_document")
    return selected, {
        "pages": len(raw_pages),
        "body_characters": body_characters,
        "reference_pages_excluded": reference_pages,
        "reference_characters_excluded": reference_characters,
        "raw_chunk_counts": raw_counts,
        "selected_chunk_counts": {
            scale: sum(chunk.scale == scale for chunk in selected) for scale in SCALES
        },
        "selected_chunks": len(selected),
        "lexical_chunks": sum(bool(sum(chunk.lexical_counts)) for chunk in selected),
    }


def select_distractor_chunks(
    chunks: Sequence[EvidenceChunk], count: int
) -> list[EvidenceChunk]:
    if count < 1:
        raise ValueError("distractor chunk count must be positive")
    return [chunks[index] for index in _even_indices(len(chunks), min(count, len(chunks)))]


def reciprocal_rank_fusion(
    dense_scores: Sequence[float], lexical_scores: Sequence[float], *, k: int = 60
) -> list[float]:
    if len(dense_scores) != len(lexical_scores):
        raise ValueError("RRF inputs must align")
    length = len(dense_scores)
    dense_values = [float(value) for value in dense_scores]
    lexical_values = [float(value) for value in lexical_scores]
    dense_unique = sorted(set(dense_values), reverse=True)
    lexical_unique = sorted(
        {value for value in lexical_values if value > 0.0}, reverse=True
    )
    dense_rank = {value: rank + 1 for rank, value in enumerate(dense_unique)}
    lexical_rank = {value: rank + 1 for rank, value in enumerate(lexical_unique)}
    return [
        1.0 / (k + dense_rank[dense_values[index]])
        + (
            1.0 / (k + lexical_rank[lexical_values[index]])
            if lexical_values[index] > 0.0
            else 0.0
        )
        for index in range(length)
    ]


def robust_distractor_statistics(values: Sequence[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        raise ValueError("distractor statistics require finite values")

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        left = int(math.floor(position))
        right = int(math.ceil(position))
        weight = position - left
        return ordered[left] * (1.0 - weight) + ordered[right] * weight

    median = quantile(0.5)
    deviations = sorted(abs(value - median) for value in ordered)
    middle = len(deviations) // 2
    mad = (
        deviations[middle]
        if len(deviations) % 2
        else 0.5 * (deviations[middle - 1] + deviations[middle])
    )
    return {
        "documents": float(len(ordered)),
        "median": median,
        "mad": mad,
        "q90": quantile(0.90),
        "q95": quantile(0.95),
    }


__all__ = [
    "CATEGORIES",
    "SCALES",
    "EvidenceChunk",
    "build_scientific_evidence_chunks",
    "compile_category_patterns",
    "reciprocal_rank_fusion",
    "robust_distractor_statistics",
    "select_distractor_chunks",
]
