#!/usr/bin/env python3
"""Conservative, label-free R7 evidence extraction from original-paper text.

The extractor converts explicit documentary statements into the structured
input consumed by :mod:`r7_pole_overlap_reasoner`.  It intentionally does not
infer missing values, join values across paragraphs, or accept record identity
and evaluation-label fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

try:
    from .r7_pole_overlap_reasoner import PERIOD_SCALE
except ImportError:
    from r7_pole_overlap_reasoner import PERIOD_SCALE


EXTRACTOR_VERSION = "r7_original_text_extractor_v1"


class R7OriginalTextBoundaryError(ValueError):
    """Raised when extractor input crosses the label-free data boundary."""


_FORBIDDEN_KEYS = {"refno", "resultno"}
_OROGENIC_RE = re.compile(
    r"\b(?:accretionary|fold(?:ed)?\s+belt|mobile\s+belt|orogen|orogenic|"
    r"suture|tectonostratigraphic\s+terrane)\b",
    re.I,
)
_UNCERTAIN_RE = re.compile(
    r"\b(?:apparently|could|failed|inconclusive|may|might|not|possibly|"
    r"tentative|uncertain)\b",
    re.I,
)
_FIELD_TEST_RE = re.compile(
    r"\b(?P<test>(?:baked\s+)?contact\s+test|conglomerate\s+test|"
    r"field\s+test|fold\s+test)\b",
    re.I,
)
_FIELD_ESTABLISHED_RE = re.compile(
    r"\b(?:demonstrates?|establish(?:es|ed)?|passes|positive|successful(?:ly)?|"
    r"shows?|yield(?:s|ed)?)\b",
    re.I,
)
_OLDER_RE = re.compile(
    r"\b(?:magneti[sz]ation\s+is\s+)?(?:older\s+than|pre[- ]?dates?|"
    r"prior\s+to)\b",
    re.I,
)
_BOUND_AGE_RE = re.compile(
    r"\b(?:older\s+than|at\s+least|minimum\s+age(?:\s+of)?|"
    r"age\s*(?:>|(?:is\s+)?greater\s+than))\s*"
    r"(?P<age>\d{1,4}(?:\.\d+)?)\s*(?:Ma|Myr)\b",
    re.I,
)
_A95_RE = re.compile(
    r"\bA\s*95\s*(?:=|:|of|is)?\s*(?P<value>\d{1,3}(?:\.\d+)?)"
    r"\s*(?:\u00b0|degrees?|deg\.?|(?=[,;.)]|$))",
    re.I,
)
_AGE_RE = re.compile(
    r"\b(?:age(?:d)?(?:\s+at|\s+of)?|dated\s+at)\s*(?:=|:)?\s*"
    r"(?P<value>\d{1,4}(?:\.\d+)?)\s*(?:Ma|Myr)\b",
    re.I,
)
_AGE_POLE_RE = re.compile(
    r"\b(?P<value>\d{1,4}(?:\.\d+)?)\s*(?:Ma|Myr)\s+"
    r"(?:old\s+)?(?:magneti[sz]ation|pole)\b",
    re.I,
)
_RELIABILITY_RE = re.compile(
    r"\b(?:reliability\s+)?R(?:[- ]?value)?\s*(?:=|:|of)\s*"
    r"(?P<value>\d+(?:\.\d+)?)\b",
    re.I,
)
_SEARCH_R_RE = re.compile(
    r"\bR(?:[- ]?values?)?\s*(?:>=|=>|\u2265|(?:of\s+)?at\s+least)\s*"
    r"3(?:\.0+)?\b",
    re.I,
)
_SEARCH_ALL_RE = re.compile(r"\b(?:all|every)\b", re.I)
_SEARCH_COMPLETE_RE = re.compile(r"\b(?:complete|comprehensive|exhaustive)\b", re.I)
_SEARCH_RE = re.compile(
    r"\b(?:compar(?:e|ed|es|ing|ison)|search|survey|compilation)\b", re.I
)
_CANDIDATE_MARKER_RE = re.compile(
    r"\b(?:younger|comparison|reference)\s+(?:comparison\s+|reference\s+)?pole\b"
    r"|\bpole\b.{0,40}\b(?:younger|comparison|reference)\b",
    re.I | re.S,
)
_POLE_CONTEXT_RE = re.compile(r"\b(?:paleomagnetic\s+pole|pole|VGP)\b", re.I)
_UNIT_RE = re.compile(
    r"\b(?P<unit>[A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,4}\s+"
    r"(?:Complex|Formation|Group|Member|Suite|Supergroup))\b"
)
_PERIOD_RE = re.compile(
    r"\b(?P<period>" + "|".join(
        re.escape(period.name) for period in sorted(PERIOD_SCALE, key=lambda item: -len(item.name))
    ) + r")\b",
    re.I,
)
_SCOPE_NAME = (
    r"[A-Za-z][A-Za-z0-9'\-]*(?:\s+[A-Za-z][A-Za-z0-9'\-]*){0,3}\s+Craton"
)
_SCOPE_PATTERNS = (
    re.compile(
        r"\bstable(?:\s*,\s*|\s+(?:and\s+)?)connected"
        r"(?:\s+(?:part|domain|region)\s+of)?\s+"
        r"(?:the\s+)?(?P<name>" + _SCOPE_NAME + r")\b",
        re.I,
    ),
    re.compile(
        r"\b(?P<name>" + _SCOPE_NAME + r")\s+(?:is|was|forms?|constitutes?)\s+"
        r"(?:a\s+)?stable(?:\s*,\s*|\s+(?:and\s+)?)connected"
        r"(?:\s+(?:craton|region|domain))?\b",
        re.I,
    ),
)
_DIRECTION_COORD_RE = re.compile(
    r"(?<![\d.])(?P<lat>\d{1,2}(?:\.\d+)?)\s*(?:\u00b0|deg\.?)?\s*"
    r"(?P<lat_dir>[NS])\s*[,;/]\s*"
    r"(?P<lon>\d{1,3}(?:\.\d+)?)\s*(?:\u00b0|deg\.?)?\s*"
    r"(?P<lon_dir>[EW])\b",
    re.I,
)
_LABEL_COORD_RE = re.compile(
    r"\b(?:P?lat(?:itude)?)\s*(?:=|:|is)?\s*"
    r"(?P<lat>[+-]?\d{1,2}(?:\.\d+)?)\s*(?:\u00b0|deg\.?)?\s*"
    r"(?P<lat_dir>[NS])?.{0,45}?"
    r"\b(?:P?lo(?:n|ng)(?:gitude)?)\s*(?:=|:|is)?\s*"
    r"(?P<lon>[+-]?\d{1,3}(?:\.\d+)?)\s*(?:\u00b0|deg\.?)?\s*"
    r"(?P<lon_dir>[EW])?\b",
    re.I | re.S,
)


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", "" if value is None else str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise R7OriginalTextBoundaryError(
            f"input must contain JSON-compatible values: {exc}"
        ) from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_boundary(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        seen: set[str] = set()
        for key, member in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if normalized in seen:
                raise R7OriginalTextBoundaryError(
                    f"ambiguous duplicate normalized field at {child_path}"
                )
            seen.add(normalized)
            if normalized in _FORBIDDEN_KEYS or "label" in normalized:
                raise R7OriginalTextBoundaryError(
                    f"forbidden record identifier or label field at {child_path}"
                )
            _validate_boundary(member, child_path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, member in enumerate(value):
            _validate_boundary(member, f"{path}[{index}]")


def _mapping(record: Mapping[str, Any]) -> dict[str, Any]:
    return {_normalized_key(key): value for key, value in record.items()}


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _slug(value: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", _fold(value)).strip("-")
    return slug or fallback


def _target_pattern(target: str) -> re.Pattern[str]:
    words = re.findall(r"[A-Za-z0-9]+", target)
    if not words:
        raise R7OriginalTextBoundaryError("target must contain letters or digits")
    return re.compile(r"\b" + r"[\s_\-/]+".join(map(re.escape, words)) + r"\b", re.I)


def _paragraphs(page_cache_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    pages = page_cache_row.get("pages")
    if not isinstance(pages, Sequence) or isinstance(pages, (str, bytes, bytearray)):
        raise R7OriginalTextBoundaryError("page_cache_row.pages must be an array")
    result: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, Mapping):
            raise R7OriginalTextBoundaryError(
                f"page_cache_row.pages[{page_index - 1}] must be an object"
            )
        text = page.get("text")
        if not isinstance(text, str):
            raise R7OriginalTextBoundaryError(
                f"page_cache_row.pages[{page_index - 1}].text must be a string"
            )
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n+", text) if chunk.strip()]
        section_hints = page.get("section_hints")
        section = ""
        if isinstance(section_hints, Sequence) and not isinstance(
            section_hints, (str, bytes, bytearray)
        ):
            section = str(section_hints[-1]) if section_hints else ""
        for paragraph_index, paragraph in enumerate(chunks, start=1):
            result.append(
                {
                    "text": re.sub(r"\s+", " ", paragraph).strip(),
                    "page": page.get("page", page_index),
                    "paragraph": paragraph_index,
                    "section": section,
                    "page_text_sha256": str(page.get("text_sha256") or _sha256(text)),
                }
            )
    return result


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [part.strip() for part in parts if part.strip()]


def _source(span: Mapping[str, Any], quote: str, method: str) -> dict[str, Any]:
    return {
        "page": span["page"],
        "paragraph": span["paragraph"],
        "section": span["section"],
        "quote": quote,
        "page_text_sha256": span["page_text_sha256"],
        "extraction_method": method,
    }


def _coordinate_values(text: str) -> list[tuple[float, float]]:
    values: set[tuple[float, float]] = set()
    for pattern in (_DIRECTION_COORD_RE, _LABEL_COORD_RE):
        for match in pattern.finditer(text):
            latitude = float(match.group("lat"))
            longitude = float(match.group("lon"))
            lat_dir = (match.group("lat_dir") or "").upper()
            lon_dir = (match.group("lon_dir") or "").upper()
            latitude = -abs(latitude) if lat_dir == "S" else (abs(latitude) if lat_dir == "N" else latitude)
            longitude = -abs(longitude) if lon_dir == "W" else (abs(longitude) if lon_dir == "E" else longitude)
            if -90.0 <= latitude <= 90.0 and -360.0 <= longitude <= 360.0:
                values.add((latitude, longitude))
    return sorted(values)


def _number_values(pattern: re.Pattern[str], text: str) -> list[float]:
    return sorted({float(match.group("value")) for match in pattern.finditer(text)})


def _age_values(text: str) -> list[float]:
    values = set(_number_values(_AGE_RE, text))
    values.update(_number_values(_AGE_POLE_RE, text))
    return sorted(value for value in values if 0.0 <= value <= 4567.3)


def _period_values(text: str) -> list[str]:
    canonical = {_fold(period.name): period.name for period in PERIOD_SCALE}
    return sorted(
        {canonical[_fold(match.group("period"))] for match in _PERIOD_RE.finditer(text)}
    )


def _scope_values(text: str) -> list[str]:
    return sorted(
        {
            re.sub(r"\s+", " ", match.group("name")).strip()
            for pattern in _SCOPE_PATTERNS
            for match in pattern.finditer(text)
        }
    )


def _pole_values(text: str, *, require_reliability: bool) -> tuple[dict[str, Any], list[str], list[str]]:
    values: dict[str, Any] = {}
    missing: list[str] = []
    ambiguous: list[str] = []

    coordinates = _coordinate_values(text)
    if len(coordinates) == 1:
        values["lat"], values["lon"] = coordinates[0]
    elif coordinates:
        ambiguous.extend(["lat", "lon"])
    else:
        missing.extend(["lat", "lon"])

    scalar_fields = {
        "A95": _number_values(_A95_RE, text),
        "age": _age_values(text),
        "period": _period_values(text),
        "craton": _scope_values(text),
    }
    if require_reliability:
        scalar_fields["reliability_R"] = _number_values(_RELIABILITY_RE, text)
    for field, candidates in scalar_fields.items():
        if len(candidates) == 1:
            values[field] = candidates[0]
        elif candidates:
            ambiguous.append(field)
        else:
            missing.append(field)
    if "craton" in values:
        values["stable_connected_region"] = values["craton"]
    return values, sorted(missing), sorted(ambiguous)


def _competing_unit(text: str, target: str) -> str | None:
    def unit_key(value: str) -> str:
        words = _fold(value).split()
        if words and words[0] == "the":
            words.pop(0)
        return " ".join(words)

    target_folded = unit_key(target)
    for match in _UNIT_RE.finditer(text):
        unit = match.group("unit")
        if unit_key(unit) != target_folded:
            return unit
    return None


def _candidate_name(text: str, fallback: str) -> str:
    patterns = (
        re.compile(
            r"\b(?:younger|comparison|reference)\s+(?:comparison\s+|reference\s+)?pole"
            r"(?:\s+(?:named|from))?\s+(?P<name>[A-Z][A-Za-z0-9_.\-]{1,50})\b"
        ),
        re.compile(
            r"\b(?P<name>[A-Z][A-Za-z0-9_.\-]{1,50})\s+"
            r"(?:younger\s+)?(?:comparison\s+|reference\s+)?pole\b"
        ),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group("name")
    return fallback


def _scope_id(value: Any) -> str:
    words = [
        word
        for word in _fold(value).split()
        if word not in {"connected", "craton", "domain", "region", "stable", "the"}
    ]
    return " ".join(words)


def _event(
    kind: str,
    status: str,
    reason_codes: Sequence[str],
    *,
    span: Mapping[str, Any] | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    event: dict[str, Any] = {
        "kind": kind,
        "status": status,
        "reason_codes": sorted(set(reason_codes)),
    }
    if span is not None:
        event.update(
            {
                "page": span["page"],
                "paragraph": span["paragraph"],
                "section": span["section"],
            }
        )
    if source_sha256:
        event["source_sha256"] = source_sha256
    return event


def _extract_target(
    target: str,
    paragraphs: Sequence[Mapping[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    target_re = _target_pattern(target)
    attempts: list[tuple[int, int, dict[str, Any], list[str], list[str], Mapping[str, Any], str, str]] = []
    for span in paragraphs:
        paragraph = str(span["text"])
        if not target_re.search(paragraph):
            continue
        competing = _competing_unit(paragraph, target)
        for sentence in _sentences(paragraph):
            if target_re.search(sentence) and _POLE_CONTEXT_RE.search(sentence):
                values, missing, ambiguous = _pole_values(sentence, require_reliability=False)
                if _competing_unit(sentence, target):
                    ambiguous = sorted(set([*ambiguous, "cross_target_unit"]))
                attempts.append((0, len(missing) + len(ambiguous), values, missing, ambiguous, span, sentence, "target_same_sentence"))
        if _POLE_CONTEXT_RE.search(paragraph):
            values, missing, ambiguous = _pole_values(paragraph, require_reliability=False)
            if competing:
                ambiguous = sorted(set([*ambiguous, "cross_target_unit"]))
            attempts.append((1, len(missing) + len(ambiguous), values, missing, ambiguous, span, paragraph, "target_same_paragraph"))

    if not attempts:
        events.append(_event("target_pole", "REJECTED", ["NO_TARGET_BOUND_POLE_SPAN"]))
        return None
    attempts.sort(key=lambda item: (item[0], item[1], int(item[5]["page"]), int(item[5]["paragraph"])))
    priority, _, values, missing, ambiguous, span, quote, method = attempts[0]
    source = _source(span, quote, method)
    if not values:
        events.append(
            _event(
                "target_pole",
                "REJECTED",
                ["NO_EXPLICIT_TARGET_POLE_VALUES"],
                span=span,
                source_sha256=_sha256(source),
            )
        )
        return None
    if "cross_target_unit" in ambiguous:
        events.append(
            _event(
                "target_pole",
                "REJECTED",
                ["AMBIGUOUS_CROSS_TARGET_UNIT"],
                span=span,
                source_sha256=_sha256(source),
            )
        )
        return None
    status = "ACCEPTED" if not missing and not ambiguous else "PARTIAL"
    reasons = [*(f"MISSING_{field.upper()}" for field in missing), *(f"AMBIGUOUS_{field.upper()}" for field in ambiguous)]
    events.append(_event("target_pole", status, reasons or ["EXPLICIT_TARGET_POLE"], span=span, source_sha256=_sha256(source)))
    return {
        "pole_id": f"target-{_slug(target, 'pole')}",
        **values,
        "source": source,
    }


def _extract_text_candidates(
    target: str,
    target_pole: Mapping[str, Any] | None,
    paragraphs: Sequence[Mapping[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_re = _target_pattern(target)
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for span in paragraphs:
        paragraph = str(span["text"])
        if not target_re.search(paragraph) or not _CANDIDATE_MARKER_RE.search(paragraph):
            continue
        candidate_spans = [
            sentence
            for sentence in _sentences(paragraph)
            if _CANDIDATE_MARKER_RE.search(sentence)
        ]
        if not candidate_spans:
            candidate_spans = [paragraph]
        for candidate_text in candidate_spans:
            values, missing, ambiguous = _pole_values(candidate_text, require_reliability=True)
            reasons = [*(f"MISSING_{field.upper()}" for field in missing), *(f"AMBIGUOUS_{field.upper()}" for field in ambiguous)]
            if _OROGENIC_RE.search(candidate_text):
                reasons.append("OROGENIC_SCOPE")
            source = _source(span, candidate_text, "candidate_target_bound_paragraph")
            if reasons:
                events.append(_event("candidate_pole", "REJECTED", reasons, span=span, source_sha256=_sha256(source)))
                continue
            name = _candidate_name(candidate_text, f"candidate-{len(candidates) + 1:03d}")
            pole_id = _slug(name, f"candidate-{len(candidates) + 1:03d}")
            candidate = {"pole_id": pole_id, **values, "source": source}
            rejection = _candidate_relation_rejection(target_pole, candidate)
            if rejection:
                events.append(_event("candidate_pole", "REJECTED", [rejection], span=span, source_sha256=_sha256(source)))
                continue
            if pole_id in seen_ids:
                events.append(_event("candidate_pole", "REJECTED", ["DUPLICATE_CANDIDATE_ID"], span=span, source_sha256=_sha256(source)))
                continue
            seen_ids.add(pole_id)
            candidates.append(candidate)
            events.append(_event("candidate_pole", "ACCEPTED", ["EXPLICIT_YOUNGER_REFERENCE_POLE"], span=span, source_sha256=_sha256(source)))
    return candidates


def _candidate_relation_rejection(
    target_pole: Mapping[str, Any] | None,
    candidate: Mapping[str, Any],
) -> str | None:
    if _OROGENIC_RE.search(str(candidate.get("craton", ""))) or _OROGENIC_RE.search(
        str(candidate.get("stable_connected_region", ""))
    ):
        return "OROGENIC_SCOPE"
    reliability = _finite_float(candidate.get("reliability_R"))
    if reliability is None:
        return "MISSING_RELIABILITY_R"
    if target_pole:
        target_age = _finite_float(target_pole.get("age"))
        candidate_age = _finite_float(candidate.get("age"))
        if target_age is not None and candidate_age is not None and candidate_age >= target_age:
            return "CANDIDATE_NOT_YOUNGER"
        target_scope = _scope_id(target_pole.get("stable_connected_region"))
        candidate_scope = _scope_id(candidate.get("stable_connected_region"))
        if target_scope and candidate_scope and target_scope != candidate_scope:
            return "DIFFERENT_STABLE_CONNECTED_CRATON"
    return None


def _structured_candidates(
    reference_poles: Any,
    target_pole: Mapping[str, Any] | None,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if reference_poles is None:
        return []
    inherited_source: Any = None
    if isinstance(reference_poles, Mapping):
        wrapper = _mapping(reference_poles)
        raw_poles = wrapper.get("poles")
        inherited_source = wrapper.get("source")
    else:
        raw_poles = reference_poles
    if not isinstance(raw_poles, Sequence) or isinstance(raw_poles, (str, bytes, bytearray)):
        raise R7OriginalTextBoundaryError("reference_poles must be an array or an object with poles")

    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, raw in enumerate(raw_poles, start=1):
        if not isinstance(raw, Mapping):
            raise R7OriginalTextBoundaryError(f"reference_poles[{index - 1}] must be an object")
        fields = _mapping(raw)
        output: dict[str, Any] = {}
        aliases = {
            "pole_id": ("poleid", "id"),
            "lat": ("lat", "latitude"),
            "lon": ("lon", "longitude"),
            "A95": ("a95",),
            "age": ("age", "agema"),
            "period": ("period",),
            "craton": ("craton",),
            "stable_connected_region": ("stableconnectedregion",),
            "reliability_R": ("reliabilityr",),
            "source": ("source",),
        }
        missing: list[str] = []
        for output_key, keys in aliases.items():
            value = next((fields[key] for key in keys if key in fields), None)
            if output_key == "source" and value in (None, "", {}):
                value = inherited_source
            if value in (None, "", {}):
                missing.append(output_key)
            else:
                output[output_key] = value
        numeric_limits = {
            "lat": (-90.0, 90.0),
            "lon": (-360.0, 360.0),
            "A95": (0.0, 180.0),
            "age": (0.0, 4567.3),
            "reliability_R": (0.0, math.inf),
        }
        invalid: list[str] = []
        for field, (minimum, maximum) in numeric_limits.items():
            value = _finite_float(output.get(field))
            if field not in missing and (value is None or not minimum <= value <= maximum):
                invalid.append(field)
            elif value is not None:
                output[field] = value
        if output.get("period") not in {period.name for period in PERIOD_SCALE}:
            if "period" not in missing:
                invalid.append("period")
        if output.get("stable_connected_region") and not _scope_id(output["stable_connected_region"]):
            invalid.append("stable_connected_region")
        reasons = [*(f"MISSING_{field.upper()}" for field in missing), *(f"INVALID_{field.upper()}" for field in invalid)]
        relation_rejection = _candidate_relation_rejection(target_pole, output)
        if relation_rejection:
            reasons.append(relation_rejection)
        source_hash = _sha256(output.get("source", {"reference_index": index}))
        if output.get("pole_id") in seen_ids:
            reasons.append("DUPLICATE_CANDIDATE_ID")
        if reasons:
            events.append(_event("reference_candidate", "REJECTED", reasons, source_sha256=source_hash))
            continue
        output["pole_id"] = str(output["pole_id"])
        output["period"] = str(output["period"])
        output["craton"] = str(output["craton"])
        output["stable_connected_region"] = str(output["stable_connected_region"])
        seen_ids.add(output["pole_id"])
        result.append(output)
        events.append(_event("reference_candidate", "ACCEPTED", ["EXPLICIT_FROZEN_REFERENCE_POLE"], source_sha256=source_hash))
    return result


def _search_from_text(
    target: str,
    paragraphs: Sequence[Mapping[str, Any]],
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    target_re = _target_pattern(target)
    accepted: dict[str, Any] | None = None
    for span in paragraphs:
        text = str(span["text"])
        if not target_re.search(text) or not _SEARCH_RE.search(text):
            continue
        reasons: list[str] = []
        if not _SEARCH_COMPLETE_RE.search(text):
            reasons.append("MISSING_COMPLETE_QUANTIFIER")
        if not _SEARCH_ALL_RE.search(text):
            reasons.append("MISSING_ALL_QUANTIFIER")
        if not _SEARCH_R_RE.search(text):
            reasons.append("MISSING_R_GE_3_SCOPE")
        scopes = _scope_values(text)
        if len(scopes) != 1:
            reasons.append("MISSING_OR_AMBIGUOUS_STABLE_CONNECTED_CRATON")
        if _OROGENIC_RE.search(text):
            reasons.append("OROGENIC_SCOPE")
        source = _source(span, text, "qualified_search_target_bound_paragraph")
        if reasons:
            events.append(_event("qualified_search", "REJECTED", reasons, span=span, source_sha256=_sha256(source)))
            continue
        declaration = {
            "qualified_search_complete": True,
            "source": {
                **source,
                "minimum_reliability_R": 3,
                "stable_connected_region": scopes[0],
                "scope_quantifier": "all",
            },
        }
        events.append(_event("qualified_search", "ACCEPTED", ["EXPLICIT_COMPLETE_QUALIFIED_SEARCH"], span=span, source_sha256=_sha256(source)))
        accepted = accepted or declaration
    return accepted


def _search_from_manifest(
    search_manifest: Mapping[str, Any] | None,
    target_pole: Mapping[str, Any] | None,
    events: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if search_manifest is None:
        return None
    if not isinstance(search_manifest, Mapping):
        raise R7OriginalTextBoundaryError("search_manifest must be an object")
    fields = _mapping(search_manifest)
    scope_raw = fields.get("scope")
    scope = _mapping(scope_raw) if isinstance(scope_raw, Mapping) else {}
    declaration = str(fields.get("declaration") or "")
    complete = fields.get("qualifiedsearchcomplete") is True
    quantifier = _fold(scope.get("quantifier") or fields.get("quantifier"))
    minimum_r = _finite_float(
        scope.get("minimumreliabilityr", fields.get("minimumreliabilityr"))
    )
    stable_region = scope.get("stableconnectedregion", fields.get("stableconnectedregion"))
    source = fields.get("source")
    reasons: list[str] = []
    if not complete or not (_SEARCH_COMPLETE_RE.search(declaration) or complete):
        reasons.append("MISSING_COMPLETE_QUANTIFIER")
    if quantifier != "all" and not _SEARCH_ALL_RE.search(declaration):
        reasons.append("MISSING_ALL_QUANTIFIER")
    if minimum_r != 3.0 and not _SEARCH_R_RE.search(declaration):
        reasons.append("MISSING_R_GE_3_SCOPE")
    if not isinstance(stable_region, str) or not stable_region.strip():
        reasons.append("MISSING_STABLE_CONNECTED_CRATON")
    elif _OROGENIC_RE.search(stable_region):
        reasons.append("OROGENIC_SCOPE")
    if source in (None, "", {}):
        reasons.append("MISSING_AUDITABLE_SOURCE")
    if target_pole and stable_region:
        target_scope = _scope_id(target_pole.get("stable_connected_region"))
        if target_scope and _scope_id(stable_region) != target_scope:
            reasons.append("DIFFERENT_STABLE_CONNECTED_CRATON")
    source_hash = _sha256(source if source not in (None, "", {}) else search_manifest)
    if reasons:
        events.append(_event("search_manifest", "REJECTED", reasons, source_sha256=source_hash))
        return None
    events.append(_event("search_manifest", "ACCEPTED", ["EXPLICIT_COMPLETE_QUALIFIED_SEARCH"], source_sha256=source_hash))
    return {
        "qualified_search_complete": True,
        "source": {
            "manifest_source": source,
            "minimum_reliability_R": 3,
            "stable_connected_region": stable_region,
            "scope_quantifier": "all",
            "manifest_sha256": _sha256(search_manifest),
        },
    }


def _field_constraints(
    target: str,
    paragraphs: Sequence[Mapping[str, Any]],
    candidate_ids: Sequence[str],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    target_re = _target_pattern(target)
    constraints: list[dict[str, Any]] = []
    for span in paragraphs:
        text = str(span["text"])
        test_match = _FIELD_TEST_RE.search(text)
        if not target_re.search(text) or not test_match or not _OLDER_RE.search(text):
            continue
        reasons: list[str] = []
        if _UNCERTAIN_RE.search(text) or not _FIELD_ESTABLISHED_RE.search(text):
            reasons.append("FIELD_TEST_NOT_EXPLICITLY_ESTABLISHED")
        ages = sorted({float(match.group("age")) for match in _BOUND_AGE_RE.finditer(text)})
        if len(ages) != 1:
            reasons.append("MISSING_OR_AMBIGUOUS_MAGNETIZATION_AGE_BOUND")
        matched_candidates = [candidate_id for candidate_id in candidate_ids if re.search(r"\b" + re.escape(candidate_id) + r"\b", _slug(text, ""), re.I)]
        if len(matched_candidates) > 1:
            reasons.append("AMBIGUOUS_FIELD_RESCUE_CANDIDATE")
        source = _source(span, text, "field_rescue_target_bound_paragraph")
        if reasons:
            events.append(_event("field_test_age_constraint", "REJECTED", reasons, span=span, source_sha256=_sha256(source)))
            continue
        constraint: dict[str, Any] = {
            "constraint_id": f"field-constraint-{len(constraints) + 1:03d}",
            "kind": "magnetization_older_than",
            "bound_age_ma": ages[0],
            "field_test": re.sub(r"\s+", "_", test_match.group("test").casefold()),
            "established": True,
            "source": source,
        }
        if matched_candidates:
            constraint["candidate_id"] = matched_candidates[0]
        constraints.append(constraint)
        events.append(_event("field_test_age_constraint", "ACCEPTED", ["EXPLICIT_FIELD_TEST_OLDER_BOUND"], span=span, source_sha256=_sha256(source)))
    return constraints


def extract_r7_original_text_evidence(
    target: str,
    page_cache_row: Mapping[str, Any],
    reference_poles: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    search_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Extract explicit R7 reasoner inputs without consulting labels or row IDs.

    ``page_cache_row`` must contain ``pages`` but must not carry ``REFNO`` or
    ``RESULTNO``.  Optional reference poles and search manifests are accepted
    only as record-independent, source-bearing knowledge inputs.
    """

    if not isinstance(target, str) or not target.strip():
        raise R7OriginalTextBoundaryError("target must be a non-empty string")
    if not isinstance(page_cache_row, Mapping):
        raise R7OriginalTextBoundaryError("page_cache_row must be an object")
    for name, value in (
        ("page_cache_row", page_cache_row),
        ("reference_poles", reference_poles),
        ("search_manifest", search_manifest),
    ):
        _validate_boundary(value, name)
        _canonical_json(value)

    paragraphs = _paragraphs(page_cache_row)
    events: list[dict[str, Any]] = []
    target_pole = _extract_target(target.strip(), paragraphs, events)
    text_candidates = _extract_text_candidates(
        target.strip(), target_pole, paragraphs, events
    )
    reference_candidates = _structured_candidates(
        reference_poles, target_pole, events
    )
    candidates: list[dict[str, Any]] = []
    seen_candidate_ids: set[str] = set()
    for candidate in [*text_candidates, *reference_candidates]:
        candidate_id = str(candidate["pole_id"])
        if candidate_id in seen_candidate_ids:
            events.append(
                _event(
                    "candidate_pole",
                    "REJECTED",
                    ["DUPLICATE_CANDIDATE_ID_ACROSS_SOURCES"],
                    source_sha256=_sha256(candidate.get("source", candidate)),
                )
            )
            continue
        seen_candidate_ids.add(candidate_id)
        candidates.append(candidate)

    text_search = _search_from_text(target.strip(), paragraphs, events)
    manifest_search = _search_from_manifest(search_manifest, target_pole, events)
    qualified_search = manifest_search or text_search
    constraints = _field_constraints(
        target.strip(), paragraphs, [str(item["pole_id"]) for item in candidates], events
    )
    target_event = next((event for event in events if event["kind"] == "target_pole"), None)
    target_complete = bool(target_event and target_event["status"] == "ACCEPTED")
    status = "COMPLETE" if target_complete else "INCOMPLETE"
    audit = {
        "schema_version": EXTRACTOR_VERSION,
        "status": status,
        "target_bound_paragraphs_examined": sum(
            bool(_target_pattern(target.strip()).search(str(span["text"])))
            for span in paragraphs
        ),
        "pages_examined": len({str(span["page"]) for span in paragraphs}),
        "accepted_counts": {
            "target": int(target_complete),
            "candidates": len(candidates),
            "field_test_age_constraints": len(constraints),
            "qualified_search": int(qualified_search is not None),
        },
        "events": events,
        "input_sha256": _sha256(
            {
                "target": target.strip(),
                "page_cache_row": page_cache_row,
                "reference_poles": reference_poles,
                "search_manifest": search_manifest,
            }
        ),
        "supervision": "none",
        "uses_record_identity": False,
        "uses_answer_lookup": False,
        "forbidden_input_fields": ["REFNO", "RESULTNO", "*label*"],
    }
    return {
        "target": target_pole,
        "candidates": candidates,
        "field_test_age_constraints": constraints,
        "qualified_search": qualified_search,
        "extraction_audit": audit,
    }


extract_r7_original_text = extract_r7_original_text_evidence


__all__ = [
    "EXTRACTOR_VERSION",
    "R7OriginalTextBoundaryError",
    "extract_r7_original_text",
    "extract_r7_original_text_evidence",
]
