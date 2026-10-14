#!/usr/bin/env python3
"""Quantitative, label-free reasoning for paleomagnetic criterion R7.

The reasoner implements the formal R7 instructions supplied with the project:

* compare only poles from the same stable connected cratonic region;
* a candidate comparison pole must have reliability R >= 3;
* the candidate must be younger by more than one geological period;
* resemblance is an overlap of the two spherical A95 envelopes; and
* field-test evidence proving that magnetization predates a resembling younger
  pole rescues that comparison.

The public API consumes JSON-like structures only.  It neither reads project
tables nor accepts row identifiers or supervised-label fields.  All returned
source references are SHA-256 hashes, so quantitative decisions can be audited
without copying source text into prediction artifacts.

Field-test age constraints use this schema::

    {
        "constraint_id": "fold-test-1",
        "kind": "magnetization_older_than",
        "bound_age_ma": 1100.0,
        "field_test": "positive_fold_test",
        "established": true,
        "candidate_id": "younger-pole-1",  # optional; otherwise applies to all
        "source": {"citation": "...", "quote": "..."}
    }

Because geological ages are measured before present, the constraint proves an
older magnetization only when ``bound_age_ma > candidate.age``.  Equality is
not enough.  A95 overlap is inclusive at the confidence-envelope boundary.
Disjoint candidates support an overall pass only when an auditable search
declaration is supplied as ``{"qualified_search_complete": true,
"source": ...}``; a finite candidate list is not assumed to be exhaustive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


ENGINE_VERSION = "r7_pole_overlap_reasoner_v1"
PERIOD_SCALE_VERSION = "r7_formal_period_scale_v1"
# SHA-256 of the supplied R7 section only, from its ``3.7`` heading through the
# end of that section.  This is deliberately not a hash of the whole attachment.
CRITERION_SOURCE_SHA256 = (
    "ff2667973344023bc2ccf078ee41cd2a5d3eced3a813320960633c4a3b2769d5"
)
RELATIONS = frozenset({"SUPPORT", "CONFLICT", "UNKNOWN", "MIXED"})


class R7InputBoundaryError(ValueError):
    """Raised when input crosses the label-free scoring boundary."""


@dataclass(frozen=True)
class GeologicalPeriod:
    """A formal time unit ordered from youngest (rank 0) to oldest."""

    name: str
    younger_bound_ma: float
    older_bound_ma: float
    rank: int


# Rounded international chronostratigraphic boundaries are sufficient for the
# categorical period-separation gate.  The exact input ages remain in the
# output, and the scale is versioned and hashed for reproducibility.
_PERIOD_BOUNDS = (
    ("Quaternary", 0.0, 2.58),
    ("Neogene", 2.58, 23.03),
    ("Paleogene", 23.03, 66.0),
    ("Cretaceous", 66.0, 145.0),
    ("Jurassic", 145.0, 201.4),
    ("Triassic", 201.4, 251.902),
    ("Permian", 251.902, 298.9),
    ("Carboniferous", 298.9, 358.9),
    ("Devonian", 358.9, 419.2),
    ("Silurian", 419.2, 443.8),
    ("Ordovician", 443.8, 485.4),
    ("Cambrian", 485.4, 538.8),
    ("Ediacaran", 538.8, 635.0),
    ("Cryogenian", 635.0, 720.0),
    ("Tonian", 720.0, 1000.0),
    ("Stenian", 1000.0, 1200.0),
    ("Ectasian", 1200.0, 1400.0),
    ("Calymmian", 1400.0, 1600.0),
    ("Statherian", 1600.0, 1800.0),
    ("Orosirian", 1800.0, 2050.0),
    ("Rhyacian", 2050.0, 2300.0),
    ("Siderian", 2300.0, 2500.0),
    ("Neoarchean", 2500.0, 2800.0),
    ("Mesoarchean", 2800.0, 3200.0),
    ("Paleoarchean", 3200.0, 3600.0),
    ("Eoarchean", 3600.0, 4031.0),
    ("Hadean", 4031.0, 4567.3),
)
PERIOD_SCALE = tuple(
    GeologicalPeriod(name, younger, older, rank)
    for rank, (name, younger, older) in enumerate(_PERIOD_BOUNDS)
)


def _fold(value: Any) -> str:
    text = unicodedata.normalize("NFKD", "" if value is None else str(value))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


_PERIOD_BY_KEY = {_fold(period.name): period for period in PERIOD_SCALE}
_BROAD_TIME_UNITS = {
    "phanerozoic": (0.0, 538.8),
    "neoproterozoic": (538.8, 1000.0),
    "mesoproterozoic": (1000.0, 1600.0),
    "paleoproterozoic": (1600.0, 2500.0),
    "proterozoic": (538.8, 2500.0),
    "archean": (2500.0, 4031.0),
    "archaean": (2500.0, 4031.0),
    "precambrian": (538.8, 4567.3),
}
_UNKNOWN_SCOPE_VALUES = {
    "",
    "n a",
    "na",
    "none",
    "not established",
    "not known",
    "uncertain",
    "undetermined",
    "unknown",
}
_OROGENIC_SCOPE_RE = re.compile(
    r"\b(?:accretionary|fold(?:ed)? belt|mobile belt|orogen|orogenic|suture|terrane)\b",
    re.I,
)
_GENERIC_SCOPE_WORDS = {
    "block",
    "connected",
    "craton",
    "domain",
    "region",
    "stable",
}
_FORBIDDEN_EXACT_KEYS = {
    "answer",
    "answers",
    "benchmarklabel",
    "expectedr7",
    "gold",
    "goldlabel",
    "groundtruth",
    "refno",
    "resultno",
    "r3",
    "r5",
    "r7",
    "r7answer",
    "r7label",
    "r7value",
}


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
        raise R7InputBoundaryError(f"input must contain JSON-compatible values: {exc}") from exc


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _walk_reject_supervision(value: Any, path: str = "$") -> None:
    if isinstance(value, Mapping):
        seen: set[str] = set()
        for key, member in value.items():
            normalized = _normalized_key(key)
            child_path = f"{path}.{key}"
            if normalized in seen:
                raise R7InputBoundaryError(
                    f"ambiguous duplicate normalized field at {child_path}"
                )
            seen.add(normalized)
            if (
                normalized in _FORBIDDEN_EXACT_KEYS
                or normalized in {"label", "labels"}
                or "label" in normalized
            ):
                raise R7InputBoundaryError(
                    f"forbidden identifier or supervised-label field at {child_path}"
                )
            _walk_reject_supervision(member, child_path)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, member in enumerate(value):
            _walk_reject_supervision(member, f"{path}[{index}]")


def validate_label_free_input(value: Any) -> None:
    """Reject row identifiers and any field that can carry an answer label."""

    _walk_reject_supervision(value)
    _canonical_json(value)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _require_coordinate(value: Any, name: str) -> float:
    result = _finite_float(value)
    if result is None:
        raise ValueError(f"{name} must be a finite number")
    if name == "latitude" and not -90.0 <= result <= 90.0:
        raise ValueError("latitude must be in [-90, 90]")
    return result


def normalize_longitude(longitude: float) -> float:
    """Return a finite longitude in the half-open interval [-180, 180)."""

    value = _require_coordinate(longitude, "longitude")
    normalized = (value + 180.0) % 360.0 - 180.0
    return 0.0 if normalized == -0.0 else normalized


def angular_distance_degrees(
    latitude_1: float,
    longitude_1: float,
    latitude_2: float,
    longitude_2: float,
) -> float:
    """Great-circle angular distance using a numerically stable haversine."""

    lat_1 = math.radians(_require_coordinate(latitude_1, "latitude"))
    lat_2 = math.radians(_require_coordinate(latitude_2, "latitude"))
    lon_1 = math.radians(normalize_longitude(longitude_1))
    lon_2 = math.radians(normalize_longitude(longitude_2))
    delta_lat = lat_2 - lat_1
    delta_lon = (lon_2 - lon_1 + math.pi) % (2.0 * math.pi) - math.pi
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat_1) * math.cos(lat_2) * math.sin(delta_lon / 2.0) ** 2
    )
    haversine = min(1.0, max(0.0, haversine))
    return math.degrees(
        2.0 * math.atan2(math.sqrt(haversine), math.sqrt(1.0 - haversine))
    )


def _validate_a95(value: Any, name: str) -> float:
    result = _finite_float(value)
    if result is None or not 0.0 <= result <= 180.0:
        raise ValueError(f"{name} must be a finite angle in [0, 180]")
    return result


def a95_overlap_details(
    latitude_1: float,
    longitude_1: float,
    a95_1: float,
    latitude_2: float,
    longitude_2: float,
    a95_2: float,
    *,
    tolerance_deg: float = 1e-9,
) -> dict[str, Any]:
    """Return the spherical-cap A95 overlap decision and formula parameters."""

    tolerance = _finite_float(tolerance_deg)
    if tolerance is None or tolerance < 0.0:
        raise ValueError("tolerance_deg must be a non-negative finite number")
    first = _validate_a95(a95_1, "a95_1")
    second = _validate_a95(a95_2, "a95_2")
    distance = angular_distance_degrees(
        latitude_1, longitude_1, latitude_2, longitude_2
    )
    raw_sum = first + second
    threshold = min(180.0, raw_sum)
    margin = threshold - distance
    return {
        "angular_distance_deg": distance,
        "target_a95_deg": first,
        "candidate_a95_deg": second,
        "raw_a95_sum_deg": raw_sum,
        "effective_overlap_threshold_deg": threshold,
        "overlap_margin_deg": margin,
        "tolerance_deg": tolerance,
        "overlaps": distance <= threshold + tolerance,
        "boundary_is_inclusive": True,
    }


def a95_envelopes_overlap(
    latitude_1: float,
    longitude_1: float,
    a95_1: float,
    latitude_2: float,
    longitude_2: float,
    a95_2: float,
    *,
    tolerance_deg: float = 1e-9,
) -> bool:
    """Return whether two circular spherical A95 envelopes overlap."""

    return bool(
        a95_overlap_details(
            latitude_1,
            longitude_1,
            a95_1,
            latitude_2,
            longitude_2,
            a95_2,
            tolerance_deg=tolerance_deg,
        )["overlaps"]
    )


def _period_for_age(age_ma: float) -> GeologicalPeriod | None:
    for period in PERIOD_SCALE:
        if period.younger_bound_ma <= age_ma <= period.older_bound_ma:
            return period
    return None


def _period_key(value: Any) -> str:
    folded = _fold(value)
    words = folded.split()
    while words and words[0] in {"early", "late", "lower", "middle", "mid", "upper"}:
        words.pop(0)
    return " ".join(words)


def _resolve_period(
    value: Any,
    age_ma: float,
    *,
    boundary_tolerance_ma: float,
) -> dict[str, Any]:
    key = _period_key(value)
    if key in _UNKNOWN_SCOPE_VALUES:
        return {
            "status": "UNKNOWN",
            "reason_code": "UNKNOWN_PERIOD",
            "input_period": "" if value is None else str(value),
        }

    period = _PERIOD_BY_KEY.get(key)
    method = "named_formal_period"
    if period is None and key in _BROAD_TIME_UNITS:
        younger, older = _BROAD_TIME_UNITS[key]
        if not (
            younger - boundary_tolerance_ma
            <= age_ma
            <= older + boundary_tolerance_ma
        ):
            return {
                "status": "UNKNOWN",
                "reason_code": "PERIOD_AGE_MISMATCH",
                "input_period": str(value),
                "input_age_ma": age_ma,
                "period_bounds_ma": [younger, older],
            }
        period = _period_for_age(age_ma)
        method = "formal_period_inferred_within_named_broad_unit"
    if period is None:
        return {
            "status": "UNKNOWN",
            "reason_code": "UNKNOWN_PERIOD",
            "input_period": "" if value is None else str(value),
        }
    if not (
        period.younger_bound_ma - boundary_tolerance_ma
        <= age_ma
        <= period.older_bound_ma + boundary_tolerance_ma
    ):
        return {
            "status": "UNKNOWN",
            "reason_code": "PERIOD_AGE_MISMATCH",
            "input_period": str(value),
            "input_age_ma": age_ma,
            "resolved_period": period.name,
            "period_bounds_ma": [period.younger_bound_ma, period.older_bound_ma],
        }
    return {
        "status": "RESOLVED",
        "reason_code": "PERIOD_RESOLVED",
        "input_period": str(value),
        "resolved_period": period.name,
        "rank_youngest_zero": period.rank,
        "period_bounds_ma": [period.younger_bound_ma, period.older_bound_ma],
        "resolution_method": method,
    }


def _mapping_by_normalized_key(record: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in record.items():
        normalized[_normalized_key(key)] = value
    return normalized


def _record_source_hash(record: Mapping[str, Any]) -> tuple[str, str]:
    normalized = _mapping_by_normalized_key(record)
    if "source" in normalized and normalized["source"] not in (None, "", {}):
        return _sha256(normalized["source"]), "source"
    fallback = {
        str(key): value
        for key, value in record.items()
        if _normalized_key(key) != "source"
    }
    return _sha256(fallback), "structured_record"


def _scope_identifier(value: str) -> str:
    words = [word for word in _fold(value).split() if word not in _GENERIC_SCOPE_WORDS]
    return " ".join(words)


def _scope_descriptor(craton: Any, stable_region: Any) -> dict[str, Any]:
    craton_text = "" if craton is None else str(craton).strip()
    craton_folded = _fold(craton_text)
    if craton_folded in _UNKNOWN_SCOPE_VALUES:
        return {"status": "UNKNOWN", "reason_code": "UNKNOWN_CRATON"}
    if _OROGENIC_SCOPE_RE.search(craton_text):
        return {"status": "OROGENIC", "reason_code": "OROGENIC_SCOPE"}

    if isinstance(stable_region, bool):
        if not stable_region:
            return {"status": "UNKNOWN", "reason_code": "UNSTABLE_OR_UNKNOWN_SCOPE"}
        region_text = craton_text
    elif isinstance(stable_region, str):
        region_text = stable_region.strip()
    else:
        return {"status": "UNKNOWN", "reason_code": "UNKNOWN_STABLE_CONNECTED_REGION"}

    region_folded = _fold(region_text)
    if region_folded in _UNKNOWN_SCOPE_VALUES:
        return {"status": "UNKNOWN", "reason_code": "UNKNOWN_STABLE_CONNECTED_REGION"}
    if _OROGENIC_SCOPE_RE.search(region_text):
        return {"status": "OROGENIC", "reason_code": "OROGENIC_SCOPE"}

    craton_id = _scope_identifier(craton_text) or craton_folded
    region_id = _scope_identifier(region_text)
    if region_folded in {"stable", "true", "yes"} or not region_id:
        region_id = craton_id
    return {
        "status": "STABLE",
        "reason_code": "STABLE_SCOPE_RESOLVED",
        "craton_id": craton_id,
        "stable_connected_region_id": region_id,
    }


def _parse_pole(
    raw: Mapping[str, Any],
    *,
    fallback_id: str,
    boundary_tolerance_ma: float,
    require_reliability: bool = True,
) -> dict[str, Any]:
    fields = _mapping_by_normalized_key(raw)
    pole_id = str(fields.get("poleid") or fields.get("id") or fallback_id).strip()
    source_hash, source_basis = _record_source_hash(raw)
    errors: list[str] = []

    latitude = _finite_float(fields.get("lat"))
    longitude = _finite_float(fields.get("lon"))
    a95 = _finite_float(fields.get("a95"))
    age = _finite_float(fields.get("age"))
    reliability = _finite_float(fields.get("reliabilityr"))
    if latitude is None or not -90.0 <= latitude <= 90.0:
        errors.append("INVALID_LATITUDE")
    if longitude is None:
        errors.append("INVALID_LONGITUDE")
    if a95 is None or not 0.0 <= a95 <= 180.0:
        errors.append("INVALID_A95")
    if age is None or not 0.0 <= age <= 4567.3 + boundary_tolerance_ma:
        errors.append("INVALID_AGE")
    if require_reliability and (reliability is None or reliability < 0.0):
        errors.append("INVALID_RELIABILITY_R")
    elif reliability is not None and reliability < 0.0:
        errors.append("INVALID_RELIABILITY_R")

    period = (
        _resolve_period(
            fields.get("period"),
            age,
            boundary_tolerance_ma=boundary_tolerance_ma,
        )
        if age is not None and age >= 0.0
        else {"status": "UNKNOWN", "reason_code": "INVALID_AGE"}
    )
    if period["status"] != "RESOLVED":
        errors.append(str(period["reason_code"]))

    scope = _scope_descriptor(
        fields.get("craton"), fields.get("stableconnectedregion")
    )
    return {
        "pole_id": pole_id,
        "lat": latitude,
        "lon": normalize_longitude(longitude) if longitude is not None else None,
        "A95": a95,
        "age_ma": age,
        "period": period,
        "craton": "" if fields.get("craton") is None else str(fields.get("craton")),
        "stable_connected_region": (
            fields.get("stableconnectedregion")
            if isinstance(fields.get("stableconnectedregion"), (str, bool))
            else None
        ),
        "scope": scope,
        "reliability_R": reliability,
        "valid_numeric_and_period": not errors,
        "validation_errors": sorted(set(errors)),
        "source_sha256": source_hash,
        "source_hash_basis": source_basis,
    }


def _parse_constraint(
    raw: Mapping[str, Any],
    *,
    fallback_id: str,
) -> dict[str, Any]:
    fields = _mapping_by_normalized_key(raw)
    constraint_id = str(
        fields.get("constraintid") or fields.get("id") or fallback_id
    ).strip()
    kind = _fold(fields.get("kind")).replace(" ", "_")
    bound_age = _finite_float(fields.get("boundagema"))
    established = fields.get("established") is True
    field_test = str(fields.get("fieldtest") or "").strip()
    candidate_id_raw = fields.get("candidateid")
    candidate_id = None if candidate_id_raw in (None, "") else str(candidate_id_raw).strip()
    source_hash, source_basis = _record_source_hash(raw)
    errors: list[str] = []
    if kind not in {"magnetization_older_than", "magnetization_minimum_age"}:
        errors.append("UNSUPPORTED_FIELD_CONSTRAINT_KIND")
    if bound_age is None or not 0.0 <= bound_age <= 4567.3:
        errors.append("INVALID_FIELD_CONSTRAINT_AGE")
    if not established:
        errors.append("FIELD_CONSTRAINT_NOT_ESTABLISHED")
    if not field_test:
        errors.append("MISSING_FIELD_TEST_TYPE")
    return {
        "constraint_id": constraint_id,
        "kind": kind,
        "bound_age_ma": bound_age,
        "field_test": field_test,
        "established": established,
        "candidate_id": candidate_id,
        "valid": not errors,
        "validation_errors": sorted(set(errors)),
        "source_sha256": source_hash,
        "source_hash_basis": source_basis,
    }


def _parse_qualified_search(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate a claim that the eligible younger-pole search is exhaustive."""

    if raw is None:
        return {
            "provided": False,
            "qualified_search_complete": False,
            "accepted_as_complete": False,
            "auditable_source": False,
            "validation_errors": ["QUALIFIED_SEARCH_DECLARATION_NOT_PROVIDED"],
            "source_sha256": _sha256({"qualified_search": "not_provided"}),
            "source_hash_basis": "absence_marker",
        }
    fields = _mapping_by_normalized_key(raw)
    complete_value = fields.get("qualifiedsearchcomplete")
    complete = complete_value is True
    source_value = fields.get("source")
    source_present = (
        isinstance(source_value, str) and bool(source_value.strip())
    ) or (
        isinstance(source_value, Mapping) and bool(source_value)
    )
    source_hash, source_basis = _record_source_hash(raw)
    errors: list[str] = []
    if not isinstance(complete_value, bool):
        errors.append("QUALIFIED_SEARCH_COMPLETE_MUST_BE_BOOLEAN")
    if complete and not source_present:
        errors.append("COMPLETE_SEARCH_REQUIRES_AUDITABLE_SOURCE")
    return {
        "provided": True,
        "qualified_search_complete": complete,
        "accepted_as_complete": complete and source_present and not errors,
        "auditable_source": source_present,
        "validation_errors": errors,
        "source_sha256": source_hash,
        "source_hash_basis": source_basis,
    }


def _public_pole(pole: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pole_id": pole["pole_id"],
        "lat": pole["lat"],
        "lon_normalized": pole["lon"],
        "A95": pole["A95"],
        "age_ma": pole["age_ma"],
        "period": pole["period"],
        "craton": pole["craton"],
        "stable_connected_region": pole["stable_connected_region"],
        "scope": pole["scope"],
        "reliability_R": pole["reliability_R"],
        "validation_errors": pole["validation_errors"],
        "source_sha256": pole["source_sha256"],
        "source_hash_basis": pole["source_hash_basis"],
    }


@dataclass(frozen=True)
class R7ReasonerConfig:
    minimum_candidate_reliability_R: float = 3.0
    minimum_period_rank_gap: int = 2
    angle_tolerance_deg: float = 1e-9
    age_tolerance_ma: float = 1e-9
    period_boundary_tolerance_ma: float = 5.0

    def __post_init__(self) -> None:
        if self.minimum_candidate_reliability_R < 0.0:
            raise ValueError("minimum_candidate_reliability_R must be non-negative")
        if self.minimum_period_rank_gap < 2:
            raise ValueError("minimum_period_rank_gap must be at least 2")
        for name in (
            "angle_tolerance_deg",
            "age_tolerance_ma",
            "period_boundary_tolerance_ma",
        ):
            value = _finite_float(getattr(self, name))
            if value is None or value < 0.0:
                raise ValueError(f"{name} must be a non-negative finite number")


class R7PoleOverlapReasoner:
    """Apply the quantitative R7 gates and emit provenance-bearing relations."""

    def __init__(self, config: R7ReasonerConfig | None = None) -> None:
        self.config = config or R7ReasonerConfig()

    def _unknown_relation(
        self,
        target: Mapping[str, Any],
        candidate: Mapping[str, Any],
        reason_code: str,
        eligibility: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "relation_id": f"R7.POLE_COMPARE.{candidate['pole_id']}",
            "candidate_id": candidate["pole_id"],
            "relation": "UNKNOWN",
            "reason_code": reason_code,
            "accepted_for_comparison": False,
            "eligibility": dict(eligibility),
            "formula_parameters": {},
            "evidence": {
                "target_source_sha256": target["source_sha256"],
                "candidate_source_sha256": candidate["source_sha256"],
                "criterion_source_sha256": CRITERION_SOURCE_SHA256,
            },
        }

    def _time_gate(
        self,
        target: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        target_age = target["age_ma"]
        candidate_age = candidate["age_ma"]
        if target_age is None or candidate_age is None:
            return {"passed": False, "reason_code": "INVALID_AGE"}
        difference = target_age - candidate_age
        target_period = target["period"]
        candidate_period = candidate["period"]
        formula = {
            "target_age_ma": target_age,
            "candidate_age_ma": candidate_age,
            "age_difference_ma": difference,
            "candidate_is_younger": difference > self.config.age_tolerance_ma,
            "minimum_period_rank_gap": self.config.minimum_period_rank_gap,
            "age_tolerance_ma": self.config.age_tolerance_ma,
        }
        if difference <= self.config.age_tolerance_ma:
            return {"passed": False, "reason_code": "CANDIDATE_NOT_YOUNGER", **formula}
        if target_period.get("status") != "RESOLVED" or candidate_period.get("status") != "RESOLVED":
            return {"passed": False, "reason_code": "UNKNOWN_PERIOD", **formula}
        rank_gap = (
            int(target_period["rank_youngest_zero"])
            - int(candidate_period["rank_youngest_zero"])
        )
        formula.update(
            {
                "target_formal_period": target_period["resolved_period"],
                "candidate_formal_period": candidate_period["resolved_period"],
                "period_rank_gap": rank_gap,
            }
        )
        if rank_gap < self.config.minimum_period_rank_gap:
            return {
                "passed": False,
                "reason_code": "NOT_YOUNGER_BY_MORE_THAN_ONE_PERIOD",
                **formula,
            }
        return {"passed": True, "reason_code": "YOUNGER_BY_MORE_THAN_ONE_PERIOD", **formula}

    def _geography_gate(
        self,
        target: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> dict[str, Any]:
        target_scope = target["scope"]
        candidate_scope = candidate["scope"]
        if target_scope.get("status") == "OROGENIC" or candidate_scope.get("status") == "OROGENIC":
            return {"passed": False, "reason_code": "OROGENIC_SCOPE"}
        if target_scope.get("status") != "STABLE" or candidate_scope.get("status") != "STABLE":
            return {"passed": False, "reason_code": "UNKNOWN_STABLE_SCOPE"}
        if target_scope["craton_id"] != candidate_scope["craton_id"]:
            return {
                "passed": False,
                "reason_code": "DIFFERENT_CRATON",
                "target_craton_id": target_scope["craton_id"],
                "candidate_craton_id": candidate_scope["craton_id"],
            }
        if (
            target_scope["stable_connected_region_id"]
            != candidate_scope["stable_connected_region_id"]
        ):
            return {
                "passed": False,
                "reason_code": "DIFFERENT_STABLE_CONNECTED_REGION",
                "target_region_id": target_scope["stable_connected_region_id"],
                "candidate_region_id": candidate_scope["stable_connected_region_id"],
            }
        return {
            "passed": True,
            "reason_code": "SAME_STABLE_CONNECTED_CRATON",
            "craton_id": target_scope["craton_id"],
            "stable_connected_region_id": target_scope["stable_connected_region_id"],
        }

    def _applicable_rescues(
        self,
        candidate: Mapping[str, Any],
        constraints: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        rescues: list[dict[str, Any]] = []
        for constraint in constraints:
            selector = constraint.get("candidate_id")
            if selector is not None and selector != candidate["pole_id"]:
                continue
            if not constraint["valid"]:
                continue
            bound_age = float(constraint["bound_age_ma"])
            candidate_age = float(candidate["age_ma"])
            margin = bound_age - candidate_age
            if margin <= self.config.age_tolerance_ma:
                continue
            rescues.append(
                {
                    "relation_id": f"R7.FIELD_RESCUE.{constraint['constraint_id']}",
                    "constraint_id": constraint["constraint_id"],
                    "relation": "SUPPORT",
                    "reason_code": "FIELD_EVIDENCE_PROVES_MAGNETIZATION_OLDER",
                    "field_test": constraint["field_test"],
                    "formula_parameters": {
                        "magnetization_older_than_bound_ma": bound_age,
                        "candidate_age_ma": candidate_age,
                        "older_age_margin_ma": margin,
                        "strict_inequality_passed": True,
                    },
                    "evidence": {
                        "field_constraint_source_sha256": constraint["source_sha256"],
                        "criterion_source_sha256": CRITERION_SOURCE_SHA256,
                    },
                }
            )
        return rescues

    def evaluate(
        self,
        target: Mapping[str, Any],
        candidates: Sequence[Mapping[str, Any]],
        field_test_age_constraints: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        qualified_search: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate target resemblance to eligible younger comparison poles."""

        if not isinstance(target, Mapping):
            raise R7InputBoundaryError("target must be an object")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes, bytearray)):
            raise R7InputBoundaryError("candidates must be an array of objects")
        if field_test_age_constraints is None:
            constraint_inputs: list[Mapping[str, Any]] = []
        elif isinstance(field_test_age_constraints, Mapping):
            constraint_inputs = [field_test_age_constraints]
        elif isinstance(field_test_age_constraints, Sequence) and not isinstance(
            field_test_age_constraints, (str, bytes, bytearray)
        ):
            constraint_inputs = list(field_test_age_constraints)
        else:
            raise R7InputBoundaryError(
                "field_test_age_constraints must be an object or array of objects"
            )
        if qualified_search is not None and not isinstance(qualified_search, Mapping):
            raise R7InputBoundaryError("qualified_search must be an object")
        if any(not isinstance(item, Mapping) for item in candidates):
            raise R7InputBoundaryError("each candidate must be an object")
        if any(not isinstance(item, Mapping) for item in constraint_inputs):
            raise R7InputBoundaryError("each field-test constraint must be an object")

        input_payload = {
            "target": target,
            "candidates": list(candidates),
            "field_test_age_constraints": constraint_inputs,
            "qualified_search": qualified_search,
        }
        validate_label_free_input(input_payload)

        parsed_target = _parse_pole(
            target,
            fallback_id="target",
            boundary_tolerance_ma=self.config.period_boundary_tolerance_ma,
            require_reliability=False,
        )
        parsed_candidates = [
            _parse_pole(
                candidate,
                fallback_id=f"candidate_{index:03d}",
                boundary_tolerance_ma=self.config.period_boundary_tolerance_ma,
            )
            for index, candidate in enumerate(candidates, start=1)
        ]
        candidate_ids = [candidate["pole_id"] for candidate in parsed_candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise R7InputBoundaryError("candidate pole_id values must be unique")
        constraints = [
            _parse_constraint(item, fallback_id=f"field_constraint_{index:03d}")
            for index, item in enumerate(constraint_inputs, start=1)
        ]
        constraint_ids = [constraint["constraint_id"] for constraint in constraints]
        if len(constraint_ids) != len(set(constraint_ids)):
            raise R7InputBoundaryError("field constraint_id values must be unique")
        search_audit = _parse_qualified_search(qualified_search)

        relations: list[dict[str, Any]] = []
        used_rescues: dict[str, set[str]] = {constraint_id: set() for constraint_id in constraint_ids}
        for candidate in parsed_candidates:
            eligibility: dict[str, Any] = {}
            if not parsed_target["valid_numeric_and_period"]:
                eligibility["target_validation"] = {
                    "passed": False,
                    "errors": parsed_target["validation_errors"],
                }
                relations.append(
                    self._unknown_relation(
                        parsed_target,
                        candidate,
                        "INVALID_TARGET_POLE",
                        eligibility,
                    )
                )
                continue
            if not candidate["valid_numeric_and_period"]:
                eligibility["candidate_validation"] = {
                    "passed": False,
                    "errors": candidate["validation_errors"],
                }
                relations.append(
                    self._unknown_relation(
                        parsed_target,
                        candidate,
                        "INVALID_CANDIDATE_POLE",
                        eligibility,
                    )
                )
                continue

            reliability_gate = {
                "passed": float(candidate["reliability_R"])
                >= self.config.minimum_candidate_reliability_R,
                "candidate_reliability_R": candidate["reliability_R"],
                "minimum_reliability_R": self.config.minimum_candidate_reliability_R,
            }
            reliability_gate["reason_code"] = (
                "RELIABILITY_R_AT_LEAST_3"
                if reliability_gate["passed"]
                else "CANDIDATE_RELIABILITY_BELOW_3"
            )
            eligibility["reliability"] = reliability_gate
            geography_gate = self._geography_gate(parsed_target, candidate)
            eligibility["geography"] = geography_gate
            time_gate = self._time_gate(parsed_target, candidate)
            eligibility["time"] = time_gate

            failed_gate = next(
                (
                    gate
                    for gate in (reliability_gate, geography_gate, time_gate)
                    if not gate["passed"]
                ),
                None,
            )
            if failed_gate is not None:
                relations.append(
                    self._unknown_relation(
                        parsed_target,
                        candidate,
                        str(failed_gate["reason_code"]),
                        eligibility,
                    )
                )
                continue

            overlap = a95_overlap_details(
                float(parsed_target["lat"]),
                float(parsed_target["lon"]),
                float(parsed_target["A95"]),
                float(candidate["lat"]),
                float(candidate["lon"]),
                float(candidate["A95"]),
                tolerance_deg=self.config.angle_tolerance_deg,
            )
            rescues = self._applicable_rescues(candidate, constraints) if overlap["overlaps"] else []
            for rescue in rescues:
                used_rescues[rescue["constraint_id"]].add(candidate["pole_id"])

            if overlap["overlaps"] and rescues:
                relation = "SUPPORT"
                reason_code = "FIELD_EVIDENCE_RESCUE"
                geometric_relation = "CONFLICT"
            elif overlap["overlaps"]:
                relation = "CONFLICT"
                reason_code = "A95_ENVELOPES_OVERLAP"
                geometric_relation = "CONFLICT"
            else:
                relation = "SUPPORT"
                reason_code = "A95_ENVELOPES_DISJOINT"
                geometric_relation = "SUPPORT"

            relations.append(
                {
                    "relation_id": f"R7.POLE_COMPARE.{candidate['pole_id']}",
                    "candidate_id": candidate["pole_id"],
                    "relation": relation,
                    "reason_code": reason_code,
                    "accepted_for_comparison": True,
                    "geometric_relation_before_field_rescue": geometric_relation,
                    "eligibility": eligibility,
                    "formula_parameters": overlap,
                    "field_rescues": rescues,
                    "evidence": {
                        "target_source_sha256": parsed_target["source_sha256"],
                        "candidate_source_sha256": candidate["source_sha256"],
                        "criterion_source_sha256": CRITERION_SOURCE_SHA256,
                    },
                }
            )

        support_count = sum(item["relation"] == "SUPPORT" for item in relations)
        conflict_count = sum(item["relation"] == "CONFLICT" for item in relations)
        unknown_count = sum(item["relation"] == "UNKNOWN" for item in relations)
        rescued_count = sum(
            item["reason_code"] == "FIELD_EVIDENCE_RESCUE" for item in relations
        )
        disjoint_count = sum(
            item["reason_code"] == "A95_ENVELOPES_DISJOINT" for item in relations
        )
        unresolved_reason_codes = {
            "INVALID_TARGET_POLE",
            "INVALID_CANDIDATE_POLE",
            "UNKNOWN_STABLE_SCOPE",
        }
        unresolved_count = sum(
            item["relation"] == "UNKNOWN"
            and item["reason_code"] in unresolved_reason_codes
            for item in relations
        )

        # One unrescued qualified overlap is sufficient to fail R7. A disjoint
        # comparison can never dilute that formal conflict into MIXED.
        if conflict_count:
            overall = "CONFLICT"
            value = 0
            outcome_reason = "UNRESCUED_YOUNGER_A95_OVERLAP"
        elif rescued_count and not unresolved_count:
            overall = "SUPPORT"
            value = 1
            outcome_reason = "ALL_OBSERVED_OVERLAPS_RESCUED_BY_FIELD_EVIDENCE"
        elif support_count and search_audit["accepted_as_complete"] and unresolved_count:
            overall = "MIXED"
            value = None
            outcome_reason = "COMPLETE_SEARCH_CONTAINS_UNRESOLVED_CANDIDATES"
        elif disjoint_count and search_audit["accepted_as_complete"]:
            overall = "SUPPORT"
            value = 1
            outcome_reason = "COMPLETE_QUALIFIED_SEARCH_FOUND_NO_A95_OVERLAP"
        elif search_audit["accepted_as_complete"] and not unresolved_count:
            overall = "SUPPORT"
            value = 1
            outcome_reason = "COMPLETE_SEARCH_FOUND_NO_ELIGIBLE_YOUNGER_POLES"
        elif support_count:
            overall = "UNKNOWN"
            value = None
            outcome_reason = "QUALIFIED_SEARCH_NOT_AUDITABLY_COMPLETE"
        else:
            overall = "UNKNOWN"
            value = None
            outcome_reason = "NO_ELIGIBLE_YOUNGER_POLE_COMPARISON"

        field_relations: list[dict[str, Any]] = []
        for constraint in constraints:
            rescued_candidates = sorted(used_rescues[constraint["constraint_id"]])
            field_relations.append(
                {
                    "relation_id": f"R7.FIELD_CONSTRAINT.{constraint['constraint_id']}",
                    "constraint_id": constraint["constraint_id"],
                    "relation": "SUPPORT" if rescued_candidates else "UNKNOWN",
                    "reason_code": (
                        "FIELD_EVIDENCE_RESCUE_APPLIED"
                        if rescued_candidates
                        else (
                            "INVALID_FIELD_CONSTRAINT"
                            if not constraint["valid"]
                            else "FIELD_CONSTRAINT_DID_NOT_RESCUE_AN_OVERLAP"
                        )
                    ),
                    "candidate_id": constraint["candidate_id"],
                    "rescued_candidate_ids": rescued_candidates,
                    "field_test": constraint["field_test"],
                    "bound_age_ma": constraint["bound_age_ma"],
                    "validation_errors": constraint["validation_errors"],
                    "evidence": {
                        "source_sha256": constraint["source_sha256"],
                        "source_hash_basis": constraint["source_hash_basis"],
                        "criterion_source_sha256": CRITERION_SOURCE_SHA256,
                    },
                }
            )

        period_scale_payload = [
            {
                "name": period.name,
                "younger_bound_ma": period.younger_bound_ma,
                "older_bound_ma": period.older_bound_ma,
                "rank": period.rank,
            }
            for period in PERIOD_SCALE
        ]
        source_hashes = {
            parsed_target["source_sha256"],
            *(candidate["source_sha256"] for candidate in parsed_candidates),
            *(constraint["source_sha256"] for constraint in constraints),
        }
        if search_audit["provided"]:
            source_hashes.add(search_audit["source_sha256"])
        return {
            "schema_version": ENGINE_VERSION,
            "indicator": "R7",
            "relation": overall,
            "decision": {
                "recommended_value": value,
                "reason_code": outcome_reason,
                "support_count": support_count,
                "conflict_count": conflict_count,
                "unknown_count": unknown_count,
                "manual_review_required": overall in {"UNKNOWN", "MIXED"},
            },
            "target": _public_pole(parsed_target),
            "candidates": [_public_pole(candidate) for candidate in parsed_candidates],
            "relations": relations,
            "field_relations": field_relations,
            "qualified_search": search_audit,
            "formula": {
                "angular_distance": (
                    "d = 2*asin(sqrt(sin^2((lat2-lat1)/2) + "
                    "cos(lat1)*cos(lat2)*sin^2((lon2-lon1)/2)))"
                ),
                "a95_envelope_overlap": "d <= min(180, A95_target + A95_candidate)",
                "candidate_younger": "age_target_ma - age_candidate_ma > age_tolerance_ma",
                "more_than_one_period": (
                    "target_period_rank - candidate_period_rank >= minimum_period_rank_gap"
                ),
                "field_rescue": (
                    "established bound_age_ma - candidate_age_ma > age_tolerance_ma"
                ),
            },
            "parameters": {
                "minimum_candidate_reliability_R": self.config.minimum_candidate_reliability_R,
                "minimum_period_rank_gap": self.config.minimum_period_rank_gap,
                "angle_tolerance_deg": self.config.angle_tolerance_deg,
                "age_tolerance_ma": self.config.age_tolerance_ma,
                "period_boundary_tolerance_ma": self.config.period_boundary_tolerance_ma,
                "period_scale_version": PERIOD_SCALE_VERSION,
                "period_scale_sha256": _sha256(period_scale_payload),
            },
            "provenance": {
                "criterion_source": "user_supplied_reliability_criterion_7",
                "criterion_source_sha256": CRITERION_SOURCE_SHA256,
                "input_sha256": _sha256(input_payload),
                "evidence_source_sha256": sorted(source_hashes),
                "supervision": "none",
                "uses_answer_lookup": False,
                "forbidden_input_fields": ["REFNO", "RESULTNO", "*label*"],
            },
        }


def evaluate_r7_pole_overlap(
    target: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    field_test_age_constraints: Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    qualified_search: Mapping[str, Any] | None = None,
    *,
    config: R7ReasonerConfig | None = None,
) -> dict[str, Any]:
    """Convenience function for one label-free R7 evaluation."""

    return R7PoleOverlapReasoner(config).evaluate(
        target,
        candidates,
        field_test_age_constraints,
        qualified_search,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate criterion R7 from structured pole data without labels."
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "JSON object with target, candidates, and optional "
            "field_test_age_constraints and qualified_search"
        ),
    )
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise R7InputBoundaryError("input JSON root must be an object")
    result = evaluate_r7_pole_overlap(
        payload.get("target"),
        payload.get("candidates", []),
        payload.get("field_test_age_constraints"),
        payload.get("qualified_search"),
    )
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
