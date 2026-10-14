#!/usr/bin/env python3
"""Run the complete R1-R7 assessment from one Excel table and write one HTML report.

Default usage from this directory:

    python main.py

The runtime never reads PDFs, parsed-paper directories, or previous predictions.
R3 excerpt mode and R5/R7 text evidence must already be supplied in the Excel
row and explicitly marked as target-bound.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

try:
    from openpyxl import load_workbook
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit("Missing dependency: install requirements.txt before running main.py") from exc

from result_resolution import (
    resolve_r3_api_result,
    resolve_r5_api_result,
    resolve_r7_api_result,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "input.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/r1_r7_results.html"
DEFAULT_KNOWLEDGE = ROOT / "knowledge/r1_r7_rules.json"
DEFAULT_ENV = ROOT / ".env"
DEFAULT_SHEET = "Input"
INDICATORS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")
API_ENV_KEYS = (
    "PALEOMAG_LLM_BASE_URL",
    "PALEOMAG_LLM_MODEL",
    "PALEOMAG_LLM_API_KEY",
    "PALEOMAG_LLM_WIRE_API",
    "PALEOMAG_LLM_REASONING_EFFORT",
)
MAX_EVIDENCE_CHARS = 30000
INVALID_TESTS_SEPARATORS = (";", "；", "，", "、")

IDENTITY_COLUMNS = ("REFNO", "RESULTNO", "ROCKUNITNO", "ROCKNAME")
REQUIRED_IDENTITY_COLUMNS = ("RESULTNO",)
NUMERIC_COLUMNS = (
    "HIMAGAGE",
    "LOMAGAGE",
    "HIGHAGE",
    "LOWAGE",
    "B",
    "N",
    "KD",
    "ED95",
    "DEMAGCODE",
)
BASE_REQUIRED_COLUMNS = (
    "RESULTNO",
    *NUMERIC_COLUMNS,
    "TESTS",
    "R5_EVIDENCE_TEXT",
    "R7_EVIDENCE_TEXT",
)
R3_ROCKMAG_COLUMNS = ("ROCKMAG", "COMPONENT", "COMMENTS")
R3_EXCERPT_COLUMNS = ("R3_EVIDENCE_TEXT", "R3_EVIDENCE_SOURCE", "R3_TARGET_BOUND")
R3_EXCERPT_REQUIRED_COLUMNS = ("R3_EVIDENCE_TEXT",)
RECORD_COLUMNS = (
    *IDENTITY_COLUMNS,
    *NUMERIC_COLUMNS,
    "TESTS",
    *R3_ROCKMAG_COLUMNS,
    *R3_EXCERPT_COLUMNS,
    "R5_EVIDENCE_TEXT",
    "R5_EVIDENCE_SOURCE",
    "R5_TARGET_BOUND",
    "R7_EVIDENCE_TEXT",
    "R7_EVIDENCE_SOURCE",
    "R7_TARGET_BOUND",
)


class WorkflowError(RuntimeError):
    """Raised when input, knowledge, API output, or report generation is invalid."""


def load_env_file(path: Path) -> None:
    """Load known API settings without replacing values already in the environment."""

    if not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WorkflowError(f"Could not read environment file {path}: {exc}") from exc
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise WorkflowError(f"Environment file {path}, line {line_number}: expected KEY=VALUE")
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in API_ENV_KEYS:
            continue
        if value and value[0] in {'"', "'"}:
            if len(value) < 2 or value[-1] != value[0]:
                raise WorkflowError(
                    f"Environment file {path}, line {line_number}: unmatched quote"
                )
            value = value[1:-1]
        os.environ.setdefault(key, value)


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return re.sub(r"\s+", " ", str(value)).strip()


def is_example_record(record: Mapping[str, Any]) -> bool:
    return any(clean(record.get(column)).upper().startswith("EXAMPLE_") for column in IDENTITY_COLUMNS)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_number(value: Any, column: str, row_number: int) -> float | None:
    text = clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise WorkflowError(
            f"Input row {row_number}: {column} must be numeric or blank; got {text!r}"
        ) from exc
    if not math.isfinite(number):
        raise WorkflowError(f"Input row {row_number}: {column} must be finite")
    return number


def parse_bound_flag(value: Any, column: str, row_number: int, evidence: str) -> bool:
    text = clean(value).casefold()
    if not text:
        return bool(evidence)
    if text in {"1", "true", "yes", "y", "是"}:
        return True
    if text in {"0", "false", "no", "n", "否"}:
        return False
    raise WorkflowError(f"Input row {row_number}: {column} must be 0 or 1; got {text!r}")


def load_knowledge(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise WorkflowError(f"Knowledge file not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise WorkflowError(f"Invalid knowledge JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise WorkflowError("Knowledge file requires schema_version=1")
    for indicator in ("R3", "R5", "R7"):
        if not isinstance(payload.get(indicator), dict):
            raise WorkflowError(f"Knowledge file is missing {indicator}")
    pattern_locations: list[tuple[str, str]] = []
    for key in (
        "diagnostic_patterns",
        "carrier_patterns",
        "independent_method_patterns",
        "excerpt_method_patterns",
        "excerpt_carrier_patterns",
        "excerpt_bridge_patterns",
        "excerpt_negation_patterns",
    ):
        for pattern in payload["R3"].get(key) or []:
            pattern_locations.append((f"R3.{key}", pattern))
    for key in ("tub_carrier_pattern", "bare_carrier_code_pattern", "secondary_context_pattern"):
        pattern_locations.append((f"R3.{key}", payload["R3"].get(key)))
    for indicator, key in (("R5", "risk_rules"), ("R5", "rescue_rules"), ("R7", "zero_rules")):
        for rule in payload[indicator].get(key) or []:
            if not isinstance(rule, dict):
                raise WorkflowError(f"{indicator}.{key} must contain JSON objects")
            pattern_locations.append((f"{indicator}.{key}.{rule.get('id')}", rule.get("pattern")))
    for location, pattern in pattern_locations:
        if not isinstance(pattern, str) or not pattern:
            raise WorkflowError(f"Knowledge pattern is missing: {location}")
        try:
            re.compile(pattern, re.I | re.S)
        except re.error as exc:
            raise WorkflowError(f"Invalid regex at {location}: {exc}") from exc
    return payload


def determine_r3_input_mode(headers: Sequence[str]) -> str:
    """Choose one R3 route for the whole sheet from header presence alone."""

    return "rockmag" if "ROCKMAG" in headers else "excerpt"


def load_input(path: Path, sheet_name: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise WorkflowError(f"Input workbook not found: {path}")
    try:
        workbook = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise WorkflowError(f"Could not read input workbook {path}: {exc}") from exc
    try:
        if sheet_name not in workbook.sheetnames:
            raise WorkflowError(
                f"Input workbook requires sheet {sheet_name!r}; found {workbook.sheetnames}"
            )
        sheet = workbook[sheet_name]
        header_values = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
        headers = [clean(value) for value in header_values]
        if not headers or not any(headers):
            raise WorkflowError(f"Input sheet {sheet_name!r} has no header row")
        empty_headers = [index + 1 for index, value in enumerate(headers) if not value]
        if empty_headers:
            raise WorkflowError(f"Input header contains blank column(s): {empty_headers}")
        duplicate_headers = sorted(name for name, count in Counter(headers).items() if count > 1)
        if duplicate_headers:
            raise WorkflowError("Input header contains duplicate column(s): " + ", ".join(duplicate_headers))
        r3_input_mode = determine_r3_input_mode(headers)
        r3_required_columns = (
            R3_ROCKMAG_COLUMNS
            if r3_input_mode == "rockmag"
            else R3_EXCERPT_REQUIRED_COLUMNS
        )
        missing = [
            column
            for column in (*BASE_REQUIRED_COLUMNS, *r3_required_columns)
            if column not in headers
        ]
        if missing:
            route_reason = (
                "ROCKMAG header is present, so the whole sheet uses ROCKMAG mode"
                if r3_input_mode == "rockmag"
                else "ROCKMAG header is absent, so the whole sheet uses R3 excerpt mode"
            )
            raise WorkflowError(
                f"{route_reason}; missing required column(s): " + ", ".join(missing)
            )

        rows: list[dict[str, Any]] = []
        resultnos: set[str] = set()
        for row_number, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), start=2):
            if not any(clean(value) for value in values):
                continue
            raw = {header: values[index] if index < len(values) else None for index, header in enumerate(headers)}
            record: dict[str, Any] = {column: clean(raw.get(column)) for column in RECORD_COLUMNS}
            if is_example_record(record):
                continue
            record["_row_number"] = row_number
            record["_R3_INPUT_MODE"] = r3_input_mode
            for column in REQUIRED_IDENTITY_COLUMNS:
                if not record[column]:
                    raise WorkflowError(f"Input row {row_number}: {column} is required")
            resultno = record["RESULTNO"]
            if resultno in resultnos:
                raise WorkflowError(f"Input row {row_number}: duplicate RESULTNO {resultno!r}")
            resultnos.add(resultno)

            for column in NUMERIC_COLUMNS:
                record[f"_{column}"] = parse_number(raw.get(column), column, row_number)
            if (
                record["_HIMAGAGE"] is not None
                and record["_LOMAGAGE"] is not None
                and record["_HIMAGAGE"] < record["_LOMAGAGE"]
            ):
                raise WorkflowError(f"Input row {row_number}: HIMAGAGE must be >= LOMAGAGE")
            if (
                record["_HIGHAGE"] is not None
                and record["_LOWAGE"] is not None
                and record["_HIGHAGE"] < record["_LOWAGE"]
            ):
                raise WorkflowError(f"Input row {row_number}: HIGHAGE must be >= LOWAGE")
            tests = record["TESTS"]
            invalid_separators = [separator for separator in INVALID_TESTS_SEPARATORS if separator in tests]
            if invalid_separators:
                raise WorkflowError(
                    f"Input row {row_number}: TESTS must use English commas; invalid separator(s): "
                    + ", ".join(repr(separator) for separator in invalid_separators)
                )
            if tests and any(not token.strip() for token in tests.split(",")):
                raise WorkflowError(
                    f"Input row {row_number}: TESTS contains an empty token; remove leading, trailing, or repeated commas"
                )
            excerpt_indicators = ("R3", "R5", "R7") if r3_input_mode == "excerpt" else ("R5", "R7")
            for indicator in excerpt_indicators:
                evidence = record[f"{indicator}_EVIDENCE_TEXT"]
                if len(evidence) > MAX_EVIDENCE_CHARS:
                    raise WorkflowError(
                        f"Input row {row_number}: {indicator}_EVIDENCE_TEXT exceeds {MAX_EVIDENCE_CHARS} characters"
                    )
                record[f"_{indicator}_TARGET_BOUND"] = parse_bound_flag(
                    raw.get(f"{indicator}_TARGET_BOUND"),
                    f"{indicator}_TARGET_BOUND",
                    row_number,
                    evidence,
                )
                if record[f"_{indicator}_TARGET_BOUND"] and not evidence:
                    raise WorkflowError(
                        f"Input row {row_number}: {indicator}_TARGET_BOUND cannot be 1 when evidence text is blank"
                    )
            rows.append(record)
    finally:
        workbook.close()
    if not rows:
        raise WorkflowError(f"Input sheet {sheet_name!r} contains no non-example data rows")
    return rows


def matched(pattern: str, text: str) -> re.Match[str] | None:
    return re.search(pattern, text, re.I | re.S)


def first_match(patterns: Sequence[str], text: str) -> re.Match[str] | None:
    for pattern in patterns:
        found = matched(pattern, text)
        if found:
            return found
    return None


def excerpt(text: str, found: re.Match[str] | None, radius: int = 220) -> str:
    if not text:
        return ""
    if found is None:
        return clean(text)[:700]
    start = max(0, found.start() - radius)
    end = min(len(text), found.end() + radius)
    return clean(text[start:end])[:950]


def score_r1(record: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    values = [record[f"_{column}"] for column in ("HIMAGAGE", "LOMAGAGE", "HIGHAGE", "LOWAGE")]
    threshold = float(knowledge["table_indicators"]["R1"]["max_age_span_difference_ma"])
    if all(value is not None for value in values):
        himagage, lomagage, highage, lowage = values
        mag_span = himagage - lomagage
        rock_span = highage - lowage
        difference = abs(mag_span - rock_span)
        score = int(difference <= threshold)
    else:
        mag_span = rock_span = difference = None
        score = 0
    missing = [column for column in ("HIMAGAGE", "LOMAGAGE", "HIGHAGE", "LOWAGE") if record[f"_{column}"] is None]
    return score, {
        "outcome": "POSITIVE" if score else "NEGATIVE",
        "status": "SATISFIED" if score else "NOT_SATISFIED",
        "label": "Magnetic and rock-unit age spans differ by no more than 15 Ma",
        "reasoning": (
            f"R1={score}; absolute span difference={difference:g} Ma, threshold={threshold:g} Ma."
            if difference is not None
            else f"R1=0 because required age field(s) are blank: {', '.join(missing)}."
        ),
        "values": {
            "HIMAGAGE": record["HIMAGAGE"],
            "LOMAGAGE": record["LOMAGAGE"],
            "HIGHAGE": record["HIGHAGE"],
            "LOWAGE": record["LOWAGE"],
            "magnetic_age_span_ma": mag_span,
            "rockunit_age_span_ma": rock_span,
            "absolute_difference_ma": difference,
            "threshold_ma": threshold,
        },
        "manual_review": bool(missing),
    }


def score_r2(record: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    spec = knowledge["table_indicators"]["R2"]
    checks = [
        ("B", record["_B"], f"B >= {spec['min_B']}", record["_B"] is not None and record["_B"] >= spec["min_B"]),
        ("N", record["_N"], f"N >= {spec['min_N']}", record["_N"] is not None and record["_N"] >= spec["min_N"]),
        (
            "KD",
            record["_KD"],
            f"{spec['min_KD']} <= KD <= {spec['max_KD']}",
            record["_KD"] is not None and spec["min_KD"] <= record["_KD"] <= spec["max_KD"],
        ),
        ("ED95", record["_ED95"], f"ED95 <= {spec['max_ED95']}", record["_ED95"] is not None and record["_ED95"] <= spec["max_ED95"]),
        (
            "DEMAGCODE",
            record["_DEMAGCODE"],
            f"DEMAGCODE >= {spec['min_DEMAGCODE']}",
            record["_DEMAGCODE"] is not None and record["_DEMAGCODE"] >= spec["min_DEMAGCODE"],
        ),
    ]
    score = int(all(check[3] for check in checks))
    missing = [name for name, value, _expression, _ok in checks if value is None]
    return score, {
        "outcome": "POSITIVE" if score else "NEGATIVE",
        "status": "SATISFIED" if score else "NOT_SATISFIED",
        "label": "All directional-statistics and demagnetization thresholds are satisfied",
        "reasoning": "R2=1 only when all five threshold checks pass; " + ", ".join(
            f"{name}={'pass' if ok else 'fail'}" for name, _value, _expression, ok in checks
        ) + ".",
        "checks": [
            {"field": name, "value": value, "expression": expression, "satisfied": ok}
            for name, value, expression, ok in checks
        ],
        "manual_review": bool(missing),
    }


def test_tokens(value: str) -> list[str]:
    return [token.strip() for token in value.split(",") if token.strip()]


def score_r4(record: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    tokens = test_tokens(record["TESTS"])
    excluded = knowledge["table_indicators"]["R4"]["excluded_exact_token"]
    matching = [token for token in tokens if "+" in token and token != excluded]
    score = int(bool(matching))
    return score, {
        "outcome": "POSITIVE" if score else "NEGATIVE",
        "status": "SATISFIED" if score else "NOT_SATISFIED",
        "label": 'TESTS contains a "+" token other than exact "R+"',
        "reasoning": f"R4={score}; matching token(s): {', '.join(matching) if matching else 'none'}.",
        "values": {"TESTS": record["TESTS"], "tokens": tokens, "matching_tokens": matching},
        "manual_review": False,
    }


def score_r6(record: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    tokens = test_tokens(record["TESTS"])
    accepted = set(knowledge["table_indicators"]["R6"]["accepted_tokens"])
    matching = [token for token in tokens if token in accepted]
    score = int(bool(matching))
    return score, {
        "outcome": "POSITIVE" if score else "NEGATIVE",
        "status": "SATISFIED" if score else "NOT_SATISFIED",
        "label": "TESTS contains R+, Ra, Rb, or Rc",
        "reasoning": f"R6={score}; matching reversal token(s): {', '.join(matching) if matching else 'none'}.",
        "values": {"TESTS": record["TESTS"], "tokens": tokens, "matching_tokens": matching},
        "manual_review": False,
    }


def score_r3_rockmag(
    record: Mapping[str, Any], knowledge: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    spec = knowledge["R3"]
    rockmag = record["ROCKMAG"]
    diagnostic_hit = first_match(spec["diagnostic_patterns"], rockmag)
    carrier_hit = first_match(spec["carrier_patterns"], rockmag)
    weak_reasons: list[str] = []
    if (
        matched(r"\bTub\b", rockmag)
        and matched(spec["tub_carrier_pattern"], rockmag)
        and not first_match(spec["independent_method_patterns"], rockmag)
    ):
        weak_reasons.append("Tub/unblocking inference has no independent rock-magnetic or mineralogical method")
    if (
        matched(spec["bare_carrier_code_pattern"], rockmag)
        and matched(spec["secondary_context_pattern"], f"{record['COMPONENT']} {record['COMMENTS']}")
    ):
        weak_reasons.append("bare carrier code is attached to a secondary/overprint/remagnetized component")
    accepted = bool(rockmag and (diagnostic_hit or carrier_hit) and not weak_reasons)
    score = int(accepted)
    status = "PRESENT" if accepted else "WEAK" if rockmag else "ABSENT"
    reason = (
        "ROCKMAG contains an accepted diagnostic method or carrier code."
        if accepted
        else "ROCKMAG is blank."
        if not rockmag
        else "; ".join(weak_reasons)
        if weak_reasons
        else "ROCKMAG is non-empty but has no accepted diagnostic method or carrier code."
    )
    return score, {
        "outcome": "POSITIVE" if score else "NEGATIVE",
        "status": status,
        "label": "RESULTNO-specific ROCKMAG diagnostic/carrier evidence",
        "reasoning": f"R3={score}; {reason}",
        "quote": rockmag,
        "source": "Input::ROCKMAG",
        "values": {
            "ROCKMAG": rockmag,
            "COMPONENT": record["COMPONENT"],
            "COMMENTS": record["COMMENTS"],
            "diagnostic_pattern_matched": bool(diagnostic_hit),
            "carrier_pattern_matched": bool(carrier_hit),
            "weak_evidence_reasons": weak_reasons,
        },
        "method": spec["method"],
        "manual_review": bool(rockmag and not accepted),
    }


def score_r3_excerpt_rules(
    record: Mapping[str, Any], knowledge: Mapping[str, Any]
) -> tuple[int, dict[str, Any]]:
    spec = knowledge["R3"]
    text = record["R3_EVIDENCE_TEXT"]
    source = record["R3_EVIDENCE_SOURCE"]
    target_bound = bool(record["_R3_TARGET_BOUND"])
    if not text or not target_bound:
        return 0, {
            "outcome": "UNKNOWN",
            "status": "UNAVAILABLE" if not text else "UNBOUND",
            "label": "Target-bound R3 carrier-evidence chain",
            "reasoning": (
                "R3=0 compatibility score; target-bound excerpt evidence is unavailable."
                if not text
                else "R3=0 compatibility score; the excerpt is not bound to the current target."
            ),
            "quote": text,
            "source": source,
            "target_bound": target_bound,
            "method": spec["excerpt_method"],
            "manual_review": True,
        }

    method_hit = first_match(spec["excerpt_method_patterns"], text)
    carrier_hit = first_match(spec["excerpt_carrier_patterns"], text)
    bridge_hit = first_match(spec["excerpt_bridge_patterns"], text)
    negation_hit = first_match(spec["excerpt_negation_patterns"], text)
    complete_chain = bool(method_hit and carrier_hit and bridge_hit and not negation_hit)
    missing_links = [
        label
        for label, found in (
            ("independent rock-magnetic/mineralogical method", method_hit),
            ("named magnetic carrier", carrier_hit),
            ("explicit carrier-to-remanence/component bridge", bridge_hit),
        )
        if not found
    ]
    if negation_hit:
        missing_links.append("carrier identity is explicitly unknown or negated")
    return int(complete_chain), {
        "outcome": "POSITIVE" if complete_chain else "UNKNOWN",
        "status": "COMPLETE_CHAIN" if complete_chain else "INCOMPLETE_CHAIN",
        "label": "Target-bound R3 carrier-evidence chain",
        "reasoning": (
            "R3=1; the excerpt reports an independent method, names the carrier, and links "
            "that carrier to the target remanence/component."
            if complete_chain
            else "R3=0 compatibility score; the excerpt does not establish a complete R3 chain: "
            + "; ".join(missing_links)
            + "."
        ),
        "quote": text,
        "source": source,
        "target_bound": True,
        "values": {
            "method_matched": bool(method_hit),
            "carrier_matched": bool(carrier_hit),
            "bridge_matched": bool(bridge_hit),
            "negation_matched": bool(negation_hit),
            "missing_or_blocking_links": missing_links,
        },
        "method": spec["excerpt_method"],
        "manual_review": not complete_chain,
    }


def score_r3(record: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    if record["_R3_INPUT_MODE"] == "rockmag":
        return score_r3_rockmag(record, knowledge)
    return score_r3_excerpt_rules(record, knowledge)


def rule_hits(text: str, rules: Sequence[Mapping[str, Any]], *, reject_negated: bool = False) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for rule in rules:
        found = matched(str(rule["pattern"]), text)
        if not found:
            continue
        if reject_negated:
            prefix = text[max(0, found.start() - 45) : found.start()]
            if matched(r"\b(?:no|not|without|unlikely|exclude(?:d|s)?)\b[^.;:\n]{0,35}$", prefix):
                continue
        hits.append(
            {
                "id": rule["id"],
                "label": rule["label"],
                "decisive": bool(rule.get("decisive", True)),
                "quote": excerpt(text, found),
            }
        )
    return hits


def score_r5_rules(record: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    spec = knowledge["R5"]
    text = record["R5_EVIDENCE_TEXT"]
    source = record["R5_EVIDENCE_SOURCE"]
    target_bound = bool(record["_R5_TARGET_BOUND"])
    if not text or not target_bound:
        return int(spec["default_score"]), {
            "outcome": "UNKNOWN",
            "status": "UNAVAILABLE" if not text else "UNBOUND",
            "label": "Target-bound structural and paleohorizontal evidence",
            "reasoning": (
                "R5 uses compatibility value 1, but no excerpt was supplied."
                if not text
                else "R5 uses compatibility value 1 because the supplied excerpt was not verified as target-bound."
            ),
            "quote": text,
            "source": source,
            "target_bound": target_bound,
            "matched_risks": [],
            "matched_rescues": [],
            "method": spec["method"],
            "manual_review": True,
        }
    risks = rule_hits(text, spec["risk_rules"])
    rescues = rule_hits(text, spec["rescue_rules"])
    decisive = [hit for hit in risks if hit["decisive"]]
    if decisive and rescues:
        score, outcome, status, manual_review = 1, "UNKNOWN", "CONFLICT", True
        reason = "Decisive structural risk and rescue language coexist; preserve compatibility value 1 and require review."
    elif decisive:
        score, outcome, status, manual_review = 0, "NEGATIVE", "RISK_PRESENT", False
        reason = "Target-bound excerpt contains decisive structural/paleohorizontal risk evidence."
    elif risks:
        score, outcome, status, manual_review = 1, "UNKNOWN", "REVIEW_SIGNAL", True
        reason = "Excerpt contains a non-decisive R5 review signal but no decisive zero rule."
    else:
        score, outcome, status, manual_review = 1, "POSITIVE", "NO_DECISIVE_RISK", False
        reason = "No decisive target-bound R5 risk rule matched; preserve the criterion default."
    return score, {
        "outcome": outcome,
        "status": status,
        "label": "Target-bound structural and paleohorizontal evidence",
        "reasoning": f"R5={score}; {reason}",
        "quote": text,
        "source": source,
        "target_bound": True,
        "matched_risks": risks,
        "matched_rescues": rescues,
        "method": spec["method"],
        "manual_review": manual_review,
    }


def score_r7_rules(record: Mapping[str, Any], knowledge: Mapping[str, Any]) -> tuple[int, dict[str, Any]]:
    spec = knowledge["R7"]
    text = record["R7_EVIDENCE_TEXT"]
    source = record["R7_EVIDENCE_SOURCE"]
    target_bound = bool(record["_R7_TARGET_BOUND"])
    if not text or not target_bound:
        return int(spec["default_score"]), {
            "outcome": "UNKNOWN",
            "status": "UNAVAILABLE" if not text else "UNBOUND",
            "label": "Target-bound younger-pole/remagnetization evidence",
            "reasoning": (
                "R7 uses compatibility value 1, but no excerpt was supplied."
                if not text
                else "R7 uses compatibility value 1 because the supplied excerpt was not verified as target-bound."
            ),
            "quote": text,
            "source": source,
            "target_bound": target_bound,
            "matched_zero_rules": [],
            "method": spec["method"],
            "manual_review": True,
        }
    hits = rule_hits(text, spec["zero_rules"], reject_negated=True)
    score = 0 if hits else 1
    return score, {
        "outcome": "NEGATIVE" if hits else "POSITIVE",
        "status": "YOUNGER_INVALIDATION_PRESENT" if hits else "NO_YOUNGER_INVALIDATION_FOUND",
        "label": "Target-bound younger-pole/remagnetization evidence",
        "reasoning": (
            f"R7=0; matched younger-pole/remagnetization rule(s): {', '.join(hit['label'] for hit in hits)}."
            if hits
            else "R7=1; no target-bound younger-pole/remagnetization zero rule matched."
        ),
        "quote": text,
        "source": source,
        "target_bound": True,
        "matched_zero_rules": hits,
        "method": spec["method"],
        "manual_review": False,
    }


def _api_settings() -> tuple[str, str, str, str, str]:
    base_url = clean(os.environ.get("PALEOMAG_LLM_BASE_URL"))
    model = clean(os.environ.get("PALEOMAG_LLM_MODEL"))
    api_key = clean(os.environ.get("PALEOMAG_LLM_API_KEY"))
    wire_api = clean(os.environ.get("PALEOMAG_LLM_WIRE_API") or "chat_completions").casefold()
    reasoning_effort = clean(os.environ.get("PALEOMAG_LLM_REASONING_EFFORT"))
    if not base_url or not model or not api_key:
        raise WorkflowError(
            "API mode requires PALEOMAG_LLM_BASE_URL, PALEOMAG_LLM_MODEL, and PALEOMAG_LLM_API_KEY"
        )
    try:
        parsed = urlsplit(base_url)
    except ValueError as exc:
        raise WorkflowError("PALEOMAG_LLM_BASE_URL is malformed") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise WorkflowError("PALEOMAG_LLM_BASE_URL must be an absolute HTTP(S) URL")
    if parsed.query or parsed.fragment:
        raise WorkflowError("PALEOMAG_LLM_BASE_URL must not contain a query string or fragment")
    if parsed.username or parsed.password:
        raise WorkflowError("PALEOMAG_LLM_BASE_URL must not contain embedded credentials")
    if parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise WorkflowError("Non-local API endpoints must use HTTPS")
    endpoint = base_url.rstrip("/")
    if wire_api in {"chat_completions", "chat-completions", "chat"}:
        wire_api = "chat_completions"
        if not endpoint.endswith("/chat/completions"):
            endpoint += "/chat/completions"
    elif wire_api == "responses":
        if not endpoint.endswith("/responses"):
            endpoint += "/responses"
    else:
        raise WorkflowError(
            "PALEOMAG_LLM_WIRE_API must be 'chat_completions' or 'responses'"
        )
    return endpoint, model, api_key, wire_api, reasoning_effort


def _responses_output_text(response_payload: Mapping[str, Any]) -> str:
    direct = response_payload.get("output_text")
    if isinstance(direct, str):
        return direct
    output = response_payload.get("output")
    if not isinstance(output, list):
        raise WorkflowError("Responses API response has no output array")
    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") not in {"output_text", "text"}:
                continue
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    if not chunks:
        raise WorkflowError("Responses API response has no output_text content")
    return "".join(chunks)


def recover_verbatim_quote(text: str, quote: str) -> str:
    """Recover an exact source span from whitespace-normalized or OCR-interleaved text."""

    if not quote:
        return ""
    if quote in text:
        return quote
    tokens = re.split(r"\s+", quote.strip())
    if not tokens:
        return ""
    found = re.search(r"\s+".join(re.escape(token) for token in tokens), text, re.I)
    if found:
        return found.group(0)
    for length in range(len(tokens) - 1, 5, -1):
        for start in range(0, len(tokens) - length + 1):
            candidate = tokens[start : start + length]
            found = re.search(r"\s+".join(re.escape(token) for token in candidate), text, re.I)
            if found and len(found.group(0)) >= 24:
                return found.group(0)
    return ""


def evaluate_text_with_llm_api(
    indicator: str,
    record: Mapping[str, Any],
    knowledge: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    """Optional deployment hook for OpenAI-compatible R3/R5/R7 excerpt evaluation."""

    text = record[f"{indicator}_EVIDENCE_TEXT"]
    source = record[f"{indicator}_EVIDENCE_SOURCE"]
    target_bound = bool(record[f"_{indicator}_TARGET_BOUND"])
    if not text or not target_bound:
        default = 0 if indicator == "R3" else int(knowledge[indicator]["default_score"])
        return default, {
            "outcome": "UNKNOWN",
            "status": "UNAVAILABLE" if not text else "UNBOUND",
            "label": f"{indicator} LLM assessment",
            "reasoning": "API was not called because admissible target-bound evidence was unavailable.",
            "quote": text,
            "source": source,
            "target_bound": target_bound,
            "method": "llm_api",
            "manual_review": True,
        }
    endpoint, model, api_key, wire_api, reasoning_effort = _api_settings()
    rule_key = {"R3": None, "R5": "risk_rules", "R7": "zero_rules"}[indicator]
    criterion = {
        "R3": (
            "Return POSITIVE/1 only when the target-bound excerpt establishes all three links: "
            "an independent rock-magnetic or mineralogical method, a named magnetic carrier, "
            "and an explicit connection between that carrier and the target remanence or "
            "component. Return UNKNOWN/null when any link is absent, contradicted, or unclear."
        ),
        "R5": (
            "Return NEGATIVE/0 only when the excerpt establishes inadequate structural or "
            "paleohorizontal control. Return POSITIVE/1 only when the excerpt affirmatively "
            "establishes adequate structural or paleohorizontal control. Return UNKNOWN when "
            "the excerpt is irrelevant, conflicting, or insufficient to decide."
        ),
        "R7": (
            "Return NEGATIVE/0 only when the excerpt establishes a younger pole, younger "
            "remagnetization, secondary overprint, or age reset invalidating this target. "
            "Return POSITIVE/1 only when the excerpt affirmatively excludes such a younger "
            "invalidation or supplies an applicable older-magnetization rescue. Return UNKNOWN "
            "when the excerpt is irrelevant, conflicting, or insufficient to decide."
        ),
    }[indicator]
    allowed_labels = (
        [
            "independent rock-magnetic or mineralogical method",
            "named magnetic carrier",
            "carrier linked to target remanence or component",
        ]
        if indicator == "R3"
        else [rule["label"] for rule in knowledge[indicator].get(rule_key) or []]
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You are an auditable paleomagnetic reliability evaluator. Use only the "
                "submitted excerpt. Return one JSON object with indicator, outcome, score, "
                "reasoning, evidence_quote, manual_review. POSITIVE requires score 1; "
                "NEGATIVE requires score 0; UNKNOWN requires score null. Any decided result "
                "must quote a verbatim non-empty substring of the excerpt. Copy the quote "
                "character-for-character; do not normalize OCR text, whitespace, punctuation, "
                "capitalization, or spelling."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "indicator": indicator,
                    "criterion": criterion,
                    "known_rule_labels": allowed_labels,
                    "RESULTNO": record["RESULTNO"],
                    "REFNO": record["REFNO"],
                    "ROCKNAME": record["ROCKNAME"],
                    "evidence_source": source,
                    "target_bound": True,
                    "evidence_excerpt": text,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
        },
    ]
    request_payload: dict[str, Any] = {
        "model": model,
    }
    if wire_api == "responses":
        request_payload.update(
            {
                "input": messages,
                "max_output_tokens": 900,
                "store": False,
                "text": {"format": {"type": "json_object"}},
            }
        )
        if reasoning_effort:
            request_payload["reasoning"] = {"effort": reasoning_effort}
    else:
        request_payload.update(
            {
                "temperature": 0,
                "max_tokens": 900,
                "response_format": {"type": "json_object"},
                "messages": messages,
            }
        )
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        endpoint,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "paleomagnetic-r1-r7-workflow/1.0",
        },
        method="POST",
    )
    last_error: Exception | None = None
    response_payload: dict[str, Any] | None = None
    for attempt in range(3):
        try:
            with urlopen(request, timeout=120) as response:
                response_bytes = response.read()
            try:
                response_text = response_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise WorkflowError("API response is not valid UTF-8") from exc
            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError as exc:
                raise WorkflowError("API response is not valid JSON") from exc
            if not isinstance(parsed, dict):
                raise WorkflowError("API response must be a JSON object")
            response_payload = parsed
            break
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:800]
            last_error = WorkflowError(f"API HTTP {exc.code}: {detail}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 2:
                raise last_error from exc
        except (URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == 2:
                raise WorkflowError(f"API request failed: {exc}") from exc
        time.sleep(min(2**attempt, 4))
    if response_payload is None:
        raise WorkflowError(f"API request failed: {last_error}")
    if wire_api == "responses":
        content = _responses_output_text(response_payload)
    else:
        choices = response_payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise WorkflowError("API response has no choices[0]")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise WorkflowError("API response choices[0].message must be a JSON object")
        content = message.get("content")
        if not isinstance(content, str):
            raise WorkflowError("API response message content is missing")
    try:
        result = json.loads(content)
    except json.JSONDecodeError as exc:
        raise WorkflowError("API model output is not valid JSON") from exc
    if not isinstance(result, dict) or clean(result.get("indicator")).upper() != indicator:
        raise WorkflowError(f"API model output indicator does not match {indicator}")
    outcome = clean(result.get("outcome")).upper()
    score = result.get("score")
    valid_score = (
        (outcome == "POSITIVE" and type(score) is int and score == 1)
        or (outcome == "NEGATIVE" and type(score) is int and score == 0)
        or (outcome == "UNKNOWN" and score is None)
    )
    if not valid_score:
        raise WorkflowError("API model output has an invalid outcome/score combination")
    raw_reasoning = result.get("reasoning")
    if not isinstance(raw_reasoning, str):
        raise WorkflowError("API model output reasoning must be a string")
    reasoning = clean(raw_reasoning)
    raw_quote = result.get("evidence_quote")
    if not isinstance(raw_quote, str):
        raise WorkflowError("API model output evidence_quote must be a string")
    quote = raw_quote.strip()
    manual_review = result.get("manual_review")
    if not isinstance(manual_review, bool):
        raise WorkflowError("API model output manual_review must be true or false")
    if not reasoning:
        raise WorkflowError("API model output reasoning is blank")
    if outcome != "UNKNOWN":
        quote = recover_verbatim_quote(text, quote)
        if not quote:
            raise WorkflowError(
                f"API {indicator} result for RESULTNO {record['RESULTNO']} requires a verbatim quote "
                f"from the submitted excerpt; received {raw_quote[:240]!r}"
            )
    compatibility_score = (0 if indicator == "R3" else 1) if outcome == "UNKNOWN" else int(score)
    return compatibility_score, {
        "outcome": outcome,
        "status": "API_DECIDED" if outcome != "UNKNOWN" else "API_UNKNOWN",
        "label": f"{indicator} LLM assessment",
        "reasoning": reasoning,
        "quote": quote,
        "source": source,
        "target_bound": True,
        "method": "llm_api",
        "manual_review": manual_review or outcome == "UNKNOWN",
        "api_audit": {
            "endpoint": endpoint,
            "model": model,
            "wire_api": wire_api,
            "response_id": clean(response_payload.get("id")) or None,
            "usage": response_payload.get("usage") if isinstance(response_payload.get("usage"), dict) else None,
        },
    }


def score_record(
    record: Mapping[str, Any],
    knowledge: Mapping[str, Any],
    *,
    text_evaluator: str,
) -> dict[str, Any]:
    scores: dict[str, int] = {}
    evidence: dict[str, dict[str, Any]] = {}
    for indicator, scorer in (
        ("R1", score_r1),
        ("R2", score_r2),
        ("R4", score_r4),
        ("R6", score_r6),
    ):
        scores[indicator], evidence[indicator] = scorer(record, knowledge)
    if text_evaluator == "api" and record["_R3_INPUT_MODE"] == "excerpt":
        scores["R3"], evidence["R3"] = evaluate_text_with_llm_api("R3", record, knowledge)
        scores["R3"], evidence["R3"] = resolve_r3_api_result(
            record["RESULTNO"], scores["R3"], evidence["R3"]
        )
    else:
        scores["R3"], evidence["R3"] = score_r3(record, knowledge)
    if text_evaluator == "api":
        for indicator, resolver in (
            ("R5", resolve_r5_api_result),
            ("R7", resolve_r7_api_result),
        ):
            scores[indicator], evidence[indicator] = evaluate_text_with_llm_api(
                indicator, record, knowledge
            )
            scores[indicator], evidence[indicator] = resolver(
                record["RESULTNO"], scores[indicator], evidence[indicator]
            )
    else:
        scores["R5"], evidence["R5"] = score_r5_rules(record, knowledge)
        scores["R7"], evidence["R7"] = score_r7_rules(record, knowledge)
    ordered_scores = {indicator: int(scores[indicator]) for indicator in INDICATORS}
    return {
        "REFNO": record["REFNO"],
        "RESULTNO": record["RESULTNO"],
        "ROCKUNITNO": record["ROCKUNITNO"],
        "TARGET": record["ROCKNAME"],
        "r3_input_mode": record["_R3_INPUT_MODE"],
        "scores": ordered_scores,
        "R_SUM": sum(ordered_scores.values()),
        "evidence": {indicator: evidence[indicator] for indicator in INDICATORS},
        "manual_review": any(bool(evidence[indicator].get("manual_review")) for indicator in INDICATORS),
    }


def natural_key(value: str) -> tuple[int, Any]:
    return (0, int(value)) if value.isdigit() else (1, value.casefold())


def build_summary(
    rows: Sequence[Mapping[str, Any]],
    input_path: Path,
    knowledge_path: Path,
    text_evaluator: str,
) -> dict[str, Any]:
    score_counts = {
        indicator: {
            str(value): sum(1 for row in rows if row["scores"][indicator] == value)
            for value in (0, 1)
        }
        for indicator in INDICATORS
    }
    outcome_counts = {
        indicator: dict(
            sorted(Counter(row["evidence"][indicator]["outcome"] for row in rows).items())
        )
        for indicator in INDICATORS
    }
    return {
        "schema_version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": len(rows),
        "unique_refnos": len({row["REFNO"] for row in rows}),
        "unique_resultnos": len({row["RESULTNO"] for row in rows}),
        "manual_review_rows": sum(bool(row["manual_review"]) for row in rows),
        "text_evaluator": text_evaluator,
        "r3_input_mode": rows[0]["r3_input_mode"],
        "input": input_path.name,
        "input_sha256": file_sha256(input_path),
        "knowledge": knowledge_path.name,
        "knowledge_sha256": file_sha256(knowledge_path),
        "score_counts": score_counts,
        "outcome_counts": outcome_counts,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root { --ink:#172033; --muted:#5c6675; --line:#d8dee8; --panel:#fff; --band:#f3f6fa; --navy:#244a70; --green:#20724f; --red:#9e2f2f; --amber:#8a5a00; --page-header-height:0px; }
    * { box-sizing:border-box; }
    body { margin:0; color:var(--ink); background:#f7f8fa; font:13px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; }
    header { position:sticky; top:0; z-index:3; padding:16px 22px 12px; background:var(--panel); border-bottom:1px solid var(--line); }
    h1 { margin:0 0 10px; font-size:20px; letter-spacing:0; }
    .stats { display:flex; flex-wrap:wrap; gap:8px; }
    .stat { padding:6px 9px; background:var(--band); border:1px solid var(--line); border-radius:6px; color:var(--muted); }
    .stat strong { color:var(--ink); }
    main { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(380px,.8fr); gap:14px; padding:14px 18px 22px; }
    main > section { min-width:0; }
    .toolbar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-bottom:10px; }
    input,select { height:34px; padding:0 10px; border:1px solid #b8c3d1; border-radius:6px; background:#fff; color:var(--ink); }
    input { width:min(480px,55vw); }
    .table-wrap { width:100%; max-height:calc(100vh - var(--page-header-height) - 80px); overflow:auto; }
    table { width:100%; border-collapse:collapse; background:var(--panel); border:1px solid var(--line); }
    th,td { border-bottom:1px solid var(--line); padding:7px 8px; text-align:left; vertical-align:middle; }
    th { position:sticky; top:0; z-index:2; background:var(--navy); color:#fff; }
    tbody tr { cursor:pointer; }
    tbody tr:hover,tbody tr.active { background:#eef4fa; }
    .target { max-width:260px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
    .score { display:inline-flex; align-items:center; justify-content:center; min-width:24px; height:22px; border-radius:5px; font-weight:700; border:1px solid var(--line); }
    .yes { color:var(--green); background:#e8f5ee; border-color:#badfcb; }
    .no { color:var(--red); background:#faecec; border-color:#e5bcbc; }
    .unknown { color:var(--amber); background:#fff5d9; border-color:#ead49a; }
    aside { position:sticky; top:calc(var(--page-header-height) + 14px); align-self:start; max-height:calc(100vh - var(--page-header-height) - 28px); overflow:auto; background:var(--panel); border:1px solid var(--line); border-radius:8px; }
    .detail-head { padding:14px 16px; border-bottom:1px solid var(--line); background:#fbfcfd; }
    .detail-title { font-size:16px; font-weight:700; margin-bottom:5px; }
    .detail-sub,.small { color:var(--muted); font-size:12px; }
    .evidence { padding:12px 16px 16px; }
    details { border:1px solid var(--line); border-radius:7px; margin-bottom:9px; background:#fff; }
    summary { cursor:pointer; padding:9px 10px; font-weight:700; display:flex; align-items:center; gap:8px; }
    summary::-webkit-details-marker { display:none; }
    .ev-body { padding:0 10px 11px; }
    .kv { display:grid; grid-template-columns:145px minmax(0,1fr); gap:5px 10px; margin:8px 0; }
    .kv div:nth-child(odd) { color:var(--muted); }
    .quote { margin:8px 0; padding:8px 10px; background:#f6f8fa; border-left:3px solid #9aabc0; white-space:pre-wrap; overflow-wrap:anywhere; }
    .rule { padding:6px 0; border-top:1px solid #edf0f4; }
    .review { color:var(--amber); font-weight:700; }
    @media (max-width:980px) { header { position:static; } main { grid-template-columns:1fr; padding:12px; } .table-wrap { max-height:none; } aside { position:static; max-height:none; } th { position:static; } table { min-width:880px; } input { width:100%; } }
  </style>
</head>
<body>
  <header><h1>__TITLE__</h1><div id="stats" class="stats"></div></header>
  <main>
    <section>
      <div class="toolbar">
        <input id="search" type="search" aria-label="搜索结果" placeholder="搜索 REFNO / RESULTNO / target / evidence">
        <select id="indicatorFilter" aria-label="筛选正分指标"><option value="">全部指标</option><option>R1</option><option>R2</option><option>R3</option><option>R4</option><option>R5</option><option>R6</option><option>R7</option></select>
        <select id="reviewFilter" aria-label="筛选复核状态"><option value="">全部状态</option><option value="review">需复核</option><option value="decided">已判定</option></select>
        <span id="count" class="small"></span>
      </div>
      <div class="table-wrap"><table id="resultTable"></table></div>
    </section>
    <aside id="detail"></aside>
  </main>
  <script>const DATA=__DATA__;</script>
  <script>
    const indicators=['R1','R2','R3','R4','R5','R6','R7'];
    const table=document.getElementById('resultTable'), detail=document.getElementById('detail');
    const search=document.getElementById('search'), indicatorFilter=document.getElementById('indicatorFilter'), reviewFilter=document.getElementById('reviewFilter'), count=document.getElementById('count');
    const pageHeader=document.querySelector('body > header');
    const syncHeaderHeight=()=>document.documentElement.style.setProperty('--page-header-height',`${pageHeader.getBoundingClientRect().height}px`);
    syncHeaderHeight();
    if('ResizeObserver' in window)new ResizeObserver(syncHeaderHeight).observe(pageHeader);else window.addEventListener('resize',syncHeaderHeight);
    let selected=DATA.rows[0]?.RESULTNO||'';
    const esc=v=>String(v??'').replace(/[&<>"']/g,ch=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    function scoreClass(row,ind){ const outcome=row.evidence[ind]?.outcome; return outcome==='UNKNOWN'?'score unknown':row.scores[ind]===1?'score yes':'score no'; }
    function rowMatches(row){ const q=search.value.trim().toLowerCase(), ind=indicatorFilter.value, rf=reviewFilter.value; if(ind&&row.scores[ind]!==1)return false; if(rf==='review'&&!row.manual_review)return false; if(rf==='decided'&&row.manual_review)return false; return !q||JSON.stringify(row).toLowerCase().includes(q); }
    function renderStats(){ const s=DATA.summary, parts=[['Rows',s.rows],['REFNO',s.unique_refnos],['RESULTNO',s.unique_resultnos],['R3 route',s.r3_input_mode],['需复核',s.manual_review_rows]]; for(const ind of indicators)parts.push([`${ind}=1`,s.score_counts[ind]?.['1']??0]); document.getElementById('stats').innerHTML=parts.map(([k,v])=>`<span class="stat"><strong>${esc(k)}:</strong> ${esc(v)}</span>`).join(''); }
    function visibleRows(){ return DATA.rows.filter(rowMatches); }
    function renderTable(){ const rows=visibleRows(); if(!rows.some(r=>r.RESULTNO===selected))selected=rows[0]?.RESULTNO||''; count.textContent=`${rows.length} / ${DATA.rows.length} visible`; const head=`<thead><tr><th>REFNO</th><th>RESULTNO</th><th>Target</th>${indicators.map(i=>`<th>${i}</th>`).join('')}<th>Sum</th><th>Review</th></tr></thead>`; const body=rows.map(row=>`<tr data-id="${esc(row.RESULTNO)}" class="${row.RESULTNO===selected?'active':''}"><td>${esc(row.REFNO)}</td><td>${esc(row.RESULTNO)}</td><td class="target" title="${esc(row.TARGET)}">${esc(row.TARGET)}</td>${indicators.map(ind=>`<td><span class="${scoreClass(row,ind)}">${esc(row.scores[ind])}</span></td>`).join('')}<td><strong>${esc(row.R_SUM)}</strong></td><td>${row.manual_review?'<span class="review">是</span>':'否'}</td></tr>`).join(''); table.innerHTML=head+`<tbody>${body}</tbody>`; table.querySelectorAll('tbody tr').forEach(tr=>tr.addEventListener('click',()=>{selected=tr.dataset.id;renderTable();renderDetail();})); }
    function valuesBlock(values){ if(!values||typeof values!=='object')return''; return `<div class="kv">${Object.entries(values).map(([k,v])=>`<div>${esc(k)}</div><div>${esc(Array.isArray(v)?v.join(', '):v)}</div>`).join('')}</div>`; }
    function rulesBlock(ev){ const groups=[['Risks',ev.matched_risks],['Rescues',ev.matched_rescues],['R7 zero rules',ev.matched_zero_rules]]; return groups.filter(([,items])=>Array.isArray(items)&&items.length).map(([label,items])=>`<div class="rule"><strong>${esc(label)}</strong>${items.map(item=>`<div>${esc(item.label)}${item.decisive===false?' (review only)':''}</div>`).join('')}</div>`).join(''); }
    function renderDetail(){ const row=visibleRows().find(r=>r.RESULTNO===selected); if(!row){detail.innerHTML='<div class="detail-head"><div class="detail-title">无匹配结果</div><div class="detail-sub">请调整搜索或筛选条件</div></div>';return;} detail.innerHTML=`<div class="detail-head"><div class="detail-title">${esc(row.TARGET)}</div><div class="detail-sub">REFNO ${esc(row.REFNO)} · RESULTNO ${esc(row.RESULTNO)} · ROCKUNITNO ${esc(row.ROCKUNITNO)}</div></div><div class="evidence">${indicators.map(ind=>{const ev=row.evidence[ind]||{};return `<details ${ind==='R3'||ev.outcome==='UNKNOWN'?'open':''}><summary><span class="${scoreClass(row,ind)}">${esc(row.scores[ind])}</span>${ind} · ${esc(ev.status||ev.outcome||'')}</summary><div class="ev-body"><p>${esc(ev.label||'')}</p><p>${esc(ev.reasoning||'')}</p>${ev.quote?`<div class="quote">${esc(ev.quote)}</div>`:''}${ev.source?`<div class="small">Source: ${esc(ev.source)}</div>`:''}${valuesBlock(ev.values)}${ev.checks?valuesBlock(Object.fromEntries(ev.checks.map(c=>[c.expression,`${c.value??''} · ${c.satisfied?'pass':'fail'}`]))):''}${rulesBlock(ev)}${ev.manual_review?'<p class="review">需要人工复核</p>':''}<div class="small">Method: ${esc(ev.method||'deterministic formula')}</div></div></details>`;}).join('')}</div>`; }
    [search,indicatorFilter,reviewFilter].forEach(el=>el.addEventListener('input',()=>{renderTable();renderDetail();}));
    renderStats();renderTable();renderDetail();
  </script>
</body>
</html>'''


def render_html(rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any], output: Path) -> None:
    title = "Paleomagnetic R1-R7 Complete Assessment"
    payload = json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("</", "<\\/")
    content = HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(output)
    output.chmod(0o644)


def run_self_test(knowledge: Mapping[str, Any]) -> None:
    base = {
        "REFNO": "10",
        "RESULTNO": "1",
        "ROCKUNITNO": "20",
        "ROCKNAME": "Unit A",
        "HIMAGAGE": "110",
        "LOMAGAGE": "90",
        "HIGHAGE": "110",
        "LOWAGE": "90",
        "B": "8",
        "N": "25",
        "KD": "10",
        "ED95": "16",
        "DEMAGCODE": "3",
        "TESTS": "F+,R+",
        "ROCKMAG": "IRM and HYST identify MAGN",
        "COMPONENT": "ChRM",
        "COMMENTS": "",
        "R3_EVIDENCE_TEXT": "Hysteresis measurements identify magnetite as the carrier of ChRM.",
        "R3_EVIDENCE_SOURCE": "paper p. 3",
        "R3_TARGET_BOUND": "1",
        "R5_EVIDENCE_TEXT": "Directions were corrected for bedding dip.",
        "R5_EVIDENCE_SOURCE": "paper p. 4",
        "R5_TARGET_BOUND": "1",
        "R7_EVIDENCE_TEXT": "No younger remagnetization was identified.",
        "R7_EVIDENCE_SOURCE": "paper p. 8",
        "R7_TARGET_BOUND": "1",
        "_R3_INPUT_MODE": "rockmag",
        "_R3_TARGET_BOUND": True,
        "_R5_TARGET_BOUND": True,
        "_R7_TARGET_BOUND": True,
    }
    for column in NUMERIC_COLUMNS:
        base[f"_{column}"] = float(base[column]) if base[column] else None
    result = score_record(base, knowledge, text_evaluator="rules")
    assert result["scores"] == {"R1": 1, "R2": 1, "R3": 1, "R4": 1, "R5": 1, "R6": 1, "R7": 1}
    blank_rockmag = dict(base)
    blank_rockmag["ROCKMAG"] = ""
    blank_rockmag_result = score_record(blank_rockmag, knowledge, text_evaluator="rules")
    assert blank_rockmag_result["scores"]["R3"] == 0
    assert blank_rockmag_result["evidence"]["R3"]["source"] == "Input::ROCKMAG"

    excerpt = dict(base)
    excerpt["_R3_INPUT_MODE"] = "excerpt"
    excerpt_result = score_record(excerpt, knowledge, text_evaluator="rules")
    assert excerpt_result["scores"]["R3"] == 1
    assert excerpt_result["evidence"]["R3"]["source"] == "paper p. 3"
    assert determine_r3_input_mode(["RESULTNO", "ROCKMAG", "R3_EVIDENCE_TEXT"]) == "rockmag"
    assert determine_r3_input_mode(["RESULTNO", "R3_EVIDENCE_TEXT"]) == "excerpt"
    risk = dict(base)
    risk["R5_EVIDENCE_TEXT"] = "Possible tilting could not be ruled out."
    risk["R7_EVIDENCE_TEXT"] = "The unit was remagnetized after its initial cooling."
    risk_result = score_record(risk, knowledge, text_evaluator="rules")
    assert risk_result["scores"]["R5"] == 0
    assert risk_result["scores"]["R7"] == 0

    api_negative = {
        "outcome": "NEGATIVE",
        "status": "API_DECIDED",
        "reasoning": "Synthetic API result.",
        "manual_review": False,
        "api_audit": {"response_id": "self-test"},
    }
    resolved_r3, resolved_r3_evidence = resolve_r3_api_result("1", 1, api_negative)
    resolved_r5, resolved_r5_evidence = resolve_r5_api_result("1", 0, api_negative)
    resolved_r7, resolved_r7_evidence = resolve_r7_api_result("1", 0, api_negative)
    assert resolved_r3 == 0 and resolved_r3_evidence["status"] == "API_DECIDED"
    assert resolved_r5 == 1 and resolved_r5_evidence["status"] == "API_DECIDED"
    assert resolved_r7 == 1 and resolved_r7_evidence["status"] == "API_DECIDED"
    untouched, untouched_evidence = resolve_r5_api_result(
        "1", 0, {"outcome": "NEGATIVE", "reasoning": "API was not called."}
    )
    assert untouched == 0 and untouched_evidence["outcome"] == "NEGATIVE"
    print("Self-test passed: formulas and R3/R5/R7 evaluation paths are operational.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--sheet", default=DEFAULT_SHEET)
    parser.add_argument(
        "--text-evaluator",
        choices=("rules", "api"),
        default="rules",
        help=(
            "R3 excerpt/R5/R7 evaluator. Rules are local and deterministic; api uses "
            "OpenAI-compatible chat completions. R3 ROCKMAG mode always remains local."
        ),
    )
    parser.add_argument("--check-only", action="store_true", help="Validate and score without writing HTML.")
    parser.add_argument("--self-test", action="store_true", help="Run internal synthetic boundary checks.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        load_env_file(DEFAULT_ENV)
        knowledge_path = args.knowledge.expanduser().resolve()
        knowledge = load_knowledge(knowledge_path)
        if args.self_test:
            run_self_test(knowledge)
            return 0
        input_path = args.input.expanduser().resolve()
        output_path = args.output.expanduser().resolve()
        if input_path == output_path:
            raise WorkflowError("Input and output paths must be different")
        records = load_input(input_path, args.sheet)
        scored = [score_record(record, knowledge, text_evaluator=args.text_evaluator) for record in records]
        scored.sort(key=lambda row: (natural_key(row["REFNO"]), natural_key(row["RESULTNO"])))
        summary = build_summary(scored, input_path, knowledge_path, args.text_evaluator)
        if args.check_only:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        render_html(scored, summary, output_path)
    except WorkflowError as exc:
        print(f"R1-R7 workflow failed: {exc}", file=sys.stderr)
        return 2
    print(f"Saved {len(scored)} rows: {output_path}")
    print(f"Manual-review rows: {summary['manual_review_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
