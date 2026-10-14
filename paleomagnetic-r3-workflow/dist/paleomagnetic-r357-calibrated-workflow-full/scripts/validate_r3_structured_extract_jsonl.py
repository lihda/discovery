#!/usr/bin/env python3
"""Validate label-free structured R3 extraction JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {"REFNO", "RESULTNO", "TARGET", "STRUCTURED_EVIDENCE"}
FORBIDDEN_TOP_LEVEL = {"R3", "STANDARD_R3", "GOLD_R3", "RETAINED_R3"}
REQUIRED_EVIDENCE = {
    "resultno",
    "target",
    "extractor",
    "source_quality",
    "target_anchor",
    "carrier_investigation",
    "carrier_identity",
    "remanence_use_bridge",
    "evidence_family",
    "risks",
    "high_risk_route",
    "extractor_notes",
    "manual_review",
}
REQUIRED_FIELD_KEYS = {"status", "quote", "reason"}
REQUIRED_RISKS = {
    "directional_substitution_risk",
    "multi_target_transfer_risk",
    "demagnetization_only_risk",
    "source_quality_risk",
    "evaluator_inference_risk",
    "separate_evidence_family_risk",
}


def parse_payload(path: Path, line_no: int, value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}:{line_no}: invalid STRUCTURED_EVIDENCE: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"{path}:{line_no}: STRUCTURED_EVIDENCE must be an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            missing = REQUIRED_TOP_LEVEL - set(row)
            if missing:
                raise SystemExit(f"{path}:{line_no}: missing top-level keys: {sorted(missing)}")
            forbidden = FORBIDDEN_TOP_LEVEL & set(row)
            if forbidden:
                raise SystemExit(f"{path}:{line_no}: extraction row contains score keys: {sorted(forbidden)}")

            payload = parse_payload(path, line_no, row["STRUCTURED_EVIDENCE"])
            missing_payload = REQUIRED_EVIDENCE - set(payload)
            if missing_payload:
                raise SystemExit(
                    f"{path}:{line_no}: missing evidence keys: {sorted(missing_payload)}"
                )
            if payload.get("extractor") != "Codex chat model (structured evidence extraction)":
                raise SystemExit(f"{path}:{line_no}: unexpected extractor")
            if str(payload.get("resultno")) != str(row["RESULTNO"]):
                raise SystemExit(f"{path}:{line_no}: payload resultno mismatch")
            if str(payload.get("target")) != str(row["TARGET"]):
                raise SystemExit(f"{path}:{line_no}: payload target mismatch")

            for key in (
                "target_anchor",
                "carrier_investigation",
                "carrier_identity",
                "remanence_use_bridge",
                "evidence_family",
            ):
                field = payload.get(key)
                if not isinstance(field, dict):
                    raise SystemExit(f"{path}:{line_no}: {key} must be an object")
                missing_field = REQUIRED_FIELD_KEYS - set(field)
                if missing_field:
                    raise SystemExit(
                        f"{path}:{line_no}: {key} missing keys: {sorted(missing_field)}"
                    )

            source_quality = payload.get("source_quality")
            if not isinstance(source_quality, dict) or "status" not in source_quality:
                raise SystemExit(f"{path}:{line_no}: source_quality must contain status")

            risks = payload.get("risks")
            if not isinstance(risks, dict):
                raise SystemExit(f"{path}:{line_no}: risks must be an object")
            missing_risks = REQUIRED_RISKS - set(risks)
            if missing_risks:
                raise SystemExit(f"{path}:{line_no}: risks missing keys: {sorted(missing_risks)}")
            for risk_key in REQUIRED_RISKS:
                if not isinstance(risks[risk_key], bool):
                    raise SystemExit(f"{path}:{line_no}: risks.{risk_key} must be boolean")

            if not isinstance(payload.get("high_risk_route"), bool):
                raise SystemExit(f"{path}:{line_no}: high_risk_route must be boolean")
            if not isinstance(payload.get("manual_review"), bool):
                raise SystemExit(f"{path}:{line_no}: manual_review must be boolean")
            rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--expected-lines", type=int)
    args = parser.parse_args()

    seen: set[str] = set()
    total = 0
    manual_review = 0
    high_risk = 0
    for path in args.paths:
        rows = load_jsonl(path)
        if args.expected_lines is not None and len(rows) != args.expected_lines:
            raise SystemExit(
                f"{path}: expected {args.expected_lines} line(s), found {len(rows)}"
            )
        for row in rows:
            resultno = str(row["RESULTNO"])
            if resultno in seen:
                raise SystemExit(f"duplicate RESULTNO across files: {resultno}")
            seen.add(resultno)
            payload = parse_payload(path, 0, row["STRUCTURED_EVIDENCE"])
            manual_review += int(payload["manual_review"])
            high_risk += int(payload["high_risk_route"])
        total += len(rows)

    print(
        json.dumps(
            {
                "files": len(args.paths),
                "rows": total,
                "unique_resultnos": len(seen),
                "manual_review": manual_review,
                "high_risk_route": high_risk,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
