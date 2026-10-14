#!/usr/bin/env python3
"""Build a source-isolated target manifest with non-label locator fields only."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "outputs/rockmag_multitarget_pilot_20260717/identity_manifest.json"
DEFAULT_PMAG = ROOT / "data/source_metadata/PMAGRESULT.xlsx"
DEFAULT_ROCKUNIT = ROOT / "data/source_metadata/ROCKUNIT.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/rockmag_multitarget_pilot_v2_locators_20260717/identity_manifest.locators.json"

PMAG_ALLOWLIST = (
    "ROCKUNITNO",
    "RESULTNO",
    "COMPONENT",
    "SLAT",
    "SLONG",
    "DEC",
    "INC",
    "PLAT",
    "PLONG",
    "PTYPE",
)
ROCKUNIT_ALLOWLIST = (
    "REFNO",
    "ROCKUNITNO",
    "ROCKNAME",
    "RLAT",
    "RLONG",
)
OUTPUT_FIELDS = (
    "REFNO",
    "RESULTNO",
    "ROCKUNITNO",
    "ROCKNAME",
    "RLAT",
    "RLONG",
    "COMPONENT",
    "SLAT",
    "SLONG",
    "DEC",
    "INC",
    "PLAT",
    "PLONG",
    "PTYPE",
)
FORBIDDEN_FIELDS = (
    "PMAGRESULT.ROCKMAG",
    "PMAGRESULT.TESTS",
    "historical R3",
    "PMAGRESULT.LABDETAILS",
    "PMAGRESULT.COMMENTS",
    "prior predictions",
    "evaluation reports",
)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def load_allowlisted(path: Path, sheet: str, columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_excel(
        path,
        sheet_name=sheet,
        usecols=list(columns),
        dtype=str,
        keep_default_na=False,
    )
    if set(frame.columns) != set(columns):
        missing = sorted(set(columns) - set(frame.columns))
        raise SystemExit(f"missing allowlisted columns in {path}: {', '.join(missing)}")
    frame = frame[list(columns)].copy()
    for column in columns:
        frame[column] = frame[column].map(clean)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--pmagresult", type=Path, default=DEFAULT_PMAG)
    parser.add_argument("--rockunit", type=Path, default=DEFAULT_ROCKUNIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = json.loads(args.base_manifest.read_text(encoding="utf-8"))
    base_identities = base.get("identities")
    if not isinstance(base_identities, list) or not base_identities:
        raise SystemExit("base manifest has no identities")

    pmag = load_allowlisted(args.pmagresult, "PMAGRESULT", PMAG_ALLOWLIST)
    rockunit = load_allowlisted(args.rockunit, "ROCKUNIT", ROCKUNIT_ALLOWLIST)
    if pmag["RESULTNO"].eq("").any() or pmag["RESULTNO"].duplicated().any():
        raise SystemExit("PMAGRESULT RESULTNO must be unique and non-empty")
    if rockunit["ROCKUNITNO"].eq("").any() or rockunit["ROCKUNITNO"].duplicated().any():
        raise SystemExit("ROCKUNIT ROCKUNITNO must be unique and non-empty")

    joined = pmag.merge(rockunit, on="ROCKUNITNO", how="left", validate="many_to_one")
    by_result = joined.set_index("RESULTNO", drop=False)
    identities: list[dict[str, str]] = []
    for base_row in base_identities:
        resultno = clean(base_row.get("RESULTNO"))
        if resultno not in by_result.index:
            raise SystemExit(f"RESULTNO {resultno} is missing from PMAGRESULT")
        source = by_result.loc[resultno]
        if isinstance(source, pd.DataFrame):
            raise SystemExit(f"RESULTNO {resultno} is not unique")
        output_row = {field: clean(source.get(field)) for field in OUTPUT_FIELDS}
        for field in ("REFNO", "RESULTNO", "ROCKUNITNO", "ROCKNAME"):
            if output_row[field] != clean(base_row.get(field)):
                raise SystemExit(
                    f"identity mismatch for RESULTNO {resultno}, field {field}: "
                    f"{output_row[field]!r} != {clean(base_row.get(field))!r}"
                )
        identities.append(output_row)

    payload = {
        "schema_version": "rockmag_multitarget_locator_identity_v2",
        "selected_refnos": base.get("selected_refnos") or sorted({row["REFNO"] for row in identities}),
        "papers": len({row["REFNO"] for row in identities}),
        "targets": len(identities),
        "identity_fields": list(OUTPUT_FIELDS),
        "allowed_source_columns": {
            "PMAGRESULT": list(PMAG_ALLOWLIST),
            "ROCKUNIT": list(ROCKUNIT_ALLOWLIST),
        },
        "forbidden_fields_not_loaded": list(FORBIDDEN_FIELDS),
        "source_sha256": {
            str(args.base_manifest): file_sha256(args.base_manifest),
            str(args.pmagresult): file_sha256(args.pmagresult),
            str(args.rockunit): file_sha256(args.rockunit),
        },
        "identities": identities,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "schema_version": "rockmag_locator_manifest_receipt_v2",
        "manifest": str(args.output),
        "manifest_sha256": file_sha256(args.output),
        "papers": payload["papers"],
        "targets": payload["targets"],
        "identity_fields": payload["identity_fields"],
        "forbidden_fields_not_loaded": payload["forbidden_fields_not_loaded"],
    }
    receipt_path = args.output.with_suffix(args.output.suffix + ".receipt.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
