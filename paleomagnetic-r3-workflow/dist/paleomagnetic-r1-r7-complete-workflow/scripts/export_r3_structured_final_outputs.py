#!/usr/bin/env python3
"""Export structured R3 arbitration JSONL as the final workbook + HTML pair."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_JSONL = (
    ROOT
    / "outputs/r3_latest_source_recovered_v9b_arbitration.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            rows.append(row)
    return rows


def normalize_row(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    missing = {"REFNO", "RESULTNO", "TARGET", "R3", "R3_RESULT_JSON"} - set(row)
    if missing:
        raise SystemExit(f"{path}: row missing key(s): {', '.join(sorted(missing))}")
    score = int(row["R3"])
    if score not in {0, 1}:
        raise SystemExit(f"{path}: RESULTNO {row.get('RESULTNO')} has non-binary R3={row['R3']!r}")
    payload = row["R3_RESULT_JSON"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"{path}: invalid R3_RESULT_JSON for RESULTNO {row.get('RESULTNO')}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: R3_RESULT_JSON must be an object for RESULTNO {row.get('RESULTNO')}")
    payload["R3"] = score
    payload["resultno"] = str(row["RESULTNO"])
    payload["target"] = str(row["TARGET"])
    return {
        "REFNO": str(row["REFNO"]),
        "RESULTNO": str(row["RESULTNO"]),
        "TARGET": str(row["TARGET"]),
        "R3": score,
        "R3_RESULT_JSON": json.dumps(payload, ensure_ascii=False),
    }


def accuracy_suffix(predictions: pd.DataFrame, targets: Path) -> tuple[str, int, int]:
    target_columns = set(pd.read_excel(targets, nrows=0).columns)
    if not {"RESULTNO", "R3"}.issubset(target_columns):
        return "unlabeled", 0, 0
    standard = pd.read_excel(targets, usecols=["RESULTNO", "R3"], dtype={"RESULTNO": str})
    comparison = predictions.merge(
        standard.rename(columns={"R3": "STANDARD_R3"}),
        on="RESULTNO",
        how="left",
        validate="one_to_one",
    )
    compared = comparison["STANDARD_R3"].isin([0, 1])
    total = int(compared.sum())
    if total == 0:
        return "unlabeled", 0, 0
    correct = int((compared & comparison["R3"].eq(comparison["STANDARD_R3"])).sum())
    return f"{round(correct / total * 10000):04d}", correct, total


def validate_stamp(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("timestamp must use MMDDHHMM, for example 06302343")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL, help="Structured arbitration JSONL.")
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", type=validate_stamp, default=None)
    parser.add_argument("--title", default="R3 Results Visual Review")
    args = parser.parse_args()

    records = [normalize_row(args.jsonl, row) for row in load_jsonl(args.jsonl)]
    predictions = pd.DataFrame(records, columns=["REFNO", "RESULTNO", "TARGET", "R3", "R3_RESULT_JSON"])
    if predictions["RESULTNO"].duplicated().any():
        duplicated = ", ".join(predictions.loc[predictions["RESULTNO"].duplicated(), "RESULTNO"].head(10))
        raise SystemExit(f"Duplicate RESULTNO in {args.jsonl}: {duplicated}")

    suffix, correct, total = accuracy_suffix(predictions, args.targets)
    stamp = args.timestamp or datetime.now().strftime("%m%d%H%M")
    output = args.output_dir / f"r3_{stamp}_{suffix}.xlsx"
    temp_predictions = args.output_dir / f".{output.stem}_working_predictions.xlsx"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_excel(temp_predictions, index=False)

    create_workbook = ROOT / "scripts/create_r3_final_workbook.py"
    create_html = ROOT / "scripts/create_r3_results_page.py"
    try:
        subprocess.run(
            [
                sys.executable,
                str(create_workbook),
                "--predictions",
                str(temp_predictions),
                "--targets",
                str(args.targets),
                "--output",
                str(output),
                "--skip-gold-audit",
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                sys.executable,
                str(create_html),
                "--input",
                str(output),
                "--standard",
                str(args.targets),
                "--title",
                args.title,
            ],
            cwd=ROOT,
            check=True,
        )
    finally:
        temp_predictions.unlink(missing_ok=True)

    print(f"Final workbook: {output}")
    print(f"Final HTML: {output.with_suffix('.html')}")
    if total:
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
    else:
        print(f"Accuracy: not computed (no standard R3 labels in {args.targets})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
