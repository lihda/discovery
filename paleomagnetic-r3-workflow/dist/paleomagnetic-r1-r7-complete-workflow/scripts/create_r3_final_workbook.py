#!/usr/bin/env python3
"""Create the final user-facing R3 workbook with target identifiers and red mismatches."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Font

from audit_r3_drift_against_gold import compare as compare_with_gold
from audit_r3_drift_against_gold import write_audit as write_gold_audit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "outputs/r3_working_predictions.xlsx"
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_GOLD = ROOT / "data/r3_gold_9408.xlsx"
DEFAULT_AUDIT_OUTPUT = ROOT / "outputs/r3_drift_audit_vs_gold.xlsx"
FINAL_COLUMNS = ["REFNO", "RESULTNO", "TARGET", "R3", "R3_RESULT_JSON"]
OUTPUT_NAME_PATTERN = re.compile(r"^r3_\d{8}_(?:\d{4,5}|unlabeled)\.xlsx$")


def build_final_frame(predictions: Path, targets: Path) -> pd.DataFrame:
    pred = pd.read_excel(predictions, dtype={"REFNO": str, "RESULTNO": str})
    required = {"REFNO", "RESULTNO", "TARGET", "R3", "R3_RESULT_JSON"}
    missing = sorted(required - set(pred.columns))
    if missing:
        raise SystemExit(f"{predictions} is missing required column(s): {', '.join(missing)}")
    known_predictions = pred["RESULTNO"].dropna()
    known_predictions = known_predictions[known_predictions.astype(str).str.upper() != "UNKNOWN"]
    if known_predictions.duplicated().any():
        duplicated = ", ".join(known_predictions[known_predictions.duplicated()].astype(str).head(10))
        raise SystemExit(f"{predictions} contains duplicate RESULTNO values: {duplicated}")

    target_columns = set(pd.read_excel(targets, nrows=0).columns)
    if {"RESULTNO", "R3"}.issubset(target_columns):
        standard = pd.read_excel(targets, usecols=["RESULTNO", "R3"], dtype={"RESULTNO": str})
        if standard["RESULTNO"].duplicated().any():
            duplicated = ", ".join(standard.loc[standard["RESULTNO"].duplicated(), "RESULTNO"].astype(str).head(10))
            raise SystemExit(f"{targets} contains duplicate RESULTNO values: {duplicated}")

        standard = standard.rename(columns={"R3": "STANDARD_R3"})
        merged = pred.merge(standard, on="RESULTNO", how="left", validate="many_to_one")
    else:
        merged = pred.copy()
        merged["STANDARD_R3"] = pd.NA
    merged["RESULTNO"] = merged["RESULTNO"].where(merged["RESULTNO"].notna(), "UNKNOWN")
    compared = merged["STANDARD_R3"].isin([0, 1])
    matches = pd.Series(False, index=merged.index)
    matches.loc[compared] = merged.loc[compared, "R3"].eq(merged.loc[compared, "STANDARD_R3"])
    merged["MISMATCH"] = compared & ~matches
    return merged


def write_final_workbook(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame[FINAL_COLUMNS].to_excel(output, index=False)

    workbook = load_workbook(output)
    worksheet = workbook.active
    worksheet.title = "R3 Results"

    for cell in worksheet[1]:
        cell.font = Font(bold=True)

    red_font = Font(color="FFFF0000")
    for excel_row, mismatch in enumerate(frame["MISMATCH"].tolist(), start=2):
        if mismatch:
            for cell in worksheet[excel_row]:
                cell.font = red_font

    widths = {"A": 12, "B": 12, "C": 34, "D": 8, "E": 120}
    for column, width in widths.items():
        worksheet.column_dimensions[column].width = width
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    workbook.save(output)


def accuracy_suffix(frame: pd.DataFrame) -> tuple[str, int, int]:
    compared = frame["STANDARD_R3"].isin([0, 1])
    total = int(compared.sum())
    if total == 0:
        return "unlabeled", 0, 0
    correct = int((compared & frame["R3"].eq(frame["STANDARD_R3"])).sum())
    suffix = f"{round(correct / total * 10000):04d}"
    return suffix, correct, total


def validate_timestamp(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("timestamp must use MMDDHHMM, for example 06281543")
    return value


def validate_output_name(path: Path) -> Path:
    if not OUTPUT_NAME_PATTERN.fullmatch(path.name):
        raise SystemExit(
            "Final workbook name must match r3_<MMDDHHMM>_<accuracy-or-unlabeled>.xlsx, "
            "for example r3_06281543_9333.xlsx or r3_06281543_unlabeled.xlsx"
        )
    return path


def accuracy_part_from_name(path: Path) -> str:
    validate_output_name(path)
    return path.stem.split("_")[-1]


def resolve_output_path(
    frame: pd.DataFrame,
    output: Path | None,
    output_dir: Path,
    timestamp: str | None,
) -> tuple[Path, str, int, int]:
    suffix, correct, total = accuracy_suffix(frame)
    if output is not None:
        validate_output_name(output)
        requested_suffix = accuracy_part_from_name(output)
        if requested_suffix != suffix:
            raise SystemExit(
                f"Output filename accuracy suffix {requested_suffix} does not match computed suffix {suffix}."
            )
        return output, suffix, correct, total
    stamp = timestamp or datetime.now().strftime("%m%d%H%M")
    return output_dir / f"r3_{stamp}_{suffix}.xlsx", suffix, correct, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional final workbook path. If provided, the filename must match r3_<MMDDHHMM>_<accuracy-or-unlabeled>.xlsx.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for auto-named final workbooks when --output is omitted.",
    )
    parser.add_argument(
        "--timestamp",
        type=validate_timestamp,
        default=None,
        help="Optional MMDDHHMM timestamp used in the auto-generated filename.",
    )
    parser.add_argument(
        "--skip-gold-audit",
        action="store_true",
        help="Skip the default R3 gold drift gate. Use only for diagnostics or intentionally uncalibrated runs.",
    )
    parser.add_argument(
        "--audit-gold",
        type=Path,
        default=DEFAULT_GOLD,
        help="Frozen R3 gold regression workbook used by the default drift gate.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT,
        help="Workbook path for the gold drift audit report.",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        default=0.93,
        help="Minimum standard-label accuracy required by the default gold drift gate.",
    )
    parser.add_argument(
        "--max-gold-regressions",
        type=int,
        default=0,
        help="Maximum allowed rows where the gold run was correct and this prediction is wrong.",
    )
    args = parser.parse_args()

    frame = build_final_frame(args.predictions, args.targets)
    has_r3_standard = {"RESULTNO", "R3"}.issubset(set(pd.read_excel(args.targets, nrows=0).columns))
    if not args.skip_gold_audit and has_r3_standard:
        audit_summary, audit_comparison = compare_with_gold(args.predictions, args.audit_gold, args.targets)
        write_gold_audit(args.audit_output, audit_summary, audit_comparison)
        audit_row = audit_summary.iloc[0]
        accuracy = float(audit_row["predicted_accuracy"])
        regressions = int(audit_row["gold_correct_to_predicted_wrong"])
        if accuracy < args.min_accuracy:
            raise SystemExit(
                f"R3 gold drift gate failed: accuracy {accuracy:.2%} is below "
                f"--min-accuracy {args.min_accuracy:.2%}. Audit: {args.audit_output}"
            )
        if regressions > args.max_gold_regressions:
            raise SystemExit(
                f"R3 gold drift gate failed: {regressions} gold-correct regression(s) exceed "
                f"--max-gold-regressions {args.max_gold_regressions}. Audit: {args.audit_output}"
            )
        print(
            "R3 gold drift gate passed: "
            f"{int(audit_row.predicted_correct)}/{int(audit_row.labelled_rows)} = {accuracy:.2%}; "
            f"gold-correct regressions={regressions}."
        )
    elif not args.skip_gold_audit:
        print(f"R3 gold drift gate skipped because {args.targets} has no standard R3 labels.")
    output, suffix, correct, total = resolve_output_path(frame, args.output, args.output_dir, args.timestamp)
    write_final_workbook(frame, output)

    mismatch_ids = (
        frame.loc[frame["MISMATCH"], ["REFNO", "RESULTNO"]]
        .astype(str)
        .apply(lambda row: f"{row['REFNO']}/{row['RESULTNO']}", axis=1)
        .tolist()
    )
    print(f"Saved: {output}")
    print(f"Rows: {len(frame)}")
    print(f"Columns: {', '.join(FINAL_COLUMNS)}")
    if total:
        print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")
    else:
        print(f"Accuracy: not computed (no standard R3 labels in {args.targets})")
    print(f"Accuracy suffix: {suffix}")
    print(f"Mismatches highlighted red: {len(mismatch_ids)}")
    if mismatch_ids:
        print(f"Mismatch REFNO/RESULTNOs: {', '.join(mismatch_ids)}")


if __name__ == "__main__":
    main()
