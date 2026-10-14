#!/usr/bin/env python3
"""Audit R3 predictions against standard labels and the frozen 94.08% gold run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GOLD = ROOT / "data/r3_gold_9408.xlsx"
DEFAULT_STANDARD = ROOT / "data/targets.xlsx"
DEFAULT_OUTPUT = ROOT / "outputs/r3_drift_audit_vs_gold.xlsx"
REQUIRED_COLUMNS = {"REFNO", "RESULTNO", "TARGET", "R3", "R3_RESULT_JSON"}


def normalize_score(value: Any, column: str) -> int:
    if pd.isna(value):
        raise ValueError(f"{column} contains a blank score")
    text = str(value).strip()
    if text in {"1", "1.0", "True", "true"}:
        return 1
    if text in {"0", "0.0", "False", "false"}:
        return 0
    raise ValueError(f"{column} contains a non-binary score: {value!r}")


def load_result_workbook(path: Path, score_column: str = "R3") -> pd.DataFrame:
    frame = pd.read_excel(path, dtype={"REFNO": str, "RESULTNO": str})
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {', '.join(missing)}")
    known = frame["RESULTNO"].dropna()
    known = known[known.astype(str).str.upper() != "UNKNOWN"]
    duplicated = known[known.duplicated()].astype(str).tolist()
    if duplicated:
        raise SystemExit(f"{path} contains duplicate RESULTNO values: {', '.join(duplicated[:10])}")
    frame = frame.copy()
    try:
        frame[score_column] = frame[score_column].map(lambda value: normalize_score(value, score_column))
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    return frame


def load_standard(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path, usecols=["REFNO", "RESULTNO", "ROCKNAME", "R3"], dtype={"REFNO": str, "RESULTNO": str})
    if frame["RESULTNO"].duplicated().any():
        duplicated = frame.loc[frame["RESULTNO"].duplicated(), "RESULTNO"].astype(str).head(10)
        raise SystemExit(f"{path} contains duplicate RESULTNO values: {', '.join(duplicated)}")
    frame = frame.rename(columns={"REFNO": "STANDARD_REFNO", "ROCKNAME": "STANDARD_TARGET", "R3": "STANDARD_R3"})
    try:
        frame["STANDARD_R3"] = frame["STANDARD_R3"].map(lambda value: normalize_score(value, "STANDARD_R3"))
    except ValueError as exc:
        raise SystemExit(f"{path}: {exc}") from exc
    return frame


def parse_json_cell(value: Any) -> dict[str, Any]:
    if pd.isna(value):
        return {}
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def manual_review_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["R3_RESULT_JSON"].map(lambda value: bool(parse_json_cell(value).get("manual_review")))


def compare(predictions: Path, gold: Path, standard: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = load_result_workbook(predictions)
    pred = pred[pred["RESULTNO"].astype(str).str.upper() != "UNKNOWN"].copy()
    gold_frame = load_result_workbook(gold).rename(
        columns={
            "REFNO": "GOLD_REFNO",
            "TARGET": "GOLD_TARGET",
            "R3": "GOLD_R3",
            "R3_RESULT_JSON": "GOLD_R3_RESULT_JSON",
        }
    )
    gold_frame = gold_frame[gold_frame["RESULTNO"].astype(str).str.upper() != "UNKNOWN"].copy()
    standard_frame = load_standard(standard)

    comparison = pred.merge(standard_frame, on="RESULTNO", how="left", indicator="PREDICTED_STANDARD_JOIN")
    comparison = comparison.merge(
        gold_frame[["RESULTNO", "GOLD_REFNO", "GOLD_TARGET", "GOLD_R3", "GOLD_R3_RESULT_JSON"]],
        on="RESULTNO",
        how="left",
        validate="one_to_one",
    )

    labelled = comparison["STANDARD_R3"].isin([0, 1])
    comparison["PREDICTED_CORRECT"] = labelled & comparison["R3"].eq(comparison["STANDARD_R3"])
    comparison["GOLD_CORRECT"] = labelled & comparison["GOLD_R3"].eq(comparison["STANDARD_R3"])
    comparison["SCORE_CHANGED_FROM_GOLD"] = comparison["GOLD_R3"].isin([0, 1]) & comparison["R3"].ne(comparison["GOLD_R3"])
    comparison["GOLD_CORRECT_TO_PREDICTED_WRONG"] = comparison["GOLD_CORRECT"] & ~comparison["PREDICTED_CORRECT"]
    comparison["GOLD_WRONG_TO_PREDICTED_CORRECT"] = ~comparison["GOLD_CORRECT"] & comparison["PREDICTED_CORRECT"] & labelled
    comparison["PREDICTED_ERROR_TYPE"] = "correct"
    comparison.loc[labelled & (comparison["R3"].eq(1)) & (comparison["STANDARD_R3"].eq(0)), "PREDICTED_ERROR_TYPE"] = "false_positive"
    comparison.loc[labelled & (comparison["R3"].eq(0)) & (comparison["STANDARD_R3"].eq(1)), "PREDICTED_ERROR_TYPE"] = "false_negative"
    comparison.loc[~labelled, "PREDICTED_ERROR_TYPE"] = "no_standard"
    comparison["MANUAL_REVIEW"] = manual_review_mask(comparison)

    n_labelled = int(labelled.sum())
    predicted_correct = int(comparison.loc[labelled, "PREDICTED_CORRECT"].sum())
    gold_correct = int(comparison.loc[labelled, "GOLD_CORRECT"].sum())
    regressions = int(comparison["GOLD_CORRECT_TO_PREDICTED_WRONG"].sum())
    improvements = int(comparison["GOLD_WRONG_TO_PREDICTED_CORRECT"].sum())
    score_changes = int(comparison["SCORE_CHANGED_FROM_GOLD"].sum())
    fp = int((labelled & comparison["R3"].eq(1) & comparison["STANDARD_R3"].eq(0)).sum())
    fn = int((labelled & comparison["R3"].eq(0) & comparison["STANDARD_R3"].eq(1)).sum())
    summary = pd.DataFrame(
        [
            {
                "predictions": str(predictions),
                "gold": str(gold),
                "standard": str(standard),
                "labelled_rows": n_labelled,
                "predicted_correct": predicted_correct,
                "predicted_accuracy": predicted_correct / n_labelled if n_labelled else None,
                "gold_correct": gold_correct,
                "gold_accuracy": gold_correct / n_labelled if n_labelled else None,
                "accuracy_delta_vs_gold": (predicted_correct - gold_correct) / n_labelled if n_labelled else None,
                "score_changes_from_gold": score_changes,
                "gold_correct_to_predicted_wrong": regressions,
                "gold_wrong_to_predicted_correct": improvements,
                "false_positive": fp,
                "false_negative": fn,
                "manual_review": int(comparison["MANUAL_REVIEW"].sum()),
            }
        ]
    )
    return summary, comparison


def write_audit(output: Path, summary: pd.DataFrame, comparison: pd.DataFrame) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    regressions = comparison[comparison["GOLD_CORRECT_TO_PREDICTED_WRONG"]].copy()
    improvements = comparison[comparison["GOLD_WRONG_TO_PREDICTED_CORRECT"]].copy()
    score_changes = comparison[comparison["SCORE_CHANGED_FROM_GOLD"]].copy()
    disagreements = comparison[comparison["PREDICTED_ERROR_TYPE"].isin(["false_positive", "false_negative"])].copy()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        regressions.to_excel(writer, sheet_name="GoldCorrectRegressions", index=False)
        improvements.to_excel(writer, sheet_name="GoldWrongImprovements", index=False)
        score_changes.to_excel(writer, sheet_name="ScoreChanges", index=False)
        disagreements.to_excel(writer, sheet_name="PredictedDisagreements", index=False)
        comparison.to_excel(writer, sheet_name="AllComparedRows", index=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--standard", type=Path, default=DEFAULT_STANDARD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-accuracy", type=float, default=0.93)
    parser.add_argument("--max-gold-regressions", type=int, default=0)
    parser.add_argument(
        "--allow-missing-gold",
        action="store_true",
        help="Allow prediction rows without a matching gold RESULTNO. Standard labelled rows are still required.",
    )
    args = parser.parse_args()

    summary, comparison = compare(args.predictions, args.gold, args.standard)
    labelled_missing_gold = comparison["STANDARD_R3"].isin([0, 1]) & comparison["GOLD_R3"].isna()
    if not args.allow_missing_gold and labelled_missing_gold.any():
        missing = ", ".join(comparison.loc[labelled_missing_gold, "RESULTNO"].astype(str).head(20))
        raise SystemExit(f"Gold workbook lacks RESULTNO row(s) present in predictions/standard comparison: {missing}")

    write_audit(args.output, summary, comparison)
    row = summary.iloc[0]
    accuracy = float(row["predicted_accuracy"])
    regressions = int(row["gold_correct_to_predicted_wrong"])
    print(f"Predicted accuracy: {int(row.predicted_correct)}/{int(row.labelled_rows)} = {accuracy:.2%}")
    print(f"Gold accuracy: {int(row.gold_correct)}/{int(row.labelled_rows)} = {float(row.gold_accuracy):.2%}")
    print(f"Score changes from gold: {int(row.score_changes_from_gold)}")
    print(f"Gold-correct regressions: {regressions}")
    print(f"Gold-wrong improvements: {int(row.gold_wrong_to_predicted_correct)}")
    print(f"False positives: {int(row.false_positive)}")
    print(f"False negatives: {int(row.false_negative)}")
    print(f"Manual review: {int(row.manual_review)}")
    print(f"Saved audit: {args.output}")

    failed = False
    if accuracy < args.min_accuracy:
        print(f"FAIL: accuracy {accuracy:.2%} is below --min-accuracy {args.min_accuracy:.2%}.")
        failed = True
    if regressions > args.max_gold_regressions:
        print(
            "FAIL: gold-correct regressions "
            f"{regressions} exceed --max-gold-regressions {args.max_gold_regressions}."
        )
        failed = True
    if failed:
        return 1
    print("PASS: R3 drift audit gate passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
