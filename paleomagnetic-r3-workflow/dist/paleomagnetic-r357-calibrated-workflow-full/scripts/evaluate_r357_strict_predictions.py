#!/usr/bin/env python3
"""Evaluate already-written strict R3/R5/R7 predictions against labels.

This command is deliberately separate from scoring. It verifies and hashes
all prediction files before it opens the label workbook, never imports or
calls scoring code, never writes record-level answers, and verifies that the
prediction files remain byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd


INDICATORS = ("R3", "R5", "R7")
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_HOLDOUT_SALT = "r357-strict-refno-holdout-v1"


class EvaluationInputError(ValueError):
    """Raised when predictions or labels cannot be compared safely."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise EvaluationInputError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise EvaluationInputError(
                        f"{path}:{line_number}: prediction must be an object"
                    )
                records.append(value)
        return records
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvaluationInputError(f"{path}: invalid JSON: {exc}") from exc
    if isinstance(value, dict):
        value = value.get("predictions", [value])
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise EvaluationInputError(f"{path}: expected a prediction object or list")
    return value


def _payload_outcome(value: Any, path: Path, resultno: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "DECIDED"
    if isinstance(value, str):
        if not value.strip():
            return "DECIDED"
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise EvaluationInputError(
                f"{path}: invalid result JSON for RESULTNO {resultno}: {exc}"
            ) from exc
    if not isinstance(value, dict):
        raise EvaluationInputError(
            f"{path}: result JSON for RESULTNO {resultno} must be an object"
        )
    return str(value.get("outcome") or "DECIDED").strip().upper()


def load_predictions(path: Path, indicator: str) -> pd.DataFrame:
    payload_column = f"{indicator}_RESULT_JSON"
    if path.suffix.casefold() == ".xlsx":
        frame = pd.read_excel(
            path,
            usecols=["REFNO", "RESULTNO", indicator, payload_column],
            dtype={"REFNO": str, "RESULTNO": str},
        )
    elif path.suffix.casefold() in {".json", ".jsonl"}:
        frame = pd.DataFrame(_load_json_records(path))
        missing = {"REFNO", "RESULTNO", indicator, payload_column} - set(frame.columns)
        if missing:
            raise EvaluationInputError(
                f"{path} is missing prediction field(s): {', '.join(sorted(missing))}"
            )
        frame = frame[["REFNO", "RESULTNO", indicator, payload_column]].copy()
    else:
        raise EvaluationInputError(
            f"Unsupported {indicator} prediction format: {path.suffix}"
        )
    for column in ("REFNO", "RESULTNO"):
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    if frame.empty or frame[["REFNO", "RESULTNO"]].eq("").any(axis=None):
        raise EvaluationInputError(f"{path} contains empty or no REFNO/RESULTNO values")
    if frame["RESULTNO"].duplicated().any():
        raise EvaluationInputError(f"{path} contains duplicate RESULTNO values")
    numeric = pd.to_numeric(frame[indicator], errors="coerce")
    if not numeric.isin([0, 1]).all():
        raise EvaluationInputError(f"{path} contains a non-binary {indicator} prediction")
    frame[indicator] = numeric.astype(int)
    frame["OUTCOME"] = [
        _payload_outcome(value, path, resultno)
        for value, resultno in zip(frame[payload_column], frame["RESULTNO"])
    ]
    return frame[["REFNO", "RESULTNO", indicator, "OUTCOME"]]


def holdout_bucket(refno: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}\0{refno}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def holdout_fraction(value: str) -> float:
    try:
        fraction = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("holdout fraction must be a number") from exc
    if not 0.0 < fraction < 1.0:
        raise argparse.ArgumentTypeError("holdout fraction must be between 0 and 1")
    return fraction


def nonempty_salt(value: str) -> str:
    if not value.strip():
        raise argparse.ArgumentTypeError("holdout salt must not be empty")
    return value


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else None


def metrics_from_counts(
    *,
    labelled: int,
    matched: int,
    known: int,
    correct: int,
    known_correct: int,
    true_negative: int,
    false_positive: int,
    false_negative: int,
    true_positive: int,
) -> dict[str, Any]:
    f1_0 = _f1(true_negative, false_negative, false_positive)
    f1_1 = _f1(true_positive, false_positive, false_negative)
    available_f1 = [value for value in (f1_0, f1_1) if value is not None]
    return {
        "labelled_rows": labelled,
        "matched_prediction_rows": matched,
        "known_rows": known,
        "correct_rows": correct,
        "known_correct_rows": known_correct,
        "prediction_coverage": _ratio(matched, labelled),
        "coverage": _ratio(known, labelled),
        "accuracy": _ratio(correct, matched),
        "known_accuracy": _ratio(known_correct, known),
        "macro_f1": sum(available_f1) / len(available_f1) if available_f1 else None,
        "f1_0": f1_0,
        "f1_1": f1_1,
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }


def evaluate_indicator(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    indicator: str,
) -> dict[str, Any]:
    standard = labels[["RESULTNO", indicator]].copy()
    standard[indicator] = pd.to_numeric(standard[indicator], errors="coerce")
    standard = standard[standard[indicator].isin([0, 1])].copy()
    standard[indicator] = standard[indicator].astype(int)
    comparison = standard.merge(
        predictions.rename(columns={indicator: "PREDICTION"}),
        on="RESULTNO",
        how="left",
        validate="one_to_one",
    )
    matched_mask = comparison["PREDICTION"].isin([0, 1])
    matched = comparison[matched_mask].copy()
    matched["PREDICTION"] = matched["PREDICTION"].astype(int)
    known_mask = matched["OUTCOME"].fillna("MISSING").ne("UNKNOWN")
    correct_mask = matched["PREDICTION"].eq(matched[indicator])
    tn = int(((matched[indicator] == 0) & (matched["PREDICTION"] == 0)).sum())
    fp = int(((matched[indicator] == 0) & (matched["PREDICTION"] == 1)).sum())
    fn = int(((matched[indicator] == 1) & (matched["PREDICTION"] == 0)).sum())
    tp = int(((matched[indicator] == 1) & (matched["PREDICTION"] == 1)).sum())
    result = metrics_from_counts(
        labelled=len(standard),
        matched=len(matched),
        known=int(known_mask.sum()),
        correct=int(correct_mask.sum()),
        known_correct=int((correct_mask & known_mask).sum()),
        true_negative=tn,
        false_positive=fp,
        false_negative=fn,
        true_positive=tp,
    )
    result["extra_prediction_rows"] = int(
        (~predictions["RESULTNO"].isin(set(standard["RESULTNO"]))).sum()
    )
    return result


def aggregate_metrics(indicators: dict[str, dict[str, Any]]) -> dict[str, Any]:
    confusion = {
        key: sum(item["confusion_matrix"][key] for item in indicators.values())
        for key in ("true_negative", "false_positive", "false_negative", "true_positive")
    }
    correct = confusion["true_negative"] + confusion["true_positive"]
    known_correct = sum(item["known_correct_rows"] for item in indicators.values())
    return metrics_from_counts(
        labelled=sum(item["labelled_rows"] for item in indicators.values()),
        matched=sum(item["matched_prediction_rows"] for item in indicators.values()),
        known=sum(item["known_rows"] for item in indicators.values()),
        correct=correct,
        known_correct=known_correct,
        **confusion,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--r3", type=Path, required=True)
    parser.add_argument("--r5", type=Path, required=True)
    parser.add_argument("--r7", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--holdout-fraction",
        type=holdout_fraction,
        default=DEFAULT_HOLDOUT_FRACTION,
        help="Stable SHA256(REFNO) holdout fraction (default: 0.20).",
    )
    parser.add_argument(
        "--holdout-salt",
        type=nonempty_salt,
        default=DEFAULT_HOLDOUT_SALT,
        help="Public salt for stable REFNO assignment; always recorded in the report.",
    )
    return parser.parse_args(argv)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    prediction_paths = {
        "R3": args.r3.resolve(strict=False),
        "R5": args.r5.resolve(strict=False),
        "R7": args.r7.resolve(strict=False),
    }
    missing = [path for path in prediction_paths.values() if not path.is_file()]
    if missing:
        raise EvaluationInputError(
            "Prediction file(s) must exist before labels are opened: "
            + ", ".join(str(path) for path in missing)
        )
    if len(set(prediction_paths.values())) != len(prediction_paths):
        raise EvaluationInputError("R3, R5, and R7 prediction paths must be distinct")

    # This ordering is part of the isolation contract: labels are not opened
    # until every prediction fingerprint has been persisted independently.
    hashes_before = {
        indicator: file_sha256(path)
        for indicator, path in prediction_paths.items()
    }
    labels_path = args.labels.resolve(strict=False)
    output_path = args.output.resolve(strict=False)
    receipt_path = output_path.with_suffix(output_path.suffix + ".prediction_hashes.json")
    protected_inputs = {*prediction_paths.values(), labels_path}
    if output_path in protected_inputs or receipt_path in protected_inputs:
        raise EvaluationInputError("Evaluation outputs must not overwrite predictions or labels")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": "r357_prediction_hash_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "before_labels_opened",
        "predictions": {
            indicator: {"path": str(path), "sha256": hashes_before[indicator]}
            for indicator, path in prediction_paths.items()
        },
    }
    receipt_temporary = receipt_path.with_suffix(receipt_path.suffix + ".tmp")
    receipt_temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    receipt_temporary.replace(receipt_path)

    predictions = {
        indicator: load_predictions(path, indicator)
        for indicator, path in prediction_paths.items()
    }

    if not labels_path.is_file():
        raise EvaluationInputError(f"Label workbook does not exist: {labels_path}")
    labels_hash_before = file_sha256(labels_path)
    labels = pd.read_excel(
        labels_path,
        usecols=["REFNO", "RESULTNO", *INDICATORS],
        dtype={"REFNO": str, "RESULTNO": str},
    )
    for column in ("REFNO", "RESULTNO"):
        labels[column] = labels[column].fillna("").astype(str).str.strip()
    if labels.empty or labels[["REFNO", "RESULTNO"]].eq("").any(axis=None):
        raise EvaluationInputError("Label workbook contains empty or no REFNO/RESULTNO values")
    if labels["RESULTNO"].duplicated().any():
        raise EvaluationInputError("Label workbook contains duplicate RESULTNO values")

    label_identity = labels[["RESULTNO", "REFNO"]]
    for indicator, frame in predictions.items():
        identity_check = label_identity.merge(
            frame[["RESULTNO", "REFNO"]],
            on="RESULTNO",
            how="inner",
            suffixes=("_LABEL", "_PREDICTION"),
            validate="one_to_one",
        )
        mismatched = identity_check["REFNO_LABEL"].ne(identity_check["REFNO_PREDICTION"])
        if mismatched.any():
            examples = identity_check.loc[mismatched, "RESULTNO"].head(10).tolist()
            raise EvaluationInputError(
                f"{indicator} prediction REFNO disagrees with labels for RESULTNO values: {examples}"
            )

    indicator_metrics = {
        indicator: evaluate_indicator(labels, predictions[indicator], indicator)
        for indicator in INDICATORS
    }
    holdout_mask = labels["REFNO"].map(
        lambda refno: holdout_bucket(refno, args.holdout_salt) < args.holdout_fraction
    )
    holdout_labels = labels[holdout_mask].copy()
    holdout_resultnos = set(holdout_labels["RESULTNO"])
    holdout_predictions = {
        indicator: frame[frame["RESULTNO"].isin(holdout_resultnos)].copy()
        for indicator, frame in predictions.items()
    }
    holdout_metrics = {
        indicator: evaluate_indicator(
            holdout_labels,
            holdout_predictions[indicator],
            indicator,
        )
        for indicator in INDICATORS
    }
    hashes_after = {
        indicator: file_sha256(path)
        for indicator, path in prediction_paths.items()
    }
    if hashes_after != hashes_before:
        raise RuntimeError("A prediction file changed during evaluation")
    labels_hash_after = file_sha256(labels_path)
    if labels_hash_after != labels_hash_before:
        raise RuntimeError("The label workbook changed during evaluation")

    report = {
        "schema_version": "r357_strict_posthoc_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_contract": {
            "scoring_code_called": False,
            "prediction_hashes_recorded_before_labels_opened": True,
            "prediction_hash_receipt_written_before_labels_opened": True,
            "prediction_files_modified": False,
            "record_level_results_written": False,
            "feedback_into_scoring": False,
            "metric_denominators": {
                "accuracy": "matched prediction rows with binary labels",
                "prediction_coverage": "matched prediction rows / labelled rows",
                "coverage": "non-UNKNOWN matched rows / labelled rows",
                "known_accuracy": "correct non-UNKNOWN rows / non-UNKNOWN matched rows",
            },
        },
        "predictions": {
            indicator: {
                "path": str(path),
                "sha256_before_labels": hashes_before[indicator],
                "sha256_after_evaluation": hashes_after[indicator],
            }
            for indicator, path in prediction_paths.items()
        },
        "prediction_hash_receipt": {
            "path": str(receipt_path),
            "sha256": file_sha256(receipt_path),
        },
        "labels": {
            "path": str(labels_path),
            "sha256": labels_hash_before,
        },
        "refno_holdout": {
            "method": "sha256_threshold_by_refno_v1",
            "salt": args.holdout_salt,
            "fraction": args.holdout_fraction,
            "assignment_rule": (
                "uint64_be(SHA256(UTF8(salt + NUL + REFNO))[0:8]) / 2^64 < fraction"
            ),
            "all_unique_refnos": int(labels["REFNO"].nunique()),
            "holdout_unique_refnos": int(holdout_labels["REFNO"].nunique()),
            "all_rows": int(len(labels)),
            "holdout_rows": int(len(holdout_labels)),
            "indicators": holdout_metrics,
            "aggregate": aggregate_metrics(holdout_metrics),
        },
        "indicators": indicator_metrics,
        "aggregate": aggregate_metrics(indicator_metrics),
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_evaluation(args)
    except EvaluationInputError as exc:
        print(f"Evaluation input error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "indicators": report["indicators"],
                "aggregate": report["aggregate"],
                "refno_holdout": report["refno_holdout"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
