#!/usr/bin/env python3
"""Evaluate frozen source-only R3 predictions on the preregistered REFNO holdout.

The evaluator is intentionally separate from scoring and knowledge induction.
It persists a prediction hash receipt before opening the label workbook, accepts
only predictions assigned to the fixed holdout, and emits aggregate metrics
without record-level answers.
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


HOLDOUT_SALT = "r3-rockmag-distillation-v1"
HOLDOUT_FRACTION = 0.20
UNKNOWN_OUTCOMES = frozenset({"ABSTAIN", "UNKNOWN"})


class EvaluationError(ValueError):
    """Raised when an evaluation input violates the frozen contract."""


class EvaluationIntegrityError(EvaluationError):
    """Raised when a protected input changes after fingerprinting."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holdout_bucket(refno: str) -> float:
    digest = hashlib.sha256(f"{HOLDOUT_SALT}\0{refno}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def is_holdout_refno(refno: str) -> bool:
    return holdout_bucket(refno) < HOLDOUT_FRACTION


def _clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prediction_outcome(row: dict[str, Any], path: Path, line_number: int) -> str:
    raw = row.get("OUTCOME", row.get("outcome"))
    if raw is None and "R3_RESULT_JSON" in row:
        payload = row["R3_RESULT_JSON"]
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise EvaluationError(
                    f"{path}:{line_number}: R3_RESULT_JSON is invalid JSON: {exc}"
                ) from exc
        if payload is not None and not isinstance(payload, dict):
            raise EvaluationError(
                f"{path}:{line_number}: R3_RESULT_JSON must be an object"
            )
        if isinstance(payload, dict):
            raw = payload.get("outcome")
    outcome = _clean(raw).upper() or "DECIDED"
    return outcome


def load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.casefold() != ".jsonl":
        raise EvaluationError("R3 distilled predictions must be a JSONL file")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise EvaluationError(f"{path}:{line_number}: prediction must be an object")
            missing = {"REFNO", "RESULTNO", "R3"} - set(row)
            if missing:
                raise EvaluationError(
                    f"{path}:{line_number}: missing field(s): {', '.join(sorted(missing))}"
                )
            refno = _clean(row["REFNO"])
            resultno = _clean(row["RESULTNO"])
            if not refno or not resultno:
                raise EvaluationError(
                    f"{path}:{line_number}: REFNO and RESULTNO must be non-empty"
                )
            value = row["R3"]
            if isinstance(value, bool) or value not in (0, 1):
                raise EvaluationError(f"{path}:{line_number}: R3 must be binary 0 or 1")
            records.append(
                {
                    "REFNO": refno,
                    "RESULTNO": resultno,
                    "R3": int(value),
                    "OUTCOME": _prediction_outcome(row, path, line_number),
                }
            )

    if not records:
        raise EvaluationError(f"{path} contains no predictions")
    frame = pd.DataFrame.from_records(records)
    duplicated = frame["RESULTNO"].duplicated(keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, "RESULTNO"].drop_duplicates().head(10).tolist()
        raise EvaluationError(f"Prediction RESULTNO values must be unique: {examples}")
    non_holdout = frame.loc[~frame["REFNO"].map(is_holdout_refno), "RESULTNO"]
    if not non_holdout.empty:
        raise EvaluationError(
            "Prediction file contains non-holdout REFNO assignments for RESULTNO values: "
            f"{non_holdout.head(10).tolist()}"
        )
    return frame


def load_holdout_labels(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    try:
        labels = pd.read_excel(
            path,
            usecols=["REFNO", "RESULTNO", "R3"],
            dtype={"REFNO": str, "RESULTNO": str},
        )
    except (ValueError, KeyError) as exc:
        raise EvaluationError(f"Could not read REFNO, RESULTNO, and R3 labels: {exc}") from exc

    for column in ("REFNO", "RESULTNO"):
        labels[column] = labels[column].map(_clean)
    if labels.empty or labels[["REFNO", "RESULTNO"]].eq("").any(axis=None):
        raise EvaluationError("Label workbook contains empty or no REFNO/RESULTNO values")
    if labels["RESULTNO"].duplicated().any():
        raise EvaluationError("Label workbook contains duplicate RESULTNO values")

    holdout_mask = labels["REFNO"].map(is_holdout_refno)
    holdout = labels.loc[holdout_mask].copy()
    if holdout.empty:
        raise EvaluationError("Label workbook contains no preregistered holdout rows")
    numeric = pd.to_numeric(holdout["R3"], errors="coerce")
    if not numeric.isin([0, 1]).all():
        raise EvaluationError("Every holdout row must have a binary R3 label")
    holdout["R3"] = numeric.astype(int)
    counts = {
        "all_rows": int(len(labels)),
        "all_unique_refnos": int(labels["REFNO"].nunique()),
        "holdout_rows": int(len(holdout)),
        "holdout_unique_refnos": int(holdout["REFNO"].nunique()),
    }
    return holdout, counts


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float | None:
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else None


def calculate_metrics(comparison: pd.DataFrame) -> dict[str, Any]:
    truth = comparison["R3_LABEL"]
    prediction = comparison["R3_PREDICTION"]
    known = ~comparison["OUTCOME"].isin(UNKNOWN_OUTCOMES)
    correct = truth.eq(prediction)
    tn = int(((truth == 0) & (prediction == 0)).sum())
    fp = int(((truth == 0) & (prediction == 1)).sum())
    fn = int(((truth == 1) & (prediction == 0)).sum())
    tp = int(((truth == 1) & (prediction == 1)).sum())
    recall_0 = _ratio(tn, tn + fp)
    recall_1 = _ratio(tp, tp + fn)
    available_recalls = [value for value in (recall_0, recall_1) if value is not None]
    f1_0 = _f1(tn, fn, fp)
    f1_1 = _f1(tp, fp, fn)
    available_f1 = [value for value in (f1_0, f1_1) if value is not None]
    return {
        "labelled_rows": int(len(comparison)),
        "prediction_rows": int(len(comparison)),
        "known_rows": int(known.sum()),
        "correct_rows": int(correct.sum()),
        "known_correct_rows": int((known & correct).sum()),
        "accuracy": _ratio(int(correct.sum()), len(comparison)),
        "balanced_accuracy": (
            sum(available_recalls) / len(available_recalls) if available_recalls else None
        ),
        "coverage": _ratio(int(known.sum()), len(comparison)),
        "known_accuracy": _ratio(int((known & correct).sum()), int(known.sum())),
        "macro_f1": sum(available_f1) / len(available_f1) if available_f1 else None,
        "f1_0": f1_0,
        "f1_1": f1_1,
        "confusion_matrix": {
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
        },
    }


def _compare_identities(labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    label_resultnos = set(labels["RESULTNO"])
    prediction_resultnos = set(predictions["RESULTNO"])
    missing = label_resultnos - prediction_resultnos
    extra = prediction_resultnos - label_resultnos
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing holdout predictions={len(missing)}")
        if extra:
            details.append(f"unknown/non-holdout predictions={len(extra)}")
        raise EvaluationError("Prediction set must equal the labelled holdout set: " + ", ".join(details))

    comparison = labels.merge(
        predictions,
        on="RESULTNO",
        how="inner",
        suffixes=("_LABEL", "_PREDICTION"),
        validate="one_to_one",
    )
    mismatched = comparison["REFNO_LABEL"].ne(comparison["REFNO_PREDICTION"])
    if mismatched.any():
        examples = comparison.loc[mismatched, "RESULTNO"].head(10).tolist()
        raise EvaluationError(
            "Prediction REFNO disagrees with the label identity for RESULTNO values: "
            f"{examples}"
        )
    return comparison


def evaluate(
    predictions_path: Path,
    labels_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    predictions_path = predictions_path.resolve(strict=False)
    labels_path = labels_path.resolve(strict=False)
    output_path = output_path.resolve(strict=False)
    receipt_path = output_path.with_suffix(output_path.suffix + ".prediction_hashes.json")

    if not predictions_path.is_file():
        raise EvaluationError(f"Prediction JSONL does not exist: {predictions_path}")
    protected = {predictions_path, labels_path}
    if output_path in protected or receipt_path in protected:
        raise EvaluationError("Evaluation outputs must not overwrite predictions or labels")

    prediction_hash_before = file_sha256(predictions_path)
    receipt = {
        "schema_version": "r3_distilled_prediction_hash_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "before_labels_opened",
        "split": {
            "salt": HOLDOUT_SALT,
            "fraction": HOLDOUT_FRACTION,
        },
        "prediction": {
            "path": str(predictions_path),
            "sha256": prediction_hash_before,
        },
    }
    _atomic_write_json(receipt_path, receipt)

    # Prediction structure and claimed split membership are checked without labels.
    predictions = load_predictions(predictions_path)

    if not labels_path.is_file():
        raise EvaluationError(f"Label workbook does not exist: {labels_path}")
    labels_hash_before = file_sha256(labels_path)
    labels, split_counts = load_holdout_labels(labels_path)

    prediction_hash_checkpoint = file_sha256(predictions_path)
    if prediction_hash_checkpoint != prediction_hash_before:
        raise EvaluationIntegrityError("Prediction JSONL changed after its receipt was persisted")
    labels_hash_checkpoint = file_sha256(labels_path)
    if labels_hash_checkpoint != labels_hash_before:
        raise EvaluationIntegrityError("Label workbook changed while it was being opened")

    comparison = _compare_identities(labels, predictions)
    metrics = calculate_metrics(comparison)

    prediction_hash_after = file_sha256(predictions_path)
    labels_hash_after = file_sha256(labels_path)
    if prediction_hash_after != prediction_hash_before:
        raise EvaluationIntegrityError("Prediction JSONL changed during evaluation")
    if labels_hash_after != labels_hash_before:
        raise EvaluationIntegrityError("Label workbook changed during evaluation")

    report = {
        "schema_version": "r3_distilled_holdout_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_contract": {
            "indicator": "R3",
            "prediction_receipt_persisted_before_labels_opened": True,
            "scoring_code_called": False,
            "record_level_answers_written": False,
            "predictions_restricted_to_preregistered_holdout": True,
            "prediction_files_modified": False,
            "metric_denominators": {
                "accuracy": "all preregistered holdout rows; compatibility values included",
                "balanced_accuracy": "mean recall across available binary classes",
                "coverage": "non-ABSTAIN/non-UNKNOWN predictions / holdout rows",
                "known_accuracy": "correct non-ABSTAIN/non-UNKNOWN predictions / known rows",
                "macro_f1": "mean F1 across available binary classes; compatibility values included",
            },
        },
        "split": {
            "method": "sha256_threshold_by_refno_v1",
            "salt": HOLDOUT_SALT,
            "fraction": HOLDOUT_FRACTION,
            "assignment_rule": (
                "uint64_be(SHA256(UTF8(salt + NUL + REFNO))[0:8]) / 2^64 < fraction"
            ),
            **split_counts,
        },
        "sources": {
            "predictions": {
                "path": str(predictions_path),
                "sha256_before_labels": prediction_hash_before,
                "sha256_after_evaluation": prediction_hash_after,
            },
            "labels": {
                "path": str(labels_path),
                "sha256_before_evaluation": labels_hash_before,
                "sha256_after_evaluation": labels_hash_after,
            },
        },
        "prediction_hash_receipt": {
            "path": str(receipt_path),
            "sha256": file_sha256(receipt_path),
        },
        "metrics": metrics,
    }
    _atomic_write_json(output_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate(args.predictions, args.labels, args.output)
    except (EvaluationError, OSError) as exc:
        print(f"R3 evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"split": report["split"], "metrics": report["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
