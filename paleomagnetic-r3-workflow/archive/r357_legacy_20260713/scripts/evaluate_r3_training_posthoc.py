#!/usr/bin/env python3
"""Post-hoc evaluation of frozen R3 predictions on the training partition.

This evaluator is deliberately downstream of scoring.  It hashes every
prediction file and persists one receipt before opening the label workbook.
Only the fixed, non-holdout REFNO partition is accepted, and reports contain
aggregate metrics rather than record-level answers.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import pandas as pd

try:
    from .evaluate_r3_distilled_predictions import (
        HOLDOUT_FRACTION,
        HOLDOUT_SALT,
        UNKNOWN_OUTCOMES,
        calculate_metrics,
        is_holdout_refno,
    )
except ImportError:
    from evaluate_r3_distilled_predictions import (
        HOLDOUT_FRACTION,
        HOLDOUT_SALT,
        UNKNOWN_OUTCOMES,
        calculate_metrics,
        is_holdout_refno,
    )


EXPECTED_TRAINING_ROWS = 2473


class TrainingEvaluationError(ValueError):
    """Raised when a training post-hoc input violates the frozen contract."""


class TrainingEvaluationIntegrityError(TrainingEvaluationError):
    """Raised when a protected input changes after fingerprinting."""


def _zip_equal(*sequences: Sequence[Any]):
    """Provide Python 3.9-compatible strict zip semantics."""

    lengths = {len(sequence) for sequence in sequences}
    if len(lengths) > 1:
        raise TrainingEvaluationIntegrityError(
            "Internal evaluation sequences have inconsistent lengths"
        )
    return zip(*sequences)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: R3_RESULT_JSON is invalid JSON: {exc}"
                ) from exc
        if payload is not None and not isinstance(payload, dict):
            raise TrainingEvaluationError(
                f"{path}:{line_number}: R3_RESULT_JSON must be an object"
            )
        if isinstance(payload, dict):
            raw = payload.get("outcome")
    return _clean(raw).upper() or "DECIDED"


def load_training_predictions(path: Path) -> pd.DataFrame:
    """Load one prediction set without consulting labels."""

    if path.suffix.casefold() != ".jsonl":
        raise TrainingEvaluationError("R3 training predictions must be JSONL files")

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: prediction must be an object"
                )
            missing = {"REFNO", "RESULTNO", "R3"} - set(row)
            if missing:
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: missing required prediction field(s)"
                )
            refno = _clean(row["REFNO"])
            resultno = _clean(row["RESULTNO"])
            if not refno or not resultno:
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: prediction identity must be non-empty"
                )
            value = row["R3"]
            if isinstance(value, bool) or value not in (0, 1):
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: R3 must be binary 0 or 1"
                )
            records.append(
                {
                    "REFNO": refno,
                    "RESULTNO": resultno,
                    "R3": int(value),
                    "OUTCOME": _prediction_outcome(row, path, line_number),
                }
            )

    if len(records) != EXPECTED_TRAINING_ROWS:
        raise TrainingEvaluationError(
            "Each prediction set must contain exactly "
            f"{EXPECTED_TRAINING_ROWS} training identities; found {len(records)}"
        )
    frame = pd.DataFrame.from_records(records)
    if frame["RESULTNO"].duplicated().any():
        raise TrainingEvaluationError(
            "Prediction RESULTNO values must be unique within each prediction set"
        )
    holdout_count = int(frame["REFNO"].map(is_holdout_refno).sum())
    if holdout_count:
        raise TrainingEvaluationError(
            "Prediction set contains preregistered holdout assignments: "
            f"count={holdout_count}"
        )
    return frame


def load_pdf_eligible_refnos(
    path: Path,
    required_refnos: set[str],
) -> tuple[set[str], dict[str, int]]:
    """Freeze a PDF-only evaluation set before labels are opened."""

    if path.suffix.casefold() != ".jsonl" or not path.is_file():
        raise TrainingEvaluationError("PDF-only page cache must be an existing JSONL file")
    found: set[str] = set()
    eligible: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: invalid page-cache JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: page-cache row must be an object"
                )
            refno = _clean(row.get("REFNO"))
            if not refno or refno in found:
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: page-cache REFNO is empty or duplicated"
                )
            found.add(refno)
            source = row.get("source")
            if not isinstance(source, dict):
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: page-cache source must be an object"
                )
            pdf_path = _clean(source.get("pdf_path"))
            raw_flags = source.get("source_quality_flags", [])
            if not isinstance(raw_flags, list):
                raise TrainingEvaluationError(
                    f"{path}:{line_number}: source_quality_flags must be a list"
                )
            flags = {_clean(flag) for flag in raw_flags}
            if pdf_path and "no_local_pdf" not in flags:
                eligible.add(refno)
    missing = sorted(required_refnos - found)
    extra = sorted(found - required_refnos)
    if missing or extra:
        raise TrainingEvaluationError(
            "PDF-only page cache REFNO set must match predictions: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return eligible, {
        "all_training_papers": len(required_refnos),
        "pdf_eligible_papers": len(eligible),
        "skipped_no_pdf_papers": len(required_refnos - eligible),
    }


def load_training_labels(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Open labels only after prediction receipts and preflight checks exist."""

    try:
        labels = pd.read_excel(
            path,
            usecols=["REFNO", "RESULTNO", "R3"],
            dtype={"REFNO": str, "RESULTNO": str},
        )
    except (ValueError, KeyError) as exc:
        raise TrainingEvaluationError(
            f"Could not read REFNO, RESULTNO, and R3 labels: {exc}"
        ) from exc

    for column in ("REFNO", "RESULTNO"):
        labels[column] = labels[column].map(_clean)
    if labels.empty or labels[["REFNO", "RESULTNO"]].eq("").any(axis=None):
        raise TrainingEvaluationError(
            "Label workbook contains empty or no prediction identities"
        )
    if labels["RESULTNO"].duplicated().any():
        raise TrainingEvaluationError("Label workbook contains duplicate RESULTNO values")

    holdout_mask = labels["REFNO"].map(is_holdout_refno)
    training = labels.loc[~holdout_mask].copy()
    if len(training) != EXPECTED_TRAINING_ROWS:
        raise TrainingEvaluationError(
            "Label workbook must define exactly "
            f"{EXPECTED_TRAINING_ROWS} fixed training identities; found {len(training)}"
        )
    numeric = pd.to_numeric(training["R3"], errors="coerce")
    if not numeric.isin([0, 1]).all():
        raise TrainingEvaluationError("Every training row must have a binary R3 label")
    training["R3"] = numeric.astype(int)
    return training, {
        "all_rows": int(len(labels)),
        "all_unique_paper_groups": int(labels["REFNO"].nunique()),
        "training_rows": int(len(training)),
        "training_unique_paper_groups": int(training["REFNO"].nunique()),
        "holdout_rows_excluded": int(holdout_mask.sum()),
        "holdout_unique_paper_groups_excluded": int(
            labels.loc[holdout_mask, "REFNO"].nunique()
        ),
    }


def _compare_identities(labels: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    label_resultnos = set(labels["RESULTNO"])
    prediction_resultnos = set(predictions["RESULTNO"])
    missing_count = len(label_resultnos - prediction_resultnos)
    extra_count = len(prediction_resultnos - label_resultnos)
    if missing_count or extra_count:
        raise TrainingEvaluationError(
            "Prediction set must exactly equal the labelled training identity set: "
            f"missing={missing_count}, extra={extra_count}"
        )

    comparison = labels.merge(
        predictions,
        on="RESULTNO",
        how="inner",
        suffixes=("_LABEL", "_PREDICTION"),
        validate="one_to_one",
    )
    mismatch_count = int(
        comparison["REFNO_LABEL"].ne(comparison["REFNO_PREDICTION"]).sum()
    )
    if mismatch_count:
        raise TrainingEvaluationError(
            "Prediction paper identities disagree with the label identities: "
            f"count={mismatch_count}"
        )
    return comparison


def _verify_prediction_hashes(
    paths: Sequence[Path], expected_hashes: Sequence[str], message: str
) -> list[str]:
    actual_hashes = [file_sha256(path) for path in paths]
    if actual_hashes != list(expected_hashes):
        raise TrainingEvaluationIntegrityError(message)
    return actual_hashes


def _change_counts(
    prediction_sets: Sequence[pd.DataFrame], prediction_ids: Sequence[str]
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for left_index, right_index in itertools.combinations(
        range(len(prediction_sets)), 2
    ):
        left = prediction_sets[left_index].set_index("RESULTNO")
        right = prediction_sets[right_index].set_index("RESULTNO").reindex(left.index)
        left_known = ~left["OUTCOME"].isin(UNKNOWN_OUTCOMES)
        right_known = ~right["OUTCOME"].isin(UNKNOWN_OUTCOMES)
        changes.append(
            {
                "from_prediction_id": prediction_ids[left_index],
                "to_prediction_id": prediction_ids[right_index],
                "compared_rows": int(len(left)),
                "r3_changed_rows": int(left["R3"].ne(right["R3"]).sum()),
                "r3_zero_to_one_rows": int(
                    ((left["R3"] == 0) & (right["R3"] == 1)).sum()
                ),
                "r3_one_to_zero_rows": int(
                    ((left["R3"] == 1) & (right["R3"] == 0)).sum()
                ),
                "outcome_changed_rows": int(
                    left["OUTCOME"].ne(right["OUTCOME"]).sum()
                ),
                "known_status_changed_rows": int(left_known.ne(right_known).sum()),
            }
        )
    return changes


def evaluate(
    predictions_paths: Path | Sequence[Path],
    labels_path: Path,
    output_path: Path,
    *,
    include_change_counts: bool = False,
    page_cache_path: Path | None = None,
    pdf_only: bool = False,
) -> dict[str, Any]:
    """Evaluate complete frozen training prediction sets after receipting them."""

    if isinstance(predictions_paths, Path):
        supplied_predictions = [predictions_paths]
    else:
        supplied_predictions = list(predictions_paths)
    if not supplied_predictions:
        raise TrainingEvaluationError("At least one prediction JSONL is required")
    resolved_predictions = [
        Path(path).expanduser().resolve(strict=False) for path in supplied_predictions
    ]
    labels_path = labels_path.expanduser().resolve(strict=False)
    output_path = output_path.expanduser().resolve(strict=False)
    page_cache_path = (
        page_cache_path.expanduser().resolve(strict=False)
        if page_cache_path is not None
        else None
    )
    if pdf_only and page_cache_path is None:
        raise TrainingEvaluationError("pdf_only evaluation requires a page cache")
    if page_cache_path is not None and not page_cache_path.is_file():
        raise TrainingEvaluationError(f"Page cache does not exist: {page_cache_path}")
    receipt_path = output_path.with_suffix(output_path.suffix + ".prediction_hashes.json")

    if len(set(resolved_predictions)) != len(resolved_predictions):
        raise TrainingEvaluationError("Prediction paths must be unique")
    for path in resolved_predictions:
        if not path.is_file():
            raise TrainingEvaluationError(f"Prediction JSONL does not exist: {path}")
    protected = set(resolved_predictions) | {labels_path}
    if page_cache_path is not None:
        protected.add(page_cache_path)
    output_paths = {
        output_path,
        output_path.with_suffix(output_path.suffix + ".tmp"),
        receipt_path,
        receipt_path.with_suffix(receipt_path.suffix + ".tmp"),
    }
    if protected & output_paths:
        raise TrainingEvaluationError(
            "Evaluation outputs must not overwrite predictions or labels"
        )

    prediction_hashes_before = [file_sha256(path) for path in resolved_predictions]
    page_cache_hash_before = (
        file_sha256(page_cache_path) if page_cache_path is not None else None
    )
    prediction_ids = [
        f"prediction_{index}" for index in range(1, len(resolved_predictions) + 1)
    ]
    receipt = {
        "schema_version": "r3_training_prediction_hash_receipt_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "before_labels_opened",
        "split": {
            "salt": HOLDOUT_SALT,
            "holdout_fraction": HOLDOUT_FRACTION,
            "partition": "training",
            "expected_training_rows": EXPECTED_TRAINING_ROWS,
        },
        "predictions": [
            {
                "prediction_id": prediction_id,
                "path": str(path),
                "sha256": sha256,
            }
            for prediction_id, path, sha256 in _zip_equal(
                prediction_ids,
                resolved_predictions,
                prediction_hashes_before,
            )
        ],
        "pdf_only_scope": (
            {
                "enabled": pdf_only,
                "page_cache_path": str(page_cache_path),
                "page_cache_sha256": page_cache_hash_before,
                "selection_rule": "non-empty source.pdf_path and no no_local_pdf flag",
            }
            if page_cache_path is not None
            else {"enabled": False}
        ),
    }
    _atomic_write_json(receipt_path, receipt)

    # No label existence check, hash, or workbook read is permitted before this
    # prediction-only preflight has completed behind the persisted receipt.
    prediction_sets = [load_training_predictions(path) for path in resolved_predictions]
    required_refnos = set(prediction_sets[0]["REFNO"])
    pdf_eligible_refnos: set[str] | None = None
    pdf_scope_counts: dict[str, int] | None = None
    if page_cache_path is not None:
        pdf_eligible_refnos, pdf_scope_counts = load_pdf_eligible_refnos(
            page_cache_path,
            required_refnos,
        )
        if file_sha256(page_cache_path) != page_cache_hash_before:
            raise TrainingEvaluationIntegrityError(
                "Page cache changed while the PDF-only scope was frozen"
            )
    _verify_prediction_hashes(
        resolved_predictions,
        prediction_hashes_before,
        "A prediction JSONL changed after its receipt was persisted",
    )

    if not labels_path.is_file():
        raise TrainingEvaluationError(f"Label workbook does not exist: {labels_path}")
    labels_hash_before = file_sha256(labels_path)
    labels, split_counts = load_training_labels(labels_path)

    prediction_hashes_checkpoint = _verify_prediction_hashes(
        resolved_predictions,
        prediction_hashes_before,
        "A prediction JSONL changed while labels were being opened",
    )
    labels_hash_checkpoint = file_sha256(labels_path)
    if labels_hash_checkpoint != labels_hash_before:
        raise TrainingEvaluationIntegrityError(
            "Label workbook changed while it was being opened"
        )

    comparisons = [
        _compare_identities(labels, predictions) for predictions in prediction_sets
    ]
    if pdf_only:
        assert pdf_eligible_refnos is not None
        comparisons = [
            comparison.loc[
                comparison["REFNO_LABEL"].isin(pdf_eligible_refnos)
            ].copy()
            for comparison in comparisons
        ]
    metrics = [calculate_metrics(comparison) for comparison in comparisons]
    changes = (
        _change_counts(prediction_sets, prediction_ids)
        if include_change_counts and len(prediction_sets) > 1
        else None
    )

    prediction_hashes_after = _verify_prediction_hashes(
        resolved_predictions,
        prediction_hashes_before,
        "A prediction JSONL changed during evaluation",
    )
    labels_hash_after = file_sha256(labels_path)
    if labels_hash_after != labels_hash_before:
        raise TrainingEvaluationIntegrityError(
            "Label workbook changed during evaluation"
        )
    page_cache_hash_after = (
        file_sha256(page_cache_path) if page_cache_path is not None else None
    )
    if page_cache_hash_after != page_cache_hash_before:
        raise TrainingEvaluationIntegrityError("Page cache changed during evaluation")

    report: dict[str, Any] = {
        "schema_version": "r3_training_posthoc_evaluation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_contract": {
            "indicator": "R3",
            "prediction_receipt_persisted_before_labels_opened": True,
            "partition": "training_only",
            "record_level_answers_written": False,
            "scoring_code_called": False,
            "prediction_files_modified": False,
            "label_file_modified": False,
            "metric_definitions": "evaluate_r3_distilled_predictions.calculate_metrics",
        },
        "split": {
            "method": "sha256_threshold_by_refno_v1",
            "salt": HOLDOUT_SALT,
            "holdout_fraction": HOLDOUT_FRACTION,
            "partition": "training",
            "assignment_rule": (
                "uint64_be(SHA256(UTF8(salt + NUL + paper_id))[0:8]) / 2^64 "
                ">= holdout_fraction"
            ),
            "expected_training_rows": EXPECTED_TRAINING_ROWS,
            **split_counts,
        },
        "sources": {
            "predictions": [
                {
                    "prediction_id": prediction_id,
                    "path": str(path),
                    "sha256_before_labels": before,
                    "sha256_at_label_checkpoint": checkpoint,
                    "sha256_after_evaluation": after,
                }
                for prediction_id, path, before, checkpoint, after in _zip_equal(
                    prediction_ids,
                    resolved_predictions,
                    prediction_hashes_before,
                    prediction_hashes_checkpoint,
                    prediction_hashes_after,
                )
            ],
            "labels": {
                "path": str(labels_path),
                "sha256_before_evaluation": labels_hash_before,
                "sha256_at_open_checkpoint": labels_hash_checkpoint,
                "sha256_after_evaluation": labels_hash_after,
            },
            "page_cache": (
                {
                    "path": str(page_cache_path),
                    "sha256_before_labels": page_cache_hash_before,
                    "sha256_after_evaluation": page_cache_hash_after,
                }
                if page_cache_path is not None
                else None
            ),
        },
        "evaluation_scope": {
            "pdf_only": pdf_only,
            "selection_frozen_before_labels_opened": bool(pdf_only),
            "selection_rule": (
                "non-empty page-cache source.pdf_path and no no_local_pdf flag"
                if pdf_only
                else "all fixed training identities"
            ),
            **(pdf_scope_counts or {}),
            "evaluated_rows": int(len(comparisons[0])),
            "skipped_rows": int(EXPECTED_TRAINING_ROWS - len(comparisons[0])),
        },
        "prediction_hash_receipt": {
            "path": str(receipt_path),
            "sha256": file_sha256(receipt_path),
        },
        "prediction_sets": [
            {"prediction_id": prediction_id, "metrics": prediction_metrics}
            for prediction_id, prediction_metrics in _zip_equal(
                prediction_ids, metrics
            )
        ],
    }
    if changes is not None:
        report["prediction_change_counts"] = changes
    _atomic_write_json(output_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-change-counts", action="store_true")
    parser.add_argument("--page-cache", type=Path)
    parser.add_argument("--pdf-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = evaluate(
            args.predictions,
            args.labels,
            args.output,
            include_change_counts=args.include_change_counts,
            page_cache_path=args.page_cache,
            pdf_only=args.pdf_only,
        )
    except (TrainingEvaluationError, OSError) as exc:
        print(f"R3 training post-hoc evaluation error: {exc}", file=sys.stderr)
        return 2
    console_report: dict[str, Any] = {
        "split": report["split"],
        "evaluation_scope": report["evaluation_scope"],
        "prediction_sets": report["prediction_sets"],
    }
    if "prediction_change_counts" in report:
        console_report["prediction_change_counts"] = report[
            "prediction_change_counts"
        ]
    print(json.dumps(console_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
