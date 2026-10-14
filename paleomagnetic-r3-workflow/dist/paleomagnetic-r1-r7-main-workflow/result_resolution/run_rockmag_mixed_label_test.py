#!/usr/bin/env python3
"""Prepare and evaluate the mixed-label multi-target ROCKMAG test set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs/rockmag_mixed_label_multitarget_20260717"
IDENTITY = ROOT / "outputs/r357_strict_eval_07111758/inputs/identity_manifest.xlsx"
TARGETS = ROOT / "data/targets.xlsx"
PMAGRESULT = ROOT / "data/source_metadata/PMAGRESULT.xlsx"
PAGE_CACHE = ROOT / "tmp/r3_all_labelled_pdf_page_cache_v1/page_cache.jsonl"
PDF_DIR = ROOT / "data/raw_pdfs"
PAIRWISE_OOF = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1/grep_pairwise_rank_oof_v2.npz"


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def load_sheet(path: Path, sheet: str | None = None) -> list[dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[sheet] if sheet else workbook.worksheets[0]
        rows = worksheet.iter_rows(values_only=True)
        header = [clean(value) for value in next(rows, ())]
        return [
            {header[index]: value for index, value in enumerate(row[: len(header)])}
            for row in rows
            if any(value not in (None, "") for value in row)
        ]
    finally:
        workbook.close()


def labelled_pdf_records() -> tuple[list[dict[str, str]], dict[str, int]]:
    identities = [
        {field: clean(row.get(field)) for field in ("REFNO", "RESULTNO", "ROCKNAME")}
        for row in load_sheet(IDENTITY)
    ]
    label_map = {
        clean(row.get("RESULTNO")): int(float(row["R3"]))
        for row in load_sheet(TARGETS)
        if clean(row.get("RESULTNO")) and row.get("R3") not in (None, "")
    }
    pdf_refnos = {path.stem for path in PDF_DIR.glob("*.pdf")}
    records = [row for row in identities if row["REFNO"] in pdf_refnos]
    if len(records) != 2751 or len({row["REFNO"] for row in records}) != 1478:
        raise ValueError("frozen PDF-backed identity membership changed")
    if any(row["RESULTNO"] not in label_map for row in records):
        raise ValueError("R3 labels do not align with PDF-backed identities")
    return records, label_map


def mixed_label_records() -> tuple[list[dict[str, str]], dict[str, int]]:
    records, label_map = labelled_pdf_records()
    by_refno: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in records:
        by_refno[row["REFNO"]].append(row)
    mixed_refnos = {
        refno
        for refno, rows in by_refno.items()
        if len(rows) >= 2 and len({label_map[row["RESULTNO"]] for row in rows}) > 1
    }
    selected = [row for row in records if row["REFNO"] in mixed_refnos]
    if len(selected) != 278 or len(mixed_refnos) != 61:
        raise ValueError("expected the frozen mixed-label subset to contain 278 rows / 61 papers")
    return selected, label_map


def prepare(output_dir: Path) -> dict[str, Any]:
    selected, label_map = mixed_label_records()
    refnos = {row["REFNO"] for row in selected}
    base_manifest = output_dir / "identity_manifest.json"
    manifest = {
        "schema_version": "rockmag_mixed_label_multitarget_identity_v1",
        "selection_definition": (
            "PDF-backed REFNO with at least two labelled targets and both historical R3=0 and R3=1"
        ),
        "selection_label_use": (
            "R3 used only to define test-set membership; no row label is present in predictor inputs"
        ),
        "selected_refnos": sorted(refnos, key=lambda value: int(value)),
        "papers": len(refnos),
        "targets": len(selected),
        "identity_fields": ["REFNO", "RESULTNO", "ROCKUNITNO", "ROCKNAME"],
        "identities": [
            {
                "REFNO": row["REFNO"],
                "RESULTNO": row["RESULTNO"],
                "ROCKUNITNO": "",
                "ROCKNAME": row["ROCKNAME"],
            }
            for row in selected
        ],
    }

    # Fill ROCKUNITNO from PMAGRESULT without loading ROCKMAG or other forbidden columns.
    workbook = load_workbook(PMAGRESULT, read_only=True, data_only=True)
    try:
        worksheet = workbook["PMAGRESULT"]
        rows = worksheet.iter_rows(values_only=True)
        header = [clean(value) for value in next(rows, ())]
        result_index = header.index("RESULTNO")
        unit_index = header.index("ROCKUNITNO")
        unit_by_result = {
            clean(row[result_index]): clean(row[unit_index])
            for row in rows
            if clean(row[result_index]) in {item["RESULTNO"] for item in manifest["identities"]}
        }
    finally:
        workbook.close()
    for row in manifest["identities"]:
        row["ROCKUNITNO"] = unit_by_result.get(row["RESULTNO"], "")
        if not row["ROCKUNITNO"]:
            raise ValueError(f"missing ROCKUNITNO for RESULTNO {row['RESULTNO']}")
    atomic_json(base_manifest, manifest)

    parsed_dir = output_dir / "parsed_papers"
    parsed_dir.mkdir(parents=True, exist_ok=True)
    found: set[str] = set()
    page_counts: dict[str, int] = {}
    with PAGE_CACHE.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            refno = clean(record.get("REFNO"))
            if refno not in refnos:
                continue
            if refno in found:
                raise ValueError(f"duplicate page-cache REFNO: {refno}")
            found.add(refno)
            pages = record.get("pages") or []
            page_counts[refno] = len(pages)
            text = "\n\n".join(
                f"### page_{int(page['page'])}\n{clean(page.get('text'))}" for page in pages
            ).rstrip() + "\n"
            target = parsed_dir / f"{refno}.txt"
            temporary = target.with_suffix(".txt.tmp")
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(target)
    if found != refnos:
        raise ValueError(f"page cache missing REFNO values: {sorted(refnos - found)}")

    audit = {
        "schema_version": "rockmag_mixed_label_multitarget_selection_audit_v1",
        "status": "PASS",
        "papers": len(refnos),
        "targets": len(selected),
        "historical_r3_distribution": dict(
            sorted(Counter(label_map[row["RESULTNO"]] for row in selected).items())
        ),
        "paper_text_coverage": {
            "available_papers": len(found),
            "required_papers": len(refnos),
            "coverage": len(found) / len(refnos),
            "total_pages": sum(page_counts.values()),
        },
        "predictor_input_excludes": [
            "PMAGRESULT.ROCKMAG",
            "PMAGRESULT.TESTS",
            "historical row R3",
            "PMAGRESULT.LABDETAILS",
            "PMAGRESULT.COMMENTS",
            "prior predictions",
            "evaluation reports",
        ],
        "source_sha256": {
            str(IDENTITY): file_sha256(IDENTITY),
            str(TARGETS): file_sha256(TARGETS),
            str(PMAGRESULT): file_sha256(PMAGRESULT),
            str(PAGE_CACHE): file_sha256(PAGE_CACHE),
        },
        "output": {
            "identity_manifest": str(base_manifest),
            "identity_manifest_sha256": file_sha256(base_manifest),
            "parsed_dir": str(parsed_dir),
        },
    }
    atomic_json(output_dir / "selection_audit.json", audit)
    return audit


def binary_metrics(truth: list[int], predicted: list[int]) -> dict[str, Any]:
    if len(truth) != len(predicted):
        raise ValueError("metric inputs differ in length")
    tn = sum(a == 0 and b == 0 for a, b in zip(truth, predicted))
    fp = sum(a == 0 and b == 1 for a, b in zip(truth, predicted))
    fn = sum(a == 1 and b == 0 for a, b in zip(truth, predicted))
    tp = sum(a == 1 and b == 1 for a, b in zip(truth, predicted))
    divide = lambda numerator, denominator: numerator / denominator if denominator else 0.0
    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    specificity = divide(tn, tn + fp)
    f1 = divide(2 * precision * recall, precision + recall)
    negative_precision = divide(tn, tn + fn)
    negative_recall = specificity
    negative_f1 = divide(
        2 * negative_precision * negative_recall, negative_precision + negative_recall
    )
    return {
        "rows": len(truth),
        "accuracy": divide(tp + tn, len(truth)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "balanced_accuracy": (recall + specificity) / 2,
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def canonical_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def evidence_summary(prediction: dict[str, Any]) -> tuple[str, str]:
    evidence = prediction.get("evidence") or []
    pages = sorted({int(item["page"]) for item in evidence if item.get("page") is not None})
    quotes = []
    for item in evidence[:3]:
        quote = clean(item.get("quote") or item.get("matched_text"))
        if quote:
            quotes.append(f"p.{item.get('page')}: {quote[:500]}")
    return ", ".join(map(str, pages)), " | ".join(quotes)


def evaluate(output_dir: Path, predictions_path: Path) -> dict[str, Any]:
    manifest_path = output_dir / "identity_manifest.locators.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identities = manifest["identities"]
    selected_resultnos = {row["RESULTNO"] for row in identities}
    predictions = [json.loads(line) for line in predictions_path.open(encoding="utf-8") if line.strip()]
    prediction_by_result = {clean(row.get("RESULTNO")): row for row in predictions}
    if set(prediction_by_result) != selected_resultnos or len(predictions) != len(selected_resultnos):
        raise ValueError("predictions do not align one-to-one with the frozen identity manifest")

    all_records, label_map = labelled_pdf_records()
    selected = [row for row in all_records if row["RESULTNO"] in selected_resultnos]
    if [row["RESULTNO"] for row in selected] != [row["RESULTNO"] for row in identities]:
        raise ValueError("selected identity row order changed")

    pmag_rows = load_sheet(PMAGRESULT, "PMAGRESULT")
    historical_rockmag = {
        clean(row.get("RESULTNO")): clean(row.get("ROCKMAG"))
        for row in pmag_rows
        if clean(row.get("RESULTNO")) in selected_resultnos
    }
    if set(historical_rockmag) != selected_resultnos:
        raise ValueError("historical PMAGRESULT rows do not align with selection")

    oof = np.load(PAIRWISE_OOF)
    baseline = oof["grep_expanded_mean4_recomputed"]
    pairwise = oof["grep_pairwise_nested_router"]
    if baseline.shape != (len(all_records),) or pairwise.shape != (len(all_records),):
        raise ValueError("prior OOF arrays do not align with PDF-backed identity")
    prior_by_result = {
        row["RESULTNO"]: (float(baseline[index]), float(pairwise[index]))
        for index, row in enumerate(all_records)
    }

    name_counts = Counter((row["REFNO"], canonical_name(row["ROCKNAME"])) for row in identities)
    details: list[dict[str, Any]] = []
    for identity in identities:
        resultno = identity["RESULTNO"]
        prediction = prediction_by_result[resultno]
        actual_rockmag = historical_rockmag[resultno]
        truth_presence = int(bool(actual_rockmag))
        predicted_presence = int(prediction.get("prediction_status") == "PRESENT")
        historical_r3 = label_map[resultno]
        baseline_probability, pairwise_probability = prior_by_result[resultno]
        pages, quotes = evidence_summary(prediction)
        details.append(
            {
                **{field: clean(identity.get(field)) for field in identity},
                "historical_rockmag": actual_rockmag,
                "truth_rockmag_present": truth_presence,
                "historical_r3": historical_r3,
                "rockmag_presence_equals_r3": int(truth_presence == historical_r3),
                "prediction_status": clean(prediction.get("prediction_status")),
                "predicted_rockmag": clean(prediction.get("predicted_rockmag")),
                "candidate_rockmag": clean(prediction.get("candidate_rockmag")),
                "predicted_present_direct_r3": predicted_presence,
                "rockmag_presence_correct": int(predicted_presence == truth_presence),
                "direct_r3_correct": int(predicted_presence == historical_r3),
                "decided": int(prediction.get("prediction_status") in {"PRESENT", "ABSENT"}),
                "confidence": clean(prediction.get("confidence")),
                "locator_resolution": clean(prediction.get("locator_resolution")),
                "locator_metadata_unique": bool(prediction.get("locator_metadata_unique")),
                "locator_component_unique": bool(prediction.get("locator_component_unique")),
                "duplicate_same_name_within_paper": int(
                    name_counts[(identity["REFNO"], canonical_name(identity["ROCKNAME"]))] > 1
                ),
                "methods": ",".join(prediction.get("methods") or []),
                "carriers": ",".join(prediction.get("carriers") or []),
                "domains": ",".join(prediction.get("domains") or []),
                "evidence_pages": pages,
                "evidence_excerpt": quotes,
                "old_baseline_probability": baseline_probability,
                "old_baseline_prediction": int(baseline_probability >= 0.5),
                "old_baseline_correct": int((baseline_probability >= 0.5) == historical_r3),
                "old_pairwise_probability": pairwise_probability,
                "old_pairwise_prediction": int(pairwise_probability >= 0.5),
                "old_pairwise_correct": int((pairwise_probability >= 0.5) == historical_r3),
            }
        )

    truth_presence = [row["truth_rockmag_present"] for row in details]
    historical_r3 = [row["historical_r3"] for row in details]
    predicted = [row["predicted_present_direct_r3"] for row in details]
    decided = [row for row in details if row["decided"]]
    actual_present_predicted_present = [
        row for row in details if row["truth_rockmag_present"] and row["predicted_present_direct_r3"]
    ]
    exact_nonempty = sum(
        row["predicted_rockmag"].casefold() == row["historical_rockmag"].casefold()
        for row in actual_present_predicted_present
    )

    by_refno: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in details:
        by_refno[row["REFNO"]].append(row)
    papers: list[dict[str, Any]] = []
    for refno, rows in by_refno.items():
        papers.append(
            {
                "REFNO": refno,
                "targets": len(rows),
                "historical_r3_0": sum(row["historical_r3"] == 0 for row in rows),
                "historical_r3_1": sum(row["historical_r3"] == 1 for row in rows),
                "historical_rockmag_nonempty": sum(row["truth_rockmag_present"] for row in rows),
                "predicted_present": sum(row["predicted_present_direct_r3"] for row in rows),
                "decided": sum(row["decided"] for row in rows),
                "abstained": sum(not row["decided"] for row in rows),
                "rockmag_presence_correct": sum(row["rockmag_presence_correct"] for row in rows),
                "direct_r3_correct": sum(row["direct_r3_correct"] for row in rows),
                "old_baseline_correct": sum(row["old_baseline_correct"] for row in rows),
                "old_pairwise_correct": sum(row["old_pairwise_correct"] for row in rows),
                "all_direct_r3_correct": int(all(row["direct_r3_correct"] for row in rows)),
                "constant_direct_prediction": int(
                    len({row["predicted_present_direct_r3"] for row in rows}) == 1
                ),
            }
        )

    summary = {
        "schema_version": "rockmag_mixed_label_multitarget_evaluation_v1",
        "status": "complete",
        "selection": {
            "papers": len(by_refno),
            "targets": len(details),
            "definition": "same REFNO has at least two targets and contains historical R3=0 and R3=1",
            "paper_text_coverage": 1.0,
        },
        "prediction_coverage": {
            "status_counts": dict(sorted(Counter(row["prediction_status"] for row in details).items())),
            "decided_rows": len(decided),
            "abstained_rows": len(details) - len(decided),
            "decided_rate": len(decided) / len(details),
        },
        "ground_truth_relationship": {
            "historical_r3_counts": dict(sorted(Counter(historical_r3).items())),
            "historical_rockmag_nonempty": sum(truth_presence),
            "historical_rockmag_empty": len(details) - sum(truth_presence),
            "rockmag_presence_as_r3": binary_metrics(historical_r3, truth_presence),
            "mismatch_rows": sum(a != b for a, b in zip(historical_r3, truth_presence)),
        },
        "rockmag_presence_prediction": {
            "full_set": binary_metrics(truth_presence, predicted),
            "decided_only": binary_metrics(
                [row["truth_rockmag_present"] for row in decided],
                [row["predicted_present_direct_r3"] for row in decided],
            ),
            "exact_string_match_among_true_and_predicted_present": {
                "matches": exact_nonempty,
                "rows": len(actual_present_predicted_present),
                "rate": exact_nonempty / len(actual_present_predicted_present)
                if actual_present_predicted_present
                else 0.0,
            },
        },
        "direct_r3_prediction": {
            "full_set": binary_metrics(historical_r3, predicted),
            "decided_only": binary_metrics(
                [row["historical_r3"] for row in decided],
                [row["predicted_present_direct_r3"] for row in decided],
            ),
            "all_targets_correct_papers": sum(row["all_direct_r3_correct"] for row in papers),
            "constant_prediction_papers": sum(row["constant_direct_prediction"] for row in papers),
        },
        "prior_table_comparison": {
            "old_expanded_baseline": binary_metrics(
                historical_r3, [row["old_baseline_prediction"] for row in details]
            ),
            "old_pairwise_router": binary_metrics(
                historical_r3, [row["old_pairwise_prediction"] for row in details]
            ),
        },
        "identity_diagnostics": {
            "duplicate_same_name_rows": sum(row["duplicate_same_name_within_paper"] for row in details),
            "locator_resolution_counts": dict(
                sorted(Counter(row["locator_resolution"] for row in details).items())
            ),
            "metadata_unique_rows": sum(row["locator_metadata_unique"] for row in details),
        },
        "artifacts": {
            "identity_manifest": str(manifest_path),
            "identity_manifest_sha256": file_sha256(manifest_path),
            "predictions": str(predictions_path),
            "predictions_sha256": file_sha256(predictions_path),
        },
    }
    summary["prior_table_comparison"]["direct_r3_delta_vs_old_baseline"] = (
        summary["direct_r3_prediction"]["full_set"]["accuracy"]
        - summary["prior_table_comparison"]["old_expanded_baseline"]["accuracy"]
    )
    summary["prior_table_comparison"]["direct_r3_delta_vs_old_pairwise"] = (
        summary["direct_r3_prediction"]["full_set"]["accuracy"]
        - summary["prior_table_comparison"]["old_pairwise_router"]["accuracy"]
    )
    atomic_json(output_dir / "evaluation_summary.json", summary)
    atomic_jsonl(output_dir / "target_results.jsonl", details)
    atomic_jsonl(output_dir / "paper_summary.jsonl", papers)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    evaluate_parser.add_argument(
        "--predictions", type=Path, default=OUTPUT_DIR / "predictions.locator.v22.blind.jsonl"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.output_dir.resolve())
    else:
        result = evaluate(args.output_dir.resolve(), args.predictions.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
