#!/usr/bin/env python3
"""Run the reproducible R1-R7 full-target workflow.

This is the retained orchestration core. The public entrypoint is
``run_r1_r7_workflow.py``. It does not read previous final workbooks as row
answers. R1/R2/R4/R6 are calculated from source tables; R3 uses the frozen
historically supervised stable model over original-paper and knowledge
features by default; R5 and R7 retain their source-text workflows.
"""

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
DEFAULT_TARGETS = ROOT / "data/targetsall.xlsx"
DEFAULT_PARSED_DIR = ROOT / "data/parsed_mds"
DEFAULT_PDF_DIR = ROOT / "data/raw_pdfs"
DEFAULT_OUTPUTS = ROOT / "outputs"
DEFAULT_R3_EXTRACTS = (
    ROOT
    / "data/calibration/r3_source_recovered_structured_evidence_v9b.jsonl"
)
DEFAULT_ROCKUNIT = ROOT / "data/source_metadata/ROCKUNIT.xlsx"
DEFAULT_PMAGRESULT = ROOT / "data/source_metadata/PMAGRESULT.xlsx"
DEFAULT_R3_STABLE_MODEL = ROOT / "data/models/r3_stable_production_v1.joblib"
DEFAULT_R3_STABLE_MANIFEST = ROOT / "data/models/r3_stable_production_v1.manifest.json"
DEFAULT_R3_MODEL_SNAPSHOT = (
    ROOT
    / "data/models/bge-base-en-v1.5"
    / "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
)
ALL_TARGETS = ROOT / "data/targetsall.xlsx"
PDF_TARGETS = ROOT / "data/targetsall_with_pdfs.xlsx"
TARGET_ID_COLUMNS = ["RESULTNO", "ROCKUNITNO", "REFNO", "ROCKNAME"]


def validate_timestamp(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("timestamp must use MMDDHHMM, for example 07010632")
    return value


def run(command: list[str], cwd: Path = ROOT) -> None:
    print("$ " + " ".join(command))
    subprocess.run(command, cwd=cwd, check=True)


def normalized_identifiers(frame: pd.DataFrame, source: Path) -> pd.DataFrame:
    missing = [column for column in TARGET_ID_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{source} is missing target identity column(s): {', '.join(missing)}")
    result = frame[TARGET_ID_COLUMNS].fillna("").copy()
    for column in TARGET_ID_COLUMNS:
        result[column] = result[column].astype(str).str.strip()
    empty = {column: int(result[column].eq("").sum()) for column in TARGET_ID_COLUMNS}
    empty = {column: count for column, count in empty.items() if count}
    if empty:
        raise SystemExit(f"{source} contains empty target identities: {empty}")
    if result["RESULTNO"].duplicated().any():
        examples = ", ".join(result.loc[result["RESULTNO"].duplicated(), "RESULTNO"].head(10))
        raise SystemExit(f"{source} contains duplicate RESULTNO values: {examples}")
    return result


def build_r3_identity_manifest(targets: Path, output: Path) -> Path:
    identity_columns = ["REFNO", "RESULTNO", "ROCKNAME"]
    try:
        identity = pd.read_excel(targets, usecols=identity_columns, dtype=str)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"Could not read R3 identity columns from {targets}: {exc}") from exc
    identity = identity[identity_columns].fillna("").copy()
    for column in identity_columns:
        identity[column] = identity[column].astype(str).str.strip()
    if identity.empty or identity.eq("").any(axis=None):
        raise SystemExit(f"{targets} contains empty R3 identity values")
    if identity["RESULTNO"].duplicated().any():
        raise SystemExit(f"{targets} contains duplicate RESULTNO values")
    output.parent.mkdir(parents=True, exist_ok=True)
    identity.to_excel(output, index=False)
    return output


def expected_targets(rockunit: Path, pmagresult: Path) -> pd.DataFrame:
    pmag = pd.read_excel(
        pmagresult,
        usecols=["RESULTNO", "ROCKUNITNO"],
        dtype={"RESULTNO": str, "ROCKUNITNO": str},
    )
    rocks = pd.read_excel(
        rockunit,
        usecols=["ROCKUNITNO", "REFNO", "ROCKNAME"],
        dtype={"ROCKUNITNO": str, "REFNO": str, "ROCKNAME": str},
    )
    pmag = pmag.fillna("").copy()
    for column in ("RESULTNO", "ROCKUNITNO"):
        pmag[column] = pmag[column].astype(str).str.strip()
    if pmag["RESULTNO"].eq("").any() or pmag["RESULTNO"].duplicated().any():
        raise SystemExit(f"{pmagresult} must contain unique, non-empty RESULTNO values")
    if pmag["ROCKUNITNO"].eq("").any():
        raise SystemExit(f"{pmagresult} contains empty ROCKUNITNO values")
    rocks = rocks.fillna("").copy()
    for column in ("ROCKUNITNO", "REFNO", "ROCKNAME"):
        rocks[column] = rocks[column].astype(str).str.strip()
    if rocks["ROCKUNITNO"].eq("").any() or rocks["ROCKUNITNO"].duplicated().any():
        raise SystemExit(f"{rockunit} must contain unique, non-empty ROCKUNITNO values")
    merged = pmag.merge(rocks, on="ROCKUNITNO", how="left", validate="many_to_one")
    return normalized_identifiers(merged, pmagresult)


def validate_declared_target_scope(
    targets: Path,
    rockunit: Path,
    pmagresult: Path,
    pdf_dir: Path,
) -> str:
    resolved = targets.resolve()
    if resolved == ALL_TARGETS.resolve():
        scope = "all_resultno_targets"
    elif resolved == PDF_TARGETS.resolve():
        scope = "pdf_backed_targets"
    else:
        return "custom_or_labelled_targets"

    expected = expected_targets(rockunit, pmagresult)
    if scope == "pdf_backed_targets":
        pdf_refnos = {path.stem for path in pdf_dir.glob("*.pdf")}
        expected = expected[expected["REFNO"].isin(pdf_refnos)].copy()
    actual_source = pd.read_excel(targets, dtype=str)
    label_columns = sorted(set(actual_source.columns) & {f"R{index}" for index in range(1, 8)})
    if label_columns:
        raise SystemExit(f"{targets} must omit R1-R7 labels for {scope}: found {label_columns}")
    actual = normalized_identifiers(actual_source, targets)

    expected_ids = set(expected["RESULTNO"])
    actual_ids = set(actual["RESULTNO"])
    missing_ids = sorted(expected_ids - actual_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    extra_ids = sorted(actual_ids - expected_ids, key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    if missing_ids or extra_ids:
        raise SystemExit(
            f"{targets} failed {scope} preflight: expected {len(expected)} RESULTNO rows, "
            f"found {len(actual)}; missing={len(missing_ids)} {missing_ids[:10]}; "
            f"extra={len(extra_ids)} {extra_ids[:10]}"
        )

    comparison = actual.merge(
        expected,
        on="RESULTNO",
        how="left",
        suffixes=("_actual", "_expected"),
        validate="one_to_one",
    )
    mismatched = pd.Series(False, index=comparison.index)
    for column in ("ROCKUNITNO", "REFNO", "ROCKNAME"):
        mismatched |= comparison[f"{column}_actual"].ne(comparison[f"{column}_expected"])
    if mismatched.any():
        examples = comparison.loc[mismatched, "RESULTNO"].head(10).tolist()
        raise SystemExit(f"{targets} failed {scope} identity preflight for RESULTNO values: {examples}")

    print(json.dumps({"target_scope": scope, "rows": len(actual), "status": "validated"}))
    return scope


def newest_final(indicator: str, timestamp: str, output_dir: Path) -> Path:
    matches = sorted(output_dir.glob(f"{indicator.lower()}_{timestamp}_*.xlsx"))
    if not matches:
        raise SystemExit(f"No final {indicator} workbook was produced for timestamp {timestamp}.")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise SystemExit(f"Multiple final {indicator} workbooks matched timestamp {timestamp}: {names}")
    return matches[0]


def metrics(indicator: str, workbook: Path, targets: Path) -> dict[str, Any]:
    pred = pd.read_excel(workbook, dtype={"REFNO": str, "RESULTNO": str})
    payload_column = f"{indicator}_RESULT_JSON"
    outcome_by_result: dict[str, str] = {}
    if payload_column in pred.columns:
        for row in pred[["RESULTNO", payload_column]].to_dict(orient="records"):
            try:
                payload = json.loads(str(row.get(payload_column) or "{}"))
            except json.JSONDecodeError:
                payload = {}
            outcome_by_result[str(row.get("RESULTNO") or "")] = str(
                payload.get("outcome") or "DECIDED"
            ).upper()
    outcome_counts: dict[str, int] = {}
    for outcome in outcome_by_result.values():
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    target_columns = set(pd.read_excel(targets, nrows=0).columns)
    if {"RESULTNO", indicator}.issubset(target_columns):
        standard = pd.read_excel(targets, usecols=["RESULTNO", indicator], dtype={"RESULTNO": str})
        merged = pred.merge(standard.rename(columns={indicator: f"STANDARD_{indicator}"}), on="RESULTNO", how="left")
        labelled = merged[f"STANDARD_{indicator}"].isin([0, 1])
        compared = merged[labelled].copy()
        correct = int(compared[indicator].eq(compared[f"STANDARD_{indicator}"]).sum())
        total = int(len(compared))
        tp = int(((compared[indicator] == 1) & (compared[f"STANDARD_{indicator}"] == 1)).sum())
        tn = int(((compared[indicator] == 0) & (compared[f"STANDARD_{indicator}"] == 0)).sum())
        fp = int(((compared[indicator] == 1) & (compared[f"STANDARD_{indicator}"] == 0)).sum())
        fn = int(((compared[indicator] == 0) & (compared[f"STANDARD_{indicator}"] == 1)).sum())
        known = compared["RESULTNO"].map(outcome_by_result).fillna("DECIDED").ne("UNKNOWN")
        known_rows = int(known.sum())
        known_correct = int(
            compared.loc[known, indicator]
            .eq(compared.loc[known, f"STANDARD_{indicator}"])
            .sum()
        )
    else:
        correct = 0
        total = 0
        tp = tn = fp = fn = 0
        known_rows = sum(outcome != "UNKNOWN" for outcome in outcome_by_result.values())
        known_correct = 0
    return {
        "workbook": str(workbook),
        "html": str(workbook.with_suffix(".html")),
        "rows": int(len(pred)),
        "labelled_rows": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "known_rows": known_rows,
        "unknown_rows": int(sum(outcome == "UNKNOWN" for outcome in outcome_by_result.values())),
        "known_accuracy": known_correct / known_rows if total and known_rows else None,
        "score_counts": {str(k): int(v) for k, v in pred[indicator].value_counts().sort_index().items()},
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--parsed-dir", type=Path, default=DEFAULT_PARSED_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--r3-extracts", type=Path, default=None)
    parser.add_argument(
        "--r3-source-mode",
        choices=["fresh", "paper_only", "retained", "supervised_stable"],
        default="supervised_stable",
        help=(
            "supervised_stable applies the frozen historically supervised stable model to "
            "original-paper and knowledge features while retaining paper-only evidence; fresh "
            "rebuilds metadata-augmented evidence; paper_only uses original-paper rules only; "
            "retained uses --r3-extracts."
        ),
    )
    parser.add_argument(
        "--r3-policy",
        default=None,
        help="R3 arbitration policy. Defaults to metadata_v1 for fresh mode and v9 for retained mode.",
    )
    parser.add_argument("--r3-page-cache", type=Path, default=None, help="Optional prebuilt R3 page source cache.")
    parser.add_argument("--r3-stable-model", type=Path, default=DEFAULT_R3_STABLE_MODEL)
    parser.add_argument("--r3-stable-manifest", type=Path, default=DEFAULT_R3_STABLE_MANIFEST)
    parser.add_argument("--r3-model-snapshot", type=Path, default=DEFAULT_R3_MODEL_SNAPSHOT)
    parser.add_argument("--r3-model-device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--r3-workers", type=int, default=8)
    parser.add_argument("--r3-pdfplumber-mode", choices=["never", "fallback", "always"], default="fallback")
    parser.add_argument("--r3-pdftotext-mode", choices=["never", "fallback", "always"], default="never")
    parser.add_argument(
        "--r5-profile",
        choices=["paper_only", "semantic", "metadata_calibrated", "source_augmented"],
        default="paper_only",
        help="R5 scoring/evidence profile; paper_only uses target-bound paper text and a frozen REFNO-grouped calibration.",
    )
    parser.add_argument(
        "--r7-profile",
        choices=["source_text", "astute_source_aware"],
        default="source_text",
    )
    parser.add_argument("--rockunit", type=Path, default=DEFAULT_ROCKUNIT)
    parser.add_argument("--pmagresult", type=Path, default=DEFAULT_PMAGRESULT)
    parser.add_argument("--output-dir", type=Path, default=None, help="Intermediate run directory.")
    parser.add_argument("--final-output-dir", type=Path, default=DEFAULT_OUTPUTS)
    parser.add_argument("--timestamp", type=validate_timestamp, default=None)
    return parser.parse_args()


def resolve_r3_extracts(path: Path | None) -> Path:
    if path:
        return path
    return DEFAULT_R3_EXTRACTS


def build_r3_page_cache(args: argparse.Namespace, run_dir: Path) -> Path:
    page_cache = args.r3_page_cache or run_dir / "r3_page_source_cache.jsonl"
    page_cache_summary = run_dir / "r3_page_source_cache_summary.json"
    if args.r3_page_cache is None:
        run(
            [
                sys.executable,
                "scripts/build_r3_page_source_cache.py",
                "--targets",
                str(args.targets),
                "--parsed-dir",
                str(args.parsed_dir),
                "--pdf-dir",
                str(args.pdf_dir),
                "--workers",
                str(args.r3_workers),
                "--pdfplumber-mode",
                args.r3_pdfplumber_mode,
                "--pdftotext-mode",
                args.r3_pdftotext_mode,
                "--output",
                str(page_cache),
                "--summary-json",
                str(page_cache_summary),
            ]
        )
    return page_cache


def build_fresh_r3_extracts(args: argparse.Namespace, run_dir: Path) -> tuple[Path, Path]:
    page_cache = build_r3_page_cache(args, run_dir)

    packs_dir = run_dir / "r3_packs_page_cache"
    run(
        [
            sys.executable,
            "scripts/build_r3_evidence_packs.py",
            "--targets",
            str(args.targets),
            "--parsed-dir",
            str(args.parsed_dir),
            "--pdf-dir",
            str(args.pdf_dir),
            "--page-cache",
            str(page_cache),
            "--pmagresult",
            str(args.pmagresult),
            "--rockunit",
            str(args.rockunit),
            "--output-dir",
            str(packs_dir),
            "--batch-size",
            "100",
        ]
    )

    structured_extracts = run_dir / "r3_structured_extract_from_page_cache.jsonl"
    run(
        [
            sys.executable,
            "scripts/build_r3_structured_extracts_from_packs.py",
            "--packs-jsonl",
            str(packs_dir / "r3_evidence_packs.jsonl"),
            "--output",
            str(structured_extracts),
        ]
    )
    return structured_extracts, page_cache


def main() -> int:
    args = parse_args()
    timestamp = args.timestamp or datetime.now().strftime("%m%d%H%M")
    run_dir = args.output_dir or DEFAULT_OUTPUTS / f"r357_latest_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    r3_policy = args.r3_policy or (
        "metadata_v1"
        if args.r3_source_mode == "fresh"
        else "paper_only_v1"
        if args.r3_source_mode == "paper_only"
        else "stable_production_v1"
        if args.r3_source_mode == "supervised_stable"
        else "v9"
    )

    required = {
        "targets": args.targets,
        "parsed_dir": args.parsed_dir,
        "pdf_dir": args.pdf_dir,
        "rockunit": args.rockunit,
        "pmagresult": args.pmagresult,
    }
    if args.r3_source_mode == "retained":
        required["r3_source_recovered_extracts"] = resolve_r3_extracts(args.r3_extracts)
    elif args.r3_source_mode == "supervised_stable":
        required.update(
            {
                "r3_stable_model": args.r3_stable_model,
                "r3_stable_manifest": args.r3_stable_manifest,
                "r3_model_snapshot": args.r3_model_snapshot,
            }
        )
    if args.r3_source_mode != "retained" and args.r3_page_cache is not None:
        required["r3_page_cache"] = args.r3_page_cache
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("Required workflow input(s) missing: " + "; ".join(missing))

    target_scope = validate_declared_target_scope(
        args.targets,
        args.rockunit,
        args.pmagresult,
        args.pdf_dir,
    )

    run_dir.mkdir(parents=True, exist_ok=True)
    args.final_output_dir.mkdir(parents=True, exist_ok=True)

    r3_jsonl = run_dir / "r3_latest_source_recovered_arbitration.jsonl"
    r5_predictions = run_dir / "r5_latest_source_recovered_predictions.xlsx"
    r7_predictions = run_dir / "r7_latest_source_recovered_predictions.xlsx"
    r1234567_html = args.final_output_dir / f"r1_r7_{timestamp}.html"
    r1234567_summary = run_dir / "r1_r7_total_review_summary.json"
    r1234567_xlsx = args.final_output_dir / f"r1_r7_{timestamp}.xlsx"
    r1234567_jsonl = args.final_output_dir / f"r1_r7_{timestamp}.jsonl.gz"
    r1234567_manifest = args.final_output_dir / f"r1_r7_{timestamp}.manifest.json"

    r3_page_cache_used: Path | None = None
    r3_identity_manifest: Path | None = None
    r3_extracts: Path | None = None
    r3_paper_only_jsonl: Path | None = None
    r3_stable_summary: Path | None = None
    if args.r3_source_mode in {"paper_only", "supervised_stable"}:
        r3_page_cache_used = build_r3_page_cache(args, run_dir)
        r3_identity_manifest = build_r3_identity_manifest(
            args.targets,
            run_dir / "r3_identity_manifest.xlsx",
        )
        r3_paper_only_jsonl = (
            r3_jsonl
            if args.r3_source_mode == "paper_only"
            else run_dir / "r3_paper_only_audit.jsonl"
        )
        run(
            [
                sys.executable,
                "scripts/build_r3_paper_only_predictions.py",
                "--identity-manifest",
                str(r3_identity_manifest),
                "--page-cache",
                str(r3_page_cache_used),
                "--output",
                str(r3_paper_only_jsonl),
                "--summary-json",
                str(run_dir / "r3_paper_only_summary.json"),
            ]
        )
        if args.r3_source_mode == "supervised_stable":
            r3_stable_summary = run_dir / "r3_supervised_stable_summary.json"
            run(
                [
                    sys.executable,
                    "scripts/build_r3_stable_production_predictions.py",
                    "--identity-manifest",
                    str(r3_identity_manifest),
                    "--page-cache",
                    str(r3_page_cache_used),
                    "--pdf-dir",
                    str(args.pdf_dir),
                    "--paper-only-jsonl",
                    str(r3_paper_only_jsonl),
                    "--model",
                    str(args.r3_stable_model),
                    "--model-manifest",
                    str(args.r3_stable_manifest),
                    "--model-snapshot",
                    str(args.r3_model_snapshot),
                    "--device",
                    args.r3_model_device,
                    "--output",
                    str(r3_jsonl),
                    "--summary-json",
                    str(r3_stable_summary),
                ]
            )
    elif args.r3_source_mode == "fresh":
        r3_extracts, r3_page_cache_used = build_fresh_r3_extracts(args, run_dir)
    else:
        r3_extracts = resolve_r3_extracts(args.r3_extracts)

    if args.r3_source_mode not in {"paper_only", "supervised_stable"}:
        assert r3_extracts is not None
        run([sys.executable, "scripts/validate_r3_structured_extract_jsonl.py", str(r3_extracts)])
        run(
            [
                sys.executable,
                "scripts/arbitrate_r3_structured_extracts.py",
                str(r3_extracts),
                "--policy",
                r3_policy,
                "--output",
                str(r3_jsonl),
            ]
        )
    run(
        [
            sys.executable,
            "scripts/export_r3_structured_final_outputs.py",
            "--jsonl",
            str(r3_jsonl),
            "--targets",
            str(args.targets),
            "--output-dir",
            str(args.final_output_dir),
            "--timestamp",
            timestamp,
            "--title",
            "R3 Latest Source-Recovered Full Review",
        ]
    )

    for indicator, predictions in (("R5", r5_predictions), ("R7", r7_predictions)):
        command = [
            sys.executable,
            "scripts/build_r5_r7_source_recovered_predictions.py",
            "--indicator",
            indicator,
            "--targets",
            str(args.targets),
            "--parsed-dir",
            str(args.parsed_dir),
            "--pdf-dir",
            str(args.pdf_dir),
            "--rockunit",
            str(args.rockunit),
            "--pmagresult",
            str(args.pmagresult),
            "--output",
            str(predictions),
        ]
        if indicator == "R5":
            command.extend(["--r5-profile", args.r5_profile])
        else:
            command.extend(["--r7-profile", args.r7_profile])
        if r3_page_cache_used is not None:
            command.extend(["--page-cache", str(r3_page_cache_used)])
        run(command)
        run(
            [
                sys.executable,
                "scripts/create_indicator_final_workbook.py",
                "--indicator",
                indicator,
                "--predictions",
                str(predictions),
                "--targets",
                str(args.targets),
                "--output-dir",
                str(args.final_output_dir),
                "--timestamp",
                timestamp,
            ]
        )
        final = newest_final(indicator, timestamp, args.final_output_dir)
        run(
            [
                sys.executable,
                "scripts/create_indicator_results_page.py",
                "--indicator",
                indicator,
                "--input",
                str(final),
                "--standard",
                str(args.targets),
                "--output",
                str(final.with_suffix(".html")),
            ]
        )

    r3_final = newest_final("R3", timestamp, args.final_output_dir)
    r5_final = newest_final("R5", timestamp, args.final_output_dir)
    r7_final = newest_final("R7", timestamp, args.final_output_dir)
    run(
        [
            sys.executable,
            "scripts/create_r1234567_total_review_html.py",
            "--targets",
            str(args.targets),
            "--rockunit",
            str(args.rockunit),
            "--pmagresult",
            str(args.pmagresult),
            "--r3",
            str(r3_final),
            "--r5",
            str(r5_final),
            "--r7",
            str(r7_final),
            "--output-html",
            str(r1234567_html),
            "--summary-json",
            str(r1234567_summary),
            "--timestamp",
            timestamp,
        ]
    )
    run(
        [
            sys.executable,
            "scripts/create_r1_r7_deliverables.py",
            "--targets",
            str(args.targets),
            "--rockunit",
            str(args.rockunit),
            "--pmagresult",
            str(args.pmagresult),
            "--r3",
            str(r3_final),
            "--r5",
            str(r5_final),
            "--r7",
            str(r7_final),
            "--html",
            str(r1234567_html),
            "--output-dir",
            str(args.final_output_dir),
            "--output-xlsx",
            str(r1234567_xlsx),
            "--output-jsonl",
            str(r1234567_jsonl),
            "--manifest-json",
            str(r1234567_manifest),
        ]
    )

    summary = {
        "workflow": "complete R1-R7 full workflow",
        "timestamp": timestamp,
        "run_dir": str(run_dir),
        "inputs": {name: str(path) for name, path in required.items()},
        "r3_source_mode": args.r3_source_mode,
        "r3_policy": r3_policy,
        "r3_extracts_used": str(r3_extracts) if r3_extracts else None,
        "r3_page_cache_used": str(r3_page_cache_used) if r3_page_cache_used else None,
        "r3_identity_manifest": str(r3_identity_manifest) if r3_identity_manifest else None,
        "r3_paper_only_audit": str(r3_paper_only_jsonl) if r3_paper_only_jsonl else None,
        "r3_supervised_stable_summary": str(r3_stable_summary) if r3_stable_summary else None,
        "r3_stable_model": (
            str(args.r3_stable_model)
            if args.r3_source_mode == "supervised_stable"
            else None
        ),
        "r3_stable_manifest": (
            str(args.r3_stable_manifest)
            if args.r3_source_mode == "supervised_stable"
            else None
        ),
        "target_scope": target_scope,
        "r5_profile": args.r5_profile,
        "r7_profile": args.r7_profile,
        "total_review_html": {
            "indicators": ["R1", "R2", "R4", "R6"],
            "combined_indicators": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
            "html": str(r1234567_html),
            "summary": str(r1234567_summary),
            "workbook": str(r1234567_xlsx),
            "jsonl": str(r1234567_jsonl),
            "manifest": str(r1234567_manifest),
        },
        "indicators": {},
        "boundary": (
            "No previous final workbook or benchmark label is used as a scoring input. "
            "Default R3 supervised_stable uses target identities, original-paper text, frozen "
            "knowledge/features, and a model trained on historical R3 labels; it does not read "
            "R3 labels at prediction time. Paper-only evidence is retained for audit and missing-text fallback. "
            "Default R5 paper_only uses target identities, original-paper page text, and frozen lexical "
            "parameters calibrated with REFNO-grouped splits; it does not read ROCKUNIT/PMAGRESULT at runtime. "
            "Legacy R3 fresh metadata_v1 and R5 metadata profiles remain explicit opt-in modes. "
            "Accuracy is computed only after final workbooks are produced. "
            "R1/R2/R4/R6 are direct table calculations from PMAGRESULT and ROCKUNIT, "
            "rendered with numeric evidence in the R1-R7 total HTML."
        ),
    }
    for indicator in ("R3", "R5", "R7"):
        final = newest_final(indicator, timestamp, args.final_output_dir)
        summary["indicators"][indicator] = metrics(indicator, final, args.targets)
    if args.r3_source_mode == "supervised_stable":
        model_manifest = json.loads(args.r3_stable_manifest.read_text(encoding="utf-8"))
        training = model_manifest.get("training") or {}
        r3_metrics = summary["indicators"]["R3"]
        r3_metrics["accuracy_interpretation"] = (
            "The post-scoring label comparison above is resubstitution on historical development "
            "labels when the current bundled targets are used; it is not reportable model accuracy."
        )
        r3_metrics["reportable_internal_grouped_oof"] = {
            "correct": int(training.get("grouped_oof_correct") or 0),
            "rows": int(training.get("grouped_oof_rows") or 0),
            "accuracy": float(training.get("grouped_oof_accuracy") or 0.0),
            "external_validation": False,
        }
    summary["deliverables"] = dict(summary["total_review_html"])

    summary_path = run_dir / "r357_latest_full_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
