#!/usr/bin/env python3
"""Inject final R3 result data into the reusable HTML review template."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs"
DEFAULT_TEMPLATE = ROOT / "assets/r3_results_template.html"
DEFAULT_STANDARD = ROOT / "data/targets.xlsx"
REQUIRED_COLUMNS = ["REFNO", "RESULTNO", "TARGET", "R3", "R3_RESULT_JSON"]
PAYLOAD_PLACEHOLDER = "__R3_RESULTS_PAYLOAD__"
WORKBOOK_NAME_PATTERN = re.compile(r"^r3_\d{8}_(?:\d{4,5}|unlabeled)\.xlsx$")


def as_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def short_text(value: Any, limit: int = 650) -> str:
    text = re.sub(r"\s+", " ", as_text(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def load_results(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        frame = pd.read_excel(path, dtype={"REFNO": str, "RESULTNO": str})
    elif suffix == ".csv":
        frame = pd.read_csv(path, dtype={"REFNO": str, "RESULTNO": str})
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("results", "rows", "data"):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        frame = pd.DataFrame(payload)
    else:
        raise SystemExit(f"Unsupported input format: {path.suffix}")

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {', '.join(missing)}")
    return frame[REQUIRED_COLUMNS].copy()


def parse_json_cell(value: Any) -> tuple[dict[str, Any], str | None]:
    text = as_text(value).strip()
    if not text:
        return {}, "empty JSON cell"
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return {}, f"invalid JSON: {exc.msg}"
    if not isinstance(parsed, dict):
        return {}, "JSON cell is not an object"
    return parsed, None


def normalize_score(value: Any) -> str:
    text = as_text(value).strip()
    if text in {"1", "1.0", "True", "true"}:
        return "1"
    if text in {"0", "0.0", "False", "false"}:
        return "0"
    return text or "UNKNOWN"


def load_standard_scores(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    columns = set(pd.read_excel(path, nrows=0).columns)
    if not {"RESULTNO", "R3"}.issubset(columns):
        return {}
    frame = pd.read_excel(path, usecols=["RESULTNO", "R3"], dtype={"RESULTNO": str})
    lookup: dict[str, str] = {}
    for _, row in frame.iterrows():
        score = normalize_score(row.get("R3"))
        if score in {"0", "1"}:
            lookup[as_text(row.get("RESULTNO"))] = score
    return lookup


def mismatch_metadata(resultno: Any, score: str, standards: dict[str, str]) -> tuple[bool, str, str]:
    standard_score = standards.get(as_text(resultno))
    if standard_score is None:
        return False, "UNKNOWN", "no_standard"
    if score == standard_score:
        return False, standard_score, "correct"
    if score == "1" and standard_score == "0":
        return True, standard_score, "false_positive"
    if score == "0" and standard_score == "1":
        return True, standard_score, "false_negative"
    return True, standard_score, "mismatch"


def get_path(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def evidence_status(parsed: dict[str, Any], key: str) -> tuple[str, bool | None]:
    block = parsed.get(key)
    if isinstance(block, str):
        text = block.strip()
        if text.startswith("SATISFIED"):
            return "SATISFIED", True
        if text.startswith("NOT_SATISFIED"):
            return "NOT_SATISFIED", False
        return text or "UNKNOWN", None
    if not isinstance(block, dict):
        return "UNKNOWN", None
    status = as_text(block.get("status")).strip() or "UNKNOWN"
    satisfied = block.get("satisfied")
    if isinstance(satisfied, bool):
        return status, satisfied
    if status.upper() == "SATISFIED":
        return status, True
    if status.upper() == "NOT_SATISFIED":
        return status, False
    return status, None


def split_status_reason(value: Any) -> tuple[str, str]:
    text = short_text(value, 900)
    if " - " in text:
        status, reasoning = text.split(" - ", 1)
        return status.strip() or "UNKNOWN", reasoning.strip()
    return text.strip() or "UNKNOWN", ""


def evidence_quote(parsed: dict[str, Any], key: str) -> str:
    quotes = get_path(parsed, key, "evidence_quotes")
    if not isinstance(quotes, list) or not quotes:
        return ""
    first = quotes[0]
    if not isinstance(first, dict):
        return ""
    quote = as_text(first.get("quote")).strip()
    page = as_text(first.get("section_or_page")).strip()
    if not quote or quote.upper() == "ABSENT":
        return ""
    quote = short_text(quote, 550)
    return f"{quote} ({page})" if page else quote


def knowledge_relation_evidence(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    analysis = parsed.get("knowledge_conflict_analysis")
    if not isinstance(analysis, dict):
        return {}
    relation = as_text(analysis.get("relation")).upper() or "UNKNOWN"
    decision = analysis.get("decision") if isinstance(analysis.get("decision"), dict) else {}
    confidence = analysis.get("confidence")
    conflict_types = ", ".join(as_text(value) for value in analysis.get("conflict_types") or [])
    output: dict[str, dict[str, Any]] = {
        "knowledge": {
            "label": "Expert knowledge relation",
            "status": relation,
            "satisfied": True if relation == "SUPPORT" else False if relation == "CONFLICT" else None,
            "quote": "",
            "reasoning": short_text(
                f"{decision.get('reason', '')} confidence={confidence}; "
                f"conflict_types={conflict_types or 'none'}",
                650,
            ),
        }
    }
    for index, item in enumerate(list(analysis.get("relations") or [])[:3], start=1):
        if not isinstance(item, dict):
            continue
        status = as_text(item.get("relation")).upper() or "UNKNOWN"
        claim = item.get("claim") if isinstance(item.get("claim"), dict) else {}
        source = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
        quote = short_text(source.get("quote"), 550)
        page = as_text(source.get("page"))
        section = as_text(source.get("section"))
        location = " | ".join(value for value in (f"page {page}" if page else "", section) if value)
        if quote and location:
            quote = f"{quote} ({location})"
        output[f"knowledge_{index}"] = {
            "label": as_text(claim.get("en") or item.get("rule_id") or "Knowledge rule"),
            "status": status,
            "satisfied": True if status == "SUPPORT" else False if status == "CONFLICT" else None,
            "quote": quote,
            "reasoning": short_text(
                f"rule={item.get('rule_id')}; confidence={item.get('confidence')}; "
                f"conflict_type={item.get('conflict_type')}",
                650,
            ),
        }
    return output


def r3_evidence(parsed: dict[str, Any]) -> dict[str, dict[str, Any]]:
    evidence: dict[str, dict[str, Any]] = {}
    for key, json_key, label in [
        ("A", "Evidence_A_Carrier_Investigation", "Carrier investigation"),
        ("B", "Evidence_B_Carrier_Identity", "Carrier identity"),
        ("C", "Evidence_C_Carrier_Significance", "Carrier significance"),
    ]:
        block = parsed.get(json_key)
        status, satisfied = evidence_status(parsed, json_key)
        if isinstance(block, dict):
            reasoning = short_text(block.get("reasoning") or block.get("reason"), 650)
        else:
            _, reasoning = split_status_reason(block)
        evidence[key] = {
            "label": label,
            "status": status,
            "satisfied": satisfied,
            "quote": evidence_quote(parsed, json_key),
            "reasoning": reasoning,
        }
    metadata = parsed.get("metadata_rockmag")
    if isinstance(metadata, dict):
        quote = short_text(metadata.get("quote"), 550)
        if quote.upper() == "ABSENT":
            quote = ""
        reason = short_text(metadata.get("reason"), 650)
        status = as_text(metadata.get("status")).strip() or ("valid" if metadata.get("valid_r3_evidence") else "absent")
        satisfied = metadata.get("valid_r3_evidence")
        evidence["metadata"] = {
            "label": "PMAGRESULT rock-magnetic evidence",
            "status": status,
            "satisfied": satisfied if isinstance(satisfied, bool) else None,
            "quote": quote,
            "reasoning": reason,
        }
    evidence.update(knowledge_relation_evidence(parsed))
    return evidence


def summarize_rows(frame: pd.DataFrame, standards: dict[str, str] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    standards = standards or {}
    stats = {
        "total": 0,
        "r3_yes": 0,
        "r3_no": 0,
        "manual_review": 0,
        "json_errors": 0,
        "mismatches": 0,
        "false_positive": 0,
        "false_negative": 0,
        "no_standard": 0,
        "a_yes": 0,
        "b_yes": 0,
        "c_yes": 0,
    }

    for _, source in frame.iterrows():
        parsed, error = parse_json_cell(source["R3_RESULT_JSON"])
        score = normalize_score(source["R3"])
        a_status, a_sat = evidence_status(parsed, "Evidence_A_Carrier_Investigation")
        b_status, b_sat = evidence_status(parsed, "Evidence_B_Carrier_Identity")
        c_status, c_sat = evidence_status(parsed, "Evidence_C_Carrier_Significance")
        manual = bool(parsed.get("manual_review")) if parsed else False
        summary = as_text(parsed.get("summary") or parsed.get("final_decision_reasoning")).strip()
        mismatch, standard_score, error_type = mismatch_metadata(source["RESULTNO"], score, standards)

        stats["total"] += 1
        if score == "1":
            stats["r3_yes"] += 1
        elif score == "0":
            stats["r3_no"] += 1
        if manual:
            stats["manual_review"] += 1
        if error:
            stats["json_errors"] += 1
        if mismatch:
            stats["mismatches"] += 1
        if error_type in {"false_positive", "false_negative", "no_standard"}:
            stats[error_type] += 1
        if a_sat is True:
            stats["a_yes"] += 1
        if b_sat is True:
            stats["b_yes"] += 1
        if c_sat is True:
            stats["c_yes"] += 1

        rows.append(
            {
                "refno": as_text(source["REFNO"]),
                "resultno": as_text(source["RESULTNO"]),
                "target": as_text(source["TARGET"]),
                "score": score,
                "summary": summary or (error or "No summary available."),
                "mismatch": mismatch,
                "standard_score": standard_score,
                "error_type": error_type,
                "evidence": r3_evidence(parsed),
            }
        )

    return rows, stats


def safe_json_for_script(payload: dict[str, Any]) -> str:
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return text.replace("</", "<\\/")


def build_payload(frame: pd.DataFrame, input_path: Path, title: str, standards: dict[str, str] | None = None) -> dict[str, Any]:
    rows, stats = summarize_rows(frame, standards)
    public_stats = {key: value for key, value in stats.items() if key not in {"manual_review", "json_errors"}}
    return {
        "title": title,
        "indicator": "R3",
        "score_label": "R3",
        "source": str(input_path),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "standard_mode": "labelled_rows" if standards else "unlabelled_extension_rows",
        "stats": public_stats,
        "rows": rows,
    }


def render_template(template_path: Path, payload: dict[str, Any]) -> str:
    template = template_path.read_text(encoding="utf-8")
    if PAYLOAD_PLACEHOLDER not in template:
        raise SystemExit(f"{template_path} is missing {PAYLOAD_PLACEHOLDER}")
    return template.replace(PAYLOAD_PLACEHOLDER, safe_json_for_script(payload))


def latest_named_workbook(input_dir: Path) -> Path:
    candidates = sorted(
        (path for path in input_dir.glob("r3_*.xlsx") if WORKBOOK_NAME_PATTERN.fullmatch(path.name)),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No final R3 workbook matching r3_<MMDDHHMM>_<accuracy-or-unlabeled>.xlsx found in {input_dir}")
    return candidates[0]


def resolve_output_path(input_path: Path, output: Path | None) -> Path:
    if input_path.suffix.lower() in {".xlsx", ".xlsm", ".xls"} and not WORKBOOK_NAME_PATTERN.fullmatch(input_path.name):
        raise SystemExit(
            "Final R3 workbook name must match r3_<MMDDHHMM>_<accuracy-or-unlabeled>.xlsx "
            "before creating the matching HTML page."
        )
    resolved = output or input_path.with_suffix(".html")
    if resolved.stem != input_path.stem:
        raise SystemExit(
            "HTML output basename must match the workbook basename, "
            f"expected {input_path.with_suffix('.html').name}"
        )
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Final R3 workbook/CSV/JSON. Defaults to the newest named workbook in outputs/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Standalone HTML output path. Defaults to the input workbook basename with .html.",
    )
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Reusable HTML template.")
    parser.add_argument("--standard", type=Path, default=DEFAULT_STANDARD, help="Standard workbook for mismatch markers.")
    parser.add_argument("--title", default="R3 Results Visual Review", help="HTML page title.")
    args = parser.parse_args()

    input_path = args.input or latest_named_workbook(DEFAULT_INPUT_DIR)
    frame = load_results(input_path)
    payload = build_payload(frame, input_path, args.title, load_standard_scores(args.standard))
    output = resolve_output_path(input_path, args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_template(args.template, payload), encoding="utf-8")

    stats = payload["stats"]
    print(f"Saved: {output}")
    print(f"Template: {args.template}")
    print(f"Rows: {stats['total']}")
    print(f"R3=1: {stats['r3_yes']} | R3=0: {stats['r3_no']}")
    print(f"Mismatches: {stats['mismatches']}")


if __name__ == "__main__":
    main()
