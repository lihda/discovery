#!/usr/bin/env python3
"""Create a single HTML review page for R1-R7 results.

R1/R2/R4/R6 are calculated directly from PMAGRESULT/ROCKUNIT and rendered with
numeric evidence. R3/R5/R7 are read from existing result workbooks and rendered
from their RESULT_JSON evidence payloads. No XLSX file is written by this script.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from calculate_r1246_table_indicators import (
    DEFAULT_PMAGRESULT,
    DEFAULT_ROCKUNIT,
    DEFAULT_TARGETS,
    build_source_frame,
    calculate_indicators,
    load_targets,
    load_unique_table,
    numeric,
    test_tokens,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
INDICATORS = ("R1", "R2", "R3", "R4", "R5", "R6", "R7")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def as_number(value: Any) -> float | None:
    text = as_text(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def score_value(value: Any) -> int | str:
    text = as_text(value)
    if text in {"1", "1.0", "True", "true"}:
        return 1
    if text in {"0", "0.0", "False", "false"}:
        return 0
    return ""


def short(value: Any, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", as_text(value)).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def parse_json_cell(value: Any) -> dict[str, Any]:
    text = as_text(value)
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def split_status_reason(value: Any) -> tuple[str, str]:
    text = short(value, 900)
    if " - " in text:
        status, reasoning = text.split(" - ", 1)
        return status.strip() or "UNKNOWN", reasoning.strip()
    return text.strip() or "UNKNOWN", ""


def status_satisfied(status: Any) -> bool | None:
    upper = as_text(status).upper()
    if not upper:
        return None
    if "NOT_SATISFIED" in upper or upper in {"ABSENT", "INVALID", "NO", "NOT_MET", "FALSE"}:
        return False
    if "SATISFIED" in upper or upper in {"PRESENT", "VALID", "YES", "MET", "TRUE"}:
        return True
    return None


def evidence_quote_from_block(block: Any, limit: int = 550) -> str:
    if not isinstance(block, dict):
        return ""
    quotes = block.get("evidence_quotes")
    if not isinstance(quotes, list):
        return ""
    for item in quotes:
        if not isinstance(item, dict):
            continue
        quote = short(item.get("quote"), limit)
        if not quote or quote.upper() == "ABSENT":
            continue
        page = short(item.get("section_or_page"), 120)
        return f"{quote} ({page})" if page else quote
    return ""


def r3_readable_conditions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, label in [
        ("Evidence_A_Carrier_Investigation", "Carrier investigation"),
        ("Evidence_B_Carrier_Identity", "Carrier identity"),
        ("Evidence_C_Carrier_Significance", "Carrier significance"),
    ]:
        block = payload.get(key)
        if isinstance(block, dict):
            status = as_text(block.get("status")) or "UNKNOWN"
            reasoning = short(block.get("reasoning") or block.get("reason"), 650)
            satisfied = block.get("satisfied")
            if not isinstance(satisfied, bool):
                satisfied = status_satisfied(status)
            quote = evidence_quote_from_block(block)
        else:
            status, reasoning = split_status_reason(block)
            satisfied = status_satisfied(status)
            quote = ""
        if status == "UNKNOWN" and not reasoning and not quote:
            continue
        rows.append(
            {
                "label": label,
                "status": status,
                "satisfied": satisfied,
                "quote": quote,
                "reasoning": reasoning,
            }
        )

    metadata = payload.get("metadata_rockmag")
    if isinstance(metadata, dict):
        quote = short(metadata.get("quote"), 550)
        if quote.upper() == "ABSENT":
            quote = ""
        reason = short(metadata.get("reason"), 650)
        status = as_text(metadata.get("status")) or ("valid" if metadata.get("valid_r3_evidence") else "absent")
        satisfied = metadata.get("valid_r3_evidence")
        if not isinstance(satisfied, bool):
            satisfied = status_satisfied(status)
        if quote or reason:
            rows.append(
                {
                    "label": "PMAGRESULT rock-magnetic evidence",
                    "status": status,
                    "satisfied": satisfied,
                    "quote": quote,
                    "reasoning": reason,
                }
            )
    analysis = payload.get("knowledge_conflict_analysis")
    if isinstance(analysis, dict):
        relation = as_text(analysis.get("relation")).upper() or "UNKNOWN"
        decision = analysis.get("decision") if isinstance(analysis.get("decision"), dict) else {}
        conflict_types = ", ".join(as_text(value) for value in analysis.get("conflict_types") or [])
        rows.append(
            {
                "label": "Expert knowledge relation",
                "status": relation,
                "satisfied": True if relation == "SUPPORT" else False if relation == "CONFLICT" else None,
                "quote": "",
                "reasoning": short(
                    f"{decision.get('reason', '')} confidence={analysis.get('confidence')}; "
                    f"conflict_types={conflict_types or 'none'}",
                    650,
                ),
            }
        )
        for item in list(analysis.get("relations") or [])[:3]:
            if not isinstance(item, dict):
                continue
            status = as_text(item.get("relation")).upper() or "UNKNOWN"
            claim = item.get("claim") if isinstance(item.get("claim"), dict) else {}
            evidence = item.get("evidence") if isinstance(item.get("evidence"), dict) else {}
            quote = short(evidence.get("quote"), 550)
            page = as_text(evidence.get("page"))
            section = as_text(evidence.get("section"))
            location = " | ".join(value for value in (f"page {page}" if page else "", section) if value)
            if quote and location:
                quote = f"{quote} ({location})"
            rows.append(
                {
                    "label": as_text(claim.get("en") or item.get("rule_id") or "Knowledge rule"),
                    "status": status,
                    "satisfied": True if status == "SUPPORT" else False if status == "CONFLICT" else None,
                    "quote": quote,
                    "reasoning": short(
                        f"rule={item.get('rule_id')}; confidence={item.get('confidence')}; "
                        f"conflict_type={item.get('conflict_type')}",
                        650,
                    ),
                }
            )
    return rows


def generic_readable_conditions(payload: dict[str, Any], indicator: str) -> list[dict[str, Any]]:
    explicit = payload.get("condition_summary")
    if not isinstance(explicit, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in explicit:
        if not isinstance(item, dict):
            continue
        key = as_text(item.get("key"))
        if indicator == "R5" and key == "source_default":
            continue
        status = as_text(item.get("status")) or "UNKNOWN"
        satisfied = item.get("satisfied")
        if not isinstance(satisfied, bool):
            satisfied = status_satisfied(status)
        quote = evidence_quote_from_block(item)
        if indicator == "R7" and satisfied is False:
            quote = ""
        rows.append(
            {
                "label": as_text(item.get("label") or key or "Judgement condition"),
                "status": status,
                "satisfied": satisfied,
                "quote": quote,
                "reasoning": short(item.get("reasoning"), 650),
            }
        )
    return rows


def latest_workbook(indicator: str, output_dir: Path) -> Path:
    pattern = re.compile(rf"^{indicator.lower()}_\d{{8}}_(?:\d{{4,5}}|unlabeled)\.xlsx$")
    candidates = sorted(
        (path for path in output_dir.glob(f"{indicator.lower()}_*.xlsx") if pattern.fullmatch(path.name)),
        key=lambda path: (path.stat().st_mtime, path.name),
        reverse=True,
    )
    if not candidates:
        raise SystemExit(f"No final {indicator} workbook found in {output_dir}.")
    return candidates[0]


def read_indicator(path: Path, indicator: str) -> pd.DataFrame:
    frame = pd.read_excel(path, dtype={"REFNO": str, "RESULTNO": str}).fillna("")
    required = {"REFNO", "RESULTNO", "TARGET", indicator, f"{indicator}_RESULT_JSON"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(missing)}")
    return frame[["REFNO", "RESULTNO", "TARGET", indicator, f"{indicator}_RESULT_JSON"]].copy()


def r1246_frame(targets_path: Path, pmagresult_path: Path, rockunit_path: Path) -> pd.DataFrame:
    pmag = load_unique_table(
        pmagresult_path,
        "RESULTNO",
        {"RESULTNO", "ROCKUNITNO", "HIMAGAGE", "LOMAGAGE", "B", "N", "KD", "ED95", "DEMAGCODE", "TESTS"},
    )
    rockunit = load_unique_table(
        rockunit_path,
        "ROCKUNITNO",
        {"ROCKUNITNO", "REFNO", "ROCKNAME", "HIGHAGE", "LOWAGE"},
    )
    targets = load_targets(targets_path)
    frame = calculate_indicators(build_source_frame(targets, pmag, rockunit))
    keep = [
        "REFNO",
        "RESULTNO",
        "TARGET",
        "ROCKUNITNO",
        "R1",
        "R2",
        "R4",
        "R6",
        "HIMAGAGE",
        "LOMAGAGE",
        "HIGHAGE",
        "LOWAGE",
        "B",
        "N",
        "KD",
        "ED95",
        "DEMAGCODE",
        "TESTS",
    ]
    return frame[keep].copy()


def evidence_item(label: str, satisfied: bool, values: dict[str, Any], reasoning: str) -> dict[str, Any]:
    return {
        "label": label,
        "status": "SATISFIED" if satisfied else "NOT_SATISFIED",
        "satisfied": bool(satisfied),
        "values": {key: as_text(value) for key, value in values.items()},
        "reasoning": reasoning,
    }


def r1_evidence(row: pd.Series) -> dict[str, Any]:
    himagage = as_number(row.get("HIMAGAGE"))
    lomagage = as_number(row.get("LOMAGAGE"))
    highage = as_number(row.get("HIGHAGE"))
    lowage = as_number(row.get("LOWAGE"))
    mag_span = None if himagage is None or lomagage is None else himagage - lomagage
    rock_span = None if highage is None or lowage is None else highage - lowage
    delta = None if mag_span is None or rock_span is None else abs(mag_span - rock_span)
    satisfied = delta is not None and delta <= 15
    return evidence_item(
        "|(HIMAGAGE-LOMAGAGE) - (HIGHAGE-LOWAGE)| <= 15",
        satisfied,
        {
            "HIMAGAGE": row.get("HIMAGAGE"),
            "LOMAGAGE": row.get("LOMAGAGE"),
            "PMAG age span": "" if mag_span is None else f"{mag_span:g}",
            "HIGHAGE": row.get("HIGHAGE"),
            "LOWAGE": row.get("LOWAGE"),
            "ROCKUNIT age span": "" if rock_span is None else f"{rock_span:g}",
            "absolute difference": "" if delta is None else f"{delta:g}",
            "threshold": "15",
        },
        "R1=1 when the magnetic age span and rock-unit age span differ by no more than 15.",
    )


def threshold_evidence(label: str, value: Any, expression: str, satisfied: bool) -> dict[str, Any]:
    return {
        "label": label,
        "value": as_text(value),
        "expression": expression,
        "satisfied": bool(satisfied),
    }


def r2_evidence(row: pd.Series) -> dict[str, Any]:
    values = {
        "B": as_number(row.get("B")),
        "N": as_number(row.get("N")),
        "KD": as_number(row.get("KD")),
        "ED95": as_number(row.get("ED95")),
        "DEMAGCODE": as_number(row.get("DEMAGCODE")),
    }
    checks = [
        threshold_evidence("B", row.get("B"), "B >= 8", values["B"] is not None and values["B"] >= 8),
        threshold_evidence("N", row.get("N"), "N >= 25", values["N"] is not None and values["N"] >= 25),
        threshold_evidence("KD", row.get("KD"), "10 <= KD <= 70", values["KD"] is not None and 10 <= values["KD"] <= 70),
        threshold_evidence("ED95", row.get("ED95"), "ED95 <= 16", values["ED95"] is not None and values["ED95"] <= 16),
        threshold_evidence("DEMAGCODE", row.get("DEMAGCODE"), "DEMAGCODE >= 3", values["DEMAGCODE"] is not None and values["DEMAGCODE"] >= 3),
    ]
    satisfied = all(check["satisfied"] for check in checks)
    return {
        "label": "B>=8, N>=25, 10<=KD<=70, ED95<=16, DEMAGCODE>=3",
        "status": "SATISFIED" if satisfied else "NOT_SATISFIED",
        "satisfied": satisfied,
        "checks": checks,
        "reasoning": "R2=1 only when all numeric thresholds are satisfied.",
    }


def r4_evidence(row: pd.Series) -> dict[str, Any]:
    tokens = sorted(test_tokens(row.get("TESTS")))
    plus_tokens = [token for token in tokens if "+" in token and token != "R+"]
    satisfied = bool(plus_tokens)
    return evidence_item(
        'TESTS contains a "+" test token other than exact "R+"',
        satisfied,
        {
            "TESTS": row.get("TESTS"),
            "tokens": ", ".join(tokens),
            "matching non-R+ plus tokens": ", ".join(plus_tokens),
        },
        "R4=1 when TESTS contains an explicit plus-sign test result, excluding the standalone R+ token.",
    )


def r6_evidence(row: pd.Series) -> dict[str, Any]:
    tokens = sorted(test_tokens(row.get("TESTS")))
    matched = [token for token in tokens if token in {"R+", "Ra", "Rb", "Rc"}]
    satisfied = bool(matched)
    return evidence_item(
        'TESTS token is one of "R+", "Ra", "Rb", or "Rc"',
        satisfied,
        {
            "TESTS": row.get("TESTS"),
            "tokens": ", ".join(tokens),
            "matched reversal tokens": ", ".join(matched),
        },
        "R6=1 when TESTS contains one of the accepted reversal-test codes.",
    )


def condition_summary(payload: dict[str, Any], indicator: str) -> list[dict[str, Any]]:
    if not payload:
        return []
    if indicator == "R3":
        return r3_readable_conditions(payload)
    return generic_readable_conditions(payload, indicator)


def json_evidence(payload: dict[str, Any], indicator: str) -> dict[str, Any]:
    summary = short(payload.get("final_decision_reasoning") or payload.get("summary"), 900) if payload else ""
    outcome = as_text(payload.get("outcome") or "DECIDED").upper() if payload else "UNKNOWN"
    return {
        "label": f"{indicator} judgement evidence",
        "status": f"{indicator}={score_value(payload.get(indicator))}" if payload else "missing evidence",
        "satisfied": score_value(payload.get(indicator)) == 1 if payload else False,
        "outcome": outcome,
        "source_status": "UNKNOWN · source unavailable" if outcome == "UNKNOWN" else "",
        "summary": summary,
        "conditions": condition_summary(payload, indicator),
    }


def combine(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    r1246 = r1246_frame(args.targets, args.pmagresult, args.rockunit)
    r3 = read_indicator(args.r3 or latest_workbook("R3", args.output_dir), "R3")
    r5 = read_indicator(args.r5 or latest_workbook("R5", args.output_dir), "R5")
    r7 = read_indicator(args.r7 or latest_workbook("R7", args.output_dir), "R7")

    combined = r1246.merge(
        r3.drop(columns=["REFNO", "TARGET"]),
        on="RESULTNO",
        how="left",
        validate="one_to_one",
    ).merge(
        r5.drop(columns=["REFNO", "TARGET"]),
        on="RESULTNO",
        how="left",
        validate="one_to_one",
    ).merge(
        r7.drop(columns=["REFNO", "TARGET"]),
        on="RESULTNO",
        how="left",
        validate="one_to_one",
    )

    for indicator in INDICATORS:
        if indicator in combined.columns:
            combined[indicator] = combined[indicator].map(score_value)

    combined["R_SUM"] = combined[[indicator for indicator in INDICATORS if indicator in combined.columns]].apply(
        lambda row: sum(value for value in row if isinstance(value, int)),
        axis=1,
    )
    combined["R3_PAYLOAD"] = combined["R3_RESULT_JSON"].map(parse_json_cell)
    combined["R5_PAYLOAD"] = combined["R5_RESULT_JSON"].map(parse_json_cell)
    combined["R7_PAYLOAD"] = combined["R7_RESULT_JSON"].map(parse_json_cell)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": int(len(combined)),
        "unique_refnos": int(combined["REFNO"].nunique()),
        "unique_resultnos": int(combined["RESULTNO"].nunique()),
        "inputs": {
            "targets": str(args.targets),
            "pmagresult": str(args.pmagresult),
            "rockunit": str(args.rockunit),
            "r3": str(args.r3 or latest_workbook("R3", args.output_dir)),
            "r5": str(args.r5 or latest_workbook("R5", args.output_dir)),
            "r7": str(args.r7 or latest_workbook("R7", args.output_dir)),
        },
        "score_counts": {
            indicator: {
                str(key): int(value)
                for key, value in combined[indicator].value_counts(dropna=False).sort_index().items()
            }
            for indicator in INDICATORS
            if indicator in combined.columns
        },
    }
    return combined, summary


def row_payload(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, row in frame.sort_values(
        by=["REFNO", "RESULTNO"],
        key=lambda series: series.map(lambda value: (0, int(value)) if as_text(value).isdigit() else (1, as_text(value))),
    ).iterrows():
        evidence = {
            "R1": r1_evidence(row),
            "R2": r2_evidence(row),
            "R3": json_evidence(row.get("R3_PAYLOAD") or {}, "R3"),
            "R4": r4_evidence(row),
            "R5": json_evidence(row.get("R5_PAYLOAD") or {}, "R5"),
            "R6": r6_evidence(row),
            "R7": json_evidence(row.get("R7_PAYLOAD") or {}, "R7"),
        }
        rows.append(
            {
                "REFNO": as_text(row.get("REFNO")),
                "RESULTNO": as_text(row.get("RESULTNO")),
                "TARGET": as_text(row.get("TARGET")),
                "ROCKUNITNO": as_text(row.get("ROCKUNITNO")),
                "scores": {indicator: score_value(row.get(indicator)) for indicator in INDICATORS},
                "R_SUM": int(row.get("R_SUM") or 0),
                "evidence": evidence,
                "source_values": {
                    key: as_text(row.get(key))
                    for key in ["HIMAGAGE", "LOMAGAGE", "HIGHAGE", "LOWAGE", "B", "N", "KD", "ED95", "DEMAGCODE", "TESTS"]
                },
            }
        )
    return rows


def safe_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")


def render_html(rows: list[dict[str, Any]], summary: dict[str, Any], output: Path, title: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = safe_json({"title": title, "summary": summary, "rows": rows})
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #5c6675;
      --line: #d8dee8;
      --panel: #ffffff;
      --band: #f3f6fa;
      --navy: #244a70;
      --green: #20724f;
      --red: #9e2f2f;
      --amber: #8a5a00;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; color: var(--ink); background: #f7f8fa; font: 13px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif; }}
    header {{ position: sticky; top: 0; z-index: 3; padding: 16px 22px 12px; background: var(--panel); border-bottom: 1px solid var(--line); }}
    h1 {{ margin: 0 0 10px; font-size: 20px; font-weight: 700; letter-spacing: 0; }}
    .stats {{ display: flex; flex-wrap: wrap; gap: 8px; }}
    .stat {{ padding: 6px 9px; background: var(--band); border: 1px solid var(--line); border-radius: 6px; color: var(--muted); }}
    .stat strong {{ color: var(--ink); }}
    main {{ display: grid; grid-template-columns: minmax(0, 1.4fr) minmax(380px, 0.8fr); gap: 14px; padding: 14px 18px 22px; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 10px; }}
    input, select {{ height: 34px; padding: 0 10px; border: 1px solid #b8c3d1; border-radius: 6px; background: #fff; color: var(--ink); }}
    input {{ width: min(480px, 55vw); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 7px 8px; text-align: left; vertical-align: middle; }}
    th {{ position: sticky; top: 84px; z-index: 2; background: var(--navy); color: #fff; font-weight: 700; }}
    tr {{ cursor: pointer; }}
    tbody tr:hover, tbody tr.active {{ background: #eef4fa; }}
    .target {{ max-width: 260px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .score {{ display: inline-flex; align-items: center; justify-content: center; min-width: 24px; height: 22px; border-radius: 5px; font-weight: 700; border: 1px solid var(--line); }}
    .score.yes {{ color: var(--green); background: #e8f5ee; border-color: #badfcb; }}
    .score.no {{ color: var(--red); background: #faecec; border-color: #e5bcbc; }}
    .score.unknown {{ color: var(--muted); background: #f1f1ee; border-color: #d3d4d0; }}
    .score.blank {{ color: var(--muted); background: #f2f3f5; }}
    .sum {{ font-weight: 700; }}
    aside {{ position: sticky; top: 84px; align-self: start; max-height: calc(100vh - 104px); overflow: auto; background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }}
    .detail-head {{ padding: 14px 16px; border-bottom: 1px solid var(--line); background: #fbfcfd; }}
    .detail-title {{ font-size: 16px; font-weight: 700; margin-bottom: 5px; }}
    .detail-sub {{ color: var(--muted); }}
    .evidence {{ padding: 12px 16px 16px; }}
    details {{ border: 1px solid var(--line); border-radius: 7px; margin-bottom: 9px; background: #fff; }}
    summary {{ cursor: pointer; padding: 9px 10px; font-weight: 700; display: flex; align-items: center; gap: 8px; }}
    summary::-webkit-details-marker {{ display: none; }}
    .ev-body {{ padding: 0 10px 11px; color: var(--ink); }}
    .kv {{ display: grid; grid-template-columns: 150px minmax(0, 1fr); gap: 4px 10px; margin: 8px 0; }}
    .kv div:nth-child(odd) {{ color: var(--muted); }}
    .condition {{ padding: 7px 0; border-top: 1px solid #edf0f4; }}
    .status {{ font-size: 12px; padding: 2px 6px; border-radius: 999px; }}
    .status.ok {{ color: var(--green); background: #e8f5ee; }}
    .status.fail {{ color: var(--red); background: #faecec; }}
    .status.warn {{ color: var(--amber); background: #fff5d9; }}
    .small {{ color: var(--muted); font-size: 12px; }}
    @media (max-width: 980px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ position: static; max-height: none; }}
      th {{ top: 112px; }}
      input {{ width: 100%; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{title}</h1>
    <div id="stats" class="stats"></div>
  </header>
  <main>
    <section>
      <div class="toolbar">
        <input id="search" type="search" placeholder="搜索 REFNO / RESULTNO / target / evidence">
        <select id="indicatorFilter">
          <option value="">全部指标</option>
          <option value="R1">R1=1</option><option value="R2">R2=1</option><option value="R3">R3=1</option>
          <option value="R4">R4=1</option><option value="R5">R5=1</option><option value="R6">R6=1</option><option value="R7">R7=1</option>
        </select>
        <span id="count" class="small"></span>
      </div>
      <table id="resultTable"></table>
    </section>
    <aside id="detail"></aside>
  </main>
  <script>const DATA = {payload};</script>
  <script>
    const indicators = ['R1','R2','R3','R4','R5','R6','R7'];
    const table = document.getElementById('resultTable');
    const detail = document.getElementById('detail');
    const search = document.getElementById('search');
    const indicatorFilter = document.getElementById('indicatorFilter');
    const count = document.getElementById('count');
    let selected = DATA.rows[0]?.RESULTNO || '';

    function esc(value) {{
      return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    }}
    function scoreClass(value, outcome = '') {{
      if (outcome === 'UNKNOWN') return 'score unknown';
      if (value === 1) return 'score yes';
      if (value === 0) return 'score no';
      return 'score blank';
    }}
    function statusClass(ev) {{
      const statusText = String(ev?.status || '');
      if (ev?.satisfied === false || statusText.includes('NOT') || statusText.includes('ABSENT')) return 'status fail';
      if (ev?.satisfied === true || statusText.includes('SATISFIED') || statusText.includes('PRESENT')) return 'status ok';
      return 'status warn';
    }}
    function rowMatches(row) {{
      const q = search.value.trim().toLowerCase();
      const filter = indicatorFilter.value;
      if (filter && row.scores[filter] !== 1) return false;
      if (!q) return true;
      return JSON.stringify(row).toLowerCase().includes(q);
    }}
    function renderStats() {{
      const stats = DATA.summary;
      const parts = [
        ['Rows', stats.rows],
        ['REFNO', stats.unique_refnos],
        ['RESULTNO', stats.unique_resultnos],
      ];
      for (const ind of indicators) {{
        const counts = stats.score_counts[ind] || {{}};
        parts.push([`${{ind}}=1`, counts['1'] ?? 0]);
      }}
      document.getElementById('stats').innerHTML = parts.map(([k,v]) => `<span class="stat"><strong>${{esc(k)}}:</strong> ${{esc(v)}}</span>`).join('');
    }}
    function renderTable() {{
      const rows = DATA.rows.filter(rowMatches);
      count.textContent = `${{rows.length}} / ${{DATA.rows.length}} visible`;
      const head = `<thead><tr><th>REFNO</th><th>RESULTNO</th><th>Target</th>${{indicators.map(i => `<th>${{i}}</th>`).join('')}}<th>Sum</th></tr></thead>`;
      const body = rows.map(row => `<tr data-resultno="${{esc(row.RESULTNO)}}" class="${{row.RESULTNO === selected ? 'active' : ''}}">
        <td>${{esc(row.REFNO)}}</td><td>${{esc(row.RESULTNO)}}</td><td class="target" title="${{esc(row.TARGET)}}">${{esc(row.TARGET)}}</td>
        ${{indicators.map(ind => `<td><span class="${{scoreClass(row.scores[ind], row.evidence[ind]?.outcome)}}">${{esc(row.scores[ind])}}</span></td>`).join('')}}
        <td class="sum">${{esc(row.R_SUM)}}</td>
      </tr>`).join('');
      table.innerHTML = head + `<tbody>${{body}}</tbody>`;
      table.querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => {{
        selected = tr.dataset.resultno;
        renderTable();
        renderDetail();
      }}));
      if (!rows.some(row => row.RESULTNO === selected) && rows[0]) selected = rows[0].RESULTNO;
    }}
    function renderValues(values) {{
      if (!values) return '';
      return `<div class="kv">${{Object.entries(values).map(([k,v]) => `<div>${{esc(k)}}</div><div>${{esc(v)}}</div>`).join('')}}</div>`;
    }}
    function renderChecks(checks) {{
      if (!Array.isArray(checks)) return '';
      return checks.map(check => `<div class="condition"><span class="${{check.satisfied ? 'status ok' : 'status fail'}}">${{check.satisfied ? 'PASS' : 'FAIL'}}</span> <strong>${{esc(check.expression)}}</strong> <span class="small">value=${{esc(check.value)}}</span></div>`).join('');
    }}
    function renderConditions(conditions) {{
      if (!Array.isArray(conditions) || !conditions.length) return '';
      return conditions.map(condition => `<div class="condition">
        <div><strong>${{esc(condition.label)}}</strong> <span class="${{statusClass(condition)}}">${{esc(condition.status ?? '')}}</span></div>
        ${{condition.quote ? `<div class="small">${{esc(condition.quote)}}</div>` : ''}}
        ${{condition.reasoning ? `<div>${{esc(condition.reasoning)}}</div>` : ''}}
      </div>`).join('');
    }}
    function renderEvidence(ind, ev, score) {{
      return `<details ${{ind === 'R1' ? 'open' : ''}}>
        <summary><span class="${{scoreClass(score, ev?.outcome)}}">${{esc(score)}}</span> ${{ind}} <span class="${{statusClass(ev)}}">${{esc(ev.status || '')}}</span>${{ev.source_status ? ` <span class="status warn">${{esc(ev.source_status)}}</span>` : ''}}</summary>
        <div class="ev-body">
          <div><strong>${{esc(ev.label || '')}}</strong></div>
          ${{ev.summary ? `<p>${{esc(ev.summary)}}</p>` : ''}}
          ${{ev.reasoning ? `<p>${{esc(ev.reasoning)}}</p>` : ''}}
          ${{renderValues(ev.values)}}
          ${{renderChecks(ev.checks)}}
          ${{renderConditions(ev.conditions)}}
        </div>
      </details>`;
    }}
    function renderDetail() {{
      const row = DATA.rows.find(item => item.RESULTNO === selected) || DATA.rows[0];
      if (!row) {{
        detail.innerHTML = '<div class="detail-head"><div class="detail-title">No rows</div></div>';
        return;
      }}
      detail.innerHTML = `<div class="detail-head">
        <div class="detail-title">${{esc(row.TARGET)}}</div>
        <div class="detail-sub">REFNO ${{esc(row.REFNO)}} · RESULTNO ${{esc(row.RESULTNO)}} · ROCKUNITNO ${{esc(row.ROCKUNITNO)}} · R_SUM ${{esc(row.R_SUM)}}</div>
      </div>
      <div class="evidence">
        ${{indicators.map(ind => renderEvidence(ind, row.evidence[ind], row.scores[ind])).join('')}}
      </div>`;
    }}
    search.addEventListener('input', () => {{ renderTable(); renderDetail(); }});
    indicatorFilter.addEventListener('change', () => {{ renderTable(); renderDetail(); }});
    renderStats();
    renderTable();
    renderDetail();
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def validate_timestamp(value: str) -> str:
    if not re.fullmatch(r"\d{8}", value):
        raise argparse.ArgumentTypeError("timestamp must use MMDDHHMM, for example 07010632")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--pmagresult", type=Path, default=DEFAULT_PMAGRESULT)
    parser.add_argument("--rockunit", type=Path, default=DEFAULT_ROCKUNIT)
    parser.add_argument("--r3", type=Path, default=None)
    parser.add_argument("--r5", type=Path, default=None)
    parser.add_argument("--r7", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-html", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--timestamp", type=validate_timestamp, default=None)
    parser.add_argument("--title", default="R1-R7 Total Reliability Review")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame, summary = combine(args)
    rows = row_payload(frame)
    stamp = args.timestamp or datetime.now().strftime("%m%d%H%M")
    output = args.output_html or args.output_dir / f"r1234567_{stamp}.html"
    render_html(rows, summary, output, args.title)
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"HTML: {output}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
