from __future__ import annotations

import json
from pathlib import Path
import re

import pandas as pd


SOURCE_ROOT = Path(
    "/Users/lihd/discovery/paleomagnetic-r3-workflow/dist/"
    "paleomagnetic-r357-calibrated-workflow-full/data"
)
OUTPUT = Path(__file__).with_name("selection.json")
SELECTED_RESULTNOS = (
    "10052",
    "10058",
    "10063",
    "10082",
    "10086",
    "10091",
    "10095",
    "10096",
    "10125",
    "10218",
)

PATTERNS = {
    "R3": (
        r"\b(?:magnetite|hematite|haematite|titanomagnetite|maghemite|pyrrhotite|goethite|greigite)\b",
        r"\bmagnetic (?:carrier|mineral(?:ogy|ogical)?)\b",
        r"\b(?:hysteresis|isothermal remanent magnetization|IRM|thermomagnetic|Curie temperature|Lowrie test|coercivit\w*)\b",
        r"\b(?:rock magnetic|mineralog\w*|X-ray|Mössbauer|scanning electron)\b",
        r"\b(?:characteristic remanent magnetization|ChRM|natural remanent magnetization|NRM)\b",
    ),
    "R5": (
        r"\b(?:fold test|tilt correction|bedding correction|structural correction)\b",
        r"\b(?:paleo|palaeo)horizontal\w*\b",
        r"\b(?:pre[- ]?fold|syn[- ]?fold|post[- ]?fold|untilted|tilt[- ]?corrected)\b",
        r"\b(?:bedding|folding|folded|tilting|tilted|tectonic rotation|structural control)\b",
    ),
    "R7": (
        r"\b(?:remagnetiz\w*|remagnetis\w*|overprint\w*)\b",
        r"\b(?:younger pole|younger magnetiz\w*|younger magnetis\w*|secondary component|secondary magnetiz\w*|secondary magnetis\w*)\b",
        r"\b(?:age reset|resetting|post[- ]?deposition\w*|post[- ]?fold\w*|chemical remanent magnetization|viscous remanence)\b",
        r"\b(?:younger|paleopole|palaeopole)\b",
    ),
}


def text_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def typed_number(value: object) -> int | float | None:
    text = text_value(value)
    if not text:
        return None
    number = float(text)
    return int(number) if number.is_integer() else number


def clean_excerpt(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def source_excerpt(text: str, patterns: tuple[str, ...], limit: int = 9000) -> str:
    intervals: list[tuple[int, int]] = [(0, min(len(text), 1500))]
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.I):
            start = max(0, match.start() - 650)
            end = min(len(text), match.end() + 650)
            if any(start <= old_end + 120 and end >= old_start - 120 for old_start, old_end in intervals):
                continue
            intervals.append((start, end))
            if len(intervals) >= 9:
                break
        if len(intervals) >= 9:
            break
    chunks: list[str] = []
    size = 0
    for index, (start, end) in enumerate(sorted(intervals), start=1):
        chunk = clean_excerpt(text[start:end])
        if not chunk:
            continue
        marked = f"[Original-paper excerpt {index}]\n{chunk}"
        if chunks and size + len(marked) + 2 > limit:
            break
        chunks.append(marked)
        size += len(marked) + 2
    return "\n\n".join(chunks)[:limit]


def main() -> None:
    targets = pd.read_excel(SOURCE_ROOT / "targets.xlsx", dtype=str).fillna("")
    pmag = pd.read_excel(SOURCE_ROOT / "source_metadata/PMAGRESULT.xlsx", dtype=str).fillna("")
    rockunit = pd.read_excel(SOURCE_ROOT / "source_metadata/ROCKUNIT.xlsx", dtype=str).fillna("")

    selected = targets[targets["RESULTNO"].isin(SELECTED_RESULTNOS)].copy()
    selected["_order"] = selected["RESULTNO"].map(
        {resultno: index for index, resultno in enumerate(SELECTED_RESULTNOS)}
    )
    selected = selected.sort_values("_order")
    if list(selected["RESULTNO"]) != list(SELECTED_RESULTNOS):
        raise RuntimeError("Selected RESULTNO set is incomplete")

    pmag_by_result = {
        text_value(row["RESULTNO"]): row for _, row in pmag.iterrows() if text_value(row["RESULTNO"])
    }
    rock_by_unit = {
        text_value(row["ROCKUNITNO"]): row
        for _, row in rockunit.iterrows()
        if text_value(row["ROCKUNITNO"])
    }

    rows: list[dict[str, object]] = []
    gold: list[dict[str, object]] = []
    for _, target in selected.iterrows():
        resultno = text_value(target["RESULTNO"])
        refno = text_value(target["REFNO"])
        pmag_row = pmag_by_result[resultno]
        rock_row = rock_by_unit[text_value(pmag_row["ROCKUNITNO"])]
        source_path = SOURCE_ROOT / "parsed_mds" / f"{refno}.txt"
        source_text = source_path.read_text(encoding="utf-8", errors="ignore")
        if not source_text.strip():
            raise RuntimeError(f"Parsed source is blank: {source_path}")

        row: dict[str, object] = {
            "REFNO": refno,
            "RESULTNO": resultno,
            "ROCKUNITNO": text_value(pmag_row["ROCKUNITNO"]),
            "ROCKNAME": text_value(target["ROCKNAME"]),
            "HIMAGAGE": typed_number(pmag_row["HIMAGAGE"]),
            "LOMAGAGE": typed_number(pmag_row["LOMAGAGE"]),
            "HIGHAGE": typed_number(rock_row["HIGHAGE"]),
            "LOWAGE": typed_number(rock_row["LOWAGE"]),
            "B": typed_number(pmag_row["B"]),
            "N": typed_number(pmag_row["N"]),
            "KD": typed_number(pmag_row["KD"]),
            "ED95": typed_number(pmag_row["ED95"]),
            "DEMAGCODE": typed_number(pmag_row["DEMAGCODE"]),
            "TESTS": text_value(pmag_row["TESTS"]),
        }
        for indicator in ("R3", "R5", "R7"):
            row[f"{indicator}_EVIDENCE_TEXT"] = source_excerpt(source_text, PATTERNS[indicator])
            row[f"{indicator}_EVIDENCE_SOURCE"] = str(source_path)
        rows.append(row)
        gold.append(
            {
                "RESULTNO": resultno,
                "REFNO": refno,
                "ROCKNAME": text_value(target["ROCKNAME"]),
                "R3": int(target["R3"]),
                "R5": int(target["R5"]),
                "R7": int(target["R7"]),
            }
        )

    OUTPUT.write_text(
        json.dumps({"input_rows": rows, "gold_rows": gold}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "rows": len(rows),
        "resultnos": [row["RESULTNO"] for row in rows],
        "gold_counts": {
            indicator: {
                str(value): sum(1 for row in gold if row[indicator] == value)
                for value in (0, 1)
            }
            for indicator in ("R3", "R5", "R7")
        },
        "evidence_lengths": {
            indicator: [len(str(row[f"{indicator}_EVIDENCE_TEXT"])) for row in rows]
            for indicator in ("R3", "R5", "R7")
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
