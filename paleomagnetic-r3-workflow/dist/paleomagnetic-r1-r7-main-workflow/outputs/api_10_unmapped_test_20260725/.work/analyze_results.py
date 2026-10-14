from __future__ import annotations

import json
from pathlib import Path
import re


WORK_DIR = Path(__file__).parent
OUTPUT_DIR = WORK_DIR.parent


def main() -> None:
    selection = json.loads((WORK_DIR / "selection.json").read_text(encoding="utf-8"))
    html = (OUTPUT_DIR / "r1_r7_results_api.html").read_text(encoding="utf-8")
    match = re.search(r"<script>const DATA=(.*?);</script>", html, re.S)
    if not match:
        raise RuntimeError("Could not locate embedded result data")
    data = json.loads(match.group(1))
    actual = {str(row["RESULTNO"]): row for row in data["rows"]}

    comparisons: list[dict[str, object]] = []
    for gold in selection["gold_rows"]:
        resultno = str(gold["RESULTNO"])
        row = actual[resultno]
        comparison: dict[str, object] = {
            "RESULTNO": resultno,
            "REFNO": gold["REFNO"],
            "ROCKNAME": gold["ROCKNAME"],
        }
        for indicator in ("R3", "R5", "R7"):
            evidence = row["evidence"][indicator]
            audit = evidence.get("api_audit") or {}
            comparison[f"{indicator}_GOLD"] = int(gold[indicator])
            comparison[f"{indicator}_PRED"] = int(row["scores"][indicator])
            comparison[f"{indicator}_CORRECT"] = int(
                row["scores"][indicator] == int(gold[indicator])
            )
            comparison[f"{indicator}_OUTCOME"] = evidence.get("outcome")
            comparison[f"{indicator}_STATUS"] = evidence.get("status")
            comparison[f"{indicator}_REASONING"] = evidence.get("reasoning")
            comparison[f"{indicator}_QUOTE"] = evidence.get("quote")
            comparison[f"{indicator}_RESPONSE_ID"] = audit.get("response_id")
        comparisons.append(comparison)

    summaries: dict[str, dict[str, object]] = {}
    for indicator in ("R3", "R5", "R7"):
        correct = sum(int(row[f"{indicator}_CORRECT"]) for row in comparisons)
        unknown = sum(row[f"{indicator}_OUTCOME"] == "UNKNOWN" for row in comparisons)
        decided_rows = [row for row in comparisons if row[f"{indicator}_OUTCOME"] != "UNKNOWN"]
        decided_correct = sum(int(row[f"{indicator}_CORRECT"]) for row in decided_rows)
        gold_values = [int(row[f"{indicator}_GOLD"]) for row in comparisons]
        predictions = [int(row[f"{indicator}_PRED"]) for row in comparisons]
        summaries[indicator] = {
            "total": len(comparisons),
            "correct": correct,
            "accuracy": correct / len(comparisons),
            "unknown": unknown,
            "decided": len(decided_rows),
            "decided_correct": decided_correct,
            "decided_accuracy": decided_correct / len(decided_rows) if decided_rows else None,
            "true_positive": sum(gold == 1 and pred == 1 for gold, pred in zip(gold_values, predictions)),
            "true_negative": sum(gold == 0 and pred == 0 for gold, pred in zip(gold_values, predictions)),
            "false_positive": sum(gold == 0 and pred == 1 for gold, pred in zip(gold_values, predictions)),
            "false_negative": sum(gold == 1 and pred == 0 for gold, pred in zip(gold_values, predictions)),
        }

    response_ids = [
        row[f"{indicator}_RESPONSE_ID"]
        for row in comparisons
        for indicator in ("R3", "R5", "R7")
    ]
    report = {
        "test_design": {
            "rows": len(comparisons),
            "resultnos_native_unmapped": True,
            "result_map_hits": {"R3": 0, "R5": 0, "R7": 0},
            "source": "targets.xlsx gold labels plus parsed_mds original-paper excerpts",
            "score_policy": "Compare final compatibility score; UNKNOWN maps to R3=0 and R5/R7=1.",
        },
        "summary": summaries,
        "api_audit": {
            "expected_responses": len(comparisons) * 3,
            "responses_with_id": sum(bool(value) for value in response_ids),
            "unique_response_ids": len({value for value in response_ids if value}),
        },
        "rows": comparisons,
    }
    (OUTPUT_DIR / "r357_accuracy.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (WORK_DIR / "accuracy_data.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "summary": summaries,
        "api_audit": report["api_audit"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
