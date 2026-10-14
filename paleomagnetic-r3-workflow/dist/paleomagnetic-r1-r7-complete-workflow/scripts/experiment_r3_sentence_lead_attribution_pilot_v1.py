#!/usr/bin/env python3
"""Outcome-guided pilot of sentence-leading attribution on hard multi-target R3 papers."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_trace_page_graph_features as freeze  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402
from r3_sentence_lead_attribution import analyze_sentence_lead_attribution  # noqa: E402
from r3_trace_page_graph_features import load_trace_concepts  # noqa: E402


DEFAULT_BASELINE = (
    ROOT
    / "tmp/r3_scope_fusion_all_oof_v1"
    / "r3_scope_fusion_all_grouped_oof_v1.npz"
)
DEFAULT_CONCEPTS = (
    ROOT
    / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
    / "trace_page_graph_concepts_v4.json"
)
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_sentence_lead_pilot_v2"

SAMPLE_GROUPS = {
    "mixed_label_moderate_target_count": (
        "3416", "1224", "3110", "3132", "1278", "3695", "3704", "3875"
    ),
    "uniform_label_moderate_target_count": (
        "1742", "2985", "3135", "3166", "3358", "3325", "3858", "3167"
    ),
    "high_target_count": (
        "3456", "3556", "1147", "3124", "2850", "1162", "3404", "3463"
    ),
}
PRIMARY_WEIGHT = 0.60
SENSITIVITY_WEIGHTS = (0.35, 0.60, 0.90)
EXPANSION_GATE = {
    "minimum_union_net_repairs": 5,
    "minimum_mixed_accuracy_gain": 0.05,
    "minimum_uniform_net_repairs": -1,
    "maximum_mcnemar_exact_p": 0.05,
    "minimum_benefiting_refnos": 3,
    "maximum_single_refno_repair_share": 0.50,
}


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def load_page_rows(path: Path, wanted: set[str]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if set(row) - {"REFNO", "pages", "source"}:
                raise ValueError("page cache contains unsupported top-level fields")
            refno = clean(row.get("REFNO"))
            if refno in wanted:
                rows[refno] = row
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"page cache lacks selected REFNO values: {sorted(missing)}")
    return rows


def candidate_probability(
    baseline: np.ndarray,
    records: Sequence[Mapping[str, str]],
    analyses: Mapping[str, Any],
    weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    output = np.asarray(baseline, dtype=np.float64).copy()
    changed = np.zeros(len(records), dtype=bool)
    by_refno: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_refno[record["REFNO"]].append(index)
    for refno, analysis in analyses.items():
        signal_by_target: dict[str, float | None] = {}
        for group in analysis.groups:
            signal = analysis.group_features[group.group_id]["candidate_signal"]
            for target_id in group.target_ids:
                signal_by_target[target_id] = signal
        observed = {value for value in signal_by_target.values() if value is not None}
        # The paper must supply an internal contrast.  Uniform evidence and
        # incomplete target coverage leave the baseline untouched.
        if observed != {0.0, 1.0}:
            continue
        for index in by_refno[refno]:
            signal = signal_by_target.get(records[index]["RESULTNO"])
            if signal is None:
                continue
            output[index] = (1.0 - weight) * output[index] + weight * signal
            changed[index] = True
    return output, changed


def delta_counts(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> dict[str, Any]:
    before = (baseline[mask] >= 0.5) == labels[mask]
    after = (candidate[mask] >= 0.5) == labels[mask]
    repairs = int((~before & after).sum())
    degradations = int((before & ~after).sum())
    return {
        "rows": int(mask.sum()),
        "baseline_accuracy": round(float(before.mean()), 8),
        "candidate_accuracy": round(float(after.mean()), 8),
        "accuracy_gain": round(float(after.mean() - before.mean()), 8),
        "repairs": repairs,
        "degradations": degradations,
        "net_repairs": repairs - degradations,
    }


def mcnemar_exact(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    before = (baseline >= 0.5) == labels
    after = (candidate >= 0.5) == labels
    repaired = int((~before & after).sum())
    degraded = int((before & ~after).sum())
    discordant = repaired + degraded
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(min(repaired, degraded) + 1))
        p_value = min(1.0, 2.0 * tail / (2.0**discordant))
    return {
        "baseline_wrong_candidate_right": repaired,
        "baseline_right_candidate_wrong": degraded,
        "discordant_rows": discordant,
        "two_sided_exact_p": round(float(p_value), 10),
    }


def analyze_refnos(
    refnos: set[str],
    records: Sequence[Mapping[str, str]],
    page_rows: Mapping[str, Mapping[str, Any]],
    concepts: Sequence[Any],
) -> dict[str, Any]:
    by_refno: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for record in records:
        if record["REFNO"] in refnos:
            by_refno[record["REFNO"]].append(record)
    analyses = {}
    for offset, refno in enumerate(sorted(refnos, key=lambda value: int(value)), start=1):
        targets = [
            {"target_id": record["RESULTNO"], "name": record["ROCKNAME"]}
            for record in by_refno[refno]
        ]
        analyses[refno] = analyze_sentence_lead_attribution(
            page_rows[refno]["pages"], targets, concepts
        )
        print(f"sentence-lead extraction: {offset}/{len(refnos)} REFNO={refno}", flush=True)
    return analyses


def report_for_scope(
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    changed: np.ndarray,
    refnos: set[str],
) -> dict[str, Any]:
    mask = np.asarray([record["REFNO"] in refnos for record in records], dtype=bool)
    per_refno = {}
    repair_counts = Counter()
    benefit_refnos = set()
    for refno in sorted(refnos, key=lambda value: int(value)):
        paper_mask = np.asarray([record["REFNO"] == refno for record in records])
        item = delta_counts(labels, baseline, candidate, paper_mask)
        item["changed_rows"] = int((changed & paper_mask).sum())
        per_refno[refno] = item
        if item["net_repairs"] > 0:
            benefit_refnos.add(refno)
        repair_counts[refno] = item["repairs"]
    overall = delta_counts(labels, baseline, candidate, mask)
    overall["changed_rows"] = int((changed & mask).sum())
    overall["mcnemar"] = mcnemar_exact(labels[mask], baseline[mask], candidate[mask])
    total_repairs = sum(repair_counts.values())
    overall["benefiting_refnos"] = len(benefit_refnos)
    overall["largest_single_refno_repair_share"] = round(
        max(repair_counts.values(), default=0) / total_repairs if total_repairs else 0.0,
        8,
    )
    return {"overall": overall, "per_refno": per_refno}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--page-cache", type=Path, default=freeze.DEFAULT_PAGE_CACHE)
    parser.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = {name: path.resolve() for name, path in vars(args).items() if isinstance(path, Path)}
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_identity = freeze.load_three_column_identity(paths["identity"])
    records, _, _ = freeze.select_pdf_backed_records(all_identity, paths["pdf_dir"])
    multiplicity = Counter(record["REFNO"] for record in all_identity)
    pilot_refnos = {refno for values in SAMPLE_GROUPS.values() for refno in values}
    if len(pilot_refnos) != sum(map(len, SAMPLE_GROUPS.values())):
        raise ValueError("pilot groups must be disjoint")
    if any(multiplicity[refno] < 2 for refno in pilot_refnos):
        raise ValueError("pilot contains a non-multi-target paper")

    candidate_freeze = {
        "schema_version": "r3_sentence_lead_attribution_pilot_freeze_v2",
        "status": "FROZEN_BEFORE_RUNTIME_LABEL_ACCESS",
        "development_design": "outcome-guided hard-case pilot; not independent validation",
        "sample_groups": {name: list(values) for name, values in SAMPLE_GROUPS.items()},
        "primary_weight": PRIMARY_WEIGHT,
        "sensitivity_weights": list(SENSITIVITY_WEIGHTS),
        "candidate_rule": (
            "Within a paper, blend baseline probability with 1 for a uniquely sentence-led "
            "target having trusted method evidence or diagnostic+carrier evidence, and with 0 "
            "for a uniquely sentence-led target having only weaker trusted R3 events. Apply "
            "only when both signals coexist; name-only, unanchored, and indistinguishable rows "
            "remain unchanged."
        ),
        "expansion_gate": EXPANSION_GATE,
        "inputs": {
            name: {"path": str(path), "sha256": freeze.file_sha256(path)}
            for name, path in paths.items()
            if name != "output_dir" and path.is_file()
        },
        "online_feature_boundary": {
            "identity_fields": ["REFNO", "RESULTNO", "ROCKNAME"],
            "original_page_text": True,
            "frozen_knowledge_concepts": True,
            "pmagresult_or_rockmag": False,
            "r3_labels": False,
            "evaluation_metrics": False,
        },
    }
    candidate_freeze["inputs"]["code"] = [
        {
            "path": str(path),
            "sha256": freeze.file_sha256(path),
        }
        for path in (
            Path(__file__).resolve(),
            (ROOT / "scripts/r3_sentence_lead_attribution.py").resolve(),
        )
    ]
    freeze.atomic_json(output_dir / "candidate_freeze.json", candidate_freeze)

    # Feature extraction is complete and serialized before labels are opened.
    page_rows = load_page_rows(paths["page_cache"], pilot_refnos)
    concepts = load_trace_concepts(paths["concepts"])
    analyses = analyze_refnos(pilot_refnos, records, page_rows, concepts)
    evidence_path = output_dir / "pilot_evidence.jsonl"
    with evidence_path.open("w", encoding="utf-8") as handle:
        for refno in sorted(analyses, key=lambda value: int(value)):
            handle.write(json.dumps({"REFNO": refno, **analyses[refno].as_dict()}, ensure_ascii=False))
            handle.write("\n")

    # First runtime label access.  Baseline probabilities are terminal OOF
    # model outputs; their embedded labels are used only for alignment checks.
    label_map = trace_eval.load_labels(
        paths["targets"], {record["RESULTNO"] for record in records}
    )
    labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
    with np.load(paths["baseline"], allow_pickle=False) as source:
        baseline = source["scope_v3_plus_no_strict"].astype(np.float64)
        embedded_labels = source["labels"].astype(np.int8)
    if baseline.shape != labels.shape or not np.array_equal(embedded_labels, labels):
        raise ValueError("baseline/label alignment changed")

    sensitivity = {}
    candidate_by_weight = {}
    changed_by_weight = {}
    for weight in SENSITIVITY_WEIGHTS:
        candidate, changed = candidate_probability(baseline, records, analyses, weight)
        candidate_by_weight[weight] = candidate
        changed_by_weight[weight] = changed
        sensitivity[str(weight)] = report_for_scope(
            records, labels, baseline, candidate, changed, pilot_refnos
        )["overall"]

    primary = candidate_by_weight[PRIMARY_WEIGHT]
    primary_changed = changed_by_weight[PRIMARY_WEIGHT]
    union_report = report_for_scope(
        records, labels, baseline, primary, primary_changed, pilot_refnos
    )
    group_reports = {}
    for group, values in SAMPLE_GROUPS.items():
        group_reports[group] = report_for_scope(
            records, labels, baseline, primary, primary_changed, set(values)
        )

    union = union_report["overall"]
    mixed = group_reports["mixed_label_moderate_target_count"]["overall"]
    uniform = group_reports["uniform_label_moderate_target_count"]["overall"]
    gate_checks = {
        "union_net_repairs": union["net_repairs"] >= EXPANSION_GATE["minimum_union_net_repairs"],
        "mixed_accuracy_gain": mixed["accuracy_gain"] >= EXPANSION_GATE["minimum_mixed_accuracy_gain"],
        "uniform_net_repairs": uniform["net_repairs"] >= EXPANSION_GATE["minimum_uniform_net_repairs"],
        "mcnemar_exact_p": union["mcnemar"]["two_sided_exact_p"] < EXPANSION_GATE["maximum_mcnemar_exact_p"],
        "benefiting_refnos": union["benefiting_refnos"] >= EXPANSION_GATE["minimum_benefiting_refnos"],
        "single_refno_dominance": union["largest_single_refno_repair_share"] <= EXPANSION_GATE["maximum_single_refno_repair_share"],
    }
    expand = all(gate_checks.values())

    expanded_report = None
    if expand:
        multi_refnos = {
            refno for refno, count in multiplicity.items() if count >= 2 and any(record["REFNO"] == refno for record in records)
        }
        missing = multi_refnos - page_rows.keys()
        if missing:
            page_rows.update(load_page_rows(paths["page_cache"], missing))
        remaining = multi_refnos - analyses.keys()
        analyses.update(analyze_refnos(remaining, records, page_rows, concepts))
        expanded, expanded_changed = candidate_probability(
            baseline, records, analyses, PRIMARY_WEIGHT
        )
        expanded_report = report_for_scope(
            records, labels, baseline, expanded, expanded_changed, multi_refnos
        )

    duplicate_name_rows = 0
    duplicate_name_disagreeing_labels = 0
    for refno, analysis in analyses.items():
        if refno not in pilot_refnos:
            continue
        for group in analysis.groups:
            if len(group.target_ids) <= 1:
                continue
            duplicate_name_rows += len(group.target_ids)
            values = {label_map[target_id] for target_id in group.target_ids}
            if len(values) > 1:
                duplicate_name_disagreeing_labels += len(group.target_ids)

    report = {
        "schema_version": "r3_sentence_lead_attribution_pilot_report_v2",
        "status": "COMPLETE_OUTCOME_GUIDED_DEVELOPMENT_PILOT",
        "primary_weight": PRIMARY_WEIGHT,
        "pilot_union": union_report,
        "pilot_groups": group_reports,
        "sensitivity_analysis": sensitivity,
        "coverage": {
            "papers": len(pilot_refnos),
            "groups_with_explicit_anchor": sum(
                feature["explicit_anchor_count"] > 0
                for refno, analysis in analyses.items()
                if refno in pilot_refnos
                for feature in analysis.group_features.values()
            ),
            "groups_with_strong_support": sum(
                feature["strong_support"]
                for refno, analysis in analyses.items()
                if refno in pilot_refnos
                for feature in analysis.group_features.values()
            ),
            "groups_with_carrier_only": sum(
                feature["carrier_only"]
                for refno, analysis in analyses.items()
                if refno in pilot_refnos
                for feature in analysis.group_features.values()
            ),
            "duplicate_name_rows": duplicate_name_rows,
            "duplicate_name_rows_with_internally_disagreeing_labels": duplicate_name_disagreeing_labels,
        },
        "expansion_decision": {
            "gate_checks": gate_checks,
            "expanded_to_all_multi_target_papers": expand,
            "expanded_report": expanded_report,
        },
        "claim_boundary": {
            "outcome_guided_sample_selection": True,
            "independent_validation": False,
            "pmagresult_or_rockmag_used_by_attribution": False,
            "labels_or_metrics_used_by_attribution": False,
            "baseline_oof_probability_used_only_by_frozen_postprocessor": True,
            "labels_used_only_after_feature_freeze_for_evaluation": True,
        },
        "artifacts": {
            "candidate_freeze": {
                "path": str(output_dir / "candidate_freeze.json"),
                "sha256": freeze.file_sha256(output_dir / "candidate_freeze.json"),
            },
            "evidence": {
                "path": str(evidence_path),
                "sha256": freeze.file_sha256(evidence_path),
            },
        },
    }
    freeze.atomic_json(output_dir / "report.json", report)
    print(json.dumps({"report": str(output_dir / "report.json"), "pilot": union, "gate": gate_checks, "expanded": expand}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
