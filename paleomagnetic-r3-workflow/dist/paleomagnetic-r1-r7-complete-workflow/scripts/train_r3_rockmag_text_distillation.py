#!/usr/bin/env python3
"""Build a frozen R3 ROCKMAG-to-original-text distillation artifact.

This is an offline, historically supervised training utility.  It uses R3 and
ROCKMAG only for the pre-registered training partition.  The held-out
partition contributes identity counts only; its labels, record identifiers,
target names, and ROCKMAG cells are never dereferenced by this process.

The emitted artifact contains aggregate concept statistics and safe patterns
for later original-paper retrieval.  It never contains row identities, target
names, raw ROCKMAG values, or source-paper quotations.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = ROOT / "data/targets.xlsx"
DEFAULT_PMAGRESULT = ROOT / "data/source_metadata/PMAGRESULT.xlsx"
DEFAULT_OUTPUT = ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json"

SPLIT_SALT = "r3-rockmag-distillation-v1"
HOLDOUT_FRACTION = 0.20
WILSON_Z_95 = 1.959963984540054


# Field aliases are used only by this offline trainer.  In particular, terse
# legacy codes such as OP and TS must never become original-paper regexes.
METHOD_TAXONOMY: tuple[dict[str, Any], ...] = (
    {
        "concept_id": "irm_acquisition",
        "field_patterns": (r"\bS?IRM\b", r"isothermal\s+remanent\s+magneti[sz]ation"),
        "runtime_original_text_patterns": (
            r"\b(?:IRM|SIRM)\b",
            r"isothermal\s+remanent\s+magneti[sz]ation",
            r"(?:IRM|remanence)\s+acquisition\s+curves?",
        ),
    },
    {
        "concept_id": "thermomagnetic_curve",
        "field_patterns": (
            r"\bJ(?:s|rs)[- ]?T\b",
            r"\bHcs[- ]?T\b",
            r"thermo[- ]?magnetic",
        ),
        "runtime_original_text_patterns": (
            r"thermo[- ]?magnetic",
            r"\bJ(?:s|rs)[- ]?\(?T\)?\b",
            r"saturation\s+magneti[sz]ation.{0,60}temperature",
        ),
    },
    {
        "concept_id": "temperature_dependent_susceptibility",
        "field_patterns": (
            r"\bSUSC[- ]?T\b",
            r"\b(?:kappa|chi|k)[- ]?T\b",
            r"temperature[- ]dependent\s+(?:magnetic\s+)?susceptibility",
        ),
        "runtime_original_text_patterns": (
            r"temperature[- ]dependent\s+(?:magnetic\s+)?susceptibility",
            r"susceptibility.{0,80}(?:heating|cooling|temperature)",
            r"(?:heating|cooling|temperature).{0,80}susceptibility",
            r"\b(?:kappa|chi|[κχ])[- ]?T\b",
        ),
    },
    {
        "concept_id": "three_axis_irm",
        "field_patterns": (r"3[- ]axis", r"three[- ]axis", r"\bLowrie\b"),
        "runtime_original_text_patterns": (
            r"three[- ]axis\s+(?:IRM|isothermal\s+remanent\s+magneti[sz]ation)",
            r"3[- ]axis\s+(?:IRM|isothermal\s+remanent\s+magneti[sz]ation)",
            r"\bLowrie(?:[- ]Fuller)?\s+test",
        ),
    },
    {
        "concept_id": "low_temperature_irm",
        "field_patterns": (
            r"low[- ]temperature.{0,30}\bIRM\b",
            r"\bIRM\b.{0,30}low[- ]temperature",
            r"\b(?:Verwey|Morin)\b",
        ),
        "runtime_original_text_patterns": (
            r"low[- ]temperature.{0,100}(?:IRM|isothermal\s+remanent\s+magneti[sz]ation)",
            r"(?:IRM|isothermal\s+remanent\s+magneti[sz]ation).{0,100}low[- ]temperature",
            r"(?:Verwey|Morin)\s+transition",
        ),
    },
    {
        "concept_id": "hysteresis_forc",
        "field_patterns": (r"\bHYST\b", r"hysteresis", r"\bFORC\b"),
        "runtime_original_text_patterns": (
            r"hysteresis\s+(?:loops?|properties|parameters)",
            r"first[- ]order\s+reversal\s+curves?",
            r"\bFORC\s+(?:diagrams?|distributions?|analys[ei]s)",
            r"(?:Day|Henkel)\s+plots?",
        ),
    },
    {
        "concept_id": "magnetic_fabric",
        # Bare legacy AN is deliberately absent.
        "field_patterns": (r"\bAMS\b", r"\bAIR\b", r"\bAAR\b", r"magnetic\s+fabric"),
        "runtime_original_text_patterns": (
            r"anisotropy\s+of\s+magnetic\s+susceptibility",
            r"anisotropy\s+of\s+isothermal\s+remanence",
            r"anisotropy\s+of\s+anhysteretic\s+remanence",
            r"\b(?:AMS|AIR|AAR)\b.{0,50}(?:measurements?|analys[ei]s|fabric)",
        ),
    },
    {
        "concept_id": "petrographic_microscopy",
        "field_patterns": (
            r"\bOP\b",
            r"\bTS\b",
            r"\bSEM\b",
            r"\bTEM\b",
            r"\bEPMA\b",
            r"\bEDS\b",
            r"petrograph",
        ),
        "runtime_original_text_patterns": (
            r"reflected[- ]light\s+microscop",
            r"polished\s+thin\s+sections?",
            r"scanning\s+electron\s+microscop",
            r"transmission\s+electron\s+microscop",
            r"electron\s+microprobe",
            r"energy[- ]dispersive\s+(?:spectroscopy|analysis)",
        ),
    },
    {
        "concept_id": "mineralogical_phase_analysis",
        "field_patterns": (r"\bXRD\b", r"X[- ]?RAY", r"M[oö]ssbauer", r"mineral\s+separation"),
        "runtime_original_text_patterns": (
            r"X[- ]?ray\s+diffraction",
            r"\bXRD\b.{0,50}(?:analys[ei]s|patterns?|measurements?)",
            r"M[oö]ssbauer\s+(?:spectroscopy|spectra)",
            r"mineral\s+separation",
        ),
    },
)

DIAGNOSTIC_TAXONOMY: tuple[dict[str, Any], ...] = (
    {
        "concept_id": "saturation_behavior",
        "field_patterns": (r"\bno\s+sat\b", r"\bsat\.?\s*[0-9]", r"saturat(?:e|ed|ion)"),
        "runtime_original_text_patterns": (
            r"(?:does\s+not|did\s+not|failed\s+to)\s+saturate",
            r"saturat(?:e|ed|ion).{0,50}(?:at|by|above|below)\s*[0-9]",
        ),
    },
    {
        "concept_id": "curie_temperature",
        "field_patterns": (r"\bCT\s*[0-9]", r"Curie\s+(?:point|temperature)"),
        "runtime_original_text_patterns": (r"Curie\s+(?:point|temperature)",),
    },
    {
        "concept_id": "domain_state",
        "field_patterns": (r"\b(?:SD|PSD|MD)\b", r"domain\s+state"),
        "runtime_original_text_patterns": (
            r"(?:single|pseudo[- ]single|multi)[- ]domain",
            r"\b(?:SD|PSD|MD)\b.{0,40}domain",
            r"domain\s+(?:state|size)",
        ),
    },
    {
        "concept_id": "mineral_transition",
        "field_patterns": (r"\b(?:Verwey|Morin)\b",),
        "runtime_original_text_patterns": (r"(?:Verwey|Morin)\s+transition",),
    },
)

CARRIER_TAXONOMY: tuple[dict[str, Any], ...] = (
    {"concept_id": "magnetite", "field_patterns": (r"\bMAGN\b", r"\bmagnetite\b"), "runtime_original_text_patterns": (r"\bmagnetite\b",)},
    {"concept_id": "titanomagnetite", "field_patterns": (r"\bTM\b", r"titano[- ]?magnetite"), "runtime_original_text_patterns": (r"\btitano[- ]?magnetite\b",)},
    {"concept_id": "hematite", "field_patterns": (r"\bHEM\b", r"ha?ematite"), "runtime_original_text_patterns": (r"\bha?ematite\b",)},
    {"concept_id": "maghemite", "field_patterns": (r"\bMAGH\b", r"maghemite"), "runtime_original_text_patterns": (r"\bmaghemite\b",)},
    {"concept_id": "pyrrhotite", "field_patterns": (r"\bPYRR\b", r"pyrrhotite"), "runtime_original_text_patterns": (r"\bpyrrhotite\b",)},
    {"concept_id": "greigite", "field_patterns": (r"\bGREIG\b", r"greigite"), "runtime_original_text_patterns": (r"\bgreigite\b",)},
    {"concept_id": "goethite", "field_patterns": (r"\bGOETH\b", r"goethite"), "runtime_original_text_patterns": (r"\bgoethite\b",)},
    {"concept_id": "ilmenite", "field_patterns": (r"\bILM\b", r"ilmenite"), "runtime_original_text_patterns": (r"\bilmenite\b",)},
)

STANDALONE_EXCLUSIONS: tuple[dict[str, str], ...] = (
    {
        "exclusion_id": "demagnetization_unblocking_interval",
        "field_pattern": r"\bT(?:ub|b)\s*(?:=|>|<|:)",
        "policy": "A component cleaning or unblocking interval is not by itself a carrier investigation.",
    },
    {
        "exclusion_id": "bare_susceptibility",
        "field_pattern": r"^\s*SUSC\s*$",
        "policy": "Bare susceptibility lacks a temperature-dependent carrier diagnostic.",
    },
    {
        "exclusion_id": "bare_anisotropy_code",
        "field_pattern": r"^\s*AN\s*$",
        "policy": "The ambiguous legacy AN code is not an online evidence pattern.",
    },
    {
        "exclusion_id": "bare_carrier_name",
        "field_pattern": "derived:carrier_without_method",
        "policy": "A mineral name without an investigation or diagnostic result is insufficient.",
    },
)


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holdout_bucket(paper_identity: str, salt: str = SPLIT_SALT) -> float:
    digest = hashlib.sha256(f"{salt}\0{paper_identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def is_holdout(paper_identity: str) -> bool:
    return holdout_bucket(paper_identity) < HOLDOUT_FRACTION


def _cell_value(cell: Any) -> Any:
    """Read one cell lazily; tests use this boundary to detect forbidden access."""

    try:
        return cell.value
    except AttributeError:
        return cell


def _header_map(header: Sequence[Any], required: Sequence[str]) -> dict[str, int]:
    names = [clean(_cell_value(cell)) for cell in header]
    result: dict[str, int] = {}
    for name in required:
        matches = [index for index, value in enumerate(names) if value == name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one {name} column; found {len(matches)}")
        result[name] = matches[0]
    return result


def partition_target_rows(
    header: Sequence[Any],
    rows: Iterable[Sequence[Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read supervised cells only after a row is known to be in training."""

    columns = _header_map(header, ("REFNO", "RESULTNO", "ROCKNAME", "R3"))
    training: list[dict[str, Any]] = []
    train_papers: set[str] = set()
    holdout_papers: set[str] = set()
    train_rows = 0
    holdout_rows = 0
    skipped_blank_identity = 0

    for row in rows:
        paper_identity = clean(_cell_value(row[columns["REFNO"]]))
        if not paper_identity:
            skipped_blank_identity += 1
            continue
        if is_holdout(paper_identity):
            # Do not dereference the remaining cells in a held-out row.
            holdout_rows += 1
            holdout_papers.add(paper_identity)
            continue

        train_rows += 1
        train_papers.add(paper_identity)
        record_identity = clean(_cell_value(row[columns["RESULTNO"]]))
        label_value = _cell_value(row[columns["R3"]])
        if not record_identity:
            raise ValueError("training row has an empty record identity")
        try:
            numeric_label = int(float(label_value))
        except (TypeError, ValueError) as exc:
            raise ValueError("training row has a non-binary R3 value") from exc
        if numeric_label not in {0, 1}:
            raise ValueError("training row has a non-binary R3 value")
        # ROCKNAME is intentionally not dereferenced even for training rows; it
        # is not needed for aggregate field-to-text terminology distillation.
        training.append(
            {
                "paper_identity": paper_identity,
                "record_identity": record_identity,
                "label": numeric_label,
            }
        )

    if train_papers & holdout_papers:
        raise AssertionError("paper-group split overlap")
    audit = {
        "training": {"rows": train_rows, "paper_groups": len(train_papers)},
        "holdout": {"rows": holdout_rows, "paper_groups": len(holdout_papers)},
        "skipped_blank_paper_identity_rows": skipped_blank_identity,
    }
    return training, audit


def load_training_targets(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[workbook.sheetnames[0]]
        rows = worksheet.iter_rows()
        header = next(rows, ())
        return partition_target_rows(header, rows)
    finally:
        workbook.close()


def collect_training_rockmag_rows(
    header: Sequence[Any],
    rows: Iterable[Sequence[Any]],
    training_record_identities: set[str],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Dereference ROCKMAG only for record identities in the training set."""

    columns = _header_map(header, ("RESULTNO", "ROCKMAG"))
    values: dict[str, list[str]] = defaultdict(list)
    scanned_rows = 0
    matched_training_rows = 0
    nonempty_training_values = 0
    for row in rows:
        scanned_rows += 1
        record_identity = clean(_cell_value(row[columns["RESULTNO"]]))
        if record_identity not in training_record_identities:
            continue
        matched_training_rows += 1
        value = clean(_cell_value(row[columns["ROCKMAG"]]))
        if value:
            values[record_identity].append(value)
            nonempty_training_values += 1
    return dict(values), {
        "source_rows_scanned_by_identity_only": scanned_rows,
        "training_rows_with_field_cell_access": matched_training_rows,
        "nonempty_training_field_values": nonempty_training_values,
    }


def load_training_rockmag(
    path: Path,
    training_record_identities: set[str],
) -> tuple[dict[str, list[str]], dict[str, int]]:
    workbook = load_workbook(path, read_only=True, data_only=True)
    try:
        worksheet = workbook[workbook.sheetnames[0]]
        rows = worksheet.iter_rows()
        header = next(rows, ())
        return collect_training_rockmag_rows(header, rows, training_record_identities)
    finally:
        workbook.close()


def _compiled_taxonomy(
    taxonomy: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, tuple[re.Pattern[str], ...]], ...]:
    return tuple(
        (
            str(item["concept_id"]),
            tuple(re.compile(pattern, re.I) for pattern in item["field_patterns"]),
        )
        for item in taxonomy
    )


_COMPILED_METHODS = _compiled_taxonomy(METHOD_TAXONOMY)
_COMPILED_DIAGNOSTICS = _compiled_taxonomy(DIAGNOSTIC_TAXONOMY)
_COMPILED_CARRIERS = _compiled_taxonomy(CARRIER_TAXONOMY)
_TUB_INTERVAL = re.compile(r"\bT(?:ub|b)\s*(?:=|>|<|:)", re.I)
_BARE_SUSC = re.compile(r"^\s*SUSC\s*$", re.I)
_BARE_AN = re.compile(r"^\s*AN\s*$", re.I)


def _matches(
    value: str,
    compiled: Sequence[tuple[str, Sequence[re.Pattern[str]]]],
) -> set[str]:
    return {
        concept_id
        for concept_id, patterns in compiled
        if any(pattern.search(value) for pattern in patterns)
    }


def classify_rockmag_value(value: Any) -> dict[str, Any]:
    """Map one legacy field value to aggregate-safe semantic concepts."""

    text = clean(value)
    methods = _matches(text, _COMPILED_METHODS)
    diagnostics = _matches(text, _COMPILED_DIAGNOSTICS)
    carriers = _matches(text, _COMPILED_CARRIERS)
    exclusions: set[str] = set()
    if _TUB_INTERVAL.search(text):
        exclusions.add("demagnetization_unblocking_interval")
    if _BARE_SUSC.fullmatch(text):
        exclusions.add("bare_susceptibility")
    if _BARE_AN.fullmatch(text):
        exclusions.add("bare_anisotropy_code")
    if carriers and not methods:
        exclusions.add("bare_carrier_name")

    # A ROCKMAG-equivalent description needs an actual carrier-oriented method
    # plus either a diagnostic result or a named carrier.  Tub intervals alone
    # are component-demagnetization metadata and therefore cannot satisfy it.
    reportable = bool(methods and (diagnostics or carriers))
    return {
        "method_concepts": sorted(methods),
        "diagnostic_concepts": sorted(diagnostics),
        "carrier_concepts": sorted(carriers),
        "standalone_exclusions": sorted(exclusions),
        "reportable_template": reportable,
    }


def wilson_lower_bound(successes: int, trials: int, z: float = WILSON_Z_95) -> float:
    if trials <= 0:
        return 0.0
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = proportion + z * z / (2.0 * trials)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    )
    return max(0.0, (centre - margin) / denominator)


def _precision_record(positive: int, negative: int, conflict: int = 0) -> dict[str, Any]:
    trials = positive + negative
    return {
        "observations": trials,
        "positive": positive,
        "negative": negative,
        "conflict_excluded": conflict,
        "precision": round(positive / trials, 8) if trials else None,
        "wilson_lower_bound_95": round(wilson_lower_bound(positive, trials), 8),
    }


def _binary_metrics(confusion: Mapping[str, int]) -> dict[str, Any]:
    true_negative = int(confusion.get("true_negative", 0))
    false_positive = int(confusion.get("false_positive", 0))
    false_negative = int(confusion.get("false_negative", 0))
    true_positive = int(confusion.get("true_positive", 0))
    rows = true_negative + false_positive + false_negative + true_positive

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    def f1(correct: int, predicted_wrong: int, missed: int) -> float | None:
        denominator = 2 * correct + predicted_wrong + missed
        return 2 * correct / denominator if denominator else None

    recall_zero = ratio(true_negative, true_negative + false_positive)
    recall_one = ratio(true_positive, true_positive + false_negative)
    balanced_values = [
        value for value in (recall_zero, recall_one) if value is not None
    ]
    return {
        "rows": rows,
        "accuracy": ratio(true_negative + true_positive, rows),
        "balanced_accuracy": (
            sum(balanced_values) / len(balanced_values)
            if balanced_values
            else None
        ),
        "precision_1": ratio(true_positive, true_positive + false_positive),
        "recall_1": recall_one,
        "f1_0": f1(true_negative, false_negative, false_positive),
        "f1_1": f1(true_positive, false_positive, false_negative),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }


def induce_training_statistics(
    training_records: Sequence[Mapping[str, Any]],
    values_by_record: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    row_counts: dict[str, Counter[int]] = defaultdict(Counter)
    paper_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    joined_rows = 0
    nonempty_rows = 0
    reportable_rows = 0
    exclusion_counts: Counter[str] = Counter()
    nonempty_confusion: Counter[str] = Counter()
    reportable_confusion: Counter[str] = Counter()

    for record in training_records:
        values = list(values_by_record.get(str(record["record_identity"]), ()))
        if values:
            joined_rows += 1
        classifications = [classify_rockmag_value(value) for value in values if clean(value)]
        if classifications:
            nonempty_rows += 1
        concepts: set[str] = set()
        reportable = False
        for classification in classifications:
            concepts.update(f"method:{value}" for value in classification["method_concepts"])
            concepts.update(f"diagnostic:{value}" for value in classification["diagnostic_concepts"])
            concepts.update(f"carrier:{value}" for value in classification["carrier_concepts"])
            exclusion_counts.update(classification["standalone_exclusions"])
            reportable = reportable or bool(classification["reportable_template"])
        if reportable:
            reportable_rows += 1
            concepts.add("template:rockmag_reportable")

        label = int(record["label"])
        nonempty_prediction = int(bool(classifications))
        reportable_prediction = int(reportable)
        for prediction, confusion in (
            (nonempty_prediction, nonempty_confusion),
            (reportable_prediction, reportable_confusion),
        ):
            key = (
                "true_positive"
                if prediction == 1 and label == 1
                else "false_positive"
                if prediction == 1
                else "true_negative"
                if label == 0
                else "false_negative"
            )
            confusion[key] += 1
        paper_identity = str(record["paper_identity"])
        for concept in concepts:
            row_counts[concept][label] += 1
            paper_labels[(paper_identity, concept)].add(label)

    paper_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for (_, concept), labels in paper_labels.items():
        status = "conflict" if len(labels) > 1 else "positive" if 1 in labels else "negative"
        paper_counts[concept][status] += 1

    concepts = sorted(set(row_counts) | set(paper_counts))
    row_statistics = {
        concept: _precision_record(row_counts[concept][1], row_counts[concept][0])
        for concept in concepts
    }
    paper_statistics = {
        concept: _precision_record(
            paper_counts[concept]["positive"],
            paper_counts[concept]["negative"],
            paper_counts[concept]["conflict"],
        )
        for concept in concepts
    }
    return {
        "aggregate_counts": {
            "training_rows": len(training_records),
            "rows_joined_to_source_table": joined_rows,
            "rows_with_nonempty_field_value": nonempty_rows,
            "rows_matching_reportable_template": reportable_rows,
        },
        "row_level_statistics": row_statistics,
        "paper_deduplicated_statistics": paper_statistics,
        "standalone_exclusion_counts": dict(sorted(exclusion_counts.items())),
        "training_rule_evaluation": {
            "rockmag_nonempty_baseline": _binary_metrics(nonempty_confusion),
            "reportable_semantic_template": _binary_metrics(reportable_confusion),
            "scope": "pre_registered_training_partition_only",
            "independent_generalization_claim": False,
        },
    }


def _runtime_taxonomy(
    taxonomy: Sequence[Mapping[str, Any]],
    statistics: Mapping[str, Any],
    prefix: str,
) -> list[dict[str, Any]]:
    row_statistics = statistics["row_level_statistics"]
    paper_statistics = statistics["paper_deduplicated_statistics"]
    empty = _precision_record(0, 0)
    output: list[dict[str, Any]] = []
    for item in taxonomy:
        concept_id = str(item["concept_id"])
        statistic_key = f"{prefix}:{concept_id}"
        output.append(
            {
                "concept_id": concept_id,
                "runtime_original_text_patterns": list(
                    item["runtime_original_text_patterns"]
                ),
                "training_statistics": {
                    "row_level": dict(row_statistics.get(statistic_key, empty)),
                    "paper_deduplicated": dict(
                        paper_statistics.get(statistic_key, empty)
                    ),
                },
            }
        )
    return output


def build_artifact(targets_path: Path, pmagresult_path: Path) -> dict[str, Any]:
    training_records, split_counts = load_training_targets(targets_path)
    training_record_identities = {
        str(record["record_identity"]) for record in training_records
    }
    values_by_record, source_access = load_training_rockmag(
        pmagresult_path,
        training_record_identities,
    )
    statistics = induce_training_statistics(training_records, values_by_record)
    return {
        "schema_version": "r3_rockmag_text_distillation_v1",
        "artifact_class": "historically_supervised_distillation",
        "profile": "rockmag_text_distilled_v1",
        "provenance": {
            "phase": "offline_training_only",
            "label_supervision": True,
            "supervision_scope": "pre_registered_training_partition_only",
            "source_fields_used_offline": ["R3", "ROCKMAG"],
            "identity_use_offline": {
                "paper_identity": "paper-group split only",
                "record_identity": "training-only table join only",
            },
            "raw_field_values_retained": False,
            "source_paper_quotes_retained": False,
            "row_identity_mapping_retained": False,
            "input_sha256": {
                "target_manifest_source": file_sha256(targets_path),
                "source_metadata_table": file_sha256(pmagresult_path),
            },
            "input_hash_access_is_nonsemantic": True,
        },
        "split": {
            "unit": "paper_identity",
            "algorithm": "SHA256(salt + NUL + canonical paper identity), first 8 bytes as unsigned big-endian / 2^64",
            "salt": SPLIT_SALT,
            "holdout_fraction": HOLDOUT_FRACTION,
            "held_out_supervision_accessed": False,
            "identity_counts": split_counts,
        },
        "offline_source_access": source_access,
        "runtime_policy": {
            "allowed_scoring_sources": [
                "identity manifest",
                "original paper page text",
                "this frozen aggregate artifact",
            ],
            "source_metadata_table_access": False,
            "supervised_field_access": False,
            "previous_prediction_access": False,
            "contains_record_answer_mapping": False,
        },
        "method_taxonomy": _runtime_taxonomy(
            METHOD_TAXONOMY,
            statistics,
            "method",
        ),
        "diagnostic_taxonomy": _runtime_taxonomy(
            DIAGNOSTIC_TAXONOMY,
            statistics,
            "diagnostic",
        ),
        "carrier_taxonomy": _runtime_taxonomy(
            CARRIER_TAXONOMY,
            statistics,
            "carrier",
        ),
        "standalone_exclusions": [dict(item) for item in STANDALONE_EXCLUSIONS],
        "decision_policy": {
            "positive_requires_any": [
                "target-bound performed method distilled from a ROCKMAG semantic class",
                "target-bound quantified diagnostic result that entails a performed method",
                "target-bound performed method plus diagnostic result",
                "target-bound performed method plus named carrier identification",
            ],
            "always_required": [
                "original-paper evidence",
                "target binding",
                "non-reference context",
                "performed-or-reported experimental context",
            ],
            "standalone_non_support": [
                "component unblocking-temperature interval",
                "bare susceptibility code",
                "bare anisotropy code",
                "bare carrier name",
            ],
            "rockmag_proxy_policy": (
                "distilled_method_or_self_authenticating_result"
            ),
            "identity_granularity": "canonical_target",
            "missing_or_ambiguous_source": "ABSTAIN",
        },
        "training_statistics": statistics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--pmagresult", type=Path, default=DEFAULT_PMAGRESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = build_artifact(args.targets, args.pmagresult)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "schema_version": artifact["schema_version"],
                "split": artifact["split"]["identity_counts"],
                **artifact["training_statistics"]["aggregate_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
