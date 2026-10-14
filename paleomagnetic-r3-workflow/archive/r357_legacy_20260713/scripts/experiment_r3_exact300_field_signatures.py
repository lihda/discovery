#!/usr/bin/env python3
"""Evaluate record-free ROCKMAG field signatures on canonical exact300 R3.

The high-resolution feature layer is derived only from the frozen aggregate
``r3_positive_rockmag_expert_kb_v2`` artifact and original-paper text.  It
matches method, diagnostic, and carrier combinations within bounded
target-local and own-study passages, guards ambiguous abbreviations with
magnetic context, and emits aggregate counts/scores only.

The 2,172-row signature matrix is cached and frozen before R3 labels are
opened.  Each canonical outer fold excludes every auxiliary row belonging to
an outer-test REFNO.  No source-field cell, answer mapping, or old prediction
is an input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_own_study_events as own_study  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/field_signatures_v1"
SIGNATURE_CACHE_NAME = "signature_features.npz"
SIGNATURE_MANIFEST_NAME = "signature_features.manifest.json"
FREEZE_NAME = "feature_freeze.json"
REPORT_NAME = "report.json"
OOF_NAME = "oof_probabilities.npz"

EXPECTED_ROWS = auxiliary.EXPECTED_EXACT_ROWS
REFERENCE_ACCURACY = 0.87333333
REFERENCE_FIRST3 = (0.90, 0.86666667, 0.78333333)
FIRST3_CLEAR_REGRESSION_MEAN = 0.82
TARGET_ACCURACY = auxiliary.TARGET_ACCURACY
SCOPES = ("target_local", "target_own", "paper_own")
FORBIDDEN_ARRAY_NAME = re.compile(
    r"(?i)^(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)$"
)

MAGNETIC_CONTEXT_RE = re.compile(
    r"\b(?:magnet\w*|reman\w*|rock[- ]magnet\w*|mineral\w*|oxide\w*|"
    r"susceptib\w*|coerciv\w*|hysteres\w*|saturat\w*|domain\w*|"
    r"curie|verwey|morin|thermo\w*|heating|cooling|demagnet\w*|"
    r"acquisition|back[- ]field|applied\s+field|microscop\w*|grain\w*|"
    r"sample\w*|specimen\w*|mT|tesla|oersted)\b",
    re.I,
)
ABBREVIATION_GUARDS: Mapping[str, re.Pattern[str]] = {
    "irm_acquisition": re.compile(
        r"\b(?:acquisition|back[- ]field|demagnet\w*|reman\w*|magnet\w*|"
        r"applied\s+field|saturat\w*|mT|tesla)\b",
        re.I,
    ),
    "thermomagnetic_curve": re.compile(
        r"\b(?:thermo\w*|temperature|heating|cooling|magneti[sz]ation|curve)\b",
        re.I,
    ),
    "temperature_dependent_susceptibility": re.compile(
        r"\b(?:susceptib\w*|temperature|heating|cooling|kappa|chi)\b",
        re.I,
    ),
    "petrographic_microscopy": re.compile(
        r"\b(?:microscop\w*|petrograph\w*|thin\s+section|grain\w*|mineral\w*|"
        r"oxide\w*|electron\w*|spectr\w*)\b",
        re.I,
    ),
    "mineralogical_phase_analysis": re.compile(
        r"\b(?:x[- ]?ray|diffraction|mineral\w*|phase\w*|pattern\w*|analys\w*)\b",
        re.I,
    ),
    "magnetic_fabric": re.compile(
        r"\b(?:magnet\w*|anisotrop\w*|susceptib\w*|reman\w*|fabric)\b",
        re.I,
    ),
    "curie_temperature": re.compile(
        r"\b(?:curie|temperature|heating|cooling|°?\s*C|kelvin)\b",
        re.I,
    ),
    "domain_state": re.compile(
        r"\b(?:domain|grain\w*|magnet\w*|hysteres\w*|single|pseudo|multi)\b",
        re.I,
    ),
    "coercivity_parameter": re.compile(
        r"\b(?:coerciv\w*|hysteres\w*|back[- ]field|ratio|mT|tesla|oersted)\b",
        re.I,
    ),
}
TUB_RE = re.compile(
    r"\bT(?:ub|b)\s*(?:=|>|<|:)|\bunblocking\s+temperature\b", re.I
)
SUSCEPTIBILITY_RE = re.compile(r"\b(?:magnetic\s+)?susceptibility\b", re.I)
AMBIGUOUS_AN_RE = re.compile(r"(?<![A-Za-z])AN(?![A-Za-z])")


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.view(np.uint8).tobytes()).hexdigest()


def concept_key(category: str, concept_id: str) -> str:
    return f"{category}:{concept_id}"


def validate_kb_boundary(artifact: Mapping[str, Any]) -> None:
    runtime = artifact.get("runtime_policy", {})
    provenance = artifact.get("provenance", {})
    if (
        artifact.get("schema_version") != "r3_positive_rockmag_expert_kb_v2"
        or runtime.get("contains_record_answer_mapping") is not False
        or runtime.get("source_table_access") is not False
        or runtime.get("label_access") is not False
        or provenance.get("raw_field_values_retained") is not False
        or provenance.get("row_identity_mapping_retained") is not False
        or provenance.get("source_quotes_retained") is not False
        or provenance.get("held_out_field_cells_opened") is not False
        or provenance.get("held_out_labels_opened") is not False
        or provenance.get("prior_predictions_or_evaluations_used") is not False
    ):
        raise ValueError("positive expert KB violates the record-free runtime boundary")


def load_signature_kb(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(artifact, Mapping):
        raise ValueError("positive expert KB must be a JSON object")
    validate_kb_boundary(artifact)

    concepts: list[dict[str, Any]] = []
    concept_by_key: dict[str, dict[str, Any]] = {}
    for taxonomy_name, category in (
        ("method_taxonomy", "method"),
        ("diagnostic_taxonomy", "diagnostic"),
        ("carrier_taxonomy", "carrier"),
    ):
        for item in artifact.get(taxonomy_name, []):
            concept_id = str(item["concept_id"])
            patterns = [
                re.compile(str(expression), re.I | re.S)
                for expression in item.get("runtime_original_text_patterns", [])
            ]
            if not patterns:
                raise ValueError(f"KB concept has no runtime patterns: {concept_id}")
            statistics = item.get("statistics", {}).get("row_level", {})
            concept = {
                "category": category,
                "concept_id": concept_id,
                "key": concept_key(category, concept_id),
                "patterns": patterns,
                "positive_precision": float(statistics.get("positive_precision") or 0.0),
                "wilson_lower": float(
                    statistics.get("positive_precision_wilson_lower_95") or 0.0
                ),
            }
            concepts.append(concept)
            concept_by_key[concept["key"]] = concept

    edges: list[dict[str, Any]] = []
    for edge in artifact.get("aggregate_statistics", {}).get(
        "positive_concept_cooccurrences", []
    ):
        source = str(edge["source_concept"])
        target = str(edge["target_concept"])
        source_category = source.split(":", 1)[0]
        target_category = target.split(":", 1)[0]
        categories = {source_category, target_category}
        if source not in concept_by_key or target not in concept_by_key:
            continue
        if "method" not in categories or categories == {"method"}:
            continue
        edges.append(
            {
                "source": source,
                "target": target,
                "positive_row_count": int(edge["positive_row_count"]),
                "positive_paper_count": int(edge["positive_paper_count"]),
                "coverage": float(edge["positive_row_coverage"]),
            }
        )

    abbreviations: list[dict[str, Any]] = []
    for item in artifact.get("abbreviation_lexicon", []):
        statistics = item.get("statistics", {}).get("row_level", {})
        abbreviations.append(
            {
                "abbreviation": str(item["abbreviation"]),
                "category": str(item["category"]),
                "concept_id": str(item["concept_id"]),
                "pattern": re.compile(str(item["field_pattern"]), re.I),
                "positive_precision": float(statistics.get("positive_precision") or 0.0),
                "wilson_lower": float(
                    statistics.get("positive_precision_wilson_lower_95") or 0.0
                ),
                "observations": int(statistics.get("observations") or 0),
            }
        )

    category_signatures = [
        str(item["class_id"])
        for item in artifact.get("aggregate_statistics", {}).get(
            "positive_category_signatures", []
        )
    ]
    evidence_classes = [
        str(item["class_id"])
        for item in artifact.get("aggregate_statistics", {}).get(
            "positive_evidence_classes", []
        )
    ]
    nonqualifying = [
        {
            "class_id": str(item["class_id"]),
            "precision": float(
                item.get("statistics", {})
                .get("row_level", {})
                .get("positive_precision")
                or 0.0
            ),
        }
        for item in artifact.get("nonqualifying_classes", [])
    ]
    kb = {
        "concepts": concepts,
        "concept_by_key": concept_by_key,
        "edges": edges,
        "abbreviations": abbreviations,
        "category_signatures": category_signatures,
        "evidence_classes": evidence_classes,
        "nonqualifying": nonqualifying,
    }
    return kb, {
        "path": str(path),
        "sha256": base.file_sha256(path),
        "concept_count": len(concepts),
        "cross_category_method_edge_count": len(edges),
        "abbreviation_count": len(abbreviations),
        "category_signature_count": len(category_signatures),
        "evidence_class_count": len(evidence_classes),
        "nonqualifying_class_count": len(nonqualifying),
        "record_answer_mapping_present": False,
        "raw_field_values_present": False,
    }


def guarded_abbreviation_counts(
    text: str,
    abbreviations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], int]:
    retained: Counter[str] = Counter()
    suppressed = 0
    for item in abbreviations:
        concept_id = str(item["concept_id"])
        guard = ABBREVIATION_GUARDS.get(concept_id, MAGNETIC_CONTEXT_RE)
        for match in item["pattern"].finditer(text):
            window = text[max(0, match.start() - 140) : min(len(text), match.end() + 140)]
            if guard.search(window) and MAGNETIC_CONTEXT_RE.search(window):
                retained[str(item["abbreviation"])] += 1
            else:
                suppressed += 1
    return dict(retained), suppressed


def category_signature(categories: set[str]) -> str:
    return "+".join(sorted(categories)) if categories else "none"


def evidence_class(categories: set[str]) -> str:
    if categories == {"method", "diagnostic", "carrier"}:
        return "method_diagnostic_carrier"
    if categories == {"method", "carrier"}:
        return "method_carrier"
    if categories == {"method", "diagnostic"}:
        return "method_diagnostic"
    if categories == {"diagnostic", "carrier"}:
        return "diagnostic_carrier_without_method"
    if categories == {"method"}:
        return "method_only"
    if categories == {"carrier"}:
        return "carrier_only"
    return "unclassified_nonempty"


def passage_nonqualifying_classes(
    text: str,
    categories: set[str],
    configured_classes: set[str],
) -> set[str]:
    output: set[str] = set()
    if (
        "tub_unblocking_interval_without_independent_method" in configured_classes
        and TUB_RE.search(text)
        and "method" not in categories
    ):
        output.add("tub_unblocking_interval_without_independent_method")
    if (
        "bare_susceptibility" in configured_classes
        and SUSCEPTIBILITY_RE.search(text)
        and "method" not in categories
    ):
        output.add("bare_susceptibility")
    ambiguous_an = bool(AMBIGUOUS_AN_RE.search(text))
    if (
        "bare_anisotropy" in configured_classes
        and ambiguous_an
        and not re.search(r"\b(?:AMS|AIR|AAR|magnetic\s+fabric|anisotrop\w*)\b", text, re.I)
    ):
        output.add("bare_anisotropy")
    if (
        "bare_susc_an_bundle" in configured_classes
        and ambiguous_an
        and SUSCEPTIBILITY_RE.search(text)
        and "method" not in categories
    ):
        output.add("bare_susc_an_bundle")
    if (
        "carrier_without_method" in configured_classes
        and "carrier" in categories
        and "method" not in categories
    ):
        output.add("carrier_without_method")
    if (
        "unclassified_nonempty" in configured_classes
        and MAGNETIC_CONTEXT_RE.search(text)
        and not categories
    ):
        output.add("unclassified_nonempty")
    return output


def cluster_signature_passages(
    paper: Mapping[str, Any],
    kb: Mapping[str, Any],
) -> list[dict[str, Any]]:
    segment_lookup = {
        int(segment["segment_id"]): segment for segment in paper["segments"]
    }
    by_segment: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for event in paper["events"]:
        by_segment[int(event["segment_id"])].append(event)
    configured_nonqualifying = {
        str(item["class_id"]) for item in kb["nonqualifying"]
    }
    passages: list[dict[str, Any]] = []
    for segment_id, events in by_segment.items():
        ordered = sorted(events, key=lambda item: int(item["start"]))
        clusters: list[list[Mapping[str, Any]]] = []
        for event in ordered:
            if not clusters or int(event["start"]) - int(clusters[-1][-1]["end"]) > 800:
                clusters.append([event])
            else:
                clusters[-1].append(event)
        segment = segment_lookup[segment_id]
        segment_text = str(segment["text"])
        for cluster in clusters:
            start = max(0, min(int(event["start"]) for event in cluster) - 360)
            end = min(
                len(segment_text),
                max(int(event["end"]) for event in cluster) + 360,
            )
            text = segment_text[start:end]
            concepts = {
                concept_key(str(event["category"]), str(event["concept_id"]))
                for event in cluster
            }
            categories = {value.split(":", 1)[0] for value in concepts}
            assigned = Counter(
                str(event["assigned_target"])
                for event in cluster
                if str(event["assigned_target"])
            )
            target_distances: dict[str, int] = {}
            for event in cluster:
                for target, distance in event["distances"].items():
                    if distance is None:
                        continue
                    target_distances[str(target)] = min(
                        int(distance), target_distances.get(str(target), 10**9)
                    )
            abbreviation_counts, suppressed = guarded_abbreviation_counts(
                text, kb["abbreviations"]
            )
            passages.append(
                {
                    "section": str(segment["section"]),
                    "text": text,
                    "concepts": concepts,
                    "categories": categories,
                    "assigned_target": assigned.most_common(1)[0][0] if assigned else "",
                    "target_distances": target_distances,
                    "own_strong": any(bool(event["own_strong"]) for event in cluster),
                    "own_likely": any(bool(event["own_likely"]) for event in cluster),
                    "external_only": all(bool(event["external"]) for event in cluster),
                    "abbreviations": abbreviation_counts,
                    "suppressed_abbreviations": suppressed,
                    "nonqualifying": passage_nonqualifying_classes(
                        text, categories, configured_nonqualifying
                    ),
                }
            )
    return passages


def target_scope_passages(
    passages: Sequence[Mapping[str, Any]],
    target_key: str,
) -> dict[str, list[Mapping[str, Any]]]:
    local = [
        passage
        for passage in passages
        if (
            passage["assigned_target"] == target_key
            or (
                not passage["assigned_target"]
                and passage["target_distances"].get(target_key, 10**9) <= 1200
            )
        )
    ]
    target_own = [
        passage
        for passage in local
        if passage["own_likely"] and not passage["external_only"]
    ]
    paper_own = [
        passage
        for passage in passages
        if passage["own_likely"] and not passage["external_only"]
    ]
    return {
        "target_local": local,
        "target_own": target_own,
        "paper_own": paper_own,
    }


def aggregate_scope(
    scope: str,
    passages: Sequence[Mapping[str, Any]],
    kb: Mapping[str, Any],
) -> tuple[list[float], list[str]]:
    values: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(f"{scope}_{name}")
        values.append(float(value))

    add("passage_count_log", math.log1p(len(passages)))
    add("own_strong_count_log", math.log1p(sum(bool(item["own_strong"]) for item in passages)))
    add("external_only_count_log", math.log1p(sum(bool(item["external_only"]) for item in passages)))
    add("suppressed_abbreviation_count_log", math.log1p(sum(int(item["suppressed_abbreviations"]) for item in passages)))

    for concept in kb["concepts"]:
        count = sum(concept["key"] in item["concepts"] for item in passages)
        add(f"concept_{concept['key'].replace(':', '_')}_count_log", math.log1p(count))

    edge_counts: list[int] = []
    for edge in kb["edges"]:
        count = sum(
            edge["source"] in item["concepts"]
            and edge["target"] in item["concepts"]
            for item in passages
        )
        edge_counts.append(count)
        edge_name = f"{edge['source']}__{edge['target']}".replace(":", "_")
        add(f"edge_{edge_name}_count_log", math.log1p(count))

    signatures = Counter(category_signature(set(item["categories"])) for item in passages)
    for signature in kb["category_signatures"]:
        add(f"category_signature_{signature}_count_log", math.log1p(signatures[signature]))
    evidence = Counter(evidence_class(set(item["categories"])) for item in passages)
    for class_id in kb["evidence_classes"]:
        add(f"evidence_class_{class_id}_count_log", math.log1p(evidence[class_id]))

    abbreviation_totals: Counter[str] = Counter()
    for passage in passages:
        abbreviation_totals.update(passage["abbreviations"])
    precision_sum = 0.0
    wilson_sum = 0.0
    observed_count = 0
    for item in kb["abbreviations"]:
        count = int(abbreviation_totals[item["abbreviation"]])
        add(f"abbreviation_{item['abbreviation']}_guarded_count_log", math.log1p(count))
        precision_sum += count * float(item["positive_precision"])
        wilson_sum += count * float(item["wilson_lower"])
        observed_count += count * int(item["observations"] > 0)
    add("abbreviation_precision_weighted_sum_log", math.log1p(precision_sum))
    add("abbreviation_wilson_weighted_sum_log", math.log1p(wilson_sum))
    add("abbreviation_with_training_support_count_log", math.log1p(observed_count))

    nonqualifying = Counter(
        class_id
        for passage in passages
        for class_id in passage["nonqualifying"]
    )
    for item in kb["nonqualifying"]:
        add(f"nonqualifying_{item['class_id']}_count_log", math.log1p(nonqualifying[item["class_id"]]))

    categories_by_passage = [set(item["categories"]) for item in passages]
    add("method_carrier_chain_count_log", math.log1p(sum({"method", "carrier"}.issubset(item) for item in categories_by_passage)))
    add("method_diagnostic_chain_count_log", math.log1p(sum({"method", "diagnostic"}.issubset(item) for item in categories_by_passage)))
    add("triple_chain_count_log", math.log1p(sum({"method", "diagnostic", "carrier"}.issubset(item) for item in categories_by_passage)))
    add("distinct_method_count_log", math.log1p(len({concept for passage in passages for concept in passage["concepts"] if concept.startswith("method:")})))
    add("distinct_diagnostic_count_log", math.log1p(len({concept for passage in passages for concept in passage["concepts"] if concept.startswith("diagnostic:")})))
    add("distinct_carrier_count_log", math.log1p(len({concept for passage in passages for concept in passage["concepts"] if concept.startswith("carrier:")})))
    edge_weight = sum(
        count * float(edge["coverage"])
        for count, edge in zip(edge_counts, kb["edges"])
    )
    add("positive_edge_coverage_weighted_sum_log", math.log1p(edge_weight))
    add("matched_edge_distinct_count_log", math.log1p(sum(count > 0 for count in edge_counts)))
    add("nonqualifying_total_log", math.log1p(sum(nonqualifying.values())))
    return values, names


def build_signature_features(
    records: Sequence[dict[str, str]],
    structured_pages: Mapping[str, Sequence[Mapping[str, Any]]],
    kb: Mapping[str, Any],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    targets_by_refno: dict[str, dict[str, str]] = defaultdict(dict)
    for record in records:
        target_key = canonical_target_identity(record["ROCKNAME"])
        targets_by_refno[record["REFNO"]].setdefault(target_key, record["ROCKNAME"])

    vectors: dict[tuple[str, str], list[float]] = {}
    feature_names: list[str] | None = None
    aggregate_audit = Counter()
    for paper_number, refno in enumerate(sorted(targets_by_refno), start=1):
        paper = own_study.extract_paper_events(
            structured_pages[refno], targets_by_refno[refno], kb["concepts"]
        )
        passages = cluster_signature_passages(paper, kb)
        aggregate_audit["papers"] += 1
        aggregate_audit["passages"] += len(passages)
        aggregate_audit["own_passages"] += sum(
            item["own_likely"] and not item["external_only"] for item in passages
        )
        for target_key in targets_by_refno[refno]:
            scoped = target_scope_passages(passages, target_key)
            vector: list[float] = []
            names: list[str] = []
            for scope in SCOPES:
                scope_values, scope_names = aggregate_scope(scope, scoped[scope], kb)
                vector.extend(scope_values)
                names.extend(scope_names)
                aggregate_audit[f"{scope}_passages"] += len(scoped[scope])
            vectors[(refno, target_key)] = vector
            feature_names = feature_names or names
            if feature_names != names:
                raise AssertionError("field-signature feature order changed")
        if paper_number % 100 == 0:
            print(
                f"signature cache: papers={paper_number}/{len(targets_by_refno)}",
                flush=True,
            )
    matrix = np.asarray(
        [
            vectors[(record["REFNO"], canonical_target_identity(record["ROCKNAME"]))]
            for record in records
        ],
        dtype=np.float32,
    )
    if matrix.shape[0] != len(records) or not np.isfinite(matrix).all():
        raise AssertionError("field-signature matrix is incomplete")
    return matrix, feature_names or [], {
        key: int(value) for key, value in sorted(aggregate_audit.items())
    }


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if any(FORBIDDEN_ARRAY_NAME.search(name) for name in arrays):
        raise ValueError("cache key contains a label or record identifier")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def make_signature_hgb() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=240,
        learning_rate=0.035,
        max_leaf_nodes=7,
        min_samples_leaf=12,
        l2_regularization=2.0,
        random_state=base.RANDOM_STATE,
    )


def make_signature_rf() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=450,
        max_features="sqrt",
        min_samples_leaf=3,
        n_jobs=1,
        random_state=base.RANDOM_STATE,
    )


def candidate_specs(
    signature_matrix: np.ndarray,
    canonical_matrix: np.ndarray,
) -> tuple[tuple[str, Any, np.ndarray], ...]:
    combined = np.hstack((canonical_matrix, signature_matrix))
    return (
        ("field_signature_hgb", make_signature_hgb, signature_matrix),
        ("field_signature_rf", make_signature_rf, signature_matrix),
        ("canonical_plus_signature_hgb", curation.make_hgb, combined),
        ("canonical_plus_signature_rf", make_signature_rf, combined),
    )


def first3_clear_regression(
    fold_accuracies: Sequence[float],
) -> bool:
    if len(fold_accuracies) != 3:
        raise ValueError("first-three gate requires exactly three fold accuracies")
    return (
        float(np.mean(fold_accuracies)) < FIRST3_CLEAR_REGRESSION_MEAN
        and float(fold_accuracies[2]) <= REFERENCE_FIRST3[2]
    )


def run_outer_folds(
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    exact_labels: np.ndarray,
    fold_ids: np.ndarray,
    specs: Sequence[tuple[str, Any, np.ndarray]],
    canonical_targets: Sequence[str],
) -> tuple[dict[str, np.ndarray], list[np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    full_lookup = {
        auxiliary.row_key(record): index
        for index, record in enumerate(full_records)
    }
    folds = auxiliary.exact_group_folds(exact_records, fold_ids)
    probabilities = {
        name: np.full(len(exact_records), np.nan, dtype=np.float64)
        for name, _, _ in specs
    }
    fold_indices: list[np.ndarray] = []
    audits: list[dict[str, Any]] = []
    stopped = False
    stop_reason = ""
    for fold, (_, exact_test) in enumerate(folds, start=1):
        train_full, test_full, test_refnos = auxiliary.auxiliary_split(
            full_records, exact_records, exact_test, full_lookup
        )
        weights = base.inverse_target_multiplicity_weights(
            train_full, full_records, canonical_targets
        )
        fold_indices.append(exact_test)
        fold_candidate_metrics: dict[str, Any] = {}
        for name, factory, matrix in specs:
            model = factory()
            model.fit(
                matrix[train_full],
                full_labels[train_full],
                sample_weight=weights,
            )
            predicted = model.predict_proba(matrix[test_full])[:, 1]
            probabilities[name][exact_test] = predicted
            fold_candidate_metrics[name] = base.metric_record(
                exact_labels[exact_test], predicted
            )
        training_refnos = {
            full_records[int(index)]["REFNO"] for index in train_full
        }
        audits.append(
            {
                "fold": fold,
                "exact_test_rows": int(len(exact_test)),
                "exact_test_papers": len(test_refnos),
                "auxiliary_train_rows": int(len(train_full)),
                "auxiliary_train_papers": len(training_refnos),
                "train_test_refno_overlap": len(training_refnos & test_refnos),
                "all_test_paper_rows_excluded": all(
                    record["REFNO"] not in test_refnos
                    for record in (full_records[int(index)] for index in train_full)
                ),
                "candidate_metrics": fold_candidate_metrics,
            }
        )
        print(
            f"fold {fold}/5: "
            + ", ".join(
                f"{name}={fold_candidate_metrics[name]['accuracy']:.4f}"
                for name in probabilities
            ),
            flush=True,
        )
        if fold == 3:
            combined_names = (
                "canonical_plus_signature_hgb",
                "canonical_plus_signature_rf",
            )
            best_name = max(
                combined_names,
                key=lambda name: np.mean(
                    [
                        audits[index]["candidate_metrics"][name]["accuracy"]
                        for index in range(3)
                    ]
                ),
            )
            first3 = [
                audits[index]["candidate_metrics"][best_name]["accuracy"]
                for index in range(3)
            ]
            if first3_clear_regression(first3):
                stopped = True
                stop_reason = (
                    "best combined signature candidate clearly regressed over the "
                    "predeclared first-three-fold gate"
                )
                break
    smoke = {
        "evaluated_folds": len(fold_indices),
        "stopped_after_first_three": stopped,
        "stop_reason": stop_reason,
        "reference_first3_accuracy": list(REFERENCE_FIRST3),
        "clear_regression_mean_threshold": FIRST3_CLEAR_REGRESSION_MEAN,
    }
    return probabilities, fold_indices, audits, smoke


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument(
        "--screening-protocol", type=Path, default=auxiliary.SCREENING_PROTOCOL
    )
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    base.require_experiment_dependencies()

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    protocol_path = args.screening_protocol.resolve()
    fold_manifest_path = args.fold_manifest.resolve()
    targets_path = args.targets.resolve()
    page_cache_path = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy_path = args.taxonomy.resolve()
    positive_kb_path = args.positive_kb.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    input_audit = auxiliary.validate_inputs(
        exact_identity,
        full_identity,
        protocol_path,
        fold_manifest_path,
        targets_path,
        page_cache_path,
        taxonomy_path,
        positive_kb_path,
    )
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(fold_manifest_path, exact_records)

    kb, kb_audit = load_signature_kb(positive_kb_path)
    refnos = {record["REFNO"] for record in full_records}
    structured_pages, paper_text = own_study.load_structured_pages(
        page_cache_path, refnos
    )
    curation_pages = curation.load_pages(page_cache_path, refnos)
    signature_matrix, signature_names, extraction_audit = build_signature_features(
        full_records, structured_pages, kb
    )
    cache_path = output_dir / SIGNATURE_CACHE_NAME
    write_npz(cache_path, {"field_signatures": signature_matrix})
    cache_manifest_path = output_dir / SIGNATURE_MANIFEST_NAME
    cache_manifest = {
        "schema_version": "r3_exact300_field_signature_cache_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "rows": len(full_records),
        "features": int(signature_matrix.shape[1]),
        "matrix_sha256": sha256_array(signature_matrix),
        "alignment_sha256": auxiliary.sha256_records(full_records),
        "feature_names": signature_names,
        "extraction_audit": extraction_audit,
        "knowledge": kb_audit,
        "contains_labels": False,
        "contains_record_identifiers": False,
        "contains_source_quotes": False,
    }
    base.atomic_json(cache_manifest_path, cache_manifest)

    concepts, baseline_knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )
    base_features = base.build_features(full_records, paper_text, concepts)
    curation_features = curation.build_features(
        full_records, curation_pages, concepts, pdf_dir
    )
    canonical_matrix = np.concatenate(
        (
            np.asarray(base_features["engineered_rows"], dtype=np.float64),
            np.asarray(curation_features["all"], dtype=np.float64),
        ),
        axis=1,
    )
    specs = candidate_specs(signature_matrix.astype(np.float64), canonical_matrix)
    canonical_targets = [
        item["canonical_target"] for item in base_features["row_meta"]
    ]

    freeze_path = output_dir / FREEZE_NAME
    freeze = {
        "schema_version": "r3_exact300_field_signatures_feature_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": {
            "evaluation_rows": len(exact_records),
            "evaluation_papers": len({record["REFNO"] for record in exact_records}),
            "auxiliary_rows": len(full_records),
            "auxiliary_papers": len(refnos),
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": auxiliary.sha256_records(full_records),
        },
        "canonical_fold_manifest": {
            "path": str(fold_manifest_path),
            "sha256": base.file_sha256(fold_manifest_path),
            "fold_id_sha256": hashlib.sha256(
                bytes(int(value) for value in fold_ids)
            ).hexdigest(),
        },
        "candidate_names": [name for name, _, _ in specs],
        "first_three_stop_gate": {
            "reference_fold_accuracy": list(REFERENCE_FIRST3),
            "clear_regression_mean_below": FIRST3_CLEAR_REGRESSION_MEAN,
            "and_fold3_not_above": REFERENCE_FIRST3[2],
        },
        "feature_matrices": {
            "field_signatures": {
                "shape": list(signature_matrix.shape),
                "sha256": sha256_array(signature_matrix),
            },
            "canonical": {
                "shape": list(canonical_matrix.shape),
                "sha256": sha256_array(canonical_matrix),
            },
        },
        "signature_cache": {
            "path": str(cache_path),
            "sha256": base.file_sha256(cache_path),
            "manifest": str(cache_manifest_path),
            "manifest_sha256": base.file_sha256(cache_manifest_path),
        },
        "runtime_boundary": {
            "labels_read": False,
            "source_field_cells_read": False,
            "prior_predictions_or_evaluations_read": False,
            "identifiers_used_as_predictive_features": False,
            "record_answer_mapping_loaded": False,
            "source_quotes_cached": False,
        },
        "inputs": input_audit,
        "signature_knowledge": kb_audit,
        "baseline_knowledge": baseline_knowledge_audit,
    }
    base.atomic_json(freeze_path, freeze)
    freeze_sha256 = base.file_sha256(freeze_path)

    # R3 is dereferenced only after the signature cache and candidate freeze.
    labels_by_resultno = base.load_training_labels(
        targets_path, {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in full_records],
        dtype=np.int8,
    )
    exact_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )
    probabilities, fold_indices, fold_audit, smoke = run_outer_folds(
        full_records,
        exact_records,
        full_labels,
        exact_labels,
        fold_ids,
        specs,
        canonical_targets,
    )

    completed = len(fold_indices) == 5
    evaluated_indices = np.concatenate(fold_indices)
    metrics = {
        name: base.metric_record(exact_labels[evaluated_indices], values[evaluated_indices])
        for name, values in probabilities.items()
    }
    fold_metrics = {
        name: [
            {
                "fold": fold,
                **base.metric_record(exact_labels[index], values[index]),
            }
            for fold, index in enumerate(fold_indices, start=1)
        ]
        for name, values in probabilities.items()
    }
    ranking = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["accuracy"],
            metrics[name]["macro_f1"],
            metrics[name]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best = ranking[0]
    oof_path = output_dir / OOF_NAME
    if completed:
        write_npz(oof_path, probabilities)
    report = {
        "schema_version": "r3_exact300_field_signatures_grouped_oof_v1",
        "status": "complete" if completed else "stopped_after_first_three",
        "scope": {
            "evaluated_rows": int(len(evaluated_indices)),
            "evaluated_papers": len(
                {exact_records[int(index)]["REFNO"] for index in evaluated_indices}
            ),
            "evaluated_folds": len(fold_indices),
            "coverage": round(len(evaluated_indices) / EXPECTED_ROWS, 8),
            "all_300_rows_predicted_once": completed,
            "auxiliary_candidate_pool": full_coverage,
            "exact_membership": exact_coverage,
        },
        "guardrails": {
            "canonical_fold_manifest_sha256": base.file_sha256(fold_manifest_path),
            "every_test_refno_fully_excluded_from_training": all(
                item["all_test_paper_rows_excluded"]
                and item["train_test_refno_overlap"] == 0
                for item in fold_audit
            ),
            "signature_cache_before_label_access": True,
            "feature_freeze_before_label_access": True,
            "feature_freeze_sha256": freeze_sha256,
            "raw_source_fields_used_as_features": False,
            "prior_predictions_or_evaluations_used_as_features": False,
            "identifiers_used_as_predictive_features": False,
            "record_answer_mapping_loaded": False,
        },
        "first_three_smoke_gate": smoke,
        "outer_fold_audit": fold_audit,
        "ranking": ranking,
        "best_evaluated": {
            "strategy": best,
            **metrics[best],
            "folds": fold_metrics[best],
        },
        "models": {
            name: {"aggregate": metrics[name], "folds": fold_metrics[name]}
            for name in probabilities
        },
        "reference_comparison": {
            "reference_strategy": "canonical aux_combined_hgb",
            "reference_accuracy": REFERENCE_ACCURACY,
            "candidate_accuracy": metrics.get(
                "canonical_plus_signature_hgb", {}
            ).get("accuracy"),
            "full_coverage_comparison_valid": completed,
            "improved": bool(
                completed
                and metrics["canonical_plus_signature_hgb"]["accuracy"]
                > REFERENCE_ACCURACY
            ),
        },
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "observed_accuracy": metrics[best]["accuracy"] if completed else None,
            "passed": bool(completed and metrics[best]["accuracy"] >= TARGET_ACCURACY),
        },
        "artifacts": {
            "signature_cache": {
                "path": str(cache_path),
                "sha256": base.file_sha256(cache_path),
            },
            "signature_manifest": {
                "path": str(cache_manifest_path),
                "sha256": base.file_sha256(cache_manifest_path),
            },
            "feature_freeze": {"path": str(freeze_path), "sha256": freeze_sha256},
            "oof_probabilities": (
                {
                    "path": str(oof_path),
                    "sha256": base.file_sha256(oof_path),
                    "rows_per_array": EXPECTED_ROWS,
                    "contains_labels": False,
                    "contains_identifiers": False,
                }
                if completed
                else None
            ),
            "script": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
    }
    report_path = output_dir / REPORT_NAME
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "status": report["status"],
                "best_evaluated": report["best_evaluated"],
                "reference_comparison": report["reference_comparison"],
                "promotion_gate": report["promotion_gate"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
