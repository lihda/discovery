#!/usr/bin/env python3
"""Canonical exact300 R3 experiment with historical annotation-policy priors.

Bibliographic entities come only from the original-paper page cache.  In each
outer fold, every auxiliary row sharing a test REFNO is excluded.  Empirical
Bayes entity posteriors and the fixed fusion with the semantic HGB are selected
from four REFNO-isolated inner OOF folds; outer-test labels are never accepted
by the prediction function.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
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

import audit_r3_exact300_bibliographic_policy_coverage as coverage  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402


EXACT_IDENTITY = auxiliary.EXACT_IDENTITY
FULL_IDENTITY = auxiliary.FULL_IDENTITY
SCREENING_PROTOCOL = auxiliary.SCREENING_PROTOCOL
FOLD_MANIFEST = auxiliary.FOLD_MANIFEST
COVERAGE_PROTOCOL = ROOT / "tmp/r3_exact_300_resultno_v2/bibliographic_policy_v2/coverage_protocol.json"
COVERAGE_AUDIT = ROOT / "tmp/r3_exact_300_resultno_v2/bibliographic_policy_v2/coverage_audit.json"
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/bibliographic_policy_model_v1"

FEATURE_FREEZE_NAME = "feature_freeze.json"
REPORT_NAME = "report.json"
OOF_NAME = "oof_probabilities.npz"

COVERAGE_PROTOCOL_SHA256 = "c395951bcece8fc016a1648d9f86a843e0c1c1a68e1d41f699c90bd3899e35b6"
COVERAGE_AUDIT_SHA256 = "30a7a328234901a1a6126a5aa8a885a096ac1ab53f01cf8e861a9922bdbddc95"
CANONICAL_FOLD_MANIFEST_SHA256 = auxiliary.FOLD_MANIFEST_SHA256
EXPECTED_ROWS = auxiliary.EXPECTED_EXACT_ROWS
INNER_FOLDS = 4
INNER_SPLIT_SALT = "r3-bibliographic-policy-inner-v1"
SMOOTHING_STRENGTHS = (4.0, 12.0)
FUSION_WEIGHTS = (0.15, 0.30, 0.50)
TARGET_ACCURACY = 0.90
CANONICAL_REFERENCE_ACCURACY = 0.87333333

ENTITY_LIMITS = {
    "authors": 30,
    "doi_prefixes": 10,
    "citation_dois": 100,
    "citation_keys": 400,
    "source_dois": 20,
    "source_author_year_keys": 30,
}
FAMILY_RULES = {
    "author": {"minimum_support": 2, "weight": 2.0, "alpha_scale": 1.0},
    "journal_decade": {"minimum_support": 10, "weight": 1.0, "alpha_scale": 1.0},
    "publisher_decade": {"minimum_support": 10, "weight": 1.0, "alpha_scale": 1.0},
    "doi_prefix": {"minimum_support": 5, "weight": 0.75, "alpha_scale": 1.0},
    "shared_citation_doi": {"minimum_support": 2, "weight": 1.25, "alpha_scale": 1.0},
    "shared_citation_key": {"minimum_support": 3, "weight": 1.25, "alpha_scale": 1.0},
    "direct_outgoing_doi": {"minimum_support": 1, "weight": 2.0, "alpha_scale": 0.25},
    "direct_outgoing_key": {"minimum_support": 1, "weight": 1.5, "alpha_scale": 0.5},
    "direct_incoming_doi": {"minimum_support": 1, "weight": 2.0, "alpha_scale": 0.25},
    "direct_incoming_key": {"minimum_support": 1, "weight": 1.5, "alpha_scale": 0.5},
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    smoothing: float | None
    fusion_weight: float


@dataclass
class PolicyEncoder:
    prior: float
    tables: dict[str, dict[str, tuple[float, int]]]
    training_papers: int


def candidate_specs() -> tuple[CandidateSpec, ...]:
    return (
        CandidateSpec("semantic_only", None, 0.0),
        *(
            CandidateSpec(
                f"policy_a{int(alpha)}_fusion{int(weight * 100):02d}",
                alpha,
                weight,
            )
            for alpha in SMOOTHING_STRENGTHS
            for weight in FUSION_WEIGHTS
        ),
    )


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def profile_digest(
    profiles: Mapping[str, coverage.BibliographicProfile],
    ordered_refnos: Sequence[str],
) -> str:
    rows = []
    for refno in ordered_refnos:
        profile = profiles[refno]
        rows.append(
            {
                "refno": refno,
                "title": profile.title,
                "authors": sorted(profile.authors),
                "year": profile.year,
                "journal": profile.journal,
                "publisher": profile.publisher,
                "doi_prefixes": sorted(profile.doi_prefixes),
                "source_dois": sorted(profile.source_dois),
                "citation_dois": sorted(profile.citation_dois),
                "citation_keys": sorted(profile.citation_keys),
                "source_author_year_keys": sorted(profile.source_author_year_keys),
            }
        )
    return sha256_json(rows)


def balanced_group_fold_ids(
    records: Sequence[Mapping[str, str]],
    indices: np.ndarray,
) -> np.ndarray:
    """Assign REFNO groups to deterministic, row-balanced inner folds."""

    counts = Counter(records[int(index)]["REFNO"] for index in indices)
    ranked = sorted(
        counts,
        key=lambda refno: (
            -counts[refno],
            hashlib.sha256(f"{INNER_SPLIT_SALT}\0{refno}".encode()).digest(),
            refno,
        ),
    )
    loads = [0] * INNER_FOLDS
    assignment: dict[str, int] = {}
    for refno in ranked:
        fold = min(range(INNER_FOLDS), key=lambda value: (loads[value], value))
        assignment[refno] = fold
        loads[fold] += counts[refno]
    fold_ids = np.asarray(
        [assignment[records[int(index)]["REFNO"]] for index in indices],
        dtype=np.int8,
    )
    if set(int(value) for value in fold_ids) != set(range(INNER_FOLDS)):
        raise ValueError("bibliographic inner split does not contain four folds")
    observed: dict[str, set[int]] = defaultdict(set)
    for index, fold in zip(indices, fold_ids):
        observed[records[int(index)]["REFNO"]].add(int(fold))
    if any(len(values) != 1 for values in observed.values()):
        raise AssertionError("REFNO split across bibliographic inner folds")
    return fold_ids


def limited(values: Sequence[str], field: str) -> tuple[str, ...]:
    return tuple(sorted(set(values))[: ENTITY_LIMITS[field]])


def profile_entities(
    profile: coverage.BibliographicProfile,
) -> dict[str, tuple[str, ...]]:
    journal_decade = f"{profile.journal}:{coverage.decade(profile.year)}"
    publisher_decade = (
        (f"{profile.publisher}:{coverage.decade(profile.year)}",)
        if profile.publisher is not None
        else ()
    )
    return {
        "author": limited(profile.authors, "authors"),
        "journal_decade": (journal_decade,),
        "publisher_decade": publisher_decade,
        "doi_prefix": limited(profile.doi_prefixes, "doi_prefixes"),
        "shared_citation_doi": limited(profile.citation_dois, "citation_dois"),
        "shared_citation_key": limited(profile.citation_keys, "citation_keys"),
        "direct_outgoing_doi": limited(profile.citation_dois, "citation_dois"),
        "direct_outgoing_key": limited(profile.citation_keys, "citation_keys"),
        "direct_incoming_doi": limited(profile.source_dois, "source_dois"),
        "direct_incoming_key": limited(
            profile.source_author_year_keys, "source_author_year_keys"
        ),
        "source_doi": limited(profile.source_dois, "source_dois"),
        "source_key": limited(
            profile.source_author_year_keys, "source_author_year_keys"
        ),
    }


def paper_label_map(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
) -> dict[str, float]:
    values: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        values[records[int(index)]["REFNO"]].append(int(labels[int(index)]))
    return {refno: float(np.mean(items)) for refno, items in values.items()}


def fit_policy_encoder(
    train_refnos: Sequence[str],
    paper_labels: Mapping[str, float],
    profiles: Mapping[str, coverage.BibliographicProfile],
) -> PolicyEncoder:
    if set(train_refnos) != set(paper_labels):
        raise ValueError("policy encoder paper labels do not match training profiles")
    prior = float(np.mean([paper_labels[refno] for refno in train_refnos]))
    raw_tables: dict[str, dict[str, list[float]]] = {
        name: defaultdict(lambda: [0.0, 0.0])
        for name in (
            "author",
            "journal_decade",
            "publisher_decade",
            "doi_prefix",
            "shared_citation_doi",
            "shared_citation_key",
            "source_doi",
            "source_key",
        )
    }
    for refno in train_refnos:
        label = paper_labels[refno]
        entities = profile_entities(profiles[refno])
        for family in raw_tables:
            for entity in entities[family]:
                raw_tables[family][entity][0] += label
                raw_tables[family][entity][1] += 1.0
    tables = {
        family: {
            entity: (float(values[0]), int(values[1]))
            for entity, values in table.items()
        }
        for family, table in raw_tables.items()
    }
    return PolicyEncoder(prior=prior, tables=tables, training_papers=len(train_refnos))


def family_table_name(family: str) -> str:
    return {
        "direct_outgoing_doi": "source_doi",
        "direct_outgoing_key": "source_key",
        "direct_incoming_doi": "shared_citation_doi",
        "direct_incoming_key": "shared_citation_key",
    }.get(family, family)


def family_posterior(
    table: Mapping[str, tuple[float, int]],
    entities: Sequence[str],
    *,
    prior: float,
    alpha: float,
    minimum_support: int,
) -> tuple[float, float, int, int]:
    values: list[tuple[float, float, int]] = []
    for entity in entities:
        total, count = table.get(entity, (0.0, 0))
        if count < minimum_support:
            continue
        posterior = (total + alpha * prior) / (count + alpha)
        reliability = count / (count + alpha)
        values.append((posterior, reliability, count))
    if not values:
        return prior, 0.0, 0, 0
    weights = np.asarray([math.sqrt(count) for _, _, count in values])
    probability = float(
        np.average(np.asarray([value for value, _, _ in values]), weights=weights)
    )
    reliability = 1.0 - float(
        np.prod([1.0 - min(0.95, item[1]) for item in values])
    )
    return probability, reliability, len(values), max(count for _, _, count in values)


def transform_policy_profile(
    encoder: PolicyEncoder,
    profile: coverage.BibliographicProfile,
    smoothing: float,
) -> tuple[float, float, dict[str, Any]]:
    entities = profile_entities(profile)
    evidence: list[tuple[float, float]] = []
    family_audit: dict[str, Any] = {}
    for family, rule in FAMILY_RULES.items():
        table = encoder.tables[family_table_name(family)]
        alpha = smoothing * float(rule["alpha_scale"])
        probability, reliability, matches, maximum_support = family_posterior(
            table,
            entities[family],
            prior=encoder.prior,
            alpha=alpha,
            minimum_support=int(rule["minimum_support"]),
        )
        weighted_reliability = float(rule["weight"]) * reliability
        if matches:
            evidence.append((probability, weighted_reliability))
        family_audit[family] = {
            "matched_entities": matches,
            "maximum_training_support": maximum_support,
            "reliability": round(reliability, 8),
        }
    evidence_weight = sum(weight for _, weight in evidence)
    prior_weight = 0.5
    probability = (
        encoder.prior * prior_weight
        + sum(value * weight for value, weight in evidence)
    ) / (prior_weight + evidence_weight)
    confidence = 1.0 - math.exp(-evidence_weight / 3.0)
    return (
        float(np.clip(probability, 0.0, 1.0)),
        float(np.clip(confidence, 0.0, 1.0)),
        {
            "supported_families": sum(
                record["matched_entities"] > 0 for record in family_audit.values()
            ),
            "evidence_weight": round(evidence_weight, 8),
            "families": family_audit,
        },
    )


def transform_policy_rows(
    encoder: PolicyEncoder,
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    profiles: Mapping[str, coverage.BibliographicProfile],
    smoothing: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    refnos = sorted({records[int(index)]["REFNO"] for index in indices})
    by_refno: dict[str, tuple[float, float, dict[str, Any]]] = {
        refno: transform_policy_profile(encoder, profiles[refno], smoothing)
        for refno in refnos
    }
    probability = np.asarray(
        [by_refno[records[int(index)]["REFNO"]][0] for index in indices],
        dtype=np.float64,
    )
    confidence = np.asarray(
        [by_refno[records[int(index)]["REFNO"]][1] for index in indices],
        dtype=np.float64,
    )
    audits = [value[2] for value in by_refno.values()]
    return probability, confidence, {
        "papers": len(refnos),
        "rows": int(len(indices)),
        "mean_confidence": round(float(confidence.mean()), 8),
        "zero_confidence_rows": int((confidence == 0).sum()),
        "mean_supported_families_per_paper": round(
            float(np.mean([item["supported_families"] for item in audits])), 8
        ),
    }


def fuse_probability(
    semantic: np.ndarray,
    policy: np.ndarray,
    confidence: np.ndarray,
    weight: float,
) -> np.ndarray:
    return np.clip(
        semantic + float(weight) * confidence * (policy - semantic), 0.0, 1.0
    )


def specificity(metric: Mapping[str, Any]) -> float:
    confusion = metric["confusion"]
    denominator = int(confusion["tn"]) + int(confusion["fp"])
    return float(confusion["tn"]) / denominator if denominator else 0.0


def select_candidate(
    labels: np.ndarray,
    semantic: np.ndarray,
    policy_by_alpha: Mapping[float, np.ndarray],
    confidence_by_alpha: Mapping[float, np.ndarray],
) -> tuple[CandidateSpec, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    specs = candidate_specs()
    for order, spec in enumerate(specs):
        probability = (
            semantic
            if spec.smoothing is None
            else fuse_probability(
                semantic,
                policy_by_alpha[spec.smoothing],
                confidence_by_alpha[spec.smoothing],
                spec.fusion_weight,
            )
        )
        metric = base.metric_record(labels, probability, threshold=0.5)
        records.append(
            {
                "candidate": spec.name,
                "candidate_order": order,
                "smoothing": spec.smoothing,
                "fusion_weight": spec.fusion_weight,
                "metric": metric,
            }
        )
    selected = max(
        records,
        key=lambda item: (
            item["metric"]["accuracy"],
            item["metric"]["macro_f1"],
            item["metric"]["balanced_accuracy"],
            specificity(item["metric"]),
            -item["fusion_weight"],
            -item["candidate_order"],
        ),
    )
    return next(spec for spec in specs if spec.name == selected["candidate"]), records


def fit_semantic_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    records: Sequence[dict[str, str]],
    canonical_targets: Sequence[str],
) -> Any:
    weights = base.inverse_target_multiplicity_weights(
        indices, records, canonical_targets
    )
    model = curation.make_hgb()
    model.fit(matrix[indices], labels[indices], sample_weight=weights)
    return model


def run_fold(
    fold: int,
    *,
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    fold_ids: np.ndarray,
    semantic_matrix: np.ndarray,
    canonical_targets: Sequence[str],
    profiles: Mapping[str, coverage.BibliographicProfile],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Produce one outer fold without accepting its exact labels."""

    exact_test = np.flatnonzero(fold_ids == fold).astype(np.int64)
    full_lookup = {
        auxiliary.row_key(record): index for index, record in enumerate(full_records)
    }
    outer_train, test_full, test_refnos = auxiliary.auxiliary_split(
        full_records, exact_records, exact_test, full_lookup
    )
    inner_ids = balanced_group_fold_ids(full_records, outer_train)
    inner_semantic = np.full(len(outer_train), np.nan, dtype=np.float64)
    inner_policy = {
        alpha: np.full(len(outer_train), np.nan, dtype=np.float64)
        for alpha in SMOOTHING_STRENGTHS
    }
    inner_confidence = {
        alpha: np.full(len(outer_train), np.nan, dtype=np.float64)
        for alpha in SMOOTHING_STRENGTHS
    }
    inner_audits: list[dict[str, Any]] = []
    for inner_fold in range(INNER_FOLDS):
        fit_position = np.flatnonzero(inner_ids != inner_fold)
        validation_position = np.flatnonzero(inner_ids == inner_fold)
        fit_indices = outer_train[fit_position]
        validation_indices = outer_train[validation_position]
        fit_refnos = {full_records[int(index)]["REFNO"] for index in fit_indices}
        validation_refnos = {
            full_records[int(index)]["REFNO"] for index in validation_indices
        }
        if (
            fit_refnos & validation_refnos
            or fit_refnos & test_refnos
            or validation_refnos & test_refnos
        ):
            raise AssertionError("REFNO leaked into bibliographic inner selection")
        semantic_model = fit_semantic_model(
            semantic_matrix,
            full_labels,
            fit_indices,
            full_records,
            canonical_targets,
        )
        inner_semantic[validation_position] = semantic_model.predict_proba(
            semantic_matrix[validation_indices]
        )[:, 1]
        labels_by_paper = paper_label_map(
            fit_indices, full_records, full_labels
        )
        encoder = fit_policy_encoder(
            sorted(fit_refnos), labels_by_paper, profiles
        )
        alpha_audits: dict[str, Any] = {}
        for alpha in SMOOTHING_STRENGTHS:
            probability, confidence, transform_audit = transform_policy_rows(
                encoder,
                validation_indices,
                full_records,
                profiles,
                alpha,
            )
            inner_policy[alpha][validation_position] = probability
            inner_confidence[alpha][validation_position] = confidence
            alpha_audits[f"alpha_{int(alpha)}"] = transform_audit
        inner_audits.append(
            {
                "inner_fold": inner_fold + 1,
                "fit_rows": int(len(fit_indices)),
                "fit_papers": len(fit_refnos),
                "validation_rows": int(len(validation_indices)),
                "validation_papers": len(validation_refnos),
                "fit_validation_refno_overlap": len(
                    fit_refnos & validation_refnos
                ),
                "outer_test_refno_overlap": len(
                    (fit_refnos | validation_refnos) & test_refnos
                ),
                "policy_transform": alpha_audits,
            }
        )
    if not np.isfinite(inner_semantic).all() or any(
        not np.isfinite(values).all()
        for mapping in (inner_policy, inner_confidence)
        for values in mapping.values()
    ):
        raise AssertionError("bibliographic inner OOF coverage is incomplete")
    selected, candidate_audit = select_candidate(
        full_labels[outer_train], inner_semantic, inner_policy, inner_confidence
    )

    semantic_model = fit_semantic_model(
        semantic_matrix,
        full_labels,
        outer_train,
        full_records,
        canonical_targets,
    )
    semantic_probability = np.asarray(
        semantic_model.predict_proba(semantic_matrix[test_full])[:, 1],
        dtype=np.float64,
    )
    smoothing = selected.smoothing or SMOOTHING_STRENGTHS[-1]
    train_refnos = {full_records[int(index)]["REFNO"] for index in outer_train}
    encoder = fit_policy_encoder(
        sorted(train_refnos),
        paper_label_map(outer_train, full_records, full_labels),
        profiles,
    )
    policy_probability, policy_confidence, policy_audit = transform_policy_rows(
        encoder, test_full, full_records, profiles, smoothing
    )
    fused_probability = (
        semantic_probability
        if selected.smoothing is None
        else fuse_probability(
            semantic_probability,
            policy_probability,
            policy_confidence,
            selected.fusion_weight,
        )
    )
    excluded_rows = sum(record["REFNO"] in test_refnos for record in full_records)
    audit = {
        "fold": fold,
        "exact_test_rows": int(len(exact_test)),
        "exact_test_papers": len(test_refnos),
        "outer_train_rows": int(len(outer_train)),
        "outer_train_papers": len(train_refnos),
        "full_rows_excluded_for_test_refnos": excluded_rows,
        "outer_train_test_refno_overlap": len(train_refnos & test_refnos),
        "outer_train_exact_test_row_overlap": len(
            set(int(index) for index in outer_train)
            & set(int(index) for index in test_full)
        ),
        "excluded_test_refno_rows_remaining_in_outer_train": sum(
            full_records[int(index)]["REFNO"] in test_refnos
            for index in outer_train
        ),
        "inner_split": {
            "method": "label-free deterministic row-balanced REFNO assignment",
            "salt": INNER_SPLIT_SALT,
            "folds": INNER_FOLDS,
            "fold_id_sha256": hashlib.sha256(bytes(inner_ids)).hexdigest(),
        },
        "inner_folds": inner_audits,
        "selection_source": "four-fold REFNO-isolated inner OOF only",
        "selection_rule": (
            "accuracy, macro-F1, balanced accuracy, specificity, lower policy "
            "weight, declaration order"
        ),
        "selected_candidate": asdict(selected),
        "candidate_metrics": candidate_audit,
        "outer_policy_transform": policy_audit,
        "outer_test_labels_received_by_run_fold": False,
    }
    return exact_test, {
        "bibliographic_policy_fusion": fused_probability,
        "semantic_hgb": semantic_probability,
        "policy_probability": policy_probability,
        "policy_confidence": policy_confidence,
    }, audit


def write_oof(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != {
        "bibliographic_policy_fusion",
        "semantic_hgb",
        "policy_probability",
        "policy_confidence",
    }:
        raise ValueError("unexpected bibliographic OOF array names")
    if any(value.shape != (EXPECTED_ROWS,) for value in arrays.values()):
        raise ValueError("bibliographic OOF arrays must cover exact300")
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("bibliographic OOF arrays contain non-finite values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(value, dtype=np.float32)
                for name, value in sorted(arrays.items())
            },
        )
    temporary.replace(path)


def candidate_protocol() -> dict[str, Any]:
    return {
        "candidates": [asdict(spec) for spec in candidate_specs()],
        "inner_split": {
            "folds": INNER_FOLDS,
            "unit": "REFNO",
            "label_free_assignment": True,
            "salt": INNER_SPLIT_SALT,
        },
        "empirical_bayes": {
            "smoothing_strengths": list(SMOOTHING_STRENGTHS),
            "family_rules": FAMILY_RULES,
            "entity_limits": ENTITY_LIMITS,
            "paper_label_unit": "mean R3 within training paper",
            "self_paper_labels_excluded_by_inner_cross_fitting": True,
        },
        "fusion": {
            "rule": "semantic + weight * policy_confidence * (policy - semantic)",
            "weights": list(FUSION_WEIGHTS),
            "threshold": 0.5,
            "outer_label_threshold_selection": False,
        },
    }


def validate_coverage_gate() -> dict[str, Any]:
    expected = (
        (COVERAGE_PROTOCOL, COVERAGE_PROTOCOL_SHA256),
        (COVERAGE_AUDIT, COVERAGE_AUDIT_SHA256),
    )
    for path, digest in expected:
        if not path.is_file() or base.file_sha256(path) != digest:
            raise ValueError(f"bibliographic coverage artifact hash mismatch: {path}")
    report = json.loads(COVERAGE_AUDIT.read_text(encoding="utf-8"))
    gate = report.get("continuation_gate", {})
    if (
        report.get("schema_version")
        != "r3_exact300_bibliographic_policy_coverage_audit_v2"
        or gate.get("passed") is not True
        or gate.get("labels_opened_for_this_audit") is not False
        or report.get("guardrails", {}).get("labels_read") is not False
        or report.get("guardrails", {}).get(
            "prior_predictions_or_evaluations_read"
        )
        is not False
    ):
        raise ValueError("bibliographic coverage gate does not authorize modeling")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=FULL_IDENTITY)
    parser.add_argument("--screening-protocol", type=Path, default=SCREENING_PROTOCOL)
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    screening_protocol = args.screening_protocol.resolve()
    fold_manifest = args.fold_manifest.resolve()
    targets = args.targets.resolve()
    page_cache = args.page_cache.resolve()
    pdf_dir = args.pdf_dir.resolve()
    taxonomy = args.taxonomy.resolve()
    positive_kb = args.positive_kb.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_report = validate_coverage_gate()
    input_audit = auxiliary.validate_inputs(
        exact_identity,
        full_identity,
        screening_protocol,
        fold_manifest,
        targets,
        page_cache,
        taxonomy,
        positive_kb,
    )
    if base.file_sha256(fold_manifest) != CANONICAL_FOLD_MANIFEST_SHA256:
        raise ValueError("bibliographic model requires canonical exact300 folds")
    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(fold_manifest, exact_records)
    refnos = {record["REFNO"] for record in full_records}
    profiles = coverage.load_profiles(page_cache, refnos)

    pages = curation.load_pages(page_cache, refnos)
    paper_text = base.load_page_text(page_cache, refnos)
    concepts, knowledge_audit = base.load_knowledge_patterns(taxonomy, positive_kb)
    base_features = base.build_features(full_records, paper_text, concepts)
    curation_features = curation.build_features(full_records, pages, concepts, pdf_dir)
    semantic_matrix = np.concatenate(
        (
            np.asarray(base_features["engineered_rows"], dtype=np.float64),
            np.asarray(curation_features["all"], dtype=np.float64),
        ),
        axis=1,
    )
    canonical_targets = [item["canonical_target"] for item in base_features["row_meta"]]
    protocol = candidate_protocol()
    freeze = {
        "schema_version": "r3_exact300_bibliographic_policy_feature_freeze_v1",
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
            "path": str(fold_manifest),
            "sha256": base.file_sha256(fold_manifest),
        },
        "coverage_gate": {
            "protocol_path": str(COVERAGE_PROTOCOL),
            "protocol_sha256": COVERAGE_PROTOCOL_SHA256,
            "audit_path": str(COVERAGE_AUDIT),
            "audit_sha256": COVERAGE_AUDIT_SHA256,
            "decision": coverage_report["continuation_gate"]["decision"],
        },
        "candidate_protocol": protocol,
        "candidate_protocol_sha256": sha256_json(protocol),
        "features": {
            "semantic_matrix": {
                "shape": list(semantic_matrix.shape),
                "sha256": auxiliary.sha256_array(semantic_matrix),
            },
            "bibliographic_profiles": {
                "papers": len(profiles),
                "sha256": profile_digest(profiles, sorted(refnos)),
            },
        },
        "allowed_predictive_sources": [
            "ROCKNAME from frozen identity for semantic target attribution",
            "original-paper text/layout from frozen page cache",
            "local PDF byte size",
            "frozen aggregate taxonomy and positive expert vocabulary",
            "bibliographic entities extracted from original-paper text",
            "R3 labels from outer-allowed auxiliary training papers only",
        ],
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_source_cells_read": False,
            "prior_predictions_or_evaluations_read": False,
            "raw_source_table_fields_read": False,
            "refno_or_resultno_used_as_predictive_features": False,
            "identifiers_used_for_alignment_and_group_isolation_only": True,
        },
        "inputs": {
            **input_audit,
            "coverage_protocol": {
                "path": str(COVERAGE_PROTOCOL),
                "sha256": COVERAGE_PROTOCOL_SHA256,
            },
            "coverage_audit": {
                "path": str(COVERAGE_AUDIT),
                "sha256": COVERAGE_AUDIT_SHA256,
            },
        },
        "knowledge": knowledge_audit,
        "implementation": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    freeze_path = output_dir / FEATURE_FREEZE_NAME
    base.atomic_json(freeze_path, freeze)
    freeze_sha = base.file_sha256(freeze_path)

    # Labels are opened only after profiles, semantic matrix, candidates, and hashes freeze.
    labels_by_resultno = base.load_training_labels(
        targets, {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in full_records],
        dtype=np.int8,
    )
    exact_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )

    arrays = {
        name: np.full(EXPECTED_ROWS, np.nan, dtype=np.float64)
        for name in (
            "bibliographic_policy_fusion",
            "semantic_hgb",
            "policy_probability",
            "policy_confidence",
        )
    }
    fold_audits: list[dict[str, Any]] = []
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(1, auxiliary.N_SPLITS + 1):
        exact_test, predictions, audit = run_fold(
            fold,
            full_records=full_records,
            exact_records=exact_records,
            full_labels=full_labels,
            fold_ids=fold_ids,
            semantic_matrix=semantic_matrix,
            canonical_targets=canonical_targets,
            profiles=profiles,
        )
        for name, values in predictions.items():
            arrays[name][exact_test] = values
        fold_audits.append(audit)
        fold_metrics.append(
            {
                "fold": fold,
                "bibliographic_policy_fusion": base.metric_record(
                    exact_labels[exact_test],
                    predictions["bibliographic_policy_fusion"],
                ),
                "semantic_hgb": base.metric_record(
                    exact_labels[exact_test], predictions["semantic_hgb"]
                ),
                "policy_only_diagnostic": base.metric_record(
                    exact_labels[exact_test], predictions["policy_probability"]
                ),
            }
        )
        print(
            f"fold {fold}/5 selected={audit['selected_candidate']['name']} "
            f"fusion_acc={fold_metrics[-1]['bibliographic_policy_fusion']['accuracy']:.4f} "
            f"semantic_acc={fold_metrics[-1]['semantic_hgb']['accuracy']:.4f}",
            flush=True,
        )
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise AssertionError("bibliographic exact300 OOF coverage is incomplete")
    oof_path = output_dir / OOF_NAME
    write_oof(oof_path, arrays)
    fusion_metric = base.metric_record(
        exact_labels, arrays["bibliographic_policy_fusion"]
    )
    semantic_metric = base.metric_record(exact_labels, arrays["semantic_hgb"])
    policy_metric = base.metric_record(exact_labels, arrays["policy_probability"])
    zero_fields = (
        "outer_train_test_refno_overlap",
        "outer_train_exact_test_row_overlap",
        "excluded_test_refno_rows_remaining_in_outer_train",
    )
    zero_overlap = all(
        audit[field] == 0 for audit in fold_audits for field in zero_fields
    ) and all(
        inner["fit_validation_refno_overlap"] == 0
        and inner["outer_test_refno_overlap"] == 0
        for audit in fold_audits
        for inner in audit["inner_folds"]
    )
    report = {
        "schema_version": "r3_exact300_bibliographic_policy_grouped_oof_v1",
        "status": "complete",
        "scope": {
            "evaluated_rows": len(exact_records),
            "evaluated_papers": len({record["REFNO"] for record in exact_records}),
            "evaluated_folds": auxiliary.N_SPLITS,
            "full_coverage": True,
            "full_2172_rows_evaluated": False,
            "full_pool_coverage": full_coverage,
            "exact_coverage": exact_coverage,
        },
        "guardrails": {
            "canonical_fold_manifest_sha256": base.file_sha256(fold_manifest),
            "feature_freeze_before_label_access": True,
            "feature_freeze_sha256": freeze_sha,
            "every_outer_test_refno_fully_excluded": True,
            "inner_empirical_bayes_self_paper_excluded": True,
            "fusion_selected_from_inner_oof_only": True,
            "fixed_outer_threshold": 0.5,
            "posthoc_outer_label_threshold_used": False,
            "outer_test_labels_used_by_run_fold": False,
            "all_outer_and_inner_overlap_audits_zero": zero_overlap,
            "pmagresult_or_rockmag_source_cells_used": False,
            "prior_predictions_or_evaluations_used": False,
            "raw_source_table_fields_used": False,
            "identifiers_used_as_predictive_features": False,
        },
        "data": {
            "full_label_counts": dict(
                sorted(Counter(str(int(value)) for value in full_labels).items())
            ),
            "exact_label_counts": dict(
                sorted(Counter(str(int(value)) for value in exact_labels).items())
            ),
            "semantic_feature_shape": list(semantic_matrix.shape),
            "profile_count": len(profiles),
        },
        "predeclared_candidate_protocol": protocol,
        "outer_fold_audit": fold_audits,
        "fold_metrics": fold_metrics,
        "primary_full_coverage_result": {
            "strategy": "inner-selected bibliographic policy fusion",
            **fusion_metric,
            "folds": [
                {"fold": item["fold"], **item["bibliographic_policy_fusion"]}
                for item in fold_metrics
            ],
        },
        "semantic_hgb_reference": {
            **semantic_metric,
            "folds": [
                {"fold": item["fold"], **item["semantic_hgb"]}
                for item in fold_metrics
            ],
        },
        "policy_only_diagnostic": policy_metric,
        "promotion_gate": {
            "target_accuracy": TARGET_ACCURACY,
            "canonical_reference_accuracy": CANONICAL_REFERENCE_ACCURACY,
            "observed_accuracy": fusion_metric["accuracy"],
            "absolute_delta_from_canonical": round(
                fusion_metric["accuracy"] - CANONICAL_REFERENCE_ACCURACY, 8
            ),
            "absolute_delta_from_recomputed_semantic": round(
                fusion_metric["accuracy"] - semantic_metric["accuracy"], 8
            ),
            "passed": fusion_metric["accuracy"] >= TARGET_ACCURACY,
            "decision": (
                "EXACT300_TARGET_REACHED"
                if fusion_metric["accuracy"] >= TARGET_ACCURACY
                else "EXACT300_TARGET_NOT_REACHED"
            ),
        },
        "artifacts": {
            "feature_freeze": {"path": str(freeze_path), "sha256": freeze_sha},
            "oof_probabilities": {
                "path": str(oof_path),
                "sha256": base.file_sha256(oof_path),
                "rows_per_array": EXPECTED_ROWS,
                "contains_labels": False,
                "contains_identifiers": False,
                "arrays": sorted(arrays),
            },
            "implementation": {
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
                "primary": report["primary_full_coverage_result"],
                "semantic_reference": report["semantic_hgb_reference"],
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
