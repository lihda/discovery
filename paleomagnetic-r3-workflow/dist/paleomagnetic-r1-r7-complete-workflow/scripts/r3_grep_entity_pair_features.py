#!/usr/bin/env python3
"""Build GREP-inspired target/evidence interaction features for R3.

The implementation adapts two ideas from Zhang, Yan, and Cheng (ACL 2025):
document-level global relation priors and interaction between entity-pair
representations. Here, a relation pair is a (target, evidence occurrence)
pair. Features are built from the identity manifest and original-paper page
text only; labels and row-level source metadata are not accepted.
"""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from .paper_text_retrieval import PaperTextRetriever, canonical_target_identity
except ImportError:
    from paper_text_retrieval import PaperTextRetriever, canonical_target_identity


CATEGORIES = ("method", "diagnostic", "carrier", "bridge")
ANCHOR_STATUSES = ("exact", "token", "generic_only", "absent")
ANCHOR_STRENGTHS = ("strong", "moderate", "weak", "none")
SOFTMAX_TEMPERATURE = 0.16
MAX_HITS_PER_CATEGORY = 50


def clean(value: Any) -> str:
    return "" if value is None else str(value).strip()


def load_page_cache_rows(
    path: Path,
    wanted_refnos: set[str],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            refno = clean(row.get("REFNO"))
            if refno not in wanted_refnos:
                continue
            if refno in rows:
                raise ValueError(f"duplicate REFNO {refno} in {path}:{line_number}")
            rows[refno] = row
    missing = sorted(wanted_refnos - rows.keys())
    if missing:
        raise ValueError(f"page cache is missing REFNO values: {missing[:20]}")
    return rows


def _log1p(value: float) -> float:
    return math.log1p(max(0.0, float(value)))


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _anchor_status(anchor: Mapping[str, Any]) -> str:
    status = clean(anchor.get("status")).casefold()
    return status if status in ANCHOR_STATUSES else "absent"


def _anchor_strength(anchor: Mapping[str, Any]) -> str:
    strength = clean(anchor.get("strength")).casefold()
    return strength if strength in ANCHOR_STRENGTHS else "none"


def _claim_key(category: str, hit: Mapping[str, Any]) -> str:
    return "|".join(
        (
            category,
            clean(hit.get("page")),
            clean(hit.get("scope_id")),
            clean(hit.get("start")),
            clean(hit.get("end")),
            clean(hit.get("subtype")),
            clean(hit.get("claim_text_sha256")),
        )
    )


def _hit_score(hit: Mapping[str, Any]) -> float:
    relevance = float(hit.get("target_relevance") or 0.0)
    binding = float(hit.get("binding") or 0.0)
    competitors = len(hit.get("scope_competing_targets") or [])
    score = (
        0.62 * relevance
        + 0.28 * binding
        + 0.18 * float(hit.get("local_target") is True)
        + 0.10 * float(hit.get("scope_target_exclusive") is True)
        + 0.08 * float(hit.get("target_anchor") == "target_same_sentence")
        + 0.04 * float(hit.get("scope_bounded") is True)
        - 0.025 * min(competitors, 6)
    )
    return max(0.0, score)


def _category_features(
    category: str,
    hits: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    valid = [hit for hit in hits if not hit.get("reference_noise")]
    relevance = [float(hit.get("target_relevance") or 0.0) for hit in valid]
    binding = [float(hit.get("binding") or 0.0) for hit in valid]
    distances = [
        float(hit.get("target_distance_chars"))
        for hit in valid
        if hit.get("target_distance_chars") is not None
    ]
    competitors = [len(hit.get("scope_competing_targets") or []) for hit in valid]
    qualified = [
        hit
        for hit in valid
        if category != "method" or clean(hit.get("subtype")) != "demagnetization"
    ]
    prefix = f"pair_{category}_"
    return {
        prefix + "present": float(bool(valid)),
        prefix + "count_log": _log1p(len(valid)),
        prefix + "qualified_count_log": _log1p(len(qualified)),
        prefix + "local_count_log": _log1p(
            sum(hit.get("local_target") is True for hit in valid)
        ),
        prefix + "same_sentence_count_log": _log1p(
            sum(hit.get("target_anchor") == "target_same_sentence" for hit in valid)
        ),
        prefix + "exclusive_count_log": _log1p(
            sum(hit.get("scope_target_exclusive") is True for hit in valid)
        ),
        prefix + "bounded_count_log": _log1p(
            sum(hit.get("scope_bounded") is True for hit in valid)
        ),
        prefix + "max_relevance": max(relevance, default=0.0),
        prefix + "mean_relevance": _mean(relevance),
        prefix + "max_binding": max(binding, default=0.0),
        prefix + "mean_binding": _mean(binding),
        prefix + "minimum_distance_score": (
            1.0 / (1.0 + min(distances) / 100.0) if distances else 0.0
        ),
        prefix + "unique_pages_log": _log1p(
            len({clean(hit.get("page")) for hit in valid if clean(hit.get("page"))})
        ),
        prefix + "unique_scopes_log": _log1p(
            len(
                {
                    clean(hit.get("scope_id"))
                    for hit in valid
                    if clean(hit.get("scope_id"))
                }
            )
        ),
        prefix + "subtype_count_log": _log1p(
            len(
                {
                    clean(hit.get("subtype"))
                    for hit in valid
                    if clean(hit.get("subtype"))
                }
            )
        ),
        prefix + "mean_competing_targets_log": _log1p(_mean(competitors)),
        prefix + "competition_free_ratio": (
            _mean([float(value == 0) for value in competitors])
            if competitors
            else 0.0
        ),
    }


def _raw_target_features(retrieval: Mapping[str, Any]) -> dict[str, float]:
    anchor = retrieval.get("target_anchor") or {}
    chain = retrieval.get("best_chain") or {}
    output: dict[str, float] = {
        "source_available": float(bool(retrieval.get("source_available"))),
        "anchor_found": float(bool(anchor.get("found"))),
        "anchor_specificity": float(anchor.get("specificity") or 0.0),
        "anchor_common_target": float(bool(anchor.get("common_target"))),
        "anchor_page_count_log": _log1p(len(anchor.get("pages") or [])),
        "anchor_exact_page_count_log": _log1p(len(anchor.get("exact_pages") or [])),
        "anchor_token_page_count_log": _log1p(len(anchor.get("token_pages") or [])),
        "scope_competitor_policy_enabled": float(
            (retrieval.get("scope_binding") or {}).get("status")
            == "enabled_with_competing_target_exclusion"
        ),
    }
    status = _anchor_status(anchor)
    strength = _anchor_strength(anchor)
    for value in ANCHOR_STATUSES:
        output[f"anchor_status_{value}"] = float(status == value)
    for value in ANCHOR_STRENGTHS:
        output[f"anchor_strength_{value}"] = float(strength == value)

    category_hits = retrieval.get("categories") or {}
    for category in CATEGORIES:
        output.update(_category_features(category, category_hits.get(category) or []))

    selected_hits = list((chain.get("hits") or {}).values())
    chain_relevance = [
        float(hit.get("target_relevance") or 0.0)
        for hit in selected_hits
        if isinstance(hit, Mapping)
    ]
    page_span = chain.get("page_span")
    output.update(
        {
            "chain_complete": float(bool(chain.get("complete"))),
            "chain_connected": float(bool(chain.get("connected"))),
            "chain_same_page": float(chain.get("connection") == "same_page"),
            "chain_adjacent_pages": float(
                chain.get("connection") == "adjacent_pages"
            ),
            "chain_score": float(chain.get("score") or 0.0),
            "chain_page_span_score": (
                1.0 / (1.0 + float(page_span)) if page_span is not None else 0.0
            ),
            "chain_minimum_relevance": min(chain_relevance, default=0.0),
            "chain_mean_relevance": _mean(chain_relevance),
            "chain_all_hits_local": float(
                bool(selected_hits)
                and all(hit.get("local_target") is True for hit in selected_hits)
            ),
        }
    )
    return output


def _claim_maps(
    retrievals: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Mapping[str, Any]]]]:
    output: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {
        category: defaultdict(dict) for category in CATEGORIES
    }
    for target, retrieval in retrievals.items():
        for category in CATEGORIES:
            for hit in (retrieval.get("categories") or {}).get(category) or []:
                if hit.get("reference_noise"):
                    continue
                key = _claim_key(category, hit)
                current = output[category][key].get(target)
                if current is None or _hit_score(hit) > _hit_score(current):
                    output[category][key][target] = hit
    return output


def _graph_features(
    targets: Sequence[str],
    claims: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {target: {} for target in targets}
    for category in CATEGORIES:
        category_claims = claims.get(category) or {}
        accumulators = {
            target: {
                "candidate": 0.0,
                "mass": 0.0,
                "winner": 0.0,
                "positive_margin": 0.0,
                "signed_margin": 0.0,
                "high_mass": 0.0,
                "unique": 0.0,
                "degree": 0.0,
            }
            for target in targets
        }
        for target_hits in category_claims.values():
            scores = {
                target: _hit_score(target_hits[target]) if target in target_hits else 0.0
                for target in targets
            }
            maximum = max(scores.values(), default=0.0)
            winners = [
                target
                for target, value in scores.items()
                if abs(value - maximum) <= 1e-12
            ]
            exponentials = {
                target: math.exp(min(40.0, value / SOFTMAX_TEMPERATURE))
                for target, value in scores.items()
            }
            denominator = sum(exponentials.values()) or 1.0
            degree = len(target_hits)
            for target in targets:
                others = [value for name, value in scores.items() if name != target]
                sibling_maximum = max(others, default=0.0)
                mass = exponentials[target] / denominator
                state = accumulators[target]
                state["candidate"] += float(target in target_hits)
                state["mass"] += mass
                state["winner"] += (
                    1.0 / len(winners) if target in winners and winners else 0.0
                )
                state["positive_margin"] += max(0.0, scores[target] - sibling_maximum)
                state["signed_margin"] += scores[target] - sibling_maximum
                state["high_mass"] += float(mass >= 0.5)
                state["unique"] += float(target in target_hits and degree == 1)
                if target in target_hits:
                    state["degree"] += degree

        global_count = len(category_claims)
        for target in targets:
            state = accumulators[target]
            candidate = state["candidate"]
            prefix = f"graph_{category}_"
            output[target].update(
                {
                    prefix + "global_relation_present": float(global_count > 0),
                    prefix + "global_claim_count_log": _log1p(global_count),
                    prefix + "pair_claim_count_log": _log1p(candidate),
                    prefix + "candidate_coverage": candidate / max(1, global_count),
                    prefix + "mass_share": state["mass"] / max(1, global_count),
                    prefix + "winner_share": state["winner"] / max(1, global_count),
                    prefix + "positive_margin_mean": state["positive_margin"]
                    / max(1.0, candidate),
                    prefix + "signed_margin_mean": state["signed_margin"]
                    / max(1, global_count),
                    prefix + "high_mass_share": state["high_mass"]
                    / max(1, global_count),
                    prefix + "unique_claim_share": state["unique"]
                    / max(1, global_count),
                    prefix + "candidate_degree_mean_log": _log1p(
                        state["degree"] / max(1.0, candidate)
                    ),
                }
            )
    return output


def _contrast_feature_names() -> list[str]:
    names = [
        "anchor_specificity",
        "anchor_exact_page_count_log",
        "anchor_strength_strong",
        "chain_connected",
        "chain_score",
        "chain_minimum_relevance",
    ]
    for category in CATEGORIES:
        names.extend(
            [
                f"pair_{category}_qualified_count_log",
                f"pair_{category}_local_count_log",
                f"pair_{category}_same_sentence_count_log",
                f"pair_{category}_exclusive_count_log",
                f"pair_{category}_max_relevance",
                f"pair_{category}_max_binding",
                f"pair_{category}_minimum_distance_score",
                f"pair_{category}_competition_free_ratio",
                f"graph_{category}_pair_claim_count_log",
                f"graph_{category}_mass_share",
                f"graph_{category}_winner_share",
                f"graph_{category}_positive_margin_mean",
                f"graph_{category}_unique_claim_share",
            ]
        )
    return names


def _add_sibling_contrasts(
    target_features: dict[str, dict[str, float]],
) -> None:
    targets = list(target_features)
    for target in targets:
        siblings = [name for name in targets if name != target]
        for name in _contrast_feature_names():
            value = target_features[target][name]
            sibling_values = [target_features[sibling][name] for sibling in siblings]
            sibling_maximum = max(sibling_values, default=0.0)
            sibling_mean = _mean(sibling_values)
            paper_total = value + sum(sibling_values)
            target_features[target][f"contrast_{name}_minus_sibling_max"] = (
                value - sibling_maximum
            )
            target_features[target][f"contrast_{name}_minus_sibling_mean"] = (
                value - sibling_mean
            )
            target_features[target][f"contrast_{name}_paper_share"] = (
                value / paper_total if paper_total > 0.0 else 0.0
            )


def build_grep_entity_pair_features(
    identity: Sequence[Mapping[str, str]],
    page_cache_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    allowed_identity_fields = {"REFNO", "RESULTNO", "ROCKNAME"}
    for row_number, record in enumerate(identity, start=1):
        extra = set(record) - allowed_identity_fields
        if extra:
            raise ValueError(
                f"identity row {row_number} contains forbidden fields: {sorted(extra)}"
            )
    records = [
        {
            "REFNO": clean(record.get("REFNO")),
            "RESULTNO": clean(record.get("RESULTNO")),
            "ROCKNAME": clean(record.get("ROCKNAME")),
        }
        for record in identity
    ]
    if any(not all(record.values()) for record in records):
        raise ValueError("identity requires nonempty REFNO, RESULTNO, and ROCKNAME")
    by_refno: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_refno[record["REFNO"]].append(index)

    rows: list[list[float] | None] = [None] * len(records)
    row_metadata: list[dict[str, Any] | None] = [None] * len(records)
    feature_names: list[str] | None = None
    paper_audit: list[dict[str, Any]] = []
    for paper_number, refno in enumerate(sorted(by_refno), start=1):
        indices = by_refno[refno]
        cache_row = page_cache_rows.get(refno)
        if cache_row is None:
            raise ValueError(f"missing page cache row for REFNO {refno}")
        target_names: dict[str, str] = {}
        for index in indices:
            record = records[index]
            canonical = canonical_target_identity(record["ROCKNAME"])
            target_names.setdefault(canonical, record["ROCKNAME"])
        canonical_targets = list(target_names)
        retriever = PaperTextRetriever(cache_row)
        retrievals: dict[str, dict[str, Any]] = {}
        for canonical in canonical_targets:
            retrievals[canonical] = retriever.retrieve(
                target_names[canonical],
                multi_target=len(canonical_targets) > 1,
                indicator="R3",
                competing_targets=[
                    target_names[sibling]
                    for sibling in canonical_targets
                    if sibling != canonical
                ],
                allow_structured_scope=True,
                allow_discourse_scope=True,
                limit=MAX_HITS_PER_CATEGORY * len(CATEGORIES),
                per_category_limit=MAX_HITS_PER_CATEGORY,
            )
        claims = _claim_maps(retrievals)
        target_features = {
            target: _raw_target_features(retrievals[target])
            for target in canonical_targets
        }
        graph = _graph_features(canonical_targets, claims)
        global_chain = all(bool(claims[category]) for category in ("method", "carrier", "bridge"))
        total_claims = sum(len(claims[category]) for category in CATEGORIES)
        for target in canonical_targets:
            target_features[target].update(graph[target])
            target_features[target].update(
                {
                    "paper_target_count_log": _log1p(len(canonical_targets)),
                    "paper_global_chain_present": float(global_chain),
                    "paper_global_claim_count_log": _log1p(total_claims),
                }
            )
        _add_sibling_contrasts(target_features)

        for index in indices:
            canonical = canonical_target_identity(records[index]["ROCKNAME"])
            feature_map = target_features[canonical]
            names = list(feature_map)
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise AssertionError("GREP entity-pair feature order changed")
            rows[index] = [float(feature_map[name]) for name in feature_names]
            row_metadata[index] = {
                "canonical_target": canonical,
                "paper_target_count": len(canonical_targets),
                "paper_global_chain_present": global_chain,
                "paper_global_claim_count": total_claims,
            }
        paper_audit.append(
            {
                "REFNO": refno,
                "target_count": len(canonical_targets),
                "global_claim_counts": {
                    category: len(claims[category]) for category in CATEGORIES
                },
                "retrieval_scan_counts": retriever.cache_info()["indicator_scan_counts"],
            }
        )
        if paper_number % 250 == 0:
            print(
                f"GREP entity-pair features: papers={paper_number}/{len(by_refno)}",
                flush=True,
            )

    if feature_names is None or any(row is None for row in rows):
        raise AssertionError("GREP entity-pair feature construction is incomplete")
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise AssertionError("GREP entity-pair feature matrix is invalid")
    return {
        "matrix": matrix,
        "feature_names": feature_names,
        "row_metadata": row_metadata,
        "paper_audit": paper_audit,
        "paper_count": len(by_refno),
    }


__all__ = [
    "build_grep_entity_pair_features",
    "load_page_cache_rows",
]
