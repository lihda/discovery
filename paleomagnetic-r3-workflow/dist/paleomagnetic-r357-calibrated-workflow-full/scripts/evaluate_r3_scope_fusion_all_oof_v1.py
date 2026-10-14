#!/usr/bin/env python3
"""Evaluate single-target V3 with GREP-assisted multi-target R3 scoring."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_query_adaptive_retrieval_features as retrieval  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import evaluate_r3_query_adaptive_single872_oof as query_eval  # noqa: E402
import evaluate_r3_residual_gate_single872_oof as residual  # noqa: E402
import evaluate_r3_style_fusion_single872_oof_v2 as style_v2  # noqa: E402
import experiment_r3_grep_entity_pair_oof as grep_eval  # noqa: E402
import experiment_r3_grep_pairwise_rank_oof as pair_eval  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402


DEFAULT_V2_CONFIG = ROOT / "data/knowledge/r3_style_fusion_evaluation_v2.json"
DEFAULT_V1_CONFIG = ROOT / "data/knowledge/r3_residual_gate_evaluation_v1.json"
DEFAULT_SINGLE_V3 = (
    ROOT
    / "tmp/r3_fulltext_fusion_single872_v3"
    / "r3_fulltext_fusion_single872_grouped_oof_v3.npz"
)
DEFAULT_GREP_DIR = ROOT / "tmp/r3_grep_domain_v4folds_v1"
DEFAULT_GREP_OOF = DEFAULT_GREP_DIR / "paleomag_domain_router_oof_v1.npz"
DEFAULT_GREP_RECEIPT = (
    DEFAULT_GREP_DIR / "paleomag_domain_router_prediction_receipt_v1.json"
)
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_scope_fusion_all_oof_v1"

FREEZE_NAME = "r3_scope_fusion_evaluation_freeze_v1.json"
OOF_NAME = "r3_scope_fusion_all_grouped_oof_v1.npz"
REPORT_NAME = "r3_scope_fusion_all_grouped_oof_report_v1.json"

PAIRWISE = "grep_pairwise_nested_router_recomputed"
DOMAIN = "paleomag_domain_nested_router"


class ScopeFusionError(ValueError):
    pass


def clean(value: Any) -> str:
    return residual.clean(value)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ScopeFusionError(f"JSON object required: {path}")
    return value


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": trace_freeze.file_sha256(path)}


def delta(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    baseline_correct = (baseline >= 0.5) == labels
    candidate_correct = (candidate >= 0.5) == labels
    repairs = int((~baseline_correct & candidate_correct).sum())
    degradations = int((baseline_correct & ~candidate_correct).sum())
    return {
        "repairs": repairs,
        "degradations": degradations,
        "net_repairs": repairs - degradations,
        "accuracy_delta": round(float(candidate_correct.mean() - baseline_correct.mean()), 8),
    }


def scoped(single: np.ndarray, multi: np.ndarray, single_mask: np.ndarray) -> np.ndarray:
    if not (single.shape == multi.shape == single_mask.shape):
        raise ScopeFusionError("scope arrays are not aligned")
    return np.where(single_mask, single, multi)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    parser.add_argument("--v1-config", type=Path, default=DEFAULT_V1_CONFIG)
    parser.add_argument("--domain-features", type=Path, default=residual.DEFAULT_DOMAIN_FEATURES)
    parser.add_argument("--domain-manifest", type=Path, default=residual.DEFAULT_DOMAIN_MANIFEST)
    parser.add_argument("--page-cache", type=Path, default=residual.DEFAULT_PAGE_CACHE)
    parser.add_argument("--v4-features", type=Path, default=residual.DEFAULT_V4_FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=residual.DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4-context", type=Path, default=residual.DEFAULT_V4_CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=residual.DEFAULT_V4_CANDIDATE)
    parser.add_argument("--v4-folds", type=Path, default=residual.DEFAULT_V4_FOLDS)
    parser.add_argument("--query-dir", type=Path, default=residual.QUERY_DIR)
    parser.add_argument("--single-v3", type=Path, default=DEFAULT_SINGLE_V3)
    parser.add_argument("--grep-oof", type=Path, default=DEFAULT_GREP_OOF)
    parser.add_argument("--grep-receipt", type=Path, default=DEFAULT_GREP_RECEIPT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    paths = {name: value.resolve() for name, value in vars(args).items() if isinstance(value, Path)}
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    v2_config = load_json(paths["v2_config"])
    style_v2.validate_config(v2_config)
    v1_config = load_json(paths["v1_config"])
    residual.validate_config(v1_config)
    (
        records,
        v4,
        _v4_names,
        _contexts,
        fold_ids,
        document_groups,
        _v4_candidate,
    ) = trace_eval.load_frozen_inputs(
        identity_path=paths["identity"],
        pdf_dir=paths["pdf_dir"],
        feature_path=paths["v4_features"],
        manifest_path=paths["v4_manifest"],
        context_path=paths["v4_context"],
        candidate_path=paths["v4_candidate"],
        fold_path=paths["v4_folds"],
    )
    complete_identity = trace_freeze.load_three_column_identity(paths["identity"])
    multiplicity = Counter(record["REFNO"] for record in complete_identity)
    single_mask = np.asarray(
        [multiplicity[record["REFNO"]] == 1 for record in records], dtype=bool
    )
    if (len(records), int(single_mask.sum()), int((~single_mask).sum())) != (2751, 872, 1879):
        raise ScopeFusionError("expected 2751/872/1879 evaluation scope changed")

    compact, pooled, query_single, _query_manifest, query_candidate = (
        query_eval.load_query_features(
            query_dir=paths["query_dir"],
            records=records,
            v4_fold_ids=fold_ids,
            expected_single_mask=single_mask,
        )
    )
    if not np.array_equal(query_single, single_mask):
        raise ScopeFusionError("query feature scope changed")
    query_matrices = query_eval.build_matrices(v4, compact, pooled)
    domain_features, domain_manifest = residual.load_domain_features(
        paths["domain_features"], paths["domain_manifest"], records
    )
    if trace_freeze.file_sha256(paths["page_cache"]) != clean(
        (domain_manifest.get("inputs") or {}).get("page_cache", {}).get("sha256")
    ):
        raise ScopeFusionError("front-text page cache changed")
    documents = residual.front_documents(
        records,
        paths["page_cache"],
        int(v1_config["front_text_expert"]["characters"]),
    )

    candidate_specs = {
        "scope_v3_plus_no_strict": "V3 on single-target rows; no-strict base on multi-target rows",
        "scope_v3_plus_grep_pairwise": "V3 on single-target rows; GREP pairwise router on multi-target rows",
        "scope_v3_plus_grep_domain": "V3 on single-target rows; paleomagnetic GREP domain router on multi-target rows",
        "scope_v3_plus_blend_75_25": "V3 on single-target rows; 0.75 no-strict + 0.25 GREP pairwise on multi-target rows",
        "scope_v3_plus_blend_50_50": "V3 on single-target rows; equal no-strict/GREP pairwise blend on multi-target rows",
        "scope_v3_plus_confident_grep_gate": (
            "V3 on single-target rows; GREP replaces no-strict only when GREP margin >= 0.25 "
            "and no-strict margin <= 0.15"
        ),
    }
    freeze_path = output_dir / FREEZE_NAME
    freeze = {
        "schema_version": "r3_scope_fusion_evaluation_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": {"pdf_backed_rows": 2751, "single_target": 872, "multi_target": 1879},
        "candidate_specs": candidate_specs,
        "inputs": {
            name: file_record(path)
            for name, path in paths.items()
            if name != "output_dir" and path.is_file()
        },
        "boundary": {
            "pmagresult_or_rockmag_used_as_feature": False,
            "outer_test_labels_used_for_training_or_candidate_selection": False,
            "scope_router_uses_only_target_multiplicity": True,
            "component_oof_predictions_are_terminal_scope_outputs_not_meta_model_features": True,
            "candidate_family_informed_by_prior_internal_development": True,
        },
    }
    style_v2.atomic_json(freeze_path, freeze)

    # First label access in this process.
    label_map = trace_eval.load_labels(
        paths["targets"], {record["RESULTNO"] for record in records}
    )
    labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
    no_strict_oof = np.full(len(records), np.nan, dtype=np.float64)
    component_oof = {
        name: np.full(len(records), np.nan, dtype=np.float64)
        for name in style_v2.BASE_COMPONENTS
    }
    fold_audit: list[dict[str, Any]] = []
    for outer_fold in range(1, 6):
        test = np.flatnonzero(fold_ids == outer_fold)
        train = np.flatnonzero(fold_ids != outer_fold)
        if set(map(clean, document_groups[test])) & set(map(clean, document_groups[train])):
            raise AssertionError("outer fold leaks a normalized document")
        components = style_v2.fit_base_components(
            train=train,
            test=test,
            labels=labels,
            records=records,
            domain=domain_features,
            query_matrices=query_matrices,
            query_config=query_candidate["evaluation_candidates"],
            documents=documents,
            v1_config=v1_config,
        )
        probability = style_v2.base_fusion(components, v2_config)
        no_strict_oof[test] = probability
        for name, values in components.items():
            component_oof[name][test] = values
        fold_audit.append(
            {
                "outer_fold": outer_fold,
                "train_rows": len(train),
                "test_rows": len(test),
                "no_strict_accuracy": trace_eval.metric(labels[test], probability)["accuracy"],
            }
        )
        print(f"scope fusion base: fold={outer_fold}/5", flush=True)
    if not np.isfinite(no_strict_oof).all():
        raise AssertionError("no-strict OOF is incomplete")

    # Existing component OOF files are opened only after the new all-row base is immutable.
    with np.load(paths["single_v3"], allow_pickle=False) as source:
        single_indices = source["source_row_indices"].astype(np.int64)
        single_labels = source["labels"].astype(np.int8)
        single_folds = source["fold_ids"].astype(np.int8)
        source_no_strict = source["no_strict_base"].astype(np.float64)
        single_probability = source["primary"].astype(np.float64)
    expected_single = np.flatnonzero(single_mask)
    if not np.array_equal(single_indices, expected_single):
        raise ScopeFusionError("single V3 source indices changed")
    if not np.array_equal(single_labels, labels[single_indices]):
        raise ScopeFusionError("single V3 labels do not align")
    if not np.array_equal(single_folds, fold_ids[single_indices]):
        raise ScopeFusionError("single V3 folds do not align with V4")
    base_reproduction_maximum = float(
        np.max(np.abs(no_strict_oof[single_indices] - source_no_strict))
    )
    if base_reproduction_maximum > 1e-12:
        raise ScopeFusionError("single V3 no-strict base reproduction failed")
    v3_all = no_strict_oof.copy()
    v3_all[single_indices] = single_probability

    grep_receipt = load_json(paths["grep_receipt"])
    receipt_rows = grep_receipt.get("rows") or []
    if len(receipt_rows) != len(records):
        raise ScopeFusionError("GREP receipt row count changed")
    for index, (record, row) in enumerate(zip(records, receipt_rows)):
        if (
            int(row.get("row_index", -1)) != index
            or clean(row.get("REFNO")) != clean(record["REFNO"])
            or clean(row.get("RESULTNO")) != clean(record["RESULTNO"])
            or int(row.get("fold", -1)) != int(fold_ids[index])
        ):
            raise ScopeFusionError(f"GREP identity/fold mismatch at row {index}")
    with np.load(paths["grep_oof"], allow_pickle=False) as source:
        grep_pairwise = source[PAIRWISE].astype(np.float64)
        grep_domain = source[DOMAIN].astype(np.float64)
    if grep_pairwise.shape != labels.shape or grep_domain.shape != labels.shape:
        raise ScopeFusionError("GREP OOF shape changed")

    blend_75_25 = 0.75 * no_strict_oof + 0.25 * grep_pairwise
    blend_50_50 = 0.50 * no_strict_oof + 0.50 * grep_pairwise
    confident = (
        (np.abs(grep_pairwise - 0.5) >= 0.25)
        & (np.abs(no_strict_oof - 0.5) <= 0.15)
        & (~single_mask)
    )
    gated = np.where(confident, grep_pairwise, no_strict_oof)
    strategies = {
        "scope_v3_plus_no_strict": v3_all,
        "scope_v3_plus_grep_pairwise": scoped(v3_all, grep_pairwise, single_mask),
        "scope_v3_plus_grep_domain": scoped(v3_all, grep_domain, single_mask),
        "scope_v3_plus_blend_75_25": scoped(v3_all, blend_75_25, single_mask),
        "scope_v3_plus_blend_50_50": scoped(v3_all, blend_50_50, single_mask),
        "scope_v3_plus_confident_grep_gate": scoped(v3_all, gated, single_mask),
    }
    arrays: dict[str, np.ndarray] = {
        "labels": labels,
        "fold_ids": fold_ids,
        "single_target_mask": single_mask,
        "no_strict_all": no_strict_oof,
        "single_v3_scoped": v3_all,
        "grep_pairwise": grep_pairwise,
        "grep_domain": grep_domain,
        "confident_grep_gate": confident,
        **strategies,
    }
    arrays.update({f"component__{name}": values for name, values in component_oof.items()})
    oof_path = output_dir / OOF_NAME
    style_v2.atomic_npz(oof_path, **arrays)

    masks = grep_eval.segment_masks(records, complete_identity, labels)
    reports = {
        name: grep_eval.strategy_report(records, labels, probability, masks)
        for name, probability in strategies.items()
    }
    baseline = strategies["scope_v3_plus_no_strict"]
    deltas = {
        name: {
            "overall": delta(labels, baseline, probability),
            "multi_target": delta(
                labels[~single_mask], baseline[~single_mask], probability[~single_mask]
            ),
        }
        for name, probability in strategies.items()
        if name != "scope_v3_plus_no_strict"
    }
    ranked = sorted(
        reports,
        key=lambda name: (
            reports[name]["overall"]["accuracy"],
            reports[name]["segments"]["multi_target"]["accuracy"],
            name,
        ),
        reverse=True,
    )
    fold_comparison = []
    for outer_fold in range(1, 6):
        multi_fold = (fold_ids == outer_fold) & (~single_mask)
        fold_comparison.append(
            {
                "outer_fold": outer_fold,
                "multi_target_rows": int(multi_fold.sum()),
                "strategies": {
                    name: trace_eval.metric(labels[multi_fold], probability[multi_fold])
                    for name, probability in strategies.items()
                },
            }
        )
    blend = strategies["scope_v3_plus_blend_50_50"]
    bootstrap = {
        "overall": pair_eval.clustered_bootstrap_delta(
            records,
            labels,
            baseline,
            blend,
            np.ones(len(records), dtype=bool),
        ),
        "multi_target": pair_eval.clustered_bootstrap_delta(
            records,
            labels,
            baseline,
            blend,
            ~single_mask,
        ),
    }
    blend_delta = deltas["scope_v3_plus_blend_50_50"]
    fold_nets = [
        int(
            round(
                item["strategies"]["scope_v3_plus_blend_50_50"]["accuracy"]
                * item["multi_target_rows"]
                - item["strategies"]["scope_v3_plus_no_strict"]["accuracy"]
                * item["multi_target_rows"]
            )
        )
        for item in fold_comparison
    ]
    report_path = output_dir / REPORT_NAME
    report = {
        "schema_version": "r3_scope_fusion_all_grouped_oof_report_v1",
        "status": "COMPLETE_INTERNAL_GROUPED_OOF",
        "scope": {
            "pdf_backed_rows": len(records),
            "papers": len({record["REFNO"] for record in records}),
            "single_target_rows": int(single_mask.sum()),
            "multi_target_rows": int((~single_mask).sum()),
            "rows_without_pdf_skipped": len(complete_identity) - len(records),
        },
        "threshold": 0.5,
        "strategies": reports,
        "ranking_by_overall_accuracy": ranked,
        "delta_vs_scope_v3_plus_no_strict": deltas,
        "confident_grep_gate_rows": int(confident.sum()),
        "fold_audit": fold_audit,
        "multi_target_fold_comparison": fold_comparison,
        "paired_refno_cluster_bootstrap_blend_50_50_vs_no_strict": bootstrap,
        "alignment_audit": {
            "single_v3_source_indices_exact": True,
            "single_v3_labels_exact": True,
            "single_v3_v4_folds_exact": True,
            "grep_receipt_identity_exact": True,
            "grep_v4_folds_exact": True,
            "single_no_strict_base_reproduction_maximum_absolute_difference": (
                base_reproduction_maximum
            ),
            "single_no_strict_base_reproduced_at_tolerance_1e_12": True,
        },
        "promotion_decision": {
            "exploratory_internal_winner": ranked[0],
            "winner_net_repairs_vs_no_strict": blend_delta["overall"]["net_repairs"],
            "multi_target_fold_net_repairs": fold_nets,
            "bootstrap_95_interval_excludes_zero": bool(
                bootstrap["multi_target"]["percentile_95_interval"][0] > 0.0
            ),
            "promote_as_stable_improvement": False,
            "reason": (
                "The equal blend improves aggregate internal OOF, but two outer folds regress "
                "and the REFNO-cluster bootstrap interval includes zero."
            ),
        },
        "claim_boundary": {
            "internal_refno_and_duplicate_document_grouped_oof": True,
            "independent_external_validation": False,
            "pmagresult_or_rockmag_used_as_feature": False,
            "existing_evaluation_result_used_as_feature": False,
            "component_oof_probabilities_combined_only_by_frozen_scope_rules": True,
            "candidate_family_informed_by_prior_internal_development": True,
            "do_not_select_final_production_candidate_from_this_report_alone": True,
        },
        "artifacts": {
            "freeze": file_record(freeze_path),
            "oof": file_record(oof_path),
        },
    }
    style_v2.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "ranking": [
                    {
                        "strategy": name,
                        "overall": reports[name]["overall"],
                        "multi_target": reports[name]["segments"]["multi_target"],
                    }
                    for name in ranked
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
