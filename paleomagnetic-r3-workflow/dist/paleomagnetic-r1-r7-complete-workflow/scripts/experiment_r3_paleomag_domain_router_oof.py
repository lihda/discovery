#!/usr/bin/env python3
"""Nested OOF evaluation of a paleomagnetic domain router for R3."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_all_labelled_pdf_grouped_oof as all_oof  # noqa: E402
import experiment_r3_exact300_paper_consistency as paper_v1  # noqa: E402
import experiment_r3_grep_entity_pair_oof as grep_oof  # noqa: E402
import experiment_r3_grep_pairwise_rank_oof as pair_oof  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


OOF_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
OUTPUT_DIR = ROOT / "tmp/r3_grep_domain_v4folds_v1"
COMBINED_FEATURES = OOF_DIR / "combined_features.npz"
COMBINED_MANIFEST = OOF_DIR / "combined_features.manifest.json"
GREP_FEATURES = OOF_DIR / "grep_entity_pair_features_v1.npz"
GREP_MANIFEST = OOF_DIR / "grep_entity_pair_features_v1.manifest.json"
PALEOMAG_FEATURES = OOF_DIR / "paleomag_scope_features_v1.npz"
PALEOMAG_MANIFEST = OOF_DIR / "paleomag_scope_features_v1.manifest.json"
FOLD_MANIFEST = OOF_DIR / "refno_fold_manifest.json"
PRIOR_PAIR_OOF = OOF_DIR / "grep_pairwise_rank_oof_v2.npz"
CANDIDATE_FREEZE = OOF_DIR / "paleomag_domain_router_candidate_freeze_v1.json"
OOF_OUTPUT = OOF_DIR / "paleomag_domain_router_oof_v1.npz"
PREDICTION_RECEIPT = OOF_DIR / "paleomag_domain_router_prediction_receipt_v1.json"
REPORT_OUTPUT = OOF_DIR / "paleomag_domain_router_report_v1.json"

PRIMARY_STRATEGY = "paleomag_domain_nested_router"
CURRENT_STRATEGY = "grep_pairwise_nested_router_recomputed"
AUGMENTED_STRATEGY = "paleomag_expanded_mean4"


@dataclass(frozen=True)
class DomainRouteSpec:
    name: str
    gate_threshold: float
    paleomag_weight: float


def domain_route_specs() -> tuple[DomainRouteSpec, ...]:
    specs = [DomainRouteSpec("identity_current_pairwise", 1.1, 0.0)]
    for threshold in (0.5, 0.6, 0.7, 0.8):
        for weight in (0.5, 0.75, 1.0):
            specs.append(
                DomainRouteSpec(
                    f"paleomag_hgb_t{threshold:.2f}_w{weight:.2f}",
                    threshold,
                    weight,
                )
            )
    return tuple(specs)


def apply_domain_route(
    current_probability: np.ndarray,
    augmented_probability: np.ndarray,
    gate_probability: np.ndarray,
    multi_mask: np.ndarray,
    spec: DomainRouteSpec,
) -> np.ndarray:
    current = np.asarray(current_probability, dtype=np.float64)
    augmented = np.asarray(augmented_probability, dtype=np.float64)
    gate = np.asarray(gate_probability, dtype=np.float64)
    if not (
        current.shape == augmented.shape == gate.shape == multi_mask.shape
        and np.isfinite(current).all()
        and np.isfinite(augmented).all()
        and np.isfinite(gate).all()
    ):
        raise ValueError("domain route arrays must be finite and aligned")
    if spec.paleomag_weight == 0.0:
        return current.copy()
    routed = pair_oof.sigmoid(
        (1.0 - spec.paleomag_weight) * pair_oof.logit(current)
        + spec.paleomag_weight * pair_oof.logit(augmented)
    )
    enabled = multi_mask & (gate >= spec.gate_threshold)
    return np.where(enabled, routed, current)


def select_domain_route(
    specs: Sequence[DomainRouteSpec],
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    multi_mask: np.ndarray,
    mixed_mask: np.ndarray,
    selected_indices: np.ndarray,
) -> tuple[DomainRouteSpec, dict[str, float]]:
    selected = np.zeros(len(labels), dtype=bool)
    selected[selected_indices] = True
    masks = (selected & multi_mask, selected, selected & mixed_mask)

    def metrics(spec: DomainRouteSpec) -> dict[str, float]:
        probability = probabilities[spec.name]
        return {
            "multi_target_accuracy": pair_oof.accuracy(labels, probability, masks[0]),
            "overall_accuracy": pair_oof.accuracy(labels, probability, masks[1]),
            "mixed_label_accuracy": pair_oof.accuracy(labels, probability, masks[2]),
        }

    best = specs[0]
    best_metrics = metrics(best)
    best_key = tuple(best_metrics[name] for name in (
        "multi_target_accuracy", "overall_accuracy", "mixed_label_accuracy"
    ))
    for spec in specs[1:]:
        observed = metrics(spec)
        key = tuple(observed[name] for name in (
            "multi_target_accuracy", "overall_accuracy", "mixed_label_accuracy"
        ))
        if key > best_key:
            best, best_metrics, best_key = spec, observed, key
    return best, best_metrics


def paper_indices(
    row_indices: np.ndarray,
    row_to_paper: np.ndarray,
    paper_target_counts: np.ndarray,
    *,
    multi_only: bool,
) -> np.ndarray:
    values = sorted({int(row_to_paper[index]) for index in row_indices})
    if multi_only:
        values = [index for index in values if paper_target_counts[index] > 1]
    return np.asarray(values, dtype=np.int64)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=all_oof.IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--combined-features", type=Path, default=COMBINED_FEATURES)
    parser.add_argument("--combined-manifest", type=Path, default=COMBINED_MANIFEST)
    parser.add_argument("--grep-features", type=Path, default=GREP_FEATURES)
    parser.add_argument("--grep-manifest", type=Path, default=GREP_MANIFEST)
    parser.add_argument("--paleomag-features", type=Path, default=PALEOMAG_FEATURES)
    parser.add_argument("--paleomag-manifest", type=Path, default=PALEOMAG_MANIFEST)
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    if runtime != paper_v1.CANONICAL_RUNTIME:
        raise ValueError(
            f"canonical runtime required: expected={paper_v1.CANONICAL_RUNTIME}, observed={runtime}"
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = output_dir / CANDIDATE_FREEZE.name
    oof_path = output_dir / OOF_OUTPUT.name
    receipt_path = output_dir / PREDICTION_RECEIPT.name
    report_path = output_dir / REPORT_OUTPUT.name

    labelled_records = all_oof.load_identity(args.identity.resolve())
    records, coverage = all_oof.pdf_only_records(labelled_records, args.pdf_dir.resolve())
    alignment_sha256 = all_oof.record_sha256(records)
    fold_ids = grep_oof.load_fold_ids(args.fold_manifest.resolve(), records)
    folds = grep_oof.fold_indices(fold_ids)

    manifests = {
        "combined": json.loads(args.combined_manifest.resolve().read_text("utf-8")),
        "grep": json.loads(args.grep_manifest.resolve().read_text("utf-8")),
        "paleomag": json.loads(args.paleomag_manifest.resolve().read_text("utf-8")),
    }
    for name, manifest in manifests.items():
        if manifest.get("alignment_sha256") != alignment_sha256:
            raise ValueError(f"{name} feature alignment hash changed")
    combined = np.load(args.combined_features.resolve())["combined"].astype(np.float32)
    grep = np.load(args.grep_features.resolve())["grep"].astype(np.float32)
    paleomag = np.load(args.paleomag_features.resolve())["paleomag"].astype(np.float32)
    if any(matrix.shape[0] != len(records) for matrix in (combined, grep, paleomag)):
        raise ValueError("feature rows do not align with evaluated identity")
    expanded = np.concatenate((combined, grep), axis=1).astype(np.float32, copy=False)
    augmented = np.concatenate((expanded, paleomag), axis=1).astype(np.float32, copy=False)
    grep_refnos, grep_paper_matrix, grep_row_to_paper = pair_oof.paper_aggregate_features(
        grep, records
    )
    domain_refnos, domain_paper_matrix, domain_row_to_paper = pair_oof.paper_aggregate_features(
        np.concatenate((grep, paleomag), axis=1), records
    )
    if grep_refnos != domain_refnos or not np.array_equal(grep_row_to_paper, domain_row_to_paper):
        raise AssertionError("paper aggregate identities changed")
    refnos = domain_refnos
    row_to_paper = domain_row_to_paper
    by_refno = pair_oof.rows_by_refno(records)
    paper_target_counts = np.asarray([len(by_refno[refno]) for refno in refnos])

    routes = domain_route_specs()
    candidate_freeze = {
        "schema_version": "r3_paleomag_domain_router_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "purpose": "use paleomagnetic structure only when a label-free paper gate predicts target heterogeneity",
        "current_pairwise_candidates": [asdict(spec) for spec in pair_oof.fusion_specs()],
        "domain_route_candidates": [asdict(spec) for spec in routes],
        "domain_gate": asdict(pair_oof.GateSpec("paleomag_domain_gate", "hgb_leaf20")),
        "selection_objective": "multi-target accuracy, then overall accuracy, then mixed-label accuracy",
        "outer_folds": all_oof.N_SPLITS,
        "inner_folds": pair_oof.INNER_SPLITS,
        "coverage": coverage,
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_model_input": False,
            "paleomagnetic_features_frozen_without_label_access": True,
            "refno_used_for_group_isolation_and_sibling_reasoning": True,
        },
        "feature_manifests": {
            name: {
                "path": str(path.resolve()),
                "sha256": base.file_sha256(path.resolve()),
            }
            for name, path in (
                ("combined", args.combined_manifest),
                ("grep", args.grep_manifest),
                ("paleomag", args.paleomag_manifest),
            )
        },
        "canonical_runtime": runtime,
    }
    base.atomic_json(candidate_path, candidate_freeze)

    labels_by_resultno = all_oof.load_labels(
        args.targets.resolve(), {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    canonical_targets = [canonical_target_identity(record["ROCKNAME"]) for record in records]
    multi_mask = pair_oof.multi_target_mask(records, labelled_records)
    mixed_mask = pair_oof.mixed_label_mask(records, labels)
    paper_labels = np.asarray(
        [int(len({int(labels[index]) for index in by_refno[refno]}) > 1) for refno in refnos],
        dtype=np.int8,
    )
    groups = [record["REFNO"] for record in records]
    current_candidates = pair_oof.fusion_specs()
    rank_by_name = {spec.name: spec for spec in pair_oof.rank_specs()}
    gate_by_name = {spec.name: spec for spec in pair_oof.gate_specs()}
    domain_gate_spec = pair_oof.GateSpec("paleomag_domain_gate", "hgb_leaf20")

    baseline_oof = np.full(len(records), np.nan)
    augmented_oof = np.full(len(records), np.nan)
    current_oof = np.full(len(records), np.nan)
    primary_oof = np.full(len(records), np.nan)
    domain_gate_oof = np.zeros(len(records))
    selections: list[dict[str, Any]] = []

    for outer_fold, (outer_train, outer_test) in enumerate(folds, start=1):
        baseline_oof[outer_test] = pair_oof.fit_baseline_mean4(
            expanded, labels, outer_train, outer_test, records, canonical_targets
        )
        augmented_oof[outer_test] = pair_oof.fit_baseline_mean4(
            augmented, labels, outer_train, outer_test, records, canonical_targets
        )
        inner_folds = pair_oof.nested.deterministic_group_folds(
            outer_train,
            [groups[int(index)] for index in outer_train],
            n_splits=pair_oof.INNER_SPLITS,
            seed=pair_oof.INNER_SEED + outer_fold,
        )
        inner_baseline = np.full(len(records), np.nan)
        inner_augmented = np.full(len(records), np.nan)
        inner_rank = {spec.name: np.zeros(len(records)) for spec in pair_oof.rank_specs()}
        inner_gate = {spec.name: np.zeros(len(records)) for spec in pair_oof.gate_specs()}
        inner_domain_gate = np.zeros(len(records))

        for inner_train, inner_validation in inner_folds:
            inner_baseline[inner_validation] = pair_oof.fit_baseline_mean4(
                expanded, labels, inner_train, inner_validation, records, canonical_targets
            )
            inner_augmented[inner_validation] = pair_oof.fit_baseline_mean4(
                augmented, labels, inner_train, inner_validation, records, canonical_targets
            )
            for spec in pair_oof.rank_specs():
                fitted, _ = pair_oof.fit_ranker(spec, combined, labels, inner_train, records)
                normalized = pair_oof.normalize_scores_by_paper(
                    fitted.score(combined), records, inner_validation
                )
                inner_rank[spec.name][inner_validation] = normalized[inner_validation]
            train_papers = paper_indices(
                inner_train, row_to_paper, paper_target_counts, multi_only=True
            )
            validation_papers = paper_indices(
                inner_validation, row_to_paper, paper_target_counts, multi_only=False
            )
            for spec in pair_oof.gate_specs():
                fitted_gate = pair_oof.fit_gate(
                    spec, grep_paper_matrix, paper_labels, train_papers
                )
                values = fitted_gate.predict(grep_paper_matrix[validation_papers])
                mapping = dict(zip(validation_papers, values))
                inner_gate[spec.name][inner_validation] = [
                    mapping[int(row_to_paper[index])] for index in inner_validation
                ]
            fitted_domain_gate = pair_oof.fit_gate(
                domain_gate_spec, domain_paper_matrix, paper_labels, train_papers
            )
            domain_values = fitted_domain_gate.predict(domain_paper_matrix[validation_papers])
            domain_mapping = dict(zip(validation_papers, domain_values))
            inner_domain_gate[inner_validation] = [
                domain_mapping[int(row_to_paper[index])] for index in inner_validation
            ]

        current_probabilities: dict[str, np.ndarray] = {}
        for spec in current_candidates:
            if spec.rank is None:
                current_probabilities[spec.name] = inner_baseline.copy()
            else:
                current_probabilities[spec.name] = pair_oof.apply_fusion(
                    inner_baseline,
                    inner_rank[spec.rank],
                    inner_gate[spec.gate],
                    multi_mask,
                    spec,
                )
        selected_current, current_metrics = pair_oof.select_fusion(
            current_candidates,
            labels,
            current_probabilities,
            multi_mask,
            mixed_mask,
            outer_train,
        )
        inner_current = current_probabilities[selected_current.name]
        safe_inner_current = np.nan_to_num(inner_current, nan=0.5)
        safe_inner_augmented = np.nan_to_num(inner_augmented, nan=0.5)
        route_probabilities = {
            spec.name: apply_domain_route(
                safe_inner_current,
                safe_inner_augmented,
                inner_domain_gate,
                multi_mask,
                spec,
            )
            for spec in routes
        }
        selected_route, route_metrics = select_domain_route(
            routes,
            labels,
            route_probabilities,
            multi_mask,
            mixed_mask,
            outer_train,
        )

        if selected_current.rank is None:
            current_test = baseline_oof[outer_test]
        else:
            ranker, _ = pair_oof.fit_ranker(
                rank_by_name[selected_current.rank], combined, labels, outer_train, records
            )
            normalized = pair_oof.normalize_scores_by_paper(
                ranker.score(combined), records, outer_test
            )
            train_papers = paper_indices(
                outer_train, row_to_paper, paper_target_counts, multi_only=True
            )
            test_papers = paper_indices(
                outer_test, row_to_paper, paper_target_counts, multi_only=False
            )
            fitted_gate = pair_oof.fit_gate(
                gate_by_name[selected_current.gate], grep_paper_matrix, paper_labels, train_papers
            )
            gate_values = fitted_gate.predict(grep_paper_matrix[test_papers])
            mapping = dict(zip(test_papers, gate_values))
            row_gate = np.zeros(len(records))
            row_gate[outer_test] = [mapping[int(row_to_paper[index])] for index in outer_test]
            current_full = pair_oof.apply_fusion(
                baseline_oof, normalized, row_gate, multi_mask, selected_current
            )
            current_test = current_full[outer_test]
        current_oof[outer_test] = current_test

        train_papers = paper_indices(
            outer_train, row_to_paper, paper_target_counts, multi_only=True
        )
        test_papers = paper_indices(
            outer_test, row_to_paper, paper_target_counts, multi_only=False
        )
        fitted_domain_gate = pair_oof.fit_gate(
            domain_gate_spec, domain_paper_matrix, paper_labels, train_papers
        )
        gate_values = fitted_domain_gate.predict(domain_paper_matrix[test_papers])
        mapping = dict(zip(test_papers, gate_values))
        test_gate = np.asarray(
            [mapping[int(row_to_paper[index])] for index in outer_test], dtype=np.float64
        )
        domain_gate_oof[outer_test] = test_gate
        primary_oof[outer_test] = apply_domain_route(
            current_test,
            augmented_oof[outer_test],
            test_gate,
            multi_mask[outer_test],
            selected_route,
        )
        selections.append(
            {
                "outer_fold": outer_fold,
                "selected_current_pairwise": asdict(selected_current),
                "current_inner_metrics": current_metrics,
                "selected_domain_route": asdict(selected_route),
                "domain_route_inner_metrics": route_metrics,
            }
        )
        print(
            f"paleomag outer fold {outer_fold}: current={selected_current.name}, "
            f"domain={selected_route.name}",
            flush=True,
        )

    for values in (baseline_oof, augmented_oof, current_oof, primary_oof, domain_gate_oof):
        if not np.isfinite(values).all():
            raise AssertionError("paleomagnetic nested OOF arrays are incomplete")
    arrays = {
        "grep_expanded_mean4_recomputed": baseline_oof,
        AUGMENTED_STRATEGY: augmented_oof,
        CURRENT_STRATEGY: current_oof,
        PRIMARY_STRATEGY: primary_oof,
        "paleomag_domain_gate_probability": domain_gate_oof,
    }
    grep_oof.atomic_npz(oof_path, **arrays)
    receipt = {
        "schema_version": "r3_paleomag_domain_router_prediction_receipt_v1",
        "status": "NESTED_OOF_PREDICTIONS_FROZEN_BEFORE_AGGREGATE_SCORING",
        "primary_strategy": PRIMARY_STRATEGY,
        "candidate_freeze_sha256": base.file_sha256(candidate_path),
        "oof": {"path": str(oof_path), "sha256": base.file_sha256(oof_path)},
        "each_row_predicted_by_model_excluding_its_refno": True,
        "each_outer_candidate_selected_without_outer_test_labels": True,
        "rows": [
            {
                "row_index": index,
                **record,
                "fold": int(fold_ids[index]),
                "probability": float(primary_oof[index]),
                "prediction": int(primary_oof[index] >= 0.5),
            }
            for index, record in enumerate(records)
        ],
    }
    base.atomic_json(receipt_path, receipt)

    masks = grep_oof.segment_masks(records, labelled_records, labels)
    reports = {
        name: grep_oof.strategy_report(records, labels, probability, masks)
        for name, probability in (
            ("grep_expanded_mean4_recomputed", baseline_oof),
            (AUGMENTED_STRATEGY, augmented_oof),
            (CURRENT_STRATEGY, current_oof),
            (PRIMARY_STRATEGY, primary_oof),
        )
    }
    current_report = reports[CURRENT_STRATEGY]
    primary_report = reports[PRIMARY_STRATEGY]
    prior_difference: float | None = None
    if PRIOR_PAIR_OOF.exists():
        prior = np.load(PRIOR_PAIR_OOF)["grep_pairwise_nested_router"].astype(np.float64)
        prior_difference = float(np.max(np.abs(prior - current_oof)))
    report = {
        "schema_version": "r3_paleomag_domain_router_nested_oof_v1",
        "status": "complete_development_nested_oof",
        "primary_strategy": PRIMARY_STRATEGY,
        "claim_boundary": {
            "internal_refno_grouped_nested_oof": True,
            "independent_external_validation": False,
            "candidate_family_informed_by_prior_development_analysis": True,
            "outer_test_labels_excluded_from_training_and_selection": True,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluations_used_as_model_input": False,
            "paleomagnetic_feature_rules_are_label_free": True,
        },
        "current_pairwise_reproduction": {
            "prior_oof_path": str(PRIOR_PAIR_OOF),
            "maximum_absolute_probability_difference": prior_difference,
            "within_numeric_tolerance_1e_7": (
                prior_difference is not None and prior_difference <= 1e-7
            ),
        },
        "nested_selection": {
            "domain_route_counts": dict(
                Counter(item["selected_domain_route"]["name"] for item in selections)
            ),
            "outer_folds": selections,
        },
        "strategies": reports,
        "delta_vs_current_pairwise": {
            "overall_accuracy": round(
                primary_report["overall"]["accuracy"] - current_report["overall"]["accuracy"], 8
            ),
            "multi_target_accuracy": round(
                primary_report["segments"]["multi_target"]["accuracy"]
                - current_report["segments"]["multi_target"]["accuracy"], 8
            ),
            "mixed_label_accuracy": round(
                primary_report["segments"]["multi_target_mixed_labels"]["accuracy"]
                - current_report["segments"]["multi_target_mixed_labels"]["accuracy"], 8
            ),
            "uniform_multi_target_accuracy": round(
                primary_report["segments"]["multi_target_uniform_labels"]["accuracy"]
                - current_report["segments"]["multi_target_uniform_labels"]["accuracy"], 8
            ),
        },
        "paired_refno_cluster_bootstrap": {
            "overall": pair_oof.clustered_bootstrap_delta(
                records, labels, current_oof, primary_oof, np.ones(len(records), dtype=bool)
            ),
            "multi_target": pair_oof.clustered_bootstrap_delta(
                records, labels, current_oof, primary_oof, masks["multi_target"]
            ),
            "mixed_label": pair_oof.clustered_bootstrap_delta(
                records, labels, current_oof, primary_oof, masks["multi_target_mixed_labels"]
            ),
        },
        "promotion_decision": {
            "promote_to_current_supervised_evaluation": bool(
                primary_report["overall"]["accuracy"] > current_report["overall"]["accuracy"]
                and primary_report["segments"]["multi_target"]["accuracy"]
                > current_report["segments"]["multi_target"]["accuracy"]
            ),
            "promote_to_strict_production_scoring": False,
        },
        "artifacts": {
            "candidate_freeze": {"path": str(candidate_path), "sha256": base.file_sha256(candidate_path)},
            "prediction_receipt": {"path": str(receipt_path), "sha256": base.file_sha256(receipt_path)},
            "oof_probabilities": {"path": str(oof_path), "sha256": base.file_sha256(oof_path)},
        },
    }
    base.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "current": current_report["overall"],
                "primary": primary_report["overall"],
                "primary_multi": primary_report["segments"]["multi_target"],
                "primary_mixed": primary_report["segments"]["multi_target_mixed_labels"],
                "delta": report["delta_vs_current_pairwise"],
                "promotion": report["promotion_decision"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
