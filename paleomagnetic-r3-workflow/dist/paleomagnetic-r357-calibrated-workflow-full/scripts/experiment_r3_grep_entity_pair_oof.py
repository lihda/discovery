#!/usr/bin/env python3
"""Evaluate GREP-inspired R3 entity-pair reasoning with REFNO-grouped OOF."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
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
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_exact300_paper_consistency as paper_v1  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402
from r3_grep_entity_pair_features import (  # noqa: E402
    build_grep_entity_pair_features,
    load_page_cache_rows,
)


OOF_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
COMBINED_FEATURES = OOF_DIR / "combined_features.npz"
COMBINED_MANIFEST = OOF_DIR / "combined_features.manifest.json"
FOLD_MANIFEST = OOF_DIR / "refno_fold_manifest.json"
BASELINE_OOF = OOF_DIR / "oof_probabilities.npz"
FEATURE_OUTPUT = OOF_DIR / "grep_entity_pair_features_v1.npz"
FEATURE_MANIFEST = OOF_DIR / "grep_entity_pair_features_v1.manifest.json"
CANDIDATE_FREEZE = OOF_DIR / "grep_entity_pair_candidate_freeze_v1.json"
OOF_OUTPUT = OOF_DIR / "grep_entity_pair_oof_v1.npz"
PREDICTION_RECEIPT = OOF_DIR / "grep_entity_pair_prediction_receipt_v1.json"
REPORT_OUTPUT = OOF_DIR / "grep_entity_pair_report_v1.json"

PAPER = {
    "title": "Document-Level Relation Extraction with Global Relations and Entity Pair Reasoning",
    "authors": ["Fu Zhang", "Yi Yan", "Jingwei Cheng"],
    "venue": "Findings of the Association for Computational Linguistics: ACL 2025",
    "publication_date": "2025-07",
    "url": "https://aclanthology.org/2025.findings-acl.1002/",
    "code": "https://github.com/yanyi74/GREP",
}
PRIMARY_STRATEGY = "grep_global_pair_router"


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-5, 1.0 - 1e-5)
    return np.log(clipped / (1.0 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_fold_ids(
    path: Path,
    records: Sequence[Mapping[str, str]],
) -> np.ndarray:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = manifest.get("rows") or []
    if len(rows) != len(records):
        raise ValueError("fold manifest row count does not match evaluated identity")
    fold_ids = np.zeros(len(records), dtype=np.int8)
    for index, (declared, record) in enumerate(zip(rows, records)):
        if int(declared.get("row_index", -1)) != index:
            raise ValueError("fold manifest row index changed")
        for field in ("REFNO", "RESULTNO"):
            if str(declared.get(field) or "") != str(record[field]):
                raise ValueError(f"fold manifest {field} alignment changed at row {index}")
        fold_ids[index] = int(declared["fold"])
    if set(int(value) for value in fold_ids) != set(range(1, all_oof.N_SPLITS + 1)):
        raise ValueError("fold manifest does not contain all expected folds")
    return fold_ids


def fold_indices(fold_ids: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    all_indices = np.arange(len(fold_ids), dtype=np.int64)
    return [
        (all_indices[fold_ids != fold], all_indices[fold_ids == fold])
        for fold in range(1, all_oof.N_SPLITS + 1)
    ]


def model_specs() -> tuple[nested.ExpertSpec, ...]:
    return (
        nested.ExpertSpec("hgb", "matrix", "hgb10"),
        nested.ExpertSpec("rf", "matrix", "rf8"),
        nested.ExpertSpec("extra", "matrix", "extra8"),
        nested.ExpertSpec("logreg", "matrix", "logreg001", standardized=True),
    )


def fit_family(
    family: str,
    matrix: np.ndarray,
    probabilities: dict[str, np.ndarray],
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> None:
    for spec in model_specs():
        fitted = nested.fit_expert(spec, matrix, train, labels, weights)
        probabilities[f"{family}_{spec.name}"][test] = fitted.predict(matrix[test])


def family_mean(
    probabilities: Mapping[str, np.ndarray],
    family: str,
) -> np.ndarray:
    return np.column_stack(
        [probabilities[f"{family}_{spec.name}"] for spec in model_specs()]
    ).mean(axis=1)


def segment_masks(
    records: Sequence[Mapping[str, str]],
    labelled_records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    full_counts = Counter(record["REFNO"] for record in labelled_records)
    single = np.asarray(
        [full_counts[record["REFNO"]] == 1 for record in records], dtype=bool
    )
    paper_labels: dict[str, set[int]] = defaultdict(set)
    for record, label in zip(records, labels):
        paper_labels[record["REFNO"]].add(int(label))
    mixed = np.asarray(
        [len(paper_labels[record["REFNO"]]) > 1 for record in records], dtype=bool
    )
    target_counts = np.asarray(
        [full_counts[record["REFNO"]] for record in records], dtype=np.int16
    )
    return {
        "single_target": single,
        "multi_target": ~single,
        "two_targets": target_counts == 2,
        "three_or_four_targets": (target_counts >= 3) & (target_counts <= 4),
        "five_or_more_targets": target_counts >= 5,
        "multi_target_uniform_labels": (~single) & (~mixed),
        "multi_target_mixed_labels": (~single) & mixed,
    }


def paper_diagnostics(
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    prediction = probabilities >= 0.5
    by_refno: dict[str, list[int]] = defaultdict(list)
    for index in np.flatnonzero(mask):
        by_refno[records[int(index)]["REFNO"]].append(int(index))
    all_correct = 0
    constant = 0
    for indices in by_refno.values():
        all_correct += int(np.array_equal(prediction[indices], labels[indices]))
        constant += int(len(set(int(value) for value in prediction[indices])) == 1)
    papers = len(by_refno)
    return {
        "papers": papers,
        "all_targets_correct_papers": all_correct,
        "all_targets_correct_rate": round(all_correct / papers, 8) if papers else None,
        "constant_prediction_papers": constant,
        "constant_prediction_rate": round(constant / papers, 8) if papers else None,
    }


def strategy_report(
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    probabilities: np.ndarray,
    masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "overall": base.metric_record(labels, probabilities),
        "segments": {
            name: base.metric_record(labels[mask], probabilities[mask])
            for name, mask in masks.items()
        },
        "mixed_label_paper_diagnostics": paper_diagnostics(
            records,
            labels,
            probabilities,
            masks["multi_target_mixed_labels"],
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=all_oof.IDENTITY)
    parser.add_argument("--page-cache", type=Path, default=all_oof.PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--combined-features", type=Path, default=COMBINED_FEATURES)
    parser.add_argument("--combined-manifest", type=Path, default=COMBINED_MANIFEST)
    parser.add_argument("--fold-manifest", type=Path, default=FOLD_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=OOF_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_output = output_dir / FEATURE_OUTPUT.name
    feature_manifest_path = output_dir / FEATURE_MANIFEST.name
    candidate_freeze_path = output_dir / CANDIDATE_FREEZE.name
    oof_output = output_dir / OOF_OUTPUT.name
    prediction_receipt_path = output_dir / PREDICTION_RECEIPT.name
    report_output = output_dir / REPORT_OUTPUT.name

    runtime = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
    }
    if runtime != paper_v1.CANONICAL_RUNTIME:
        raise ValueError(
            f"canonical runtime required: expected={paper_v1.CANONICAL_RUNTIME}, observed={runtime}"
        )

    labelled_records = all_oof.load_identity(args.identity.resolve())
    records, coverage = all_oof.pdf_only_records(
        labelled_records, args.pdf_dir.resolve()
    )
    alignment_sha256 = all_oof.record_sha256(records)
    refnos = {record["REFNO"] for record in records}
    page_rows = load_page_cache_rows(args.page_cache.resolve(), refnos)
    grep_features = build_grep_entity_pair_features(records, page_rows)
    grep_matrix = np.asarray(grep_features["matrix"], dtype=np.float32)
    if grep_matrix.shape[0] != len(records):
        raise ValueError("GREP feature rows do not align with evaluated identity")

    combined_manifest = json.loads(
        args.combined_manifest.resolve().read_text(encoding="utf-8")
    )
    if combined_manifest.get("alignment_sha256") != alignment_sha256:
        raise ValueError("combined feature alignment hash changed")
    if base.file_sha256(args.combined_features.resolve()) != str(
        (combined_manifest.get("output") or {}).get("sha256")
    ):
        raise ValueError("combined feature archive hash changed")
    combined = np.load(args.combined_features.resolve())["combined"].astype(
        np.float32, copy=False
    )
    if combined.shape[0] != len(records) or not np.isfinite(combined).all():
        raise ValueError("combined feature matrix is invalid")
    expanded = np.concatenate((combined, grep_matrix), axis=1).astype(
        np.float32, copy=False
    )
    fold_ids = load_fold_ids(args.fold_manifest.resolve(), records)
    folds = fold_indices(fold_ids)

    atomic_npz(feature_output, grep=grep_matrix)
    target_count_distribution = Counter(
        int(item["paper_target_count"])
        for item in grep_features["row_metadata"]
        if isinstance(item, Mapping)
    )
    feature_manifest = {
        "schema_version": "r3_grep_entity_pair_features_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "paper_reference": PAPER,
        "adaptation": {
            "global_relations": "paper-level presence of R3 method, diagnostic, carrier, and bridge evidence",
            "entity_pairs": "each original-paper evidence occurrence paired with every canonical sibling target",
            "pair_interaction": "softmax responsibility, winner credit, sibling margins, and shared-evidence degree",
            "neural_architecture_copied": False,
        },
        "rows": len(records),
        "papers": len(refnos),
        "features": grep_matrix.shape[1],
        "feature_names": grep_features["feature_names"],
        "matrix_sha256": auxiliary.sha256_array(grep_matrix),
        "alignment_sha256": alignment_sha256,
        "target_count_distribution_by_row": {
            str(key): value for key, value in sorted(target_count_distribution.items())
        },
        "runtime_boundary": {
            "labels_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "allowed_sources": [
                "REFNO/RESULTNO/ROCKNAME identity",
                "original-paper page cache",
                "frozen R3 retrieval patterns",
            ],
        },
        "inputs": {
            "identity": {
                "path": str(args.identity.resolve()),
                "sha256": base.file_sha256(args.identity.resolve()),
            },
            "page_cache": {
                "path": str(args.page_cache.resolve()),
                "sha256": base.file_sha256(args.page_cache.resolve()),
            },
        },
        "output": {
            "path": str(feature_output),
            "sha256": base.file_sha256(feature_output),
        },
    }
    base.atomic_json(feature_manifest_path, feature_manifest)

    experts = [asdict(spec) for spec in model_specs()]
    candidate_freeze = {
        "schema_version": "r3_grep_entity_pair_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "primary_strategy": PRIMARY_STRATEGY,
        "paper_reference": PAPER,
        "experts": experts,
        "families": {
            "global": "existing combined label-free original-paper features",
            "pair": "GREP-inspired target/evidence interaction features",
            "expanded": "global and pair features concatenated",
        },
        "fixed_strategies": {
            "baseline_mean4": "uniform mean of four global experts",
            "grep_pair_mean4": "uniform mean of four entity-pair experts",
            "grep_expanded_mean4": "uniform mean of four concatenated-feature experts",
            "grep_global_pair_fusion": "sigmoid(mean(logit(global), logit(pair)))",
            "grep_global_expanded_blend": "uniform arithmetic mean of global and expanded mean4",
            "grep_global_pair_router": "global mean4 for single-target papers; global/pair logit fusion for multi-target papers",
        },
        "threshold": 0.5,
        "coverage": coverage,
        "feature_manifest": {
            "path": str(feature_manifest_path),
            "sha256": base.file_sha256(feature_manifest_path),
        },
        "combined_manifest": {
            "path": str(args.combined_manifest.resolve()),
            "sha256": base.file_sha256(args.combined_manifest.resolve()),
        },
        "fold_manifest": {
            "path": str(args.fold_manifest.resolve()),
            "sha256": base.file_sha256(args.fold_manifest.resolve()),
        },
        "runtime_boundary": {
            "labels_read_at_freeze": False,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_oof_or_evaluation_read": False,
            "refno_used_only_for_group_isolation_and_sibling_interaction": True,
            "resultno_used_only_for_alignment": True,
        },
        "canonical_runtime": runtime,
    }
    base.atomic_json(candidate_freeze_path, candidate_freeze)

    labels_by_resultno = all_oof.load_labels(
        args.targets.resolve(), {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    canonical_targets = [
        canonical_target_identity(record["ROCKNAME"]) for record in records
    ]
    probabilities = {
        f"{family}_{spec.name}": np.full(len(records), np.nan, dtype=np.float64)
        for family in ("baseline", "pair", "expanded")
        for spec in model_specs()
    }
    for fold, (train, test) in enumerate(folds, start=1):
        train_refnos = {records[int(index)]["REFNO"] for index in train}
        test_refnos = {records[int(index)]["REFNO"] for index in test}
        if train_refnos & test_refnos:
            raise AssertionError("GREP outer fold splits a REFNO")
        weights = base.inverse_target_multiplicity_weights(
            train, records, canonical_targets
        )
        fit_family(
            "baseline", combined, probabilities, train, test, labels, weights
        )
        fit_family("pair", grep_matrix, probabilities, train, test, labels, weights)
        fit_family("expanded", expanded, probabilities, train, test, labels, weights)
        print(
            f"GREP fold {fold}: train_rows={len(train)}, test_rows={len(test)}, "
            f"test_papers={len(test_refnos)}",
            flush=True,
        )
    if any(not np.isfinite(values).all() for values in probabilities.values()):
        raise AssertionError("GREP OOF probabilities are incomplete")

    probabilities["baseline_mean4"] = family_mean(probabilities, "baseline")
    probabilities["grep_pair_mean4"] = family_mean(probabilities, "pair")
    probabilities["grep_expanded_mean4"] = family_mean(probabilities, "expanded")
    probabilities["grep_global_pair_fusion"] = sigmoid(
        0.5 * logit(probabilities["baseline_mean4"])
        + 0.5 * logit(probabilities["grep_pair_mean4"])
    )
    probabilities["grep_global_expanded_blend"] = 0.5 * (
        probabilities["baseline_mean4"]
        + probabilities["grep_expanded_mean4"]
    )
    full_counts = Counter(record["REFNO"] for record in labelled_records)
    multi_target = np.asarray(
        [full_counts[record["REFNO"]] > 1 for record in records], dtype=bool
    )
    probabilities[PRIMARY_STRATEGY] = np.where(
        multi_target,
        probabilities["grep_global_pair_fusion"],
        probabilities["baseline_mean4"],
    )
    all_oof.write_array_archive(oof_output, probabilities)

    receipt = {
        "schema_version": "r3_grep_entity_pair_prediction_receipt_v1",
        "status": "OOF_PREDICTIONS_FROZEN_BEFORE_AGGREGATE_SCORING",
        "primary_strategy": PRIMARY_STRATEGY,
        "candidate_freeze_sha256": base.file_sha256(candidate_freeze_path),
        "oof": {"path": str(oof_output), "sha256": base.file_sha256(oof_output)},
        "each_row_predicted_by_model_excluding_its_refno": True,
        "rows": [
            {
                "row_index": index,
                **record,
                "fold": int(fold_ids[index]),
                "probability": float(probabilities[PRIMARY_STRATEGY][index]),
                "prediction": int(probabilities[PRIMARY_STRATEGY][index] >= 0.5),
            }
            for index, record in enumerate(records)
        ],
    }
    base.atomic_json(prediction_receipt_path, receipt)

    masks = segment_masks(records, labelled_records, labels)
    strategy_names = (
        "baseline_mean4",
        "grep_pair_mean4",
        "grep_expanded_mean4",
        "grep_global_pair_fusion",
        "grep_global_expanded_blend",
        PRIMARY_STRATEGY,
    )
    reports = {
        name: strategy_report(records, labels, probabilities[name], masks)
        for name in strategy_names
    }
    baseline = reports["baseline_mean4"]
    primary = reports[PRIMARY_STRATEGY]
    old_baseline = np.load(BASELINE_OOF)["mean4"].astype(np.float64)
    baseline_max_abs_difference = float(
        np.max(np.abs(old_baseline - probabilities["baseline_mean4"]))
    )
    report = {
        "schema_version": "r3_grep_entity_pair_grouped_oof_v1",
        "status": "complete_development_oof",
        "paper_reference": PAPER,
        "primary_strategy": PRIMARY_STRATEGY,
        "claim_boundary": {
            "internal_refno_grouped_oof": True,
            "independent_external_validation": False,
            "feature_and_candidate_artifacts_frozen_before_label_access": True,
            "pmagresult_or_row_rockmag_used_as_feature": False,
            "prior_oof_or_evaluation_used_as_model_input": False,
            "paper_method_adapted_not_architecture_reproduced": True,
        },
        "baseline_reproduction": {
            "stored_oof_path": str(BASELINE_OOF),
            "maximum_absolute_probability_difference": baseline_max_abs_difference,
            "exact_within_1e_12": baseline_max_abs_difference <= 1e-12,
            "within_numeric_tolerance_1e_7": baseline_max_abs_difference <= 1e-7,
        },
        "strategies": reports,
        "primary_delta_vs_baseline": {
            "overall_accuracy": round(
                primary["overall"]["accuracy"] - baseline["overall"]["accuracy"],
                8,
            ),
            "multi_target_accuracy": round(
                primary["segments"]["multi_target"]["accuracy"]
                - baseline["segments"]["multi_target"]["accuracy"],
                8,
            ),
            "mixed_label_accuracy": round(
                primary["segments"]["multi_target_mixed_labels"]["accuracy"]
                - baseline["segments"]["multi_target_mixed_labels"]["accuracy"],
                8,
            ),
        },
        "artifacts": {
            "feature_manifest": {
                "path": str(feature_manifest_path),
                "sha256": base.file_sha256(feature_manifest_path),
            },
            "candidate_freeze": {
                "path": str(candidate_freeze_path),
                "sha256": base.file_sha256(candidate_freeze_path),
            },
            "prediction_receipt": {
                "path": str(prediction_receipt_path),
                "sha256": base.file_sha256(prediction_receipt_path),
            },
            "oof_probabilities": {
                "path": str(oof_output),
                "sha256": base.file_sha256(oof_output),
            },
        },
    }
    base.atomic_json(report_output, report)
    print(
        json.dumps(
            {
                "report": str(report_output),
                "baseline": baseline["overall"],
                "primary": primary["overall"],
                "primary_multi_target": primary["segments"]["multi_target"],
                "primary_mixed_label": primary["segments"][
                    "multi_target_mixed_labels"
                ],
                "delta": report["primary_delta_vs_baseline"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
