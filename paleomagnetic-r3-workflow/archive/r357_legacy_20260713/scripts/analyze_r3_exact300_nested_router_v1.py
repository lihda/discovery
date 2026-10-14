#!/usr/bin/env python3
"""Aggregate diagnostic audit for the completed exact300 nested-router v1.

This script intentionally reads the completed OOF artifact and R3 labels, so
its output is diagnostic-only and must never be consumed as a runtime feature,
candidate selector, or calibration input.  It emits no row identifiers.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/nested_router_v1"
DEFAULT_REPORT = DEFAULT_DIR / "report.json"
DEFAULT_OOF = DEFAULT_DIR / "oof_probabilities.npz"
DEFAULT_OUTPUT = DEFAULT_DIR / "diagnostic.json"
ROUTER_NAME = "router_loss_hgb_hard"


def rounded(value: float) -> float:
    return round(float(value), 8)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--oof", type=Path, default=DEFAULT_OOF)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report_path = args.report.resolve()
    oof_path = args.oof.resolve()
    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    fold_manifest = args.fold_manifest.resolve()
    targets = args.targets.resolve()
    pdf_dir = args.pdf_dir.resolve()
    output = args.output.resolve()

    if base.file_sha256(exact_identity) != auxiliary.EXACT_IDENTITY_SHA256:
        raise ValueError("exact300 identity hash mismatch")
    if base.file_sha256(full_identity) != auxiliary.FULL_IDENTITY_SHA256:
        raise ValueError("full identity hash mismatch")
    if base.file_sha256(fold_manifest) != auxiliary.FOLD_MANIFEST_SHA256:
        raise ValueError("fold manifest hash mismatch")
    if base.file_sha256(targets) != auxiliary.TARGETS_SHA256:
        raise ValueError("target workbook hash mismatch")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema_version")
        != "r3_exact300_nested_router_grouped_oof_v1"
        or report.get("status") != "complete"
        or report.get("mode") != "full_five_fold"
        or report.get("scope", {}).get("evaluated_rows") != 300
        or report.get("scope", {}).get("full_coverage") is not True
        or report.get("compliance_audit", {}).get(
            "canonical_outer_fold_manifest_sha256"
        )
        != auxiliary.FOLD_MANIFEST_SHA256
        or report.get("artifacts", {})
        .get("oof_probabilities", {})
        .get("sha256")
        != base.file_sha256(oof_path)
    ):
        raise ValueError("v1 report does not prove a canonical complete run")
    for outer in report.get("outer_fold_audit", []):
        if (
            outer.get("train_test_refno_overlap") != 0
            or outer.get("all_outer_test_refno_rows_excluded") is not True
            or any(
                inner.get("fit_validation_refno_overlap") != 0
                or inner.get("outer_test_refno_overlap") != 0
                for inner in outer.get("inner_cross_fitting", [])
            )
        ):
            raise ValueError("v1 report contains a failed REFNO isolation audit")

    full_records, _ = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, _ = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    fold_ids = auxiliary.load_fold_manifest(fold_manifest, exact_records)
    labels_by_resultno = base.load_training_labels(
        targets, {record["RESULTNO"] for record in full_records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )

    with np.load(oof_path, allow_pickle=False) as payload:
        expert_names = sorted(name for name in payload.files if name.startswith("expert_"))
        if ROUTER_NAME not in payload.files or len(expert_names) != 6:
            raise ValueError("v1 OOF does not contain the frozen expert/router family")
        expert_probability = np.column_stack(
            [np.asarray(payload[name], dtype=np.float64) for name in expert_names]
        )
        router_probability = np.asarray(payload[ROUTER_NAME], dtype=np.float64)
    if (
        expert_probability.shape != (300, 6)
        or router_probability.shape != (300,)
        or not np.isfinite(expert_probability).all()
        or not np.isfinite(router_probability).all()
    ):
        raise ValueError("v1 OOF arrays are incomplete")

    distances = np.abs(expert_probability - router_probability[:, None])
    selected = distances.argmin(axis=1)
    if float(distances.min(axis=1).max()) > 1e-7:
        raise ValueError("hard router probability does not match a declared expert")
    expert_decisions = expert_probability >= 0.5
    router_decision = router_probability >= 0.5
    errors = router_decision != labels
    any_expert_correct = np.any(expert_decisions == labels[:, None], axis=1)
    all_experts_agree = np.all(
        expert_decisions == expert_decisions[:, [0]], axis=1
    )
    actual_loss = (expert_probability - labels[:, None]) ** 2
    selected_loss = actual_loss[np.arange(len(labels)), selected]
    best_actual_loss = actual_loss.min(axis=1)
    regret = selected_loss - best_actual_loss
    confidence = np.abs(router_probability - 0.5)
    vote_fraction = expert_decisions.mean(axis=1)
    paper_counts = Counter(record["REFNO"] for record in exact_records)

    selected_summary: dict[str, Any] = {}
    correction_summary: dict[str, int] = {}
    for column, name in enumerate(expert_names):
        chosen = selected == column
        selected_summary[name] = {
            "selected_rows": int(chosen.sum()),
            "selected_errors": int((chosen & errors).sum()),
            "conditional_error_rate": rounded(errors[chosen].mean())
            if chosen.any()
            else 0.0,
        }
        correction_summary[name] = int(
            (errors & (expert_decisions[:, column] == labels)).sum()
        )

    confidence_bands = []
    for lower, upper in ((0.0, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.50)):
        member = (confidence >= lower) & (confidence < upper)
        confidence_bands.append(
            {
                "absolute_margin_lower": lower,
                "absolute_margin_upper": upper,
                "rows": int(member.sum()),
                "errors": int((member & errors).sum()),
                "error_rate": rounded(errors[member].mean()) if member.any() else 0.0,
            }
        )

    vote_summary = []
    for value in sorted(set(float(item) for item in vote_fraction)):
        member = np.isclose(vote_fraction, value)
        vote_summary.append(
            {
                "positive_vote_fraction": rounded(value),
                "rows": int(member.sum()),
                "errors": int((member & errors).sum()),
            }
        )

    diagnostic = {
        "schema_version": "r3_exact300_nested_router_v1_diagnostic_v1",
        "status": "complete",
        "usage_boundary": {
            "diagnostic_only": True,
            "contains_oof_derived_statistics": True,
            "contains_label_derived_statistics": True,
            "contains_row_identifiers": False,
            "permitted_as_runtime_feature": False,
            "permitted_for_candidate_or_threshold_selection": False,
        },
        "inputs": {
            "report": {"path": str(report_path), "sha256": base.file_sha256(report_path)},
            "oof": {"path": str(oof_path), "sha256": base.file_sha256(oof_path)},
            "fold_manifest_sha256": auxiliary.FOLD_MANIFEST_SHA256,
            "targets_sha256": auxiliary.TARGETS_SHA256,
        },
        "canonical_audit": {
            "rows": 300,
            "outer_folds": len(report.get("outer_fold_audit", [])),
            "outer_refno_overlap_zero": True,
            "inner_refno_overlap_zero": True,
            "oof_full_coverage": True,
        },
        "router": {
            "name": ROUTER_NAME,
            "errors": int(errors.sum()),
            "accuracy": rounded((~errors).mean()),
            "false_positive": int((router_decision & (labels == 0)).sum()),
            "false_negative": int((~router_decision & (labels == 1)).sum()),
        },
        "error_decomposition": {
            "all_six_experts_wrong": int((errors & ~any_expert_correct).sum()),
            "at_least_one_expert_could_fix": int((errors & any_expert_correct).sum()),
            "all_six_experts_agree_on_wrong_decision": int(
                (errors & all_experts_agree).sum()
            ),
            "six_expert_oracle_errors": int((~any_expert_correct).sum()),
            "six_expert_oracle_accuracy": rounded(any_expert_correct.mean()),
        },
        "selected_expert": selected_summary,
        "expert_corrections_available_on_router_errors": correction_summary,
        "confidence": {
            "mean_absolute_margin_on_errors": rounded(confidence[errors].mean()),
            "mean_absolute_margin_on_correct_rows": rounded(
                confidence[~errors].mean()
            ),
            "bands": confidence_bands,
            "expert_vote_summary": vote_summary,
        },
        "routing_loss": {
            "observed_loss": "squared probability error (p-y)^2",
            "mean_selected_loss_regret_all_rows": rounded(regret.mean()),
            "mean_selected_loss_regret_on_router_errors": rounded(regret[errors].mean()),
            "router_errors_where_selected_expert_has_minimum_actual_loss": int(
                (errors & np.isclose(selected_loss, best_actual_loss)).sum()
            ),
            "predicted_route_loss_matrix_persisted": False,
            "interpretation": (
                "The hard output identifies the selected expert exactly, but v1 did "
                "not persist each expert's predicted route loss; diagnostics therefore "
                "measure realized loss regret rather than calibration of latent losses."
            ),
        },
        "structure": {
            "errors_by_fold": {
                str(fold): int((errors & (fold_ids == fold)).sum())
                for fold in range(1, auxiliary.N_SPLITS + 1)
            },
            "errors_by_exact300_paper_multiplicity": dict(
                sorted(
                    Counter(
                        str(paper_counts[exact_records[int(index)]["REFNO"]])
                        for index in np.flatnonzero(errors)
                    ).items()
                )
            ),
            "errors_in_multirow_exact300_papers": int(
                sum(
                    paper_counts[exact_records[int(index)]["REFNO"]] > 1
                    for index in np.flatnonzero(errors)
                )
            ),
        },
    }
    base.atomic_json(output, diagnostic)
    print(json.dumps(diagnostic, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
