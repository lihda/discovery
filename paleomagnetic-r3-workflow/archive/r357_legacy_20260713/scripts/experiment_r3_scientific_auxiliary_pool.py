#!/usr/bin/env python3
"""Leakage-safe exact300 fold-1 SciNCL training with a 2,172-row auxiliary pool."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_scientific_embeddings as scientific  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_scientific_finetune as exact_scincl  # noqa: E402


OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/scincl_auxiliary_pool_v1"
DEFAULT_EXACT_IDENTITY = ROOT / "tmp/r3_exact_300_resultno_v2/identity.xlsx"
DEFAULT_FULL_IDENTITY = ROOT / "tmp/r3_distilled_training_preflight/identity.xlsx"
DEFAULT_INPUTS = OUTPUT_DIR / "full_pool_inputs.npz"
DEFAULT_INPUT_MANIFEST = OUTPUT_DIR / "full_pool_inputs.manifest.json"
DEFAULT_PROTOCOL = OUTPUT_DIR / "fold1_protocol.json"
DEFAULT_FOLD_MANIFEST = ROOT / "tmp/r3_exact_300_resultno_v2/refno_fold_manifest.json"
DEFAULT_SCREENING_PROTOCOL = ROOT / "tmp/r3_exact_300_resultno_v2/screening_protocol.json"
DEFAULT_BASELINE_REPORT = ROOT / "tmp/r3_exact_300_resultno_v2/baseline.json"
DEFAULT_MODEL = ROOT / "tmp/models/scincl"
DEFAULT_REPORT = OUTPUT_DIR / "fold1_smoke.json"
DEFAULT_PROBABILITIES = OUTPUT_DIR / "fold1_probabilities.npz"

EXPECTED_EXACT_ROWS = 300
EXPECTED_EXACT_GROUPS = 258
EXPECTED_FULL_ROWS = 2172
EXPECTED_FULL_GROUPS = 1174
EXPECTED_COMPONENTS = exact_scincl.EXPECTED_COMPONENTS
N_SPLITS = 5
AUX_RANDOM_STATE = 7357
MAX_SELECTION_EPOCHS = 3
BATCH_SIZE = 16
EVALUATION_BATCH_SIZE = 32
LAST_UNFROZEN_LAYERS = 2
ENCODER_LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
SMOKE_MINIMUM_ABSOLUTE_GAIN = 0.02
THRESHOLD_GRID = tuple(float(value) for value in np.arange(0.15, 0.851, 0.025))
INNER_VALIDATION_FRACTION = 0.20
INNER_SPLIT_SALT = "r3-scincl-auxiliary-inner-v1"

EXACT_IDENTITY_SHA256 = auxiliary.EXACT_IDENTITY_SHA256
FULL_IDENTITY_SHA256 = auxiliary.FULL_IDENTITY_SHA256
SCREENING_PROTOCOL_SHA256 = auxiliary.SCREENING_PROTOCOL_SHA256
TARGETS_SHA256 = auxiliary.TARGETS_SHA256
PAGE_CACHE_SHA256 = auxiliary.PAGE_CACHE_SHA256
TAXONOMY_SHA256 = auxiliary.TAXONOMY_SHA256
POSITIVE_KB_SHA256 = auxiliary.POSITIVE_KB_SHA256
BASELINE_REPORT_SHA256 = "5b2b80228af3985bbaaba40dfdbeea5e84e4d58d1dfe8af7aaed7c20a20bc9bb"
SCINCL_WEIGHT_SHA256 = exact_scincl.SCINCL_WEIGHT_SHA256
FOLD_MANIFEST_SHA256 = "8f6ab8f1a071878e14d4db8e19a7ec0519ed612be107b0f0fdf0489dd59aaf2d"
FOLD_ID_SHA256 = "28552aa3627c2c808ff49ac50c7c9d67a6f32daaad0e9aff5ae6a96fbe61f08f"
EXACT_ALIGNMENT_SHA256 = "ea3b591bdcaa26d8bfa86aa379c070626c436b889b7ea79cc1385611c6f99126"
AUX_PROTOCOL_SHA256 = "d0a822b934fb1405ce6721cc54233a3d45a8927ae8b3d98d9b499fa71384dfff"


def file_audit(path: Path, expected_sha256: str, name: str) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"missing frozen {name}: {path}")
    observed = base.file_sha256(path)
    if observed != expected_sha256:
        raise ValueError(f"frozen {name} hash mismatch")
    return {"path": str(path), "sha256": observed}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def load_canonical_fold_ids(
    path: Path,
    exact_records: Sequence[Mapping[str, str]],
) -> np.ndarray:
    if base.file_sha256(path) != FOLD_MANIFEST_SHA256:
        raise ValueError("canonical exact300 fold manifest hash mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    if (
        manifest.get("schema_version") != "r3_exact300_label_free_refno_folds_v1"
        or manifest.get("status") != "FROZEN_WITHOUT_LABEL_ACCESS"
        or manifest.get("input", {}).get("identity_sha256") != EXACT_IDENTITY_SHA256
        or manifest.get("input", {}).get("alignment_sha256")
        != EXACT_ALIGNMENT_SHA256
        or manifest.get("fold_id_sha256") != FOLD_ID_SHA256
        or any(
            boundary.get(name) is not False
            for name in (
                "labels_read",
                "targets_workbook_read",
                "pdf_text_read",
                "pmagresult_or_rockmag_read",
                "prior_predictions_or_evaluations_read",
            )
        )
    ):
        raise ValueError("canonical exact300 fold manifest boundary mismatch")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != len(exact_records):
        raise ValueError("canonical exact300 fold manifest row count mismatch")
    fold_ids = np.zeros(len(exact_records), dtype=np.int8)
    for index, (item, record) in enumerate(zip(rows, exact_records)):
        if (
            item.get("row_index") != index
            or str(item.get("REFNO")) != record["REFNO"]
            or str(item.get("RESULTNO")) != record["RESULTNO"]
            or int(item.get("fold", 0)) not in range(1, N_SPLITS + 1)
        ):
            raise ValueError("canonical exact300 fold manifest lost row alignment")
        fold_ids[index] = int(item["fold"])
    if hashlib.sha256(bytes(int(value) for value in fold_ids)).hexdigest() != FOLD_ID_SHA256:
        raise ValueError("canonical exact300 fold-id digest mismatch")
    if Counter(int(value) for value in fold_ids) != Counter(
        {fold: 60 for fold in range(1, N_SPLITS + 1)}
    ):
        raise ValueError("canonical exact300 fold sizes differ from 60 rows")
    group_folds: dict[str, set[int]] = {}
    for record, fold in zip(exact_records, fold_ids):
        group_folds.setdefault(record["REFNO"], set()).add(int(fold))
    if any(len(values) != 1 for values in group_folds.values()):
        raise ValueError("canonical exact300 manifest splits one REFNO")
    return fold_ids


def deterministic_inner_split(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    """Choose a label-free, version-independent REFNO validation partition."""

    counts = Counter(records[int(index)]["REFNO"] for index in indices)
    ranked_groups = sorted(
        counts,
        key=lambda refno: (
            hashlib.sha256(f"{INNER_SPLIT_SALT}\0{refno}".encode("utf-8")).digest(),
            refno,
        ),
    )
    target_rows = max(1, round(len(indices) * INNER_VALIDATION_FRACTION))
    validation_groups: set[str] = set()
    validation_rows = 0
    for refno in ranked_groups:
        if validation_rows >= target_rows:
            break
        validation_groups.add(refno)
        validation_rows += counts[refno]
    validation = np.asarray(
        [index for index in indices if records[int(index)]["REFNO"] in validation_groups],
        dtype=np.int64,
    )
    fit = np.asarray(
        [index for index in indices if records[int(index)]["REFNO"] not in validation_groups],
        dtype=np.int64,
    )
    if len(fit) == 0 or len(validation) == 0:
        raise ValueError("deterministic auxiliary inner split is empty")
    if (
        {records[int(index)]["REFNO"] for index in fit}
        & {records[int(index)]["REFNO"] for index in validation}
    ):
        raise AssertionError("REFNO leaked across deterministic auxiliary inner split")
    return fit, validation


def paper_class_balanced_weights(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Give each paper unit mass, then equalize weighted class mass."""

    if len(indices) == 0 or len(set(int(labels[int(index)]) for index in indices)) != 2:
        raise ValueError("class-balanced auxiliary training requires both labels")
    paper_counts = Counter(records[int(index)]["REFNO"] for index in indices)
    paper_weights = np.asarray(
        [1.0 / paper_counts[records[int(index)]["REFNO"]] for index in indices],
        dtype=np.float64,
    )
    selected_labels = labels[indices].astype(np.int8)
    raw_class_mass = {
        value: float(paper_weights[selected_labels == value].sum()) for value in (0, 1)
    }
    class_factor = {value: 0.5 / raw_class_mass[value] for value in (0, 1)}
    weights = paper_weights * np.asarray(
        [class_factor[int(value)] for value in selected_labels], dtype=np.float64
    )
    weights *= len(weights) / weights.sum()
    balanced_mass = {
        value: float(weights[selected_labels == value].sum()) for value in (0, 1)
    }
    return weights.astype(np.float32), {
        "rows": int(len(indices)),
        "papers": len(paper_counts),
        "label_rows": dict(sorted(Counter(int(value) for value in selected_labels).items())),
        "raw_inverse_paper_class_mass": {
            str(key): round(value, 8) for key, value in raw_class_mass.items()
        },
        "balanced_class_mass": {
            str(key): round(value, 8) for key, value in balanced_mass.items()
        },
        "weight_sum": round(float(weights.sum()), 8),
    }


def weighted_loader(
    indices: np.ndarray,
    *,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    batch_size: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    weights, audit = paper_class_balanced_weights(indices, records, labels)
    dataset = TensorDataset(
        torch.from_numpy(input_ids[indices]),
        torch.from_numpy(attention_mask[indices]),
        torch.from_numpy(labels[indices].astype(np.int64)),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        ),
        audit,
    )


def threshold_candidates(probabilities: np.ndarray) -> list[float]:
    values = np.unique(np.asarray(probabilities, dtype=np.float64))
    midpoints = (
        ((values[:-1] + values[1:]) / 2.0).tolist() if len(values) > 1 else []
    )
    return sorted(
        {
            round(float(value), 12)
            for value in (*THRESHOLD_GRID, *midpoints, 0.5)
            if 0.0 < float(value) < 1.0
        }
    )


def select_inner_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, Any], bool, int]:
    """Optimize inner accuracy while forbidding one-class validation decisions."""

    candidates = threshold_candidates(probabilities)
    metrics = [base.metric_record(labels, probabilities, threshold=value) for value in candidates]
    eligible = [
        item for item in metrics if 0 < int(item["predicted_positive"]) < len(labels)
    ]
    if not eligible:
        fallback = base.metric_record(labels, probabilities, threshold=0.5)
        return 0.5, fallback, False, len(candidates)
    selected = max(
        eligible,
        key=lambda item: (
            item["accuracy"],
            item["balanced_accuracy"],
            item["macro_f1"],
            -abs(float(item["threshold"]) - 0.5),
        ),
    )
    return float(selected["threshold"]), selected, True, len(candidates)


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    full_identity: Path,
    inputs: Path,
    page_cache: Path,
    taxonomy: Path,
    positive_kb: Path,
    records: Sequence[Mapping[str, str]],
) -> None:
    if manifest.get("schema_version") != "r3_scientific_finetune_inputs_v2":
        raise ValueError("auxiliary text manifest must use the positive-KB v2 schema")
    boundary = manifest.get("runtime_boundary", {})
    for name in (
        "labels_read",
        "pmagresult_or_rockmag_read",
        "prior_predictions_or_evaluations_read",
        "cache_contains_record_identifiers",
    ):
        if boundary.get(name) is not False:
            raise ValueError("auxiliary text manifest violates the source boundary")
    expected_inputs = {
        "identity": (full_identity, FULL_IDENTITY_SHA256),
        "page_cache": (page_cache, PAGE_CACHE_SHA256),
        "taxonomy": (taxonomy, TAXONOMY_SHA256),
        "positive_kb": (positive_kb, POSITIVE_KB_SHA256),
    }
    for name, (path, expected_hash) in expected_inputs.items():
        item = manifest.get("inputs", {}).get(name, {})
        if item.get("sha256") != expected_hash or base.file_sha256(path) != expected_hash:
            raise ValueError(f"auxiliary text manifest {name} hash mismatch")
    if (
        manifest.get("inputs", {}).get("alignment_sha256")
        != scientific.alignment_sha256(records)
        or manifest.get("output", {}).get("sha256") != base.file_sha256(inputs)
        or manifest.get("scope", {}).get("component_rows") != EXPECTED_FULL_ROWS
    ):
        raise ValueError("auxiliary text cache alignment or output hash mismatch")
    knowledge = manifest.get("knowledge", {})
    if (
        knowledge.get("positive_kb_status") != "loaded"
        or knowledge.get("runtime_pattern_count") != 80
        or knowledge.get("runtime_pattern_counts_by_source", {}).get(
            "positive_rockmag_expert_kb"
        )
        != 32
    ):
        raise ValueError("auxiliary text cache did not use the frozen positive expert KB")


def validate_protocol(
    path: Path,
    *,
    exact_identity: Path,
    full_identity: Path,
    inputs: Path,
    manifest: Path,
    fold_manifest: Path,
    screening_protocol: Path,
    baseline_report: Path,
    targets: Path,
    page_cache: Path,
    taxonomy: Path,
    positive_kb: Path,
    model: Path,
) -> dict[str, Any]:
    if base.file_sha256(path) != AUX_PROTOCOL_SHA256:
        raise ValueError("auxiliary SciNCL protocol hash mismatch")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema_version") != "r3_scientific_auxiliary_pool_fold1_protocol_v1"
        or protocol.get("status") != "FROZEN_BEFORE_OUTER_TEST"
    ):
        raise ValueError("auxiliary SciNCL protocol schema or status mismatch")
    expected_hashes = {
        "exact_identity": (exact_identity, EXACT_IDENTITY_SHA256),
        "full_identity": (full_identity, FULL_IDENTITY_SHA256),
        "text_cache": (inputs, base.file_sha256(inputs)),
        "text_manifest": (manifest, base.file_sha256(manifest)),
        "fold_manifest": (fold_manifest, FOLD_MANIFEST_SHA256),
        "screening_protocol": (screening_protocol, SCREENING_PROTOCOL_SHA256),
        "baseline_report": (baseline_report, BASELINE_REPORT_SHA256),
        "targets": (targets, TARGETS_SHA256),
        "page_cache": (page_cache, PAGE_CACHE_SHA256),
        "taxonomy": (taxonomy, TAXONOMY_SHA256),
        "positive_kb": (positive_kb, POSITIVE_KB_SHA256),
        "model_weights": (model / "model.safetensors", SCINCL_WEIGHT_SHA256),
    }
    for name, (input_path, expected_hash) in expected_hashes.items():
        item = protocol.get("inputs", {}).get(name, {})
        if base.file_sha256(input_path) != expected_hash or item.get("sha256") != expected_hash:
            raise ValueError(f"auxiliary SciNCL protocol input mismatch: {name}")
    training = protocol.get("training", {})
    expected_training = {
        "random_state": AUX_RANDOM_STATE,
        "maximum_selection_epochs": MAX_SELECTION_EPOCHS,
        "batch_size": BATCH_SIZE,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "last_unfrozen_encoder_layers": LAST_UNFROZEN_LAYERS,
        "encoder_learning_rate": ENCODER_LEARNING_RATE,
        "head_learning_rate": HEAD_LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "paper_inverse_multiplicity_weight": True,
        "class_balanced_weighted_loss": True,
        "inner_selected_noncollapsed_threshold": True,
        "deterministic_label_free_inner_split": True,
        "inner_split_salt": INNER_SPLIT_SALT,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
    }
    if any(training.get(name) != value for name, value in expected_training.items()):
        raise ValueError("auxiliary SciNCL runtime differs from the frozen protocol")
    scope = protocol.get("scope", {})
    if (
        scope.get("evaluation_rows") != EXPECTED_EXACT_ROWS
        or scope.get("evaluation_groups") != EXPECTED_EXACT_GROUPS
        or scope.get("auxiliary_rows") != EXPECTED_FULL_ROWS
        or scope.get("auxiliary_groups") != EXPECTED_FULL_GROUPS
        or scope.get("first_stage_fold") != 1
    ):
        raise ValueError("auxiliary SciNCL protocol scope mismatch")
    return protocol


def run_fold1(
    *,
    full_records: Sequence[dict[str, str]],
    exact_records: Sequence[dict[str, str]],
    full_labels: np.ndarray,
    exact_labels: np.ndarray,
    exact_fold_ids: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    model_path: Path,
    device: Any,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    full_lookup = {
        auxiliary.row_key(record): index for index, record in enumerate(full_records)
    }
    exact_test = np.flatnonzero(exact_fold_ids == 1).astype(np.int64)
    outer_train, test_full, test_refnos = auxiliary.auxiliary_split(
        full_records, exact_records, exact_test, full_lookup
    )
    full_groups = np.asarray([record["REFNO"] for record in full_records], dtype=object)
    fit_index, validation_index = deterministic_inner_split(
        outer_train, full_records
    )
    test_groups = set(test_refnos)
    fit_groups = set(full_groups[fit_index])
    validation_groups = set(full_groups[validation_index])
    outer_train_groups = set(full_groups[outer_train])
    if (
        fit_groups & validation_groups
        or fit_groups & test_groups
        or validation_groups & test_groups
        or outer_train_groups & test_groups
    ):
        raise AssertionError("REFNO leaked across auxiliary SciNCL partitions")

    loader, fit_weight_audit = weighted_loader(
        fit_index,
        records=full_records,
        labels=full_labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_size=BATCH_SIZE,
        seed=AUX_RANDOM_STATE + 1,
    )
    set_seed(AUX_RANDOM_STATE + 1)
    model, _ = exact_scincl.make_model(model_path, LAST_UNFROZEN_LAYERS, device)
    optimizer, scheduler = exact_scincl.optimizer_and_scheduler(
        model,
        loader,
        epochs=MAX_SELECTION_EPOCHS,
        encoder_learning_rate=ENCODER_LEARNING_RATE,
        head_learning_rate=HEAD_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    epoch_records: list[dict[str, Any]] = []
    best_key: tuple[Any, ...] | None = None
    best_epoch = 0
    best_threshold = 0.5
    for epoch in range(1, MAX_SELECTION_EPOCHS + 1):
        train_loss = exact_scincl.train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            device=device,
            last_layers=LAST_UNFROZEN_LAYERS,
        )
        validation_probability, validation_loss = exact_scincl.evaluate_model(
            model,
            input_ids,
            attention_mask,
            full_labels,
            validation_index,
            device=device,
            batch_size=EVALUATION_BATCH_SIZE,
        )
        threshold, metric, threshold_stable, candidate_count = select_inner_threshold(
            full_labels[validation_index], validation_probability
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 8),
            "validation_loss": round(float(validation_loss), 8),
            "selected_threshold": round(float(threshold), 8),
            "threshold_candidate_count": candidate_count,
            "threshold_noncollapsed": threshold_stable,
            "validation": metric,
        }
        epoch_records.append(epoch_record)
        key = (
            int(threshold_stable),
            metric["accuracy"],
            metric["balanced_accuracy"],
            metric["macro_f1"],
            -float(validation_loss),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            best_threshold = threshold
        print(
            f"aux fold1 epoch {epoch}/{MAX_SELECTION_EPOCHS}: "
            f"loss={train_loss:.4f}, val_acc={metric['accuracy']:.4f}, "
            f"threshold={threshold:.4f}, noncollapsed={threshold_stable}",
            flush=True,
        )

    del model, optimizer, scheduler, loader
    if getattr(torch, "mps", None) is not None:
        torch.mps.empty_cache()
    refit_loader, refit_weight_audit = weighted_loader(
        outer_train,
        records=full_records,
        labels=full_labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_size=BATCH_SIZE,
        seed=AUX_RANDOM_STATE + 1001,
    )
    set_seed(AUX_RANDOM_STATE + 1001)
    model, parameter_audit = exact_scincl.make_model(
        model_path, LAST_UNFROZEN_LAYERS, device
    )
    optimizer, scheduler = exact_scincl.optimizer_and_scheduler(
        model,
        refit_loader,
        epochs=best_epoch,
        encoder_learning_rate=ENCODER_LEARNING_RATE,
        head_learning_rate=HEAD_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    refit_records: list[dict[str, Any]] = []
    for epoch in range(1, best_epoch + 1):
        train_loss = exact_scincl.train_epoch(
            model,
            refit_loader,
            optimizer,
            scheduler,
            device=device,
            last_layers=LAST_UNFROZEN_LAYERS,
        )
        refit_records.append({"epoch": epoch, "train_loss": round(train_loss, 8)})
        print(
            f"aux fold1 refit {epoch}/{best_epoch}: loss={train_loss:.4f}",
            flush=True,
        )
    test_probability, test_loss = exact_scincl.evaluate_model(
        model,
        input_ids,
        attention_mask,
        full_labels,
        test_full,
        device=device,
        batch_size=EVALUATION_BATCH_SIZE,
    )
    test_labels = exact_labels[exact_test]
    selected_metric = base.metric_record(
        test_labels, test_probability, threshold=best_threshold
    )
    default_metric = base.metric_record(test_labels, test_probability, threshold=0.5)
    decision = (test_probability >= best_threshold).astype(np.int8)
    stable = bool(
        np.isfinite(test_probability).all()
        and math.isfinite(test_loss)
        and 0 < int(decision.sum()) < len(decision)
    )
    excluded_rows = sum(record["REFNO"] in test_groups for record in full_records)
    audit = {
        "fold": 1,
        "exact_test_rows": int(len(exact_test)),
        "exact_test_groups": len(test_groups),
        "auxiliary_candidate_rows": len(full_records),
        "outer_train_rows": int(len(outer_train)),
        "outer_train_groups": len(outer_train_groups),
        "excluded_rows_for_outer_test_groups": int(excluded_rows),
        "train_test_refno_overlap": len(outer_train_groups & test_groups),
        "inner_fit_rows": int(len(fit_index)),
        "inner_fit_groups": len(fit_groups),
        "inner_validation_rows": int(len(validation_index)),
        "inner_validation_groups": len(validation_groups),
        "inner_fit_validation_refno_overlap": len(fit_groups & validation_groups),
        "selected_epoch": int(best_epoch),
        "selected_threshold": round(float(best_threshold), 8),
        "selection_rule": (
            "inner REFNO-grouped validation; require both predicted classes, then "
            "accuracy, balanced accuracy, macro-F1, loss"
        ),
        "inner_selection_epochs": epoch_records,
        "fit_weight_audit": fit_weight_audit,
        "refit_weight_audit": refit_weight_audit,
        "outer_train_refit_epochs": refit_records,
        "outer_test_loss": round(float(test_loss), 8),
        "outer_test_selected_threshold": selected_metric,
        "outer_test_default_threshold_diagnostic": default_metric,
        "training_stable_and_test_noncollapsed": stable,
        "parameters": parameter_audit,
    }
    del model, optimizer, scheduler, refit_loader
    if getattr(torch, "mps", None) is not None:
        torch.mps.empty_cache()
    return test_probability, decision, audit


def write_fold1_artifact(
    path: Path,
    probability: np.ndarray,
    decision: np.ndarray,
) -> None:
    if probability.shape != (60,) or decision.shape != (60,):
        raise ValueError("auxiliary fold1 artifact must contain exactly 60 predictions")
    if not np.isfinite(probability).all() or not np.isin(decision, (0, 1)).all():
        raise ValueError("auxiliary fold1 artifact contains invalid values")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            fold_1_probability=probability.astype(np.float32),
            fold_1_decision=decision.astype(np.int8),
        )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=DEFAULT_EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=DEFAULT_FULL_IDENTITY)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--fold-manifest", type=Path, default=DEFAULT_FOLD_MANIFEST)
    parser.add_argument(
        "--screening-protocol", type=Path, default=DEFAULT_SCREENING_PROTOCOL
    )
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=base.DEFAULT_POSITIVE_KB)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import AutoTokenizer
        import transformers
    except ModuleNotFoundError as exc:
        raise SystemExit("auxiliary SciNCL requires torch and transformers") from exc
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")

    paths = {
        name: value.resolve()
        for name, value in {
            "exact_identity": args.exact_identity,
            "full_identity": args.full_identity,
            "inputs": args.inputs,
            "manifest": args.input_manifest,
            "protocol": args.protocol,
            "fold_manifest": args.fold_manifest,
            "screening": args.screening_protocol,
            "baseline": args.baseline_report,
            "targets": args.targets,
            "page_cache": args.page_cache,
            "pdf_dir": args.pdf_dir,
            "taxonomy": args.taxonomy,
            "positive_kb": args.positive_kb,
            "model": args.model,
        }.items()
    }
    frozen_inputs = {
        "exact_identity": file_audit(
            paths["exact_identity"], EXACT_IDENTITY_SHA256, "exact identity"
        ),
        "full_identity": file_audit(
            paths["full_identity"], FULL_IDENTITY_SHA256, "full identity"
        ),
        "screening_protocol": file_audit(
            paths["screening"], SCREENING_PROTOCOL_SHA256, "screening protocol"
        ),
        "fold_manifest": file_audit(
            paths["fold_manifest"], FOLD_MANIFEST_SHA256, "canonical fold manifest"
        ),
        "baseline_report": file_audit(
            paths["baseline"], BASELINE_REPORT_SHA256, "baseline report"
        ),
        "targets": file_audit(paths["targets"], TARGETS_SHA256, "targets"),
        "page_cache": file_audit(
            paths["page_cache"], PAGE_CACHE_SHA256, "page cache"
        ),
        "taxonomy": file_audit(paths["taxonomy"], TAXONOMY_SHA256, "taxonomy"),
        "positive_kb": file_audit(
            paths["positive_kb"], POSITIVE_KB_SHA256, "positive expert KB"
        ),
        "model_weights": file_audit(
            paths["model"] / "model.safetensors",
            SCINCL_WEIGHT_SHA256,
            "SciNCL weights",
        ),
    }
    full_records, full_coverage = base.load_pdf_only_identity(
        paths["full_identity"], paths["pdf_dir"]
    )
    exact_records, exact_coverage = base.load_pdf_only_identity(
        paths["exact_identity"], paths["pdf_dir"]
    )
    auxiliary.validate_membership(full_records, exact_records)
    manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    validate_manifest(
        manifest,
        full_identity=paths["full_identity"],
        inputs=paths["inputs"],
        page_cache=paths["page_cache"],
        taxonomy=paths["taxonomy"],
        positive_kb=paths["positive_kb"],
        records=full_records,
    )
    protocol = validate_protocol(
        paths["protocol"],
        exact_identity=paths["exact_identity"],
        full_identity=paths["full_identity"],
        inputs=paths["inputs"],
        manifest=paths["manifest"],
        fold_manifest=paths["fold_manifest"],
        screening_protocol=paths["screening"],
        baseline_report=paths["baseline"],
        targets=paths["targets"],
        page_cache=paths["page_cache"],
        taxonomy=paths["taxonomy"],
        positive_kb=paths["positive_kb"],
        model=paths["model"],
    )
    components = exact_scincl.load_components(paths["inputs"], len(full_records))
    exact_fold_ids = load_canonical_fold_ids(
        paths["fold_manifest"], exact_records
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(paths["model"]), local_files_only=True
    )
    input_ids, attention_mask = exact_scincl.encode_components(tokenizer, components)
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "PASS_BEFORE_LABEL_ACCESS",
                    "full_rows": len(full_records),
                    "exact_rows": len(exact_records),
                    "canonical_fold_rows": dict(
                        sorted(Counter(int(value) for value in exact_fold_ids).items())
                    ),
                    "input_shape": list(input_ids.shape),
                    "attention_shape": list(attention_mask.shape),
                    "protocol_sha256": base.file_sha256(paths["protocol"]),
                    "fold_manifest_sha256": base.file_sha256(paths["fold_manifest"]),
                    "labels_read": False,
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    # R3 is opened only after the full-pool text cache and protocol pass validation.
    labels_by_resultno = base.load_training_labels(
        paths["targets"], {record["RESULTNO"] for record in full_records}
    )
    full_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in full_records],
        dtype=np.int8,
    )
    exact_labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in exact_records],
        dtype=np.int8,
    )
    device = torch.device(args.device)
    probability, decision, fold = run_fold1(
        full_records=full_records,
        exact_records=exact_records,
        full_labels=full_labels,
        exact_labels=exact_labels,
        exact_fold_ids=exact_fold_ids,
        input_ids=input_ids,
        attention_mask=attention_mask,
        model_path=paths["model"],
        device=device,
    )
    write_fold1_artifact(args.probabilities.resolve(), probability, decision)
    baseline_accuracy = float(protocol["baseline"]["fold1_accuracy"])
    observed_accuracy = float(fold["outer_test_selected_threshold"]["accuracy"])
    stable = bool(fold["training_stable_and_test_noncollapsed"])
    minimum_accuracy = baseline_accuracy + SMOKE_MINIMUM_ABSOLUTE_GAIN
    smoke_gate = {
        "baseline_fold1_accuracy": baseline_accuracy,
        "minimum_absolute_gain": SMOKE_MINIMUM_ABSOLUTE_GAIN,
        "minimum_accuracy": round(minimum_accuracy, 8),
        "observed_accuracy": round(observed_accuracy, 8),
        "observed_absolute_gain": round(observed_accuracy - baseline_accuracy, 8),
        "training_stable_and_test_noncollapsed": stable,
        "continue_to_all_folds": bool(stable and observed_accuracy >= minimum_accuracy),
    }
    report = {
        "schema_version": "r3_scientific_auxiliary_pool_fold1_result_v1",
        "status": "complete",
        "scope": {
            "evaluation_identity": "frozen exact300",
            "evaluated_folds": [1],
            "evaluated_rows": 60,
            "full_exact300_rows": EXPECTED_EXACT_ROWS,
            "auxiliary_candidate_rows": EXPECTED_FULL_ROWS,
            "auxiliary_candidate_groups": EXPECTED_FULL_GROUPS,
            "full_2172_rows_evaluated": False,
        },
        "guardrails": {
            "outer_split": "frozen exact300 5-fold GroupKFold by REFNO",
            "training_rule": (
                "all 2,172 PDF-backed fixed-training rows except every row whose "
                "REFNO belongs to exact300 outer fold 1"
            ),
            "inner_split": (
                "label-free SHA256-ranked REFNO validation split within allowed "
                "auxiliary rows"
            ),
            "canonical_outer_fold_manifest_sha256": FOLD_MANIFEST_SHA256,
            "outer_test_refno_overlap": fold["train_test_refno_overlap"],
            "class_balanced_weighted_loss": True,
            "inverse_paper_multiplicity_weight": True,
            "threshold_selected_from_inner_validation_only": True,
            "outer_test_labels_used_for_training_epoch_or_threshold_selection": False,
            "pmagresult_or_rockmag_used_as_predictive_input": False,
            "prior_predictions_or_evaluations_used_as_input": False,
            "identifier_arrays_in_text_cache": False,
        },
        "inputs": {
            **frozen_inputs,
            "text_cache": {
                "path": str(paths["inputs"]),
                "sha256": base.file_sha256(paths["inputs"]),
            },
            "text_manifest": {
                "path": str(paths["manifest"]),
                "sha256": base.file_sha256(paths["manifest"]),
            },
            "candidate_protocol": {
                "path": str(paths["protocol"]),
                "sha256": base.file_sha256(paths["protocol"]),
            },
            "implementation": {
                "path": str(Path(__file__).resolve()),
                "sha256": base.file_sha256(Path(__file__).resolve()),
            },
        },
        "data": {
            "full_coverage": full_coverage,
            "exact_coverage": exact_coverage,
            "full_label_counts": dict(sorted(Counter(int(v) for v in full_labels).items())),
            "exact_label_counts": dict(sorted(Counter(int(v) for v in exact_labels).items())),
        },
        "model": {
            "upstream": scientific.SCINCL_UPSTREAM,
            "revision": scientific.SCINCL_REVISION,
            "maximum_sequence_length": exact_scincl.MAX_SEQUENCE_LENGTH,
            "component_token_budgets": exact_scincl.COMPONENT_TOKEN_BUDGET,
            "maximum_selection_epochs": MAX_SELECTION_EPOCHS,
            "batch_size": BATCH_SIZE,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "last_unfrozen_encoder_layers": LAST_UNFROZEN_LAYERS,
            "encoder_learning_rate": ENCODER_LEARNING_RATE,
            "head_learning_rate": HEAD_LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "device": args.device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "folds": [fold],
        "smoke_gate": smoke_gate,
        "artifacts": {
            "probabilities": {
                "path": str(args.probabilities.resolve()),
                "sha256": base.file_sha256(args.probabilities.resolve()),
                "arrays": {
                    "fold_1_probability": [60],
                    "fold_1_decision": [60],
                },
            }
        },
    }
    base.atomic_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "fold1": fold["outer_test_selected_threshold"],
                "default_threshold_diagnostic": fold[
                    "outer_test_default_threshold_diagnostic"
                ],
                "smoke_gate": smoke_gate,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
