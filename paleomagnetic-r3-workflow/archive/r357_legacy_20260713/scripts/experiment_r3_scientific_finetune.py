#!/usr/bin/env python3
"""Fold-isolated supervised SciNCL fine-tuning for the frozen R3 screen."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_scientific_embeddings as scientific  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


DEFAULT_IDENTITY = ROOT / "tmp/r3_exact_300_resultno_v2/identity.xlsx"
DEFAULT_INPUTS = ROOT / "tmp/r3_scientific_finetune/scincl_exact300_v2_inputs.npz"
DEFAULT_INPUT_MANIFEST = (
    ROOT / "tmp/r3_scientific_finetune/scincl_exact300_v2_inputs.manifest.json"
)
DEFAULT_MODEL = ROOT / "tmp/models/scincl"
DEFAULT_SCREENING_PROTOCOL = (
    ROOT / "tmp/r3_exact_300_resultno_v2/screening_protocol.json"
)
DEFAULT_CANDIDATE_PROTOCOL = (
    ROOT / "tmp/r3_scientific_finetune/fold1_protocol_exact300_v2.json"
)
DEFAULT_BASELINE_REPORT = ROOT / "tmp/r3_exact_300_resultno_v2/baseline.json"
DEFAULT_OUTPUT = ROOT / "tmp/r3_scientific_finetune/scincl_exact300_v2_fold1_smoke.json"
DEFAULT_PROBABILITIES = (
    ROOT / "tmp/r3_scientific_finetune/scincl_exact300_v2_fold1_smoke_probabilities.npz"
)
EXPECTED_COMPONENTS = {"target", "abstract", "local_evidence", "paper_evidence"}
COMPONENT_ORDER = ("target", "abstract", "local_evidence", "paper_evidence")
COMPONENT_TOKEN_BUDGET = {
    "target": 24,
    "abstract": 150,
    "local_evidence": 105,
    "paper_evidence": 90,
}
MAX_SEQUENCE_LENGTH = 384
SMOKE_MINIMUM_ABSOLUTE_GAIN = 0.02
SCINCL_WEIGHT_SHA256 = "d65da219f67f051fdd065bfec666285c136e23002787603b2565e467bfed3c68"
CANDIDATE_PROTOCOL_SHA256 = "4de34c7fdf128770d9a5a3998b705a983fef32e9d4d122e4d9a1403238c734e7"
SCREENING_PROTOCOL_SHA256 = "6412244a5d268e6dcabfaed63f6f529d4976dd17ef6c4313f70bdb6748c04b4f"


def load_components(path: Path, row_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        names = set(payload.files)
        if names != EXPECTED_COMPONENTS:
            raise ValueError(
                "fine-tuning cache must contain only text components; "
                f"missing={sorted(EXPECTED_COMPONENTS - names)}, "
                f"extra={sorted(names - EXPECTED_COMPONENTS)}"
            )
        arrays = {name: np.asarray(payload[name], dtype=str) for name in names}
    if any(values.ndim != 1 or len(values) != row_count for values in arrays.values()):
        raise ValueError("fine-tuning text cache is not row aligned")
    if any(any(not str(value).strip() for value in values) for values in arrays.values()):
        raise ValueError("fine-tuning text cache contains an empty component")
    return arrays


def encode_components(
    tokenizer: Any,
    components: Mapping[str, np.ndarray],
    *,
    maximum_length: int = MAX_SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Allocate tokens explicitly so late evidence cannot be truncated away."""

    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id
    if cls_id is None or sep_id is None or pad_id is None:
        raise ValueError("SciNCL tokenizer lacks required BERT special tokens")
    row_count = len(components[COMPONENT_ORDER[0]])
    input_ids = np.full((row_count, maximum_length), pad_id, dtype=np.int64)
    attention_mask = np.zeros((row_count, maximum_length), dtype=np.int64)
    for row in range(row_count):
        ids = [cls_id]
        for name in COMPONENT_ORDER:
            tokens = tokenizer.encode(
                str(components[name][row]),
                add_special_tokens=False,
                truncation=True,
                max_length=COMPONENT_TOKEN_BUDGET[name],
            )
            ids.extend(tokens)
            ids.append(sep_id)
        if len(ids) > maximum_length:
            raise AssertionError("fixed component budgets exceed maximum sequence length")
        input_ids[row, : len(ids)] = ids
        attention_mask[row, : len(ids)] = 1
    return input_ids, attention_mask


def parse_folds(value: str) -> list[int]:
    if value.casefold() == "all":
        return list(range(1, base.N_SPLITS + 1))
    try:
        folds = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise ValueError("--folds must be 'all' or comma-separated integers") from exc
    if not folds or any(fold < 1 or fold > base.N_SPLITS for fold in folds):
        raise ValueError(f"folds must be within 1..{base.N_SPLITS}")
    return folds


def fold1_smoke_required(folds: Sequence[int]) -> bool:
    return list(folds) != [1]


def probability_cache(
    path: Path,
    probabilities: Mapping[str, np.ndarray],
) -> None:
    forbidden = re.compile(r"(?i)(?:label|answer|refno|resultno|rockname|identifier)")
    invalid = sorted(name for name in probabilities if forbidden.search(name))
    if invalid:
        raise ValueError(f"probability cache contains forbidden fields: {invalid}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(values, dtype=np.float32)
                for name, values in sorted(probabilities.items())
            },
        )
    temporary.replace(path)


def smoke_decision(
    accuracy: float,
    baseline_accuracy: float,
    *,
    stable: bool,
) -> dict[str, Any]:
    minimum = baseline_accuracy + SMOKE_MINIMUM_ABSOLUTE_GAIN
    return {
        "strict_baseline_fold_accuracy": baseline_accuracy,
        "minimum_absolute_gain": SMOKE_MINIMUM_ABSOLUTE_GAIN,
        "minimum_smoke_accuracy": round(float(minimum), 8),
        "observed_accuracy": round(float(accuracy), 8),
        "observed_absolute_gain": round(float(accuracy - baseline_accuracy), 8),
        "training_stable": bool(stable),
        "continue_to_all_folds": bool(stable and accuracy >= minimum),
    }


def validate_fold1_smoke_report(
    report: Mapping[str, Any],
    *,
    expected_inputs: Mapping[str, Any],
    expected_model: Mapping[str, Any],
    expected_rows: int,
    expected_groups: int,
    expected_test_rows: int,
    baseline_accuracy: float,
) -> None:
    """Bind an all-fold promotion to the exact frozen fold-1 experiment."""

    if (
        report.get("schema_version") != "r3_supervised_scincl_grouped_fold_v1"
        or report.get("status") != "complete"
    ):
        raise ValueError("fold1 smoke report has an invalid schema or status")
    scope = report.get("scope", {})
    data = report.get("data", {})
    if scope.get("evaluated_folds") != [1]:
        raise ValueError("fold1 smoke report must evaluate only fold 1")
    if (
        scope.get("pdf_only_rows") != expected_rows
        or scope.get("pdf_only_paper_groups") != expected_groups
        or data.get("rows") != expected_rows
        or data.get("groups") != expected_groups
    ):
        raise ValueError("fold1 smoke report exact300 scope mismatch")

    inputs = report.get("inputs", {})
    mismatched_inputs = sorted(
        name for name, value in expected_inputs.items() if inputs.get(name) != value
    )
    if mismatched_inputs:
        raise ValueError(
            "fold1 smoke report input hash mismatch: "
            + ", ".join(mismatched_inputs)
        )
    model = report.get("model", {})
    mismatched_model = sorted(
        name for name, value in expected_model.items() if model.get(name) != value
    )
    if mismatched_model:
        raise ValueError(
            "fold1 smoke report model/protocol mismatch: "
            + ", ".join(mismatched_model)
        )

    folds = report.get("folds")
    if not isinstance(folds, list) or len(folds) != 1:
        raise ValueError("fold1 smoke report must contain one fold audit")
    fold = folds[0]
    if (
        fold.get("fold") != 1
        or fold.get("outer_test_rows") != expected_test_rows
        or fold.get("training_stable") is not True
    ):
        raise ValueError("fold1 smoke report fold audit mismatch")
    outer_accuracy = fold.get("outer_test", {}).get("accuracy")
    gate = report.get("smoke_gate", {})
    expected_minimum = baseline_accuracy + SMOKE_MINIMUM_ABSOLUTE_GAIN
    numeric_checks = (
        (gate.get("strict_baseline_fold_accuracy"), baseline_accuracy),
        (gate.get("minimum_absolute_gain"), SMOKE_MINIMUM_ABSOLUTE_GAIN),
        (gate.get("minimum_smoke_accuracy"), expected_minimum),
        (gate.get("observed_accuracy"), outer_accuracy),
    )
    if any(
        not isinstance(observed, (int, float))
        or not isinstance(expected, (int, float))
        or not math.isclose(float(observed), float(expected), abs_tol=1e-8)
        for observed, expected in numeric_checks
    ):
        raise ValueError("fold1 smoke report gate metrics are inconsistent")
    if (
        gate.get("training_stable") is not True
        or gate.get("continue_to_all_folds") is not True
        or float(outer_accuracy) < expected_minimum
    ):
        raise ValueError("fold1 smoke did not authorize all-fold training")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)


def make_model(model_path: Path, last_layers: int, device: Any) -> tuple[Any, dict[str, int]]:
    import torch
    from transformers import AutoModel

    class SciNCLClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(
                str(model_path), local_files_only=True
            )
            self.dropout = torch.nn.Dropout(0.10)
            self.classifier = torch.nn.Linear(self.encoder.config.hidden_size, 2)

        def forward(self, input_ids: Any, attention_mask: Any) -> Any:
            encoded = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            return self.classifier(self.dropout(encoded.last_hidden_state[:, 0]))

    model = SciNCLClassifier()
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    layers = model.encoder.encoder.layer
    if last_layers < 1 or last_layers > len(layers):
        raise ValueError("last_layers must select at least one valid encoder layer")
    for layer in layers[-last_layers:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    model.to(device)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return model, {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "trainable_fraction": round(float(trainable / total), 8),
        "unfrozen_encoder_layers": int(last_layers),
    }


def keep_frozen_layers_in_eval(model: Any, last_layers: int) -> None:
    model.train()
    model.encoder.embeddings.eval()
    for layer in model.encoder.encoder.layer[:-last_layers]:
        layer.eval()
    if getattr(model.encoder, "pooler", None) is not None:
        model.encoder.pooler.eval()


def evaluate_model(
    model: Any,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, float]:
    import torch
    from torch.nn import functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(
        torch.from_numpy(input_ids[indices]),
        torch.from_numpy(attention_mask[indices]),
        torch.from_numpy(labels[indices].astype(np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    probabilities: list[np.ndarray] = []
    loss_total = 0.0
    count = 0
    with torch.no_grad():
        for batch_ids, batch_mask, batch_labels in loader:
            batch_ids = batch_ids.to(device)
            batch_mask = batch_mask.to(device)
            batch_labels = batch_labels.to(device)
            logits = model(batch_ids, batch_mask)
            loss = functional.cross_entropy(logits, batch_labels, reduction="sum")
            probability = torch.softmax(logits, dim=1)[:, 1]
            probabilities.append(probability.detach().cpu().numpy())
            loss_total += float(loss.detach().cpu())
            count += len(batch_labels)
    return np.concatenate(probabilities).astype(np.float64), loss_total / max(1, count)


def weighted_train_loader(
    indices: np.ndarray,
    *,
    records: Sequence[dict[str, str]],
    canonical_targets: Sequence[str],
    labels: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    batch_size: int,
    seed: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    weights = base.inverse_target_multiplicity_weights(
        indices, records, canonical_targets
    ).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(input_ids[indices]),
        torch.from_numpy(attention_mask[indices]),
        torch.from_numpy(labels[indices].astype(np.int64)),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def optimizer_and_scheduler(
    model: Any,
    loader: Any,
    *,
    epochs: int,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> tuple[Any, Any]:
    import torch
    from transformers import get_linear_schedule_with_warmup

    encoder_parameters = [
        parameter
        for parameter in model.encoder.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_parameters, "lr": encoder_learning_rate},
            {"params": list(model.classifier.parameters()), "lr": head_learning_rate},
        ],
        weight_decay=weight_decay,
    )
    total_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(total_steps * 0.10)),
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    device: Any,
    last_layers: int,
) -> float:
    import torch
    from torch.nn import functional as functional

    keep_frozen_layers_in_eval(model, last_layers)
    train_loss = 0.0
    train_examples = 0
    for batch_ids, batch_mask, batch_labels, batch_weights in loader:
        batch_ids = batch_ids.to(device)
        batch_mask = batch_mask.to(device)
        batch_labels = batch_labels.to(device)
        batch_weights = batch_weights.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_ids, batch_mask)
        losses = functional.cross_entropy(logits, batch_labels, reduction="none")
        loss = (losses * batch_weights).sum() / batch_weights.sum().clamp_min(1e-8)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite supervised SciNCL training loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()
        scheduler.step()
        train_loss += float(loss.detach().cpu()) * len(batch_labels)
        train_examples += len(batch_labels)
    return train_loss / max(1, train_examples)


def run_fold(
    fold: int,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    *,
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    groups: np.ndarray,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    model_path: Path,
    device: Any,
    epochs: int,
    batch_size: int,
    evaluation_batch_size: int,
    last_layers: int,
    encoder_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    inner_groups = groups[outer_train]
    inner_splitter = GroupKFold(n_splits=5)
    inner_fit_position, inner_validation_position = next(
        inner_splitter.split(
            np.zeros(len(outer_train)), labels[outer_train], inner_groups
        )
    )
    fit_index = outer_train[inner_fit_position]
    validation_index = outer_train[inner_validation_position]
    if (
        set(groups[fit_index]) & set(groups[validation_index])
        or set(groups[fit_index]) & set(groups[outer_test])
        or set(groups[validation_index]) & set(groups[outer_test])
    ):
        raise AssertionError("REFNO leaked across supervised fine-tuning partitions")

    canonical_targets = [canonical_target_identity(record["ROCKNAME"]) for record in records]
    loader = weighted_train_loader(
        fit_index,
        records=records,
        canonical_targets=canonical_targets,
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_size=batch_size,
        seed=base.RANDOM_STATE + fold,
    )

    set_seed(base.RANDOM_STATE + fold)
    model, _ = make_model(model_path, last_layers, device)
    optimizer, scheduler = optimizer_and_scheduler(
        model,
        loader,
        epochs=epochs,
        encoder_learning_rate=encoder_learning_rate,
        head_learning_rate=head_learning_rate,
        weight_decay=weight_decay,
    )

    epoch_records: list[dict[str, Any]] = []
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    stable = True
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            device=device,
            last_layers=last_layers,
        )

        validation_probability, validation_loss = evaluate_model(
            model,
            input_ids,
            attention_mask,
            labels,
            validation_index,
            device=device,
            batch_size=evaluation_batch_size,
        )
        validation_metric = base.metric_record(
            labels[validation_index], validation_probability
        )
        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 8),
            "validation_loss": round(float(validation_loss), 8),
            "validation": validation_metric,
        }
        epoch_records.append(record)
        key = (
            validation_metric["accuracy"],
            validation_metric["macro_f1"],
            -validation_loss,
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
        print(
            f"fold {fold} epoch {epoch}/{epochs}: "
            f"train_loss={record['train_loss']:.4f}, "
            f"val_acc={validation_metric['accuracy']:.4f}",
            flush=True,
        )

    # Refit from the fixed pretrained checkpoint on every outer-training row.
    del model, optimizer, scheduler, loader
    if getattr(torch, "mps", None) is not None:
        torch.mps.empty_cache()
    refit_loader = weighted_train_loader(
        outer_train,
        records=records,
        canonical_targets=canonical_targets,
        labels=labels,
        input_ids=input_ids,
        attention_mask=attention_mask,
        batch_size=batch_size,
        seed=base.RANDOM_STATE + 1000 + fold,
    )
    set_seed(base.RANDOM_STATE + 1000 + fold)
    model, parameter_audit = make_model(model_path, last_layers, device)
    optimizer, scheduler = optimizer_and_scheduler(
        model,
        refit_loader,
        epochs=best_epoch,
        encoder_learning_rate=encoder_learning_rate,
        head_learning_rate=head_learning_rate,
        weight_decay=weight_decay,
    )
    refit_records: list[dict[str, Any]] = []
    for refit_epoch in range(1, best_epoch + 1):
        refit_loss = train_epoch(
            model,
            refit_loader,
            optimizer,
            scheduler,
            device=device,
            last_layers=last_layers,
        )
        refit_records.append(
            {"epoch": refit_epoch, "train_loss": round(refit_loss, 8)}
        )
        print(
            f"fold {fold} refit {refit_epoch}/{best_epoch}: "
            f"train_loss={refit_loss:.4f}",
            flush=True,
        )
    test_probability, test_loss = evaluate_model(
        model,
        input_ids,
        attention_mask,
        labels,
        outer_test,
        device=device,
        batch_size=evaluation_batch_size,
    )
    test_metric = base.metric_record(labels[outer_test], test_probability)
    stable = stable and np.isfinite(test_probability).all() and math.isfinite(test_loss)
    audit = {
        "fold": fold,
        "outer_train_rows": int(len(outer_train)),
        "outer_train_groups": int(len(set(groups[outer_train]))),
        "inner_fit_rows": int(len(fit_index)),
        "inner_fit_groups": int(len(set(groups[fit_index]))),
        "inner_validation_rows": int(len(validation_index)),
        "inner_validation_groups": int(len(set(groups[validation_index]))),
        "outer_test_rows": int(len(outer_test)),
        "outer_test_groups": int(len(set(groups[outer_test]))),
        "selected_epoch": int(best_epoch),
        "epoch_selection": "inner REFNO-grouped validation accuracy, then macro-F1, then loss",
        "inner_selection_epochs": epoch_records,
        "outer_train_refit_epochs": refit_records,
        "outer_test_loss": round(float(test_loss), 8),
        "outer_test": test_metric,
        "training_stable": bool(stable),
        "parameters": parameter_audit,
    }
    del model, optimizer, scheduler, refit_loader
    if getattr(torch, "mps", None) is not None:
        torch.mps.empty_cache()
    return test_probability, audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT
    )
    parser.add_argument(
        "--screening-protocol", type=Path, default=DEFAULT_SCREENING_PROTOCOL
    )
    parser.add_argument(
        "--candidate-protocol", type=Path, default=DEFAULT_CANDIDATE_PROTOCOL
    )
    parser.add_argument("--folds", default="1")
    parser.add_argument("--fold1-smoke-report", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--evaluation-batch-size", type=int, default=16)
    parser.add_argument("--last-layers", type=int, default=2)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    args = parser.parse_args(argv)

    try:
        import torch
        from transformers import AutoTokenizer
        import transformers
    except ModuleNotFoundError as exc:
        raise SystemExit("supervised SciNCL requires torch and transformers") from exc
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")
    if args.epochs < 1 or args.batch_size < 1 or args.evaluation_batch_size < 1:
        raise SystemExit("epoch and batch counts must be positive")

    selected_folds = parse_folds(args.folds)
    if fold1_smoke_required(selected_folds) and args.fold1_smoke_report is None:
        raise SystemExit("every non-smoke fold run requires --fold1-smoke-report")

    identity_path = args.identity.resolve()
    labels_path = args.targets.resolve()
    inputs_path = args.inputs.resolve()
    manifest_path = args.input_manifest.resolve()
    model_path = args.model.resolve()
    baseline_report_path = args.baseline_report.resolve()
    screening_protocol_path = args.screening_protocol.resolve()
    candidate_protocol_path = args.candidate_protocol.resolve()
    records, coverage = base.load_pdf_only_identity(identity_path, args.pdf_dir.resolve())
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    boundary = manifest.get("runtime_boundary", {})
    if any(
        boundary.get(name) is not False
        for name in (
            "labels_read",
            "pmagresult_or_rockmag_read",
            "prior_predictions_or_evaluations_read",
            "cache_contains_record_identifiers",
        )
    ):
        raise ValueError("fine-tuning input manifest violates the source boundary")
    if manifest.get("inputs", {}).get("identity", {}).get("sha256") != base.file_sha256(identity_path):
        raise ValueError("fine-tuning input identity hash mismatch")
    if manifest.get("inputs", {}).get("alignment_sha256") != scientific.alignment_sha256(records):
        raise ValueError("fine-tuning input row alignment mismatch")
    if manifest.get("output", {}).get("sha256") != base.file_sha256(inputs_path):
        raise ValueError("fine-tuning text cache hash mismatch")
    components = load_components(inputs_path, len(records))

    model_weight_path = model_path / "model.safetensors"
    if scientific.sha256_file(model_weight_path) != SCINCL_WEIGHT_SHA256:
        raise ValueError("local SciNCL weight hash mismatch")

    if base.file_sha256(screening_protocol_path) != SCREENING_PROTOCOL_SHA256:
        raise ValueError("exact300 screening protocol hash mismatch")
    screening_protocol = json.loads(
        screening_protocol_path.read_text(encoding="utf-8")
    )
    if screening_protocol["selection"]["identity_sha256"] != base.file_sha256(identity_path):
        raise ValueError("exact300 screening protocol identity mismatch")
    if screening_protocol["baseline"]["report_sha256"] != base.file_sha256(baseline_report_path):
        raise ValueError("exact300 screening protocol baseline mismatch")
    baseline_report = json.loads(
        baseline_report_path.read_text(encoding="utf-8")
    )
    if (
        baseline_report.get("inputs", {}).get("training_identity", {}).get("sha256")
        != base.file_sha256(identity_path)
        or baseline_report.get("inputs", {}).get("targets", {}).get("sha256")
        != base.file_sha256(labels_path)
    ):
        raise ValueError("current identity or labels differ from the frozen baseline")

    if base.file_sha256(candidate_protocol_path) != CANDIDATE_PROTOCOL_SHA256:
        raise ValueError("supervised SciNCL candidate protocol hash mismatch")
    candidate_protocol = json.loads(
        candidate_protocol_path.read_text(encoding="utf-8")
    )
    if (
        candidate_protocol["baseline"]["report_sha256"]
        != base.file_sha256(baseline_report_path)
    ):
        raise ValueError("candidate protocol baseline report mismatch")
    candidate_training = candidate_protocol["training"]
    expected_hyperparameters = {
        "epochs": candidate_training["maximum_selection_epochs"],
        "batch_size": candidate_training["batch_size"],
        "evaluation_batch_size": candidate_training["evaluation_batch_size"],
        "last_layers": candidate_protocol["model"]["trainable_encoder_layers"],
        "encoder_learning_rate": candidate_training["encoder_learning_rate"],
        "head_learning_rate": candidate_training["head_learning_rate"],
        "weight_decay": candidate_training["weight_decay"],
    }
    actual_hyperparameters = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "evaluation_batch_size": args.evaluation_batch_size,
        "last_layers": args.last_layers,
        "encoder_learning_rate": args.encoder_learning_rate,
        "head_learning_rate": args.head_learning_rate,
        "weight_decay": args.weight_decay,
    }
    if actual_hyperparameters != expected_hyperparameters:
        raise ValueError("runtime hyperparameters differ from the frozen candidate protocol")
    if candidate_protocol["source_boundary"]["text_cache_sha256"] != base.file_sha256(inputs_path):
        raise ValueError("candidate protocol text cache mismatch")
    if candidate_protocol["source_boundary"]["text_manifest_sha256"] != base.file_sha256(manifest_path):
        raise ValueError("candidate protocol text manifest mismatch")
    if candidate_protocol["scope"]["identity_sha256"] != base.file_sha256(identity_path):
        raise ValueError("candidate protocol identity mismatch")
    if (
        candidate_protocol["scope"]["rows"] != len(records)
        or candidate_protocol["scope"]["refno_groups"]
        != len({record["REFNO"] for record in records})
    ):
        raise ValueError("candidate protocol exact300 scope mismatch")
    if (
        candidate_protocol["model"]["upstream"] != scientific.SCINCL_UPSTREAM
        or candidate_protocol["model"]["revision"] != scientific.SCINCL_REVISION
        or candidate_protocol["model"]["weight_sha256"] != SCINCL_WEIGHT_SHA256
        or candidate_training["random_state"] != base.RANDOM_STATE
    ):
        raise ValueError("model or random state differs from the frozen candidate protocol")
    if (
        candidate_protocol["tokens"]["maximum_sequence_length"]
        != MAX_SEQUENCE_LENGTH
        or candidate_protocol["tokens"]["component_order"]
        != list(COMPONENT_ORDER)
        or candidate_protocol["tokens"]["component_budgets"]
        != COMPONENT_TOKEN_BUDGET
        or candidate_protocol["fold1_smoke_gate"]["minimum_absolute_gain_over_primary"]
        != SMOKE_MINIMUM_ABSOLUTE_GAIN
    ):
        raise ValueError("code constants differ from the frozen candidate protocol")

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    input_ids, attention_mask = encode_components(tokenizer, components)
    # R3 is opened only after the label-free cache and protocol pass validation.
    labels_by_resultno = base.load_training_labels(
        labels_path, {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    splits = list(
        GroupKFold(n_splits=base.N_SPLITS).split(
            np.zeros(len(records)), labels, groups
        )
    )
    if fold1_smoke_required(selected_folds):
        smoke = json.loads(
            args.fold1_smoke_report.resolve().read_text(encoding="utf-8")
        )
        validate_fold1_smoke_report(
            smoke,
            expected_inputs={
                "identity_sha256": base.file_sha256(identity_path),
                "labels_sha256": base.file_sha256(labels_path),
                "text_cache_sha256": base.file_sha256(inputs_path),
                "text_manifest_sha256": base.file_sha256(manifest_path),
                "baseline_report_sha256": base.file_sha256(baseline_report_path),
                "screening_protocol_sha256": base.file_sha256(screening_protocol_path),
                "candidate_protocol_sha256": base.file_sha256(candidate_protocol_path),
                "model_files": {
                    "model.safetensors": scientific.sha256_file(model_weight_path),
                    "config.json": scientific.sha256_file(model_path / "config.json"),
                    "vocab.txt": scientific.sha256_file(model_path / "vocab.txt"),
                },
            },
            expected_model={
                "upstream": scientific.SCINCL_UPSTREAM,
                "revision": scientific.SCINCL_REVISION,
                "token_budgets": COMPONENT_TOKEN_BUDGET,
                "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
                "last_unfrozen_encoder_layers": args.last_layers,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "evaluation_batch_size": args.evaluation_batch_size,
                "encoder_learning_rate": args.encoder_learning_rate,
                "head_learning_rate": args.head_learning_rate,
                "weight_decay": args.weight_decay,
            },
            expected_rows=len(records),
            expected_groups=len(set(groups)),
            expected_test_rows=len(splits[0][1]),
            baseline_accuracy=float(
                candidate_protocol["baseline"]["primary_fold_accuracy"][0]
            ),
        )
    device = torch.device(args.device)
    fold_reports: list[dict[str, Any]] = []
    saved_probabilities: dict[str, np.ndarray] = {}
    oof = np.full(len(records), np.nan, dtype=np.float64)
    for fold in selected_folds:
        outer_train, outer_test = splits[fold - 1]
        predicted, audit = run_fold(
            fold,
            outer_train,
            outer_test,
            records=records,
            labels=labels,
            groups=groups,
            input_ids=input_ids,
            attention_mask=attention_mask,
            model_path=model_path,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            evaluation_batch_size=args.evaluation_batch_size,
            last_layers=args.last_layers,
            encoder_learning_rate=args.encoder_learning_rate,
            head_learning_rate=args.head_learning_rate,
            weight_decay=args.weight_decay,
        )
        oof[outer_test] = predicted
        saved_probabilities[f"fold_{fold}_probability"] = predicted
        fold_reports.append(audit)

    smoke_gate = None
    if selected_folds == [1]:
        smoke_gate = smoke_decision(
            fold_reports[0]["outer_test"]["accuracy"],
            float(candidate_protocol["baseline"]["primary_fold_accuracy"][0]),
            stable=bool(fold_reports[0]["training_stable"]),
        )
    aggregate = None
    promotion_gate = None
    if selected_folds == list(range(1, base.N_SPLITS + 1)):
        if not np.isfinite(oof).all():
            raise AssertionError("all-fold supervised SciNCL OOF is incomplete")
        aggregate = base.metric_record(labels, oof)
        baseline_folds = candidate_protocol["baseline"]["primary_fold_accuracy"]
        deltas = [
            report["outer_test"]["accuracy"] - float(baseline_accuracy)
            for report, baseline_accuracy in zip(fold_reports, baseline_folds)
        ]
        gate = screening_protocol["promotion_gate"]
        promotion_gate = {
            "fold_accuracy_deltas": [round(float(value), 8) for value in deltas],
            "folds_not_worse": sum(value >= 0.0 for value in deltas),
            "worst_fold_regression": round(float(min(deltas)), 8),
            "target_accuracy": gate["minimum_aggregate_accuracy"],
            "target_reached_on_exact300": bool(
                aggregate["accuracy"] >= gate["minimum_aggregate_accuracy"]
            ),
            "full_2172_row_run_forbidden": True,
        }
        saved_probabilities = {"supervised_scincl_probability": oof}

    report = {
        "schema_version": "r3_supervised_scincl_grouped_fold_v1",
        "status": "complete",
        "scope": {**coverage, "evaluated_folds": selected_folds},
        "guardrails": {
            "outer_split": "5-fold GroupKFold by REFNO",
            "epoch_selection": "inner 5-fold GroupKFold first validation fold within outer training only",
            "selected_epoch_refit": "fresh checkpoint refit on every outer-training row before outer test",
            "outer_test_labels_used_for_training_or_epoch_selection": False,
            "labels_opened_after_label_free_cache_validation": True,
            "identifier_features": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_used_as_features": False,
            "all_fold_run_requires_passing_fold1_smoke_report": True,
            "partial_non_smoke_fold_runs_also_require_fold1_smoke_report": True,
        },
        "inputs": {
            "identity_sha256": base.file_sha256(identity_path),
            "labels_sha256": base.file_sha256(labels_path),
            "text_cache_sha256": base.file_sha256(inputs_path),
            "text_manifest_sha256": base.file_sha256(manifest_path),
            "baseline_report_sha256": base.file_sha256(baseline_report_path),
            "screening_protocol_sha256": base.file_sha256(screening_protocol_path),
            "candidate_protocol_sha256": base.file_sha256(candidate_protocol_path),
            "model_files": {
                "model.safetensors": scientific.sha256_file(model_weight_path),
                "config.json": scientific.sha256_file(model_path / "config.json"),
                "vocab.txt": scientific.sha256_file(model_path / "vocab.txt"),
            },
        },
        "model": {
            "upstream": scientific.SCINCL_UPSTREAM,
            "revision": scientific.SCINCL_REVISION,
            "classification_representation": "last_hidden_state CLS",
            "token_budgets": COMPONENT_TOKEN_BUDGET,
            "maximum_sequence_length": MAX_SEQUENCE_LENGTH,
            "last_unfrozen_encoder_layers": args.last_layers,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "evaluation_batch_size": args.evaluation_batch_size,
            "encoder_learning_rate": args.encoder_learning_rate,
            "head_learning_rate": args.head_learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
        },
        "data": {
            "rows": len(records),
            "groups": len(set(groups)),
            "labels": dict(sorted(Counter(int(value) for value in labels).items())),
        },
        "folds": fold_reports,
        "smoke_gate": smoke_gate,
        "aggregate": aggregate,
        "promotion_gate": promotion_gate,
        "interpretation": {
            "exact300_primary_fold1_baseline": candidate_protocol["baseline"]["primary_fold_accuracy"][0],
            "exact300_primary_aggregate_baseline": candidate_protocol["baseline"]["primary_accuracy"],
            "exact300_secondary_strict_aggregate_baseline": candidate_protocol["baseline"]["secondary_strict_propagation_accuracy"],
            "exact300_target_accuracy": candidate_protocol["conditional_all_fold_gate"]["minimum_aggregate_accuracy"],
            "full_weighted_reference": 0.83103131,
            "full_2172_row_run_forbidden": True,
        },
    }
    base.atomic_json(args.output.resolve(), report)
    probability_cache(args.probabilities.resolve(), saved_probabilities)
    print(
        json.dumps(
            {
                "folds": selected_folds,
                "fold_metrics": [item["outer_test"] for item in fold_reports],
                "smoke_gate": smoke_gate,
                "aggregate": aggregate,
                "promotion_gate": promotion_gate,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
