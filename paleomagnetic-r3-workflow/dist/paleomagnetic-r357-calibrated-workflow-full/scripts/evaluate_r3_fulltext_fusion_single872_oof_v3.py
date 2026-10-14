#!/usr/bin/env python3
"""Evaluate the frozen full-document text extension of the R3 V2 fusion."""

from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_query_adaptive_retrieval_features as retrieval  # noqa: E402
import build_r3_style_features_single872_v2 as style_freeze  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import evaluate_r3_query_adaptive_single872_oof as query_eval  # noqa: E402
import evaluate_r3_residual_gate_single872_oof as residual  # noqa: E402
import evaluate_r3_style_fusion_single872_oof_v2 as v2  # noqa: E402
import experiment_r3_pdf_only_strategies as pdf_text  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402


DEFAULT_CONFIG = ROOT / "data/knowledge/r3_fulltext_fusion_evaluation_v3.json"
DEFAULT_V2_CONFIG = ROOT / "data/knowledge/r3_style_fusion_evaluation_v2.json"
DEFAULT_V2_OOF = (
    ROOT
    / "tmp/r3_style_fusion_single872_v2"
    / "r3_style_fusion_single872_grouped_oof_v2.npz"
)
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_fulltext_fusion_single872_v3"

FREEZE_NAME = "r3_fulltext_fusion_evaluation_freeze_v3.json"
OOF_NAME = "r3_fulltext_fusion_single872_grouped_oof_v3.npz"
RECEIPT_NAME = "r3_fulltext_fusion_single872_grouped_oof_receipt_v3.json"
REPORT_NAME = "r3_fulltext_fusion_single872_grouped_oof_report_v3.json"
RUN_STATE_NAME = "r3_fulltext_fusion_evaluation_run_state_v3.json"
FULLTEXT_EXPERTS = (
    "fulltext_word_c05",
    "fulltext_word_c1",
    "fulltext_word_c2",
)


class FulltextFusionError(ValueError):
    pass


def clean(value: Any) -> str:
    return residual.clean(value)


def file_sha256(path: Path) -> str:
    return trace_freeze.file_sha256(path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FulltextFusionError(f"{label} must be a JSON object")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        Path(temporary_name).replace(path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def acquire_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    handle = (output_dir / ".r3_fulltext_fusion_v3.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise FulltextFusionError("another V3 fulltext evaluation is running") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


class RunState:
    def __init__(self, path: Path, freeze_sha256: str) -> None:
        self.path = path
        self.freeze_sha256 = freeze_sha256
        self.finished = False
        atomic_json(
            path,
            {
                "schema_version": "r3_fulltext_fusion_evaluation_run_state_v3",
                "status": "RUNNING",
                "pid": os.getpid(),
                "evaluation_freeze_sha256": freeze_sha256,
            },
        )

    def complete(self, report: Path, oof: Path, receipt: Path) -> None:
        atomic_json(
            self.path,
            {
                "schema_version": "r3_fulltext_fusion_evaluation_run_state_v3",
                "status": "COMPLETE",
                "evaluation_freeze_sha256": self.freeze_sha256,
                "report": {"path": str(report), "sha256": file_sha256(report)},
                "oof": {"path": str(oof), "sha256": file_sha256(oof)},
                "receipt": {"path": str(receipt), "sha256": file_sha256(receipt)},
            },
        )
        self.finished = True

    def fail(self, error: BaseException) -> None:
        if not self.finished:
            atomic_json(
                self.path,
                {
                    "schema_version": "r3_fulltext_fusion_evaluation_run_state_v3",
                    "status": "FAILED_OR_INTERRUPTED",
                    "evaluation_freeze_sha256": self.freeze_sha256,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != "r3_fulltext_fusion_evaluation_v3"
        or config.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
    ):
        raise FulltextFusionError("V3 evaluation config schema changed")
    protocol = config.get("protocol") or {}
    if (
        int(protocol.get("outer_folds", -1)) != 5
        or float(protocol.get("threshold", math.nan)) != 0.5
        or protocol.get("outer_test_selection") is not False
        or protocol.get("single_target_training_only_for_fulltext") is not True
    ):
        raise FulltextFusionError("V3 evaluation protocol changed")
    text = config.get("fulltext_expert") or {}
    if (
        text.get("analyzer") != "word"
        or tuple(text.get("ngram_range") or ()) != (1, 2)
        or tuple(map(float, text.get("C_values") or ())) != (0.5, 1.0, 2.0)
        or text.get("aggregation") != "uniform_arithmetic_mean"
        or int(text.get("max_features", -1)) != 100000
    ):
        raise FulltextFusionError("V3 fulltext expert contract changed")
    fusion = config.get("primary_fusion") or {}
    if (
        float(fusion.get("v2_weight", math.nan)) != 0.95
        or float(fusion.get("fulltext_weight", math.nan)) != 0.05
        or float(fusion.get("threshold", math.nan)) != 0.5
    ):
        raise FulltextFusionError("V3 primary fusion changed")
    boundary = config.get("runtime_boundary") or {}
    for name in (
        "pmagresult_or_rockmag_read",
        "prior_predictions_read_before_provisional",
        "evaluation_results_read_before_provisional",
        "outer_test_labels_used_for_training_or_selection",
        "posthoc_threshold_search_used",
        "selective_abstention_used",
    ):
        if boundary.get(name) is not False:
            raise FulltextFusionError(f"invalid V3 runtime boundary: {name}")


def full_documents(
    records: Sequence[Mapping[str, str]], page_cache: Path
) -> list[str]:
    refnos = {record["REFNO"] for record in records}
    papers = pdf_text.load_page_text(page_cache, refnos)
    return [
        f"{record['ROCKNAME']}\n{papers[record['REFNO']]}" for record in records
    ]


def fit_fulltext_experts(
    documents: Sequence[str],
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    spec = config["fulltext_expert"]
    vectorizer = TfidfVectorizer(
        analyzer=clean(spec["analyzer"]),
        ngram_range=tuple(map(int, spec["ngram_range"])),
        min_df=int(spec["min_df"]),
        max_df=float(spec["max_df"]),
        max_features=int(spec["max_features"]),
        sublinear_tf=bool(spec["sublinear_tf"]),
        strip_accents=clean(spec["strip_accents"]),
        dtype=np.float32,
    )
    fit = vectorizer.fit_transform([documents[int(index)] for index in train])
    predict = vectorizer.transform([documents[int(index)] for index in test])
    output: dict[str, np.ndarray] = {}
    for name, c_value in zip(FULLTEXT_EXPERTS, spec["C_values"]):
        model = LogisticRegression(
            C=float(c_value),
            solver=clean(spec["solver"]),
            max_iter=int(spec["max_iter"]),
            random_state=int(config["protocol"]["random_state"]),
        )
        model.fit(fit, labels[train])
        output[name] = np.asarray(model.predict_proba(predict)[:, 1], dtype=np.float64)
    if any(
        value.shape != (len(test),) or not np.isfinite(value).all()
        for value in output.values()
    ):
        raise AssertionError("V3 fulltext expert prediction is invalid")
    return output


def primary_fusion(
    v2_probability: np.ndarray,
    fulltext_probability: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    spec = config["primary_fusion"]
    return (
        float(spec["v2_weight"]) * np.asarray(v2_probability)
        + float(spec["fulltext_weight"]) * np.asarray(fulltext_probability)
    )


def metric(labels: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    return trace_eval.metric(np.asarray(labels), np.asarray(probability))


def delta_record(
    labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray
) -> dict[str, Any]:
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


def audit_v2_reproduction(provisional: Path, v2_oof: Path) -> dict[str, Any]:
    if not v2_oof.is_file():
        raise FulltextFusionError("V2 OOF audit file is missing")
    names = (
        "source_row_indices",
        "labels",
        "fold_ids",
        "no_strict_base",
        "knowledge_mean3",
        "primary",
        *(f"component__{name}" for name in v2.BASE_COMPONENTS),
        *(f"expert__{name}" for name in v2.KNOWLEDGE_EXPERTS),
    )
    maximum: dict[str, float] = {}
    with np.load(provisional, allow_pickle=False) as observed, np.load(
        v2_oof, allow_pickle=False
    ) as expected:
        for name in names:
            if name in {"source_row_indices", "labels", "fold_ids"}:
                if not np.array_equal(observed[name], expected[name]):
                    raise FulltextFusionError(f"V2 audit alignment mismatch: {name}")
                continue
            difference = np.abs(observed[name] - expected[name])
            maximum[name] = float(difference.max(initial=0.0))
            if not np.allclose(observed[name], expected[name], rtol=0.0, atol=1e-12):
                raise FulltextFusionError(f"V2 reproduction failed: {name}")
    return {
        "v2_oof": {"path": str(v2_oof), "sha256": file_sha256(v2_oof)},
        "opened_only_after_provisional_predictions_written": True,
        "alignment_exact": True,
        "maximum_absolute_difference": maximum,
        "v2_reproduced_at_tolerance_1e_12": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v2-config", type=Path, default=DEFAULT_V2_CONFIG)
    parser.add_argument("--v1-config", type=Path, default=v2.DEFAULT_V1_CONFIG)
    parser.add_argument("--style-features", type=Path, default=v2.DEFAULT_STYLE_FEATURES)
    parser.add_argument("--style-manifest", type=Path, default=v2.DEFAULT_STYLE_MANIFEST)
    parser.add_argument("--domain-features", type=Path, default=residual.DEFAULT_DOMAIN_FEATURES)
    parser.add_argument("--domain-manifest", type=Path, default=residual.DEFAULT_DOMAIN_MANIFEST)
    parser.add_argument("--page-cache", type=Path, default=residual.DEFAULT_PAGE_CACHE)
    parser.add_argument("--v4-features", type=Path, default=residual.DEFAULT_V4_FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=residual.DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4-context", type=Path, default=residual.DEFAULT_V4_CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=residual.DEFAULT_V4_CANDIDATE)
    parser.add_argument("--v4-folds", type=Path, default=residual.DEFAULT_V4_FOLDS)
    parser.add_argument("--query-dir", type=Path, default=residual.QUERY_DIR)
    parser.add_argument("--v2-oof-audit", type=Path, default=DEFAULT_V2_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    lock = acquire_lock(output_dir)
    run_state: RunState | None = None
    try:
        config = load_json(args.config.resolve(), "V3 config")
        validate_config(config)
        v2_config = load_json(args.v2_config.resolve(), "V2 config")
        v2.validate_config(v2_config)
        v1_config = load_json(args.v1_config.resolve(), "V1 config")
        residual.validate_config(v1_config)
        (
            records,
            v4_matrix,
            _v4_names,
            _contexts,
            fold_ids,
            document_groups,
            _candidate,
        ) = trace_eval.load_frozen_inputs(
            identity_path=args.identity.resolve(),
            pdf_dir=args.pdf_dir.resolve(),
            feature_path=args.v4_features.resolve(),
            manifest_path=args.v4_manifest.resolve(),
            context_path=args.v4_context.resolve(),
            candidate_path=args.v4_candidate.resolve(),
            fold_path=args.v4_folds.resolve(),
        )
        complete_identity = trace_freeze.load_three_column_identity(args.identity.resolve())
        multiplicity = Counter(record["REFNO"] for record in complete_identity)
        single_mask = np.asarray(
            [multiplicity[record["REFNO"]] == 1 for record in records], dtype=bool
        )
        single_indices = np.flatnonzero(single_mask)
        if len(single_indices) != 872:
            raise FulltextFusionError("single-target scope changed")
        style = v2.load_style_features(
            args.style_features.resolve(),
            args.style_manifest.resolve(),
            records,
            fold_ids,
            single_indices,
        )
        compact, pooled, query_single, _query_manifest, query_candidate = query_eval.load_query_features(
            query_dir=args.query_dir.resolve(),
            records=records,
            v4_fold_ids=fold_ids,
            expected_single_mask=single_mask,
        )
        if not np.array_equal(query_single, single_mask):
            raise FulltextFusionError("query feature scope changed")
        query_matrices = query_eval.build_matrices(v4_matrix, compact, pooled)
        domain, domain_manifest = residual.load_domain_features(
            args.domain_features.resolve(), args.domain_manifest.resolve(), records
        )
        page_cache = args.page_cache.resolve()
        if file_sha256(page_cache) != clean(
            (domain_manifest.get("inputs") or {}).get("page_cache", {}).get("sha256")
        ):
            raise FulltextFusionError("page cache changed")
        front_documents = residual.front_documents(
            records, page_cache, int(v1_config["front_text_expert"]["characters"])
        )
        all_full_documents = full_documents(records, page_cache)
        single_full_documents = [all_full_documents[int(index)] for index in single_indices]

        freeze_path = output_dir / FREEZE_NAME
        freeze_inputs = (
            ("config", args.config.resolve()),
            ("v2_model_config", args.v2_config.resolve()),
            ("v1_model_config", args.v1_config.resolve()),
            ("style_features", args.style_features.resolve()),
            ("style_manifest", args.style_manifest.resolve()),
            ("domain_features", args.domain_features.resolve()),
            ("domain_manifest", args.domain_manifest.resolve()),
            ("page_cache", page_cache),
            ("v4_features", args.v4_features.resolve()),
            ("v4_manifest", args.v4_manifest.resolve()),
            ("v4_context", args.v4_context.resolve()),
            ("v4_candidate", args.v4_candidate.resolve()),
            ("v4_folds", args.v4_folds.resolve()),
            ("query_features", args.query_dir.resolve() / retrieval.FEATURE_NAME),
            ("query_manifest", args.query_dir.resolve() / retrieval.MANIFEST_NAME),
            ("query_candidate", args.query_dir.resolve() / retrieval.CANDIDATE_NAME),
            ("v2_evaluator_code", Path(v2.__file__).resolve()),
            ("evaluator_code", Path(__file__).resolve()),
        )
        atomic_json(
            freeze_path,
            {
                "schema_version": "r3_fulltext_fusion_evaluation_freeze_v3",
                "status": "FROZEN_BEFORE_LABEL_ACCESS",
                "scope": "872 PDF-backed single-target papers at full coverage",
                "config": config,
                "inputs": {
                    name: {"path": str(path), "sha256": file_sha256(path)}
                    for name, path in freeze_inputs
                },
                "runtime_boundary": {
                    "labels_read": False,
                    "pmagresult_or_rockmag_read": False,
                    "prior_predictions_read": False,
                    "evaluation_results_read": False,
                    "outer_test_labels_used_for_training_or_selection": False,
                },
            },
        )
        run_state = RunState(output_dir / RUN_STATE_NAME, file_sha256(freeze_path))

        # First label access in this process.
        label_map = trace_eval.load_labels(
            args.targets.resolve(), {record["RESULTNO"] for record in records}
        )
        labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
        single_labels = labels[single_indices]
        global_to_single = np.full(len(records), -1, dtype=np.int32)
        global_to_single[single_indices] = np.arange(len(single_indices), dtype=np.int32)
        base_components = {
            name: np.full(len(single_indices), np.nan) for name in v2.BASE_COMPONENTS
        }
        knowledge_experts = {
            name: np.full(len(single_indices), np.nan) for name in v2.KNOWLEDGE_EXPERTS
        }
        fulltext_experts = {
            name: np.full(len(single_indices), np.nan) for name in FULLTEXT_EXPERTS
        }
        base_oof = np.full(len(single_indices), np.nan)
        knowledge_mean_oof = np.full(len(single_indices), np.nan)
        v2_oof = np.full(len(single_indices), np.nan)
        fulltext_mean_oof = np.full(len(single_indices), np.nan)
        primary_oof = np.full(len(single_indices), np.nan)
        fold_audit: list[dict[str, Any]] = []

        for outer_fold in range(1, 6):
            outer_test = np.flatnonzero(single_mask & (fold_ids == outer_fold))
            outer_all_train = np.flatnonzero(fold_ids != outer_fold)
            outer_single_train = np.flatnonzero(single_mask & (fold_ids != outer_fold))
            if set(map(clean, document_groups[outer_test])) & set(
                map(clean, document_groups[outer_all_train])
            ):
                raise AssertionError("outer fold leaks a normalized document")
            components = v2.fit_base_components(
                train=outer_all_train,
                test=outer_test,
                labels=labels,
                records=records,
                domain=domain,
                query_matrices=query_matrices,
                query_config=query_candidate["evaluation_candidates"],
                documents=front_documents,
                v1_config=v1_config,
            )
            train_local = global_to_single[outer_single_train]
            test_local = global_to_single[outer_test]
            knowledge = v2.fit_knowledge_experts(
                style.knowledge,
                train_local,
                test_local,
                single_labels,
                v2_config,
            )
            text_values = fit_fulltext_experts(
                single_full_documents,
                train_local,
                test_local,
                single_labels,
                config,
            )
            base = v2.base_fusion(components, v2_config)
            knowledge_mean = np.column_stack(
                [knowledge[name] for name in v2.KNOWLEDGE_EXPERTS]
            ).mean(axis=1)
            v2_probability = v2.primary_fusion(base, knowledge_mean, v2_config)
            text_mean = np.column_stack(
                [text_values[name] for name in FULLTEXT_EXPERTS]
            ).mean(axis=1)
            primary = primary_fusion(v2_probability, text_mean, config)
            for name, probability in components.items():
                base_components[name][test_local] = probability
            for name, probability in knowledge.items():
                knowledge_experts[name][test_local] = probability
            for name, probability in text_values.items():
                fulltext_experts[name][test_local] = probability
            base_oof[test_local] = base
            knowledge_mean_oof[test_local] = knowledge_mean
            v2_oof[test_local] = v2_probability
            fulltext_mean_oof[test_local] = text_mean
            primary_oof[test_local] = primary
            fold_audit.append(
                {
                    "outer_fold": outer_fold,
                    "all_pdf_training_rows": len(outer_all_train),
                    "single_training_rows": len(outer_single_train),
                    "test_rows": len(outer_test),
                    "outer_test_metrics": metric(labels[outer_test], primary),
                }
            )
            print(
                f"fulltext fusion OOF: fold={outer_fold}/5 "
                f"accuracy={fold_audit[-1]['outer_test_metrics']['accuracy']:.4f}",
                flush=True,
            )

        arrays: dict[str, np.ndarray] = {
            "source_row_indices": single_indices.astype(np.int32),
            "labels": single_labels.astype(np.int8),
            "fold_ids": fold_ids[single_indices].astype(np.int8),
            "no_strict_base": base_oof,
            "knowledge_mean3": knowledge_mean_oof,
            "v2_primary": v2_oof,
            "fulltext_word_mean3": fulltext_mean_oof,
            "primary": primary_oof,
        }
        arrays.update(
            {f"component__{name}": value for name, value in base_components.items()}
        )
        arrays.update(
            {f"expert__{name}": value for name, value in knowledge_experts.items()}
        )
        arrays.update(
            {f"expert__{name}": value for name, value in fulltext_experts.items()}
        )
        if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
            raise AssertionError("V3 OOF predictions are incomplete")
        provisional = output_dir / (OOF_NAME + ".preaudit")
        provisional.unlink(missing_ok=True)
        # V2 used the key `primary`; expose an alias only inside the audit archive.
        audit_arrays = dict(arrays)
        audit_arrays["primary"] = v2_oof
        atomic_npz(provisional, **audit_arrays)

        reproduction = audit_v2_reproduction(provisional, args.v2_oof_audit.resolve())
        # Replace the audit alias with the actual V3 primary before publishing.
        atomic_npz(provisional, **arrays)
        oof_path = output_dir / OOF_NAME
        provisional.replace(oof_path)
        with np.load(oof_path, allow_pickle=False) as frozen:
            frozen_labels = frozen["labels"].copy()
            frozen_v2 = frozen["v2_primary"].copy()
            frozen_text = frozen["fulltext_word_mean3"].copy()
            frozen_primary = frozen["primary"].copy()
        v2_metric = metric(frozen_labels, frozen_v2)
        text_metric = metric(frozen_labels, frozen_text)
        primary_metric = metric(frozen_labels, frozen_primary)

        receipt_rows = []
        for local_index, global_index in enumerate(single_indices):
            receipt_rows.append(
                {
                    "row": local_index,
                    "source_row_index": int(global_index),
                    "REFNO": records[int(global_index)]["REFNO"],
                    "RESULTNO": records[int(global_index)]["RESULTNO"],
                    "ROCKNAME": records[int(global_index)]["ROCKNAME"],
                    "fold": int(fold_ids[int(global_index)]),
                    "label": int(frozen_labels[local_index]),
                    "v2_probability": float(frozen_v2[local_index]),
                    "fulltext_probability": float(frozen_text[local_index]),
                    "probability": float(frozen_primary[local_index]),
                    "prediction": int(frozen_primary[local_index] >= 0.5),
                }
            )
        receipt_path = output_dir / RECEIPT_NAME
        atomic_json(
            receipt_path,
            {
                "schema_version": "r3_fulltext_fusion_single872_grouped_oof_receipt_v3",
                "rows": receipt_rows,
            },
        )
        report_path = output_dir / REPORT_NAME
        report = {
            "schema_version": "r3_fulltext_fusion_single872_grouped_oof_report_v3",
            "status": "POSTHOC_EVALUATION_OF_FROZEN_DEVELOPMENT_CANDIDATE",
            "scope": {"rows": len(single_indices), "coverage": 1.0},
            "primary_strategy": config["primary_fusion"]["name"],
            "threshold": 0.5,
            "metrics": {
                "v2_primary": v2_metric,
                "fulltext_word_mean3": text_metric,
                "primary": primary_metric,
                "fulltext_experts": {
                    name: metric(frozen_labels, fulltext_experts[name])
                    for name in FULLTEXT_EXPERTS
                },
            },
            "delta_vs_v2": delta_record(frozen_labels, frozen_v2, frozen_primary),
            "fold_audit": fold_audit,
            "v2_reproduction_audit": reproduction,
            "boundary": {
                "feature_generation_used_labels": False,
                "pmagresult_or_rockmag_read": False,
                "all_models_fit_inside_outer_training_partitions": True,
                "outer_test_labels_used_for_training_or_selection": False,
                "prior_oof_opened_only_after_provisional_predictions_written": True,
                "development_internal_oof_not_external_validation": True,
            },
            "promotion_decision": {
                "reaches_90_percent_full_coverage_internal_oof": primary_metric["accuracy"] >= 0.90,
                "promote_without_external_validation": False,
                "reason": "The internal target is reached, but the corpus informed candidate development and external validation remains required.",
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            "outputs": {
                "oof": {"path": str(oof_path), "sha256": file_sha256(oof_path)},
                "receipt": {"path": str(receipt_path), "sha256": file_sha256(receipt_path)},
            },
        }
        atomic_json(report_path, report)
        run_state.complete(report_path, oof_path, receipt_path)
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "v2_primary": v2_metric,
                    "fulltext_word_mean3": text_metric,
                    "primary": primary_metric,
                    "delta_vs_v2": report["delta_vs_v2"],
                    "promotion": report["promotion_decision"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as error:
        if run_state is not None:
            run_state.fail(error)
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "FULLTEXT_EXPERTS",
    "FulltextFusionError",
    "fit_fulltext_experts",
    "full_documents",
    "primary_fusion",
    "validate_config",
]
