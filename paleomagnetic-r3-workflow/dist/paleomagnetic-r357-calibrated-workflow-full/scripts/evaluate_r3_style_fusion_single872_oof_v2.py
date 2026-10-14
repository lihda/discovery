#!/usr/bin/env python3
"""Evaluate the frozen original-text knowledge fusion on 872 R3 papers."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_query_adaptive_retrieval_features as retrieval  # noqa: E402
import build_r3_style_features_single872_v2 as style_freeze  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import evaluate_r3_query_adaptive_single872_oof as query_eval  # noqa: E402
import evaluate_r3_residual_gate_single872_oof as residual  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402


DEFAULT_CONFIG = ROOT / "data/knowledge/r3_style_fusion_evaluation_v2.json"
DEFAULT_STYLE_DIR = ROOT / "tmp/r3_style_features_single872_v2"
DEFAULT_STYLE_FEATURES = DEFAULT_STYLE_DIR / style_freeze.FEATURE_NAME
DEFAULT_STYLE_MANIFEST = DEFAULT_STYLE_DIR / style_freeze.MANIFEST_NAME
DEFAULT_V1_CONFIG = ROOT / "data/knowledge/r3_residual_gate_evaluation_v1.json"
DEFAULT_V1_OOF = (
    ROOT
    / "tmp/r3_residual_gate_single872_v1"
    / "r3_residual_gate_single872_grouped_oof_v1.npz"
)
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_style_fusion_single872_v2"

FREEZE_NAME = "r3_style_fusion_evaluation_freeze_v2.json"
OOF_NAME = "r3_style_fusion_single872_grouped_oof_v2.npz"
RECEIPT_NAME = "r3_style_fusion_single872_grouped_oof_receipt_v2.json"
REPORT_NAME = "r3_style_fusion_single872_grouped_oof_report_v2.json"
RUN_STATE_NAME = "r3_style_fusion_evaluation_run_state_v2.json"

BASE_COMPONENTS = (
    "domain_mean4",
    "query_pca64_logreg",
    "front_word_logreg",
)
KNOWLEDGE_EXPERTS = (
    "knowledge_hgb",
    "knowledge_extra",
    "knowledge_gradient",
)


class StyleFusionError(ValueError):
    pass


def clean(value: Any) -> str:
    return residual.clean(value)


def file_sha256(path: Path) -> str:
    return trace_freeze.file_sha256(path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise StyleFusionError(f"{label} must be a JSON object")
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
    handle = (output_dir / ".r3_style_fusion_v2.lock").open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise StyleFusionError("another V2 style-fusion evaluation is running") from exc
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
                "schema_version": "r3_style_fusion_evaluation_run_state_v2",
                "status": "RUNNING",
                "pid": os.getpid(),
                "evaluation_freeze_sha256": freeze_sha256,
            },
        )

    def complete(self, report: Path, oof: Path, receipt: Path) -> None:
        atomic_json(
            self.path,
            {
                "schema_version": "r3_style_fusion_evaluation_run_state_v2",
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
                    "schema_version": "r3_style_fusion_evaluation_run_state_v2",
                    "status": "FAILED_OR_INTERRUPTED",
                    "evaluation_freeze_sha256": self.freeze_sha256,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != "r3_style_fusion_evaluation_v2"
        or config.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
    ):
        raise StyleFusionError("V2 evaluation config schema changed")
    protocol = config.get("protocol") or {}
    if (
        int(protocol.get("outer_folds", -1)) != 5
        or float(protocol.get("threshold", math.nan)) != 0.5
        or protocol.get("outer_test_selection") is not False
        or protocol.get("single_target_training_only_for_knowledge") is not True
    ):
        raise StyleFusionError("V2 evaluation protocol changed")
    base = config.get("base") or {}
    weights = base.get("weights") or {}
    if (
        tuple(base.get("components") or ()) != BASE_COMPONENTS
        or set(weights) != set(BASE_COMPONENTS)
        or not math.isclose(sum(map(float, weights.values())), 1.0, abs_tol=1e-12)
    ):
        raise StyleFusionError("V2 base fusion changed")
    knowledge = config.get("knowledge_experts") or {}
    if (
        set(knowledge)
        != {
            "hist_gradient_boosting",
            "extra_trees",
            "gradient_boosting",
            "aggregation",
        }
        or knowledge.get("aggregation") != "uniform_arithmetic_mean"
    ):
        raise StyleFusionError("V2 knowledge expert set changed")
    primary = config.get("primary_fusion") or {}
    if (
        float(primary.get("base_weight", math.nan)) != 0.85
        or float(primary.get("knowledge_weight", math.nan)) != 0.15
        or float(primary.get("threshold", math.nan)) != 0.5
        or not math.isclose(
            float(primary["base_weight"]) + float(primary["knowledge_weight"]),
            1.0,
            abs_tol=1e-12,
        )
    ):
        raise StyleFusionError("V2 primary fusion changed")
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
            raise StyleFusionError(f"invalid V2 runtime boundary: {name}")


@dataclass(frozen=True)
class StyleFeatures:
    knowledge: np.ndarray
    available: np.ndarray
    source_row_indices: np.ndarray
    fold_ids: np.ndarray
    manifest: dict[str, Any]


def load_style_features(
    path: Path,
    manifest_path: Path,
    records: Sequence[Mapping[str, str]],
    fold_ids: np.ndarray,
    single_indices: np.ndarray,
) -> StyleFeatures:
    manifest = load_json(manifest_path, "V2 style feature manifest")
    outputs = manifest.get("outputs") or {}
    if (
        manifest.get("schema_version") != style_freeze.MANIFEST_SCHEMA
        or manifest.get("feature_schema_version") != style_freeze.FEATURE_SCHEMA
        or manifest.get("status") != "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS"
        or int(manifest.get("rows", -1)) != len(single_indices)
        or int(manifest.get("knowledge_feature_count", -1)) != 84
        or file_sha256(path) != clean((outputs.get("features") or {}).get("sha256"))
    ):
        raise StyleFusionError("V2 style feature manifest is invalid")
    boundary = manifest.get("runtime_boundary") or {}
    for name in (
        "labels_read",
        "pmagresult_or_rockmag_read",
        "prior_predictions_read",
        "evaluation_results_read",
    ):
        if boundary.get(name) is not False:
            raise StyleFusionError(f"style feature boundary changed: {name}")
    for item in (manifest.get("inputs") or {}).get("code") or []:
        code_path = Path(clean(item.get("path"))).resolve()
        if not code_path.is_file() or file_sha256(code_path) != clean(item.get("sha256")):
            raise StyleFusionError(f"style feature source code changed: {code_path}")
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "knowledge",
            "knowledge_feature_names",
            "available",
            "source_row_indices",
            "fold_ids",
            "refnos",
            "resultnos",
        }
        if not required.issubset(archive.files):
            raise StyleFusionError("V2 style feature archive is incomplete")
        for name, expected in (manifest.get("array_sha256") or {}).items():
            if name not in archive.files or trace_freeze.array_sha256(archive[name]) != clean(expected):
                raise StyleFusionError(f"V2 style feature array changed: {name}")
        matrix = archive["knowledge"].astype(np.float64)
        names = archive["knowledge_feature_names"].astype(str)
        available = archive["available"].astype(bool)
        source = archive["source_row_indices"].astype(np.int32)
        saved_folds = archive["fold_ids"].astype(np.int8)
        refnos = archive["refnos"].astype(str)
        resultnos = archive["resultnos"].astype(str)
    expected_refnos = np.asarray([records[int(index)]["REFNO"] for index in single_indices])
    expected_resultnos = np.asarray([records[int(index)]["RESULTNO"] for index in single_indices])
    if (
        matrix.shape != (len(single_indices), 84)
        or names.shape != (84,)
        or available.shape != (len(single_indices),)
        or not available.all()
        or not np.array_equal(source, single_indices.astype(np.int32))
        or not np.array_equal(saved_folds, fold_ids[single_indices])
        or not np.array_equal(refnos, expected_refnos)
        or not np.array_equal(resultnos, expected_resultnos)
        or not np.isfinite(matrix).all()
    ):
        raise StyleFusionError("V2 style features do not align with V4")
    return StyleFeatures(matrix, available, source, saved_folds, manifest)


def predict_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    probability = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if probability.shape != (len(matrix),) or not np.isfinite(probability).all():
        raise AssertionError("model produced invalid probabilities")
    return probability


def fit_base_components(
    *,
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, str]],
    domain: np.ndarray,
    query_matrices: Mapping[str, np.ndarray],
    query_config: Mapping[str, Any],
    documents: Sequence[str],
    v1_config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    weights = trace_eval.inverse_identity_weights(train, records)
    domain_values: dict[str, np.ndarray] = {}
    for expert in residual.domain_specs(v1_config):
        fitted = nested.fit_expert(expert, domain, train, labels, weights)
        domain_values[expert.name] = fitted.predict(domain[test])
    domain_mean = np.column_stack(
        [domain_values[name] for name in ("domain_hgb", "domain_rf", "domain_extra", "domain_logreg")]
    ).mean(axis=1)
    q_config = v1_config["query_experts"]["pca"]
    q_spec = query_eval.RawSpec(
        clean(q_config["name"]),
        clean(q_config["family"]),
        clean(q_config["matrix"]),
        c_value=float(q_config["C"]),
        pca_dimension=int(q_config["pca_dimension"]),
    )
    query_probability = query_eval.fit_predict(
        q_spec,
        query_matrices[q_spec.matrix],
        train,
        test,
        labels,
        weights,
        query_config,
    )
    front_probability = residual.fit_front_text(
        train,
        test,
        labels,
        records,
        documents,
        v1_config["front_text_expert"],
    )
    output = {
        "domain_mean4": np.asarray(domain_mean, dtype=np.float64),
        "query_pca64_logreg": np.asarray(query_probability, dtype=np.float64),
        "front_word_logreg": np.asarray(front_probability, dtype=np.float64),
    }
    if set(output) != set(BASE_COMPONENTS) or any(
        value.shape != (len(test),) or not np.isfinite(value).all()
        for value in output.values()
    ):
        raise AssertionError("V2 base component contract changed")
    return output


def fit_knowledge_experts(
    matrix: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    seed = int(config["protocol"]["random_state"])
    specifications = config["knowledge_experts"]
    hgb = specifications["hist_gradient_boosting"]
    extra = specifications["extra_trees"]
    gradient = specifications["gradient_boosting"]
    models = {
        "knowledge_hgb": HistGradientBoostingClassifier(
            max_iter=int(hgb["max_iter"]),
            learning_rate=float(hgb["learning_rate"]),
            max_leaf_nodes=int(hgb["max_leaf_nodes"]),
            min_samples_leaf=int(hgb["min_samples_leaf"]),
            l2_regularization=float(hgb["l2_regularization"]),
            random_state=seed,
        ),
        "knowledge_extra": ExtraTreesClassifier(
            n_estimators=int(extra["n_estimators"]),
            max_features=clean(extra["max_features"]),
            min_samples_leaf=int(extra["min_samples_leaf"]),
            n_jobs=int(extra["n_jobs"]),
            random_state=seed,
        ),
        "knowledge_gradient": GradientBoostingClassifier(
            n_estimators=int(gradient["n_estimators"]),
            learning_rate=float(gradient["learning_rate"]),
            max_depth=int(gradient["max_depth"]),
            min_samples_leaf=int(gradient["min_samples_leaf"]),
            random_state=seed,
        ),
    }
    output: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(matrix[train], labels[train])
        output[name] = predict_probability(model, matrix[test])
    return output


def base_fusion(
    components: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> np.ndarray:
    weights = config["base"]["weights"]
    return sum(float(weights[name]) * components[name] for name in BASE_COMPONENTS)


def primary_fusion(
    base: np.ndarray, knowledge_mean: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    spec = config["primary_fusion"]
    return (
        float(spec["base_weight"]) * np.asarray(base)
        + float(spec["knowledge_weight"]) * np.asarray(knowledge_mean)
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


def audit_v1_reproduction(provisional: Path, v1_oof: Path) -> dict[str, Any]:
    if not v1_oof.is_file():
        raise StyleFusionError("V1 OOF audit file is missing")
    maximum: dict[str, float] = {}
    with np.load(provisional, allow_pickle=False) as observed, np.load(
        v1_oof, allow_pickle=False
    ) as expected:
        for name in ("source_row_indices", "labels", "fold_ids"):
            if not np.array_equal(observed[name], expected[name]):
                raise StyleFusionError(f"V1 audit alignment mismatch: {name}")
        mapping = {
            "domain_mean4": "component__domain_mean4",
            "query_pca64_logreg": "component__query_pca64_logreg",
            "front_word_logreg": "component__front_word_logreg",
        }
        for name, v1_name in mapping.items():
            difference = np.abs(observed[f"component__{name}"] - expected[v1_name])
            maximum[name] = float(difference.max(initial=0.0))
            if not np.allclose(
                observed[f"component__{name}"], expected[v1_name], rtol=0.0, atol=1e-12
            ):
                raise StyleFusionError(f"V1 component reproduction failed: {name}")
        expected_base = (
            0.43 * expected["component__domain_mean4"]
            + 0.27 * expected["component__query_pca64_logreg"]
            + 0.25 * expected["component__front_word_logreg"]
        ) / 0.95
        maximum["no_strict_base"] = float(
            np.abs(observed["no_strict_base"] - expected_base).max(initial=0.0)
        )
        if not np.allclose(observed["no_strict_base"], expected_base, rtol=0.0, atol=1e-12):
            raise StyleFusionError("V1 no-strict base reproduction failed")
    return {
        "v1_oof": {"path": str(v1_oof), "sha256": file_sha256(v1_oof)},
        "opened_only_after_provisional_predictions_written": True,
        "alignment_exact": True,
        "component_maximum_absolute_difference": maximum,
        "components_reproduced_at_tolerance_1e_12": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--v1-config", type=Path, default=DEFAULT_V1_CONFIG)
    parser.add_argument("--style-features", type=Path, default=DEFAULT_STYLE_FEATURES)
    parser.add_argument("--style-manifest", type=Path, default=DEFAULT_STYLE_MANIFEST)
    parser.add_argument("--domain-features", type=Path, default=residual.DEFAULT_DOMAIN_FEATURES)
    parser.add_argument("--domain-manifest", type=Path, default=residual.DEFAULT_DOMAIN_MANIFEST)
    parser.add_argument("--page-cache", type=Path, default=residual.DEFAULT_PAGE_CACHE)
    parser.add_argument("--v4-features", type=Path, default=residual.DEFAULT_V4_FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=residual.DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4-context", type=Path, default=residual.DEFAULT_V4_CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=residual.DEFAULT_V4_CANDIDATE)
    parser.add_argument("--v4-folds", type=Path, default=residual.DEFAULT_V4_FOLDS)
    parser.add_argument("--query-dir", type=Path, default=residual.QUERY_DIR)
    parser.add_argument("--v1-oof-audit", type=Path, default=DEFAULT_V1_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    lock = acquire_lock(output_dir)
    run_state: RunState | None = None
    try:
        config = load_json(args.config.resolve(), "V2 evaluation config")
        validate_config(config)
        v1_config = load_json(args.v1_config.resolve(), "V1 residual config")
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
            raise StyleFusionError("single-target scope changed")
        style = load_style_features(
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
            raise StyleFusionError("query feature scope changed")
        query_matrices = query_eval.build_matrices(v4, compact, pooled)
        domain, domain_manifest = residual.load_domain_features(
            args.domain_features.resolve(), args.domain_manifest.resolve(), records
        )
        page_cache = args.page_cache.resolve()
        if file_sha256(page_cache) != clean(
            (domain_manifest.get("inputs") or {}).get("page_cache", {}).get("sha256")
        ):
            raise StyleFusionError("front-text page cache changed")
        documents = residual.front_documents(
            records, page_cache, int(v1_config["front_text_expert"]["characters"])
        )

        freeze_path = output_dir / FREEZE_NAME
        freeze_inputs = (
            ("config", args.config.resolve()),
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
            ("evaluator_code", Path(__file__).resolve()),
        )
        freeze = {
            "schema_version": "r3_style_fusion_evaluation_freeze_v2",
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
        }
        atomic_json(freeze_path, freeze)
        run_state = RunState(output_dir / RUN_STATE_NAME, file_sha256(freeze_path))

        # First label access in this process.
        label_map = trace_eval.load_labels(
            args.targets.resolve(), {record["RESULTNO"] for record in records}
        )
        labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
        single_labels = labels[single_indices]
        global_to_single = np.full(len(records), -1, dtype=np.int32)
        global_to_single[single_indices] = np.arange(len(single_indices), dtype=np.int32)
        component_oof = {
            name: np.full(len(single_indices), np.nan, dtype=np.float64)
            for name in BASE_COMPONENTS
        }
        knowledge_oof = {
            name: np.full(len(single_indices), np.nan, dtype=np.float64)
            for name in KNOWLEDGE_EXPERTS
        }
        base_oof = np.full(len(single_indices), np.nan, dtype=np.float64)
        knowledge_mean_oof = np.full(len(single_indices), np.nan, dtype=np.float64)
        primary_oof = np.full(len(single_indices), np.nan, dtype=np.float64)
        fold_audit: list[dict[str, Any]] = []

        for outer_fold in range(1, 6):
            outer_test = np.flatnonzero(single_mask & (fold_ids == outer_fold))
            outer_all_train = np.flatnonzero(fold_ids != outer_fold)
            outer_single_train = np.flatnonzero(single_mask & (fold_ids != outer_fold))
            test_groups = set(map(clean, document_groups[outer_test]))
            if test_groups & set(map(clean, document_groups[outer_all_train])):
                raise AssertionError("outer fold leaks a normalized document")
            components = fit_base_components(
                train=outer_all_train,
                test=outer_test,
                labels=labels,
                records=records,
                domain=domain,
                query_matrices=query_matrices,
                query_config=query_candidate["evaluation_candidates"],
                documents=documents,
                v1_config=v1_config,
            )
            train_local = global_to_single[outer_single_train]
            test_local = global_to_single[outer_test]
            knowledge = fit_knowledge_experts(
                style.knowledge,
                train_local,
                test_local,
                single_labels,
                config,
            )
            base = base_fusion(components, config)
            knowledge_mean = np.column_stack(
                [knowledge[name] for name in KNOWLEDGE_EXPERTS]
            ).mean(axis=1)
            primary = primary_fusion(base, knowledge_mean, config)
            for name, probability in components.items():
                component_oof[name][test_local] = probability
            for name, probability in knowledge.items():
                knowledge_oof[name][test_local] = probability
            base_oof[test_local] = base
            knowledge_mean_oof[test_local] = knowledge_mean
            primary_oof[test_local] = primary
            fold_audit.append(
                {
                    "outer_fold": outer_fold,
                    "all_pdf_training_rows": len(outer_all_train),
                    "single_knowledge_training_rows": len(outer_single_train),
                    "test_rows": len(outer_test),
                    "outer_test_metrics": metric(labels[outer_test], primary),
                }
            )
            print(
                f"style fusion OOF: fold={outer_fold}/5 "
                f"accuracy={fold_audit[-1]['outer_test_metrics']['accuracy']:.4f}",
                flush=True,
            )

        arrays: dict[str, np.ndarray] = {
            "source_row_indices": single_indices.astype(np.int32),
            "labels": single_labels.astype(np.int8),
            "fold_ids": fold_ids[single_indices].astype(np.int8),
            "no_strict_base": base_oof,
            "knowledge_mean3": knowledge_mean_oof,
            "primary": primary_oof,
        }
        arrays.update(
            {f"component__{name}": value for name, value in component_oof.items()}
        )
        arrays.update(
            {f"expert__{name}": value for name, value in knowledge_oof.items()}
        )
        if any(not np.isfinite(value).all() for value in arrays.values() if value.dtype.kind == "f"):
            raise AssertionError("V2 OOF predictions are incomplete")
        provisional = output_dir / (OOF_NAME + ".preaudit")
        provisional.unlink(missing_ok=True)
        atomic_npz(provisional, **arrays)

        # Prior predictions are first opened here, after V2 predictions are immutable.
        reproduction = audit_v1_reproduction(provisional, args.v1_oof_audit.resolve())
        oof_path = output_dir / OOF_NAME
        provisional.replace(oof_path)
        with np.load(oof_path, allow_pickle=False) as frozen:
            frozen_labels = frozen["labels"].copy()
            frozen_base = frozen["no_strict_base"].copy()
            frozen_primary = frozen["primary"].copy()
            frozen_knowledge = frozen["knowledge_mean3"].copy()
        base_metric = metric(frozen_labels, frozen_base)
        primary_metric = metric(frozen_labels, frozen_primary)
        knowledge_metric = metric(frozen_labels, frozen_knowledge)

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
                    "base_probability": float(frozen_base[local_index]),
                    "knowledge_probability": float(frozen_knowledge[local_index]),
                    "probability": float(frozen_primary[local_index]),
                    "prediction": int(frozen_primary[local_index] >= 0.5),
                }
            )
        receipt_path = output_dir / RECEIPT_NAME
        atomic_json(
            receipt_path,
            {
                "schema_version": "r3_style_fusion_single872_grouped_oof_receipt_v2",
                "rows": receipt_rows,
            },
        )
        report_path = output_dir / REPORT_NAME
        report = {
            "schema_version": "r3_style_fusion_single872_grouped_oof_report_v2",
            "status": "POSTHOC_EVALUATION_OF_FROZEN_DEVELOPMENT_CANDIDATE",
            "scope": {"rows": len(single_indices), "coverage": 1.0},
            "primary_strategy": config["primary_fusion"]["name"],
            "threshold": 0.5,
            "metrics": {
                "no_strict_base": base_metric,
                "knowledge_mean3": knowledge_metric,
                "primary": primary_metric,
                "knowledge_experts": {
                    name: metric(frozen_labels, knowledge_oof[name])
                    for name in KNOWLEDGE_EXPERTS
                },
            },
            "delta_vs_no_strict_base": delta_record(
                frozen_labels, frozen_base, frozen_primary
            ),
            "fold_audit": fold_audit,
            "v1_reproduction_audit": reproduction,
            "boundary": {
                "feature_generation_used_labels": False,
                "pmagresult_or_rockmag_read": False,
                "all_models_fit_inside_outer_training_partitions": True,
                "knowledge_models_use_only_single_target_outer_train_rows": True,
                "outer_test_labels_used_for_training_or_selection": False,
                "prior_oof_opened_only_after_provisional_predictions_written": True,
                "development_internal_oof_not_external_validation": True,
            },
            "promotion_decision": {
                "reaches_90_percent_full_coverage_internal_oof": primary_metric["accuracy"] >= 0.90,
                "promote_without_external_validation": False,
                "reason": "The 872-paper corpus informed candidate development; a locked external set remains required.",
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
                    "no_strict_base": base_metric,
                    "knowledge_mean3": knowledge_metric,
                    "primary": primary_metric,
                    "delta": report["delta_vs_no_strict_base"],
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
    "BASE_COMPONENTS",
    "KNOWLEDGE_EXPERTS",
    "StyleFusionError",
    "base_fusion",
    "fit_knowledge_experts",
    "primary_fusion",
    "validate_config",
]
