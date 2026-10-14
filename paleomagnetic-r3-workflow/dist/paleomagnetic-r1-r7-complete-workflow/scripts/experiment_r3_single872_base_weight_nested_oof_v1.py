#!/usr/bin/env python3
"""Explore and nested-validate the three R3 single-target base weights."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_query_adaptive_retrieval_features as retrieval  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import evaluate_r3_fulltext_fusion_single872_oof_v3 as v3  # noqa: E402
import evaluate_r3_query_adaptive_single872_oof as query_eval  # noqa: E402
import evaluate_r3_residual_gate_single872_oof as residual  # noqa: E402
import evaluate_r3_style_fusion_single872_oof_v2 as v2  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_grep_pairwise_rank_oof as pair_eval  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_single872_base_weight_nested_oof_v1"
DEFAULT_V3_OOF = (
    ROOT
    / "tmp/r3_fulltext_fusion_single872_v3"
    / "r3_fulltext_fusion_single872_grouped_oof_v3.npz"
)
FREEZE_NAME = "r3_single872_base_weight_candidate_freeze_v1.json"
OOF_NAME = "r3_single872_base_weight_nested_oof_v1.npz"
REPORT_NAME = "r3_single872_base_weight_nested_oof_report_v1.json"

COMPONENTS = v2.BASE_COMPONENTS
GRID_STEP = 0.01
INNER_SPLITS = 4
INNER_SEED = 35731


class BaseWeightExperimentError(ValueError):
    pass


def clean(value: Any) -> str:
    return residual.clean(value)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BaseWeightExperimentError(f"JSON object required: {path}")
    return value


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": trace_freeze.file_sha256(path)}


def weight_candidates(
    current: Sequence[float], *, step: float = GRID_STEP
) -> np.ndarray:
    units = int(round(1.0 / step))
    if units <= 0 or not np.isclose(units * step, 1.0):
        raise ValueError("weight step must exactly divide one")
    rows = [
        (first / units, second / units, (units - first - second) / units)
        for first in range(units + 1)
        for second in range(units - first + 1)
    ]
    current_row = tuple(float(value) for value in current)
    if len(current_row) != 3 or not np.isclose(sum(current_row), 1.0):
        raise ValueError("current weights must contain three values summing to one")
    if not any(np.allclose(row, current_row, rtol=0.0, atol=1e-12) for row in rows):
        rows.append(current_row)
    output = np.asarray(rows, dtype=np.float64)
    if (
        output.ndim != 2
        or output.shape[1] != 3
        or np.any(output < 0.0)
        or not np.allclose(output.sum(axis=1), 1.0)
    ):
        raise AssertionError("invalid weight candidate grid")
    return output


def base_probability(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or weight.shape != (3,):
        raise ValueError("base fusion expects an n-by-3 matrix and three weights")
    if not np.isclose(weight.sum(), 1.0) or np.any(weight < 0.0):
        raise ValueError("base weights must be nonnegative and sum to one")
    # A three-column elementwise reduction avoids noisy small-matrix BLAS paths.
    probability = np.sum(values * weight[np.newaxis, :], axis=1)
    if not np.isfinite(probability).all():
        raise AssertionError("base fusion produced non-finite probabilities")
    return probability


def final_probability(
    base: np.ndarray,
    knowledge: np.ndarray,
    fulltext: np.ndarray,
    v2_config: Mapping[str, Any],
    v3_config: Mapping[str, Any],
) -> np.ndarray:
    v2_probability = v2.primary_fusion(base, knowledge, v2_config)
    return v3.primary_fusion(v2_probability, fulltext, v3_config)


def correct_count(labels: np.ndarray, probability: np.ndarray) -> int:
    return int(((np.asarray(probability) >= 0.5) == np.asarray(labels)).sum())


def balanced_accuracy(labels: np.ndarray, probability: np.ndarray) -> float:
    observed = trace_eval.metric(np.asarray(labels), np.asarray(probability))
    return float(observed["balanced_accuracy"])


def candidate_record(
    weights: np.ndarray,
    labels: np.ndarray,
    components: np.ndarray,
    knowledge: np.ndarray,
    fulltext: np.ndarray,
    current_weights: np.ndarray,
    v2_config: Mapping[str, Any],
    v3_config: Mapping[str, Any],
) -> dict[str, Any]:
    base = base_probability(components, weights)
    final = final_probability(base, knowledge, fulltext, v2_config, v3_config)
    return {
        "weights": {
            name: float(value) for name, value in zip(COMPONENTS, weights)
        },
        "final_correct": correct_count(labels, final),
        "final_balanced_accuracy": balanced_accuracy(labels, final),
        "base_correct": correct_count(labels, base),
        "base_balanced_accuracy": balanced_accuracy(labels, base),
        "distance_from_current_l1": float(np.abs(weights - current_weights).sum()),
    }


def selection_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    weights = record["weights"]
    return (
        int(record["final_correct"]),
        float(record["final_balanced_accuracy"]),
        int(record["base_correct"]),
        -float(record["distance_from_current_l1"]),
        -float(weights[COMPONENTS[0]]),
        -float(weights[COMPONENTS[1]]),
        -float(weights[COMPONENTS[2]]),
    )


def select_weights(
    candidates: np.ndarray,
    labels: np.ndarray,
    components: np.ndarray,
    knowledge: np.ndarray,
    fulltext: np.ndarray,
    current_weights: np.ndarray,
    v2_config: Mapping[str, Any],
    v3_config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    best_weights: np.ndarray | None = None
    best_record: dict[str, Any] | None = None
    for weights in candidates:
        record = candidate_record(
            weights,
            labels,
            components,
            knowledge,
            fulltext,
            current_weights,
            v2_config,
            v3_config,
        )
        if best_record is None or selection_key(record) > selection_key(best_record):
            best_weights = weights.copy()
            best_record = record
    if best_weights is None or best_record is None:
        raise AssertionError("weight selection produced no candidate")
    return best_weights, best_record


def component_matrix(values: Mapping[str, np.ndarray]) -> np.ndarray:
    matrix = np.column_stack([values[name] for name in COMPONENTS]).astype(np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3 or not np.isfinite(matrix).all():
        raise AssertionError("base component matrix is invalid")
    return matrix


def fit_base_components_quiet(**kwargs: Any) -> dict[str, np.ndarray]:
    """Run the established base fit while checking, rather than printing, PCA warnings."""

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            module=r"sklearn\.(utils\.extmath|decomposition\._base)",
        )
        output = v2.fit_base_components(**kwargs)
    if any(not np.isfinite(values).all() for values in output.values()):
        raise AssertionError("base component fit produced non-finite probabilities")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--v3-config", type=Path, default=v3.DEFAULT_CONFIG)
    parser.add_argument("--v2-config", type=Path, default=v3.DEFAULT_V2_CONFIG)
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
    parser.add_argument("--v3-oof-audit", type=Path, default=DEFAULT_V3_OOF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    paths = {name: value.resolve() for name, value in vars(args).items() if isinstance(value, Path)}
    output_dir = paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    v3_config = load_json(paths["v3_config"])
    v3.validate_config(v3_config)
    v2_config = load_json(paths["v2_config"])
    v2.validate_config(v2_config)
    v1_config = load_json(paths["v1_config"])
    residual.validate_config(v1_config)
    current_weights = np.asarray(
        [float(v2_config["base"]["weights"][name]) for name in COMPONENTS],
        dtype=np.float64,
    )
    candidates = weight_candidates(current_weights)

    (
        records,
        v4_matrix,
        _v4_names,
        _contexts,
        fold_ids,
        document_groups,
        _candidate,
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
    single_indices = np.flatnonzero(single_mask)
    if len(single_indices) != 872:
        raise BaseWeightExperimentError("single-target scope changed")
    style = v2.load_style_features(
        paths["style_features"],
        paths["style_manifest"],
        records,
        fold_ids,
        single_indices,
    )
    compact, pooled, query_single, _query_manifest, query_candidate = (
        query_eval.load_query_features(
            query_dir=paths["query_dir"],
            records=records,
            v4_fold_ids=fold_ids,
            expected_single_mask=single_mask,
        )
    )
    if not np.array_equal(query_single, single_mask):
        raise BaseWeightExperimentError("query feature scope changed")
    query_matrices = query_eval.build_matrices(v4_matrix, compact, pooled)
    domain, domain_manifest = residual.load_domain_features(
        paths["domain_features"], paths["domain_manifest"], records
    )
    if trace_freeze.file_sha256(paths["page_cache"]) != clean(
        (domain_manifest.get("inputs") or {}).get("page_cache", {}).get("sha256")
    ):
        raise BaseWeightExperimentError("page cache changed")
    front_documents = residual.front_documents(
        records,
        paths["page_cache"],
        int(v1_config["front_text_expert"]["characters"]),
    )
    full_documents = v3.full_documents(records, paths["page_cache"])
    single_documents = [full_documents[int(index)] for index in single_indices]

    freeze_path = output_dir / FREEZE_NAME
    freeze = {
        "schema_version": "r3_single872_base_weight_candidate_freeze_v1",
        "status": "FROZEN_BEFORE_LABEL_ACCESS",
        "scope": "872 PDF-backed single-target papers",
        "candidate_space": {
            "components": list(COMPONENTS),
            "simplex_grid_step": GRID_STEP,
            "includes_zero_weight_ablations": True,
            "current_weights_added_exactly": True,
            "candidate_count": len(candidates),
        },
        "selection": {
            "outer_folds": 5,
            "inner_folds": INNER_SPLITS,
            "inner_group": "normalized document identity",
            "objective_order": [
                "final_accuracy",
                "final_balanced_accuracy",
                "base_accuracy",
                "minimum_L1_distance_from_current",
            ],
            "threshold": 0.5,
            "downstream_fusions_remain_fixed": True,
        },
        "inputs": {
            name: file_record(path)
            for name, path in paths.items()
            if name not in {"output_dir", "v3_oof_audit"} and path.is_file()
        },
        "boundary": {
            "features_frozen_before_labels": True,
            "pmagresult_or_rockmag_used_as_feature": False,
            "prior_predictions_or_evaluation_reports_used_for_selection": False,
            "outer_test_labels_used_for_weight_selection": False,
            "global_grid_result_is_posthoc_development_only": True,
        },
    }
    v2.atomic_json(freeze_path, freeze)

    # First label access in this process.
    label_map = trace_eval.load_labels(
        paths["targets"], {record["RESULTNO"] for record in records}
    )
    labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
    single_labels = labels[single_indices]
    global_to_single = np.full(len(records), -1, dtype=np.int32)
    global_to_single[single_indices] = np.arange(len(single_indices), dtype=np.int32)

    outer_components = {
        name: np.full(len(single_indices), np.nan, dtype=np.float64) for name in COMPONENTS
    }
    outer_knowledge = np.full(len(single_indices), np.nan, dtype=np.float64)
    outer_fulltext = np.full(len(single_indices), np.nan, dtype=np.float64)
    current_base = np.full(len(single_indices), np.nan, dtype=np.float64)
    current_final = np.full(len(single_indices), np.nan, dtype=np.float64)
    nested_base = np.full(len(single_indices), np.nan, dtype=np.float64)
    nested_final = np.full(len(single_indices), np.nan, dtype=np.float64)
    selected_weight_rows = np.full((len(single_indices), 3), np.nan, dtype=np.float64)
    fold_audit: list[dict[str, Any]] = []

    for outer_fold in range(1, 6):
        outer_test = np.flatnonzero(single_mask & (fold_ids == outer_fold))
        outer_all_train = np.flatnonzero(fold_ids != outer_fold)
        outer_single_train = np.flatnonzero(single_mask & (fold_ids != outer_fold))
        if set(map(clean, document_groups[outer_test])) & set(
            map(clean, document_groups[outer_all_train])
        ):
            raise AssertionError("outer fold leaks a normalized document")

        inner_components = {
            name: np.full(len(single_indices), np.nan, dtype=np.float64)
            for name in COMPONENTS
        }
        inner_knowledge = np.full(len(single_indices), np.nan, dtype=np.float64)
        inner_fulltext = np.full(len(single_indices), np.nan, dtype=np.float64)
        inner_folds = nested.deterministic_group_folds(
            outer_all_train,
            [clean(document_groups[int(index)]) for index in outer_all_train],
            n_splits=INNER_SPLITS,
            seed=INNER_SEED + outer_fold,
        )
        for inner_fold, (inner_train, inner_validation) in enumerate(inner_folds, start=1):
            validation_single = inner_validation[single_mask[inner_validation]]
            training_single = inner_train[single_mask[inner_train]]
            if not len(validation_single) or not len(training_single):
                raise AssertionError("inner fold lacks single-target rows")
            values = fit_base_components_quiet(
                train=inner_train,
                test=validation_single,
                labels=labels,
                records=records,
                domain=domain,
                query_matrices=query_matrices,
                query_config=query_candidate["evaluation_candidates"],
                documents=front_documents,
                v1_config=v1_config,
            )
            train_local = global_to_single[training_single]
            validation_local = global_to_single[validation_single]
            knowledge_values = v2.fit_knowledge_experts(
                style.knowledge,
                train_local,
                validation_local,
                single_labels,
                v2_config,
            )
            fulltext_values = v3.fit_fulltext_experts(
                single_documents,
                train_local,
                validation_local,
                single_labels,
                v3_config,
            )
            for name in COMPONENTS:
                inner_components[name][validation_local] = values[name]
            inner_knowledge[validation_local] = np.column_stack(
                [knowledge_values[name] for name in v2.KNOWLEDGE_EXPERTS]
            ).mean(axis=1)
            inner_fulltext[validation_local] = np.column_stack(
                [fulltext_values[name] for name in v3.FULLTEXT_EXPERTS]
            ).mean(axis=1)
            print(
                f"base-weight nested OOF: outer={outer_fold}/5 inner={inner_fold}/{INNER_SPLITS}",
                flush=True,
            )

        selection_local = global_to_single[outer_single_train]
        if any(
            not np.isfinite(value[selection_local]).all()
            for value in (*inner_components.values(), inner_knowledge, inner_fulltext)
        ):
            raise AssertionError("inner OOF predictions are incomplete")
        inner_component_matrix = np.column_stack(
            [inner_components[name][selection_local] for name in COMPONENTS]
        ).astype(np.float64)
        if not np.isfinite(inner_component_matrix).all():
            raise AssertionError("inner component selection matrix is incomplete")
        selected_weights, inner_selection = select_weights(
            candidates,
            single_labels[selection_local],
            inner_component_matrix,
            inner_knowledge[selection_local],
            inner_fulltext[selection_local],
            current_weights,
            v2_config,
            v3_config,
        )

        test_values = fit_base_components_quiet(
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
        outer_train_local = global_to_single[outer_single_train]
        outer_test_local = global_to_single[outer_test]
        knowledge_values = v2.fit_knowledge_experts(
            style.knowledge,
            outer_train_local,
            outer_test_local,
            single_labels,
            v2_config,
        )
        fulltext_values = v3.fit_fulltext_experts(
            single_documents,
            outer_train_local,
            outer_test_local,
            single_labels,
            v3_config,
        )
        knowledge_mean = np.column_stack(
            [knowledge_values[name] for name in v2.KNOWLEDGE_EXPERTS]
        ).mean(axis=1)
        fulltext_mean = np.column_stack(
            [fulltext_values[name] for name in v3.FULLTEXT_EXPERTS]
        ).mean(axis=1)
        matrix = component_matrix(test_values)
        current_base_values = base_probability(matrix, current_weights)
        nested_base_values = base_probability(matrix, selected_weights)
        current_final_values = final_probability(
            current_base_values, knowledge_mean, fulltext_mean, v2_config, v3_config
        )
        nested_final_values = final_probability(
            nested_base_values, knowledge_mean, fulltext_mean, v2_config, v3_config
        )
        for name in COMPONENTS:
            outer_components[name][outer_test_local] = test_values[name]
        outer_knowledge[outer_test_local] = knowledge_mean
        outer_fulltext[outer_test_local] = fulltext_mean
        current_base[outer_test_local] = current_base_values
        current_final[outer_test_local] = current_final_values
        nested_base[outer_test_local] = nested_base_values
        nested_final[outer_test_local] = nested_final_values
        selected_weight_rows[outer_test_local] = selected_weights
        fold_audit.append(
            {
                "outer_fold": outer_fold,
                "all_training_rows": len(outer_all_train),
                "single_training_rows": len(outer_single_train),
                "single_test_rows": len(outer_test),
                "selected_weights": inner_selection["weights"],
                "inner_selection_metrics": inner_selection,
                "outer_current_metrics": trace_eval.metric(
                    single_labels[outer_test_local], current_final_values
                ),
                "outer_nested_metrics": trace_eval.metric(
                    single_labels[outer_test_local], nested_final_values
                ),
            }
        )
        print(
            f"base-weight nested OOF: outer={outer_fold}/5 selected="
            f"{selected_weights.tolist()}",
            flush=True,
        )

    arrays_to_check = (
        *outer_components.values(),
        outer_knowledge,
        outer_fulltext,
        current_base,
        current_final,
        nested_base,
        nested_final,
        selected_weight_rows,
    )
    if any(not np.isfinite(value).all() for value in arrays_to_check):
        raise AssertionError("nested OOF outputs are incomplete")

    all_component_matrix = component_matrix(outer_components)
    global_records = [
        candidate_record(
            weights,
            single_labels,
            all_component_matrix,
            outer_knowledge,
            outer_fulltext,
            current_weights,
            v2_config,
            v3_config,
        )
        for weights in candidates
    ]
    global_ranked = sorted(global_records, key=selection_key, reverse=True)
    base_ranked = sorted(
        global_records,
        key=lambda item: (
            int(item["base_correct"]),
            float(item["base_balanced_accuracy"]),
            int(item["final_correct"]),
            -float(item["distance_from_current_l1"]),
        ),
        reverse=True,
    )
    global_best_weights = np.asarray(
        [global_ranked[0]["weights"][name] for name in COMPONENTS], dtype=np.float64
    )
    global_best_base = base_probability(all_component_matrix, global_best_weights)
    global_best_final = final_probability(
        global_best_base, outer_knowledge, outer_fulltext, v2_config, v3_config
    )
    base_best_weights = np.asarray(
        [base_ranked[0]["weights"][name] for name in COMPONENTS], dtype=np.float64
    )
    base_best_probability = base_probability(all_component_matrix, base_best_weights)
    base_best_final = final_probability(
        base_best_probability, outer_knowledge, outer_fulltext, v2_config, v3_config
    )
    maximum_final_correct = int(global_ranked[0]["final_correct"])
    maximum_base_correct = max(int(item["base_correct"]) for item in global_records)

    oof_path = output_dir / OOF_NAME
    v2.atomic_npz(
        oof_path,
        source_row_indices=single_indices.astype(np.int32),
        labels=single_labels.astype(np.int8),
        fold_ids=fold_ids[single_indices].astype(np.int8),
        current_weights=current_weights,
        selected_weight_rows=selected_weight_rows,
        current_base=current_base,
        current_final=current_final,
        nested_selected_base=nested_base,
        nested_selected_final=nested_final,
        global_development_best_weights=global_best_weights,
        global_development_best_base=global_best_base,
        global_development_best_final=global_best_final,
        base_development_best_weights=base_best_weights,
        base_development_best_probability=base_best_probability,
        base_development_best_final=base_best_final,
        knowledge_mean3=outer_knowledge,
        fulltext_word_mean3=outer_fulltext,
        **{f"component__{name}": value for name, value in outer_components.items()},
    )

    # Prior V3 predictions are opened only as a post-prediction reproduction audit.
    with np.load(paths["v3_oof_audit"], allow_pickle=False) as prior:
        for name, observed in (
            ("source_row_indices", single_indices.astype(np.int32)),
            ("labels", single_labels.astype(np.int8)),
            ("fold_ids", fold_ids[single_indices].astype(np.int8)),
        ):
            if not np.array_equal(prior[name], observed):
                raise BaseWeightExperimentError(f"V3 audit alignment mismatch: {name}")
        reproduction = {
            "current_base": float(np.max(np.abs(prior["no_strict_base"] - current_base))),
            "knowledge_mean3": float(
                np.max(np.abs(prior["knowledge_mean3"] - outer_knowledge))
            ),
            "fulltext_word_mean3": float(
                np.max(np.abs(prior["fulltext_word_mean3"] - outer_fulltext))
            ),
            "current_final": float(np.max(np.abs(prior["primary"] - current_final))),
        }
    if any(value > 1e-12 for value in reproduction.values()):
        raise BaseWeightExperimentError("current V3 reproduction failed")

    current_metric = trace_eval.metric(single_labels, current_final)
    nested_metric = trace_eval.metric(single_labels, nested_final)
    global_metric = trace_eval.metric(single_labels, global_best_final)
    current_base_metric = trace_eval.metric(single_labels, current_base)
    nested_base_metric = trace_eval.metric(single_labels, nested_base)
    base_best_metric = trace_eval.metric(single_labels, base_best_probability)
    base_best_final_metric = trace_eval.metric(single_labels, base_best_final)
    single_records = [records[int(index)] for index in single_indices]
    nested_bootstrap = pair_eval.clustered_bootstrap_delta(
        single_records,
        single_labels,
        current_final,
        nested_final,
        np.ones(len(single_labels), dtype=bool),
    )
    report = {
        "schema_version": "r3_single872_base_weight_nested_oof_report_v1",
        "status": "COMPLETE_INTERNAL_NESTED_OOF_AND_POSTHOC_GRID",
        "scope": {"rows": len(single_indices), "coverage": 1.0},
        "components": list(COMPONENTS),
        "current_weights": {
            name: float(value) for name, value in zip(COMPONENTS, current_weights)
        },
        "candidate_grid": {
            "step": GRID_STEP,
            "candidate_count": len(candidates),
            "threshold": 0.5,
        },
        "strict_nested_oof": {
            "current_fixed": {
                "base": current_base_metric,
                "final_v3": current_metric,
            },
            "inner_selected_weights": {
                "base": nested_base_metric,
                "final_v3": nested_metric,
            },
            "delta": v3.delta_record(single_labels, current_final, nested_final),
            "folds": fold_audit,
            "paired_refno_cluster_bootstrap": nested_bootstrap,
        },
        "posthoc_full_oof_grid": {
            "development_only": True,
            "best_for_final_v3": {
                "weights": global_ranked[0]["weights"],
                "base_metrics_at_these_weights": trace_eval.metric(
                    single_labels, global_best_base
                ),
                "final_metrics": global_metric,
            },
            "best_final_delta_vs_current": v3.delta_record(
                single_labels, current_final, global_best_final
            ),
            "maximum_final_correct": maximum_final_correct,
            "candidates_at_maximum_final_correct": sum(
                int(item["final_correct"]) == maximum_final_correct
                for item in global_records
            ),
            "maximum_base_correct": maximum_base_correct,
            "candidates_at_maximum_base_correct": sum(
                int(item["base_correct"]) == maximum_base_correct
                for item in global_records
            ),
            "best_for_base_only": {
                "weights": base_ranked[0]["weights"],
                "base_metrics": base_best_metric,
                "final_metrics_at_these_weights": base_best_final_metric,
            },
            "top_25_by_final_objective": global_ranked[:25],
            "top_25_by_base_objective": base_ranked[:25],
            "warning": (
                "The same 872 labels were used to rank this global grid. These weights are a "
                "development proposal, not an unbiased confirmation."
            ),
        },
        "reproduction_audit": {
            "prior_v3_opened_after_new_predictions_frozen": True,
            "maximum_absolute_difference": reproduction,
            "reproduced_at_tolerance_1e_12": True,
        },
        "promotion_decision": {
            "promote_global_grid_best_without_external_validation": False,
            "promote_nested_weight_selection": bool(
                nested_metric["accuracy"] > current_metric["accuracy"]
                and nested_bootstrap["percentile_95_interval"][0] > 0.0
            ),
            "reason": (
                "Promotion requires nested OOF improvement with a REFNO-cluster bootstrap "
                "interval excluding zero; a global-grid maximum alone is insufficient."
            ),
        },
        "boundary": {
            "internal_development_oof_not_external_validation": True,
            "outer_test_labels_excluded_from_weight_selection": True,
            "normalized_document_groups_isolated_in_outer_and_inner_folds": True,
            "pmagresult_or_rockmag_used_as_feature": False,
            "prior_predictions_or_reports_used_for_selection": False,
        },
        "artifacts": {
            "freeze": file_record(freeze_path),
            "oof": file_record(oof_path),
        },
    }
    report_path = output_dir / REPORT_NAME
    v2.atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "current": current_metric,
                "nested": nested_metric,
                "global_development_best": {
                    "weights": global_ranked[0]["weights"],
                    "metrics": global_metric,
                },
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
