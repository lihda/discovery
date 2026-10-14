#!/usr/bin/env python3
"""Fit and run the deployable form of the frozen R3 stable combination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ARTIFACT_SCHEMA = "r3_stable_production_model_v1"
THRESHOLD = 0.5


@dataclass(frozen=True)
class StablePrediction:
    probability: np.ndarray
    route: np.ndarray
    components: dict[str, np.ndarray]


def _predict(model: Any, matrix: Any) -> np.ndarray:
    with warnings.catch_warnings(), np.errstate(
        divide="ignore", over="ignore", invalid="ignore"
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        value = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("R3 stable model produced invalid probabilities")
    return value


def _word_vectorizer(spec: Mapping[str, Any]) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=str(spec["analyzer"]),
        ngram_range=tuple(map(int, spec["ngram_range"])),
        min_df=int(spec["min_df"]),
        max_df=float(spec["max_df"]),
        max_features=int(spec["max_features"]),
        sublinear_tf=bool(spec["sublinear_tf"]),
        strip_accents=str(spec["strip_accents"]),
        dtype=np.float32,
    )


def _fit_domain_models(
    matrix: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    import evaluate_r3_residual_gate_single872_oof as residual
    import experiment_r3_exact300_nested_router as nested
    import experiment_r3_trace_clean_nested_oof as trace_eval

    indices = np.arange(len(records), dtype=np.int64)
    weights = trace_eval.inverse_identity_weights(indices, records)
    fitted = []
    for spec in residual.domain_specs(config):
        expert = nested.fit_expert(spec, matrix, indices, labels, weights)
        fitted.append(
            {
                "name": spec.name,
                "model": expert.model,
                "scaler": expert.scaler,
            }
        )
    return fitted


def _fit_query_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    import evaluate_r3_query_adaptive_single872_oof as query_eval
    import experiment_r3_trace_clean_nested_oof as trace_eval

    spec = config["query_experts"]["pca"]
    indices = np.arange(len(records), dtype=np.int64)
    weights = trace_eval.inverse_identity_weights(indices, records)
    scaler = StandardScaler()
    scaled = np.clip(scaler.fit_transform(matrix), -12.0, 12.0)
    pca = PCA(
        n_components=int(spec["pca_dimension"]),
        svd_solver="randomized",
        random_state=query_eval.RANDOM_STATE,
    )
    transformed = pca.fit_transform(scaled)
    model = LogisticRegression(
        C=float(spec["C"]),
        solver="liblinear",
        max_iter=3000,
        random_state=query_eval.RANDOM_STATE,
    )
    model.fit(transformed, labels, sample_weight=weights)
    return {"scaler": scaler, "pca": pca, "model": model}


def _fit_front_model(
    documents: Sequence[str],
    labels: np.ndarray,
    records: Sequence[Mapping[str, str]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    import experiment_r3_trace_clean_nested_oof as trace_eval

    vectorizer = _word_vectorizer(spec)
    matrix = vectorizer.fit_transform(documents)
    indices = np.arange(len(records), dtype=np.int64)
    model = LogisticRegression(
        C=float(spec["C"]),
        solver=str(spec["solver"]),
        max_iter=int(spec["max_iter"]),
        random_state=35_720_260,
    )
    model.fit(
        matrix,
        labels,
        sample_weight=trace_eval.inverse_identity_weights(indices, records),
    )
    return {"vectorizer": vectorizer, "model": model}


def _fit_knowledge_models(
    matrix: np.ndarray, labels: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    seed = int(config["protocol"]["random_state"])
    specifications = config["knowledge_experts"]
    hgb = specifications["hist_gradient_boosting"]
    extra = specifications["extra_trees"]
    gradient = specifications["gradient_boosting"]
    models: dict[str, Any] = {
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
            max_features=str(extra["max_features"]),
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
    for model in models.values():
        model.fit(matrix, labels)
    return models


def _fit_fulltext_models(
    documents: Sequence[str], labels: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    import evaluate_r3_fulltext_fusion_single872_oof_v3 as fulltext_v3

    spec = config["fulltext_expert"]
    vectorizer = _word_vectorizer(spec)
    matrix = vectorizer.fit_transform(documents)
    models = {}
    for name, c_value in zip(fulltext_v3.FULLTEXT_EXPERTS, spec["C_values"]):
        model = LogisticRegression(
            C=float(c_value),
            solver=str(spec["solver"]),
            max_iter=int(spec["max_iter"]),
            random_state=int(config["protocol"]["random_state"]),
        )
        model.fit(matrix, labels)
        models[name] = model
    return {"vectorizer": vectorizer, "models": models}


def fit_stable_artifact(
    *,
    records: Sequence[Mapping[str, str]],
    labels: np.ndarray,
    domain: np.ndarray,
    v4: np.ndarray,
    compact: np.ndarray,
    pooled: np.ndarray,
    front_documents: Sequence[str],
    single_indices: np.ndarray,
    knowledge: np.ndarray,
    full_documents: Sequence[str],
    v1_config: Mapping[str, Any],
    v2_config: Mapping[str, Any],
    v3_config: Mapping[str, Any],
) -> dict[str, Any]:
    count = len(records)
    if labels.shape != (count,) or single_indices.ndim != 1:
        raise ValueError("R3 stable training alignment is invalid")
    query_matrix = np.concatenate((v4, compact, pooled), axis=1)
    single_labels = labels[single_indices]
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "threshold": THRESHOLD,
        "domain_models": _fit_domain_models(domain, labels, records, v1_config),
        "query_model": _fit_query_model(query_matrix, labels, records, v1_config),
        "front_model": _fit_front_model(
            front_documents, labels, records, v1_config["front_text_expert"]
        ),
        "knowledge_models": _fit_knowledge_models(knowledge, single_labels, v2_config),
        "fulltext_models": _fit_fulltext_models(full_documents, single_labels, v3_config),
        "weights": {
            "base": dict(v2_config["base"]["weights"]),
            "v2": {
                "base": float(v2_config["primary_fusion"]["base_weight"]),
                "knowledge": float(v2_config["primary_fusion"]["knowledge_weight"]),
            },
            "v3": {
                "v2": float(v3_config["primary_fusion"]["v2_weight"]),
                "fulltext": float(v3_config["primary_fusion"]["fulltext_weight"]),
            },
        },
        "dimensions": {
            "domain": int(domain.shape[1]),
            "v4": int(v4.shape[1]),
            "compact": int(compact.shape[1]),
            "pooled": int(pooled.shape[1]),
            "query": int(query_matrix.shape[1]),
            "knowledge": int(knowledge.shape[1]),
        },
        "training": {
            "rows": count,
            "single_target_rows": int(len(single_indices)),
            "multi_target_rows": int(count - len(single_indices)),
            "historical_r3_labels_used": True,
        },
    }
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    if artifact.get("schema_version") != ARTIFACT_SCHEMA:
        raise ValueError("unsupported R3 stable production artifact")
    if float(artifact.get("threshold", -1)) != THRESHOLD:
        raise ValueError("R3 stable production threshold changed")
    required = {
        "domain_models",
        "query_model",
        "front_model",
        "knowledge_models",
        "fulltext_models",
        "weights",
        "dimensions",
        "training",
    }
    if required - set(artifact):
        raise ValueError("R3 stable production artifact is incomplete")


def predict_stable(
    *,
    artifact: Mapping[str, Any],
    domain: np.ndarray,
    v4: np.ndarray,
    compact: np.ndarray,
    pooled: np.ndarray,
    front_documents: Sequence[str],
    single_mask: np.ndarray,
    knowledge_single: np.ndarray,
    full_documents_single: Sequence[str],
) -> StablePrediction:
    validate_artifact(artifact)
    count = len(domain)
    dimensions = artifact["dimensions"]
    if (
        domain.shape != (count, int(dimensions["domain"]))
        or v4.shape != (count, int(dimensions["v4"]))
        or compact.shape != (count, int(dimensions["compact"]))
        or pooled.shape != (count, int(dimensions["pooled"]))
        or single_mask.shape != (count,)
        or len(front_documents) != count
    ):
        raise ValueError("R3 stable inference feature dimensions changed")

    domain_values = []
    for item in artifact["domain_models"]:
        matrix = np.asarray(domain, dtype=np.float64)
        if item["scaler"] is not None:
            matrix = np.clip(item["scaler"].transform(matrix), -12.0, 12.0)
        domain_values.append(_predict(item["model"], matrix))
    domain_mean = np.column_stack(domain_values).mean(axis=1)

    query_matrix = np.concatenate((v4, compact, pooled), axis=1)
    query = artifact["query_model"]
    query_scaled = np.clip(query["scaler"].transform(query_matrix), -12.0, 12.0)
    with warnings.catch_warnings(), np.errstate(
        divide="ignore", over="ignore", invalid="ignore"
    ):
        warnings.simplefilter("ignore", RuntimeWarning)
        query_transformed = query["pca"].transform(query_scaled)
    query_probability = _predict(query["model"], query_transformed)

    front = artifact["front_model"]
    front_probability = _predict(
        front["model"], front["vectorizer"].transform(front_documents)
    )
    base_components = {
        "domain_mean4": domain_mean,
        "query_pca64_logreg": query_probability,
        "front_word_logreg": front_probability,
    }
    base_weights = artifact["weights"]["base"]
    base_probability = sum(
        float(base_weights[name]) * value for name, value in base_components.items()
    )
    probability = np.asarray(base_probability, dtype=np.float64).copy()
    route = np.full(count, "multi_no_strict", dtype="<U24")

    single_indices = np.flatnonzero(single_mask)
    if len(single_indices):
        if (
            knowledge_single.shape
            != (len(single_indices), int(dimensions["knowledge"]))
            or len(full_documents_single) != len(single_indices)
        ):
            raise ValueError("R3 stable single-target feature alignment changed")
        knowledge_values = [
            _predict(model, knowledge_single)
            for model in artifact["knowledge_models"].values()
        ]
        knowledge_mean = np.column_stack(knowledge_values).mean(axis=1)
        v2_weights = artifact["weights"]["v2"]
        v2_probability = (
            float(v2_weights["base"]) * base_probability[single_indices]
            + float(v2_weights["knowledge"]) * knowledge_mean
        )
        full = artifact["fulltext_models"]
        full_matrix = full["vectorizer"].transform(full_documents_single)
        full_mean = np.column_stack(
            [_predict(model, full_matrix) for model in full["models"].values()]
        ).mean(axis=1)
        v3_weights = artifact["weights"]["v3"]
        probability[single_indices] = (
            float(v3_weights["v2"]) * v2_probability
            + float(v3_weights["fulltext"]) * full_mean
        )
        route[single_indices] = "single_v3"
    else:
        knowledge_mean = np.empty(0, dtype=np.float64)
        full_mean = np.empty(0, dtype=np.float64)

    if not np.isfinite(probability).all():
        raise ValueError("R3 stable production probability is non-finite")
    components = {
        **base_components,
        "base_probability": base_probability,
        "single_knowledge_mean": knowledge_mean,
        "single_fulltext_mean": full_mean,
    }
    return StablePrediction(probability=probability, route=route, components=components)


__all__ = [
    "ARTIFACT_SCHEMA",
    "StablePrediction",
    "fit_stable_artifact",
    "predict_stable",
    "validate_artifact",
]
