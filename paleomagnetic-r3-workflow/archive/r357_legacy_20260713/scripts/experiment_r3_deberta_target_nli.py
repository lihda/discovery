#!/usr/bin/env python3
"""Evaluate target-bound DeBERTa NLI R3 features with REFNO-grouped OOF."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_deberta_target_nli_embeddings as builder  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402
import experiment_r3_small_screen_curation_style as curation  # noqa: E402
from paper_text_retrieval import canonical_target_identity  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/deberta_target_nli_v2"
DEFAULT_FEATURES = DEFAULT_OUTPUT_DIR / "features.npz"
DEFAULT_IDENTITY = ROOT / "tmp/r3_exact_300_resultno_v2/identity.xlsx"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "features.manifest.json"
DEFAULT_FEATURE_FREEZE = DEFAULT_OUTPUT_DIR / "feature_freeze.json"
DEFAULT_PROTOCOL = ROOT / "tmp/r3_exact_300_resultno_v2/screening_protocol.json"
DEFAULT_BASELINE = ROOT / "tmp/r3_exact_300_resultno_v2/baseline.json"
DEFAULT_OUTPUT = DEFAULT_OUTPUT_DIR / "grouped_oof.json"
DEFAULT_PROBABILITIES = DEFAULT_OUTPUT_DIR / "grouped_oof.npz"
SCREENING_PROTOCOL_SHA256 = (
    "6412244a5d268e6dcabfaed63f6f529d4976dd17ef6c4313f70bdb6748c04b4f"
)
BASELINE_REPORT_SHA256 = (
    "5b2b80228af3985bbaaba40dfdbeea5e84e4d58d1dfe8af7aaed7c20a20bc9bb"
)
FEATURE_FREEZE_SHA256 = (
    "3401b9a492f4e31866ccb8077bd6fafaacad4ae76cb6c4dbb6fd961942fe7c7d"
)
PRIMARY_STRATEGY = (
    "dense__target_chain_taxonomy__linear_svc_scaled_c0.001__multi_paper_50"
)
REPAIR_PRIMARY_STRATEGY = (
    "repair__joint_ensemble__curation_deberta_taxonomy_relative__paper_style_25"
)
FROZEN_IDENTITY_SHA256 = builder.FROZEN_IDENTITY_SHA256
FROZEN_ALIGNMENT_SHA256 = builder.FROZEN_ALIGNMENT_SHA256
FROZEN_INPUT_SHA256 = {
    "page_cache": "f8ec25c1148c04112ccfc75076743803417effbb833834a136c714dff4e146d7",
    "taxonomy": "149df539a937678c8204aa6d6e60e77bd901e4a77975d26caa5362749aa18ad4",
    "positive_kb": "b0fc28866fe6702938a7cc02fcf7b87b68abc9e3b4e391a53a53de1615b5e9ef",
    "reused_passages": "a704219df287ea37d71f1571c47cd17aa8b23227c78e756bc96106a9d39b7438",
    "model_safetensors": "d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa",
    "model_config": "897e756eb59d3183adb505952e7910e7cbc7750a43f3b3747a96b688d2b02a47",
    "model_tokenizer": "5124ef2ead1a10a717703bc436de7f353da76d6340e4587719b42b1693707964",
    "model_spm": "c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd",
    "targets": "378c17338f3b5cfeff3ccd496894e0f6183f747119b2624c4a9cefaeda02501e",
}
FROZEN_CURATION_MATRIX_SHA256 = {
    "style": "52766326a0cd4303614bf0e219dffb775abb2456d51a4bfd0b8c54d0c791b248",
    "knowledge": "e228c9e1cb42657921db7fa34639b679f1e40532fd603cfd331eb977538fe674",
    "all": "e9d7d973bcf07da0d9f274d2bdee9fccd4a1ad35255b6466c969d6679dc427a9",
}
TARGET_ACCURACY = 0.90


def expected_arrays() -> set[str]:
    return {
        "structured",
        *{
            f"{hypothesis}__{pool}"
            for hypothesis in builder.HYPOTHESIS_NAMES
            for pool in ("mean", "cls", "top")
        },
    }


def load_feature_views(path: Path, row_count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        names = set(payload.files)
        expected = expected_arrays()
        if names != expected:
            raise ValueError(
                "DeBERTa cache contains unexpected arrays; "
                f"missing={sorted(expected - names)}, extra={sorted(names - expected)}"
            )
        arrays = {
            name: np.asarray(payload[name], dtype=np.float32) for name in payload.files
        }
    if any(matrix.ndim != 2 or matrix.shape[0] != row_count for matrix in arrays.values()):
        raise ValueError("DeBERTa feature arrays are not aligned to every PDF-only row")
    if any(not np.isfinite(matrix).all() for matrix in arrays.values()):
        raise ValueError("DeBERTa feature cache contains non-finite values")

    def join(*names: str) -> np.ndarray:
        return np.concatenate([arrays[name] for name in names], axis=1)

    mean_names = tuple(f"{name}__mean" for name in builder.HYPOTHESIS_NAMES)
    cls_names = tuple(f"{name}__cls" for name in builder.HYPOTHESIS_NAMES)
    return {
        "structured": arrays["structured"],
        "paper_criterion_mean": arrays["paper_criterion__mean"],
        "target_criterion_mean": arrays["target_criterion__mean"],
        "target_criterion_cls": arrays["target_criterion__cls"],
        "target_criterion_top": arrays["target_criterion__top"],
        "paper_target_mean": join(
            "paper_criterion__mean", "target_criterion__mean"
        ),
        "method_carrier_mean": join(
            "target_method__mean", "target_carrier__mean"
        ),
        "target_chain_mean": join(
            "target_criterion__mean",
            "target_method__mean",
            "target_carrier__mean",
        ),
        "expert_mean": join(
            "paper_criterion__mean",
            "target_criterion__mean",
            "concept_method__mean",
        ),
        "all_mean": join(*mean_names),
        "all_cls": join(*cls_names),
        "target_mean_cls": join(
            "target_criterion__mean", "target_criterion__cls"
        ),
        "all_mean_structured": join(*mean_names, "structured"),
    }


def model_specs() -> list[tuple[str, bool, Callable[[], Any]]]:
    return [
        (
            "logreg_raw_c0.01",
            False,
            lambda: LogisticRegression(
                C=0.01,
                solver="liblinear",
                max_iter=2_000,
                random_state=base.RANDOM_STATE,
            ),
        ),
        (
            "logreg_raw_c0.1",
            False,
            lambda: LogisticRegression(
                C=0.1,
                solver="liblinear",
                max_iter=2_000,
                random_state=base.RANDOM_STATE,
            ),
        ),
        (
            "linear_svc_raw_c0.001",
            False,
            lambda: LinearSVC(
                C=0.001, max_iter=8_000, random_state=base.RANDOM_STATE
            ),
        ),
        (
            "linear_svc_raw_c0.01",
            False,
            lambda: LinearSVC(
                C=0.01, max_iter=8_000, random_state=base.RANDOM_STATE
            ),
        ),
        (
            "logreg_scaled_c0.001",
            True,
            lambda: LogisticRegression(
                C=0.001,
                solver="liblinear",
                max_iter=2_000,
                random_state=base.RANDOM_STATE,
            ),
        ),
        (
            "logreg_scaled_c0.01",
            True,
            lambda: LogisticRegression(
                C=0.01,
                solver="liblinear",
                max_iter=2_000,
                random_state=base.RANDOM_STATE,
            ),
        ),
        (
            "linear_svc_scaled_c0.0001",
            True,
            lambda: LinearSVC(
                C=0.0001, max_iter=8_000, random_state=base.RANDOM_STATE
            ),
        ),
        (
            "linear_svc_scaled_c0.001",
            True,
            lambda: LinearSVC(
                C=0.001, max_iter=8_000, random_state=base.RANDOM_STATE
            ),
        ),
    ]


def model_probability(model: Any, matrix: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    return expit(np.asarray(model.decision_function(matrix), dtype=np.float64))


def inverse_target_weights(
    indices: np.ndarray,
    records: Sequence[Mapping[str, str]],
) -> np.ndarray:
    keys = [
        (record["REFNO"], canonical_target_identity(record["ROCKNAME"]))
        for record in records
    ]
    counts = Counter(keys[index] for index in indices)
    return np.asarray([1.0 / counts[keys[index]] for index in indices], dtype=np.float64)


def paper_relative_features(
    records: Sequence[Mapping[str, str]], matrix: np.ndarray
) -> np.ndarray:
    """Express target evidence relative to competing targets in the same paper."""

    matrix = np.asarray(matrix, dtype=np.float32)
    paper_targets: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for index, record in enumerate(records):
        paper_targets[record["REFNO"]][
            canonical_target_identity(record["ROCKNAME"])
        ].append(index)
    output = np.zeros((len(records), matrix.shape[1] * 2), dtype=np.float32)
    for targets in paper_targets.values():
        target_keys = sorted(targets)
        target_matrix = np.asarray(
            [matrix[targets[key]].mean(axis=0) for key in target_keys], dtype=np.float32
        )
        mean = target_matrix.mean(axis=0)
        minimum = target_matrix.min(axis=0)
        spread = target_matrix.max(axis=0) - minimum
        for position, key in enumerate(target_keys):
            difference = target_matrix[position] - mean
            relative_position = np.full(matrix.shape[1], 0.5, dtype=np.float32)
            variable = spread > 1e-7
            relative_position[variable] = (
                target_matrix[position, variable] - minimum[variable]
            ) / spread[variable]
            values = np.concatenate((difference, relative_position))
            output[np.asarray(targets[key], dtype=np.int64)] = values
    return output


def outer_folds(
    records: Sequence[Mapping[str, str]], labels: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    folds = list(
        GroupKFold(n_splits=base.N_SPLITS).split(
            np.zeros(len(records)), labels, groups
        )
    )
    for train_index, test_index in folds:
        if set(groups[train_index]) & set(groups[test_index]):
            raise AssertionError("REFNO leaked across a DeBERTa outer fold")
    return folds


def run_dense_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    views: Mapping[str, np.ndarray],
    *,
    fold_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]], list[np.ndarray]]:
    folds = outer_folds(records, labels)
    if fold_limit is not None:
        folds = folds[:fold_limit]
    probabilities = {
        f"dense__{view}__{model_name}": np.full(len(records), np.nan, dtype=np.float64)
        for view in views
        for model_name, _, _ in model_specs()
    }
    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    test_indices: list[np.ndarray] = []
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        test_indices.append(test_index)
        weights = inverse_target_weights(train_index, records)
        for view_name, matrix in views.items():
            raw_train = matrix[train_index]
            raw_test = matrix[test_index]
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(raw_train)
            scaled_test = scaler.transform(raw_test)
            for model_name, scaled, factory in model_specs():
                model = factory()
                model.fit(
                    scaled_train if scaled else raw_train,
                    labels[train_index],
                    sample_weight=weights,
                )
                predicted = model_probability(
                    model, scaled_test if scaled else raw_test
                )
                name = f"dense__{view_name}__{model_name}"
                probabilities[name][test_index] = predicted
                fold_metrics[name].append(
                    {"fold": fold, **base.metric_record(labels[test_index], predicted)}
                )
        print(f"dense fold {fold}/{len(folds)} complete", flush=True)
    return probabilities, dict(fold_metrics), test_indices


def paper_training_arrays(
    records: Sequence[dict[str, str]],
    matrix: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, list[int]]]:
    members: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        members[record["REFNO"]].append(index)
    refnos = sorted(members)
    features = np.asarray(
        [matrix[np.asarray(members[refno], dtype=np.int64)].mean(axis=0) for refno in refnos],
        dtype=np.float32,
    )
    row_weights = np.asarray([len(members[refno]) for refno in refnos], dtype=np.float64)
    return refnos, features, row_weights, dict(members)


def run_paper_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    views: Mapping[str, np.ndarray],
    *,
    fold_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    folds = outer_folds(records, labels)
    if fold_limit is not None:
        folds = folds[:fold_limit]
    groups = np.asarray([record["REFNO"] for record in records], dtype=object)
    probabilities = {
        f"paper__{view}__{model_name}": np.full(len(records), np.nan, dtype=np.float64)
        for view in views
        for model_name, _, _ in model_specs()
    }
    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        train_refnos = set(str(groups[index]) for index in train_index)
        test_refnos = set(str(groups[index]) for index in test_index)
        for view_name, row_matrix in views.items():
            refnos, matrix, weights, members = paper_training_arrays(
                records, row_matrix
            )
            position = {refno: index for index, refno in enumerate(refnos)}
            train_position = np.asarray(
                [position[refno] for refno in refnos if refno in train_refnos], dtype=np.int64
            )
            train_labels = np.asarray(
                [
                    int(labels[members[refno]].mean() >= 0.5)
                    for refno in refnos
                    if refno in train_refnos
                ],
                dtype=np.int8,
            )
            test_order = [refno for refno in refnos if refno in test_refnos]
            test_position = np.asarray([position[refno] for refno in test_order], dtype=np.int64)
            raw_train = matrix[train_position]
            raw_test = matrix[test_position]
            scaler = StandardScaler()
            scaled_train = scaler.fit_transform(raw_train)
            scaled_test = scaler.transform(raw_test)
            for model_name, scaled, factory in model_specs():
                model = factory()
                model.fit(
                    scaled_train if scaled else raw_train,
                    train_labels,
                    sample_weight=weights[train_position],
                )
                paper_probability = model_probability(
                    model, scaled_test if scaled else raw_test
                )
                lookup = dict(zip(test_order, paper_probability))
                row_probability = np.asarray(
                    [lookup[str(groups[index])] for index in test_index], dtype=np.float64
                )
                name = f"paper__{view_name}__{model_name}"
                probabilities[name][test_index] = row_probability
                fold_metrics[name].append(
                    {"fold": fold, **base.metric_record(labels[test_index], row_probability)}
                )
        print(f"paper fold {fold}/{len(folds)} complete", flush=True)
    return probabilities, dict(fold_metrics)


def joint_tree_factories() -> tuple[tuple[str, Callable[[], Any]], ...]:
    return (
        ("hgb", curation.make_hgb),
        ("extra", curation.make_extra),
        ("gradient", curation.make_gradient),
    )


def add_joint_repair_strategies(
    probabilities: dict[str, np.ndarray],
    view_names: Sequence[str],
) -> None:
    for view_name in view_names:
        members = [
            probabilities[f"joint_tree__{view_name}__{model_name}"]
            for model_name, _ in joint_tree_factories()
        ]
        probabilities[f"joint_ensemble__{view_name}"] = np.mean(members, axis=0)
    joint_name = "joint_ensemble__curation_deberta_taxonomy_relative"
    paper_name = "joint_tree__paper_style__hgb"
    if joint_name not in probabilities or paper_name not in probabilities:
        raise AssertionError("joint repair ensemble inputs were not generated")
    probabilities[REPAIR_PRIMARY_STRATEGY] = (
        0.75 * probabilities[joint_name] + 0.25 * probabilities[paper_name]
    )


def run_joint_tree_oof(
    records: Sequence[dict[str, str]],
    labels: np.ndarray,
    views: Mapping[str, np.ndarray],
    curation_features: Mapping[str, Any],
    *,
    fold_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, list[dict[str, Any]]]]:
    """Fit low-dimensional raw-feature repair models strictly inside each fold."""

    folds = outer_folds(records, labels)
    if fold_limit is not None:
        folds = folds[:fold_limit]
    probabilities = {
        f"joint_tree__{view_name}__{model_name}": np.full(
            len(records), np.nan, dtype=np.float64
        )
        for view_name in views
        for model_name, _ in joint_tree_factories()
    }
    paper_name = "joint_tree__paper_style__hgb"
    probabilities[paper_name] = np.full(len(records), np.nan, dtype=np.float64)
    fold_metrics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for fold, (train_index, test_index) in enumerate(folds, start=1):
        weights = inverse_target_weights(train_index, records)
        for view_name, matrix in views.items():
            for model_name, factory in joint_tree_factories():
                model = factory()
                model.fit(
                    matrix[train_index], labels[train_index], sample_weight=weights
                )
                probability = np.asarray(
                    model.predict_proba(matrix[test_index])[:, 1], dtype=np.float64
                )
                name = f"joint_tree__{view_name}__{model_name}"
                probabilities[name][test_index] = probability
                fold_metrics[name].append(
                    {"fold": fold, **base.metric_record(labels[test_index], probability)}
                )

        paper_train, paper_labels, paper_test, test_groups = curation.paper_training_data(
            train_index,
            test_index,
            records,
            labels,
            curation_features,
            include_knowledge=False,
        )
        paper_model = curation.make_hgb()
        paper_model.fit(paper_train, paper_labels)
        paper_lookup = dict(
            zip(test_groups, paper_model.predict_proba(paper_test)[:, 1])
        )
        paper_probability = np.asarray(
            [paper_lookup[records[index]["REFNO"]] for index in test_index],
            dtype=np.float64,
        )
        probabilities[paper_name][test_index] = paper_probability
        fold_metrics[paper_name].append(
            {
                "fold": fold,
                **base.metric_record(labels[test_index], paper_probability),
            }
        )
        print(f"joint tree fold {fold}/{len(folds)} complete", flush=True)

    add_joint_repair_strategies(probabilities, tuple(views))
    return probabilities, dict(fold_metrics)


def propagation_strategies(
    records: Sequence[Mapping[str, str]],
    probabilities: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    members: dict[str, list[int]] = defaultdict(list)
    target_counts: dict[str, set[str]] = defaultdict(set)
    for index, record in enumerate(records):
        refno = record["REFNO"]
        members[refno].append(index)
        target_counts[refno].add(canonical_target_identity(record["ROCKNAME"]))
    multi = np.asarray([len(target_counts[record["REFNO"]]) > 1 for record in records])
    output = {name: values.copy() for name, values in probabilities.items()}
    for name, values in probabilities.items():
        paper_mean = values.copy()
        for indices in members.values():
            finite = values[indices][np.isfinite(values[indices])]
            if len(finite):
                paper_mean[indices] = float(finite.mean())
        for weight in (0.25, 0.50, 0.75):
            blended = values.copy()
            blended[multi] = (
                (1.0 - weight) * values[multi] + weight * paper_mean[multi]
            )
            output[f"{name}__multi_paper_{int(weight * 100):02d}"] = blended
    return output


def fixed_family_blends(probabilities: dict[str, np.ndarray]) -> None:
    """Add a small set of preregistered equal-weight semantic fusions."""

    for model_name, _, _ in model_specs():
        criterion = f"dense__target_criterion_mean__{model_name}"
        chain = f"dense__method_carrier_mean__{model_name}"
        paper = f"paper__paper_criterion_mean__{model_name}"
        if criterion in probabilities and chain in probabilities:
            probabilities[f"blend__criterion_chain__{model_name}"] = (
                probabilities[criterion] + probabilities[chain]
            ) / 2.0
        if criterion in probabilities and paper in probabilities:
            probabilities[f"blend__target_paper__{model_name}"] = (
                probabilities[criterion] + probabilities[paper]
            ) / 2.0


def fold_metric_records(
    labels: np.ndarray,
    values: np.ndarray,
    test_indices: Sequence[np.ndarray],
) -> list[dict[str, Any]]:
    return [
        {"fold": fold, **base.metric_record(labels[index], values[index])}
        for fold, index in enumerate(test_indices, start=1)
    ]


def representation_ceilings(
    records: Sequence[Mapping[str, str]], labels: np.ndarray
) -> dict[str, Any]:
    def ceiling(keys: Sequence[Any]) -> dict[str, Any]:
        members: dict[Any, list[int]] = defaultdict(list)
        for index, key in enumerate(keys):
            members[key].append(index)
        correct = sum(
            max(sum(labels[indices] == 0), sum(labels[indices] == 1))
            for indices in members.values()
        )
        mixed = sum(len(set(int(labels[index]) for index in indices)) > 1 for indices in members.values())
        return {
            "groups": len(members),
            "mixed_label_groups": mixed,
            "row_accuracy_ceiling": round(correct / len(labels), 8),
        }

    return {
        "canonical_target_within_paper": ceiling(
            [
                (record["REFNO"], canonical_target_identity(record["ROCKNAME"]))
                for record in records
            ]
        ),
        "paper_only": ceiling([record["REFNO"] for record in records]),
    }


def write_oof_npz(
    path: Path,
    probabilities: Mapping[str, np.ndarray],
    *,
    expected_rows: int | None = None,
) -> None:
    forbidden = re.compile(
        r"(?i)(?:^|__|_)(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)(?:$|__|_)"
    )
    if any(forbidden.search(name) for name in probabilities):
        raise ValueError("OOF strategy name contains a forbidden label or identifier field")
    arrays: dict[str, np.ndarray] = {}
    for name, values in sorted(probabilities.items()):
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 1 or not np.isfinite(values).all():
            raise ValueError("full OOF output requires finite one-dimensional arrays")
        if expected_rows is not None and len(values) != expected_rows:
            raise ValueError("OOF output does not have the frozen full row coverage")
        arrays[f"prob__{name}"] = values
        arrays[f"decision__{name}"] = (values >= 0.5).astype(np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=base.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--page-cache", type=Path, default=base.DEFAULT_PAGE_CACHE)
    parser.add_argument("--taxonomy", type=Path, default=base.DEFAULT_TAXONOMY)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--feature-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--feature-freeze", type=Path, default=DEFAULT_FEATURE_FREEZE)
    parser.add_argument("--screening-protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--smoke-folds", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probabilities", type=Path, default=DEFAULT_PROBABILITIES)
    args = parser.parse_args(argv)

    if args.smoke_folds is not None and not 1 <= args.smoke_folds < base.N_SPLITS:
        raise SystemExit("--smoke-folds must be between 1 and 4")
    identity_path = args.identity.resolve()
    targets_path = args.targets.resolve()
    pdf_dir = args.pdf_dir.resolve()
    features_path = args.features.resolve()
    manifest_path = args.feature_manifest.resolve()
    feature_freeze_path = args.feature_freeze.resolve()
    protocol_path = args.screening_protocol.resolve()
    baseline_path = args.baseline_report.resolve()
    identity_hash = base.file_sha256(identity_path)
    if identity_hash != FROZEN_IDENTITY_SHA256:
        raise ValueError("evaluation is restricted to the frozen exact-300 identity")
    if base.file_sha256(protocol_path) != SCREENING_PROTOCOL_SHA256:
        raise ValueError("exact-300 screening protocol hash mismatch before model fitting")
    if base.file_sha256(baseline_path) != BASELINE_REPORT_SHA256:
        raise ValueError("exact-300 baseline report hash mismatch")
    if base.file_sha256(targets_path) != FROZEN_INPUT_SHA256["targets"]:
        raise ValueError("R3 labels differ from the truth set used by the frozen baseline")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("selection", {}).get("identity_sha256") != identity_hash
        or protocol.get("selection", {}).get("pdf_only_rows") != builder.FROZEN_ROWS
        or protocol.get("selection", {}).get("pdf_only_paper_groups")
        != builder.FROZEN_PAPERS
    ):
        raise ValueError("screening protocol does not match the exact-300 scope")
    if (
        protocol.get("baseline", {}).get("report_sha256") != BASELINE_REPORT_SHA256
        or protocol.get("baseline", {}).get("strict_strategy")
        != "row_knowledge_extra_trees"
        or protocol.get("promotion_gate", {}).get("minimum_aggregate_accuracy")
        != TARGET_ACCURACY
    ):
        raise ValueError("screening protocol baseline or target differs from the freeze")
    records, coverage = base.load_pdf_only_identity(identity_path, pdf_dir)
    if (
        coverage.get("pdf_only_rows") != builder.FROZEN_ROWS
        or coverage.get("pdf_only_paper_groups") != builder.FROZEN_PAPERS
        or builder.alignment_sha256(records) != FROZEN_ALIGNMENT_SHA256
    ):
        raise ValueError("PDF membership differs from the frozen exact-300 scope")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "r3_exact300_deberta_target_nli_embeddings_v2":
        raise ValueError("feature manifest is not the exact-300 DeBERTa v2 artifact")
    if base.file_sha256(feature_freeze_path) != FEATURE_FREEZE_SHA256:
        raise ValueError("exact-300 label-free feature freeze hash mismatch")
    feature_freeze = json.loads(feature_freeze_path.read_text(encoding="utf-8"))
    if (
        feature_freeze.get("features_sha256") != base.file_sha256(features_path)
        or feature_freeze.get("manifest_sha256") != base.file_sha256(manifest_path)
        or feature_freeze.get("builder_sha256")
        != manifest.get("generator", {}).get("sha256")
        or feature_freeze.get("primary_strategy") != REPAIR_PRIMARY_STRATEGY
    ):
        raise ValueError("feature freeze does not bind the exact candidate artifacts")
    boundary = manifest.get("runtime_boundary", {})
    required_false = (
        "labels_read",
        "pmagresult_or_rockmag_read",
        "prior_predictions_or_evaluations_read",
        "identifier_features",
        "numeric_cache_contains_identifiers",
    )
    if manifest.get("status") != "PASS" or any(
        boundary.get(name) is not False for name in required_false
    ):
        raise ValueError("DeBERTa manifest does not prove a strict label-free build")
    if manifest.get("inputs", {}).get("identity", {}).get("sha256") != base.file_sha256(identity_path):
        raise ValueError("DeBERTa manifest identity hash mismatch")
    if manifest.get("inputs", {}).get("alignment_sha256") != builder.alignment_sha256(records):
        raise ValueError("DeBERTa feature row alignment mismatch")
    if manifest.get("output", {}).get("sha256") != base.file_sha256(features_path):
        raise ValueError("DeBERTa numerical cache hash mismatch")
    manifest_inputs = manifest.get("inputs", {})
    for name in ("page_cache", "taxonomy", "positive_kb", "reused_passages"):
        if manifest_inputs.get(name, {}).get("sha256") != FROZEN_INPUT_SHA256[name]:
            raise ValueError(f"DeBERTa manifest does not use the frozen {name} input")
    model_files = manifest.get("nli", {}).get("model_files", {})
    expected_model_files = {
        "config.json": FROZEN_INPUT_SHA256["model_config"],
        "model.safetensors": FROZEN_INPUT_SHA256["model_safetensors"],
        "tokenizer.json": FROZEN_INPUT_SHA256["model_tokenizer"],
        "spm.model": FROZEN_INPUT_SHA256["model_spm"],
    }
    if (
        manifest.get("nli", {}).get("revision") != builder.MODEL_REVISION
        or model_files != expected_model_files
    ):
        raise ValueError("DeBERTa manifest does not use the frozen model revision")
    retrieval_config = manifest.get("retrieval", {})
    if (
        manifest.get("scope", {}).get("target_limit") is not None
        or manifest.get("nli", {}).get("max_length") != 256
        or manifest.get("nli", {}).get("hypothesis_names")
        != list(builder.HYPOTHESIS_NAMES)
        or manifest.get("nli", {}).get("hypothesis_templates")
        != builder.HYPOTHESIS_TEMPLATES
        or retrieval_config.get("fresh_passages_per_paper") != 8
        or retrieval_config.get("maximum_selected_passages_per_target") != 4
        or retrieval_config.get("target_local")
        != {
            "maximum_passages_per_target": 4,
            "radius_characters": 700,
            "target_anchor_prefixed": True,
        }
    ):
        raise ValueError("DeBERTa feature-build configuration is not the frozen candidate")
    views = load_feature_views(features_path, len(records))

    # Rebuild the frozen taxonomy features from original text. This is the raw,
    # label-free feature family behind the baseline, not a reused prediction.
    page_cache_path = args.page_cache.resolve()
    taxonomy_path = args.taxonomy.resolve()
    if base.file_sha256(page_cache_path) != FROZEN_INPUT_SHA256["page_cache"]:
        raise ValueError("runtime page cache differs from the frozen DeBERTa screen")
    if base.file_sha256(taxonomy_path) != FROZEN_INPUT_SHA256["taxonomy"]:
        raise ValueError("runtime taxonomy differs from the frozen DeBERTa screen")
    paper_text = base.load_page_text(
        page_cache_path, {record["REFNO"] for record in records}
    )
    concepts, knowledge_audit = base.load_knowledge_patterns(taxonomy_path, None)
    source_features = base.build_features(records, paper_text, concepts)
    taxonomy_engineered = np.asarray(
        source_features["engineered_rows"], dtype=np.float32
    )
    taxonomy_article = np.asarray(
        [source_features["article_features"][record["REFNO"]] for record in records],
        dtype=np.float32,
    )
    views.update(
        {
            "taxonomy_engineered": taxonomy_engineered,
            "taxonomy_article": taxonomy_article,
            "target_criterion_taxonomy": np.concatenate(
                (views["target_criterion_mean"], taxonomy_engineered), axis=1
            ),
            "target_chain_taxonomy": np.concatenate(
                (views["target_chain_mean"], taxonomy_engineered), axis=1
            ),
            "all_mean_taxonomy": np.concatenate(
                (views["all_mean"], taxonomy_engineered), axis=1
            ),
            "deberta_structured_taxonomy": np.concatenate(
                (views["structured"], taxonomy_engineered), axis=1
            ),
        }
    )

    positive_kb_path = builder.DEFAULT_POSITIVE_KB.resolve()
    if base.file_sha256(positive_kb_path) != FROZEN_INPUT_SHA256["positive_kb"]:
        raise ValueError("positive ROCKMAG expert KB differs from the frozen screen")
    curation_concepts, curation_knowledge_audit = base.load_knowledge_patterns(
        taxonomy_path, positive_kb_path
    )
    pages = curation.load_pages(
        page_cache_path, {record["REFNO"] for record in records}
    )
    curation_features = curation.build_features(
        records, pages, curation_concepts, pdf_dir
    )
    curation_hashes = {
        name: curation.sha256_array(np.asarray(curation_features[name]))
        for name in ("style", "knowledge", "all")
    }
    if curation_hashes != FROZEN_CURATION_MATRIX_SHA256:
        raise ValueError("raw curation/style matrices differ from their label-free freeze")
    joint_base = np.concatenate(
        (
            curation_features["all"],
            views["structured"],
            taxonomy_engineered,
        ),
        axis=1,
    ).astype(np.float32)
    relative_source = np.concatenate(
        (
            curation_features["knowledge"],
            views["structured"],
            taxonomy_engineered,
        ),
        axis=1,
    ).astype(np.float32)
    joint_views = {
        "curation_all": np.asarray(curation_features["all"], dtype=np.float32),
        "curation_deberta": np.concatenate(
            (curation_features["all"], views["structured"]), axis=1
        ).astype(np.float32),
        "curation_taxonomy": np.concatenate(
            (curation_features["all"], taxonomy_engineered), axis=1
        ).astype(np.float32),
        "curation_deberta_taxonomy": joint_base,
        "curation_deberta_taxonomy_relative": np.concatenate(
            (joint_base, paper_relative_features(records, relative_source)), axis=1
        ).astype(np.float32),
    }
    raw_joint_feature_freeze = {
        "status": "validated_before_label_access",
        "sources": [
            "ROCKNAME identity text",
            "original-paper page-cache text and layout metadata",
            "local PDF byte size",
            "frozen taxonomy",
            "frozen positive ROCKMAG expert vocabulary",
            "label-free DeBERTa structured evidence scores",
        ],
        "curation_matrix_sha256": curation_hashes,
        "joint_view_shapes": {
            name: list(matrix.shape) for name, matrix in joint_views.items()
        },
        "labels_read": False,
        "prior_predictions_or_evaluations_read": False,
        "knowledge": curation_knowledge_audit,
    }

    # Labels are opened only after PDF scope and label-free cache validation.
    labels_by_resultno = base.load_training_labels(
        targets_path, {record["RESULTNO"] for record in records}
    )
    labels = np.asarray(
        [labels_by_resultno[record["RESULTNO"]] for record in records], dtype=np.int8
    )
    fold_limit = args.smoke_folds
    dense_views = views
    if fold_limit is not None:
        smoke_names = (
            "structured",
            "target_criterion_mean",
            "target_chain_mean",
            "target_chain_taxonomy",
            "all_mean_taxonomy",
        )
        dense_views = {name: views[name] for name in smoke_names}
    dense, dense_folds, test_indices = run_dense_oof(
        records, labels, dense_views, fold_limit=fold_limit
    )
    paper_view_names = (
        ("paper_criterion_mean", "deberta_structured_taxonomy")
        if fold_limit is not None
        else (
            "paper_criterion_mean",
            "structured",
            "expert_mean",
            "all_mean_structured",
            "taxonomy_article",
            "deberta_structured_taxonomy",
        )
    )
    paper, paper_folds = run_paper_oof(
        records,
        labels,
        {name: views[name] for name in paper_view_names},
        fold_limit=fold_limit,
    )
    joint, joint_folds = run_joint_tree_oof(
        records,
        labels,
        joint_views,
        curation_features,
        fold_limit=fold_limit,
    )
    raw = {**dense, **paper, **joint}
    fixed_family_blends(raw)
    probabilities = propagation_strategies(records, raw)
    evaluated = {
        name: values
        for name, values in probabilities.items()
        if all(np.isfinite(values[index]).all() for index in test_indices)
    }
    fold_metrics = {
        name: dense_folds.get(name)
        or paper_folds.get(name)
        or joint_folds.get(name)
        or fold_metric_records(labels, values, test_indices)
        for name, values in evaluated.items()
    }

    if fold_limit is None:
        metrics = {
            name: base.metric_record(labels, values) for name, values in evaluated.items()
        }
    else:
        covered = np.concatenate(test_indices)
        metrics = {
            name: base.metric_record(labels[covered], values[covered])
            for name, values in evaluated.items()
        }
    ranking = sorted(
        metrics,
        key=lambda name: (
            metrics[name]["accuracy"],
            metrics[name]["macro_f1"],
            metrics[name]["balanced_accuracy"],
        ),
        reverse=True,
    )
    best_name = ranking[0]

    gate_result = None
    if fold_limit is None:
        baseline_report = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline = baseline_report["best_full_coverage"]
        baseline_strategy = str(baseline["strategy"])
        baseline_folds = baseline_report["strategies"][baseline_strategy]["folds"]
        if REPAIR_PRIMARY_STRATEGY not in metrics:
            raise AssertionError("the preregistered repair strategy was not evaluated")
        primary_metrics = metrics[REPAIR_PRIMARY_STRATEGY]
        candidate_folds = fold_metrics[REPAIR_PRIMARY_STRATEGY]
        fold_deltas = [
            candidate["accuracy"] - baseline_fold["accuracy"]
            for candidate, baseline_fold in zip(candidate_folds, baseline_folds)
        ]
        folds_not_worse = sum(delta >= 0.0 for delta in fold_deltas)
        gain = primary_metrics["accuracy"] - baseline["accuracy"]
        passed = bool(primary_metrics["accuracy"] >= TARGET_ACCURACY)
        gate_result = {
            "passed": passed,
            "evaluated_scope": "exact_300_resultno_only",
            "other_scope_run_authorized": False,
            "baseline_strategy": baseline_strategy,
            "baseline_accuracy": baseline["accuracy"],
            "preregistered_primary_strategy": REPAIR_PRIMARY_STRATEGY,
            "target_accuracy": TARGET_ACCURACY,
            "observed_accuracy": primary_metrics["accuracy"],
            "observed_absolute_gain": round(float(gain), 8),
            "observed_folds_not_worse": folds_not_worse,
            "fold_accuracy_deltas": [round(float(value), 8) for value in fold_deltas],
            "worst_fold_regression": round(float(min(fold_deltas)), 8),
            "decision": (
                "EXACT300_TARGET_REACHED" if passed else "EXACT300_TARGET_NOT_REACHED"
            ),
            "warning": "Exact-300 grouped OOF is development evidence, not an external validation claim.",
        }

    homogeneous, mixed = base.paper_homogeneity_masks(records, labels)
    structured_names = manifest.get("output", {}).get("structured_feature_names", [])
    structured_lookup = {name: index for index, name in enumerate(structured_names)}
    coverage_names = (
        "retrieval__strong_chain_passages",
        "retrieval__positive_kb_passages",
        "retrieval__target_local_passages",
        "target__exact_passages",
        "target__token_hit_passages",
    )
    evidence_coverage: dict[str, Any] = {}
    for name in coverage_names:
        if name not in structured_lookup:
            continue
        values = views["structured"][:, structured_lookup[name]]
        evidence_coverage[name] = {
            subset: {
                "rows": int(mask.sum()),
                "nonzero_fraction": round(float((values[mask] > 0).mean()), 8),
                "mean": round(float(values[mask].mean()), 8),
            }
            for subset, mask in (
                ("all", np.ones(len(records), dtype=bool)),
                ("homogeneous_paper_rows", homogeneous),
                ("mixed_paper_rows", mixed),
            )
        }
    result = {
        "schema_version": "r3_exact300_deberta_target_nli_grouped_oof_v2",
        "status": "complete",
        "mode": "full_five_fold" if fold_limit is None else "technical_smoke",
        "scope": {
            **coverage,
            "evaluated_rows": len(records) if fold_limit is None else int(sum(len(v) for v in test_indices)),
            "evaluated_folds": base.N_SPLITS if fold_limit is None else fold_limit,
        },
        "guardrails": {
            "outer_split": "5-fold GroupKFold by REFNO",
            "full_row_coverage": fold_limit is None,
            "feature_cache_label_free": True,
            "labels_opened_after_scope_and_cache_validation": True,
            "pmagresult_or_rockmag_used_as_runtime_feature": False,
            "prior_predictions_or_oof_probabilities_used_as_feature": False,
            "baseline_report_used_only_after_oof_prediction": fold_limit is None,
            "identifier_features": False,
            "training_weight": "inverse REFNO/canonical-target row multiplicity",
            "screening_protocol_frozen_before_exact300_candidate_results": True,
            "old_640_row_probabilities_or_results_used": False,
            "model_selection_warning": (
                "Only the preregistered primary is used for the 90% target decision; "
                "the full ranking is diagnostic."
            ),
        },
        "inputs": {
            "identity": {"path": str(identity_path), "sha256": base.file_sha256(identity_path)},
            "targets_post_gate": {"path": str(targets_path), "sha256": base.file_sha256(targets_path)},
            "features": {"path": str(features_path), "sha256": base.file_sha256(features_path)},
            "feature_manifest": {"path": str(manifest_path), "sha256": base.file_sha256(manifest_path)},
            "page_cache": {"path": str(page_cache_path), "sha256": base.file_sha256(page_cache_path)},
            "taxonomy": {"path": str(taxonomy_path), "sha256": base.file_sha256(taxonomy_path)},
            "screening_protocol": (
                {"path": str(protocol_path), "sha256": base.file_sha256(protocol_path)}
            ),
            "feature_freeze": {
                "path": str(feature_freeze_path),
                "sha256": base.file_sha256(feature_freeze_path),
            },
            "baseline_report_comparison_only": {
                "path": str(baseline_path),
                "sha256": base.file_sha256(baseline_path),
            },
        },
        "feature_views": {name: int(matrix.shape[1]) for name, matrix in views.items()},
        "raw_joint_feature_freeze": raw_joint_feature_freeze,
        "joint_feature_views": {
            name: int(matrix.shape[1]) for name, matrix in joint_views.items()
        },
        "knowledge": knowledge_audit,
        "evidence_coverage_diagnostics": evidence_coverage,
        "representation_diagnostics": representation_ceilings(records, labels),
        "ranking": ranking,
        "preregistered_primary_candidate": (
            {
                "strategy": REPAIR_PRIMARY_STRATEGY,
                **metrics[REPAIR_PRIMARY_STRATEGY],
                "folds": fold_metrics[REPAIR_PRIMARY_STRATEGY],
                "homogeneous_paper_rows": base.subset_metric(
                    labels, evaluated[REPAIR_PRIMARY_STRATEGY], homogeneous
                ),
                "mixed_paper_rows": base.subset_metric(
                    labels, evaluated[REPAIR_PRIMARY_STRATEGY], mixed
                ),
            }
            if fold_limit is None
            else {
                "strategy": REPAIR_PRIMARY_STRATEGY,
                "evaluated_in_smoke": REPAIR_PRIMARY_STRATEGY in metrics,
            }
        ),
        "best_full_coverage" if fold_limit is None else "best_smoke_candidate": {
            "strategy": best_name,
            **metrics[best_name],
            "folds": fold_metrics[best_name],
            "homogeneous_paper_rows": (
                base.subset_metric(labels, evaluated[best_name], homogeneous)
                if fold_limit is None
                else None
            ),
            "mixed_paper_rows": (
                base.subset_metric(labels, evaluated[best_name], mixed)
                if fold_limit is None
                else None
            ),
        },
        "models": {
            name: {"aggregate": metrics[name], "folds": fold_metrics[name]}
            for name in evaluated
        },
        "promotion_gate": gate_result,
        "interpretation": {
            "target_accuracy": TARGET_ACCURACY,
            "target_reached_by_preregistered_primary": (
                metrics[REPAIR_PRIMARY_STRATEGY]["accuracy"] >= TARGET_ACCURACY
                if fold_limit is None
                else False
            ),
            "diagnostic_best_reached_target": (
                metrics[best_name]["accuracy"] >= TARGET_ACCURACY
            ),
            "no_other_scope_was_run": True,
        },
    }
    output_path = args.output.resolve()
    base.atomic_json(output_path, result)
    artifacts: dict[str, Any] = {"report": {"path": str(output_path)}}
    if fold_limit is None:
        probability_path = args.probabilities.resolve()
        write_oof_npz(probability_path, evaluated, expected_rows=len(records))
        artifacts["oof_probabilities_and_decisions"] = {
            "path": str(probability_path),
            "sha256": base.file_sha256(probability_path),
            "contains_labels": False,
            "contains_identifiers": False,
        }
        result["artifacts"] = artifacts
        base.atomic_json(output_path, result)
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "best": result[
                    "best_full_coverage" if fold_limit is None else "best_smoke_candidate"
                ],
                "promotion_gate": gate_result,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
