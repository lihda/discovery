#!/usr/bin/env python3
"""Freeze label-blind strict-trusted summaries from frozen R3 MIL bags.

The command accepts only the frozen MIL directory, this builder's frozen
knowledge contract, and an output directory. It has no input for labels,
source-table evidence, predictions, or evaluation reports.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_evidence_mil_bags as mil_freeze  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402


DEFAULT_MIL_DIR = ROOT / "tmp/r3_evidence_mil_single872_v1"
DEFAULT_KNOWLEDGE = ROOT / "data/knowledge/r3_strict_trusted_summary_v1.json"
DEFAULT_OUTPUT_DIR = ROOT / "tmp/r3_strict_trusted_summary_v1"

FEATURE_NAME = "r3_strict_trusted_summary_v1.npz"
RECEIPT_NAME = "r3_strict_trusted_summary_v1.receipt.jsonl"
MANIFEST_NAME = "r3_strict_trusted_summary_v1.manifest.json"

KB_SCHEMA = "r3_strict_trusted_summary_v1"
FEATURE_SCHEMA = "r3_strict_trusted_summary_v1"
MANIFEST_SCHEMA = "r3_strict_trusted_summary_manifest_v1"
RECEIPT_SCHEMA = "r3_strict_trusted_summary_receipt_v1"

SOURCE_ARRAY_NAMES = (
    "bag_indptr",
    "instance_features",
    "fold_ids",
    "source_row_indices",
    "refnos",
    "resultnos",
    "rocknames",
    "document_groups",
    "instance_bag_indices",
    "instance_pages",
    "instance_scale_ids",
    "instance_route_ids",
    "instance_chunk_ids",
    "instance_text_sha256",
)
IDENTITY_ARRAY_NAMES = (
    "fold_ids",
    "source_row_indices",
    "refnos",
    "resultnos",
    "rocknames",
    "document_groups",
)
AGGREGATE_INSTANCE_FEATURES = (
    "lexical_method_count_clip",
    "lexical_diagnostic_count_clip",
    "lexical_carrier_count_clip",
    "lexical_bridge_count_clip",
    "lexical_total_log",
    "lexical_category_coverage",
    "scale_sentence_window",
    "scale_paragraph",
    "section_methods",
    "section_results",
    "section_discussion",
    "own_study_language",
    "result_language",
    "support_without_citation",
    "event_first_person",
    "event_passive_execution",
    "event_procedure",
    "event_result_verb",
    "event_quantitative",
    "event_figure_table",
    "event_sample",
    "event_own_strong",
    "event_own_likely",
    "event_score_scaled",
    "trusted_method",
    "trusted_diagnostic",
    "trusted_carrier",
    "trusted_bridge",
)
TRUSTED_CATEGORY_FEATURES = (
    "trusted_method",
    "trusted_diagnostic",
    "trusted_carrier",
    "trusted_bridge",
)
APPENDED_FEATURES = (
    "strict_trusted_count_log1p",
    "strict_trusted_fraction",
    "strict_trusted_method_count_log1p",
    "strict_trusted_diagnostic_count_log1p",
    "strict_trusted_carrier_count_log1p",
    "strict_trusted_bridge_count_log1p",
)
STRICT_TRUE_FEATURES = ("trusted_any",)
STRICT_FALSE_FEATURES = (
    "support_blocked",
    "event_external",
    "event_future",
    "event_method_negation",
)
EVENT_FEATURES_REVALIDATED = (
    "first_person",
    "passive_execution",
    "procedure",
    "result_verb",
    "quantitative",
    "figure_table",
    "sample",
    "citation",
    "future",
    "method_negation",
    "own_strong",
    "own_likely",
    "external",
)
BOUNDARY_FALSE_FIELDS = (
    "labels_read",
    "pmagresult_or_rockmag_read",
    "prior_predictions_read",
    "evaluation_results_read",
    "source_table_fields_read",
    "identity_values_used_as_features",
)


class StrictTrustedSummaryError(ValueError):
    pass


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def json_sha256(value: Any) -> str:
    return mil_freeze.json_sha256(value)


def _require_exact_keys(
    value: Mapping[str, Any], expected: Sequence[str], label: str
) -> None:
    if set(value) != set(expected):
        missing = sorted(set(expected) - set(value))
        extra = sorted(set(value) - set(expected))
        raise StrictTrustedSummaryError(
            f"{label} keys changed; missing={missing}, extra={extra}"
        )


def load_summary_knowledge(path: Path) -> dict[str, Any]:
    payload = mil_freeze.load_json(path, "strict-trusted summary knowledge")
    if (
        payload.get("schema_version") != KB_SCHEMA
        or payload.get("status") != "FROZEN_LABEL_FREE_KNOWLEDGE"
    ):
        raise StrictTrustedSummaryError("summary knowledge has an invalid schema or status")

    source = payload.get("source_contract") or {}
    if (
        source.get("manifest_schema_version") != mil_freeze.MANIFEST_SCHEMA
        or source.get("feature_schema_version") != mil_freeze.FEATURE_SCHEMA
        or source.get("receipt_schema_version")
        != "r3_evidence_mil_bag_receipt_v1"
    ):
        raise StrictTrustedSummaryError("summary knowledge source contract changed")

    strict_filter = payload.get("strict_trusted_filter") or {}
    conditional = strict_filter.get("conditional_requirements") or []
    if (
        tuple(strict_filter.get("require_true_features") or ())
        != STRICT_TRUE_FEATURES
        or tuple(strict_filter.get("require_empty_receipt_fields") or ())
        != ("exclusion_codes",)
        or tuple(strict_filter.get("require_false_features") or ())
        != STRICT_FALSE_FEATURES
        or conditional
        != [
            {
                "if_feature": "event_citation",
                "then_true_feature": "event_own_likely",
            }
        ]
    ):
        raise StrictTrustedSummaryError("strict-trusted filter contract changed")

    summary = payload.get("summary") or {}
    if (
        tuple(summary.get("aggregate_instance_features") or ())
        != AGGREGATE_INSTANCE_FEATURES
        or tuple(summary.get("reducers") or ()) != ("mean", "max")
        or tuple(summary.get("appended_features") or ()) != APPENDED_FEATURES
        or int(summary.get("maximum_feature_count", 0)) != 64
        or summary.get("empty_bag_policy")
        != "retain_zero_summary_and_set_available_false"
    ):
        raise StrictTrustedSummaryError("strict-trusted summary contract changed")

    names = summary_feature_names(payload)
    if len(names) != 62 or len(names) > int(summary["maximum_feature_count"]):
        raise StrictTrustedSummaryError("summary feature dimension is invalid")

    boundary = payload.get("runtime_boundary") or {}
    _require_exact_keys(boundary, BOUNDARY_FALSE_FIELDS, "runtime boundary")
    if any(boundary[name] is not False for name in BOUNDARY_FALSE_FIELDS):
        raise StrictTrustedSummaryError("summary knowledge violates label-blind boundary")
    return payload


def summary_feature_names(kb: Mapping[str, Any]) -> list[str]:
    summary = kb.get("summary") or {}
    base = tuple(summary.get("aggregate_instance_features") or ())
    reducers = tuple(summary.get("reducers") or ())
    appended = tuple(summary.get("appended_features") or ())
    return [*(f"{reducer}_{name}" for reducer in reducers for name in base), *appended]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise StrictTrustedSummaryError(
                    f"source receipt contains a blank line at {line_number}"
                )
            value = json.loads(line)
            if not isinstance(value, dict):
                raise StrictTrustedSummaryError(
                    f"source receipt line {line_number} is not an object"
                )
            rows.append(value)
    return rows


def _feature_value(
    vector: np.ndarray, feature_index: Mapping[str, int], name: str
) -> float:
    return float(vector[feature_index[name]])


def _assert_feature_value(
    vector: np.ndarray,
    feature_index: Mapping[str, int],
    name: str,
    expected: float,
    global_index: int,
) -> None:
    actual = _feature_value(vector, feature_index, name)
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-6):
        raise StrictTrustedSummaryError(
            f"receipt/NPZ feature drift at instance {global_index}: "
            f"{name}={actual}, expected {expected}"
        )


def _as_mapping(value: Any, label: str, global_index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise StrictTrustedSummaryError(
            f"receipt instance {global_index} has invalid {label}"
        )
    return value


def _validate_receipt_instance_features(
    instance: Mapping[str, Any],
    vector: np.ndarray,
    feature_index: Mapping[str, int],
    global_index: int,
) -> None:
    lexical = _as_mapping(instance.get("lexical_counts"), "lexical_counts", global_index)
    lexical_values = []
    for category in mil_freeze.CATEGORIES:
        raw = lexical.get(category)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise StrictTrustedSummaryError(
                f"receipt instance {global_index} has invalid lexical count"
            )
        lexical_values.append(raw)
        _assert_feature_value(
            vector,
            feature_index,
            f"lexical_{category}_count_clip",
            min(raw, 4) / 4.0,
            global_index,
        )
    lexical_total = sum(lexical_values)
    _assert_feature_value(
        vector,
        feature_index,
        "lexical_total_log",
        math.log1p(lexical_total) / math.log1p(99 * len(mil_freeze.CATEGORIES)),
        global_index,
    )
    _assert_feature_value(
        vector,
        feature_index,
        "lexical_category_coverage",
        sum(value > 0 for value in lexical_values) / len(mil_freeze.CATEGORIES),
        global_index,
    )

    scale = clean(instance.get("scale"))
    if scale not in mil_freeze.ALLOWED_SCALES:
        raise StrictTrustedSummaryError(
            f"receipt instance {global_index} has invalid scale"
        )
    for name in mil_freeze.ALLOWED_SCALES:
        _assert_feature_value(
            vector,
            feature_index,
            f"scale_{name}",
            float(scale == name),
            global_index,
        )

    section = clean(instance.get("section_class"))
    if section not in mil_freeze.SECTION_CLASSES:
        raise StrictTrustedSummaryError(
            f"receipt instance {global_index} has invalid section class"
        )
    for name in ("methods", "results", "discussion"):
        _assert_feature_value(
            vector,
            feature_index,
            f"section_{name}",
            float(section == name),
            global_index,
        )

    language = _as_mapping(instance.get("language_flags"), "language_flags", global_index)
    for receipt_name, feature_name in (
        ("own_study", "own_study_language"),
        ("result", "result_language"),
    ):
        if not isinstance(language.get(receipt_name), bool):
            raise StrictTrustedSummaryError(
                f"receipt instance {global_index} has invalid language flag"
            )
        _assert_feature_value(
            vector,
            feature_index,
            feature_name,
            float(language[receipt_name]),
            global_index,
        )
    if not isinstance(language.get("citation"), bool):
        raise StrictTrustedSummaryError(
            f"receipt instance {global_index} has invalid citation language flag"
        )
    _assert_feature_value(
        vector,
        feature_index,
        "support_without_citation",
        float(lexical_total > 0 and not language["citation"]),
        global_index,
    )

    events = _as_mapping(instance.get("event_signals"), "event_signals", global_index)
    for name in EVENT_FEATURES_REVALIDATED:
        if not isinstance(events.get(name), bool):
            raise StrictTrustedSummaryError(
                f"receipt instance {global_index} has invalid event signal {name}"
            )
        _assert_feature_value(
            vector,
            feature_index,
            f"event_{name}",
            float(events[name]),
            global_index,
        )
    score = events.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise StrictTrustedSummaryError(
            f"receipt instance {global_index} has invalid event score"
        )
    _assert_feature_value(
        vector,
        feature_index,
        "event_score_scaled",
        float(np.clip(float(score) / 12.0, -1.0, 1.0)),
        global_index,
    )

    trusted = _as_mapping(
        instance.get("trusted_categories"), "trusted_categories", global_index
    )
    trusted_values = []
    for category in mil_freeze.CATEGORIES:
        raw = trusted.get(category)
        if not isinstance(raw, bool):
            raise StrictTrustedSummaryError(
                f"receipt instance {global_index} has invalid trusted category"
            )
        trusted_values.append(raw)
        _assert_feature_value(
            vector,
            feature_index,
            f"trusted_{category}",
            float(raw),
            global_index,
        )
    _assert_feature_value(
        vector,
        feature_index,
        "trusted_any",
        float(any(trusted_values)),
        global_index,
    )
    _assert_feature_value(
        vector,
        feature_index,
        "support_blocked",
        float(lexical_total > 0 and not any(trusted_values)),
        global_index,
    )


def is_strict_trusted_instance(
    instance: Mapping[str, Any],
    vector: np.ndarray,
    feature_index: Mapping[str, int],
    global_index: int = 0,
) -> bool:
    _validate_receipt_instance_features(instance, vector, feature_index, global_index)
    exclusions = instance.get("exclusion_codes")
    if not isinstance(exclusions, list) or any(not clean(value) for value in exclusions):
        raise StrictTrustedSummaryError(
            f"receipt instance {global_index} has invalid exclusion_codes"
        )
    if exclusions:
        return False
    if any(_feature_value(vector, feature_index, name) != 1.0 for name in STRICT_TRUE_FEATURES):
        return False
    if any(_feature_value(vector, feature_index, name) != 0.0 for name in STRICT_FALSE_FEATURES):
        return False
    citation = _feature_value(vector, feature_index, "event_citation") == 1.0
    own_likely = _feature_value(vector, feature_index, "event_own_likely") == 1.0
    return not citation or own_likely


def _validate_source_manifest(
    manifest: Mapping[str, Any], kb: Mapping[str, Any]
) -> tuple[list[str], int, int]:
    source_contract = kb["source_contract"]
    if (
        manifest.get("schema_version") != source_contract["manifest_schema_version"]
        or manifest.get("feature_schema_version")
        != source_contract["feature_schema_version"]
        or manifest.get("status") != "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS"
    ):
        raise StrictTrustedSummaryError("source MIL manifest is not a completed label-blind freeze")
    boundary = manifest.get("runtime_boundary") or {}
    if any(
        boundary.get(name) is not False
        for name in (
            "labels_read",
            "pmagresult_or_rockmag_read",
            "prior_predictions_read",
            "evaluation_results_read",
        )
    ):
        raise StrictTrustedSummaryError("source MIL manifest violates label-blind boundary")
    feature_names = manifest.get("instance_feature_names") or []
    if (
        feature_names != mil_freeze.instance_feature_names()
        or int(manifest.get("instance_feature_count", -1)) != len(feature_names)
        or clean(manifest.get("instance_feature_names_sha256"))
        != json_sha256(feature_names)
    ):
        raise StrictTrustedSummaryError("source MIL instance feature contract changed")
    bags = int(manifest.get("bags", -1))
    instances = int(manifest.get("instances", -1))
    if bags < 1 or instances < 0:
        raise StrictTrustedSummaryError("source MIL manifest has invalid dimensions")
    return list(feature_names), bags, instances


def _validate_source_arrays(
    arrays: Mapping[str, np.ndarray],
    manifest: Mapping[str, Any],
    bags: int,
    instances: int,
    feature_count: int,
) -> None:
    _require_exact_keys(arrays, SOURCE_ARRAY_NAMES, "source NPZ")
    indptr = arrays["bag_indptr"]
    features = arrays["instance_features"]
    if (
        indptr.dtype != np.int64
        or indptr.shape != (bags + 1,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != instances
        or np.any(np.diff(indptr) < 0)
        or features.dtype != np.float32
        or features.shape != (instances, feature_count)
        or not np.isfinite(features).all()
    ):
        raise StrictTrustedSummaryError("source MIL bag arrays have invalid dimensions or values")
    for name in IDENTITY_ARRAY_NAMES:
        if arrays[name].shape != (bags,):
            raise StrictTrustedSummaryError(f"source identity array {name} is misaligned")
    for name in (
        "instance_bag_indices",
        "instance_pages",
        "instance_scale_ids",
        "instance_route_ids",
        "instance_chunk_ids",
        "instance_text_sha256",
    ):
        if arrays[name].shape != (instances,):
            raise StrictTrustedSummaryError(f"source instance array {name} is misaligned")
    expected_bags = np.repeat(np.arange(bags, dtype=np.int32), np.diff(indptr))
    if not np.array_equal(arrays["instance_bag_indices"], expected_bags):
        raise StrictTrustedSummaryError("source instance-to-bag alignment changed")
    hashes = manifest.get("array_sha256") or {}
    _require_exact_keys(hashes, SOURCE_ARRAY_NAMES, "source array hashes")
    for name in SOURCE_ARRAY_NAMES:
        if clean(hashes[name]) != trace_freeze.array_sha256(arrays[name]):
            raise StrictTrustedSummaryError(f"source array hash changed: {name}")


def _validate_source_receipts(
    receipts: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    receipt_schema: str,
    feature_names: Sequence[str],
) -> np.ndarray:
    bags = len(arrays["fold_ids"])
    instances = len(arrays["instance_features"])
    if len(receipts) != bags:
        raise StrictTrustedSummaryError("source receipt bag count is misaligned")
    feature_index = {name: index for index, name in enumerate(feature_names)}
    strict_mask = np.zeros(instances, dtype=np.bool_)
    for bag_index, receipt in enumerate(receipts):
        start = int(arrays["bag_indptr"][bag_index])
        end = int(arrays["bag_indptr"][bag_index + 1])
        if (
            receipt.get("schema_version") != receipt_schema
            or int(receipt.get("bag_index", -1)) != bag_index
            or int(receipt.get("bag_start", -1)) != start
            or int(receipt.get("bag_end", -1)) != end
            or int(receipt.get("source_row_index", -1))
            != int(arrays["source_row_indices"][bag_index])
            or int(receipt.get("fold", -1)) != int(arrays["fold_ids"][bag_index])
            or clean(receipt.get("REFNO")) != clean(arrays["refnos"][bag_index])
            or clean(receipt.get("RESULTNO")) != clean(arrays["resultnos"][bag_index])
            or clean(receipt.get("ROCKNAME")) != clean(arrays["rocknames"][bag_index])
            or clean(receipt.get("document_group"))
            != clean(arrays["document_groups"][bag_index])
        ):
            raise StrictTrustedSummaryError(f"source receipt identity drift at bag {bag_index}")
        receipt_instances = receipt.get("instances")
        if not isinstance(receipt_instances, list) or len(receipt_instances) != end - start:
            raise StrictTrustedSummaryError(f"source receipt instances drift at bag {bag_index}")
        for offset, instance in enumerate(receipt_instances):
            if not isinstance(instance, Mapping):
                raise StrictTrustedSummaryError("source receipt instance is not an object")
            global_index = start + offset
            scale = clean(instance.get("scale"))
            route = clean(instance.get("primary_route"))
            if (
                int(instance.get("global_instance_index", -1)) != global_index
                or int(instance.get("bag_instance_index", -1)) != offset
                or clean(instance.get("chunk_id"))
                != clean(arrays["instance_chunk_ids"][global_index])
                or clean(instance.get("text_sha256"))
                != clean(arrays["instance_text_sha256"][global_index])
                or int(instance.get("page", -1))
                != int(arrays["instance_pages"][global_index])
                or scale not in mil_freeze.ALLOWED_SCALES
                or mil_freeze.ALLOWED_SCALES.index(scale)
                != int(arrays["instance_scale_ids"][global_index])
                or route not in mil_freeze.ROUTES
                or mil_freeze.ROUTES.index(route)
                != int(arrays["instance_route_ids"][global_index])
            ):
                raise StrictTrustedSummaryError(
                    f"source receipt instance identity drift at {global_index}"
                )
            strict_mask[global_index] = is_strict_trusted_instance(
                instance,
                arrays["instance_features"][global_index],
                feature_index,
                global_index,
            )
    return strict_mask


def build_summary_arrays(
    source_arrays: Mapping[str, np.ndarray],
    strict_mask: np.ndarray,
    feature_names: Sequence[str],
    kb: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    indptr = source_arrays["bag_indptr"]
    source_features = source_arrays["instance_features"]
    bags = len(indptr) - 1
    if strict_mask.dtype != np.bool_ or strict_mask.shape != (len(source_features),):
        raise StrictTrustedSummaryError("strict-trusted mask is misaligned")
    feature_index = {name: index for index, name in enumerate(feature_names)}
    aggregate_indices = np.asarray(
        [feature_index[name] for name in AGGREGATE_INSTANCE_FEATURES], dtype=np.int64
    )
    trusted_indices = np.asarray(
        [feature_index[name] for name in TRUSTED_CATEGORY_FEATURES], dtype=np.int64
    )
    names = summary_feature_names(kb)
    summary = np.zeros((bags, len(names)), dtype=np.float32)
    available = np.zeros(bags, dtype=np.bool_)
    counts = np.zeros(bags, dtype=np.int32)
    fractions = np.zeros(bags, dtype=np.float32)
    category_counts = np.zeros((bags, len(TRUSTED_CATEGORY_FEATURES)), dtype=np.int32)
    strict_indptr = np.zeros(bags + 1, dtype=np.int64)
    strict_global_indices = np.flatnonzero(strict_mask).astype(np.int64, copy=False)

    for bag_index in range(bags):
        start = int(indptr[bag_index])
        end = int(indptr[bag_index + 1])
        selected = np.flatnonzero(strict_mask[start:end]) + start
        count = len(selected)
        counts[bag_index] = count
        strict_indptr[bag_index + 1] = strict_indptr[bag_index] + count
        fractions[bag_index] = count / max(1, end - start)
        if not count:
            continue
        available[bag_index] = True
        values = source_features[selected][:, aggregate_indices]
        categories = source_features[selected][:, trusted_indices]
        category_counts[bag_index] = np.rint(categories.sum(axis=0)).astype(np.int32)
        summary[bag_index, : len(AGGREGATE_INSTANCE_FEATURES)] = values.mean(axis=0)
        summary[
            bag_index,
            len(AGGREGATE_INSTANCE_FEATURES) : 2 * len(AGGREGATE_INSTANCE_FEATURES),
        ] = values.max(axis=0)
        summary[bag_index, -len(APPENDED_FEATURES) :] = np.asarray(
            [
                math.log1p(count),
                fractions[bag_index],
                *(math.log1p(int(value)) for value in category_counts[bag_index]),
            ],
            dtype=np.float32,
        )

    arrays = {
        "summary": summary,
        "available": available,
        "source_row_indices": source_arrays["source_row_indices"].copy(),
        "fold_ids": source_arrays["fold_ids"].copy(),
        "refnos": source_arrays["refnos"].copy(),
        "resultnos": source_arrays["resultnos"].copy(),
        "rocknames": source_arrays["rocknames"].copy(),
        "document_groups": source_arrays["document_groups"].copy(),
        "strict_trusted_counts": counts,
        "strict_trusted_fraction": fractions,
        "strict_trusted_category_counts": category_counts,
        "strict_trusted_indptr": strict_indptr,
        "strict_trusted_global_indices": strict_global_indices,
        "summary_feature_names": np.asarray(names, dtype="U96"),
    }
    if (
        not np.isfinite(summary).all()
        or summary.shape != (bags, 62)
        or int(strict_indptr[-1]) != len(strict_global_indices)
        or not np.array_equal(np.diff(strict_indptr), counts)
        or not np.array_equal(available, counts > 0)
        or np.any(summary[~available] != 0.0)
    ):
        raise AssertionError("strict-trusted summary arrays are invalid")
    return arrays


def _summary_receipts(
    source_arrays: Mapping[str, np.ndarray], output_arrays: Mapping[str, np.ndarray]
) -> list[dict[str, Any]]:
    rows = []
    for bag_index in range(len(output_arrays["available"])):
        strict_start = int(output_arrays["strict_trusted_indptr"][bag_index])
        strict_end = int(output_arrays["strict_trusted_indptr"][bag_index + 1])
        rows.append(
            {
                "schema_version": RECEIPT_SCHEMA,
                "bag_index": bag_index,
                "source_row_index": int(output_arrays["source_row_indices"][bag_index]),
                "REFNO": clean(output_arrays["refnos"][bag_index]),
                "RESULTNO": clean(output_arrays["resultnos"][bag_index]),
                "ROCKNAME": clean(output_arrays["rocknames"][bag_index]),
                "fold": int(output_arrays["fold_ids"][bag_index]),
                "document_group": clean(output_arrays["document_groups"][bag_index]),
                "source_bag_start": int(source_arrays["bag_indptr"][bag_index]),
                "source_bag_end": int(source_arrays["bag_indptr"][bag_index + 1]),
                "strict_trusted_start": strict_start,
                "strict_trusted_end": strict_end,
                "strict_trusted_count": int(
                    output_arrays["strict_trusted_counts"][bag_index]
                ),
                "strict_trusted_fraction": float(
                    output_arrays["strict_trusted_fraction"][bag_index]
                ),
                "strict_trusted_category_counts": {
                    category.removeprefix("trusted_"): int(
                        output_arrays["strict_trusted_category_counts"][bag_index, index]
                    )
                    for index, category in enumerate(TRUSTED_CATEGORY_FEATURES)
                },
                "available": bool(output_arrays["available"][bag_index]),
                "strict_trusted_global_indices": output_arrays[
                    "strict_trusted_global_indices"
                ][strict_start:strict_end].tolist(),
            }
        )
    return rows


def _acquire_output_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".r3_strict_trusted_summary_builder.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise StrictTrustedSummaryError(
            f"another strict-trusted summary builder owns {output_dir}"
        ) from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def build_strict_trusted_summary(
    *, mil_dir: Path, knowledge_path: Path, output_dir: Path
) -> dict[str, Path]:
    mil_dir = mil_dir.resolve()
    knowledge_path = knowledge_path.resolve()
    output_dir = output_dir.resolve()
    source_paths = {
        "mil_npz": mil_dir / mil_freeze.FEATURE_NAME,
        "mil_receipt": mil_dir / mil_freeze.RECEIPT_NAME,
        "mil_manifest": mil_dir / mil_freeze.MANIFEST_NAME,
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise StrictTrustedSummaryError(f"missing source {label}: {path}")
    if not knowledge_path.is_file():
        raise StrictTrustedSummaryError(f"missing summary knowledge: {knowledge_path}")

    output_lock = _acquire_output_lock(output_dir)
    try:
        input_hashes = {
            name: trace_freeze.file_sha256(path) for name, path in source_paths.items()
        }
        knowledge_hash = trace_freeze.file_sha256(knowledge_path)
        kb = load_summary_knowledge(knowledge_path)
        source_manifest = mil_freeze.load_json(
            source_paths["mil_manifest"], "source MIL manifest"
        )
        feature_names, bags, instances = _validate_source_manifest(source_manifest, kb)
        outputs = source_manifest.get("outputs") or {}
        if (
            clean((outputs.get("bags") or {}).get("sha256")) != input_hashes["mil_npz"]
            or clean((outputs.get("receipt") or {}).get("sha256"))
            != input_hashes["mil_receipt"]
        ):
            raise StrictTrustedSummaryError("source MIL output file hash changed")

        with np.load(source_paths["mil_npz"], allow_pickle=False) as loaded:
            source_arrays = {name: loaded[name].copy() for name in loaded.files}
        _validate_source_arrays(
            source_arrays, source_manifest, bags, instances, len(feature_names)
        )
        source_receipts = _read_jsonl(source_paths["mil_receipt"])
        strict_mask = _validate_source_receipts(
            source_receipts,
            source_arrays,
            kb["source_contract"]["receipt_schema_version"],
            feature_names,
        )
        if any(
            trace_freeze.file_sha256(path) != input_hashes[name]
            for name, path in source_paths.items()
        ):
            raise StrictTrustedSummaryError("source MIL inputs changed during the build")
        if trace_freeze.file_sha256(knowledge_path) != knowledge_hash:
            raise StrictTrustedSummaryError("summary knowledge changed during the build")

        arrays = build_summary_arrays(source_arrays, strict_mask, feature_names, kb)
        receipts = _summary_receipts(source_arrays, arrays)
        feature_path = output_dir / FEATURE_NAME
        receipt_path = output_dir / RECEIPT_NAME
        manifest_path = output_dir / MANIFEST_NAME
        mil_freeze.atomic_npz(feature_path, **arrays)
        mil_freeze.atomic_jsonl(receipt_path, receipts)
        array_hashes = {
            name: trace_freeze.array_sha256(value) for name, value in arrays.items()
        }
        names = summary_feature_names(kb)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "feature_schema_version": FEATURE_SCHEMA,
            "status": "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS",
            "bags": bags,
            "source_instances": instances,
            "strict_trusted_instances": int(strict_mask.sum()),
            "available_bags": int(arrays["available"].sum()),
            "unavailable_bags": int((~arrays["available"]).sum()),
            "summary_feature_count": len(names),
            "summary_feature_names": names,
            "summary_feature_names_sha256": json_sha256(names),
            "aggregate_instance_features": list(AGGREGATE_INSTANCE_FEATURES),
            "strict_trusted_filter": kb["strict_trusted_filter"],
            "empty_bag_policy": kb["summary"]["empty_bag_policy"],
            "array_sha256": array_hashes,
            "inputs": {
                name: {"path": str(source_paths[name]), "sha256": digest}
                for name, digest in input_hashes.items()
            },
            "knowledge": {
                "path": str(knowledge_path),
                "sha256": knowledge_hash,
            },
            "code": {
                "path": str(Path(__file__).resolve()),
                "sha256": trace_freeze.file_sha256(Path(__file__).resolve()),
            },
            "outputs": {
                "summary": {
                    "path": str(feature_path),
                    "sha256": trace_freeze.file_sha256(feature_path),
                },
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": trace_freeze.file_sha256(receipt_path),
                },
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
            },
            "runtime_boundary": kb["runtime_boundary"],
        }
        mil_freeze.atomic_json(manifest_path, manifest)
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "bags": bags,
                    "strict_trusted_instances": int(strict_mask.sum()),
                    "available_bags": int(arrays["available"].sum()),
                    "summary": list(arrays["summary"].shape),
                },
                indent=2,
            ),
            flush=True,
        )
        return {
            "features": feature_path,
            "receipt": receipt_path,
            "manifest": manifest_path,
        }
    finally:
        output_lock.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mil-dir", type=Path, default=DEFAULT_MIL_DIR)
    parser.add_argument("--knowledge", type=Path, default=DEFAULT_KNOWLEDGE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_strict_trusted_summary(
        mil_dir=args.mil_dir,
        knowledge_path=args.knowledge,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGGREGATE_INSTANCE_FEATURES",
    "APPENDED_FEATURES",
    "StrictTrustedSummaryError",
    "build_parser",
    "build_strict_trusted_summary",
    "build_summary_arrays",
    "is_strict_trusted_instance",
    "load_summary_knowledge",
    "summary_feature_names",
]
