#!/usr/bin/env python3
"""Fixed nested grouped-OOF evaluation of frozen R3 evidence bags."""

from __future__ import annotations

import argparse
import atexit
from collections import Counter
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import sklearn
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_evidence_mil_bags as mil_freeze  # noqa: E402
import build_r3_trace_page_graph_features as trace_freeze  # noqa: E402
import experiment_r3_exact300_nested_router as nested  # noqa: E402
import experiment_r3_trace_clean_nested_oof as trace_eval  # noqa: E402


V4_DIR = ROOT / "tmp/r3_all_labelled_pdf_grouped_oof_v1"
DEFAULT_MIL_DIR = ROOT / "tmp/r3_evidence_mil_single872_v1"
DEFAULT_CONFIG = ROOT / "data/knowledge/r3_evidence_mil_evaluation_v1.json"
DEFAULT_V4_FEATURES = V4_DIR / trace_freeze.FEATURE_NAME
DEFAULT_V4_MANIFEST = V4_DIR / trace_freeze.MANIFEST_NAME
DEFAULT_V4_CONTEXT = V4_DIR / trace_freeze.CONTEXT_NAME
DEFAULT_V4_CANDIDATE = V4_DIR / trace_freeze.CANDIDATE_FREEZE_NAME
DEFAULT_V4_FOLDS = V4_DIR / trace_freeze.FOLD_MANIFEST_NAME
DEFAULT_V4_OOF = V4_DIR / "trace_clean_nested_oof_v4.npz"
DEFAULT_V4_REPORT = V4_DIR / "trace_clean_nested_oof_report_v4.json"

EVALUATION_FREEZE_NAME = "r3_evidence_mil_evaluation_freeze_v1.json"
RUN_STATE_NAME = "r3_evidence_mil_evaluation_run_state_v1.json"
OOF_NAME = "r3_evidence_mil_single872_grouped_oof_v1.npz"
RECEIPT_NAME = "r3_evidence_mil_single872_grouped_oof_receipt_v1.json"
REPORT_NAME = "r3_evidence_mil_single872_grouped_oof_report_v1.json"
LOCK_NAME = ".r3_evidence_mil_evaluation.lock"

CONFIG_SCHEMA = "r3_evidence_mil_evaluation_v1"
EVALUATION_FREEZE_SCHEMA = "r3_evidence_mil_evaluation_freeze_v1"
REPORT_SCHEMA = "r3_evidence_mil_single872_grouped_oof_report_v1"
EXPECTED_BAGS = 872
EXPECTED_INSTANCE_FEATURES = 61
EXPECTED_STACK_FEATURES = 131
COMPONENT_NAMES = (
    "v4_random_forest_all_pdf",
    "event_word_logistic",
    "event_char_logistic",
    "hard_em_c01_k1_max",
    "hard_em_c1_k2_top2mean",
)
HARD_EM_NAMES = COMPONENT_NAMES[-2:]
TRUSTED_FEATURE_NAMES = tuple(f"trusted_{name}" for name in mil_freeze.CATEGORIES)
FORBIDDEN_RECEIPT_KEYS = {
    "R3",
    "PMAGRESULT",
    "ROCKMAG",
    "LABEL",
    "OUTCOME",
    "PREDICTION",
    "GOLD",
    "TARGET_LABEL",
    "R3_LABEL",
}


class MilEvaluationError(ValueError):
    pass


@dataclass(frozen=True)
class FrozenMilInputs:
    arrays: Mapping[str, np.ndarray]
    receipt_rows: tuple[Mapping[str, Any], ...]
    event_documents: tuple[str, ...]
    word_matrix: sp.csr_matrix
    char_matrix: sp.csr_matrix
    static_bag_features: np.ndarray
    empty_bags: np.ndarray
    instance_metadata: Mapping[int, Mapping[str, Any]]
    manifest: Mapping[str, Any]
    candidate: Mapping[str, Any]


@dataclass(frozen=True)
class EmTrainingRows:
    instance_indices: np.ndarray
    labels: np.ndarray
    sample_weight: np.ndarray
    bag_indices: np.ndarray


@dataclass(frozen=True)
class EvidencePrediction:
    probability: np.ndarray
    instance_mean: np.ndarray
    instance_max: np.ndarray
    top_global_index: np.ndarray
    top_probability: np.ndarray


class EvaluationRunState:
    def __init__(self, path: Path, evaluation_freeze_sha256: str) -> None:
        self.path = path
        self.evaluation_freeze_sha256 = evaluation_freeze_sha256
        self.completed = False
        mil_freeze.atomic_json(
            path,
            {
                "schema_version": "r3_evidence_mil_evaluation_run_state_v1",
                "status": "RUNNING",
                "evaluation_freeze_sha256": evaluation_freeze_sha256,
            },
        )
        atexit.register(self.finalize)

    def complete(self, report: Path, oof: Path, receipt: Path) -> None:
        mil_freeze.atomic_json(
            self.path,
            {
                "schema_version": "r3_evidence_mil_evaluation_run_state_v1",
                "status": "COMPLETE",
                "evaluation_freeze_sha256": self.evaluation_freeze_sha256,
                "report": {"path": str(report), "sha256": file_sha256(report)},
                "oof": {"path": str(oof), "sha256": file_sha256(oof)},
                "receipt": {"path": str(receipt), "sha256": file_sha256(receipt)},
            },
        )
        self.completed = True

    def finalize(self) -> None:
        if self.completed:
            return
        mil_freeze.atomic_json(
            self.path,
            {
                "schema_version": "r3_evidence_mil_evaluation_run_state_v1",
                "status": "FAILED_OR_INTERRUPTED",
                "evaluation_freeze_sha256": self.evaluation_freeze_sha256,
            },
        )


def clean(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def file_sha256(path: Path) -> str:
    return trace_freeze.file_sha256(path)


def json_sha256(value: Any) -> str:
    return mil_freeze.json_sha256(value)


def sparse_matrix_sha256(matrix: sp.spmatrix) -> str:
    value = matrix.tocsr(copy=True)
    value.sum_duplicates()
    value.sort_indices()
    payload = {
        "format": "csr",
        "shape": [int(part) for part in value.shape],
        "dtype": value.dtype.str,
        "data": trace_freeze.array_sha256(value.data),
        "indices": trace_freeze.array_sha256(value.indices),
        "indptr": trace_freeze.array_sha256(value.indptr),
    }
    return json_sha256(payload)


def _verify_file(path: Path, expected: Any, label: str) -> None:
    if not path.is_file() or file_sha256(path) != clean(expected):
        raise MilEvaluationError(f"frozen {label} hash changed: {path}")


def _acquire_lock(output_dir: Path) -> Any:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / LOCK_NAME
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise MilEvaluationError("another MIL evaluator owns the output directory") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "output_dir": str(output_dir)}))
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def load_evaluation_config(path: Path) -> dict[str, Any]:
    config = mil_freeze.load_json(path, "MIL evaluation config")
    protocol = config.get("protocol") or {}
    stack = config.get("stack") or {}
    components = config.get("components") or {}
    em_specs = components.get("hard_em") or []
    boundary = config.get("runtime_boundary") or {}
    source_refs = config.get("source_refs") or []
    if len(source_refs) != 3 or not all(
        isinstance(item, Mapping) for item in source_refs
    ) or tuple(
        clean(item.get("source_id"))
        for item in source_refs
    ) != (
        "PATEL_ET_AL_ACL_2025_R2E",
        "TAN_ET_AL_SIGIR_2025_EDU_FILTERING",
        "DU_ET_AL_ICLR_2025_PSMIL",
    ) or any(
        "not" not in clean(item.get("adaptation_boundary")).casefold()
        or "direct reproduction"
        not in clean(item.get("adaptation_boundary")).casefold()
        for item in source_refs
    ):
        raise MilEvaluationError("MIL methodological source boundary changed")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
        or int(protocol.get("outer_folds", -1)) != 5
        or int(protocol.get("inner_folds", -1)) != 3
        or int(protocol.get("inner_seed", -1)) != 35_720_260_761
        or int(protocol.get("random_state", -1)) != 35_720_260
        or float(protocol.get("threshold", -1)) != 0.5
        or protocol.get("outer_test_selection") is not False
    ):
        raise MilEvaluationError("MIL evaluation protocol changed")
    if (
        tuple(stack.get("component_logits") or ()) != COMPONENT_NAMES
        or tuple(stack.get("hard_em_instance_statistics") or ())
        != (
            "hard_em_c01_k1_max_mean",
            "hard_em_c01_k1_max_max",
            "hard_em_c1_k2_top2mean_mean",
            "hard_em_c1_k2_top2mean_max",
        )
        or stack.get("static_bag_features")
        != "61 instance-feature means followed by 61 maxima"
        or int(stack.get("expected_dimensions", -1)) != EXPECTED_STACK_FEATURES
        or float(stack.get("C", -1)) != 0.1
        or stack.get("strategy") != "fixed_logistic_stack"
    ):
        raise MilEvaluationError("MIL stack contract changed")
    expected_em = (
        ("hard_em_c01_k1_max", 0.1, 1, "max", 3),
        ("hard_em_c1_k2_top2mean", 1.0, 2, "top2mean", 3),
    )
    observed_em = tuple(
        (
            clean(item.get("name")),
            float(item.get("C", -1)),
            int(item.get("positive_top_k", -1)),
            clean(item.get("aggregation")),
            int(item.get("rounds", -1)),
        )
        for item in em_specs
        if isinstance(item, Mapping)
    )
    if observed_em != expected_em:
        raise MilEvaluationError("MIL hard-EM candidates changed")
    empty_policy = config.get("empty_bag_policy") or {}
    receipt = config.get("receipt") or {}
    if (
        float(empty_policy.get("static_bag_features", -1)) != 0.0
        or float(empty_policy.get("evidence_component_probability", -1)) != 0.5
        or float(empty_policy.get("hard_em_instance_statistics", -1)) != 0.5
        or empty_policy.get("final_probability") != "v4_random_forest_all_pdf"
        or receipt.get("top_instance_source") != "hard_em_c01_k1_max"
        or receipt.get("retain_top_instance_for_each_hard_em_component") is not True
    ):
        raise MilEvaluationError("MIL empty-bag or receipt contract changed")
    hashing = config.get("event_documents") or {}
    for name, analyzer, ngrams in (
        ("word_hashing", "word", (1, 2)),
        ("char_hashing", "char_wb", (3, 5)),
    ):
        item = hashing.get(name) or {}
        if (
            clean(item.get("analyzer")) != analyzer
            or tuple(item.get("ngram_range") or ()) != ngrams
            or int(item.get("n_features", -1)) != 262_144
            or item.get("alternate_sign") is not False
            or clean(item.get("norm")) != "l2"
        ):
            raise MilEvaluationError(f"MIL {name} contract changed")
    if any(
        boundary.get(name) is not False
        for name in (
            "labels_read_before_evaluation_freeze",
            "pmagresult_or_rockmag_read",
            "prior_predictions_read_before_preaudit",
            "evaluation_results_read_before_preaudit",
            "outer_test_labels_used_for_training_or_selection",
            "missing_pdf_targets_in_denominator",
            "selective_abstention_used",
        )
    ) or boundary.get("event_document_features_are_label_blind") is not True:
        raise MilEvaluationError("MIL evaluation source boundary changed")
    return config


def make_hashing_vectorizer(spec: Mapping[str, Any]) -> HashingVectorizer:
    kwargs: dict[str, Any] = {
        "analyzer": clean(spec["analyzer"]),
        "ngram_range": tuple(int(value) for value in spec["ngram_range"]),
        "n_features": int(spec["n_features"]),
        "alternate_sign": bool(spec["alternate_sign"]),
        "lowercase": bool(spec["lowercase"]),
        "norm": clean(spec["norm"]),
        "dtype": np.float64,
    }
    if kwargs["analyzer"] == "word":
        kwargs["token_pattern"] = clean(spec["token_pattern"])
    return HashingVectorizer(**kwargs)


def build_event_document(row: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    template = clean(config["event_documents"]["instance_format"])
    separator = str(config["event_documents"]["separator"])
    values = []
    for instance in row.get("instances") or []:
        trusted = ",".join(
            name
            for name in mil_freeze.CATEGORIES
            if bool((instance.get("trusted_categories") or {}).get(name))
        ) or "none"
        exclusions = ",".join(map(clean, instance.get("exclusion_codes") or [])) or "none"
        values.append(
            template.format(
                section_class=clean(instance.get("section_class")) or "other",
                primary_route=clean(instance.get("primary_route")) or "unknown",
                trusted=trusted,
                exclusions=exclusions,
                text=clean(instance.get("text")),
            )
        )
    return separator.join(values)


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        if any(clean(key).upper() in FORBIDDEN_RECEIPT_KEYS for key in value):
            return True
        return any(_contains_forbidden_key(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_forbidden_key(item) for item in value)
    return False


def validate_trusted_instance(
    instance: Mapping[str, Any], feature_row: np.ndarray, feature_names: Sequence[str]
) -> None:
    lexical_map = instance.get("lexical_counts") or {}
    lexical = tuple(int(lexical_map.get(name, 0)) for name in mil_freeze.CATEGORIES)
    counter_map = instance.get("counter_counts") or {}
    counters = tuple(int(counter_map.get(name, 0)) for name in mil_freeze.COUNTER_GROUPS)
    signals = instance.get("event_signals") or {}
    section = clean(instance.get("section_class"))
    blocked = bool(
        section == "references"
        or signals.get("external")
        or signals.get("future")
        or signals.get("method_negation")
    )
    execution = bool(
        signals.get("first_person")
        or signals.get("passive_execution")
        or signals.get("procedure")
    )
    assertion = bool(signals.get("result_verb") or signals.get("quantitative"))
    owned = bool(signals.get("own_likely") or execution)
    method, diagnostic, carrier, bridge = lexical
    expected = (
        bool(method and (owned or assertion) and not blocked),
        bool(diagnostic and assertion and not blocked),
        bool(carrier and (owned or assertion) and not blocked),
        bool(bridge and (owned or assertion) and not blocked),
    )
    observed = tuple(
        bool((instance.get("trusted_categories") or {}).get(name))
        for name in mil_freeze.CATEGORIES
    )
    if observed != expected:
        raise MilEvaluationError("receipt trusted-category constraint changed")
    exclusions = []
    if signals.get("external"):
        exclusions.append("EXTERNAL")
    if signals.get("future"):
        exclusions.append("FUTURE")
    if signals.get("method_negation"):
        exclusions.append("METHOD_NEGATION")
    if counters[0] and not any(lexical):
        exclusions.append("DEMAG_ONLY")
    if signals.get("citation") and not signals.get("own_likely"):
        exclusions.append("CITATION_WITHOUT_OWNERSHIP")
    if not any(lexical):
        exclusions.append("NO_SUPPORT_ANCHOR")
    if list(instance.get("exclusion_codes") or []) != exclusions:
        raise MilEvaluationError("receipt exclusion-code constraint changed")
    positions = {name: feature_names.index(name) for name in TRUSTED_FEATURE_NAMES}
    for index, name in enumerate(TRUSTED_FEATURE_NAMES):
        if float(feature_row[positions[name]]) != float(expected[index]):
            raise MilEvaluationError("trusted-category feature differs from receipt")
    checks = {
        "trusted_category_fraction": sum(expected) / len(expected),
        "trusted_any": float(any(expected)),
        "support_blocked": float(any(lexical) and not any(expected)),
    }
    for name, expected_value in checks.items():
        if not math.isclose(
            float(feature_row[feature_names.index(name)]),
            float(expected_value),
            abs_tol=1e-7,
        ):
            raise MilEvaluationError(f"{name} feature differs from receipt")


def validate_bag_alignment(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, str]],
    v4_fold_ids: np.ndarray,
    v4_document_groups: np.ndarray,
    feature_names: Sequence[str],
) -> None:
    required = {
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
    }
    if set(arrays) != required:
        raise MilEvaluationError("MIL bag archive fields changed")
    indptr = arrays["bag_indptr"]
    features = arrays["instance_features"]
    source = arrays["source_row_indices"]
    bags = len(source)
    instances = len(features)
    if (
        indptr.dtype != np.int64
        or indptr.shape != (bags + 1,)
        or int(indptr[0]) != 0
        or int(indptr[-1]) != instances
        or np.any(np.diff(indptr) < 0)
        or np.any(np.diff(indptr) > 64)
        or features.dtype != np.float32
        or features.shape != (instances, len(feature_names))
        or not np.isfinite(features).all()
        or source.dtype != np.int32
        or len(np.unique(source)) != bags
        or not np.array_equal(source, np.sort(source))
    ):
        raise MilEvaluationError("MIL bag structure changed")
    bag_vectors = (
        arrays["fold_ids"],
        arrays["refnos"],
        arrays["resultnos"],
        arrays["rocknames"],
        arrays["document_groups"],
    )
    if any(value.shape != (bags,) for value in bag_vectors):
        raise MilEvaluationError("MIL bag identity arrays changed")
    if arrays["fold_ids"].dtype != np.int8:
        raise MilEvaluationError("MIL fold dtype changed")
    for bag_index, source_index in enumerate(source):
        index = int(source_index)
        if index < 0 or index >= len(records):
            raise MilEvaluationError("MIL source row is out of range")
        record = records[index]
        if (
            clean(arrays["refnos"][bag_index]) != clean(record["REFNO"])
            or clean(arrays["resultnos"][bag_index]) != clean(record["RESULTNO"])
            or clean(arrays["rocknames"][bag_index]) != clean(record["ROCKNAME"])
            or int(arrays["fold_ids"][bag_index]) != int(v4_fold_ids[index])
            or clean(arrays["document_groups"][bag_index])
            != clean(v4_document_groups[index])
        ):
            raise MilEvaluationError("MIL bags no longer align to V4 rows")
    instance_vectors = tuple(
        arrays[name]
        for name in (
            "instance_bag_indices",
            "instance_pages",
            "instance_scale_ids",
            "instance_route_ids",
            "instance_chunk_ids",
            "instance_text_sha256",
        )
    )
    if any(value.shape != (instances,) for value in instance_vectors):
        raise MilEvaluationError("MIL instance identity arrays changed")
    expected_bags = np.repeat(np.arange(bags, dtype=np.int32), np.diff(indptr))
    if not np.array_equal(arrays["instance_bag_indices"], expected_bags):
        raise MilEvaluationError("MIL instance-to-bag mapping changed")


def bag_feature_mean_max(features: np.ndarray, indptr: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or indptr.ndim != 1 or int(indptr[-1]) != len(features):
        raise ValueError("bag feature inputs do not align")
    output = np.zeros((len(indptr) - 1, features.shape[1] * 2), dtype=np.float64)
    for bag in range(len(output)):
        start, end = int(indptr[bag]), int(indptr[bag + 1])
        if end > start:
            current = np.asarray(features[start:end], dtype=np.float64)
            output[bag, : features.shape[1]] = current.mean(axis=0)
            output[bag, features.shape[1] :] = current.max(axis=0)
    if not np.isfinite(output).all():
        raise AssertionError("static MIL bag features are non-finite")
    return output


def _load_receipt_rows(path: Path) -> tuple[Mapping[str, Any], ...]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise MilEvaluationError("MIL receipt row is not an object")
                rows.append(value)
    return tuple(rows)


def _validate_receipt(
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
) -> tuple[tuple[str, ...], dict[int, Mapping[str, Any]]]:
    if len(rows) != len(arrays["source_row_indices"]):
        raise MilEvaluationError("MIL receipt bag count changed")
    metadata: dict[int, Mapping[str, Any]] = {}
    documents = []
    for bag_index, row in enumerate(rows):
        if _contains_forbidden_key(row):
            raise MilEvaluationError("MIL receipt contains a forbidden outcome field")
        start = int(arrays["bag_indptr"][bag_index])
        end = int(arrays["bag_indptr"][bag_index + 1])
        if (
            row.get("schema_version") != "r3_evidence_mil_bag_receipt_v1"
            or int(row.get("bag_index", -1)) != bag_index
            or int(row.get("source_row_index", -1))
            != int(arrays["source_row_indices"][bag_index])
            or clean(row.get("REFNO")) != clean(arrays["refnos"][bag_index])
            or clean(row.get("RESULTNO")) != clean(arrays["resultnos"][bag_index])
            or clean(row.get("ROCKNAME")) != clean(arrays["rocknames"][bag_index])
            or int(row.get("fold", -1)) != int(arrays["fold_ids"][bag_index])
            or clean(row.get("document_group"))
            != clean(arrays["document_groups"][bag_index])
            or int(row.get("bag_start", -1)) != start
            or int(row.get("bag_end", -1)) != end
            or len(row.get("instances") or []) != end - start
        ):
            raise MilEvaluationError("MIL receipt bag alignment changed")
        for offset, instance in enumerate(row.get("instances") or []):
            global_index = start + offset
            text = clean(instance.get("text"))
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if (
                int(instance.get("global_instance_index", -1)) != global_index
                or int(instance.get("bag_instance_index", -1)) != offset
                or clean(instance.get("chunk_id"))
                != clean(arrays["instance_chunk_ids"][global_index])
                or clean(instance.get("text_sha256")) != digest
                or digest != clean(arrays["instance_text_sha256"][global_index])
                or int(instance.get("page", -1))
                != int(arrays["instance_pages"][global_index])
                or clean(instance.get("scale"))
                != mil_freeze.ALLOWED_SCALES[
                    int(arrays["instance_scale_ids"][global_index])
                ]
                or clean(instance.get("primary_route"))
                != mil_freeze.ROUTES[int(arrays["instance_route_ids"][global_index])]
            ):
                raise MilEvaluationError("MIL receipt instance alignment changed")
            validate_trusted_instance(
                instance, arrays["instance_features"][global_index], feature_names
            )
            metadata[global_index] = instance
        documents.append(row)
    return tuple(documents), metadata


def _validate_manifest_inputs(manifest: Mapping[str, Any], candidate: Mapping[str, Any]) -> None:
    manifest_inputs = manifest.get("inputs") or {}
    candidate_inputs = candidate.get("inputs") or {}
    if manifest_inputs != candidate_inputs:
        raise MilEvaluationError("MIL manifest/candidate inputs differ")
    for item in manifest_inputs.get("code") or []:
        path = Path(clean(item.get("path"))).resolve()
        _verify_file(path, item.get("sha256"), "MIL builder code")
    for name in ("identity", "page_cache", "target_context", "fold_manifest", "concept_snapshot", "knowledge"):
        item = manifest_inputs.get(name) or {}
        _verify_file(Path(clean(item.get("path"))).resolve(), item.get("sha256"), name)


def load_frozen_mil_inputs(
    *,
    mil_dir: Path,
    config: Mapping[str, Any],
    records: Sequence[Mapping[str, str]],
    v4_fold_ids: np.ndarray,
    v4_document_groups: np.ndarray,
    complete_identity: Sequence[Mapping[str, str]],
    pdf_dir: Path,
) -> FrozenMilInputs:
    manifest_path = mil_dir / mil_freeze.MANIFEST_NAME
    bags_path = mil_dir / mil_freeze.FEATURE_NAME
    receipt_path = mil_dir / mil_freeze.RECEIPT_NAME
    candidate_path = mil_dir / mil_freeze.CANDIDATE_NAME
    manifest = mil_freeze.load_json(manifest_path, "MIL manifest")
    candidate = mil_freeze.load_json(candidate_path, "MIL candidate")
    if (
        manifest.get("schema_version") != mil_freeze.MANIFEST_SCHEMA
        or manifest.get("feature_schema_version") != mil_freeze.FEATURE_SCHEMA
        or manifest.get("status") != "COMPLETE_FROZEN_WITHOUT_LABEL_ACCESS"
        or candidate.get("schema_version") != mil_freeze.CANDIDATE_SCHEMA
        or candidate.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
    ):
        raise MilEvaluationError("MIL frozen schema or status changed")
    for payload in (manifest, candidate):
        boundary = payload.get("runtime_boundary") or {}
        if any(
            boundary.get(name) is not False
            for name in (
                "labels_read",
                "pmagresult_or_rockmag_read",
                "prior_predictions_read",
                "evaluation_results_read",
            )
        ):
            raise MilEvaluationError("MIL frozen source boundary changed")
    outputs = manifest.get("outputs") or {}
    for name, path in (
        ("bags", bags_path),
        ("receipt", receipt_path),
        ("candidate_freeze", candidate_path),
    ):
        item = outputs.get(name) or {}
        _verify_file(path, item.get("sha256"), f"MIL {name}")
    if (
        clean(manifest.get("candidate_freeze_sha256")) != file_sha256(candidate_path)
        or clean(candidate.get("complete_alignment_sha256"))
        != trace_freeze._record_alignment_sha256(records)
        or clean((manifest.get("alignment") or {}).get("alignment_sha256"))
        != trace_freeze._record_alignment_sha256(records)
    ):
        raise MilEvaluationError("MIL frozen alignment hash changed")
    _validate_manifest_inputs(manifest, candidate)
    _, _, membership_sha = trace_freeze.select_pdf_backed_records(
        list(complete_identity), pdf_dir
    )
    if membership_sha != clean(
        ((manifest.get("inputs") or {}).get("pdf_membership") or {}).get("sha256")
    ):
        raise MilEvaluationError("MIL PDF membership changed")
    with np.load(bags_path) as frozen:
        arrays = {name: frozen[name].copy() for name in frozen.files}
    feature_names = list(manifest.get("instance_feature_names") or [])
    if (
        len(feature_names) != EXPECTED_INSTANCE_FEATURES
        or feature_names != list(candidate.get("instance_feature_names") or [])
        or clean(manifest.get("instance_feature_names_sha256"))
        != json_sha256(feature_names)
        or clean(candidate.get("instance_feature_names_sha256"))
        != json_sha256(feature_names)
    ):
        raise MilEvaluationError("MIL instance feature names changed")
    expected_hashes = manifest.get("array_sha256") or {}
    for name, value in arrays.items():
        if trace_freeze.array_sha256(value) != clean(expected_hashes.get(name)):
            raise MilEvaluationError(f"MIL {name} array hash changed")
    validate_bag_alignment(
        arrays, records, v4_fold_ids, v4_document_groups, feature_names
    )
    multiplicity = Counter(clean(record["REFNO"]) for record in complete_identity)
    expected_source = np.asarray(
        [
            index
            for index, record in enumerate(records)
            if multiplicity[clean(record["REFNO"])] == 1
        ],
        dtype=np.int32,
    )
    if (
        len(arrays["source_row_indices"]) != EXPECTED_BAGS
        or not np.array_equal(arrays["source_row_indices"], expected_source)
        or int(candidate.get("single_target_rows", -1)) != EXPECTED_BAGS
        or trace_freeze.array_sha256(arrays["source_row_indices"])
        != clean(candidate.get("single_source_row_indices_sha256"))
        or trace_freeze.array_sha256(arrays["fold_ids"])
        != clean(candidate.get("single_fold_ids_sha256"))
        or trace_freeze.array_sha256(arrays["document_groups"])
        != clean(candidate.get("single_document_groups_sha256"))
    ):
        raise MilEvaluationError("MIL single-target scope hash changed")
    receipt_rows = _load_receipt_rows(receipt_path)
    validated_rows, metadata = _validate_receipt(receipt_rows, arrays, feature_names)
    event_documents = tuple(build_event_document(row, config) for row in validated_rows)
    vectorizers = config["event_documents"]
    word_matrix = make_hashing_vectorizer(vectorizers["word_hashing"]).transform(
        event_documents
    ).tocsr()
    char_matrix = make_hashing_vectorizer(vectorizers["char_hashing"]).transform(
        event_documents
    ).tocsr()
    static = bag_feature_mean_max(arrays["instance_features"], arrays["bag_indptr"])
    empty = np.diff(arrays["bag_indptr"]) == 0
    if (
        word_matrix.shape[0] != EXPECTED_BAGS
        or char_matrix.shape[0] != EXPECTED_BAGS
        or static.shape != (EXPECTED_BAGS, EXPECTED_INSTANCE_FEATURES * 2)
        or int(empty.sum()) != int(manifest.get("papers_without_instances", -1))
    ):
        raise MilEvaluationError("MIL label-blind document features do not align")
    return FrozenMilInputs(
        arrays=arrays,
        receipt_rows=tuple(validated_rows),
        event_documents=event_documents,
        word_matrix=word_matrix,
        char_matrix=char_matrix,
        static_bag_features=static,
        empty_bags=empty,
        instance_metadata=metadata,
        manifest=manifest,
        candidate=candidate,
    )


def select_top_k_instances(
    scores: np.ndarray,
    bag_indptr: np.ndarray,
    bag_indices: Sequence[int],
    top_k: int,
) -> dict[int, np.ndarray]:
    values = np.asarray(scores, dtype=np.float64)
    if top_k < 1 or len(values) != int(bag_indptr[-1]) or not np.isfinite(values).all():
        raise ValueError("invalid hard-EM selection inputs")
    output: dict[int, np.ndarray] = {}
    for raw_bag in bag_indices:
        bag = int(raw_bag)
        start, end = int(bag_indptr[bag]), int(bag_indptr[bag + 1])
        indices = np.arange(start, end, dtype=np.int64)
        if len(indices):
            order = np.lexsort((indices, -values[indices]))
            output[bag] = indices[order[: min(top_k, len(indices))]]
        else:
            output[bag] = np.empty(0, dtype=np.int64)
    return output


def build_hard_em_training_rows(
    train_bags: Sequence[int],
    bag_labels: np.ndarray,
    bag_indptr: np.ndarray,
    selected_positive: Mapping[int, np.ndarray],
) -> EmTrainingRows:
    indices = []
    labels = []
    weights = []
    owners = []
    for raw_bag in train_bags:
        bag = int(raw_bag)
        start, end = int(bag_indptr[bag]), int(bag_indptr[bag + 1])
        if int(bag_labels[bag]) == 1:
            current = np.asarray(selected_positive.get(bag, ()), dtype=np.int64)
            if any(index < start or index >= end for index in current):
                raise ValueError("positive hard-EM selection leaves its bag")
        else:
            current = np.arange(start, end, dtype=np.int64)
        if not len(current):
            continue
        indices.extend(map(int, current))
        labels.extend([int(bag_labels[bag])] * len(current))
        weights.extend([1.0 / len(current)] * len(current))
        owners.extend([bag] * len(current))
    return EmTrainingRows(
        instance_indices=np.asarray(indices, dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int8),
        sample_weight=np.asarray(weights, dtype=np.float64),
        bag_indices=np.asarray(owners, dtype=np.int32),
    )


def aggregate_instance_probabilities(
    instance_probability: np.ndarray,
    bag_indptr: np.ndarray,
    bag_indices: Sequence[int],
    mode: str,
    *,
    empty_probability: float = 0.5,
) -> EvidencePrediction:
    values = np.asarray(instance_probability, dtype=np.float64)
    if len(values) != int(bag_indptr[-1]) or not np.isfinite(values).all():
        raise ValueError("invalid MIL instance probabilities")
    count = len(bag_indices)
    probability = np.full(count, empty_probability, dtype=np.float64)
    means = np.full(count, empty_probability, dtype=np.float64)
    maxima = np.full(count, empty_probability, dtype=np.float64)
    top_index = np.full(count, -1, dtype=np.int32)
    top_probability = np.full(count, empty_probability, dtype=np.float64)
    for offset, raw_bag in enumerate(bag_indices):
        bag = int(raw_bag)
        start, end = int(bag_indptr[bag]), int(bag_indptr[bag + 1])
        if end <= start:
            continue
        current = values[start:end]
        global_indices = np.arange(start, end, dtype=np.int64)
        order = np.lexsort((global_indices, -current))
        best = int(global_indices[order[0]])
        means[offset] = float(current.mean())
        maxima[offset] = float(current.max())
        top_index[offset] = best
        top_probability[offset] = float(values[best])
        if mode == "max":
            probability[offset] = maxima[offset]
        elif mode == "top2mean":
            probability[offset] = float(current[order[: min(2, len(order))]].mean())
        else:
            raise ValueError(f"unknown MIL aggregation mode: {mode}")
    return EvidencePrediction(probability, means, maxima, top_index, top_probability)


def _initial_instance_scores(
    features: np.ndarray, feature_names: Sequence[str], weights: Mapping[str, Any]
) -> np.ndarray:
    output = np.zeros(len(features), dtype=np.float64)
    for name, raw_weight in weights.items():
        if name not in feature_names:
            raise MilEvaluationError(f"hard-EM seed feature is absent: {name}")
        output += float(raw_weight) * np.asarray(
            features[:, feature_names.index(name)], dtype=np.float64
        )
    return output


def _predict_instance_rows(
    model: LogisticRegression,
    scaler: StandardScaler | None,
    features: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    if not len(indices):
        return np.empty(0, dtype=np.float64)
    matrix = np.asarray(features[indices], dtype=np.float64)
    if scaler is not None:
        matrix = np.clip(scaler.transform(matrix), -12.0, 12.0)
    probability = np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
    if not np.isfinite(probability).all():
        raise AssertionError("hard-EM model produced non-finite probabilities")
    return probability


def fit_hard_em(
    *,
    features: np.ndarray,
    feature_names: Sequence[str],
    bag_indptr: np.ndarray,
    bag_labels: np.ndarray,
    train_bags: np.ndarray,
    test_bags: np.ndarray,
    spec: Mapping[str, Any],
    common: Mapping[str, Any],
) -> EvidencePrediction:
    positive = train_bags[bag_labels[train_bags] == 1]
    scores = _initial_instance_scores(
        features, feature_names, common["initial_score_features"]
    )
    selected = select_top_k_instances(
        scores, bag_indptr, positive, int(spec["positive_top_k"])
    )
    model: LogisticRegression | None = None
    scaler: StandardScaler | None = None
    for iteration in range(int(spec["rounds"])):
        rows = build_hard_em_training_rows(
            train_bags, bag_labels, bag_indptr, selected
        )
        if len(rows.instance_indices) == 0 or len(np.unique(rows.labels)) < 2:
            model = None
            break
        train = np.asarray(features[rows.instance_indices], dtype=np.float64)
        scaler = StandardScaler() if bool(common.get("standardize")) else None
        if scaler is not None:
            train = np.clip(scaler.fit_transform(train), -12.0, 12.0)
        model = LogisticRegression(
            C=float(spec["C"]),
            solver=clean(common["solver"]),
            max_iter=int(common["max_iter"]),
            random_state=int(common["random_state"]),
        )
        model.fit(train, rows.labels, sample_weight=rows.sample_weight)
        if iteration + 1 < int(spec["rounds"]):
            positive_indices = np.concatenate(
                [
                    np.arange(int(bag_indptr[bag]), int(bag_indptr[bag + 1]))
                    for bag in positive
                    if int(bag_indptr[bag + 1]) > int(bag_indptr[bag])
                ]
            ) if any(
                int(bag_indptr[bag + 1]) > int(bag_indptr[bag]) for bag in positive
            ) else np.empty(0, dtype=np.int64)
            scores = np.full(len(features), 0.5, dtype=np.float64)
            scores[positive_indices] = _predict_instance_rows(
                model, scaler, features, positive_indices
            )
            selected = select_top_k_instances(
                scores, bag_indptr, positive, int(spec["positive_top_k"])
            )
    instance_probability = np.full(len(features), 0.5, dtype=np.float64)
    test_indices = np.concatenate(
        [
            np.arange(int(bag_indptr[bag]), int(bag_indptr[bag + 1]))
            for bag in test_bags
            if int(bag_indptr[bag + 1]) > int(bag_indptr[bag])
        ]
    ) if any(
        int(bag_indptr[bag + 1]) > int(bag_indptr[bag]) for bag in test_bags
    ) else np.empty(0, dtype=np.int64)
    if model is not None:
        instance_probability[test_indices] = _predict_instance_rows(
            model, scaler, features, test_indices
        )
    return aggregate_instance_probabilities(
        instance_probability,
        bag_indptr,
        test_bags,
        clean(spec["aggregation"]),
        empty_probability=float(common["empty_bag_probability"]),
    )


def _fit_sparse_logistic(
    matrix: sp.csr_matrix,
    labels: np.ndarray,
    train_bags: np.ndarray,
    test_bags: np.ndarray,
    empty_bags: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    output = np.full(len(test_bags), 0.5, dtype=np.float64)
    active_train = train_bags[~empty_bags[train_bags]]
    active_test_mask = ~empty_bags[test_bags]
    if len(active_train) == 0 or len(np.unique(labels[active_train])) < 2:
        return output
    model = LogisticRegression(
        C=float(spec["C"]),
        solver=clean(spec["solver"]),
        max_iter=int(spec["max_iter"]),
        random_state=int(spec["random_state"]),
    )
    model.fit(matrix[active_train], labels[active_train])
    if active_test_mask.any():
        output[active_test_mask] = np.asarray(
            model.predict_proba(matrix[test_bags[active_test_mask]])[:, 1],
            dtype=np.float64,
        )
    if not np.isfinite(output).all():
        raise AssertionError("event-document logistic produced non-finite probabilities")
    return output


def _fit_v4_random_forest(
    matrix: np.ndarray,
    labels: np.ndarray,
    records: Sequence[Mapping[str, str]],
    train: np.ndarray,
    test: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    weights = trace_eval.inverse_identity_weights(train, records)
    model = RandomForestClassifier(
        n_estimators=int(spec["n_estimators"]),
        max_features=float(spec["max_features"]),
        min_samples_leaf=int(spec["min_samples_leaf"]),
        n_jobs=int(spec["n_jobs"]),
        random_state=int(spec["random_state"]),
    )
    model.fit(matrix[train], labels[train], sample_weight=weights)
    probability = np.asarray(model.predict_proba(matrix[test])[:, 1], dtype=np.float64)
    if not np.isfinite(probability).all():
        raise AssertionError("V4 random forest produced non-finite probabilities")
    return probability


def _component_predictions(
    *,
    config: Mapping[str, Any],
    mil: FrozenMilInputs,
    v4_matrix: np.ndarray,
    v4_records: Sequence[Mapping[str, str]],
    labels_all: np.ndarray,
    labels_single: np.ndarray,
    v4_train: np.ndarray,
    train_bags: np.ndarray,
    test_bags: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, EvidencePrediction]]:
    component = config["components"]
    source = mil.arrays["source_row_indices"]
    output = {
        "v4_random_forest_all_pdf": _fit_v4_random_forest(
            v4_matrix,
            labels_all,
            v4_records,
            v4_train,
            source[test_bags].astype(np.int64),
            component["v4_random_forest_all_pdf"],
        ),
        "event_word_logistic": _fit_sparse_logistic(
            mil.word_matrix,
            labels_single,
            train_bags,
            test_bags,
            mil.empty_bags,
            component["event_word_logistic"],
        ),
        "event_char_logistic": _fit_sparse_logistic(
            mil.char_matrix,
            labels_single,
            train_bags,
            test_bags,
            mil.empty_bags,
            component["event_char_logistic"],
        ),
    }
    em = {}
    feature_names = list(mil.manifest["instance_feature_names"])
    for spec in component["hard_em"]:
        name = clean(spec["name"])
        em[name] = fit_hard_em(
            features=mil.arrays["instance_features"],
            feature_names=feature_names,
            bag_indptr=mil.arrays["bag_indptr"],
            bag_labels=labels_single,
            train_bags=train_bags,
            test_bags=test_bags,
            spec=spec,
            common=component["hard_em_common"],
        )
        output[name] = em[name].probability
    if set(output) != set(COMPONENT_NAMES):
        raise AssertionError("MIL component set changed")
    return output, em


def probability_logit(values: np.ndarray, clip: float) -> np.ndarray:
    probability = np.clip(np.asarray(values, dtype=np.float64), clip, 1.0 - clip)
    return np.log(probability / (1.0 - probability))


def build_stack_features(
    component_probability: Mapping[str, np.ndarray],
    em_predictions: Mapping[str, EvidencePrediction],
    static_bag_features: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    stack = config["stack"]
    clip = float(stack["probability_clip"])
    logits = [
        probability_logit(component_probability[name], clip)
        for name in stack["component_logits"]
    ]
    em_statistics = []
    for name in HARD_EM_NAMES:
        em_statistics.extend(
            [em_predictions[name].instance_mean, em_predictions[name].instance_max]
        )
    matrix = np.column_stack((*logits, *em_statistics, static_bag_features))
    if (
        matrix.ndim != 2
        or matrix.shape[1] != int(stack["expected_dimensions"])
        or not np.isfinite(matrix).all()
    ):
        raise AssertionError("MIL stack feature matrix changed")
    return matrix.astype(np.float64, copy=False)


def _fit_stack(
    train: np.ndarray,
    test: np.ndarray,
    labels: np.ndarray,
    spec: Mapping[str, Any],
) -> np.ndarray:
    scaler = StandardScaler() if bool(spec.get("standardize")) else None
    fit = np.asarray(train, dtype=np.float64)
    predict = np.asarray(test, dtype=np.float64)
    if scaler is not None:
        fit = np.clip(scaler.fit_transform(fit), -12.0, 12.0)
        predict = np.clip(scaler.transform(predict), -12.0, 12.0)
    if len(np.unique(labels)) < 2:
        return np.full(len(test), 0.5, dtype=np.float64)
    model = LogisticRegression(
        C=float(spec["C"]),
        solver=clean(spec["solver"]),
        max_iter=int(spec["max_iter"]),
        random_state=int(spec["random_state"]),
    )
    model.fit(fit, labels)
    probability = np.asarray(model.predict_proba(predict)[:, 1], dtype=np.float64)
    if not np.isfinite(probability).all():
        raise AssertionError("MIL stack produced non-finite probabilities")
    return probability


def _full_prediction_buffers(length: int) -> tuple[
    dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]
]:
    components = {
        name: np.full(length, np.nan, dtype=np.float64) for name in COMPONENT_NAMES
    }
    em = {
        name: {
            "mean": np.full(length, np.nan, dtype=np.float64),
            "max": np.full(length, np.nan, dtype=np.float64),
            "top_index": np.full(length, -1, dtype=np.int32),
            "top_probability": np.full(length, np.nan, dtype=np.float64),
        }
        for name in HARD_EM_NAMES
    }
    return components, em


def _em_views(
    values: Mapping[str, Mapping[str, np.ndarray]],
    component_probability: Mapping[str, np.ndarray],
    indices: np.ndarray,
) -> dict[str, EvidencePrediction]:
    return {
        name: EvidencePrediction(
            probability=component_probability[name][indices],
            instance_mean=values[name]["mean"][indices],
            instance_max=values[name]["max"][indices],
            top_global_index=values[name]["top_index"][indices],
            top_probability=values[name]["top_probability"][indices],
        )
        for name in HARD_EM_NAMES
    }


def paired_change(labels: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    base_correct = (baseline >= 0.5) == labels
    candidate_correct = (candidate >= 0.5) == labels
    repairs = int((~base_correct & candidate_correct).sum())
    degradations = int((base_correct & ~candidate_correct).sum())
    discordant = repairs + degradations
    p_value = (
        float(binomtest(min(repairs, degradations), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "repairs": repairs,
        "degradations": degradations,
        "net_repairs": repairs - degradations,
        "accuracy_delta": round(
            float(
                (
                    candidate_correct.astype(np.float64)
                    - base_correct.astype(np.float64)
                ).mean()
            ),
            8,
        ),
        "mcnemar_exact_two_sided_p": round(p_value, 8),
    }


def _top_instance_receipt(
    global_index: int,
    probability: float,
    metadata: Mapping[int, Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    if global_index < 0:
        return None
    item = metadata[int(global_index)]
    return {
        "global_instance_index": int(global_index),
        "probability": float(probability),
        "chunk_id": clean(item.get("chunk_id")),
        "text_sha256": clean(item.get("text_sha256")),
        "page": int(item.get("page", 0)),
        "scale": clean(item.get("scale")),
        "section": clean(item.get("section")),
        "primary_route": clean(item.get("primary_route")),
        "trusted_categories": item.get("trusted_categories") or {},
        "exclusion_codes": list(item.get("exclusion_codes") or []),
        "text": clean(item.get("text")),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mil-dir", type=Path, default=DEFAULT_MIL_DIR)
    parser.add_argument("--v4-features", type=Path, default=DEFAULT_V4_FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4-context", type=Path, default=DEFAULT_V4_CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=DEFAULT_V4_CANDIDATE)
    parser.add_argument("--v4-folds", type=Path, default=DEFAULT_V4_FOLDS)
    parser.add_argument("--v4-oof", type=Path, default=DEFAULT_V4_OOF)
    parser.add_argument("--v4-report", type=Path, default=DEFAULT_V4_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MIL_DIR)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    output_lock = _acquire_lock(output_dir)
    try:
        config_path = args.config.resolve()
        config = load_evaluation_config(config_path)
        (
            records,
            v4_matrix,
            _v4_names,
            _v4_documents,
            v4_fold_ids,
            v4_document_groups,
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
        mil = load_frozen_mil_inputs(
            mil_dir=args.mil_dir.resolve(),
            config=config,
            records=records,
            v4_fold_ids=v4_fold_ids,
            v4_document_groups=v4_document_groups,
            complete_identity=complete_identity,
            pdf_dir=args.pdf_dir.resolve(),
        )
        source = mil.arrays["source_row_indices"].astype(np.int64)
        event_document_hash = json_sha256(list(mil.event_documents))
        evaluation_freeze_path = output_dir / EVALUATION_FREEZE_NAME
        evaluation_freeze = {
            "schema_version": EVALUATION_FREEZE_SCHEMA,
            "status": "FROZEN_BEFORE_LABEL_ACCESS",
            "scope": {
                "rows": EXPECTED_BAGS,
                "coverage": 1.0,
                "missing_pdf_targets_in_denominator": False,
            },
            "fixed_config": config,
            "label_blind_event_features": {
                "event_documents_sha256": event_document_hash,
                "word_matrix_sha256": sparse_matrix_sha256(mil.word_matrix),
                "word_matrix_shape": list(mil.word_matrix.shape),
                "char_matrix_sha256": sparse_matrix_sha256(mil.char_matrix),
                "char_matrix_shape": list(mil.char_matrix.shape),
                "static_bag_features_sha256": trace_freeze.array_sha256(
                    mil.static_bag_features
                ),
                "static_bag_features_shape": list(mil.static_bag_features.shape),
                "empty_bag_mask_sha256": trace_freeze.array_sha256(mil.empty_bags),
            },
            "inputs": {
                "config": {"path": str(config_path), "sha256": file_sha256(config_path)},
                "mil_manifest": {
                    "path": str(args.mil_dir.resolve() / mil_freeze.MANIFEST_NAME),
                    "sha256": file_sha256(args.mil_dir.resolve() / mil_freeze.MANIFEST_NAME),
                },
                "mil_bags": {
                    "path": str(args.mil_dir.resolve() / mil_freeze.FEATURE_NAME),
                    "sha256": file_sha256(args.mil_dir.resolve() / mil_freeze.FEATURE_NAME),
                },
                "mil_receipt": {
                    "path": str(args.mil_dir.resolve() / mil_freeze.RECEIPT_NAME),
                    "sha256": file_sha256(args.mil_dir.resolve() / mil_freeze.RECEIPT_NAME),
                },
                "mil_candidate": {
                    "path": str(args.mil_dir.resolve() / mil_freeze.CANDIDATE_NAME),
                    "sha256": file_sha256(args.mil_dir.resolve() / mil_freeze.CANDIDATE_NAME),
                },
                "v4_manifest": {
                    "path": str(args.v4_manifest.resolve()),
                    "sha256": file_sha256(args.v4_manifest.resolve()),
                },
                "v4_features": {
                    "path": str(args.v4_features.resolve()),
                    "sha256": file_sha256(args.v4_features.resolve()),
                },
                "v4_folds": {
                    "path": str(args.v4_folds.resolve()),
                    "sha256": file_sha256(args.v4_folds.resolve()),
                },
                "evaluator_code": {
                    "path": str(Path(__file__).resolve()),
                    "sha256": file_sha256(Path(__file__).resolve()),
                },
            },
            "postfreeze_only_inputs": {
                "labels": {"path": str(args.targets.resolve()), "columns": ["RESULTNO", "R3"]},
                "saved_v4_oof_audit": {"path": str(args.v4_oof.resolve())},
                "saved_v4_report_audit": {"path": str(args.v4_report.resolve())},
            },
            "runtime_boundary": {
                "labels_read": False,
                "pmagresult_or_rockmag_read": False,
                "prior_predictions_read": False,
                "evaluation_results_read": False,
                "outer_test_labels_used_for_training_or_selection": False,
            },
        }
        mil_freeze.atomic_json(evaluation_freeze_path, evaluation_freeze)
        run_state = EvaluationRunState(
            output_dir / RUN_STATE_NAME, file_sha256(evaluation_freeze_path)
        )

        # This is the first outcome access in this process.
        label_map = trace_eval.load_labels(
            args.targets.resolve(), {record["RESULTNO"] for record in records}
        )
        labels_all = np.asarray(
            [label_map[record["RESULTNO"]] for record in records], dtype=np.int8
        )
        labels_single = labels_all[source]
        if set(map(int, np.unique(labels_single))) != {0, 1}:
            raise MilEvaluationError("single-target MIL evaluation requires both classes")

        length = EXPECTED_BAGS
        fixed_components, fixed_em = _full_prediction_buffers(length)
        primary = np.full(length, np.nan, dtype=np.float64)
        fold_audit = []
        protocol = config["protocol"]
        for outer_fold in range(1, int(protocol["outer_folds"]) + 1):
            outer_test = np.flatnonzero(mil.arrays["fold_ids"] == outer_fold)
            outer_train = np.flatnonzero(mil.arrays["fold_ids"] != outer_fold)
            outer_v4_pool = np.flatnonzero(v4_fold_ids != outer_fold)
            test_groups = set(map(clean, mil.arrays["document_groups"][outer_test]))
            if test_groups & set(map(clean, v4_document_groups[outer_v4_pool])):
                raise AssertionError("MIL outer fold leaks a normalized document")
            inner_splits = nested.deterministic_group_folds(
                outer_train,
                [clean(mil.arrays["document_groups"][index]) for index in outer_train],
                n_splits=int(protocol["inner_folds"]),
                seed=int(protocol["inner_seed"]) + outer_fold,
            )
            inner_components, inner_em = _full_prediction_buffers(length)
            for inner_train, inner_validation in inner_splits:
                validation_groups = set(
                    map(clean, mil.arrays["document_groups"][inner_validation])
                )
                v4_train = np.asarray(
                    [
                        index
                        for index in outer_v4_pool
                        if clean(v4_document_groups[index]) not in validation_groups
                    ],
                    dtype=np.int64,
                )
                component_values, em_values = _component_predictions(
                    config=config,
                    mil=mil,
                    v4_matrix=v4_matrix,
                    v4_records=records,
                    labels_all=labels_all,
                    labels_single=labels_single,
                    v4_train=v4_train,
                    train_bags=inner_train,
                    test_bags=inner_validation,
                )
                for name, values in component_values.items():
                    inner_components[name][inner_validation] = values
                for name, values in em_values.items():
                    inner_em[name]["mean"][inner_validation] = values.instance_mean
                    inner_em[name]["max"][inner_validation] = values.instance_max
                    inner_em[name]["top_index"][inner_validation] = values.top_global_index
                    inner_em[name]["top_probability"][inner_validation] = values.top_probability
            if any(
                not np.isfinite(values[outer_train]).all()
                for values in inner_components.values()
            ) or any(
                not np.isfinite(inner_em[name][stat][outer_train]).all()
                for name in HARD_EM_NAMES
                for stat in ("mean", "max", "top_probability")
            ):
                raise AssertionError("MIL inner OOF components are incomplete")
            inner_stack = build_stack_features(
                {name: values[outer_train] for name, values in inner_components.items()},
                _em_views(inner_em, inner_components, outer_train),
                mil.static_bag_features[outer_train],
                config,
            )
            outer_components, outer_em_values = _component_predictions(
                config=config,
                mil=mil,
                v4_matrix=v4_matrix,
                v4_records=records,
                labels_all=labels_all,
                labels_single=labels_single,
                v4_train=outer_v4_pool,
                train_bags=outer_train,
                test_bags=outer_test,
            )
            outer_stack = build_stack_features(
                outer_components,
                outer_em_values,
                mil.static_bag_features[outer_test],
                config,
            )
            probability = _fit_stack(
                inner_stack,
                outer_stack,
                labels_single[outer_train],
                config["stack"],
            )
            empty_test = mil.empty_bags[outer_test]
            probability[empty_test] = outer_components["v4_random_forest_all_pdf"][empty_test]
            primary[outer_test] = probability
            for name, values in outer_components.items():
                fixed_components[name][outer_test] = values
            for name, values in outer_em_values.items():
                fixed_em[name]["mean"][outer_test] = values.instance_mean
                fixed_em[name]["max"][outer_test] = values.instance_max
                fixed_em[name]["top_index"][outer_test] = values.top_global_index
                fixed_em[name]["top_probability"][outer_test] = values.top_probability
            fold_audit.append(
                {
                    "outer_fold": outer_fold,
                    "test_rows": len(outer_test),
                    "single_training_rows": len(outer_train),
                    "all_pdf_training_rows": len(outer_v4_pool),
                    "inner_folds": len(inner_splits),
                    "empty_test_bags": int(empty_test.sum()),
                    "stack_training_fit_metrics": trace_eval.metric(
                        labels_single[outer_train],
                        _fit_stack(
                            inner_stack,
                            inner_stack,
                            labels_single[outer_train],
                            config["stack"],
                        ),
                    ),
                    "outer_test_metrics": trace_eval.metric(
                        labels_single[outer_test], probability
                    ),
                }
            )
            print(
                f"evidence MIL OOF: fold={outer_fold}/5 "
                f"accuracy={fold_audit[-1]['outer_test_metrics']['accuracy']:.4f}",
                flush=True,
            )

        if not np.isfinite(primary).all() or any(
            not np.isfinite(values).all() for values in fixed_components.values()
        ) or any(
            not np.isfinite(fixed_em[name][stat]).all()
            for name in HARD_EM_NAMES
            for stat in ("mean", "max", "top_probability")
        ):
            raise AssertionError("MIL outer OOF predictions are incomplete")
        oof_arrays: dict[str, np.ndarray] = {
            "source_row_indices": source.astype(np.int32),
            "labels": labels_single,
            "fold_ids": mil.arrays["fold_ids"].astype(np.int8),
            "empty_bags": mil.empty_bags.astype(bool),
            "nested_primary": primary.astype(np.float64),
        }
        for name, values in fixed_components.items():
            oof_arrays[f"component__{name}"] = values.astype(np.float64)
        for name in HARD_EM_NAMES:
            for stat in ("mean", "max", "top_probability"):
                oof_arrays[f"{name}__instance_{stat}"] = fixed_em[name][stat].astype(
                    np.float64
                )
            oof_arrays[f"{name}__top_global_index"] = fixed_em[name][
                "top_index"
            ].astype(np.int32)
        oof_path = output_dir / OOF_NAME
        provisional_path = oof_path.with_name(oof_path.name + ".preaudit")
        provisional_path.unlink(missing_ok=True)
        mil_freeze.atomic_npz(provisional_path, **oof_arrays)
        with np.load(provisional_path) as preaudit:
            preaudit_source = preaudit["source_row_indices"].copy()
            preaudit_labels = preaudit["labels"].copy()
            preaudit_folds = preaudit["fold_ids"].copy()
            preaudit_empty = preaudit["empty_bags"].copy()
            base_probability = preaudit[
                "component__v4_random_forest_all_pdf"
            ].astype(np.float64)
        if (
            not np.array_equal(preaudit_source, source.astype(np.int32))
            or not np.array_equal(preaudit_labels, labels_single)
            or not np.array_equal(preaudit_folds, mil.arrays["fold_ids"])
            or not np.array_equal(preaudit_empty, mil.empty_bags)
            or not np.array_equal(
                base_probability,
                fixed_components["v4_random_forest_all_pdf"],
            )
        ):
            provisional_path.unlink(missing_ok=True)
            raise AssertionError("MIL preaudit archive changed predictions or identity")
        try:
            v4_report_path = args.v4_report.resolve()
            v4_oof_path = args.v4_oof.resolve()
            v4_report = mil_freeze.load_json(v4_report_path, "saved V4 report")
            _verify_file(
                v4_oof_path,
                (v4_report.get("output") or {}).get("sha256"),
                "saved V4 OOF",
            )
            with np.load(v4_oof_path) as saved:
                saved_v4 = saved["random_forest"].copy()
                saved_labels = saved["labels"].copy()
                saved_folds = saved["fold_ids"].copy()
            if (
                saved_v4.dtype != np.float32
                or saved_labels.dtype != np.int8
                or saved_folds.dtype != np.int8
                or saved_v4.shape != labels_all.shape
                or not np.array_equal(saved_labels, labels_all)
                or not np.array_equal(saved_folds, v4_fold_ids)
            ):
                raise MilEvaluationError("saved V4 identity arrays changed")
            float32_reproduction = np.array_equal(
                saved_v4[source], base_probability.astype(np.float32)
            )
            maximum_v4_difference = float(
                np.max(
                    np.abs(
                        saved_v4[source].astype(np.float64) - base_probability
                    )
                )
            )
            if not float32_reproduction:
                raise MilEvaluationError("MIL evaluator does not exactly reproduce V4 RF")
        except BaseException:
            provisional_path.unlink(missing_ok=True)
            raise
        provisional_path.replace(oof_path)

        with np.load(oof_path) as frozen:
            frozen_source = frozen["source_row_indices"].astype(np.int32)
            frozen_labels = frozen["labels"].astype(np.int8)
            frozen_folds = frozen["fold_ids"].astype(np.int8)
            frozen_empty = frozen["empty_bags"].astype(bool)
            frozen_primary = frozen["nested_primary"].astype(np.float64)
            frozen_components = {
                name: frozen[f"component__{name}"].astype(np.float64)
                for name in COMPONENT_NAMES
            }
            frozen_top = {
                name: frozen[f"{name}__top_global_index"].astype(np.int32)
                for name in HARD_EM_NAMES
            }
            frozen_top_probability = {
                name: frozen[f"{name}__instance_top_probability"].astype(np.float64)
                for name in HARD_EM_NAMES
            }
        if (
            not np.array_equal(frozen_source, source.astype(np.int32))
            or not np.array_equal(frozen_labels, labels_single)
            or not np.array_equal(frozen_folds, mil.arrays["fold_ids"])
            or not np.array_equal(frozen_empty, mil.empty_bags)
            or not np.isfinite(frozen_primary).all()
            or any(not np.isfinite(value).all() for value in frozen_components.values())
            or any(
                not np.isfinite(value).all()
                for value in frozen_top_probability.values()
            )
        ):
            raise AssertionError("published MIL OOF identity or probabilities changed")
        receipt_path = output_dir / RECEIPT_NAME
        receipt_rows = []
        primary_top_source = clean(config["receipt"]["top_instance_source"])
        for bag_index in range(length):
            top_by_model = {
                name: _top_instance_receipt(
                    int(frozen_top[name][bag_index]),
                    float(frozen_top_probability[name][bag_index]),
                    mil.instance_metadata,
                )
                for name in HARD_EM_NAMES
            }
            receipt_rows.append(
                {
                    "bag_index": bag_index,
                    "v4_row_index": int(source[bag_index]),
                    "REFNO": clean(mil.arrays["refnos"][bag_index]),
                    "RESULTNO": clean(mil.arrays["resultnos"][bag_index]),
                    "fold": int(mil.arrays["fold_ids"][bag_index]),
                    "label": int(frozen_labels[bag_index]),
                    "empty_bag": bool(mil.empty_bags[bag_index]),
                    "components": {
                        name: float(frozen_components[name][bag_index])
                        for name in COMPONENT_NAMES
                    },
                    "probability": float(frozen_primary[bag_index]),
                    "prediction": int(frozen_primary[bag_index] >= 0.5),
                    "top_instance": top_by_model[primary_top_source],
                    "top_instances_by_hard_em_component": top_by_model,
                }
            )
        receipt = {
            "schema_version": "r3_evidence_mil_single872_grouped_oof_receipt_v1",
            "status": "OOF_PREDICTIONS_ATOMICALLY_PUBLISHED_AFTER_V4_REPRODUCTION_AUDIT",
            "evaluation_freeze_sha256": file_sha256(evaluation_freeze_path),
            "oof": {"path": str(oof_path), "sha256": file_sha256(oof_path)},
            "rows": receipt_rows,
        }
        mil_freeze.atomic_json(receipt_path, receipt)

        component_metrics = {
            name: trace_eval.metric(frozen_labels, values)
            for name, values in frozen_components.items()
        }
        primary_metric = trace_eval.metric(frozen_labels, frozen_primary)
        base_metric = component_metrics["v4_random_forest_all_pdf"]
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "POSTHOC_EVALUATION_OF_FROZEN_BAGS_AND_FIXED_MIL_STACK",
            "scope": {
                "rows": length,
                "papers": length,
                "coverage": 1.0,
                "positive": int(frozen_labels.sum()),
                "negative": int((frozen_labels == 0).sum()),
                "empty_bags": int(mil.empty_bags.sum()),
            },
            "primary_strategy": "fixed_nested_logistic_mil_stack",
            "threshold": 0.5,
            "metrics": {
                "strict_v4_random_forest": base_metric,
                "fixed_nested_logistic_mil_stack": primary_metric,
                "components": component_metrics,
            },
            "delta_vs_strict_v4_random_forest": paired_change(
                frozen_labels,
                frozen_components["v4_random_forest_all_pdf"],
                frozen_primary,
            ),
            "fold_audit": fold_audit,
            "v4_random_forest_reproduction_audit": {
                "saved_oof_opened_only_after_new_predictions_written_to_provisional_archive": True,
                "saved_report": {
                    "path": str(v4_report_path),
                    "sha256": file_sha256(v4_report_path),
                },
                "saved_oof": {
                    "path": str(v4_oof_path),
                    "sha256": file_sha256(v4_oof_path),
                },
                "saved_labels_exact": True,
                "saved_fold_ids_exact": True,
                "float32_probability_array_exact": float32_reproduction,
                "maximum_absolute_probability_difference": maximum_v4_difference,
            },
            "promotion_decision": {
                "reaches_90_percent_full_coverage_internal_oof": primary_metric["accuracy"]
                >= 0.90,
                "improves_strict_v4": primary_metric["accuracy"] > base_metric["accuracy"],
                "promote_to_strict_production_scoring": False,
                "reason": "Packaging and an independent boundary audit remain required.",
            },
            "claim_boundary": {
                "internal_grouped_nested_oof": True,
                "independent_external_validation": False,
                "bag_and_hash_feature_generation_used_labels": False,
                "pmagresult_or_rockmag_used": False,
                "prior_predictions_opened_before_preaudit": False,
                "outer_test_labels_used_for_training_or_selection": False,
                "missing_pdf_targets_in_denominator": False,
                "selective_abstention_used": False,
            },
            "runtime": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "scipy": __import__("scipy").__version__,
                "sklearn": sklearn.__version__,
            },
            "inputs": {
                "evaluation_freeze": {
                    "path": str(evaluation_freeze_path),
                    "sha256": file_sha256(evaluation_freeze_path),
                },
                "targets_postfreeze_only": {
                    "path": str(args.targets.resolve()),
                    "sha256": file_sha256(args.targets.resolve()),
                    "columns": ["RESULTNO", "R3"],
                },
            },
            "outputs": {
                "oof": {"path": str(oof_path), "sha256": file_sha256(oof_path)},
                "receipt": {
                    "path": str(receipt_path),
                    "sha256": file_sha256(receipt_path),
                },
            },
        }
        report_path = output_dir / REPORT_NAME
        mil_freeze.atomic_json(report_path, report)
        run_state.complete(report_path, oof_path, receipt_path)
        print(
            json.dumps(
                {
                    "report": str(report_path),
                    "strict_v4": base_metric,
                    "fixed_mil_stack": primary_metric,
                    "delta": report["delta_vs_strict_v4_random_forest"],
                    "promotion": report["promotion_decision"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0
    finally:
        output_lock.close()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPONENT_NAMES",
    "EXPECTED_STACK_FEATURES",
    "EmTrainingRows",
    "EvidencePrediction",
    "MilEvaluationError",
    "aggregate_instance_probabilities",
    "bag_feature_mean_max",
    "build_event_document",
    "build_hard_em_training_rows",
    "build_stack_features",
    "fit_hard_em",
    "load_evaluation_config",
    "probability_logit",
    "select_top_k_instances",
    "sparse_matrix_sha256",
    "validate_bag_alignment",
    "validate_trusted_instance",
]
