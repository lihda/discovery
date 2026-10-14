#!/usr/bin/env python3
"""Freeze label-free component-manifest features for exact300 R3 evaluation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_r3_exact300_auxiliary_numeric_cache as numeric_builder  # noqa: E402
import build_r3_exact300_own_study_cache as own_builder  # noqa: E402
import experiment_r3_exact300_auxiliary_model_sweep as numeric_sweep  # noqa: E402
import experiment_r3_exact300_auxiliary_training as auxiliary  # noqa: E402
import experiment_r3_exact300_own_study_auxiliary as own_evaluation  # noqa: E402
import experiment_r3_pdf_only_strategies as base  # noqa: E402


DEFAULT_COMPONENT_MANIFEST = ROOT / "tmp/r3_component_manifest/component_manifest.jsonl"
DEFAULT_COMPONENT_RECEIPT = ROOT / "tmp/r3_component_manifest/freeze_receipt.json"
OUTPUT_DIR = ROOT / "tmp/r3_exact_300_resultno_v2/component_bridge_cache_v1"
DEFAULT_CACHE = OUTPUT_DIR / "features.npz"
DEFAULT_MANIFEST = OUTPUT_DIR / "features.manifest.json"
EXPECTED_ARRAY_NAMES = {
    "component_all",
    "component_bridge",
    "combined_plus_component_all",
    "combined_plus_component_bridge",
    "combined_plus_own_target_component_all",
    "combined_plus_own_target_component_bridge",
}
FORBIDDEN_NAME_RE = re.compile(
    r"(?i)^(?:labels?|answers?|refnos?|resultnos?|rocknames?|ids?)$"
)
SEMANTIC_COMPONENTS = (
    "nrm",
    "chrm",
    "characteristic",
    "primary",
    "secondary",
    "normal",
    "reverse",
    "reversed",
    "low_temperature",
    "intermediate_temperature",
    "high_temperature",
    "low_coercivity",
    "high_coercivity",
    "irm",
    "trm",
)


def validate_component_inputs(
    manifest_path: Path, receipt_path: Path, full_identity: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest_record = receipt.get("manifest", {})
    script_record = receipt.get("script", {})
    identity_record = receipt.get("inputs", {}).get("identity", {})
    if (
        receipt.get("schema_version") != "r3_source_component_manifest_freeze_v1"
        or receipt.get("status") != "FROZEN_BEFORE_LABEL_ACCESS"
        or identity_record.get("sha256") != auxiliary.FULL_IDENTITY_SHA256
        or base.file_sha256(full_identity) != auxiliary.FULL_IDENTITY_SHA256
        or manifest_record.get("sha256") != base.file_sha256(manifest_path)
        or script_record.get("sha256")
        != base.file_sha256(Path(str(script_record.get("path", ""))))
        or int(manifest_record.get("identity_rows", -1))
        != auxiliary.EXPECTED_FULL_PDF_ROWS
        or int(manifest_record.get("paper_records", -1))
        != auxiliary.EXPECTED_FULL_PDF_PAPERS
    ):
        raise ValueError("component manifest receipt is not a valid pre-label freeze")
    papers: list[dict[str, Any]] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                papers.append(json.loads(line))
    if len(papers) != auxiliary.EXPECTED_FULL_PDF_PAPERS:
        raise ValueError("component manifest paper count differs from frozen receipt")
    return receipt, papers


def build_component_features(
    records: Sequence[Mapping[str, str]], papers: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict[str, Any]]:
    concept_vocabulary = sorted(
        {
            str(concept)
            for paper in papers
            for concept in paper["article"]["concepts"]
        }
        | {
            str(concept)
            for paper in papers
            for target in paper["targets"]
            for concept in target["direct_concepts"] + target["bridged_concepts"]
        }
    )
    target_lookup: dict[tuple[str, str], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for paper in papers:
        refno = str(paper["refno"])
        for target in paper["targets"]:
            for resultno in target["resultnos"]:
                key = (refno, str(resultno))
                if key in target_lookup:
                    raise ValueError(f"duplicate component target row: {key}")
                target_lookup[key] = (paper, target)

    core_names = [
        "article_page_count_log",
        "article_target_count_log",
        "article_relevant_scope_count_log",
        "article_knowledge_scope_count_log",
        "article_shared_knowledge_scope_count_log",
        "article_collective_knowledge_scope_count_log",
        "article_has_shared_evidence",
        "target_direct_scope_count_log",
        "target_linked_scope_count_log",
        "target_table_scope_count_log",
        "target_caption_scope_count_log",
        "target_sample_id_count_log",
        "target_component_count_log",
        "target_direct_concept_count_log",
        "target_bridged_concept_count_log",
        "target_sample_bridge",
        "target_component_bridge",
        "target_collective_scope",
        "target_article_propagation_eligible",
        "target_identity_alias_count_log",
        "target_source_confirmed_alias_count_log",
        "target_source_derived_alias_count_log",
        "target_sample_bridge_scope_count_log",
        "target_component_bridge_scope_count_log",
        "target_bridged_scope_count_log",
        "target_direct_to_article_scope_ratio",
        "target_linked_to_article_scope_ratio",
        "target_bridged_to_linked_scope_ratio",
        "article_shared_to_knowledge_scope_ratio",
    ]
    direct_names = [f"target_direct_concept__{value}" for value in concept_vocabulary]
    bridged_names = [f"target_bridged_concept__{value}" for value in concept_vocabulary]
    article_names = [f"article_concept__{value}" for value in concept_vocabulary]
    component_names = [f"target_component__{value}" for value in SEMANTIC_COMPONENTS]
    all_names = core_names + direct_names + bridged_names + article_names + component_names
    bridge_names = [
        name
        for name in all_names
        if name.startswith("target_")
        or name in {
            "article_target_count_log",
            "article_relevant_scope_count_log",
            "article_knowledge_scope_count_log",
            "article_shared_to_knowledge_scope_ratio",
        }
    ]
    bridge_indices = [all_names.index(name) for name in bridge_names]
    rows: list[list[float]] = []
    missing: list[tuple[str, str]] = []
    identity_mismatch: list[tuple[str, str]] = []
    for record in records:
        key = (record["REFNO"], record["RESULTNO"])
        item = target_lookup.get(key)
        if item is None:
            missing.append(key)
            continue
        paper, target = item
        if record["ROCKNAME"] not in set(map(str, target["rocknames"])):
            identity_mismatch.append(key)
        article = paper["article"]
        features = target["features"]
        gate = target["gate"]
        aliases = target["aliases"]
        links = target["scope_links"]
        relevant = float(article["relevant_scope_count"])
        knowledge = float(article["knowledge_scope_count"])
        linked = float(features["linked_scope_count"])
        bridged = float(len(links["bridged_scope_ids"]))
        values = [
            math.log1p(float(paper["page_count"])),
            math.log1p(float(paper["target_count"])),
            math.log1p(relevant),
            math.log1p(knowledge),
            math.log1p(float(article["shared_knowledge_scope_count"])),
            math.log1p(float(article["collective_knowledge_scope_count"])),
            float(bool(article["has_shared_evidence"])),
            math.log1p(float(features["direct_scope_count"])),
            math.log1p(linked),
            math.log1p(float(features["table_scope_count"])),
            math.log1p(float(features["caption_scope_count"])),
            math.log1p(float(features["sample_id_count"])),
            math.log1p(float(features["component_count"])),
            math.log1p(float(features["direct_concept_count"])),
            math.log1p(float(features["bridged_concept_count"])),
            float(bool(gate["sample_bridge"])),
            float(bool(gate["component_bridge"])),
            float(bool(gate["collective_target_scope"])),
            float(bool(gate["eligible_for_article_propagation"])),
            math.log1p(float(len(aliases["identity_derived"]))),
            math.log1p(float(len(aliases["source_confirmed_identity"]))),
            math.log1p(float(len(aliases["source_derived"]))),
            math.log1p(float(len(links["sample_bridge_scope_ids"]))),
            math.log1p(float(len(links["component_bridge_scope_ids"]))),
            math.log1p(bridged),
            float(features["direct_scope_count"]) / max(1.0, relevant),
            linked / max(1.0, relevant),
            bridged / max(1.0, linked),
            float(article["shared_knowledge_scope_count"]) / max(1.0, knowledge),
        ]
        direct = set(map(str, target["direct_concepts"]))
        bridged_concepts = set(map(str, target["bridged_concepts"]))
        article_concepts = set(map(str, article["concepts"]))
        components = set(map(str, target["components"]))
        values.extend(float(value in direct) for value in concept_vocabulary)
        values.extend(float(value in bridged_concepts) for value in concept_vocabulary)
        values.extend(float(value in article_concepts) for value in concept_vocabulary)
        values.extend(float(value in components) for value in SEMANTIC_COMPONENTS)
        rows.append(values)
    if missing or identity_mismatch or len(rows) != len(records):
        raise ValueError(
            "component manifest does not align with full identity: "
            f"missing={missing[:3]}, identity_mismatch={identity_mismatch[:3]}"
        )
    all_matrix = np.asarray(rows, dtype=np.float32)
    bridge_matrix = np.asarray(all_matrix[:, bridge_indices], dtype=np.float32)
    return all_matrix, bridge_matrix, all_names, bridge_names, {
        "concept_vocabulary": concept_vocabulary,
        "semantic_components": list(SEMANTIC_COMPONENTS),
        "manifest_row_links": len(target_lookup),
    }


def validate_arrays(
    arrays: Mapping[str, np.ndarray], expected_rows: int
) -> dict[str, Any]:
    if set(arrays) != EXPECTED_ARRAY_NAMES:
        raise ValueError("component cache arrays differ from the frozen schema")
    if any(FORBIDDEN_NAME_RE.fullmatch(name) for name in arrays):
        raise ValueError("component cache contains a forbidden label or identifier array")
    audit: dict[str, Any] = {}
    for name in sorted(arrays):
        matrix = np.asarray(arrays[name])
        if matrix.ndim != 2 or matrix.shape[0] != expected_rows or matrix.shape[1] < 1:
            raise ValueError(f"unexpected shape for {name}: {matrix.shape}")
        if matrix.dtype != np.float32:
            raise ValueError(f"component matrix {name} is not float32")
        if not np.isfinite(matrix).all():
            raise ValueError(f"component matrix {name} contains non-finite values")
        audit[name] = {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "sha256": auxiliary.sha256_array(matrix),
            "finite": True,
        }
    return audit


def write_cache(path: Path, arrays: Mapping[str, np.ndarray], expected_rows: int) -> None:
    validate_arrays(arrays, expected_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **{
                name: np.asarray(arrays[name], dtype=np.float32)
                for name in sorted(arrays)
            },
        )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-identity", type=Path, default=auxiliary.EXACT_IDENTITY)
    parser.add_argument("--full-identity", type=Path, default=auxiliary.FULL_IDENTITY)
    parser.add_argument("--fold-manifest", type=Path, default=auxiliary.FOLD_MANIFEST)
    parser.add_argument("--pdf-dir", type=Path, default=base.DEFAULT_PDF_DIR)
    parser.add_argument("--numeric-cache", type=Path, default=numeric_builder.DEFAULT_CACHE)
    parser.add_argument("--numeric-manifest", type=Path, default=numeric_builder.DEFAULT_MANIFEST)
    parser.add_argument("--own-cache", type=Path, default=own_builder.DEFAULT_CACHE)
    parser.add_argument("--own-manifest", type=Path, default=own_builder.DEFAULT_MANIFEST)
    parser.add_argument("--component-manifest", type=Path, default=DEFAULT_COMPONENT_MANIFEST)
    parser.add_argument("--component-receipt", type=Path, default=DEFAULT_COMPONENT_RECEIPT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)

    exact_identity = args.exact_identity.resolve()
    full_identity = args.full_identity.resolve()
    fold_manifest = args.fold_manifest.resolve()
    pdf_dir = args.pdf_dir.resolve()
    numeric_cache_path = args.numeric_cache.resolve()
    numeric_manifest_path = args.numeric_manifest.resolve()
    own_cache_path = args.own_cache.resolve()
    own_manifest_path = args.own_manifest.resolve()
    component_manifest_path = args.component_manifest.resolve()
    component_receipt_path = args.component_receipt.resolve()
    cache_path = args.cache.resolve()
    manifest_path = args.manifest.resolve()

    full_records, full_coverage = base.load_pdf_only_identity(full_identity, pdf_dir)
    exact_records, exact_coverage = base.load_pdf_only_identity(exact_identity, pdf_dir)
    auxiliary.validate_membership(full_records, exact_records)
    auxiliary.load_fold_manifest(fold_manifest, exact_records)
    full_alignment = auxiliary.sha256_records(full_records)
    numeric_arrays, numeric_manifest = numeric_sweep.validate_cache(
        numeric_cache_path, numeric_manifest_path
    )
    if numeric_manifest.get("scope", {}).get("full_alignment_sha256") != full_alignment:
        raise ValueError("numeric cache alignment differs from auxiliary identity")
    own_arrays, _own_manifest = own_evaluation.validate_own_cache(
        own_cache_path, own_manifest_path, len(full_records), full_alignment
    )
    receipt, papers = validate_component_inputs(
        component_manifest_path, component_receipt_path, full_identity
    )
    component_all, component_bridge, all_names, bridge_names, feature_audit = (
        build_component_features(full_records, papers)
    )
    combined = numeric_arrays["combined"]
    own_target = own_arrays["own_target"]
    arrays = {
        "component_all": component_all,
        "component_bridge": component_bridge,
        "combined_plus_component_all": np.concatenate(
            (combined, component_all), axis=1
        ),
        "combined_plus_component_bridge": np.concatenate(
            (combined, component_bridge), axis=1
        ),
        "combined_plus_own_target_component_all": np.concatenate(
            (combined, own_target, component_all), axis=1
        ),
        "combined_plus_own_target_component_bridge": np.concatenate(
            (combined, own_target, component_bridge), axis=1
        ),
    }
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}
    matrix_audit = validate_arrays(arrays, len(full_records))
    write_cache(cache_path, arrays, len(full_records))
    manifest = {
        "schema_version": "r3_exact300_component_bridge_cache_v1",
        "status": "FROZEN_WITHOUT_LABEL_ACCESS",
        "scope": {
            "exact_evaluation": exact_coverage,
            "auxiliary_candidate_pool": full_coverage,
            "exact_alignment_sha256": auxiliary.sha256_records(exact_records),
            "full_alignment_sha256": full_alignment,
        },
        "runtime_boundary": {
            "labels_read": False,
            "targets_workbook_read": False,
            "pmagresult_or_rockmag_read": False,
            "prior_predictions_or_evaluations_read": False,
            "numeric_cache_contains_identifiers": False,
        },
        "allowed_sources": [
            "ROCKNAME and RESULTNO used only to align frozen identity rows",
            "frozen source-derived component manifest",
            "frozen label-free auxiliary numeric cache",
            "frozen label-free own-study cache",
        ],
        "inputs": {
            "component_manifest": {
                "path": str(component_manifest_path),
                "sha256": base.file_sha256(component_manifest_path),
            },
            "component_receipt": {
                "path": str(component_receipt_path),
                "sha256": base.file_sha256(component_receipt_path),
                "status": receipt["status"],
            },
            "numeric_cache": {
                "path": str(numeric_cache_path),
                "sha256": base.file_sha256(numeric_cache_path),
            },
            "own_cache": {
                "path": str(own_cache_path),
                "sha256": base.file_sha256(own_cache_path),
            },
            "fold_manifest": {
                "path": str(fold_manifest),
                "sha256": base.file_sha256(fold_manifest),
            },
        },
        "feature_names": {
            "component_all": all_names,
            "component_bridge": bridge_names,
        },
        "feature_audit": feature_audit,
        "matrices": matrix_audit,
        "output": {
            "path": str(cache_path),
            "sha256": base.file_sha256(cache_path),
            "contains_labels": False,
            "contains_identifiers": False,
        },
        "builder": {
            "path": str(Path(__file__).resolve()),
            "sha256": base.file_sha256(Path(__file__).resolve()),
        },
    }
    base.atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "cache": manifest["output"],
                "manifest": {
                    "path": str(manifest_path),
                    "sha256": base.file_sha256(manifest_path),
                },
                "feature_audit": feature_audit,
                "matrices": matrix_audit,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
