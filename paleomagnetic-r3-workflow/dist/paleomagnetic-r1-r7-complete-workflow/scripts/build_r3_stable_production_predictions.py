#!/usr/bin/env python3
"""Score R3 with the supervised stable model and retain paper-only evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np

import build_r3_paper_only_predictions as paper_only
import build_r3_query_adaptive_retrieval_features as retrieval
import build_r3_style_features_single872_v2 as style_freeze
import build_r3_trace_page_graph_features as trace_freeze
import experiment_r3_pdf_only_strategies as domain_base
import experiment_r3_small_screen_curation_style as curation
import r3_trace_page_graph_features as trace_features
from r3_stable_production import predict_stable, validate_artifact


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "data/models/r3_stable_production_v1.joblib"
DEFAULT_MANIFEST = ROOT / "data/models/r3_stable_production_v1.manifest.json"
DEFAULT_MODEL_SNAPSHOT = (
    ROOT
    / "data/models/bge-base-en-v1.5"
    / retrieval.MODEL_REVISION
)
DEFAULT_QUERY_KB = ROOT / "data/knowledge/r3_query_adaptive_retrieval_v1.json"
DEFAULT_TRACE_CONCEPTS = ROOT / "data/models/r3_stable_trace_concepts_v4.json"
DEFAULT_TAXONOMY = ROOT / "data/calibration/r3_rockmag_text_distillation_v1.json"
DEFAULT_POSITIVE_KB = ROOT / "data/calibration/r3_positive_rockmag_expert_kb_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    return rows


def nonempty_cache_row(value: Mapping[str, Any] | None) -> bool:
    return bool(
        value
        and any(
            isinstance(page, Mapping) and str(page.get("text") or "").strip()
            for page in value.get("pages") or []
        )
    )


def eligible_record_indices(
    records: Sequence[Mapping[str, str]],
    cache_rows: Mapping[str, Mapping[str, Any]],
    pdf_refnos: set[str],
) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, record in enumerate(records)
            if record["REFNO"] in pdf_refnos
            and nonempty_cache_row(cache_rows.get(record["REFNO"]))
        ],
        dtype=np.int64,
    )


def combine_with_paper_only(
    *,
    records: Sequence[Mapping[str, str]],
    paper_rows: Sequence[Mapping[str, Any]],
    probability_by_result: Mapping[str, float],
    route_by_result: Mapping[str, str],
    component_by_result: Mapping[str, Mapping[str, float]],
    receipt_by_result: Mapping[str, Mapping[str, Any]],
    model_sha256: str,
    grouped_oof_accuracy: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_result = {str(row.get("RESULTNO") or ""): row for row in paper_rows}
    if set(by_result) != {str(record["RESULTNO"]) for record in records}:
        raise ValueError("paper-only evidence rows do not align with R3 identity")
    output_rows = []
    disagreements = 0
    fallback_rows = 0
    route_counts: Counter[str] = Counter()
    for record in records:
        resultno = str(record["RESULTNO"])
        base_row = by_result[resultno]
        payload = evidence_payload(base_row)
        paper_score = int(base_row.get("R3") or 0)
        paper_outcome = str(payload.get("outcome") or "UNKNOWN")
        paper_reasoning = str(payload.get("final_decision_reasoning") or "")
        if resultno in probability_by_result:
            probability = float(probability_by_result[resultno])
            score = int(probability >= 0.5)
            route = str(route_by_result[resultno])
            route_counts[route] += 1
            disagreements += int(score != paper_score)
            payload.update(
                {
                    "R3": score,
                    "outcome": "POSITIVE" if score else "NEGATIVE",
                    "final_decision_reasoning": (
                        f"R3={score} from the frozen historically supervised stable model; "
                        f"route={route}; probability={probability:.6f}. Paper-only evidence is "
                        "retained below for scientific audit."
                    ),
                    "supervised_stable_model": {
                        "status": "SCORED",
                        "probability": probability,
                        "threshold": 0.5,
                        "route": route,
                        "components": dict(component_by_result[resultno]),
                        "query_receipt": dict(receipt_by_result.get(resultno, {})),
                        "model_sha256": model_sha256,
                        "grouped_oof_accuracy": grouped_oof_accuracy,
                        "historical_r3_labels_used_in_training": True,
                        "runtime_r3_labels_read": False,
                    },
                    "paper_only_audit": {
                        "score": paper_score,
                        "outcome": paper_outcome,
                        "final_decision_reasoning": paper_reasoning,
                    },
                }
            )
        else:
            fallback_rows += 1
            score = paper_score
            payload["supervised_stable_model"] = {
                "status": "FALLBACK_TO_PAPER_ONLY",
                "reason": "No local PDF or nonempty page text was available for model features.",
                "historical_r3_labels_used_in_training": True,
                "runtime_r3_labels_read": False,
            }
        output_rows.append(
            {
                "REFNO": str(record["REFNO"]),
                "RESULTNO": resultno,
                "TARGET": str(record["ROCKNAME"]),
                "R3": score,
                "R3_RESULT_JSON": payload,
            }
        )
    return output_rows, {
        "paper_only_fallback_rows": fallback_rows,
        "paper_only_disagreements": disagreements,
        "route_counts": dict(sorted(route_counts.items())),
    }


def build_domain_features(
    records: Sequence[Mapping[str, str]],
    page_cache: Path,
    pdf_dir: Path,
    taxonomy: Path,
    positive_kb: Path,
) -> tuple[np.ndarray, dict[str, str], dict[str, list[dict[str, Any]]], Sequence[Mapping[str, Any]]]:
    refnos = {str(record["REFNO"]) for record in records}
    papers = domain_base.load_page_text(page_cache, refnos)
    pages = curation.load_pages(page_cache, refnos)
    concepts, _audit = domain_base.load_knowledge_patterns(taxonomy, positive_kb)
    base = domain_base.build_features(records, papers, concepts)
    style = curation.build_features(records, pages, concepts, pdf_dir)
    matrix = np.concatenate(
        (
            np.asarray(base["engineered_rows"], dtype=np.float32),
            np.asarray(style["all"], dtype=np.float32),
        ),
        axis=1,
    ).astype(np.float32, copy=False)
    return matrix, papers, pages, concepts


def build_trace_features(
    records: Sequence[Mapping[str, str]],
    cache_rows: Mapping[str, Mapping[str, Any]],
    concept_path: Path,
) -> np.ndarray:
    concepts = trace_features.load_trace_concepts(concept_path)
    result = trace_features.build_trace_page_graph_features(records, cache_rows, concepts)
    return np.asarray(result["matrix"], dtype=np.float32)


def build_query_features(
    *,
    records: Sequence[Mapping[str, str]],
    cache_rows: Mapping[str, Mapping[str, Any]],
    single_mask: np.ndarray,
    query_kb_path: Path,
    concept_path: Path,
    model_snapshot: Path,
    device: str,
    distractor: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, Any]]]:
    query_kb = retrieval.load_query_kb(query_kb_path)
    concepts = retrieval.load_concept_snapshot(concept_path)
    inventory, audits, _inventory_audit = retrieval.build_chunk_inventory(
        records, cache_rows, concepts, query_kb["chunking"]
    )
    encoder = retrieval.BGEEncoder(model_snapshot, query_kb["encoder"], device)
    compact, names, pooled, receipts = retrieval.build_features(
        records=records,
        inventory=inventory,
        chunk_audits=audits,
        query_kb=query_kb,
        distractor=distractor,
        encoder=encoder,
        target_enabled_mask=single_mask,
    )
    return compact, pooled, names, receipts


def evidence_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    value = row.get("R3_RESULT_JSON")
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str) and value.strip():
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity-manifest", type=Path, required=True)
    parser.add_argument("--page-cache", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, required=True)
    parser.add_argument("--paper-only-jsonl", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-snapshot", type=Path, default=DEFAULT_MODEL_SNAPSHOT)
    parser.add_argument("--query-kb", type=Path, default=DEFAULT_QUERY_KB)
    parser.add_argument("--trace-concepts", type=Path, default=DEFAULT_TRACE_CONCEPTS)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--positive-kb", type=Path, default=DEFAULT_POSITIVE_KB)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    required = {
        name: value
        for name, value in vars(args).items()
        if isinstance(value, Path) and name not in {"output", "summary_json"}
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise SystemExit("Missing R3 stable runtime input(s): " + "; ".join(missing))

    manifest = load_json(args.model_manifest)
    if sha256(args.model) != str((manifest.get("model") or {}).get("sha256") or ""):
        raise SystemExit("R3 stable model hash does not match its manifest")
    snapshot = retrieval.inspect_model_snapshot(args.model_snapshot)
    expected_snapshot = str(
        (manifest.get("bge_model_snapshot") or {}).get("snapshot_sha256") or ""
    )
    if snapshot["snapshot_sha256"] != expected_snapshot:
        raise SystemExit("R3 stable BGE snapshot hash changed")
    artifact = joblib.load(args.model)
    validate_artifact(artifact)

    identity = paper_only.load_targets(args.identity_manifest)
    records_all = identity.to_dict(orient="records")
    cache_all = paper_only.load_page_cache(args.page_cache)
    pdf_refnos = {
        path.stem
        for path in args.pdf_dir.glob("*.pdf")
        if path.is_file()
    }
    eligible_indices = eligible_record_indices(records_all, cache_all, pdf_refnos)
    records = [records_all[int(index)] for index in eligible_indices]
    multiplicity = Counter(record["REFNO"] for record in records_all)
    single_mask = np.asarray(
        [multiplicity[record["REFNO"]] == 1 for record in records], dtype=bool
    )
    cache_rows = {record["REFNO"]: cache_all[record["REFNO"]] for record in records}

    probability_by_result: dict[str, float] = {}
    route_by_result: dict[str, str] = {}
    component_by_result: dict[str, dict[str, float]] = {}
    receipt_by_result: dict[str, dict[str, Any]] = {}
    if records:
        domain, papers, _pages, concepts = build_domain_features(
            records,
            args.page_cache,
            args.pdf_dir,
            args.taxonomy,
            args.positive_kb,
        )
        v4 = build_trace_features(records, cache_rows, args.trace_concepts)
        compact, pooled, query_names, receipts = build_query_features(
            records=records,
            cache_rows=cache_rows,
            single_mask=single_mask,
            query_kb_path=args.query_kb,
            concept_path=args.trace_concepts,
            model_snapshot=args.model_snapshot,
            device=args.device,
            distractor=artifact["query_distractor"],
        )
        expected_names = list(artifact.get("query_feature_names") or [])
        if expected_names and query_names != expected_names:
            raise SystemExit("R3 stable query feature names changed")
        expected_v4 = int(artifact["dimensions"]["v4"])
        if v4.shape[1] != expected_v4:
            raise SystemExit(f"R3 stable V4 feature count changed: {v4.shape[1]}")
        front_characters = 16000
        front_documents = [
            f"{record['ROCKNAME']}\n{papers[record['REFNO']][:front_characters]}"
            for record in records
        ]
        single_records = [record for record, is_single in zip(records, single_mask) if is_single]
        pages_by_refno, page_rows_present = style_freeze.load_style_pages(
            args.page_cache, {record["REFNO"] for record in single_records}
        )
        style = (
            style_freeze.build_matrices(
                single_records,
                pages_by_refno,
                page_rows_present,
                concepts,
                args.pdf_dir,
            )
            if single_records
            else {"knowledge": np.empty((0, int(artifact["dimensions"]["knowledge"]))) }
        )
        full_documents = [
            f"{record['ROCKNAME']}\n{papers[record['REFNO']]}" for record in single_records
        ]
        predicted = predict_stable(
            artifact=artifact,
            domain=domain,
            v4=v4,
            compact=compact,
            pooled=pooled,
            front_documents=front_documents,
            single_mask=single_mask,
            knowledge_single=np.asarray(style["knowledge"], dtype=np.float64),
            full_documents_single=full_documents,
        )
        single_position = {global_index: local for local, global_index in enumerate(np.flatnonzero(single_mask))}
        for index, record in enumerate(records):
            resultno = str(record["RESULTNO"])
            probability_by_result[resultno] = float(predicted.probability[index])
            route_by_result[resultno] = str(predicted.route[index])
            components = {
                name: float(value[index])
                for name, value in predicted.components.items()
                if value.shape == (len(records),)
            }
            if index in single_position:
                local = single_position[index]
                components["knowledge_mean"] = float(
                    predicted.components["single_knowledge_mean"][local]
                )
                components["fulltext_mean"] = float(
                    predicted.components["single_fulltext_mean"][local]
                )
            component_by_result[resultno] = components
        receipt_by_result = {
            str(item["RESULTNO"]): item for item in receipts
        }

    model_hash = sha256(args.model)
    grouped_oof_accuracy = float(
        (manifest.get("training") or {}).get("grouped_oof_accuracy")
    )
    try:
        output_rows, merge_summary = combine_with_paper_only(
            records=records_all,
            paper_rows=load_jsonl(args.paper_only_jsonl),
            probability_by_result=probability_by_result,
            route_by_result=route_by_result,
            component_by_result=component_by_result,
            receipt_by_result=receipt_by_result,
            model_sha256=model_hash,
            grouped_oof_accuracy=grouped_oof_accuracy,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema_version": "r3_stable_production_predictions_v1",
        "rows": len(output_rows),
        "model_scored_rows": len(probability_by_result),
        **merge_summary,
        "score_counts": dict(sorted(Counter(row["R3"] for row in output_rows).items())),
        "model": str(args.model),
        "model_sha256": model_hash,
        "model_grouped_oof_accuracy": grouped_oof_accuracy,
        "boundary": {
            "runtime_r3_labels_read": False,
            "runtime_pmagresult_or_rockmag_read": False,
            "historical_r3_labels_encoded_in_model": True,
            "paper_only_evidence_retained": True,
        },
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
