#!/usr/bin/env python3
"""Train and freeze the production model for the 84.19% R3 stable protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import tempfile
from typing import Any

import joblib
import numpy as np
import sklearn

import build_r3_query_adaptive_retrieval_features as retrieval
import build_r3_trace_page_graph_features as trace_freeze
import evaluate_r3_fulltext_fusion_single872_oof_v3 as fulltext_v3
import evaluate_r3_query_adaptive_single872_oof as query_eval
import evaluate_r3_residual_gate_single872_oof as residual
import evaluate_r3_style_fusion_single872_oof_v2 as style_v2
import experiment_r3_trace_clean_nested_oof as trace_eval
from r3_stable_production import fit_stable_artifact, predict_stable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/models/r3_stable_production_v1.joblib"
DEFAULT_MANIFEST = ROOT / "data/models/r3_stable_production_v1.manifest.json"
DEFAULT_OOF_REPORT = (
    ROOT
    / "tmp/r3_scope_fusion_all_oof_v1"
    / "r3_scope_fusion_all_grouped_oof_report_v1.json"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_joblib(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        joblib.dump(value, temporary, compress=3)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", type=Path, default=trace_freeze.DEFAULT_IDENTITY)
    parser.add_argument("--targets", type=Path, default=trace_eval.DEFAULT_TARGETS)
    parser.add_argument("--pdf-dir", type=Path, default=trace_freeze.DEFAULT_PDF_DIR)
    parser.add_argument("--page-cache", type=Path, default=residual.DEFAULT_PAGE_CACHE)
    parser.add_argument("--domain-features", type=Path, default=residual.DEFAULT_DOMAIN_FEATURES)
    parser.add_argument("--domain-manifest", type=Path, default=residual.DEFAULT_DOMAIN_MANIFEST)
    parser.add_argument("--v4-features", type=Path, default=residual.DEFAULT_V4_FEATURES)
    parser.add_argument("--v4-manifest", type=Path, default=residual.DEFAULT_V4_MANIFEST)
    parser.add_argument("--v4-context", type=Path, default=residual.DEFAULT_V4_CONTEXT)
    parser.add_argument("--v4-candidate", type=Path, default=residual.DEFAULT_V4_CANDIDATE)
    parser.add_argument("--v4-folds", type=Path, default=residual.DEFAULT_V4_FOLDS)
    parser.add_argument("--query-dir", type=Path, default=residual.QUERY_DIR)
    parser.add_argument("--style-features", type=Path, default=style_v2.DEFAULT_STYLE_FEATURES)
    parser.add_argument("--style-manifest", type=Path, default=style_v2.DEFAULT_STYLE_MANIFEST)
    parser.add_argument("--v1-config", type=Path, default=residual.DEFAULT_CONFIG)
    parser.add_argument("--v2-config", type=Path, default=style_v2.DEFAULT_CONFIG)
    parser.add_argument("--v3-config", type=Path, default=fulltext_v3.DEFAULT_CONFIG)
    parser.add_argument("--oof-report", type=Path, default=DEFAULT_OOF_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = {
        name: value.resolve()
        for name, value in vars(args).items()
        if isinstance(value, Path)
    }
    for name, path in paths.items():
        if name not in {"output", "manifest"} and not path.exists():
            raise SystemExit(f"Missing R3 stable training input {name}: {path}")

    v1_config = residual.load_json(paths["v1_config"], "V1 config")
    residual.validate_config(v1_config)
    v2_config = style_v2.load_json(paths["v2_config"], "V2 config")
    style_v2.validate_config(v2_config)
    v3_config = fulltext_v3.load_json(paths["v3_config"], "V3 config")
    fulltext_v3.validate_config(v3_config)

    (
        records,
        v4,
        v4_names,
        _contexts,
        fold_ids,
        _document_groups,
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
    label_map = trace_eval.load_labels(
        paths["targets"], {record["RESULTNO"] for record in records}
    )
    labels = np.asarray([label_map[record["RESULTNO"]] for record in records], dtype=np.int8)
    multiplicity: dict[str, int] = {}
    for record in trace_freeze.load_three_column_identity(paths["identity"]):
        multiplicity[record["REFNO"]] = multiplicity.get(record["REFNO"], 0) + 1
    single_mask = np.asarray(
        [multiplicity[record["REFNO"]] == 1 for record in records], dtype=bool
    )
    single_indices = np.flatnonzero(single_mask)
    if (len(records), len(single_indices)) != (2751, 872):
        raise SystemExit("Frozen R3 stable training scope changed")

    domain, _domain_manifest = residual.load_domain_features(
        paths["domain_features"], paths["domain_manifest"], records
    )
    compact, pooled, query_single, query_manifest, _query_candidate = (
        query_eval.load_query_features(
            query_dir=paths["query_dir"],
            records=records,
            v4_fold_ids=fold_ids,
            expected_single_mask=single_mask,
        )
    )
    if not np.array_equal(query_single, single_mask):
        raise SystemExit("R3 query feature scope changed")
    style = style_v2.load_style_features(
        paths["style_features"],
        paths["style_manifest"],
        records,
        fold_ids,
        single_indices,
    )
    front_documents = residual.front_documents(
        records,
        paths["page_cache"],
        int(v1_config["front_text_expert"]["characters"]),
    )
    single_records = [records[int(index)] for index in single_indices]
    full_documents = fulltext_v3.full_documents(single_records, paths["page_cache"])

    artifact = fit_stable_artifact(
        records=records,
        labels=labels,
        domain=domain,
        v4=v4,
        compact=compact,
        pooled=pooled,
        front_documents=front_documents,
        single_indices=single_indices,
        knowledge=style.knowledge,
        full_documents=full_documents,
        v1_config=v1_config,
        v2_config=v2_config,
        v3_config=v3_config,
    )
    artifact["v4_feature_names"] = list(v4_names)
    artifact["query_feature_names"] = list(query_manifest.get("feature_names") or [])
    artifact["knowledge_feature_names"] = list(
        np.load(paths["style_features"], allow_pickle=False)["knowledge_feature_names"].astype(str)
    )
    distractor_path = paths["query_dir"] / retrieval.DISTRACTOR_NAME
    artifact["query_distractor"] = retrieval.load_json(
        distractor_path, "query distractor statistics"
    )

    fitted = predict_stable(
        artifact=artifact,
        domain=domain,
        v4=v4,
        compact=compact,
        pooled=pooled,
        front_documents=front_documents,
        single_mask=single_mask,
        knowledge_single=style.knowledge,
        full_documents_single=full_documents,
    )
    resubstitution_accuracy = float(((fitted.probability >= 0.5) == labels).mean())
    atomic_joblib(artifact, paths["output"])

    oof_report = json.loads(paths["oof_report"].read_text(encoding="utf-8"))
    stable = (oof_report.get("strategies") or {}).get("scope_v3_plus_no_strict") or {}
    oof_accuracy = float((stable.get("overall") or {}).get("accuracy", -1))
    if not np.isclose(oof_accuracy, 2316 / 2751):
        raise SystemExit("Frozen 84.19% OOF report changed")
    model_snapshot = retrieval.inspect_model_snapshot(retrieval.default_model_snapshot())
    manifest = {
        "schema_version": "r3_stable_production_model_manifest_v1",
        "status": "TRAINED_ON_ALL_2751_PDF_BACKED_LABELLED_ROWS",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {
            "path": str(paths["output"]),
            "sha256": sha256(paths["output"]),
            "schema_version": artifact["schema_version"],
        },
        "training": {
            **artifact["training"],
            "grouped_oof_accuracy": oof_accuracy,
            "grouped_oof_correct": 2316,
            "grouped_oof_rows": 2751,
            "resubstitution_accuracy_not_for_reporting": resubstitution_accuracy,
            "labels_source": {"path": str(paths["targets"]), "sha256": sha256(paths["targets"])},
            "identity": {"path": str(paths["identity"]), "sha256": sha256(paths["identity"])},
            "oof_report": {"path": str(paths["oof_report"]), "sha256": sha256(paths["oof_report"])},
        },
        "runtime_boundary": {
            "runtime_inputs": [
                "REFNO/RESULTNO/ROCKNAME identity",
                "original-paper page text",
                "frozen knowledge and feature specifications",
                "this historically supervised model",
            ],
            "runtime_reads_r3_labels": False,
            "runtime_reads_pmagresult_or_rockmag": False,
            "historical_r3_labels_encoded_in_model": True,
            "paper_only_evidence_retained_for_audit_and_fallback": True,
        },
        "bge_model_snapshot": model_snapshot,
        "versions": {
            "python": os.sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
        },
        "warning": (
            "84.19% is grouped OOF on the development corpus, not external validation. "
            "The final model is fit on all labelled rows and its resubstitution score must not be reported."
        ),
    }
    paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
