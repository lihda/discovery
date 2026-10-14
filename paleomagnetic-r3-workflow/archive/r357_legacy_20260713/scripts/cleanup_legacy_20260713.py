#!/usr/bin/env python3
"""Archive obsolete workflow iterations and remove regenerable caches."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path(
    "/Users/lihd/discovery/paleomagnetic-r3-workflow/archive/"
    "r357_legacy_20260713"
)

KEEP_SCRIPTS = {
    "analyze_r3_single_target_selective_policy.py",
    "analyze_r3_target_multiplicity_oof.py",
    "arbitrate_r3_structured_extracts.py",
    "audit_r357_scoring_boundary.py",
    "audit_r3_drift_against_gold.py",
    "build_r3_all_labelled_fused_page_cache.py",
    "build_r3_distilled_predictions.py",
    "build_r3_evidence_packs.py",
    "build_r3_exact300_auxiliary_numeric_cache.py",
    "build_r3_exact300_component_bridge_cache.py",
    "build_r3_exact300_own_study_cache.py",
    "build_r3_mineru_page_cache.py",
    "build_r3_page_source_cache.py",
    "build_r3_paper_only_predictions.py",
    "build_r3_structured_extracts_from_packs.py",
    "build_r5_r7_source_recovered_predictions.py",
    "build_strict_identity_manifest.py",
    "calculate_r1246_table_indicators.py",
    "create_indicator_final_workbook.py",
    "create_indicator_results_page.py",
    "create_r1234567_total_review_html.py",
    "create_r3_final_workbook.py",
    "create_r3_results_page.py",
    "criterion_spec.py",
    "evaluate_r357_strict_predictions.py",
    "experiment_r3_all_labelled_nested_threshold.py",
    "experiment_r3_all_labelled_pdf_grouped_oof.py",
    "experiment_r3_exact300_auxiliary_model_sweep.py",
    "experiment_r3_exact300_auxiliary_training.py",
    "experiment_r3_exact300_component_bridge_auxiliary.py",
    "experiment_r3_exact300_nested_router.py",
    "experiment_r3_exact300_own_study_auxiliary.py",
    "experiment_r3_exact300_paper_consistency.py",
    "experiment_r3_own_study_events.py",
    "experiment_r3_pdf_only_strategies.py",
    "experiment_r3_single_target_practice_semantics.py",
    "experiment_r3_small_screen_curation_style.py",
    "expert_conflict_engine.py",
    "export_r3_structured_final_outputs.py",
    "filter_identity_manifest_by_refno_split.py",
    "filter_page_cache_for_identity.py",
    "indicator_workflow.py",
    "induce_r3_rockmag_lexicon.py",
    "paper_text_calibration.py",
    "paper_text_retrieval.py",
    "r3_calibrated_practice_criterion.py",
    "r3_carrier_claim_status.py",
    "r3_practice_criterion.py",
    "r3_rockmag_distilled_reasoner.py",
    "r5_r7_expert_rules.py",
    "r7_original_text_extractor.py",
    "r7_pole_overlap_reasoner.py",
    "run_r357_latest_full_workflow.py",
    "run_r357_strict_knowledge_workflow.py",
    "strict_scoring_worker.py",
    "train_r3_positive_rockmag_expert_kb_v2.py",
    "train_r3_rockmag_text_distillation.py",
    "train_r5_paper_text_calibration.py",
    "validate_r3_structured_extract_jsonl.py",
}

KEEP_TESTS = {
    "test_build_r3_distilled_predictions.py",
    "test_criterion_spec.py",
    "test_expert_conflict_engine.py",
    "test_identity_refno_split.py",
    "test_page_cache_filter.py",
    "test_page_source_cache.py",
    "test_paper_text_calibration.py",
    "test_paper_text_retrieval.py",
    "test_r3_calibrated_practice_criterion.py",
    "test_r3_carrier_claim_status.py",
    "test_r3_exact300_auxiliary_model_sweep.py",
    "test_r3_exact300_auxiliary_numeric_cache.py",
    "test_r3_exact300_auxiliary_training.py",
    "test_r3_exact300_nested_router.py",
    "test_r3_exact300_paper_consistency.py",
    "test_r3_mineru_page_cache.py",
    "test_r3_own_study_events.py",
    "test_r3_paper_only_predictions.py",
    "test_r3_pdf_only_strategies.py",
    "test_r3_positive_rockmag_expert_kb_v2.py",
    "test_r3_practice_criterion.py",
    "test_r3_rockmag_distilled_reasoner.py",
    "test_r3_rockmag_lexicon.py",
    "test_r3_rockmag_text_distillation.py",
    "test_r3_small_screen_curation_style.py",
    "test_r5_paper_only_predictions.py",
    "test_r7_original_text_extractor.py",
    "test_r7_pole_overlap_reasoner.py",
    "test_scoring_boundary.py",
    "test_strict_knowledge_payloads.py",
    "test_strict_knowledge_workflow.py",
    "test_total_review_html.py",
}

KEEP_DOCS = {
    "r357_strict_knowledge_conflict_patent_note.md",
    "r3_all_labelled_pdf_grouped_oof_20260713.md",
}

KEEP_OUTPUTS = {
    "r357_latest_full_07111019",
    "r357_paper_only_eval_07111638",
    "r357_strict_eval_07111758",
    "r357_strict_v21_07111943",
    "r1234567_07111019.html",
    "r1234567_07111638.html",
    "r357_ablation_report_0711.json",
    "r3_07111019_unlabeled.html",
    "r3_07111019_unlabeled.xlsx",
    "r3_07111638_6454.html",
    "r3_07111638_6454.xlsx",
    "r5_07111019_unlabeled.html",
    "r5_07111019_unlabeled.xlsx",
    "r5_07111638_8824.html",
    "r5_07111638_8824.xlsx",
    "r7_07111019_unlabeled.html",
    "r7_07111019_unlabeled.xlsx",
    "r7_07111638_8966.html",
    "r7_07111638_8966.xlsx",
}

KEEP_TMP = {
    "mineru_cache_all_hard_risk.jsonl",
    "mineru_cache_all_hard_risk_summary.json",
    "r3_all_labelled_pdf_grouped_oof_v1",
    "r3_all_labelled_pdf_page_cache_v1",
    "r3_proxy_v2",
}

KEEP_PROXY = {
    "full_cache_holdout_mineru.jsonl",
    "full_cache_holdout_mineru_summary.json",
}

DELETE_TMP = {
    "hf_cache",
    "mineru_3244_page1.png",
    "mineru_direct_3244",
    "mineru_runs",
    "mineru_venv",
    "models",
    "pdfs",
    "r3_embedding_venv",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def files_under(path: Path) -> Iterable[Path]:
    if path.is_file() or path.is_symlink():
        yield path
        return
    yield from sorted(item for item in path.rglob("*") if item.is_file())


def tree_stats(path: Path) -> tuple[int, int]:
    files = list(files_under(path))
    return len(files), sum(item.stat().st_size for item in files)


def remove_path(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def main() -> int:
    if ARCHIVE.exists():
        raise SystemExit(f"archive already exists: {ARCHIVE}")
    ARCHIVE.mkdir(parents=True)

    deleted: list[dict[str, Any]] = []
    explicit_delete = [ROOT / "tmp" / name for name in sorted(DELETE_TMP)]
    generated = sorted(ROOT.rglob(".DS_Store"))
    generated.extend(sorted(ROOT.rglob("__pycache__")))
    seen_delete: set[Path] = set()
    for path in explicit_delete + generated:
        if not path.exists() or path in seen_delete:
            continue
        if any(parent in seen_delete for parent in path.parents):
            continue
        count, size = tree_stats(path)
        deleted.append(
            {
                "path": relative(path),
                "files": count,
                "bytes": size,
                "reason": "regenerable cache, environment, model, or bytecode",
            }
        )
        remove_path(path)
        seen_delete.add(path)

    archive_paths: list[Path] = []
    archive_paths.extend(
        path
        for path in sorted((ROOT / "scripts").glob("*.py"))
        if path.name not in KEEP_SCRIPTS
    )
    archive_paths.extend(
        path
        for path in sorted((ROOT / "tests").glob("test_*.py"))
        if path.name not in KEEP_TESTS
    )
    archive_paths.extend(
        path
        for path in sorted((ROOT / "docs").glob("*"))
        if path.is_file() and path.name not in KEEP_DOCS
    )
    if (ROOT / "research").exists():
        archive_paths.append(ROOT / "research")
    archive_paths.extend(
        path
        for path in sorted((ROOT / "outputs").iterdir())
        if path.name not in KEEP_OUTPUTS
    )
    archive_paths.extend(
        path
        for path in sorted((ROOT / "tmp").iterdir())
        if path.name not in KEEP_TMP
    )
    proxy = ROOT / "tmp/r3_proxy_v2"
    archive_paths.extend(
        path
        for path in sorted(proxy.iterdir())
        if path.name not in KEEP_PROXY
    )

    archived_files: list[dict[str, Any]] = []
    for path in archive_paths:
        for item in files_under(path):
            archived_files.append(
                {
                    "path": relative(item),
                    "bytes": item.stat().st_size,
                    "sha256": file_sha256(item),
                }
            )

    for path in archive_paths:
        destination = ARCHIVE / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))

    manifest = {
        "schema_version": "r357_cleanup_archive_manifest_v1",
        "project_root": str(ROOT),
        "archive_root": str(ARCHIVE),
        "policy": {
            "archived": "obsolete source, tests, docs, outputs, research, and experiment artifacts",
            "deleted": "regenerable environments, models, caches, bytecode, and Finder metadata",
        },
        "archived": {
            "top_level_items": len(archive_paths),
            "files": len(archived_files),
            "bytes": sum(row["bytes"] for row in archived_files),
            "entries": archived_files,
        },
        "deleted": {
            "items": len(deleted),
            "files": sum(row["files"] for row in deleted),
            "bytes": sum(row["bytes"] for row in deleted),
            "entries": deleted,
        },
        "retained_counts": {
            "scripts": len(KEEP_SCRIPTS),
            "tests": len(KEEP_TESTS),
            "docs": len(KEEP_DOCS),
            "output_items": len(KEEP_OUTPUTS),
        },
    }
    manifest_path = ARCHIVE / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "archive": str(ARCHIVE),
                "manifest": str(manifest_path),
                "archived_files": manifest["archived"]["files"],
                "archived_bytes": manifest["archived"]["bytes"],
                "deleted_files": manifest["deleted"]["files"],
                "deleted_bytes": manifest["deleted"]["bytes"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
