---
name: paleomagnetic-r357-calibrated-workflow
description: Run and audit the retained paleomagnetic R3/R5/R7 original-paper workflows, strict expert-knowledge scoring, and current grouped R3 OOF evaluation.
---

# Paleomagnetic R1-R7 Workflow

Work from the project root. Treat `README.md` as the canonical command and
result index, and `docs/project_integrity_20260717.md` as the retained-file and
verification record. Do not restore archived experiments into the active tree
unless a user explicitly asks to revisit that method.

## Runtime Boundaries

R3 prediction-time evidence may come from original papers, identity fields,
frozen knowledge and the frozen historically supervised production model. Do
not use `PMAGRESULT`, `ROCKUNIT`, current R3 labels, prior predictions or
evaluation reports as online R3 scoring inputs. R1/R2/R4/R6 still use their
declared source-table fields.

The default public R3 mode is `supervised_stable`. It first builds paper-only
evidence, then routes single-target records through V3 and multi-target records
through `multi_no_strict`. Missing PDF/text features must fall back to the
paper-only result and remain explicit in the audit payload. Keep `paper_only`
available as a compatibility mode.

The strict scorer accepts only:

- an identity manifest containing `REFNO`, `RESULTNO`, and `ROCKNAME`;
- a complete original-paper page cache;
- `data/knowledge/r357_expert_kb_v1.json`;
- `data/knowledge/r357_criterion_spec_v2.json`.

The production model was fitted on historical R3 labels after model selection.
Its reportable internal estimate remains grouped OOF 2,316/2,751 = 84.19%.
Never report the final model's 92.33% training-set resubstitution as accuracy,
or describe supervised production accuracy as a label-free strict result.

## Environments

Use `requirements.txt` for the business and strict workflows. The frozen R3
OOF runtime is Python 3.9.6, NumPy 2.0.2 and scikit-learn 1.6.1. Use
`requirements-r3-retrieval-model.txt` when rebuilding query-adaptive retrieval
features. `requirements-r3-text-model.txt` is not the current V3 environment.

## Main Workflows

Validate the current complete R1-R7 inputs without scoring:

```bash
.venv/bin/python3 run_r1_r7_workflow.py --check-only
```

Run the current complete business workflow:

```bash
./run_r1_r7_workflow.sh
```

The public workflow configuration is `config/workflow.current.json`; its model,
manifest and bundled BGE snapshot hashes must pass preflight. The
retained `scripts/run_r357_latest_full_workflow.py` is an internal compatibility
orchestrator, not the primary handoff entrypoint. A successful public run must
produce consolidated R1-R7 XLSX, JSONL, HTML, artifact manifest, and run
manifest files under one timestamped run directory.

Run physically isolated strict scoring:

```bash
./run_strict_knowledge_workflow.sh \
  outputs/r357_strict_eval_07111758/inputs/identity_manifest.xlsx \
  outputs/r357_strict_eval_07111758/inputs/page_cache.jsonl \
  outputs/r357_strict_knowledge_<timestamp>
```

Evaluate strict predictions only after their hashes are frozen, using
`scripts/evaluate_r357_strict_predictions.py` in a separate process. Require
both `strict_run_manifest.json` and `scoring_boundary_audit.json` to pass.

## Current R3 Evaluation

The retained current results are:

- single-target V3: 785/872 = 90.02%, full coverage;
- stable all-target scope/V3 combination: 2,316/2,751 = 84.19%;
- exploratory 50/50 GREP blend: 2,325/2,751 = 84.51%, not promoted because
  two outer folds regress and the REFNO bootstrap interval includes zero;
- prior R2E/GREP pairwise stage: 2,300/2,751 = 83.61%, retained as an input
  component rather than the current best result.

Recompute the current evaluators with:

```bash
.venv/bin/python3 scripts/evaluate_r3_fulltext_fusion_single872_oof_v3.py
.venv/bin/python3 scripts/evaluate_r3_scope_fusion_all_oof_v1.py
.venv/bin/python3 scripts/experiment_r3_single872_base_weight_nested_oof_v1.py
```

The weight experiment confirms the fixed three-component base mixture. A
global full-data grid reaches 786/872 only after reading all development
labels; strict nested selection falls to 777/872. Do not promote either over
the frozen 785/872 V3 result.

All current R3 folds are grouped by `REFNO`. Targets without local PDFs are
skipped and excluded from the denominator. OOF results remain internal
development estimates, not independent external validation.

## Rebuild Order

When a full feature rebuild is required, preserve this dependency order:

1. `scripts/build_r3_all_labelled_fused_page_cache.py`
2. `scripts/experiment_r3_all_labelled_pdf_grouped_oof.py`
3. V4 trace features and grouped OOF via
   `scripts/build_r3_trace_page_graph_features.py` and
   `scripts/experiment_r3_trace_clean_nested_oof.py`
4. query-adaptive features via
   `scripts/build_r3_query_adaptive_retrieval_features.py`
5. residual, style and full-text V3 evaluators
6. paleomagnetic scope/GREP evaluator and all-target scope fusion

Do not bypass frozen input hashes. The page-cache audit deliberately retains
the original seven-cache declaration; only the first two sources contribute
records. Four lower-priority files are hard-linked to the external legacy
archive so the original audit remains exactly reproducible without duplicate
disk storage.

## Expert Knowledge

Validate normative criterion and KB provenance through
`scripts/criterion_spec.py` and `scripts/expert_conflict_engine.py`. Fail closed
on a version/hash mismatch or identifier-bearing knowledge artifact.

Offline knowledge builders may read source metadata only to produce aggregate,
identifier-free artifacts. Retain no raw answer map, source row identifier,
held-out answer or quotation in a deployable knowledge package.

## R7 and MinerU

`scripts/r7_original_text_extractor.py` extracts target-bound pole, A95, age,
craton and comparison evidence. `scripts/r7_pole_overlap_reasoner.py` performs
the quantitative overlap reasoning. Keep unresolved evidence unresolved.

MinerU remains an optional label-free parser fallback. Recreate its isolated
environment from `requirements-mineru.txt` and invoke
`scripts/build_r3_mineru_page_cache.py` only with identity, local PDFs and a
declared base cache. Its quality gate must not read labels or evaluations.

## Verification

After any workflow or cleanup change, run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover -s tests -v
.venv/bin/python3 -m pip check
bash -n run_r1_r7_workflow.command run_r1_r7_workflow.sh run_latest_workflow.command run_latest_workflow.sh run_strict_knowledge_workflow.sh
.venv/bin/python3 run_r1_r7_workflow.py --help
.venv/bin/python3 run_r1_r7_workflow.py --check-only
.venv/bin/python3 scripts/run_r357_latest_full_workflow.py --help
.venv/bin/python3 scripts/create_r1_r7_deliverables.py --help
.venv/bin/python3 scripts/run_r357_strict_knowledge_workflow.py --help
.venv/bin/python3 scripts/evaluate_r3_fulltext_fusion_single872_oof_v3.py --help
.venv/bin/python3 scripts/evaluate_r3_scope_fusion_all_oof_v1.py --help
```

Treat 90% as achieved only for the 872-paper single-target internal OOF scope.
Do not turn selective coverage, a tuned development grid or the 84.51%
exploratory blend into a stronger claim than its protocol supports.
