# Latent-Guided R5/R7 Research Sandbox

This directory is an isolated research sandbox for exploring latent-guided
improvements to the paleomagnetic R5/R7 indicators. It does not modify the
production R3/R5/R7 workflow.

## Current Entry Point

```bash
.venv/bin/python3 research/latent_guided_r57/scripts/run_r57_latent_exploration.py \
  --embedding-provider hash
```

Optional Qwen/OpenAI-compatible embeddings:

```bash
EMBEDDING_URL="<endpoint>" \
EMBEDDING_TOKEN="<token>" \
.venv/bin/python3 research/latent_guided_r57/scripts/run_r57_latent_exploration.py \
  --embedding-provider auto
```

The script falls back to local hash embeddings when the remote endpoint is not
available. Do not commit tokens or endpoint secrets into this directory.

## What The Script Does

1. Recovers R5/R7 gold labels from existing HTML review payloads.
2. Joins workbook JSON payloads and PDF/page-cache source text.
3. Builds compact evidence text for each RESULTNO.
4. Scores samples against R5/R7 expert latent prototypes.
5. Evaluates multiple research variants:
   - `prototype_only`: direct latent-prototype classifier.
   - `crag_correction`: corrective RAG-style baseline preservation.
   - `selfrag_abstain`: self-reflection/manual-review gate.
   - `pattern_crag`: domain pattern retrieval gate.
   - `hybrid_crag`: max of embedding and pattern evidence.
   - `contrastive_crag`: dev-only contrastive error mining.
6. Writes a run folder under `research/latent_guided_r57/outputs/`.

## Main Outputs

- `REPORT.md`: readable run summary.
- `experiment_results.json`: full metrics, thresholds, and top contrastive tokens.
- `dataset_compact.jsonl`: recovered research dataset without full compact source text.
- `changed_or_review_cases.csv`: candidate flips/manual-review cases for expert audit.

## Current Findings

The baseline R5 workflow is already strong, so naive latent/prototype
replacement tends to over-correct. R5 is better treated as a conservative
CRAG-style correction task.

R7 has much larger improvement space. Contrastive mining can sharply improve
gold-zero recall on the full data, but the current split exposes overfitting to
target/geographic names. The next research step should mask target names and
force concept-level evidence chains:

```text
target/component -> magnetization age -> paleopole/APWP relation
-> younger event or overprint -> rescue/field-test evidence
```

That observation is useful for both paper framing and patent claims: the core
innovation should not be "embedding retrieval" alone, but conflict-aware,
concept-level latent rule guidance.

## Second-Stage R7 Evidence-Chain Experiment

Run:

```bash
.venv/bin/python3 research/latent_guided_r57/scripts/run_r7_evidence_chain_experiment.py
```

Latest report:

`research/latent_guided_r57/outputs/r7_chain_20260708_163349/REPORT.md`

This script is closer to the paper implementations:

- Self-RAG: retrieval/reflection flags for relevant, supported, and conflict.
- CRAG: retrieval-quality gate plus conservative correction.
- GraphRAG: explicit evidence nodes and same-page support/conflict edges.
- Knowledge conflict: negative R7 evidence vs rescue evidence.

Best current test-set result:

```text
baseline R7=0 F1:                 0.1481
targeted_conservative_crag F1:    0.2775
targeted_crag_graph F1:           0.3904
```

The high-recall targeted graph versions are better as expert-review discovery
tools; the conservative CRAG version is the safer candidate for integration.
