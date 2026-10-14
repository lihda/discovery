# Research Plan: Latent-Guided R5/R7 Reliability Assessment

## Working Title

Conflict-aware latent-guided expert-rule retrieval for paleomagnetic R5/R7
reliability assessment.

## Motivation

The existing deterministic workflow is a strong baseline, but the error modes
are different for R5 and R7:

- R5: high accuracy, mainly false positives where structural-control risk is
  under-detected.
- R7: lower accuracy, many false positives where younger-pole, remagnetization,
  or age-reset evidence is not detected.

This makes R5 suitable for conservative corrective retrieval, while R7 is a
better target for latent-guided discovery and conflict-aware evidence chains.

## Paper Ideas Being Reproduced

### 1. Self-RAG-style Reflection

Idea: a model should decide whether retrieval is needed, critique retrieved
evidence, and abstain or request review when evidence is conflicting.

Local reproduction:

- `selfrag_abstain`
- uses negative vs rescue evidence scores;
- preserves baseline unless retrieved evidence is strong;
- marks near-tie negative/rescue evidence as manual-review cases.

Research value:

- Turns R5/R7 scoring from pure classification into auditable decision support.
- Useful for patent claims around automatic expert-review triggering.

### 2. CRAG-style Corrective Retrieval

Idea: retrieval should not blindly override a base model; the system should
evaluate retrieval quality and only correct when evidence is reliable.

Local reproduction:

- `crag_correction`
- `pattern_crag`
- `hybrid_crag`

Research value:

- Especially appropriate for R5, whose current baseline is already strong.
- Provides a conservative method for reducing false positives without
  introducing many false negatives.

### 3. Knowledge-Conflict / Adaptive Chameleon-style Behavior

Idea: external evidence, model priors, and existing knowledge can conflict; the
system should expose the conflict rather than silently choosing one.

Local reproduction:

- negative prototypes vs rescue prototypes;
- `manual_review` when both sides are active;
- changed/review cases are exported for expert arbitration.

Research value:

- This is the most patentable part: a domain-specific rule conflict detector
  for paleomagnetic reliability indicators.

### 4. GraphRAG-style Evidence Organization

Idea: domain evidence is relational, not just a bag of text chunks.

Planned local extension:

```text
RESULTNO
  -> target unit
  -> component
  -> magnetization age
  -> pole/APWP relation
  -> field tests
  -> expert rules
  -> rescue/conflict status
```

Research value:

- This should be the next step for R7, because R7 errors often require an
  evidence chain rather than a single keyword match.

### 5. Contrastive Error Mining

Idea: use errors to discover latent risk factors that are missing from explicit
rules.

Local reproduction:

- `contrastive_crag`
- learns gold-zero risk tokens only from the dev split;
- applies the learned risk score to test/all splits.

Finding:

- It can substantially improve full-set R7 gold-zero F1, but the current version
  overfits to target/geographic names.

Next fix:

- mask target names, places, author names, and numeric ages;
- keep only geological-process and reliability-concept tokens;
- convert validated tokens into rule-card prototypes.

## Near-Term Experimental Roadmap

1. **Name-masked contrastive mining**
   - Mask target, place, terrane, formation/group/suite names.
   - Re-run `contrastive_crag`.
   - Compare R7 test F1_0 before/after masking.

2. **R7 evidence-chain extraction**
   - Extract candidate snippets for:
     - target/component;
     - paleopole/APWP;
     - younger event/remagnetization;
     - rescue field tests.
   - Score a sample only when at least two linked evidence nodes are present.

3. **Conflict-aware graph export**
   - Build JSON graph records:
     - nodes: RESULTNO, rule, snippet, component, pole, age, field test;
     - edges: supports, contradicts, rescues, requires-review.
   - Use this graph as the source for a GraphRAG-style report.

4. **Expert feedback loop**
   - Ask domain experts to review `changed_or_review_cases.csv`.
   - Convert accepted cases into new `rule_card` entries.
   - Re-run experiments and measure whether new rules improve held-out cases.

5. **Patent-ready system claims**
   - rule-card construction;
   - latent prototype retrieval;
   - negative/rescue/conflict scoring;
   - manual-review triggering;
   - feedback-driven rule update.

## Current Best Interpretation

R5 should be improved as a conservative correction layer over the existing
workflow. R7 should become the main research contribution, because its false
positive problem is large and semantically rich.

The most defensible final method is:

```text
baseline deterministic R5/R7
  + latent expert-rule retrieval
  + concept-masked contrastive discovery
  + evidence-chain graph
  + conflict/rescue/manual-review gate
```

## Second-Stage Evidence-Chain Results

The second-stage script
`research/latent_guided_r57/scripts/run_r7_evidence_chain_experiment.py`
implements a more faithful paper-inspired pipeline:

```text
Self-RAG: retrieve / relevant / supported / conflict reflection flags
CRAG: retrieval-quality and conservative correction gate
GraphRAG: evidence nodes and same-page support/conflict edges
Knowledge conflict: negative evidence vs rescue evidence
```

Latest run:

`research/latent_guided_r57/outputs/r7_chain_20260708_163349/REPORT.md`

Key R7 results on the held-out test split:

| Method | Accuracy | Macro-F1 | R7=0 F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.8787 | 0.5414 | 0.1481 | 108 | 7 |
| targeted_graph_chain | 0.8354 | 0.6454 | 0.3858 | 69 | 87 |
| targeted_crag_graph | 0.8386 | 0.6487 | 0.3904 | 69 | 84 |
| targeted_conservative_crag | 0.8681 | 0.6025 | 0.2775 | 94 | 31 |

Interpretation:

- Target-anchored evidence chains clearly improve R7=0 detection.
- The unconstrained/high-recall versions are useful as expert-review discovery
  tools, but not yet safe as automatic replacements.
- The conservative CRAG variant is the best candidate for a workflow extension:
  it nearly doubles R7=0 F1 on the test split while limiting the accuracy drop.
- A component-aware rule improved the conservative method by reducing false
  flips where the RESULTNO itself is an overprint/secondary component.

Remaining scientific problem:

Many false flips come from cases where the paper discusses APWP, overprint, or
remagnetization, but the target RESULTNO is still considered reliable by the
gold label. This suggests the next research module should explicitly classify:

```text
overprint as target component
vs.
overprint as contamination of a claimed primary component
```

That distinction is a strong thesis/patent point because it is a real
domain-specific knowledge conflict, not a generic RAG issue.
