# REAR Strict Reproduction Alignment

## Selected Paper

Paper: **REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering**

Venue: EMNLP 2024 main conference.

Official sources used:

- ACL Anthology paper page: https://aclanthology.org/2024.emnlp-main.321/
- Official code/data link listed by the paper: https://github.com/RUCAIBox/REAR
- Downloaded ACL artifacts:
  - `research/rear_strict_reproduction/external/rear_paper.pdf`
  - `research/rear_strict_reproduction/external/rear_paper.txt`
  - `research/rear_strict_reproduction/external/rear_software.zip`
  - `research/rear_strict_reproduction/external/rear_data.zip`

## What REAR Does

REAR targets noisy retrieval-augmented QA. Its key point is that retrieved
documents should not be treated equally. The paper introduces:

1. **Relevance assessment**
   - Map a query-document pair into a relevance embedding:
     `v_rel = LLM(q, d)[-1]`
   - Project it to a scalar relevance score:
     `s_rel = Assess(v_rel)`
   - In official code, this is implemented as a linear rank head over the hidden
     state at the relevance token.

2. **Relevance-guided generation**
   - Map `s_rel` into a dense guide vector and let the LLM decide whether to rely
     on internal knowledge or external evidence.

3. **Knowledge reliability verification**
   - Source-reliability: trust answers supported by high-relevance documents.
   - Knowledge-consistency: combine document relevance with a consistency score
     computed by asking how strongly the LLM keeps the same answer when
     relevance is set to zero.

4. **Training**
   - Bi-granularity relevance fusion:
     `L_bi-granularity = L_coarse + L_fine`
   - Eq. 10 training labels:
     `s_rel = 0.5 * (s_ce + y)`
   - Noise-resistant training with irrelevant negative documents.

## Official Code Anchors

Relevant files in the downloaded software artifact:

- `external/software/LLaMA-Efficient-Tuning/src/llmtuner/tuner/core/modeling_rankllama.py`
  - `self.rel_score = nn.Linear(config.hidden_size, num_labels, bias=False)`
  - `rank_loss()` extracts the hidden state at `rel_score_id`.
  - `LossForRank` implements BCE, pointwise, RankNet, RankNet+BCE, and listwise variants.

- `external/software/LLaMA-Efficient-Tuning/src/llmtuner/tuner/ranksft/workflow.py`
  - Loads model with `stage="ranksft"`.
  - Runs rank-SFT training and the custom rank data collator.

- `external/software/LLaMA-Efficient-Tuning/src/llmtuner/tuner/ranksft/collator.py`
  - Carries `classes` and `ans_score`.
  - Supports masking `<IRRELEVANT>` / `<RELEVANT>` judgment tokens.

- Official test data format from ACL data zip:
  - `question`
  - `reference`
  - `ctxs`

## Paleomagnetic Reproduction Scope

The local script:

`research/rear_strict_reproduction/scripts/run_rear_r7_reproduction.py`

strictly reproduces these REAR objects and operations:

1. Converts each R7 RESULTNO into REAR-style `question/reference/ctxs`.
2. Builds query-document pairs from PMAGRESULT/ROCKUNIT metadata, R7 JSON
   evidence, and source-text snippets.
3. Computes `s_ce_proxy` for a local small-reranker substitute.
4. Computes Eq. 10 labels exactly:
   `s_rel_eq10_label = 0.5 * (s_ce_proxy + binary_y)`.
5. Exports rank-SFT style examples with `<RELEVANT>` / `<IRRELEVANT>`.
6. Runs source-reliability and knowledge-consistency selection variants.
7. Adds a CPU-feasible linear relevance assessor as a proxy for the paper's
   explicit assessment module.

## Non-Reproduced Parts

These parts are not fully reproduced in the local paleomagnetic run:

1. No LLaMA2-7B fine-tuning is performed.
2. No hidden-state rank head is trained on GPU.
3. `s_ce_proxy` is not the original paper's cross-encoder reranker score.
4. The knowledge-consistency score is approximated with the deterministic R7
   baseline as the `s_rel = 0` internal-knowledge proxy, rather than inverse PPL.

This means the current output is a strict **method-shape reproduction and data
adapter**, not a full 7B model reproduction.

## Latest Local Result

Latest run:

`research/rear_strict_reproduction/outputs/run_20260708_185146`

All rows: 3173.

| Method | Accuracy | Macro-F1 | R7=0 F1 | FP | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.8966 | 0.5364 | 0.1277 | 300 | 28 |
| rear_source_reliability | 0.5159 | 0.4137 | 0.1688 | 168 | 1368 |
| rear_knowledge_consistency_proxy | 0.8966 | 0.5364 | 0.1277 | 300 | 28 |
| rear_trained_source_reliability | 0.8976 | 0.4730 | 0.0000 | 324 | 1 |
| rear_trained_knowledge_consistency | 0.8960 | 0.5288 | 0.1129 | 303 | 27 |

## Scientific Interpretation

The strict REAR reproduction exposes an important domain issue:

- A naive source-reliability strategy can find more R7=0 cases, but causes a
  large accuracy collapse because paleomagnetic papers contain many distracting
  mentions of remagnetization, APWP, younger poles, or impact structures that
  are not necessarily about the target RESULTNO.
- A conservative knowledge-consistency strategy protects baseline accuracy but
  does not yet improve R7=0 recall.
- A small local linear assessor is not enough. The paper's core claim matters:
  REAR needs a strong learned relevance module, ideally using hidden-state rank
  features or a domain-trained reranker.

For the paleomagnetic project, the most defensible next step is therefore not
"replace R7 with REAR", but:

```text
baseline R7
  + REAR-style query-document construction
  + domain relevance labels from Eq. 10
  + trained paleomagnetic relevance assessor
  + knowledge-consistency gate
```

This can become a paper/patent contribution as a domain-specific
relevance-aware reliability assessment framework.
