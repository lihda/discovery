# REAR R7 Strict Reproduction Shell

Generated: 2026-07-08T18:52:18

## Alignment Boundary

- Official REAR code/data were downloaded under `external/`.
- This run reproduces the REAR data objects, Eq. 10 relevance label construction, source-reliability verification, and knowledge-consistency selection shape.
- It does not train LLaMA2-7B with a learned linear rank head; `s_ce` is a deterministic local scorer used as the small-reranker substitute.
- A local linear relevance assessor is trained on Eq. 10 labels as a CPU-feasible proxy for the paper's explicit assessment module.

## Metrics All Rows

| Method | Accuracy | Macro-F1 | R7=0 F1 | FP | FN | Changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.8966 | 0.5364 | 0.1277 | 300 | 28 | 0 |
| rear_source_reliability | 0.5159 | 0.4137 | 0.1688 | 168 | 1368 | 1500 |
| rear_knowledge_consistency_proxy | 0.8966 | 0.5364 | 0.1277 | 300 | 28 | 0 |
| rear_trained_source_reliability | 0.8976 | 0.4730 | 0.0000 | 324 | 1 | 53 |
| rear_trained_knowledge_consistency | 0.8960 | 0.5288 | 0.1129 | 303 | 27 | 4 |

## Outputs

- `rear_r7_dataset.json`: REAR-style question/reference/ctxs records.
- `rear_rank_sft_examples.jsonl`: rank-SFT pair examples with Eq. 10 labels.
- `cases.csv`: per-RESULTNO decisions and top relevance evidence.
- `summary.json`: full metrics and alignment metadata.
