# REAR R7 Strict Reproduction Shell

Generated: 2026-07-08T18:42:41

## Alignment Boundary

- Official REAR code/data were downloaded under `external/`.
- This run reproduces the REAR data objects, Eq. 10 relevance label construction, source-reliability verification, and knowledge-consistency selection shape.
- It does not train LLaMA2-7B with a learned linear rank head; `s_ce` is a deterministic local scorer used as the small-reranker substitute.

## Metrics All Rows

| Method | Accuracy | Macro-F1 | R7=0 F1 | FP | FN | Changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 0.8966 | 0.5364 | 0.1277 | 300 | 28 | 0 |
| rear_source_reliability | 0.1891 | 0.1890 | 0.1782 | 45 | 2528 | 2769 |
| rear_knowledge_consistency_proxy | 0.8966 | 0.5364 | 0.1277 | 300 | 28 | 0 |

## Outputs

- `rear_r7_dataset.json`: REAR-style question/reference/ctxs records.
- `rear_rank_sft_examples.jsonl`: rank-SFT pair examples with Eq. 10 labels.
- `cases.csv`: per-RESULTNO decisions and top relevance evidence.
- `summary.json`: full metrics and alignment metadata.
