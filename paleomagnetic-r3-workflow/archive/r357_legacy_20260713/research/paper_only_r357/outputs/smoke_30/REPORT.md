# Paper-Only R357 Exploration

Generated: 2026-07-11T11:49:57

## Leakage Boundary

- Scoring reads only REFNO, RESULTNO, ROCKNAME and original-paper page text.
- PMAGRESULT and ROCKUNIT are not imported or read.
- R3/R5/R7 labels are loaded only after predictions are frozen, for evaluation.

## Metrics

| Indicator | Accuracy | Macro-F1 | F1(0) | FP | FN | Review rate | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R3 | 0.7000 | 0.6970 | 0.6667 | 4 | 5 | 0.4667 | 0.7500 |
| R5 | 0.5667 | 0.5543 | 0.4800 | 10 | 3 | 0.4667 | 0.7500 |
| R7 | 0.5333 | 0.4034 | 0.1250 | 1 | 13 | 0.6000 | 0.6667 |

## Method

The workflow recovers domain attributes from paper text, binds evidence to the current target, and performs Astute-RAG-inspired source-aware consolidation over internal criterion defaults, external opposition, rescue evidence, and filtered noise.
