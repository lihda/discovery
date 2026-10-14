# Paper-Only R357 Exploration

Generated: 2026-07-11T11:51:40

## Leakage Boundary

- Scoring reads only REFNO, RESULTNO, ROCKNAME and original-paper page text.
- PMAGRESULT and ROCKUNIT are not imported or read.
- R3/R5/R7 labels are loaded only after predictions are frozen, for evaluation.

## Metrics

| Indicator | Accuracy | Macro-F1 | F1(0) | FP | FN | Review rate | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R3 | 0.7000 | 0.6970 | 0.6667 | 4 | 5 | 0.8000 | 0.6667 |
| R5 | 0.4667 | 0.3182 | 0.0000 | 16 | 0 | 0.6000 | 0.6667 |
| R7 | 0.9000 | 0.4737 | 0.0000 | 2 | 1 | 0.8000 | 1.0000 |

## Method

The workflow recovers domain attributes from paper text, binds evidence to the current target, and performs Astute-RAG-inspired source-aware consolidation over internal criterion defaults, external opposition, rescue evidence, and filtered noise.
