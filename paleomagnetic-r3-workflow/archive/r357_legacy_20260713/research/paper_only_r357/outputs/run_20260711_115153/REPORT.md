# Paper-Only R357 Exploration

Generated: 2026-07-11T11:53:41

## Leakage Boundary

- Scoring reads only REFNO, RESULTNO, ROCKNAME and original-paper page text.
- PMAGRESULT and ROCKUNIT are not imported or read.
- R3/R5/R7 labels are loaded only after predictions are frozen, for evaluation.

## Metrics

| Indicator | Accuracy | Macro-F1 | F1(0) | FP | FN | Review rate | Accepted accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| R3 | 0.6565 | 0.6356 | 0.5485 | 600 | 490 | 0.4437 | 0.7201 |
| R5 | 0.7425 | 0.4389 | 0.0262 | 798 | 19 | 0.5761 | 0.8223 |
| R7 | 0.8431 | 0.5092 | 0.1043 | 295 | 203 | 0.7110 | 0.7623 |

## Method

The workflow recovers domain attributes from paper text, binds evidence to the current target, and performs Astute-RAG-inspired source-aware consolidation over internal criterion defaults, external opposition, rescue evidence, and filtered noise.
