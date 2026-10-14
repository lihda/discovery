# Literature-Only Embedding Prototype Experiment

Generated: 2026-07-11T13:04:18

Apple NaturalLanguage English sentence embeddings are averaged over target-centered evidence windows.
The same REFNO never appears in both train and test.

| Indicator | Dev-selected method | Test accuracy | Macro-F1 | F1(0) | FP | FN | Pred 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R5 | hybrid_11 | 0.7266 | 0.4563 | 0.0729 | 177 | 1 | 8 |
| R7 | centroid | 0.8740 | 0.4664 | 0.0000 | 78 | 4 | 4 |

## Production Baseline Comparison

The production workbooks were evaluated on the same 651-row held-out test split.

| Indicator | Method | Test accuracy | Macro-F1 | F1(0) | FP | FN | Pred 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R5 | production baseline | 0.9524 | 0.9379 | 0.9080 | 31 | 0 | 153 |
| R5 | embedding selected on dev | 0.7266 | 0.4563 | 0.0729 | 177 | 1 | 8 |
| R7 | production baseline | 0.8771 | 0.5322 | 0.1304 | 72 | 8 | 14 |
| R7 | embedding selected on dev | 0.8740 | 0.4664 | 0.0000 | 78 | 4 | 4 |

## Validation

- Embedding rows: 3173; dimensions: 512; all values finite.
- Split rows: train 2076, dev 446, test 651.
- Unique REFNOs: train 1054, dev 234, test 339.
- REFNO overlap between train/dev/test: zero.
- Cached-vector re-evaluation reproduced the selected methods and metrics exactly.

## Conclusion

The prototype/kNN embedding methods are not suitable as replacements for the
current production R5/R7 classifiers. R5 loses substantial accuracy and R7
fails to identify any held-out class-0 case. The embeddings may still be tested
as a review-ranking or conflict-retrieval signal layered over the production
baseline, where they cannot directly overwrite the final indicator score.
