# R3 exact300 sparse-text auxiliary experiment

## Scope and leakage boundary

- Evaluation membership: frozen exact300 identity, 300 rows from 258 REFNO groups.
- Canonical folds: `tmp/r3_exact_300_resultno_v2/refno_fold_manifest.json`, SHA256
  `8f6ab8f1a071878e14d4db8e19a7ec0519ed612be107b0f0fdf0489dd59aaf2d`.
- Auxiliary pool: 2,172 PDF-backed rows from 1,174 papers.
- In each fold, all auxiliary rows whose REFNO occurs in the 60-row outer test
  fold are excluded before fitting.
- Predictive inputs are limited to ROCKNAME, frozen original-paper page-cache
  text, and frozen aggregate terminology patterns. Alignment identifiers are
  not vectorized. Raw source fields and old predictions/evaluations are not
  model inputs.
- All text views and candidate specifications are hashed before R3 labels are
  opened.

## Canonical result

The best predeclared full-coverage candidate is
`paper_sparse__target_local_25`: 75% paper-level sparse ensemble plus 25%
target-local word NB-SVM.

| Metric | Result |
| --- | ---: |
| Accuracy | 85.00% (255/300) |
| Macro F1 | 82.62% |
| Balanced accuracy | 80.89% |
| TN / FP / FN / TP | 72 / 38 / 7 / 183 |
| Fold accuracy | 95.00 / 86.67 / 75.00 / 80.00 / 88.33% |

The 90% gate was not reached. The dominant residual error is false-positive
paper attribution: a paper can contain extensive magnetic-mineral terminology
without reporting qualifying rock-magnetic characterization for its own study
targets. Fold 3 contributes 14 false positives.

Post-result threshold diagnosis was not used for fitting or promotion. Even an
invalid global threshold chosen with all exact300 answers would raise the best
sparse candidate only to 85.33%, so threshold calibration is not the limiting
factor.

## Multiple-instance follow-up

A single extra candidate split the frozen knowledge neighborhoods into 7,633
passages, trained passage NB-SVM with equal total weight per training paper,
and propagated the mean probability of each test paper's top three passages.
It reached only 65.33% with TN/FP/FN/TP = 6/104/0/190. Extreme-passage pooling
selects citation and background mentions in negative papers and therefore
amplifies the existing positive bias. Its fixed 25% fusion reached 84.00% and
did not improve the 85.00% lightweight result.

## Artifacts

- Script: `scripts/experiment_r3_exact300_sparse_text.py`, SHA256
  `757e2a01985e2574e421cf1aca930a10f5bb22662642138092beaebd124a73f0`.
- Tests: `tests/test_r3_exact300_sparse_text.py`.
- Report: `tmp/r3_exact_300_resultno_v2/sparse_text_v3_passage/report.json`,
  SHA256 `0867dc76e25a9af2828ad4ec649ea7089f1fe96ef77a961d8e07378c4443f2a4`.
- Feature freeze: `tmp/r3_exact_300_resultno_v2/sparse_text_v3_passage/feature_freeze.json`,
  SHA256 `e53f51e6ab9fd79010da90dd9c58a188fd0a2ac219538a0f44316570ee8aeb6f`.
- Identifier-free OOF probabilities:
  `tmp/r3_exact_300_resultno_v2/sparse_text_v3_passage/oof_probabilities.npz`,
  SHA256 `144cafcaaeb8651b574c87fea7aaa45794a532579032a760095258e45be4be7c`.

Reproduction:

```bash
.venv/bin/python scripts/experiment_r3_exact300_sparse_text.py
.venv/bin/python -m unittest \
  tests.test_r3_exact300_sparse_text \
  tests.test_r3_exact300_auxiliary_training \
  tests.test_r3_exact300_fold_manifest -v
```

The sparse branch is useful as a complementary representation, but its own
85% result is insufficient for promotion. Further work should model whether a
claim describes the authors' study samples, rather than add broader keyword
coverage or tune a global decision threshold.
