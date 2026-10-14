# R3 exact300 auxiliary-training diagnostic

Date: 2026-07-13

> Update: this document records the earlier 87.33% auxiliary baseline. The
> later nested router plus paper-consistency result reaches 90.00%; see
> `docs/r3_exact300_paper_consistency_90pct_20260713.md`.

## Outcome

The strongest canonical, full-coverage result in this branch is **87.33%**
(262/300) from `aux_combined_hgb`.  It uses the 2,172 PDF-backed rows only as
an auxiliary training pool.  Every outer fold excludes every auxiliary row
whose `REFNO` occurs in that fold's 60 exact300 test rows.

Confusion matrix: TN/FP/FN/TP = `87/23/15/175`.

Canonical fold accuracies: `90.00 / 86.67 / 78.33 / 90.00 / 91.67%`.

This auxiliary-only branch does not reach 90%; eight additional correct rows
would be required. The later nested-router/paper-consistency branch is a
separate promoted result and reaches 90.00%.

## Canonical fold freeze

`GroupKFold` tie assignment differs between sklearn 1.6.1 and 1.9.0.  The
initial 86.67% auxiliary result used the noncanonical 1.6.1 assignment and is
diagnostic only.  A label-free fold manifest now makes every branch use the
same outer membership as the frozen baseline:

- Manifest: `tmp/r3_exact_300_resultno_v2/refno_fold_manifest.json`
- Manifest SHA-256: `8f6ab8f1a071878e14d4db8e19a7ec0519ed612be107b0f0fdf0489dd59aaf2d`
- Fold-id SHA-256: `28552aa3627c2c808ff49ac50c7c9d67a6f32daaad0e9aff5ae6a96fbe61f08f`
- Rows per fold: `60/60/60/60/60`
- Post-freeze positive-count audit: `40/37/33/34/46`
- REFNO assigned to multiple folds: `0`

The builder reads only exact identity fields and uses all-zero placeholders as
the sklearn `y` argument.  It does not open targets, PDF text, source tables,
predictions, or evaluations.

## Auxiliary training boundary

Canonical outer-fold auxiliary training row counts are
`2035/2015/2028/2036/2026`.  The corresponding complete-paper exclusions are
`137/157/144/136/146` rows.  Every reported outer train/test REFNO overlap is
zero.

Allowed features are recomputable from:

- frozen identity `ROCKNAME` text;
- the original-paper page cache and local PDF byte size;
- the frozen aggregate R3 taxonomy;
- the frozen positive ROCKMAG expert vocabulary.

No `PMAGRESULT`, row-level `ROCKMAG`, old prediction, old evaluation, numeric
`REFNO`, or numeric `RESULTNO` is a model feature.

## Candidate results

| Candidate | Accuracy | Notes |
|---|---:|---|
| Combined curation + expert-count HGB | 87.33% | Best canonical candidate |
| Curation-only HGB | 87.00% | Similar error set |
| Numeric + target-local text, 50% | 87.00% | Fewer FN but more FP |
| Nested numeric threshold | 86.33% | Valid inner OOF, no improvement |
| Sparse full-paper + target NB-SVM | 85.00% | Independent canonical branch |
| Exact300 baseline | 83.33% | 250/300 |

The nested threshold used four REFNO-grouped inner folds within each allowed
outer auxiliary set.  Selected thresholds were
`0.675/0.500/0.650/0.625/0.525`; all 20 inner fit/validation overlaps and all
inner/outer-test overlaps were zero.  Its lower result proves that per-fold
post-hoc threshold ceilings are not an attainable calibration claim.

## Remaining bottleneck

For the best canonical model, 10 of 23 false positives are strict expert-text
positives despite historical R3 label 0.  Conversely, 11 of 15 false negatives
are strict expert-text negatives despite historical label 1.  Thus 21 of the
38 residual model errors point in the same direction as the frozen semantic
expert layer and conflict with the historical evaluation label.

Nine rows are wrong under every one of the 26 auxiliary candidates.  They are
all labelled 0:

- Chenjiang Formation (two RESULTNO rows, REFNO 3124)
- Nankoweap Formation (REFNO 3576)
- Kamchatka igneous rocks (REFNO 3038)
- Labaztakh and Kotuykan Formations (REFNO 3875)
- Inuyama Chert (REFNO 3371)
- Linok Formation, upper part (REFNO 4086)
- Sunnhordland Dykes (REFNO 4123)
- Uaua Dykes (REFNO 4127)

Six are target-attribution conflicts: the paper has positive rock-magnetic
evidence, but the strict target-bound layer is negative.  Three are direct
source/label conflicts.  Inuyama Chert, Sunnhordland Dykes, and Uaua Dykes are
strict target-bound positives; the latter two contain accepted method plus
carrier/diagnostic evidence but remain R3=0 in the truth workbook.

This is not primarily a PDF parser failure.  More permissive semantic retrieval
would increase several of these counted false positives.

## Route subsequently promoted

The next defensible model identified by this diagnostic was conflict-aware and
hierarchical; it was subsequently implemented by the nested-router branch:

1. A semantic fact head predicts method, diagnostic result, carrier, target
   binding, own-study status, and paper/target scope from original text.
2. An annotation-policy head learns historical R3 assignment from curation
   metadata and the semantic fact vector using only allowed outer-training
   papers.
3. A small router is fitted from inner, group-isolated predictions inside each
   outer fold.  It must never train on ordinary OOF probabilities whose base
   fits included the current outer-test papers.
4. The report should expose both historical-label accuracy and semantic
   conflict counts.  This preserves practical scoring while making knowledge
   conflict detection an explicit patentable output rather than hidden noise.

The other high-value route is auxiliary-pool SciNCL fine-tuning on target,
abstract, target-local evidence, and paper evidence.  It must load the frozen
fold manifest above and exclude all rows from every outer-test REFNO.

## Artifacts

- Canonical report SHA-256: `d205ee13feedc11516761a5c99ce41adc569a1cb32605b2076f977e64b8cfbd5`
- Canonical OOF SHA-256: `007a55402053b1226a10f49286514f024752fec2c67492f1b2a382e8c47b04df`
- Feature freeze SHA-256: `828d289c8fc88150682458b17ef2b5a2273404a5c6984411c1362afc22447e73`
- Frozen source-evidence candidates SHA-256: `8ff1a752c079474b22ce48bea88e9d48fe8825c6f381bb353ff10db2216984d7`

The 90% gate remains failed for this earlier auxiliary-only branch. Oracle
routing, selective coverage, post-hoc outer-fold thresholds, and noncanonical
sklearn folds are not promoted as full-coverage results. The separate canonical
nested-router plus paper-consistency v2 branch later passes the exact300 gate
at 270/300; see `docs/r3_exact300_paper_consistency_90pct_20260713.md`.

## Compact numeric model-family sweep

A reusable label-free numeric cache was added after the main auxiliary run:

- Cache: `tmp/r3_exact_300_resultno_v2/auxiliary_numeric_cache_v1/features.npz`
- Cache SHA-256: `12b600c620ce069f3762a2741aff6b803f2295a2d3c834097eb5bdfbebbe07b5`
- Manifest SHA-256: `1dad1f75cbfdb1f1d906e9370951bc8005cb28963ac37ba2d58cf11416246cbb`
- `curation_all` matrix SHA-256: `b3287f28ec1f532d1c5425614c538435556b173c3d59c4486637c6490b885638`
- `combined` matrix SHA-256: `b1ca1515ad31ff10da86971fe91a66a4376e824d169c229821a8317bbbdf8b98`

The cache contains five float32 feature matrices and no labels or identifiers.
Its matrices reproduce the independently frozen CatBoost inputs.

Twenty-six predeclared candidates covered standardized logistic regression,
linear SVM, RBF SVM, random forests, new ExtraTrees settings, distance-weighted
kNN, and a fixed 32-unit MLP.  On folds 1-3, only linear, forest, and MLP
families met the predeclared continuation threshold.  RBF SVM reached 83.33%
and kNN 80.00%, so those families stopped before folds 4-5.

The best full result was `rf_combined_leaf8_half`:

- Accuracy: **87.33%** (`85/25/13/177`)
- Folds: `90.00 / 88.33 / 80.00 / 91.67 / 86.67%`
- Report SHA-256: `9814e57ae621a5b3115d46b5762f2f91c66261320f1f65242ec3f6786bf3eca0`
- OOF SHA-256: `f124a4c073ba0120a63a37cc43dc3ac0fd3e7d624ad28aaa327710562c6aad80`

This ties, but does not exceed, the combined HGB result.  Random forest fixes
seven HGB errors and introduces seven different errors.  The two-model
row-level oracle is only 89.67%, and every fixed arithmetic blend remains at
or below 87.33%.  Linear models top out at 84.33% and MLP at 84.00%.

CatBoost with inner REFNO-isolated depth, weight, iteration, and threshold
selection independently reached only 83.67%.  Together these results rule out
classifier-family choice and ordinary probability calibration as the main
remaining bottleneck; new target-attribution or conflict-state information is
required.
