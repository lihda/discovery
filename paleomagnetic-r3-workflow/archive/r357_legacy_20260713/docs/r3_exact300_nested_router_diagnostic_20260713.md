# R3 exact300 nested-router diagnostic (2026-07-13)

> Status update: the v1 router below remains the frozen 89.33% control. Its
> predeclared paper-consistency follow-up has now reached 270/300 = 90.00% on
> the same exact300 grouped-OOF scope. See the outcome section below and
> `docs/r3_exact300_paper_consistency_90pct_20260713.md`.

## Canonical result

The v1 nested loss router is a canonical, full-coverage REFNO-grouped result:

- Accuracy: 268/300 = 89.333333%
- Confusion: TN=89, FP=21, FN=11, TP=179
- Fold accuracy: 91.667%, 91.667%, 81.667%, 90.000%, 91.667%
- Best single expert: 263/300 = 87.666667%
- Router transition: 9 single-expert errors resolved, 4 new errors introduced

All 300 rows have one OOF prediction. The canonical fold-manifest SHA-256 is
`8f6ab8f1...`. Every outer-test REFNO is fully absent from auxiliary fitting,
and every inner validation REFNO is absent from its expert fit.

## Error decomposition

Of the 32 router errors, 20 are common errors of all six experts. These cannot
be fixed by changing the router alone. The remaining 12 have at least one
correct expert and are genuine routing opportunities. The six-expert oracle is
280/300 (93.333333%), so expert selection remains useful but is not sufficient
to explain the full gap to the labels.

The selected expert has minimum realized squared probability loss on only 2 of
the 32 error rows. Mean realized loss regret on errors is 0.24699713. v1 did
not persist the latent predicted-loss matrix, so this measures selection regret
against observed labels rather than direct calibration error of each predicted
route loss.

Confidence alone is a weak correction signal. Router errors have mean absolute
margin 0.3131 versus 0.3555 on correct rows, and 19 of 32 errors have margin at
least 0.30. Twenty errors occur under unanimous expert decisions. This pattern
is consistent with systematic label/criterion conflict or missing shared
evidence, not merely uncertain classification.

Eight errors belong to REFNOs with multiple exact300 rows. A nested paper-level
consistency candidate is therefore justified, but pooling weights must be
chosen only from inner REFNO-cross-fitted probabilities and inner labels.

## Paper-consistency outcome

The first predeclared follow-up was implemented as paper-consistency v2. It
recomputes the six experts and v1 router inside every outer fold and does not
accept a prior OOF file as input. Before any outer-test label can affect branch
selection, all fixed paper smoothers are compared by their binary decisions:

- folds 1, 3, 4 and 5 were decision-invariant, so the label-free early exit
  retained identity;
- fold 2 was non-invariant, so the smoother was selected only from a fully
  nested router-level OOF over the outer-training papers;
- that training OOF selected `paper_min_blend_100` at 83.62% versus 82.38%
  for identity;
- the selected fold-2 rule resolved two outer-test false positives and
  introduced no new errors.

The resulting full-coverage metric is 270/300 = 90.00%, with TN/FP/FN/TP =
91/19/11/179, macro-F1 89.06%, balanced accuracy 88.47%, and fold accuracies
91.67/95.00/81.67/90.00/91.67%. The result is a canonical development OOF
result on the fixed 300-row, 258-REFNO benchmark. It is not an independent
external validation result and does not establish 90% on the 2,172-row
PDF-only pool.

## v2 representation screen

The independent v2 screen froze record-free 531-dimensional field signatures
and 165-dimensional PDF structural metadata before labels. Its best folds 1-3
router reached 159/180 = 88.333333%, exactly tying the frozen v1 reference.
Because the gate required strict improvement, folds 4-5 were not run.

Raw field-signature and metadata concatenation therefore does not independently
justify a full canonical run. Metadata reduced some false negatives in the
router, but the field-signature experts were weaker as standalone models.

## Remaining tests after the 90% gate

1. Pairwise disagreement routing: train only on inner rows where two experts
   disagree, predict which expert is correct, and use a predeclared consensus
   fallback elsewhere. This targets the 12 selectable errors directly.
2. Nested loss calibration: cross-fit the loss router itself or calibrate
   per-expert predicted loss by inner REFNO folds. Persist only aggregate audit
   diagnostics, not row-level loss outputs usable as later features.
3. New contradiction-aware semantic experts: the 20 unanimous errors require
   a different evidence view, especially explicit own-study execution versus
   cited/background evidence and target/component attribution. More stacking
   of the same six decisions cannot fix these rows.
4. External validation: freeze the implementation and evaluate new papers or a
   new time-based cohort without using exact300 errors for further selection.

Because the exact300 gate has been reached after multiple development rounds,
the remaining model candidates should not be promoted by repeatedly comparing
them on exact300. They belong in a newly frozen external protocol.

The reproducible aggregate diagnostic is
`tmp/r3_exact_300_resultno_v2/nested_router_v1/diagnostic.json`. It is explicitly
label/OOF-derived and prohibited from runtime feature construction or formal
candidate and threshold selection.

The promoted paper-consistency report is
`tmp/r3_exact_300_resultno_v2/paper_consistency_v2/report.json`, SHA-256
`c77788373ddc3c227504f9f544f2df0c3348ceb201d6764da4178613d72b2cd9`.
