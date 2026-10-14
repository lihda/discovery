# R5/R7 Latent-Guided Exploration Report

Generated: 2026-07-08T15:44:17
Embedding backend: `hash`
Records: 6346

## Baseline
- R5: accuracy=0.9559, macro_f1=0.9383, f1_0=0.9053, fp=140, fn=0
- R7: accuracy=0.8966, macro_f1=0.5364, f1_0=0.1277, fp=300, fn=28

## Methods
### R5
- `baseline` all: accuracy=0.9559, macro_f1=0.9383, f1_0=0.9053, fp=140, fn=0
- `prototype_only` test: accuracy=0.5105, macro_f1=0.4511, f1_0=0.2704, fp=137, fn=327, changed=452, manual_review=366
  all: accuracy=0.5307, macro_f1=0.4864, f1_0=0.3356, fp=433, fn=1056, changed=1475, manual_review=1332
- `crag_correction` test: accuracy=0.9409, macro_f1=0.9147, f1_0=0.8673, fp=40, fn=16, changed=16, manual_review=8
  all: accuracy=0.9420, macro_f1=0.9208, f1_0=0.8799, fp=135, fn=49, changed=54, manual_review=33
- `selfrag_abstain` test: accuracy=0.9409, macro_f1=0.9147, f1_0=0.8673, fp=40, fn=16, changed=16, manual_review=8
  all: accuracy=0.9420, macro_f1=0.9208, f1_0=0.8799, fp=135, fn=49, changed=54, manual_review=33
- `pattern_crag` test: accuracy=0.9430, macro_f1=0.9177, f1_0=0.8720, fp=39, fn=15, changed=16, manual_review=0
  all: accuracy=0.9398, macro_f1=0.9178, f1_0=0.8752, fp=139, fn=52, changed=63, manual_review=0
- `hybrid_crag` test: accuracy=0.9430, macro_f1=0.9177, f1_0=0.8720, fp=39, fn=15, changed=16, manual_review=0
  all: accuracy=0.9398, macro_f1=0.9178, f1_0=0.8752, fp=139, fn=52, changed=63, manual_review=0
- `contrastive_crag` test: accuracy=0.8966, macro_f1=0.8610, f1_0=0.7906, fp=38, fn=60, changed=78, manual_review=1
  all: accuracy=0.9442, macro_f1=0.9279, f1_0=0.8936, fp=66, fn=111, changed=201, manual_review=2
- `masked_contrastive_crag` test: accuracy=0.8977, macro_f1=0.8618, f1_0=0.7914, fp=39, fn=58, changed=75, manual_review=4
  all: accuracy=0.9452, macro_f1=0.9290, f1_0=0.8951, fp=67, fn=107, changed=196, manual_review=6
### R7
- `baseline` all: accuracy=0.8966, macro_f1=0.5364, f1_0=0.1277, fp=300, fn=28
- `prototype_only` test: accuracy=0.8038, macro_f1=0.5499, f1_0=0.2119, fp=93, fn=93, changed=121, manual_review=442
  all: accuracy=0.8276, macro_f1=0.5267, f1_0=0.1493, fp=276, fn=271, changed=343, manual_review=1489
- `crag_correction` test: accuracy=0.8660, macro_f1=0.5485, f1_0=0.1699, fp=105, fn=22, changed=18, manual_review=19
  all: accuracy=0.8818, macro_f1=0.5412, f1_0=0.1458, fp=292, fn=83, changed=63, manual_review=66
- `selfrag_abstain` test: accuracy=0.8660, macro_f1=0.5485, f1_0=0.1699, fp=105, fn=22, changed=18, manual_review=19
  all: accuracy=0.8818, macro_f1=0.5412, f1_0=0.1458, fp=292, fn=83, changed=63, manual_review=66
- `pattern_crag` test: accuracy=0.8492, macro_f1=0.5305, f1_0=0.1437, fp=106, fn=37, changed=36, manual_review=9
  all: accuracy=0.8686, macro_f1=0.5328, f1_0=0.1366, fp=291, fn=126, changed=125, manual_review=14
- `hybrid_crag` test: accuracy=0.8492, macro_f1=0.5305, f1_0=0.1437, fp=106, fn=37, changed=36, manual_review=9
  all: accuracy=0.8686, macro_f1=0.5328, f1_0=0.1366, fp=291, fn=126, changed=125, manual_review=14
- `contrastive_crag` test: accuracy=0.8460, macro_f1=0.5180, f1_0=0.1205, fp=108, fn=38, changed=47, manual_review=2
  all: accuracy=0.8944, macro_f1=0.7392, f1_0=0.5379, fp=129, fn=206, changed=391, manual_review=4
- `masked_contrastive_crag` test: accuracy=0.8502, macro_f1=0.5154, f1_0=0.1125, fp=109, fn=33, changed=41, manual_review=1
  all: accuracy=0.8960, macro_f1=0.7396, f1_0=0.5378, fp=132, fn=198, changed=380, manual_review=1

## Interpretation
The experiments are lightweight reproductions of retrieval correction and conflict-aware reasoning ideas. They should be treated as research probes, not replacements for the calibrated production rules.
A promising paper/patent direction is to use the generated changed/manual-review cases as expert-feedback seeds, then iteratively convert validated cases into rule cards and latent prototypes.
