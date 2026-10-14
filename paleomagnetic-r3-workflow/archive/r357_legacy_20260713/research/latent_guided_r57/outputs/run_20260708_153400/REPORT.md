# R5/R7 Latent-Guided Exploration Report

Generated: 2026-07-08T15:35:03
Embedding backend: `hash`
Records: 20

## Baseline
- R5: accuracy=1.0000, macro_f1=1.0000, f1_0=1.0000, fp=0, fn=0
- R7: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0

## Methods
### R5
- `baseline` all: accuracy=1.0000, macro_f1=1.0000, f1_0=1.0000, fp=0, fn=0
- `prototype_only` test: accuracy=0.8571, macro_f1=0.7879, f1_0=0.9091, fp=0, fn=1, changed=1, manual_review=0
  all: accuracy=0.8000, macro_f1=0.7980, f1_0=0.7778, fp=1, fn=3, changed=4, manual_review=0
- `crag_correction` test: accuracy=1.0000, macro_f1=1.0000, f1_0=1.0000, fp=0, fn=0, changed=0, manual_review=0
  all: accuracy=1.0000, macro_f1=1.0000, f1_0=1.0000, fp=0, fn=0, changed=0, manual_review=0
- `selfrag_abstain` test: accuracy=1.0000, macro_f1=1.0000, f1_0=1.0000, fp=0, fn=0, changed=0, manual_review=0
  all: accuracy=1.0000, macro_f1=1.0000, f1_0=1.0000, fp=0, fn=0, changed=0, manual_review=0
### R7
- `baseline` all: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0
- `prototype_only` test: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0, changed=0, manual_review=0
  all: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0, changed=0, manual_review=0
- `crag_correction` test: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0, changed=0, manual_review=0
  all: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0, changed=0, manual_review=0
- `selfrag_abstain` test: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0, changed=0, manual_review=0
  all: accuracy=0.0000, macro_f1=0.0000, f1_0=0.0000, fp=0, fn=0, changed=0, manual_review=0

## Interpretation
The experiments are lightweight reproductions of retrieval correction and conflict-aware reasoning ideas. They should be treated as research probes, not replacements for the calibrated production rules.
A promising paper/patent direction is to use the generated changed/manual-review cases as expert-feedback seeds, then iteratively convert validated cases into rule cards and latent prototypes.
