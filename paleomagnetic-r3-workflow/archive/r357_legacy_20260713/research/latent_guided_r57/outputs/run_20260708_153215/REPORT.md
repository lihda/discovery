# R5/R7 Latent-Guided Exploration Report

Generated: 2026-07-08T15:33:28
Embedding backend: `hash`
Records: 6346

## Baseline
- R5: accuracy=0.9559, macro_f1=0.9383, f1_0=0.9053, fp=140, fn=0
- R7: accuracy=0.8966, macro_f1=0.5364, f1_0=0.1277, fp=300, fn=28

## Methods
### R5
- `baseline` all: accuracy=0.9559, macro_f1=0.9383, f1_0=0.9053, fp=140, fn=0
- `prototype_only` test: accuracy=0.5992, macro_f1=0.4670, f1_0=0.2017, fp=175, fn=205, changed=352, manual_review=2
  all: accuracy=0.6124, macro_f1=0.4839, f1_0=0.2264, fp=629, fn=601, changed=1142, manual_review=7
- `crag_correction` test: accuracy=0.9578, macro_f1=0.9373, f1_0=0.9015, fp=40, fn=0, changed=0, manual_review=2
  all: accuracy=0.9559, macro_f1=0.9383, f1_0=0.9053, fp=140, fn=0, changed=0, manual_review=7
- `selfrag_abstain` test: accuracy=0.9578, macro_f1=0.9373, f1_0=0.9015, fp=40, fn=0, changed=0, manual_review=2
  all: accuracy=0.9559, macro_f1=0.9383, f1_0=0.9053, fp=140, fn=0, changed=0, manual_review=7
### R7
- `baseline` all: accuracy=0.8966, macro_f1=0.5364, f1_0=0.1277, fp=300, fn=28
- `prototype_only` test: accuracy=0.7131, macro_f1=0.4782, f1_0=0.1282, fp=98, fn=174, changed=199, manual_review=23
  all: accuracy=0.7249, macro_f1=0.4953, f1_0=0.1549, fp=244, fn=629, changed=715, manual_review=52
- `crag_correction` test: accuracy=0.8787, macro_f1=0.5414, f1_0=0.1481, fp=108, fn=7, changed=0, manual_review=23
  all: accuracy=0.8954, macro_f1=0.5376, f1_0=0.1309, fp=299, fn=33, changed=6, manual_review=52
- `selfrag_abstain` test: accuracy=0.8787, macro_f1=0.5414, f1_0=0.1481, fp=108, fn=7, changed=0, manual_review=23
  all: accuracy=0.8954, macro_f1=0.5376, f1_0=0.1309, fp=299, fn=33, changed=6, manual_review=52

## Interpretation
The experiments are lightweight reproductions of retrieval correction and conflict-aware reasoning ideas. They should be treated as research probes, not replacements for the calibrated production rules.
A promising paper/patent direction is to use the generated changed/manual-review cases as expert-feedback seeds, then iteratively convert validated cases into rule cards and latent prototypes.
