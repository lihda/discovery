# R7 Evidence-Chain Experiment Report

Generated: 2026-07-08T16:35:04

## Baseline
- R7 baseline: accuracy=0.8966, macro_f1=0.5364, f1_0=0.1277, fp=300, fn=28

## Paper-Inspired Variants
- `graph_chain` test: accuracy=0.8428, macro_f1=0.5494, f1_0=0.1858, fp=101, fn=48, changed=48, review=0
  all: accuracy=0.8560, macro_f1=0.5320, f1_0=0.1426, fp=286, fn=171, changed=157, review=0
- `targeted_graph_chain` test: accuracy=0.8354, macro_f1=0.6454, f1_0=0.3858, fp=69, fn=87, changed=119, review=0
  all: accuracy=0.8383, macro_f1=0.5927, f1_0=0.2764, fp=226, fn=287, changed=333, review=1
- `crag_graph` test: accuracy=0.8397, macro_f1=0.5335, f1_0=0.1556, fp=104, fn=48, changed=51, review=0
  all: accuracy=0.8553, macro_f1=0.5267, f1_0=0.1323, fp=289, fn=170, changed=161, review=0
- `targeted_crag_graph` test: accuracy=0.8386, macro_f1=0.6487, f1_0=0.3904, fp=69, fn=84, changed=122, review=0
  all: accuracy=0.8415, macro_f1=0.5946, f1_0=0.2783, fp=227, fn=276, changed=345, review=1
- `targeted_conservative_crag` test: accuracy=0.8681, macro_f1=0.6025, f1_0=0.2775, fp=94, fn=31, changed=56, review=0
  all: accuracy=0.8743, macro_f1=0.5612, f1_0=0.1907, fp=277, fn=122, changed=175, review=0
- `selfrag_graph` test: accuracy=0.8544, macro_f1=0.5343, f1_0=0.1481, fp=106, fn=32, changed=27, review=670
  all: accuracy=0.8783, macro_f1=0.5287, f1_0=0.1227, fp=297, fn=89, changed=64, review=2200

## Interpretation
This run operationalizes GraphRAG as explicit R7 evidence nodes/edges, CRAG as a retrieval-quality gate, and Self-RAG as reflection flags for retrieve/relevant/supported/conflict.
If test-set f1_0 improves while accuracy remains acceptable, the evidence-chain idea is a candidate method. If it only improves all-set metrics, expert-reviewed graph nodes should be used to build stronger rule cards before claiming generalization.
