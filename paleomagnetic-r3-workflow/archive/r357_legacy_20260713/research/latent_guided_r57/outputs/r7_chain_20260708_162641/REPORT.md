# R7 Evidence-Chain Experiment Report

Generated: 2026-07-08T16:27:50

## Baseline
- R7 baseline: accuracy=0.8966, macro_f1=0.5364, f1_0=0.1277, fp=300, fn=28

## Paper-Inspired Variants
- `graph_chain` test: accuracy=0.8428, macro_f1=0.5494, f1_0=0.1858, fp=101, fn=48, changed=48, review=0
  all: accuracy=0.8550, macro_f1=0.5297, f1_0=0.1386, fp=287, fn=173, changed=158, review=0
- `crag_graph` test: accuracy=0.8397, macro_f1=0.5335, f1_0=0.1556, fp=104, fn=48, changed=51, review=0
  all: accuracy=0.8544, macro_f1=0.5244, f1_0=0.1283, fp=290, fn=172, changed=162, review=0
- `selfrag_graph` test: accuracy=0.8544, macro_f1=0.5343, f1_0=0.1481, fp=106, fn=32, changed=27, review=625
  all: accuracy=0.8749, macro_f1=0.5281, f1_0=0.1236, fp=296, fn=101, changed=77, review=2061

## Interpretation
This run operationalizes GraphRAG as explicit R7 evidence nodes/edges, CRAG as a retrieval-quality gate, and Self-RAG as reflection flags for retrieve/relevant/supported/conflict.
If test-set f1_0 improves while accuracy remains acceptable, the evidence-chain idea is a candidate method. If it only improves all-set metrics, expert-reviewed graph nodes should be used to build stronger rule cards before claiming generalization.
