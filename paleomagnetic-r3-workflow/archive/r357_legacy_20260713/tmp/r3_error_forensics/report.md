# R3 PDF-only Error Forensics

Development-only analysis. Candidate decisions were frozen before labels or prior predictions were opened.

## Full-coverage results

| Strategy | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| direct_strict | 65.56% | 663 | 141 | 607 | 761 |
| canonical_target_propagation | 65.56% | 663 | 141 | 607 | 761 |
| single_target_propagation | 65.56% | 663 | 141 | 607 | 761 |
| explicit_shared_pair_propagation | 73.43% | 573 | 231 | 346 | 1022 |
| explicit_shared_episode_propagation | 73.25% | 568 | 236 | 345 | 1023 |
| explicit_shared_strict_trigger_propagation | 66.71% | 631 | 173 | 550 | 818 |
| criterion_ordered_pair_propagation | 73.43% | 573 | 231 | 346 | 1022 |
| criterion_ordered_episode_propagation | 73.25% | 568 | 236 | 345 | 1023 |
| unsafe_any_paper_propagation | 79.97% | 432 | 372 | 63 | 1305 |
| frozen_latest_target_section | 65.56% | 663 | 141 | 607 | 761 |

## Main bottleneck

Latest false negatives: 607; latest false positives: 141.

FN structure: `{"homogeneous|multi_target": 501, "homogeneous|single_target": 31, "mixed|multi_target": 72, "mixed|single_target": 3}`

FN primary gate: `{"accepted_method_missing": 35, "cross_target_or_unbound_rejection": 548, "no_eligible_evidence": 24}`

FP decision paths: `{"ACCEPTED_DISTILLED_ROCKMAG_METHOD": 38, "ACCEPTED_METHOD_WITH_CARRIER_IDENTIFICATION": 43, "ACCEPTED_METHOD_WITH_DIAGNOSTIC_RESULT": 28, "ACCEPTED_SELF_AUTHENTICATING_DIAGNOSTIC_RESULT": 1, "MICROSCOPY_WITH_CARRIER_OR_ALTERATION": 31}`

## Oracle ceilings

Canonical target-equivalence ceiling: 99.72%.
Paper-uniform ceiling: 96.82%.
Permitted single/shared grouping ceiling: 98.71%.

The ceilings use labels only after candidate freeze and are not deployable classifiers.
