# Literature-Only R5/R7 Target-Centered Experiment

Generated: 2026-07-11T12:40:44

Scoring uses only target identifiers and original-paper text. The same REFNO never appears in both train and test.

| Indicator | Method | Test accuracy | Macro-F1 | F1(0) | FP | FN | Pred 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| R5 | all_one | 0.7174 | 0.4177 | 0.0000 | 184 | 0 | 0 |
| R5 | legacy_rule | 0.7127 | 0.4671 | 0.1053 | 173 | 14 | 25 |
| R5 | text_calibrated_accuracy | 0.7419 | 0.5534 | 0.2632 | 154 | 14 | 44 |
| R5 | text_calibrated_precision | 0.7419 | 0.5534 | 0.2632 | 154 | 14 | 44 |
| R5 | concept_calibrated_accuracy | 0.7312 | 0.5125 | 0.1860 | 164 | 11 | 31 |
| R5 | concept_calibrated_precision | 0.7312 | 0.5125 | 0.1860 | 164 | 11 | 31 |
| R7 | all_one | 0.8802 | 0.4681 | 0.0000 | 78 | 0 | 0 |
| R7 | legacy_rule | 0.8771 | 0.5322 | 0.1304 | 72 | 8 | 14 |
| R7 | text_calibrated_accuracy | 0.8802 | 0.4681 | 0.0000 | 78 | 0 | 0 |
| R7 | text_calibrated_precision | 0.8802 | 0.4681 | 0.0000 | 78 | 0 | 0 |
| R7 | concept_calibrated_accuracy | 0.8802 | 0.4681 | 0.0000 | 78 | 0 | 0 |
| R7 | concept_calibrated_precision | 0.8802 | 0.4681 | 0.0000 | 78 | 0 | 0 |

Thresholds are calibrated only on the dev REFNO split; test metrics are held out.
