# R3 Source Evidence vs Historical Label Conflict Audit

## Boundary

- Candidate SHA-256: `5927af212f6cbca9fa79ac72e033e6b402afc28430e08cb4c646fbca6b833ca7`
- Candidates were frozen before training R3 cells were opened.
- Runtime evidence inputs were identity fields, original-paper text, frozen taxonomy/positive KB, criterion spec, and the frozen component manifest.
- PMAGRESULT, ROCKMAG source cells, held-out labels, old predictions, and old evaluations were not read.

## Empirical Flags

| Audit class | Rows | Denominator | Rate |
|---|---:|---:|---:|
| `high_confidence_proxy_support_but_gold_zero` | 7 | 2172 | 0.3223% |
| `high_confidence_complete_criterion_chain_but_gold_zero` | 3 | 2172 | 0.1381% |
| `gold_one_but_no_frozen_proxy_candidate` | 1209 | 1368 | 88.3772% |
| `gold_one_but_no_high_confidence_proxy` | 1317 | 1368 | 96.2719% |
| `gold_one_but_incomplete_frozen_criterion_chain` | 1343 | 1368 | 98.1725% |

These are empirical audit flags, not unavoidable errors. A support/gold-zero row still requires manual confirmation; a gold-one/no-candidate row may be an extraction miss.

## Representation-Specific Bound

- Mixed exact-representation groups: 8
- Minimum errors for deterministic classifiers restricted to this exact frozen representation: 9 / 2172 (0.4144%)
- Corresponding representation-specific accuracy ceiling: 99.5856%
- This is not a ceiling for all source-only systems. Richer source parsing or an original-text component locator can split currently identical rows.

## Relation To The 90% Target

- Maximum errors at 90% accuracy: 217
- Exact frozen-representation minimum errors: 9
- Remaining error budget after that bound: 208
- Exact representation conflicts alone prevent 90%: false
- 1209 of 1368 gold-positive rows have no candidate under the conservative frozen extractor.

The exact mixed-label lower bound does not explain failure to reach 90%. The much larger no-candidate count identifies retrieval and target/component attribution as the observed bottleneck, but it remains an extractor-recall statistic rather than an all-source impossibility bound.

## Criterion Logic

A manually confirmed complete source chain with gold R3=0 conflicts with the uploaded criterion's positive condition. Gold R3=1 without a recovered complete chain is unresolved rather than a logical contradiction because the extractor may have missed evidence and historical practice may use a weaker ROCKMAG-style proxy.

## Review Examples

| Class | REFNO | RESULTNO | Target | Gold | Evidence excerpt |
|---|---:|---:|---|---:|---|
| `high_proxy_support_gold_zero` | 1007 | 3556 | Great Dyke | 0 | Palaeomagnetic Measurements on the Great Dyke, S. Rhodesia 219 A number of thermo-magnetic curves were made on a modified Chevallier balance, built by Parry (1957), in order to determine the Curie point of the magnetic phase. Specimens were mounted on a torsion arm lying between the poles of an electromagnet and movement of this arm reflects changes in the magnetic properties of the specimen as it is heated, the bala |
| `high_proxy_support_gold_zero` | 3106 | 8126 | Sinyai Metadolerite | 0 | Most samples show two components of magnetization (figure 3). The first, low-temperature component is removed by 500°C. The declinations for this component show considerable scatter between northwest and northeast with consistently shallow inclinations. We attribute this direction to a viscous overprint of recent origin superimposed on the ancient remanence. The second, hightemperature vector is directed to the north |
| `high_proxy_support_gold_zero` | 3166 | 8230 | Akasaki Group | 0 | 580øC, while AF demagnetization in peak fields up to 100 mT was not sufficient to demagnetize the ChRMs (Figure 4c). These demagnetization results suggest that hematite is the principal magnetic carrier. Pilot specimens from the other three sites showed unstable behavior of NRM in the demagnetization levels above about 400øC and/or 20 mT, although some of the specimens showed linear trends of vector endpoints which d |
| `high_proxy_support_gold_zero` | 3187 | 8272 | Teall Nunatak | 0 | The IRM measurements showed that saturation is always reached at ﬁeld values between 0.1 and 0.2 T, and the remanent coercive force in the specimens from Mount Keinath (16–23 mT) is lower than in those from Teall Nunatak (50–70 mT, 28 mT at site TN3). These results point to low-Ti titanomagnet- ite as the main ferromagnetic mineral in both plutons, and suggest that the grains are mainly multidomain in the Mount Keina |
| `high_proxy_support_gold_zero` | 3332 | 8533 | Elgee Formation | 0 | No pervasive regional overprint was reported in either study. a small low-temperature component, most probably due to recent weathering, is visible in many specimens, it is prominentThe target of this study, the Elgee Formation, consists of purple siltstone and silty shale with minor quartz sandstone. enough to give a site mean direction for two of the seven sites only (Table 1). The unblocking temperature of the cha |
| `high_proxy_support_gold_zero` | 3434 | 8740 | Lawn Hill Formation | 0 | AGICO CS-3 Kappabridge. Normally two or more specimens were demagnetised per sample.Both thermomagnetic analysis (Figure 3a) and RESULTS remanence intensity behaviour during thermal demag- Four formations were sampled in the McArthur Basin and netisation (Figure 3b) indicated the presence in the nine formations in the Lawn Hill Platform,yielding a total volcanics of a mineral with a Curie temperature of ≤580°C, 9002  |
| `high_proxy_support_gold_zero` | 3986 | 9745 | Chilgok Formation | 0 | 3.2. Results One of the main objectives of this study is to reveal the characteristic magnetization in the red beds of the Chilgok Formation. Typical thermomagnetic results in this study suggest magnetite and hematite to be the predominant magnetic minerals in the samples. The unblocking temperatures of about 570° –575 °C and 675° –680 °C conﬁrm the presence of magnetite and hematite, respectively. The irreversible p |
| `complete_criterion_chain_gold_zero` | 3106 | 8126 | Sinyai Metadolerite | 0 | Most samples show two components of magnetization (figure 3). The first, low-temperature component is removed by 500°C. The declinations for this component show considerable scatter between northwest and northeast with consistently shallow inclinations. We attribute this direction to a viscous overprint of recent origin superimposed on the ancient remanence. The second, hightemperature vector is directed to the north |
| `complete_criterion_chain_gold_zero` | 3166 | 8230 | Akasaki Group | 0 | 580øC, while AF demagnetization in peak fields up to 100 mT was not sufficient to demagnetize the ChRMs (Figure 4c). These demagnetization results suggest that hematite is the principal magnetic carrier. Pilot specimens from the other three sites showed unstable behavior of NRM in the demagnetization levels above about 400øC and/or 20 mT, although some of the specimens showed linear trends of vector endpoints which d |
| `complete_criterion_chain_gold_zero` | 3332 | 8533 | Elgee Formation | 0 | No pervasive regional overprint was reported in either study. a small low-temperature component, most probably due to recent weathering, is visible in many specimens, it is prominentThe target of this study, the Elgee Formation, consists of purple siltstone and silty shale with minor quartz sandstone. enough to give a site mean direction for two of the seven sites only (Table 1). The unblocking temperature of the cha |
| `gold_one_no_frozen_candidate` | 1089 | 2146 | Gowganda and Chibougamau Formations | 1 |  |
| `gold_one_no_frozen_candidate` | 1142 | 4 | Ingleside Formation | 1 |  |
| `gold_one_no_frozen_candidate` | 1145 | 8 | Proterozoic granites | 1 | University of Nairobi, PO Box 30197, Nairobi, Kenya Received 1978 December 1; in original form 1978 April 18 Summary. This paper reports a palaeomagnetic study of the lower Proterozoic granites, exposed in the western part of Kenya and dated at 2530 k 50 Ma (ARb = 1.39 x lO-"yr-'). Ten sites out of 13 have yielded stable palaeo- magnetic directions which are believed to be primary. Polished section and thermomagnetic |
| `gold_one_no_frozen_candidate` | 1147 | 16 | Douglas Peninsula Formation, Pethei Group | 1 | Douglas Peninsula Formation The Douglas Peninsula Formation, the lowest formation of the Pethei Group, occurs throughout the aulacogen and varies in thickness from 16-33 m (Hoffman 1968). The type section was sampled. It is 25 m thick and it is autochthonous (Hoffman er al. 1977). The rock is a thin-bedded, fissile, red marlstone, that is composed of small lenticles of pink or brown limestone in dark reddish-brown mu |
| `gold_one_no_frozen_candidate` | 1147 | 17 | Douglas Peninsula Formation, Pethei Group | 1 | Douglas Peninsula Formation The Douglas Peninsula Formation, the lowest formation of the Pethei Group, occurs throughout the aulacogen and varies in thickness from 16-33 m (Hoffman 1968). The type section was sampled. It is 25 m thick and it is autochthonous (Hoffman er al. 1977). The rock is a thin-bedded, fissile, red marlstone, that is composed of small lenticles of pink or brown limestone in dark reddish-brown mu |
| `gold_one_no_frozen_candidate` | 1147 | 19 | Peninsular Sill | 1 |  |
| `gold_one_no_frozen_candidate` | 1147 | 14 | Seton Formation, Kahochella Group | 1 |  |
| `gold_one_no_frozen_candidate` | 1147 | 15 | Seton Formation, Kahochella Group | 1 |  |
| `gold_one_no_frozen_candidate` | 1147 | 18 | Takiyuak Formation | 1 |  |
| `gold_one_no_frozen_candidate` | 1148 | 20 | Quartz-magnetites | 1 | This suggests that the primary magnetite of the quartz-magnetites had been remobilized and recrystallized during the second phase, which must be very intensive. The third event was characterized by the absence of any visible alteration and this is positively due to its lesser intensity. |

## Near-Duplicate Opposite Labels

Retained review pairs: 3. These are heuristic review pairs, not mathematical duplicates.

