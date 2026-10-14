# POLARIS 论文中与 R3 相关的内容摘选

来源：`/Users/lihd/Downloads/POLARIS_An_LLM_Agent_for_R.pdf`

## 范围说明

当前项目中的 R3 是 **Characterization of magnetic mineralogy and rock magnetism（磁性矿物学与岩石磁学表征）**，在论文中对应 `magnetic carriers`（磁性载体）证据族，以及围绕可靠性证据、专家规则、证据链、质量评估所写的共享框架内容。

论文第 6、8 页的 `RQ3` 是第三个研究问题 **Database Utility for Downstream gAPWP Reconstruction**，不是当前项目的 R3 判据，不能混为一谈。

## 一、与 R3 直接对应的原文

### 1. R3 被定义为八类文本证据之一

位置：第 5 页，4.2 `Expert-Guided Scientific Constraints`

> Table 1 instantiates K at the Site level. Registered transformations parse numbers, coordinates, tilt state, and polarity while preserving the source expression; non-unique owners, scopes, or variants remain unresolved. **For prose, we define eight evidence families: age, stability tests, coordinates, demagnetization and statistics, magnetic carriers, structural context, polarity groups, and pole comparison. A missed family remains unresolved rather than becoming evidence of absence.**

中文要点：论文把“磁性载体”列为八类专家定义的文本证据之一。系统未检索到该类证据时，状态应保持为“未解决”，不能直接当作论文没有相关证据。这与 R3 工作流的 `UNKNOWN` 原则直接对应。

### 2. Gold benchmark 中包含 magnetic carriers 证据

位置：第 6 页，5.1 `Experimental Setup`

> At the evidence granularity, it contains **4,712 evidence objects over 4,879 anchored quotations from eight families: ages, field tests, coordinates, demagnetization and statistics, magnetic carriers, structural context, polarity groups, and pole comparisons.** Two peer-blind annotation channels ground eligible claims independently, and substantive disagreements are adjudicated.

中文要点：磁性载体证据与其他七类证据一起进入证据级 Gold benchmark。论文只给出八类合计规模，没有单列 magnetic-carriers/R3 的样本数或准确率。

### 3. R3 所属的可靠性证据层是论文要恢复的核心内容

位置：第 2 页，1 `Introduction`

> Taken together, however, they do not provide a general mechanism for retrospectively reconstructing from legacy publications the links among a reported value, the object and result variant to which it belongs, the applied corrections, **the reliability statements that support its use**, and its precise provenance. Consequently, a reusable value may remain disconnected from the scientific context that gives it meaning.

位置：第 3 页，1 `Introduction` 的贡献总结

> We identify and quantify the missing evidence layer between curated paleomagnetic databases and source publications, showing that **critical site-level observations, reliability information, and provenance are frequently lost during data aggregation.**

中文要点：R3 对磁性矿物学、岩石磁学及载磁矿物证据的判断，属于论文所称的 `reliability statements / reliability information`。

## 二、与 R3 实现直接相关的共享方法描述

### 1. 专家规则与可追溯证据链

位置：第 3 页，1 `Introduction`

> POLARIS adopts an expert-in-the-loop design philosophy: at design time, specialists’ reading and assessment logic is externalized as reusable entity definitions, scope constraints, registered transformations, and validation rules. Operating on locator-preserving paper content, the framework organizes reconstruction as retrieval, scientific resolution, and record materialization, thereby creating traceable links between reported observation values and their supporting scientific context while retaining explicit missing, conflicting, and unresolved states. As illustrated in Figure 1, the LLM proposes and organizes semantic interpretations, whereas deterministic operations and expert rules govern value recovery, validation, and materialization.

中文要点：专家阅读与判定逻辑被外化为实体定义、范围约束、确定性转换和验证规则；LLM 负责语义解释，确定性操作和专家规则负责恢复、校验及物化。

### 2. 不替作者选择解释，保留未知、冲突与歧义

位置：第 4 页，3.2 `Computational Task`

> The computational task is to reconstruct this paper-local scientific context, rather than to select a preferred geological interpretation. Given the layout-preserving text and tables of a paper, the task identifies its reported locations, geological units, sites, result variants, poles, measurements, and typed relations; separates alternative processing states; binds each field and relation to the correct owner and applicable scope; and retains the source evidence needed to verify the reconstruction. **Conflicting, ambiguous, missing, and unresolved cases must remain explicit rather than being replaced by a plausible value or default owner.**

中文要点：这段支撑 R3 的严格判定策略，即证据不足时保留 `UNKNOWN`，不根据方法名、矿物名或常识强行给出正负判断。

### 3. LLM 只作语义解释，原值、定位和对象标识由代码锚定

位置：第 4 页，4.1 `Framework Overview`

> POLARIS first resolves site-result tables and then processes eight expert-defined text questions in a bounded agenda. At step t, the LLM receives q_t, E_t, S_t, and K, and returns a quotation, semantic label, and applies-to expression. Code then anchors the quotation, parses the matched source span, and binds its owner. **The LLM never authors reported values, locators, or object identifiers.**

中文要点：R3 输出中的证据引文、证据适用对象和判定标签应相互绑定；模型不能自行编造论文原值、定位或对象 ID。

### 4. 专家知识不是提示词，而是独立版本化规范

位置：第 5 页，4.2 `Expert-Guided Scientific Constraints`

> We encode expert knowledge as a versioned, paper-independent specification K, rather than as prompt hints. Its naming layer uses MagIC field identifiers and historical labels; its semantic layer uses GPMDB parameter definitions and expert paleomagnetic rules [15, 23]. We include no paper-specific values, benchmark labels, or adjudication decisions.

中文要点：R3 的专家知识应是与单篇论文和测试标签无关的冻结规则，不能把测试集标签、裁决结果或论文特定值写入知识库。

### 5. 双重凭据：原文凭据 + 科学凭据

位置：第 5 页，4.3 `Evidence-Grounded Resolution`

> We materialize a claim only when it has two warrants. The source warrant anchors its raw value or quotation in B_i; the scientific warrant verifies its field, owner, scope, and ResultVariant under K and the registered paleomagnetic rules. **A claim lacking either warrant remains ambiguous or unresolved.**

中文要点：R3 证据既要能回到原文，又要满足载体证据的科学规则并正确绑定对象；缺一项都不能当作已确认结论。

### 6. 文本证据的逐字引文、适用范围和证据链

位置：第 5 页，4.3 `Evidence-Grounded Resolution`

> We retrieve candidate blocks and ask the LLM for a **verbatim quotation, semantic label, and the paper’s applies-to expression**. We anchor the quotation, parse only the matched substring, and bind it through a verified Site set, entity, unit, pole, or study. If no unique binding exists, we retain the claim as unresolved.

> For each accepted field or relation, we build an EvidenceChain linking its source elements to the owner, ResultVariant, scope, and registered transformation. It is an inspectable provenance path, not hidden chain-of-thought.

中文要点：这两段直接对应 R3 的原文引文、证据分类、适用对象、来源定位与可审计证据链。

### 7. 停止条件和未完成状态

位置：第 5 页，4.5 `Bounded Agent Reasoning`

> We measure progress from S_t, not from an LLM declaration of completion. We stop when the agenda is empty, two rounds leave the state digest unchanged, or K_max is reached. We select K_max on validation and fix it for testing. **Failed anchors, unbound scopes, and unfinished questions retain explicit outcomes rather than becoming “not reported.”**

中文要点：引文锚定失败、对象范围未绑定或问题未完成时，必须保留明确状态，不能改写成“未报道”。

## 三、正确率与评估内容

### 1. 摘要中的正确率表述

位置：第 1 页，`Abstract`

> Independent expert evaluation on a representative sample validates the reconstructed evidence objects, scientific relationships, and provenance links **over 90% accuracy**.

注意：这是摘要中的总体表述，不是 R3/magnetic-carriers 子任务的单独准确率。正文第 7 页又说明相应独立盲审仍为 pending，见下文第 3.3 至 3.5 节。

### 2. 评测集与指标定义

位置：第 6 页，5.1 `Experimental Setup`

> We construct a single paper-disjoint benchmark over **100 GPMDB-linked publications**, parsed once into a frozen, locator-preserving representation B_i that every evaluated condition receives identically: **1,537 pages, 17,202 blocks, and 383 tables.**

> At the record granularity, it contains **4,131 Sites, 5,229 direction results, and 41,790 field values grounded in 141,262 source elements.** At the evidence granularity, it contains **4,712 evidence objects over 4,879 anchored quotations** from eight families.

> A predicted claim forms a complete chain only when its evidence object or field, scientific owner and scope, ResultVariant where applicable, typed relationship, and supporting provenance are all correct. We report paper-macro complete-chain precision, recall, and F1; **complete-chain F1 is the primary endpoint** because it penalizes both unsupported output and selective abstention. Exact-match Site × ResultVariant F1 is reported separately for the record layer. Decomposed measures include evidence-object accuracy, relationship F1, provenance-link accuracy, object recall, and the verifiable fraction.

中文要点：论文的主指标是完整证据链 F1，不是简单 accuracy；证据对象、对象/范围、结果变体、关系和来源链都正确时才算完整链正确。

### 3. Table 3：摘要所依据的三条 accuracy 轴

位置：第 7 页，Table 3

| Axis | Registered dev | Blind audit | 95% CI |
|---|---:|---:|---:|
| Evidence objects | 0.974 | - | [-, -] |
| Scientific relationships | 0.949 | - | [-, -] |
| Provenance links | 0.927 | - | [-, -] |

表题原文：

> Development estimates for the three abstract-level accuracy axes and the pending independent blind audit. “Registered dev” reports the original conditional development estimate; **these estimates do not substitute for the held-out audit.**

中文要点：开发集上的三项数值分别是 **97.4%、94.9%、92.7%**，但盲审结果和置信区间均未填写。这些是系统总体指标，不是 R3 单项指标。

### 4. 正文对开发集数字的限制

位置：第 7 页，5.2 `RQ1: Reconstruction Accuracy and Baseline Superiority`

> These runs were used for loss attribution and configuration development and **are not held-out evidence**. The legacy evaluator reported a chain rate, the fraction of materialized claims for which all scored links held, rather than the complete-chain F1 defined above. We therefore retain the original metric name and do not relabel it as F1.

> The development estimates used register-specific denominators and did not record the independent audit sample size or confidence intervals; they therefore remain diagnostics.

> The same development analysis reported **object recall 0.201, a verifiable fraction of 0.669, an adjudicated-valid fraction of 0.942, and 55 non-compliant objects among 453 assessed objects (0.121).** These quantities use different registered populations and are not pooled with Table 4.

中文要点：对象召回率为 **20.1%**，可验证比例为 **66.9%**，裁决后有效比例为 **94.2%**，不合规对象为 **55/453 = 12.1%**；这些指标的分母不同，不能相互合并，也不能直接当作最终测试集准确率。

### 5. “超过 90%”声明成立的预设条件

位置：第 7 页，5.2 `Independent expert audit`

> The “over 90%” statement is licensed only if **all three blind-audit point estimates exceed 0.90**; the stronger statement that all axes are above 0.90 requires every simultaneous confidence-interval lower bound to exceed 0.90. If either condition fails, the abstract claim will be revised rather than changing the audit population, matching rule, or denominator after test inspection.

中文要点：只有三条盲审轴的点估计都超过 90%，摘要才能写 `over 90%`。若要声称三条轴在统计上都高于 90%，则每条同步置信区间的下界也必须超过 90%。

### 6. Table 4：各模型和流程的开发集诊断结果

位置：第 8 页，Table 4。原表明确注明：`not held-out test results`。

| Setting | Backbone | Anchor | Evid. | Rel. | Prov. | Chain | Obj. recall | Verif. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Direct LLM | Qwen 4B | 0.641 | 0.542 | 0.330 | 0.347 | 0.060 | 0.059 | 0.176 |
| Direct LLM | Qwen 27B | 0.773 | 0.704 | 0.510 | 0.372 | 0.154 | 0.113 | 0.585 |
| Direct LLM | GPT-5 API | 0.599 | 0.468 | 0.502 | 0.501 | 0.170 | 0.272 | 0.291 |
| Retrieval + one pass | Qwen 4B | - | - | - | - | - | - | - |
| Retrieval + one pass | Qwen 27B | 0.887 | 0.806 | 0.571 | 0.502 | 0.149 | 0.094 | 0.645 |
| Retrieval + one pass | GPT-5 API | 0.907 | 0.649 | 0.554 | 0.723 | 0.180 | 0.200 | 0.444 |
| Generic tool Agent | Qwen 4B | - | - | - | - | - | - | - |
| Generic tool Agent | Qwen 27B | 0.816 | 0.775 | 0.449 | 0.355 | 0.115 | 0.070 | 0.673 |
| Generic tool Agent | GPT-5 API | 0.739 | 0.679 | 0.464 | 0.493 | 0.140 | 0.324 | 0.682 |
| POLARIS (Ours) | Qwen 4B | 1.000 | 0.867 | 0.543 | 0.662 | 0.312 | 0.053 | 0.526 |
| POLARIS (Ours) | Qwen 27B | 1.000 | 0.875 | 0.722 | 0.608 | 0.382 | 0.202 | 0.683 |
| POLARIS (Ours) | GPT-5 API | 1.000 | 0.744 | 0.729 | 0.634 | 0.415 | 0.495 | 0.485 |

指标含义：`Anchor` 为引文锚定；`Evid.` 为证据对象；`Rel.` 为科学关系；`Prov.` 为来源链接；`Chain` 为全部已评分链接均正确的比例；`Obj. recall` 为对象召回率；`Verif.` 为可验证比例。

### 7. Table 5：专家约束和证据合约的消融结果

位置：第 8 页，Table 5，Qwen 27B 开发集、legacy scorer。

| Setting | Anchor | Evid. | Rel. | Prov. | Chain | Obj. recall |
|---|---:|---:|---:|---:|---:|---:|
| POLARIS | 1.000 | 0.865 +/- 0.010 | 0.721 | - | 0.382 | 0.202 |
| w/o expert constraints K | 1.000 | 0.788 | 0.630 | - | 0.337 | 0.230 |
| w/o evidence contract | 0.969 | 0.840 | 0.701 | - | 0.272 | 0.205 |
| legacy single-pass | 1.000 | 0.886 | 0.405 | - | 0.224 | 0.202 |

相关解释原文（第 7 页）：

> Expert constraints target evidence-object and relationship correctness; the evidence contract targets anchoring, provenance, and complete-chain F1; state carry-over targets owner, scope, and ResultVariant binding; and iterative retrieval targets recall and complete-chain F1.

中文要点：移除专家约束后，证据对象指标从 **0.865 降至 0.788**，关系指标从 **0.721 降至 0.630**，Chain 从 **0.382 降至 0.337**；这组消融最能说明 R3 所依赖的专家规则对整体系统的贡献。但它仍是总体消融，不是 R3 单项消融。

## 四、规模、结论与准确率边界

### 1. 部署规模不是准确率

位置：第 7 页，5.3 `Deployment`

> Beyond the benchmark, we apply POLARIS to **3,528 electronically accessible publications** and materialize **85,644 source-grounded site-level records**. **These deployment counts are not accuracy estimates**: accepted, unresolved, and rejected candidates are traced separately, and the application corpus does not score individual-record correctness.

### 2. gAPWP 结果也不是提取准确率

位置：第 8 页，5.4 `RQ3: Database Utility for Downstream gAPWP Reconstruction`

> Thus, PDF-derived records preserved overall path geometry while exposing age-local changes. This demonstrates auditable computational usability within the matched subset, **not greater extraction accuracy, completeness, or tectonic validity**.

### 3. 全量语料没有逐条专家核验

位置：第 8 页，7 `Limitations and Ethical Considerations`

> The 3,528-paper corpus is therefore a bounded application set, and **its full-corpus records are not individually expert-verified.**

## 五、核对结论

1. 论文中能唯一对应当前 R3 的术语是 `magnetic carriers`，而不是 `RQ3`。
2. 论文完整描述了 R3 所依赖的专家约束、原文锚定、适用范围绑定、证据链和未知状态处理，但没有展开 R3 的具体三组判据，也没有列出 R3 支持的方法清单。
3. 论文没有报告 R3/magnetic-carriers 子任务独立正确率。现有正确率均是八类证据和完整流程的总体结果。
4. 摘要写有 `over 90% accuracy`，对应的开发集三项指标是 97.4%、94.9%、92.7%；但正文 Table 3 的独立盲审仍为空，且正文明确说开发集估计不能替代盲审。因此当前稿件中“超过 90%”属于尚待盲审支撑的声明，而不是已经完整呈现的最终结果。
5. 若希望论文明确体现 R3 贡献，建议补充 magnetic-carriers 的样本数、precision/recall/F1 或 accuracy、UNKNOWN/abstention 比例，以及 R3 专项消融或典型案例；目前稿件无法从总体数字中分离出 R3 的实际表现。

## 六、本地项目结果（不属于论文原文）

本地文件 `outputs/api_10_unmapped_test_20260725/r357_accuracy.json` 另有一组 10 条记录的 R3 测试：总计 10 条、正确 7 条、accuracy = 0.70；其中 6 条为 UNKNOWN，4 条作出明确判断且 4 条均正确，decided accuracy = 1.00；混淆计数为 TP=4、TN=3、FP=0、FN=3。

这组数字没有出现在 PDF 中，样本量也很小，不能用于替代论文的 held-out benchmark 或独立盲审结果；如需写入论文，应明确测试集来源、映射策略、UNKNOWN 计分规则和置信区间。
