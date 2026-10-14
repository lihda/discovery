# R3 MinerU 质量门控解析实验

日期：2026-07-11  
实验性质：训练分区 post-hoc 开发实验，不是新的独立留出集验证

## 1. 目的与边界

本实验验证两个问题：

1. PDF 文本层缺失是否会阻止 R3 从原文恢复 ROCKMAG 式证据；
2. MinerU 是否可以作为无标签、可审计的条件式解析回退，而不是替换全部 PDF 解析结果。

MinerU 运行阶段只读取 `REFNO/RESULTNO/ROCKNAME` 身份清单、现有页缓存和本地 PDF。它不读取 PMAGRESULT、R3 标签、旧评测或旧预测。标签只在预测落盘后用于本文的 post-hoc 聚合分析。

## 2. 实现

- 适配器：`scripts/build_r3_mineru_page_cache.py`；
- 可选依赖：`requirements-mineru.txt`；
- MinerU：3.4.4，pipeline OCR，公式识别关闭，表格识别开启；
- 输出优先读取 MinerU `content_list.json`，回退到 Markdown；
- 保留页号、正文、图题、表格文本、页哈希和文档哈希；
- patch 模式只处理显式 REFNO 或由缓存质量标志选出的本地 PDF；
- 非空 base cache 通过页数、字符数、风险标志、乱码量和领域关键词覆盖进行确定性质量仲裁；
- 缺少可靠物理页边界的 content-list/Markdown 不进入可检索页缓存，避免把全文伪装成同一页；
- 非空 base cache 中已经存在的 R3 核心概念必须被候选保留，字符数增长不能覆盖 IRM、载体或剩磁联系的丢失；
- 只有 base 为空、强风险被消除、内容显著增加或多个质量维度共同改善时才替换；
- 所有选择、替换、保留原因和运行边界写入 summary。

MinerU 3.4.4 的 pipeline OCR 会导入未声明的 `six` 依赖，因此隔离环境显式固定 `six==1.17.0`。

## 3. 解析结果

训练分区包含 30 篇有本地 PDF 的硬解析风险文献：21 篇缓存零文本，9 篇已有文本但存在强风险。

- 21/21 篇零文本 PDF 均恢复出非空内容，共新增 231 个有效页；
- 9 篇强风险 PDF 中，质量仲裁替换 8 篇，保留原解析 1 篇；
- 被保留文献的原缓存有 23 页、62,114 字符，MinerU 候选构成内容回退，因此没有覆盖；
- 最终页缓存仍严格包含 1,292 个 REFNO 和 19,435 个有效页；
- 最终缓存 SHA-256：`f8ec25c1148c04112ccfc75076743803417effbb833834a136c714dff4e146d7`。

首个实测样本 `REFNO=3244` 原缓存为 0 页、0 字符。MinerU 从 9 页扫描 PDF 中恢复 7 个有效正文页、20,276 字符和 1 个表格，提取出 SIRM acquisition/decay、coercivity、pyrrhotite、magnetite、titanomagnetite、hematite 和 domain-state 描述。同代码对照中该记录由 `R3=0/UNKNOWN` 转为 `R3=1/POSITIVE`，正式三段式链和 ROCKMAG 代理链均满足。

## 4. 训练期结果

所有版本均使用相同冻结 artifact、30/12 检索限额和 source-only 运行边界。

| 版本 | Accuracy | Coverage | Known accuracy | Balanced accuracy | Macro-F1 | TN/FP/FN/TP |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 安全身份基线，原缓存 | 58.5524% | 68.5402% | 64.6018% | 64.8048% | 57.9499% | 872/78/947/576 |
| 安全身份 + 全部硬风险 MinerU，句级绑定 | 58.7141% | 70.0768% | 64.5124% | 64.8767% | 58.1544% | 869/81/940/583 |
| 安全身份 + MinerU + 独占段落/表格行作用域 | 58.9163% | 70.0768% | 64.8009% | 65.0210% | 58.3866% | 868/82/934/589 |

“安全身份”表示：同一 `REFNO + canonical ROCKNAME` 对应多个 `RESULTNO`、而三列身份清单不能从原文区分其磁化分量时，行级判决必须 `ABSTAIN`。早期实验曾把这些行无条件共享为同一个文章-目标结论，得到 64.3348% accuracy；复核发现至少 11 个同名组的历史 R3 值彼此相反，因此该增益混入了无法由运行时输入证明的分量映射，现已撤回，不作为效果主张。

在安全身份基线上，MinerU 带来 7 个新增 TP 和 3 个新增 FP，净增加 4 个正确结果，coverage 增加 1.5366 个百分点。随后加入“同一有界段落或表格行内出现当前目标、且不出现任何 sibling target”的作用域规则，7 行由 0 变为 1，其中 6 行历史标签为 1、1 行为 0，净增加 5 个正确结果；作用域消融报告为 `tmp/r3_scope_ablation_training_posthoc.json`，SHA-256 为 `ec37c33b9f3c05c109cc0d60758104d3b379e07f88444b871abd083831dcc2b8`。最终预测 SHA-256 为 `6173d22a7da16e3d7245dea735d97b2ac803acfa90b4d09ef87125742159fd97`。

作为非独立的回归检查，当前作用域代码还重跑了项目中已经评测过的 700 行固定留出分区。700 行的 `R3/OUTCOME/policy_decision` 与既有冻结预测逐行完全相同，accuracy 仍为 59.00%、coverage 为 65.1429%、known accuracy 为 60.0877%。该检查只能说明本次改动没有在已知留出集上造成可见退化，不能被重新解释为新的独立验证或作用域增益。

3 个新增 FP 的 OCR 证据分别包含 `IRM saturation + Curie temperature`、`hysteresis + multidomain magnetite`、`reflected-light microscopy + titanomagnetite`。这些内容按上传 R3 criterion 属于实质性载体表征证据，说明其中至少部分是“原文专家规则与历史评价不一致”，不宜通过删除真实证据来追逐标签准确率。

## 5. 结论与下一步

MinerU 可以解决扫描件没有文本层的问题，但单靠解析器无法把 R3 推到 90%。当前最终版本仍有 934 个 FN：154 行来源不可用，252 行属于无法从三列身份映射到具体磁化分量的身份歧义，528 行在来源可用且身份明确时仍未形成强证据链；后者有 469 行至少出现一次 `cross_target_or_unbound` 拒绝。最大后续方向仍是多目标文献的目标绑定、论文内别名和证据连接，而不是继续全量 OCR。

早期对 467 个 `cross_target_or_unbound` FN 的 post-hoc 分析显示：202 行需要恢复论文内别名、缩写或表格行标签；264 行存在真实 sibling target 竞争。当前已实现第一阶段：每条命中携带 `scope_id/scope_kind`，只有独占目标的有界段落或表格行能够升级为局部绑定；若证据依赖该升级，方法与载体/诊断还必须具有兼容 `scope_id`。不能继续把多目标约束简单放宽为同页或邻页。

本实验支持的专利实施特征包括：基于无标签质量状态的多解析器路由、解析候选核心概念保真仲裁、可靠页边界门控、段落/表格行作用域及 sibling target 排除、正式判据与实践代理双轨推理，以及对原文专家结论和历史评价冲突的保留与审计。由于本文结果来自训练分区 post-hoc 开发实验，这些特征的精度效果仍需新的冻结外部集验证。

## 6. 2026-07-12 页级融合追加实验

整篇替换无法处理“MinerU 只改善部分页、同时损失另一些页核心概念”的情况，因此新增 `--page-fusion`。它按物理页号对齐 base 与 MinerU，同一页只保留一个完整来源，禁止跨解析器拼接文字。候选页须满足字符保留、block confidence、低置信 block 比例和 R3 核心概念保真，并在段落边界、表格行、乱码或内容量上形成可审计改善。

从 1,026 篇无标签检测为 flattened-layout 且有本地 PDF 的训练论文中，按 PDF 文件大小选择 3 篇小样本进行 `pipeline/auto` 实测。融合器采用 15 个 MinerU 页、保留 9 个 base 页；其中 2 个候选页因丢失 R3 概念被明确拒绝。3 篇文献的段落边界总数由 0 增至 90，表格行由 0 增至 114，flattened-layout 状态全部解除。

在相同冻结 artifact 和 `exclusive_scope_plus_discourse` 策略下完整重跑训练分区，页级融合新增 1 条正式三段式规范链，但二元 ROCKMAG 代理没有任何行变化，accuracy 仍为 63.0813%。该结果进一步说明 MinerU 能改善来源结构和正式证据链，却不是当前二元标签准确率的主要瓶颈。
