# R3 当前保留结果

日期：2026-07-17

2026-07-22 更新：用户授权使用历史 R3 标签后，全量稳定组合已训练为最终生产
模型并成为完整 R1-R7 入口的默认 R3 流程。最终模型在全部 2,751 条开发记录上
拟合；其 92.33% 训练集重代入不可报告，正式内部开发口径仍为本文的 grouped OOF
2,316/2,751（84.19%）。

## 结论

当前单目标正式内部 OOF 方法是 V3，全覆盖正确率为 **785/872（90.02%）**。
当前全量稳定组合为 **2,316/2,751（84.19%）**。50/50 叠加多目标 GREP 后的
探索最高值为 **2,325/2,751（84.51%）**，但没有通过稳定晋升条件。

这些结果均不使用 PMAGRESULT、当前行 ROCKMAG、已有预测或评测结果作为模型
输入。它们会使用其他训练折的 R3 标签拟合监督模型，因此属于按 REFNO 隔离的
内部 OOF 评测，不是严格无标签评分，也不是独立外部验证。

## 结果口径

| 范围与方法 | 正确数/总数 | Accuracy | 状态 |
| --- | ---: | ---: | --- |
| 单目标 V3 | 785/872 | 90.02% | 当前单目标结果 |
| 全量 scope + V3，不叠加 GREP | 2,316/2,751 | 84.19% | 当前稳定全量组合 |
| 全量 scope + V3、GREP 50/50 | 2,325/2,751 | 84.51% | 探索最高，不晋升 |
| R2E/GREP 成对排序 | 2,300/2,751 | 83.61% | 前序组件 |

全量范围包含 1,478 篇有本地 PDF 的论文。无本地 PDF 的 422 个标签目标跳过且
不计入分母；所有纳入的 PDF 均有非空解析文本。

## 单目标 V3

V3 以 V2 原文知识融合概率为主体，并加入小权重的完整论文词级 TF-IDF 专家。
五个外层折以 REFNO 隔离；每篇论文只对应一个目标，因此不受多目标证据归属
歧义干扰。混淆矩阵为 TN/FP/FN/TP = 244/51/36/541，balanced accuracy 为
88.24%，macro-F1 为 88.71%。

三部分基础概率采用固定比例：

| 组成 | 权重 |
| --- | ---: |
| domain mean4 | 0.4526315789 |
| query PCA64 logistic | 0.2842105263 |
| front-document word logistic | 0.2631578947 |

全局网格在同一批 872 个答案上得到的最高值为 786/872，只能作为开发上限。
严格嵌套选权在外层测试标签不可见时得到 777/872，比固定比例少 8 个正确目标；
配对 REFNO bootstrap 的差值区间也确认该退步。因此固定比例不变。

机器产物：

- `tmp/r3_fulltext_fusion_single872_v3/r3_fulltext_fusion_single872_grouped_oof_v3.npz`
- `tmp/r3_fulltext_fusion_single872_v3/r3_fulltext_fusion_single872_grouped_oof_report_v3.json`
- `tmp/r3_single872_base_weight_nested_oof_v1/r3_single872_base_weight_nested_oof_report_v1.json`

## 全量组合

全量 evaluator 对单目标行使用 V3；多目标行使用已冻结的 scope 模型。该无额外
GREP 组合得到 2,316 个正确目标，是当前稳定版本。

将多目标 scope 与同折 GREP 概率等权融合后净修复 9 行，达到 2,325 个正确目标。
但五个多目标外层折的净修复为 `-3/+12/+1/-7/+6`，两个折发生回退；按 REFNO
聚类的 10,000 次 bootstrap，其总体 accuracy 增量 95% 区间为
`-0.51` 到 `+1.35` 个百分点，跨过 0。因此 84.51% 保留为探索结果，不替代
84.19% 的稳定组合。

机器产物：

- `tmp/r3_scope_fusion_all_oof_v1/r3_scope_fusion_all_grouped_oof_v1.npz`
- `tmp/r3_scope_fusion_all_oof_v1/r3_scope_fusion_all_grouped_oof_report_v1.json`
- `tmp/r3_grep_domain_v4folds_v1/`

## 复核入口

```bash
.venv/bin/python3 scripts/evaluate_r3_fulltext_fusion_single872_oof_v3.py
.venv/bin/python3 scripts/evaluate_r3_scope_fusion_all_oof_v1.py
.venv/bin/python3 scripts/experiment_r3_single872_base_weight_nested_oof_v1.py
```

重新运行会覆盖同名产物。项目清理验收使用独立 NPZ 重算和冻结哈希检查，以避免
仅凭报告文本确认结果。
