# R3 保守融合与 ROCKMAG episode 扩展探索

日期：2026-07-13

## 总结

本轮沿“保守软融合、ROCKMAG 方法-结果 episode、目标归属门控”继续探索，并用
两个在预测前封存的新 300 行队列验证。结果证明 hard router 确有跨论文失稳，
但融合和现有 episode 特征不足以把新论文稳定提升到 90%。

| 队列 | 论文 | 正式策略 | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | --- | ---: | ---: | ---: |
| canonical exact300 | 258 | hard router + paper consistency | 90.00% | 89.06% | 88.47% |
| random300 extension | 263 | hard router + paper consistency | 84.67% | 82.18% | 81.83% |
| sealed third300 | 263 | 90% mean + 10% hard router | 87.00% | 85.41% | 85.16% |
| sealed fourth300 | 262 | fixed six-expert mean | 82.33% | 78.90% | 78.47% |

第三批的预声明诊断中，`episode_mean7` 为 87.67%，hard router 为 85.67%；第四批
中位数为 83.67%，`episode_mean7` 为 82.67%。这些诊断均未替换冻结的主结果。

## 封存边界

- third300 与 exact300/random300 的 REFNO 零重叠；300 行对应 263 篇论文，相关
  452 行从所有拟合中排除。
- fourth300 与前三批的 REFNO 均零重叠；300 行对应 262 篇论文，相关 338 行从
  所有拟合中排除。
- 两轮都先写 candidate freeze 和不含答案的 prediction receipt，再读取测试标签。
- 预测特征不读取 PMAGRESULT、逐行 ROCKMAG、旧 OOF 或旧评测。
- 两轮 300/300 均有本地 PDF 和非空页面缓存。

第三批队列 SHA-256：
`a275dd97b234a746ce3ba564157de3c7cf9766f37a445f4a8ae1b30307d2e3e1`

第四批队列 SHA-256：
`1ad4d2a61fc8fbb09c67730d1b2c5a433abca3f5765211469615cd5a4c0cad5d`

## 已尝试方向

### 保守软融合

random300 上固定均值为 88.33%，hard router 为 84.67%，说明逐样本选择一个专家
会放大路由误差。基于 exact300/random300 开发结果，third300 预先冻结
`0.9 * mean6 + 0.1 * hard_router`，得到 87.00%；其固定均值为 87.33%。方向正确，
但增益上限有限。

### 字段签名与全证据专家

现有 record-free signature 覆盖 OP、IRM、Js-T、SUSC-T、HYST、载磁矿物和
method-carrier/diagnostic episode。将该专家加入均值后，third300 从 87.33%变为
87.67%，fourth300 从 82.33%变为82.67%。单独 signature HGB 分别为83.67%和
82.00%，不能独立承担字段重建。

### OCR 鲁棒全文模型

对全文字符 3-5 gram、词 1-3 gram、NBSVM 和 calibrated LinearSVC 进行了
ROCKMAG-paper presence 重建。third300 的行级 R3 仅约76%-81%。这说明原文中
出现岩磁词汇并不等同于数据库对当前 RESULTNO 填写 ROCKMAG。

### ROCKMAG 监督重建头

在预测冻结后的开发诊断中，使用允许训练论文的 ROCKMAG 是否非空作为离线监督，
运行时仍只输入原文数值特征。该头在 third300 对字段状态最高约89.67%，对 R3
约87.67%；fourth300 对字段状态最高约86%，对 R3 约83.67%。阈值扫描也未达到
90%，因此瓶颈不是分类阈值。

## 核心原因

预测冻结后，ROCKMAG 是否非空与 R3 的关系为：

| 队列 | ROCKMAG presence -> R3 accuracy | TN/FP/FN/TP |
| --- | ---: | --- |
| random300 | 94.33% | 86/10/7/197 |
| third300 | 98.00% | 98/4/2/196 |
| fourth300 | 96.33% | 83/9/2/206 |

正式模型错误中，与 ROCKMAG presence 重建方向一致的错误分别为：

- random300：34/46；
- third300：34/39；
- fourth300：46/53。

这些错误不是大规模 PDF 空解析。第四批正确/错误文本字符数中位数约49,083/
47,784，所有页面均非空。原文中经常能找到有效岩磁科学证据，但历史表可能没有
给该 RESULTNO 填 ROCKMAG；反过来，字段可能有值而 identity-only 输入不能指出
它对应原文中的哪个 component、sample 或 table row。

因此当前约束下存在信息不足：运行时只有 `REFNO/RESULTNO/ROCKNAME`，但
`RESULTNO` 不能作为数值身份特征，也没有 component/sample locator。对于同论文
多行记录，原文科学证据和历史行级字段赋值不是可唯一恢复的映射。

## 专利化建议

最实用的专利主线应从“强制复刻历史 R3 到 90%”调整为：

1. 原文科学证据头：输出方法、诊断、载磁矿物、剩磁意义和来源片段；
2. ROCKMAG episode 重建头：输出可报告字段片段及目标绑定置信度；
3. 历史策略头：输出与数据库赋值习惯的兼容概率；
4. 知识冲突头：当科学证据与历史字段策略不一致时显式分类，不静默覆盖；
5. 身份不足检测：缺少 source-derived component/sample locator 时输出
   `IDENTITY_AMBIGUOUS`，进入人工复核；
6. 论文级一致性只作用于 historical-policy，不传播科学事实。

若仍要求行级 historical-label accuracy >= 90%，需要新增一种不属于 PMAGRESULT
的可审计输入：从原文抽取并冻结的 `RESULTNO -> component/sample/table-row` 定位清单。
继续换分类器、扩普通关键词或调整阈值，现有四批结果不支持其能稳定解决问题。

## 产物

- `tmp/r3_third300_validation_v1/cohort_manifest.json`
- `tmp/r3_third300_validation_v1/robust_holdout/report.json`
- `tmp/r3_fourth300_validation_v1/cohort_manifest.json`
- `tmp/r3_fourth300_validation_v1/mean_holdout/report.json`
- `scripts/freeze_r3_third300_validation_cohort.py`
- `scripts/experiment_r3_third300_robust_holdout.py`
- `scripts/freeze_r3_fourth300_validation_cohort.py`
- `scripts/experiment_r3_fourth300_mean_holdout.py`
