# R3 exact300 规范 90% 结果说明

日期：2026-07-13

## 结论

在固定 exact300、REFNO 分组、300 行全覆盖的评测上，R3 达到
**270/300 = 90.00%**。

- 混淆矩阵：TN/FP/FN/TP = `91/19/11/179`
- 五折准确率：`91.67 / 95.00 / 81.67 / 90.00 / 91.67%`
- Macro-F1：`89.06%`
- Balanced accuracy：`88.47%`
- exact300 的 300 行均有本地 PDF；辅助候选池中无本地 PDF 的 301 行按既定规则跳过，不进入训练池或评测分母

正式报告：
`tmp/r3_exact_300_resultno_v2/paper_consistency_v2/report.json`

## 方法

最终流程由四层组成：

1. 从原文页缓存及冻结专家知识生成 curation、own-study、目标归属和
   source-derived component bridge 特征，不在线读取 PMAGRESULT/ROCKMAG。
2. 在每个外折内完整排除测试 REFNO 的全部辅助训练行，使用六个异构专家模型。
3. 仅用外折训练论文的 REFNO 隔离 OOF，训练 historical-policy router；该层把
   “原文科学语义”与“历史 R3 赋值策略”分开建模。
4. 对同论文多行结果实施 paper-consistency 校准。所有平滑候选先在不知道外折
   标签的情况下比较二元判决；若判决完全一致，直接选择 identity。仅当判决
   不一致时，才使用完整二级 REFNO cross-fit 的训练 OOF 选择平滑规则。

五个外折中，折 1/3/4/5 的所有平滑候选判决完全相同，走标签无关早停；只有
折 2 进入二级选择，并由训练 OOF 选择 `paper_min_blend_100`。该规则把同论文
行概率替换为论文内最小概率，修正 2 行且未引入新错误，使结果从 268/300
提升到 270/300。

## 合规边界

- exact300 身份 SHA-256：`93ea5dfba0a32501a564d06eb020386fd86b11f59ded0982043f90c064ab960b`
- 规范折清单 SHA-256：`8f6ab8f1a071878e14d4db8e19a7ec0519ed612be107b0f0fdf0489dd59aaf2d`
- 每折 60 行，同一 REFNO 不跨折
- 在 2,172 行候选池中，凡 REFNO 属于外折测试论文的行均从该折训练中排除
- 平滑分支和规则选择不读取外折标签
- 不读取旧 OOF 或旧评测作为特征或模型输入
- 不读取 PMAGRESULT 或逐行 ROCKMAG 作为运行特征
- REFNO 仅用于分组隔离和同论文一致性计算，不作为数值预测特征

规范运行时被冻结为 Python `3.9.6`、NumPy `2.0.2`、scikit-learn `1.6.1`。
脚本含逐折复现断言，重新得到 v1 的 `55/55/49/54/55` 个正确结果后才允许
paper-consistency 晋级。sklearn 1.9 的诊断运行未复现该基线，未用于正式结论。

## 专利价值

可概括为“科学证据头 + 历史策略头 + 冲突输出 + 论文级一致性校准”的多层系统：

- 科学证据头从原文识别方法、诊断结果、载磁矿物、own-study 和目标绑定；
- 历史策略头从训练论文的隔离 OOF 学习数据库编辑/字段赋值习惯；
- 两头不一致时输出知识冲突，而不是静默覆盖科学判断；
- 论文级校准仅在源记录属于同一论文时生效，并由训练 OOF 或标签无关等价性证明控制；
- 该结构可同时给出历史标签预测、科学 criterion 结论、冲突类型和证据来源。

这比单纯关键词规则更实用：它允许系统在追求数据库兼容准确率的同时，保留
Teall Nunatak、Inuyama Chert 等“原文证据与历史标签冲突”的可审计事实。

## 产物与哈希

- 正式脚本：`scripts/experiment_r3_exact300_paper_consistency_v2.py`
  - SHA-256：`443fed7f086ca36299257092fe3f4b039bbdfc72041dcbf69441480a7dcb764f`
- 候选冻结：`tmp/r3_exact_300_resultno_v2/paper_consistency_v2/candidate_freeze.json`
  - SHA-256：`3494bb936b5da556ef8d105c9174f71c365b63780bea7de3631818c87895a447`
- 正式报告：`tmp/r3_exact_300_resultno_v2/paper_consistency_v2/report.json`
  - SHA-256：`c77788373ddc3c227504f9f544f2df0c3348ceb201d6764da4178613d72b2cd9`
- 300 行 OOF：`tmp/r3_exact_300_resultno_v2/paper_consistency_v2/oof_probabilities.npz`
  - SHA-256：`611ef9e4cfbd084ab35f2c0b8df914b24f22d768a7bbd6d512048b87b9c5b83e`

## 复现与独立核验

必须使用项目根目录的 `.venv`；其他 scikit-learn 版本不属于规范运行时。完整五折命令为：

```bash
.venv/bin/python -m pip install -r requirements-r3-exact300.txt
.venv/bin/python scripts/experiment_r3_exact300_paper_consistency_v2.py --folds all
```

脚本会先校验 identity、fold manifest、targets、冻结特征缓存和依赖脚本哈希，逐折断言 v1 router 正确数为 `55/55/49/54/55`，然后才写入 paper-consistency 报告和 300 行 OOF。它不接受旧 OOF、旧报告或 PMAGRESULT/ROCKMAG 路径作为命令行输入。

本次另行从 `identity.xlsx`、`targets.xlsx` 和 `oof_probabilities.npz` 对齐重算，得到 300 行、258 个 REFNO、全部概率有限，accuracy `0.90000000`、macro-F1 `0.89058549`、balanced accuracy `0.88468900`、TN/FP/FN/TP `91/19/11/179`，与正式报告完全一致。以下相关测试在规范环境通过：

```bash
.venv/bin/python -m unittest \
  tests.test_r3_exact300_paper_consistency \
  tests.test_r3_exact300_paper_consistency_v2 \
  tests.test_r3_exact300_auxiliary_training \
  tests.test_r3_exact300_fold_manifest -v
```

## 解释限制

90% 是多轮研究后在固定 exact300 上得到的规范交叉验证结果。它满足本项目的
当前校准目标，但不应写成完全独立的外部泛化率。专利或论文材料应把它表述为
“固定基准上的全覆盖分组交叉验证结果”，并另取新论文或新冻结时间段做一次
不可回看的外部验证。

剩余 30 个错误中仍有大量历史标签与严格专家语义冲突。下一阶段的重点应是
扩大独立 PDF 验证集，并分别报告 historical-policy accuracy 与 semantic conflict
accuracy，而不是继续针对 exact300 调整规则。
