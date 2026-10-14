# R3 random300 新论文扩展性留出验证

日期：2026-07-13

## 结论

当前 exact300 流程没有在新的随机论文留出集上保持 90%。正式结果为：

- 300/300 条有本地 PDF，来自 263 篇未进入 canonical exact300 的论文；
- Accuracy：`254/300 = 84.67%`；
- Macro-F1：`82.18%`；
- Balanced accuracy：`81.83%`；
- TN/FP/FN/TP：`71/25/21/183`；
- 六专家固定均值仅作事后诊断，达到 `265/300 = 88.33%`，未替换正式结果。

因此，canonical exact300 的 `270/300 = 90.00%` 应继续限定为固定开发基准的
REFNO 分组 OOF 结果，不能外推为新论文准确率。

## 冻结设计

1. 从 2,172 条 PDF-backed 固定训练分区中，先排除 canonical exact300 的全部
   258 个 REFNO，得到 1,452 条、916 篇候选论文。
2. 在不读取 R3、PMAGRESULT、旧预测或旧评测的情况下，用固定种子
   `35720260713` 无放回随机抽取 300 行。
3. 抽中 263 个 REFNO；训练时完整排除这些论文的全部 520 行，包括未被抽中的
   220 条 sibling 行。
4. 其余 1,652 行、911 篇论文用于六专家和 historical-policy router 拟合。
5. 训练内部专家概率继续使用四折 REFNO 隔离 OOF。论文一致性候选在测试集上
   的二元判决完全相同，因此按标签无关规则选择 identity。
6. 预测回执写入后，才打开 300 条测试标签计算指标。

队列 SHA-256：
`f3692dbf1908fbd00e274491822955cbad27d1d6c9db79b02c01108a56efb5c3`

样本对齐 SHA-256：
`cba3544ebd4bd0b178d12df87fda0c46a0257d2ecc85a209abc222ab377b3288`

## 瓶颈诊断

### 1. 路由器跨论文失稳

同一批冻结预测上的事后诊断如下；这些候选没有参与正式结果选择：

| 模型 | Accuracy | Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| historical-policy hard router（正式） | 84.67% | 82.18% | 81.83% |
| 六专家固定均值 | 88.33% | 86.63% | 86.73% |
| 六专家固定中位数 | 87.67% | 85.94% | 86.24% |
| 最佳单专家 HGB combined | 87.33% | 85.60% | 86.00% |

hard router 在 exact300 上优于均值，但在新留出集上低 3.67 个百分点，说明其
“按上下文选择单个专家”的策略适应了开发基准，跨论文稳健性不足。

六个专家至少一个判断正确的事后 oracle 上限为 91.67%，但 251/300 行的六专家
二元判决完全一致，其中 25 行全体一致判断错误。换路由器最多只能解决部分错误。

### 2. 不是单纯阈值问题

正式阈值为 0.5。读取测试答案后的诊断性阈值扫描在 0.58 达到最高 87.00%，
仍低于 90%，因此不能仅靠调阈值修复。

### 3. 目标归属比 PDF 缺失更关键

46 个错误的原文缓存均存在，空页数为 0。错误记录原文字符数中位数约 54,711，
正确记录约 57,674；解析长度有小幅差异，但没有形成大规模空解析。

特征对比显示：

- FN 相比 TP 明显缺少 target-bound method、strong episode、chain episode 和
  component bridge；
- FP 相比 TN 具有更多论文级方法/知识片段，但 target source-confirmed alias 和
  direct-to-article scope ratio 更低；
- 这符合“论文确有岩磁实验，但证据未可靠绑定当前目标”与“历史字段有值，但
  原文目标定位未恢复”两类相反错误。

### 4. 历史字段边界与科学证据边界不完全一致

预测冻结后的 PMAGRESULT 事后诊断显示，单纯用 ROCKMAG 非空预测 R3 在该批
样本上为 94.33%（TN/FP/FN/TP=`86/10/7/197`）。在正式模型的 46 个错误中：

- 17/21 个 FN 的 ROCKMAG 有值；
- 17/25 个 FP 的 ROCKMAG 为空；
- 其余 12 个错误同时反映 R3 标签与 ROCKMAG 是否为空本身并不完全一致。

PMAGRESULT 仅在预测回执和评分完成后用于此诊断，不是预测特征。结果说明当前
主要任务仍是从原文重建“该 RESULTNO 是否具有可报告的 ROCKMAG episode”，
并把该 historical-policy proxy 与严格科学 R3 证据头分开输出。

## 下一轮建议

下一版不应继续在本 random300 上选择并宣称改进。应先冻结改动，再抽取第三批
未参与 exact300 和 random300 的 REFNO 做不可回看的验证。优先改动方向为：

1. 将 hard expert selection 改为训练期可冻结的保守共识/软融合，并用训练 OOF
   决定何时允许路由器偏离均值；
2. 对 ROCKMAG 风格的 OP、IRM、Js-T、SUSC-T、HYST 证据建立方法-结果 episode
   恢复头，尤其补强表格、图注和缩写；
3. 增加“论文级证据存在但目标 alias 未确认”的负向门控，减少 article evidence
   被错误传播到 ROCKMAG 为空的 sibling target；
4. 保留 science-fact、practice-proxy、historical-policy 三头和冲突输出，不用
   historical-policy 覆盖严格科学结论。

## 复现

```bash
.venv/bin/python scripts/freeze_r3_random300_extension_cohort.py
.venv/bin/python scripts/experiment_r3_random300_extension_holdout.py
.venv/bin/python -m unittest tests.test_r3_random300_extension -v
```

正式产物：

- `tmp/r3_random300_extension_v1/cohort_manifest.json`
- `tmp/r3_random300_extension_v1/holdout/candidate_freeze.json`
- `tmp/r3_random300_extension_v1/holdout/prediction_receipt.json`
- `tmp/r3_random300_extension_v1/holdout/holdout_probabilities.npz`
- `tmp/r3_random300_extension_v1/holdout/expert_diagnostic_probabilities.npz`
- `tmp/r3_random300_extension_v1/holdout/report.json`
