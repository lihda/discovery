# 项目清理与完整性报告

日期：2026-07-17

## 2026-07-22 R1-R7 监督稳定模型部署补充

项目统一公开入口为 `run_r1_r7_workflow.py` / `run_r1_r7_workflow.sh`。经用户明确
授权使用历史 R3 标签后，默认 R3 已从 `paper_only` 替换为
`supervised_stable`：先保留 paper-only 原文审计，再由最终监督模型按单/多目标
路由；缺 PDF 或有效页文本时显式回退。公开运行清单同时纳入生产模型、模型
manifest、冻结知识特征和内置 BGE 快照，并在每次预检中校验哈希。

当前 10,421 条目标的输入身份预检通过；4,196 个目标 REFNO 中 3,384 个具有本地
PDF 或解析文本，812 个无本地原文并被显式报告。正式内部 R3 精度仍为 grouped OOF
2,316/2,751（84.19%），最终模型 92.33% 的训练集重代入不可报告。部署后全项目
回归为 453 tests OK。

本补充没有重跑或改写下文的冻结 OOF 产物，所以下文 2026-07-17 的哈希、结果数字
和归档记录保持原值。

## 结论

活动项目已从约 9.7 GB 收敛到约 8.5 GB。当前业务 R3/R5/R7、strict v2.1、
单目标 V3、全量 scope/GREP、权重稳健性实验及其真实上游依赖均保留。未晋升
实验和旧输出已外部归档，没有直接丢弃可恢复的研究文件。

清理后独立复算结果：

| 结果 | 独立重算 |
| --- | ---: |
| 单目标 V3 | 785/872 = 90.02% |
| 全量稳定组合 | 2,316/2,751 = 84.19% |
| 全量探索最高 | 2,325/2,751 = 84.51% |

84.51% 的 REFNO 聚类 bootstrap 区间跨 0，仍按探索结果保留，不替代已部署的稳定
84.19% 组合。监督业务流与严格知识评分仍保持不同的数据边界。

## 本次归档

归档目录：

`/Users/lihd/discovery/paleomagnetic-r3-workflow/archive/r357_current_cleanup_20260717`

- `MOVE_LIST.txt`：107 个原相对路径移动项；
- `SHA256SUMS.txt`：255 个归档文件的 SHA-256；
- 归档逻辑大小：约 1.6 GB；
- `SHA256SUMS.txt` 自身 SHA-256：
  `352ed3ffc5c672ce0f4d1aa80085737e321e21a37144b8d972c53e0bbfcafb5f`；
- 全部 255 条归档哈希已通过 `shasum -a 256 -c`。

主要归档内容包括：2026-07-11 旧业务输出、旧 strict scoring、entity/layout/NLI
未晋升实验、历史选择性 criterion、MIL 评测头、trace v1-v3、旧路由副本和不再
属于当前运行的专利草稿。归档保持原相对路径，可按 `MOVE_LIST.txt` 定向恢复。

## 冻结审计修复

清理前发现页缓存本体哈希正确，但 audit 元数据曾从原七级缓存声明缩成两级，
导致下游冻结清单期待的 audit 哈希失配。逐条核验确认：

- 1,478 篇 PDF 全部对齐；
- 1,457 篇来自主缓存，21 篇来自 MinerU 后备缓存；
- 其余五个低优先级候选实际贡献 0 篇；
- 页缓存记录、文本长度、选择来源和内容哈希均无变化。

现已恢复原声明并重新生成缓存审计，冻结链重新一致：

| 产物 | SHA-256 |
| --- | --- |
| `audit.json` | `a12ce6a2b930eb4feef49338ecf04c9c5b4e75364f9981d401b704ea963612c2` |
| `page_cache.jsonl` | `388bd9d405119a2ee19aa08d9b0194c3902bdcb0b52b9d5e8925f7ff2405a076` |
| `combined_features.manifest.json` | `cd9263e1a8c018f9f0d4cc25c8a1878ed1ae52254423c17968b104d07079bea6` |
| `combined_features.npz` | `dc9163c35699582acda29c457eb8ab2eeef41b8e50eba02af6807e56e90b6f7f` |

四个未被选择的旧 fallback 通过硬链接指向 2026-07-13 外部归档；旧业务页缓存
和一个 MinerU 身份清单也以硬链接保留审计路径。硬链接不重复占用物理磁盘块，
但不能随意删除，否则冻结路径完整性会再次被破坏。

## 保留范围

保留的核心产物：

- `outputs/r357_paper_only_eval_07111638/` 及 07111638 的 HTML/XLSX；
- `outputs/r357_strict_v21_07111943/`；
- `outputs/r357_strict_eval_07111758/inputs/`；
- `tmp/r3_all_labelled_pdf_page_cache_v1/`；
- `tmp/r3_all_labelled_pdf_grouped_oof_v1/` 的基础、GREP、scope 和 trace v4；
- query、residual、style V2、fulltext V3、GREP v4-fold、scope 和权重实验目录；
- MIL bag 与 strict-trusted summary 上游；
- 3,500 个原始 PDF、解析文本、知识库、校准产物和源元数据。

活动目录仍保留带 `exact300`、`small_screen`、`v1/v2` 名称的少量模块，因为当前
V3 和 scope evaluator 会传递导入其中的折分、特征或模型帮助函数；它们不是可随
文件名直接删除的孤立旧版本。

## 完整性检查

| 检查 | 结果 |
| --- | --- |
| 清理前测试基线 | 532 tests OK |
| 清理后活动测试 | 444 tests OK；含新增 R3 多目标句首归属试验的 12 项测试，归档分支的测试随代码成套移走 |
| Python 编译 | 125 个活动 Python 文件全部通过 |
| 本地 import | 0 个缺失模块 |
| Python 依赖 | `pip check` PASS |
| shell 入口 | 3 个入口 `bash -n` PASS |
| 主要 Python 入口 | 8 个 `--help` 预检 PASS |
| 当前 JSON | 67 个均可解析 |
| 文件哈希引用 | 111 条全部存在且匹配 |
| 语义分组哈希 | PDF membership、PDF 内容组、规范化文本组全部匹配 |
| 原始 PDF | 3,500 个，全部非空且具有 `%PDF-` 文件头 |
| PDF 总字节数 | 7,176,838,238 |
| R3 标签身份 | 3,173 行、1,627 篇、RESULTNO 全局唯一 |
| 有 PDF 的评测范围 | 2,751 行、1,478 篇；无 PDF 跳过 422 行 |
| strict 页缓存 | 1,627 篇、23,858 页，与身份 REFNO 精确对齐 |
| 全量 R3 页缓存 | 1,478 篇、24,055 页，REFNO 唯一 |
| strict v2.1 | manifest、7 个输入/输出哈希及 boundary audit 全部 PASS |
| criterion | 冻结来源与规范校验 PASS |
| 归档 | 255 条 SHA-256 全部 PASS |

工作簿行数也已逐行检查：

- `targets.xlsx`：3,173；
- `targetsall.xlsx`：10,421；
- `targetsall_with_pdfs.xlsx`：7,241；
- `PMAGRESULT.xlsx`：10,421；
- `ROCKUNIT.xlsx`：8,634。

## 已知边界

1. 当前 90.02% 只适用于 872 篇单目标文献的内部 grouped OOF；全量稳定结果仍为
   84.19%，尚未达到 90%。
2. 当前没有全新 REFNO 外部验证集；任何 OOF 数字都不能写成外部验证结果。
3. 业务一键入口已部署 V3/scope 稳定监督模型；严格知识流的效果仍不能用该 OOF
   数字替代。
4. 冻结清单包含绝对路径；移动整个项目后，需要按新根路径正式重建/重冻结产物。
5. `pdftotext` 不是当前默认路径且本机未配置；显式启用该可选解析后端前需另行
   安装。MinerU fallback 通过独立环境按需重建。

## 验收命令

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover -s tests -v
.venv/bin/python3 -m pip check
bash -n run_r1_r7_workflow.command run_r1_r7_workflow.sh run_latest_workflow.command run_latest_workflow.sh run_strict_knowledge_workflow.sh
.venv/bin/python3 run_r1_r7_workflow.py --check-only
.venv/bin/python3 scripts/run_r357_latest_full_workflow.py --help
.venv/bin/python3 scripts/run_r357_strict_knowledge_workflow.py --help
.venv/bin/python3 scripts/evaluate_r3_fulltext_fusion_single872_oof_v3.py --help
.venv/bin/python3 scripts/evaluate_r3_scope_fusion_all_oof_v1.py --help
```
