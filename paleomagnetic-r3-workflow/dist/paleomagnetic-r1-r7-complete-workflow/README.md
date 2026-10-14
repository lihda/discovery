# 古地磁 R1-R7 完整评测项目

本目录提供从现有 Excel 来源表、目标清单和论文原文生成 R1-R7 结果的完整链路。
接手人只需要维护一份配置并运行一个入口；分项算法、证据文件、统一结果和运行
清单仍保持可审计。输入格式后续变化时，应在入口适配层归一化为当前字段契约，
不要修改七个指标的判定模块。

最快的安全检查：

```bash
.venv/bin/python3 run_r1_r7_workflow.py --check-only
```

完整运行：

```bash
./run_r1_r7_workflow.sh
```

详细交接说明见 `docs/R1-R7_项目交付与使用说明.md`。

## 当前主线

| 主线 | 入口 | 边界 |
| --- | --- | --- |
| R1-R7 完整业务流 | `run_r1_r7_workflow.command` / `run_r1_r7_workflow.sh` / `run_r1_r7_workflow.py` | 预检输入，计算七项指标，生成统一 Excel、JSONL、HTML 和 manifest |
| 严格知识流 | `run_strict_knowledge_workflow.sh` | 评分时只接收身份、原文页缓存、冻结知识库和 criterion |
| R3 单目标最佳评测 | `scripts/evaluate_r3_fulltext_fusion_single872_oof_v3.py` | 872 篇单目标文献的五折 REFNO 分组 OOF |
| R3 全量最佳评测 | `scripts/evaluate_r3_scope_fusion_all_oof_v1.py` | 单目标 V3 与多目标模型的 2,751 行组合评测 |
| R3 比例稳健性实验 | `scripts/experiment_r3_single872_base_weight_nested_oof_v1.py` | 比较全局调权与严格嵌套调权，不用于在线评分 |

完整业务流中，R1/R2/R4/R6 从 PMAGRESULT/ROCKUNIT 表字段确定性计算，R3 默认走
已部署的全量稳定监督组合，R5/R7 走论文原文和冻结规则/校准分支。R3 先生成
`paper_only` 原文证据，再从论文文本和冻结知识特征构造模型输入：单目标走 V3，
多目标走 `no-strict`；缺少本地 PDF 或有效页文本时回退到 `paper_only`。该模型使用
过历史 R3 标签训练，因此不能表述为严格无标签知识流。

## 环境

当前冻结评测环境为 Python 3.9.6、NumPy 2.0.2、scikit-learn 1.6.1。研发目录可
直接使用已有 `.venv/`；独立交付包不包含虚拟环境，应按下方依赖文件重新创建。

业务流和严格流依赖：

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

从头重建当前 R3 查询检索特征及 OOF 时使用：

```bash
.venv/bin/python3 -m pip install -r requirements-r3-retrieval-model.txt
```

`requirements-r3-text-model.txt` 属于隔离的旧文本实验环境，不是当前 V3 冻结
环境。`requirements-r3-exact300.txt` 保留用于当前代码仍依赖的历史策略组件。

## 当前 R3 结果

| 结果 | Accuracy | 状态 |
| --- | ---: | --- |
| 单目标 V3 | 785/872 = **90.02%** | 当前单目标正式内部 OOF 结果，100% 覆盖 |
| 全量 scope + V3，不叠加 GREP | 2,316/2,751 = **84.19%** | 当前稳定全量组合，已部署为 R3 默认生产模型 |
| 全量 scope + V3 与 GREP 50/50 | 2,325/2,751 = **84.51%** | 探索最高；两个外层折回退且 REFNO bootstrap 区间跨 0，不晋升 |
| 旧 R2E/GREP 成对排序 | 2,300/2,751 = 83.61% | 当前全量融合的输入组件，不再是最高结果 |

单目标三部分基础概率的现行比例为
`domain_mean4/query_pca64_logreg/front_word_logreg = 0.4526/0.2842/0.2632`。
在同一批 872 个标签上做全局网格可见 786/872，但这是读取全部开发答案后的上限；
严格嵌套选权只有 777/872，并显著低于固定比例的 785/872。因此当前比例保持不变。

以上均为按论文隔离的内部 OOF 开发结果，不是独立外部验证。生产模型最终使用
全部 2,751 条有 PDF 的历史标签重新拟合；其训练集重代入为 92.33%，该数字不可
作为正式精度。在线评分不读取 R3 标签、PMAGRESULT、当前行 ROCKMAG、已有预测或
评测报告。无本地 PDF 的 422 行被跳过，不计入 2,751 行 OOF 分母。

完整口径见 `docs/current_r3_results_20260717.md`。R2E/GREP 的中间方法与论文依据
保留在 `docs/r3_r2e_pairwise_ranking_20260715.md`，当前技术演进说明见
`docs/r3_conflict_paper_accuracy_evolution_20260716.md`。

## 输入配置与运行

当前输入集中声明在 `config/workflow.current.json`：

- `targets`：目标身份，必须含 `RESULTNO/ROCKUNITNO/REFNO/ROCKNAME`；
- `pmagresult`、`rockunit`：R1/R2/R4/R6 与身份关联来源表；
- `parsed_dir`、`pdf_dir`：R3/R5/R7 原文来源；
- `profiles`：R3/R5/R7 当前运行策略；
- `runtime.r3_stable_*`、`runtime.r3_model_*`：冻结 R3 模型、manifest、内置 BGE
  快照和推理设备；
- `output.root`：每次运行的独立输出根目录。

先做只读输入检查：

```bash
.venv/bin/python3 run_r1_r7_workflow.py \
  --config config/workflow.current.json \
  --check-only
```

运行当前全量业务流：

```bash
./run_r1_r7_workflow.sh
```

也可覆盖路径而不修改配置：

```bash
.venv/bin/python3 run_r1_r7_workflow.py \
  --targets /path/to/targets.xlsx \
  --pmagresult /path/to/PMAGRESULT.xlsx \
  --rockunit /path/to/ROCKUNIT.xlsx \
  --parsed-dir /path/to/parsed_text \
  --pdf-dir /path/to/pdfs
```

每次运行写入 `outputs/r1_r7_runs/r1_r7_<时间>/`：

- `results/r1_r7_<时间>.xlsx`：统一 R1-R7 结果与表字段证据；
- `results/r1_r7_<时间>.jsonl.gz`：无 Excel 单元格长度损失的完整压缩证据；
- `results/r1_r7_<时间>.html`：可搜索的逐条证据审阅页；
- `results/r1_r7_<时间>.manifest.json`：结果文件哈希；
- `run_manifest.json`：解析后的配置、输入预检、分项统计和产物索引；
- `work/`：页缓存、分项预测和内部摘要。

R3 的 `work/r3_paper_only_audit.jsonl` 保留规则流的原文证据，
`work/r3_supervised_stable_summary.json` 记录模型覆盖、单/多目标路由和回退数量。
如需复现旧流程，可显式传入 `--r3-source-mode paper_only`，但它不再是默认值。

运行严格知识评分：

```bash
./run_strict_knowledge_workflow.sh \
  outputs/r357_strict_eval_07111758/inputs/identity_manifest.xlsx \
  outputs/r357_strict_eval_07111758/inputs/page_cache.jsonl
```

复核当前 R3 三个核心结论：

```bash
.venv/bin/python3 scripts/evaluate_r3_fulltext_fusion_single872_oof_v3.py
.venv/bin/python3 scripts/evaluate_r3_scope_fusion_all_oof_v1.py
.venv/bin/python3 scripts/experiment_r3_single872_base_weight_nested_oof_v1.py
```

这些命令会覆盖同名冻结评测产物。只核对现有产物时应运行完整性报告中的独立
NPZ 重算命令，不必重新拟合模型。

## 数据边界

- `data/raw_pdfs/`、`data/parsed_mds/`：原始文献和解析文本。
- `data/knowledge/`：严格运行可用的 criterion 与专家知识，以及当前模型配置。
- `data/calibration/`：明确声明监督来源的离线校准产物，不能混入严格知识输入。
- `data/models/`：最终 R3 生产模型、训练/评测声明以及内置 BGE 模型快照。
- `data/source_metadata/`：只允许离线知识归纳或事后评测使用。
- `outputs/r1_r7_runs/`：新入口生成的逐次完整业务输出。
- `outputs/r357_paper_only_eval_07111638/`：历史冻结业务输出。
- `outputs/r357_strict_v21_07111943/`：当前严格知识输出。
- `tmp/r3_fulltext_fusion_single872_v3/`：单目标 V3 冻结评测。
- `tmp/r3_scope_fusion_all_oof_v1/`：稳定和探索全量融合评测。

## 科研说明与图片

- R3/R5/R7 业务流、严格流与评测流：
  `docs/r357_workflow_technical_explainer_20260714.md`
- R3 原文取证、目标绑定、证据链和监督评测：
  `docs/r3_complete_workflow_technical_explainer_20260714.md`
- R3 分拆科研图：`docs/figures/r3-*-workflow.{png,svg}`

## 归档与完整性

历史归档均位于活动项目外：

- `/Users/lihd/discovery/paleomagnetic-r3-workflow/archive/r357_legacy_20260713/`：前一次大规模清理；
- `/Users/lihd/discovery/paleomagnetic-r3-workflow/archive/r357_current_cleanup_20260717/`：本次未晋升实验、旧输出和旧代码。

归档保持原相对路径，并带文件级 SHA-256 清单。页缓存冻结审计中四个从未被
选中的低优先级缓存以硬链接保留活动路径，因此不重复占用磁盘块，同时可精确
复现原审计哈希。

完整保留清单、归档哈希、测试数量和数据检查见
`docs/project_integrity_20260717.md`。

## 快速验证

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python3 -m unittest discover -s tests -v
.venv/bin/python3 -m pip check
bash -n run_r1_r7_workflow.command run_r1_r7_workflow.sh run_latest_workflow.command run_latest_workflow.sh run_strict_knowledge_workflow.sh
.venv/bin/python3 run_r1_r7_workflow.py --help
.venv/bin/python3 run_r1_r7_workflow.py --check-only
.venv/bin/python3 scripts/run_r357_latest_full_workflow.py --help
.venv/bin/python3 scripts/create_r1_r7_deliverables.py --help
.venv/bin/python3 scripts/run_r357_strict_knowledge_workflow.py --help
.venv/bin/python3 scripts/evaluate_r3_fulltext_fusion_single872_oof_v3.py --help
.venv/bin/python3 scripts/evaluate_r3_scope_fusion_all_oof_v1.py --help
```
