# 古地磁 R1-R7 完整独立交付包

本目录是与研发目录分离的自包含工程，包含当前 Excel/论文输入、R1-R7 代码、
冻结知识与校准资产、R3 最终生产模型、内置 BGE 快照、文档和自动化测试。

默认 R3 已从 `paper_only` 替换为 `supervised_stable`：先生成 paper-only 原文证据，
再由历史监督模型评分；单目标走 V3，多目标走 `multi_no_strict`，缺 PDF 或有效页
文本时回退 paper-only。可报告的内部开发精度为 grouped OOF
`2,316/2,751 = 84.19%`，不是外部验证；最终模型 92.33% 的训练集重代入不可作为
正式精度。

## 首次运行

```bash
cd paleomagnetic-r1-r7-complete-workflow
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
.venv/bin/python3 run_r1_r7_workflow.py --check-only
./run_r1_r7_workflow.sh
```

macOS 也可双击 `run_r1_r7_workflow.command`。首次安装包含 PyTorch 和 Transformers，
需要预留下载和安装时间；BGE 模型权重已在 `data/models/` 内置，运行时不需要联网
下载模型。

## 重要入口

- `run_r1_r7_workflow.py`：统一 Python 入口；
- `run_r1_r7_workflow.sh`：自动准备环境并运行；
- `config/workflow.current.json`：当前输入、策略和模型路径；
- `config/public_runtime_manifest.json`：公开运行所需代码与资产清单；
- `docs/R1-R7_项目交付与使用说明.md`：输入、输出和后续适配说明；
- `README.md`：业务流、严格流和 R3 研究评测总索引。

## 当前输入边界

当前配置包含 10,421 条 RESULTNO、4,196 个目标 REFNO。输入预检已确认三张 Excel
身份一致。3,384 个目标 REFNO 具有本地 PDF 或解析文本，812 个缺少本地原文并会
被显式报告，不会静默删除。

## 包边界

本包不包含可重建的 `.venv`、`.DS_Store`、`__pycache__`、`.pyc` 和
`node_modules`。目录版本使用 APFS 写时复制生成，具有独立 inode；同级 ZIP 是可
直接传输的普通自包含文件。
