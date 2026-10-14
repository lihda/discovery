# R3/R5/R7 大模型 API 评测接口

## 1. 用途与边界

项目预留了独立入口 `scripts/llm_api_evaluator.py`，用于在普通终端、服务器或其他
程序中调用 OpenAI-compatible `/chat/completions` 接口。该入口仅支持 R3、R5、R7；
R1、R2、R4、R6 继续使用确定性表字段公式，不应交给大模型计算。

现有 `run_r1_r7_workflow.sh` 默认不会导入该模块、不会联网，也不会改变当前精度。
只有显式运行 `run_llm_api_evaluation.sh` 或在程序中调用 `LLMApiEvaluator` 才会发送
API 请求。

## 2. 配置 API

配置文件为 `config/llm_api.example.json`。API 地址、模型名和密钥建议通过环境变量
提供：

```bash
export PALEOMAG_LLM_BASE_URL="https://your-api-host.example/v1"
export PALEOMAG_LLM_MODEL="your-model-name"
export PALEOMAG_LLM_API_KEY="your-secret-key"
```

密钥不会写入配置、输出或审计信息。非本机地址必须使用 HTTPS；本地部署可使用
`http://127.0.0.1:<port>/v1` 或 `http://localhost:<port>/v1`。

只检查配置、不发送请求：

```bash
.venv/bin/python3 scripts/llm_api_evaluator.py --check-config
```

## 3. 输入 JSONL

一行对应一个指标和一个 `RESULTNO`。最小示例：

```json
{"indicator":"R3","RESULTNO":"101","REFNO":"22","ROCKNAME":"Unit A","parameters":{"method":"IRM and hysteresis were performed"},"knowledge":{"carrier_requirement":"carrier evidence must be target-bound"},"evidence":[{"page":8,"source":"22.pdf","text":"Magnetite carries the stable ChRM in Unit A."}]}
```

| 字段 | 要求 |
| --- | --- |
| `indicator` | 只能是 `R3`、`R5`、`R7` |
| `RESULTNO/REFNO/ROCKNAME` | 必填身份字段 |
| `parameters` | mapping 表整理出的结构化参数，可为空对象 |
| `knowledge` | 本次判定允许使用的冻结知识，可为空对象 |
| `evidence` | 原文证据数组，建议包含 `text/page/source` |
| `paper_text` | 没有 `evidence` 时可用的简化全文字段 |

输入中的 `R1` 至 `R7`、`STANDARD_R*`、`*_RESULT_JSON`、`score`、`label`、旧预测，
以及 `PMAGRESULT/ROCKUNIT/SOURCE_METADATA` 容器会在请求前剔除并记入审计，防止把
历史答案或来源数据库字段发送给模型。

## 4. 运行

```bash
./run_llm_api_evaluation.sh \
  /path/to/api_input.jsonl \
  /path/to/api_output.jsonl
```

也可以只处理前几行进行测试：

```bash
./run_llm_api_evaluation.sh input.jsonl output.jsonl --limit 3
```

脚本采用临时文件后原子替换输出；任一请求或结果校验失败时整批失败，不会用默认
分数静默补齐。

## 5. Python 调用方法

```python
from pathlib import Path
import sys

sys.path.insert(0, "scripts")
from llm_api_evaluator import LLMApiEvaluator, load_api_settings

settings = load_api_settings(Path("config/llm_api.example.json"))
evaluator = LLMApiEvaluator(settings)

result = evaluator.evaluate(
    {
        "indicator": "R5",
        "RESULTNO": "101",
        "REFNO": "22",
        "ROCKNAME": "Unit A",
        "parameters": {"structural_correction": "reported"},
        "knowledge": {},
        "evidence": [
            {
                "page": 10,
                "source": "22.pdf",
                "text": "Directions were corrected for bedding tilt.",
            }
        ],
    }
)
```

公开方法地址：

- 配置加载：`scripts/llm_api_evaluator.py :: load_api_settings`
- 单条评测：`scripts/llm_api_evaluator.py :: LLMApiEvaluator.evaluate`
- 批量输出：`scripts/llm_api_evaluator.py :: write_evaluations`
- 终端入口：`run_llm_api_evaluation.sh`

## 6. 输出与校验

每行输出包含身份、`indicator`、`score`、`outcome`、`confidence`、中文推理、引用证据
和 `api_audit`。接口强制以下规则：

- `POSITIVE` 必须对应 `score=1`；
- `NEGATIVE` 必须对应 `score=0`；
- `UNKNOWN` 必须对应 `score=null` 并进入人工复核；
- 确定结论至少引用一条证据；
- 引用必须是输入证据中的逐字片段，模型编造的引文会使该行失败；
- 审计只记录 endpoint、模型、请求/响应 SHA-256、用量和被剔除字段，不记录密钥。

API 输出目前是独立评测 JSONL，不会自动覆盖当前正式 R3/R5/R7 结果。正式替换前，
需要在同一批数据上独立评测 API 流程，并明确版本、提示词、模型和准确率口径。
