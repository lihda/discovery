# Paleomagnetic R1-R7 评测

项目读取一张 Excel 输入表，计算 R1-R7，并生成一个交互式 HTML 报告。

## 1. 输入文件

将文件命名为 `input.xlsx` 并放在项目根目录。工作表名称必须为 `Input`，每行对应一个唯一的 `RESULTNO`。

### 必须字段

这些字段直接用于 R1-R7 判别：

```text
RESULTNO,
HIMAGAGE, LOMAGAGE, HIGHAGE, LOWAGE,
B, N, KD, ED95, DEMAGCODE, TESTS,
R3_EVIDENCE_TEXT, R5_EVIDENCE_TEXT, R7_EVIDENCE_TEXT
```

### 可选字段

这些字段只用于身份补充、来源记录或兼容旧输入，不影响 LLM 判别：

```text
REFNO, ROCKUNITNO, ROCKNAME,
R3_EVIDENCE_SOURCE, R5_EVIDENCE_SOURCE, R7_EVIDENCE_SOURCE,
R3_TARGET_BOUND, R5_TARGET_BOUND, R7_TARGET_BOUND
```

如果不提供 `*_TARGET_BOUND`，程序会默认对应证据属于当前记录。

其他要求：

- `REFNO`、`RESULTNO`、`ROCKUNITNO` 建议保存为文本。
- 年龄字段单位为 Ma。
- `TESTS` 使用英文逗号分隔，例如 `F+,R+`。

## 2. 使用 LLM 运行全部评测

需要 Python 3.10 或更高版本。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

正式运行前可以先检查输入表（可选）：

```bash
.venv/bin/python main.py --check-only
```

`--check-only` 会检查 `input.xlsx`、`Input` 工作表、必须字段、`RESULTNO` 重复情况以及数字和字段格式。它不会调用 LLM，也不会生成 HTML，只在终端输出检查摘要。

复制配置模板：

```bash
cp .env.example .env
```

在 `.env` 中填写自己的 API 配置：

```dotenv
PALEOMAG_LLM_BASE_URL=https://your-api.example/v1
PALEOMAG_LLM_MODEL=your-model-name
PALEOMAG_LLM_API_KEY=your-secret
PALEOMAG_LLM_WIRE_API=responses
PALEOMAG_LLM_REASONING_EFFORT=xhigh
```

填写完成后执行：

```bash
.venv/bin/python main.py --text-evaluator api
```

这一条命令会校验输入、执行全部 R1-R7 评测并生成 `outputs/r1_r7_results.html`。其中 R1、R2、R4、R6 使用本地规则，R3、R5、R7 使用输入表中的原文证据调用 LLM。Responses API 请求会设置 `store=false`。

Windows 下将 `.venv/bin/python` 替换为 `.venv\Scripts\python`。

不要分享包含真实密钥的 `.env` 文件。
