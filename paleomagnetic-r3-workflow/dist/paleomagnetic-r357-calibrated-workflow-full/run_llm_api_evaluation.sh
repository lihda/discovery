#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INPUT_JSONL OUTPUT_JSONL [additional evaluator arguments]" >&2
  echo "Configure PALEOMAG_LLM_BASE_URL, PALEOMAG_LLM_MODEL, and PALEOMAG_LLM_API_KEY first." >&2
  exit 2
fi

if [[ -x ".venv/bin/python3" ]]; then
  PYTHON=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "python3 was not found. Please install Python 3.9+ and rerun." >&2
  exit 1
fi

INPUT="$1"
OUTPUT="$2"
shift 2

exec "$PYTHON" scripts/llm_api_evaluator.py \
  --config config/llm_api.example.json \
  --input "$INPUT" \
  --output "$OUTPUT" \
  "$@"
