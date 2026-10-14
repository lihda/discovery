#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 IDENTITY_MANIFEST_XLSX PAGE_CACHE_JSONL [OUTPUT_DIR]" >&2
  echo "This scoring command intentionally accepts no labels or source-table paths." >&2
  exit 2
fi

if [[ -x ".venv/bin/python3" ]]; then
  PYTHON=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "python3 was not found. Please install Python 3.10+ and rerun." >&2
  exit 1
fi

IDENTITY_MANIFEST="$1"
PAGE_CACHE="$2"
OUTPUT_DIR="${3:-outputs/r357_strict_knowledge_$(date +%Y%m%d_%H%M%S)}"

"$PYTHON" scripts/run_r357_strict_knowledge_workflow.py \
  --identity-manifest "$IDENTITY_MANIFEST" \
  --page-cache "$PAGE_CACHE" \
  --knowledge-base data/knowledge/r357_expert_kb_v1.json \
  --output-dir "$OUTPUT_DIR"

echo
echo "Strict predictions: $OUTPUT_DIR"
echo "Run post-hoc evaluation separately only after selecting an external/held-out label workbook."
