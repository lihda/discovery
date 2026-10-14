#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

if [[ -x ".venv/bin/python3" ]]; then
  PYTHON=".venv/bin/python3"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "python3 was not found. Please install Python 3.9+ and rerun." >&2
  exit 1
fi

if ! "$PYTHON" - <<'PY' >/dev/null 2>&1
import joblib, numpy, openpyxl, pandas, pypdf, sklearn, torch, transformers
PY
then
  echo "Installing Python dependencies into .venv ..."
  python3 -m venv .venv
  PYTHON=".venv/bin/python3"
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install -r requirements.txt
fi

TIMESTAMP="$(date +%m%d%H%M)"

"$PYTHON" run_r1_r7_workflow.py \
  --config config/workflow.current.json \
  --timestamp "$TIMESTAMP" \
  "$@"

if [[ " $* " != *" --check-only "* ]]; then
  echo
  echo "Done."
  echo "Run directory: outputs/r1_r7_runs/r1_r7_${TIMESTAMP}"
  echo "Manifest: outputs/r1_r7_runs/r1_r7_${TIMESTAMP}/run_manifest.json"
  echo "Combined workbook: outputs/r1_r7_runs/r1_r7_${TIMESTAMP}/results/r1_r7_${TIMESTAMP}.xlsx"
  echo "Detailed review: outputs/r1_r7_runs/r1_r7_${TIMESTAMP}/results/r1_r7_${TIMESTAMP}.html"
fi
