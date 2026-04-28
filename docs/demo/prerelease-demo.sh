#!/usr/bin/env bash
set -euo pipefail

echo "== UESpec-Mcp prerelease demo =="
echo "Installing package in editable mode..."
python -m pip install -e .

echo
echo "Running offline compile-error eval..."
export UESPEC_BRIDGE_MODE=mock
export UESPEC_LLM_PROVIDER=mock
python evals/run_eval.py --category compile_error --provider mock --bridge-mode mock

echo
echo "Report preview:"
sed -n '1,40p' evals/last-report.md
