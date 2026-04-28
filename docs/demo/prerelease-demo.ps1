$ErrorActionPreference = "Stop"

Write-Host "== UESpec-Mcp prerelease demo =="
Write-Host "Installing package in editable mode..."
python -m pip install -e .

Write-Host ""
Write-Host "Running offline compile-error eval..."
$env:UESPEC_BRIDGE_MODE = "mock"
$env:UESPEC_LLM_PROVIDER = "mock"
python evals\run_eval.py --category compile_error --provider mock --bridge-mode mock

Write-Host ""
Write-Host "Report preview:"
Get-Content evals\last-report.md -TotalCount 40
