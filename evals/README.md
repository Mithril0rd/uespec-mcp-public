# UESpec Round 5 Evaluations

`evals/` contains manual, real-LLM evaluation assets for the Round 5 autofix loop.

## Layout

`cases/`
Holds one folder per evaluation case. Each case contains:
- `case.json`
- `broken-spec.json`
- `test-spec.json`
- `expected-fix.json`

`run_eval.py`
Runs the orchestrator against one or more cases, aggregates success metrics, and writes JSON + Markdown reports.

`report_template.md`
Base template for a human-readable evaluation summary.

## How To Run

Set the usual bridge environment first:
- `UESPEC_UPROJECT_PATH`
- `UESPEC_UE_ENGINE_PATH` or `UESPEC_UE_EDITOR_CMD`

Set the LLM provider explicitly:
- `UESPEC_LLM_PROVIDER=claude`
- or `UESPEC_LLM_PROVIDER=openai`
- or `UESPEC_LLM_PROVIDER=mock`

Optional token-cost inputs for report cost estimates:
- `UESPEC_LLM_INPUT_COST_PER_1K`
- `UESPEC_LLM_OUTPUT_COST_PER_1K`

Example:

```powershell
python .\evals\run_eval.py --provider claude --max-attempts 3
```

Filter one category:

```powershell
python .\evals\run_eval.py --provider openai --category COMPILE_ERROR
```

## Notes

- This runner is meant for manual evaluation, not CI.
- Cases are copied to a temp workspace before mutation, so source fixtures stay clean.
- `expected-fix.json` is compared semantically by JSON value, not raw text bytes.
