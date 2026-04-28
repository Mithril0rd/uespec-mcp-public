# Round 5 Eval Report

## Summary

- Provider: `claude`
- Cases run: `4`
- Fixed: `4`
- Success rate: `100.0%`
- Avg attempts: `1.0`
- Input tokens: `2406`
- Output tokens: `139`
- Estimated USD: `n/a`

## Category Table

| Category | Total | Fixed | Rate | Avg Attempts | Estimated USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| COMPILE_ERROR | 4 | 4 | 100.0% | 1.0 | n/a |

## Best Case

{
  "caseId": "compile_error/case-01-unknown-converter",
  "status": "success",
  "fixed": true,
  "attemptCount": 1,
  "inputTokens": 463,
  "outputTokens": 37
}

## Worst Case

No failed case.

## Notes

Eligible cases: 4/4.
Run this script manually with a real LLM provider to generate the weekly Round 5 eval snapshot.

## Manual Review

- Run date: 2026-04-28 Asia/Shanghai.
- Bridge mode: `mock`, using offline compile-error semantics for the four checked `compile_error` fixtures.
- Requested model: `claude-sonnet-4-5`; relay response model field: `claude-sonnet-4.6`.
- Scope note: the current repository has 4 `compile_error` eval cases, not 8.
- Mock error-code scope: `UNKNOWN_SIGNAL` is currently produced by the Python mock bridge for eval coverage; the UE compiler does not emit this exact code yet, so mock compile-error codes are a documented superset until Round 8 alignment.
- Result: all 4 eligible cases reached `success` in 1 attempt and the final mutated JSON matched `expected-fix.json` exactly.
- Token usage: 2406 input tokens, 139 output tokens.

Per-case review:

| Case | Result | Manual conclusion |
| --- | --- | --- |
| `compile_error/case-01-unknown-converter` | Fixed in 1 attempt | Correctly replaced `BoolToVisiblity` with `BoolToVisibility`. |
| `compile_error/case-02-vm-method-typo` | Fixed in 1 attempt | Correctly replaced `vm.UseSkil(0)` with `vm.UseSkill(0)`. |
| `compile_error/case-03-unknown-signal` | Fixed in 1 attempt | Correctly replaced `onSkillUseed` with `onSkillUsed`. |
| `compile_error/case-04-unknown-widget-type` | Fixed in 1 attempt | Correctly replaced `Bttton` with `Button`. |

Earlier calibration note: `claude-haiku-4-5-20251001` produced only 1/4 fixed because three responses did not parse as JSON Patch arrays. The committed report uses the stronger Sonnet run above.
