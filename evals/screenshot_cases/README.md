# Screenshot-to-UISpec Evaluation Cases

This folder defines the manual screenshot-to-UISpec evaluation set for the
experimental `generate_spec_from_image_with_autofix` pipeline.

## Layout

`manifest.json`
Lists the 5 planned screenshot cases, their target kind, prompt, expected UI
structure, input image path, and scoring focus.

`scorecard-template.json`
Template for human review after running `generate_spec_from_image_with_autofix`.

`screenshots/`
Place local PNG/JPG/WEBP inputs here using the filenames from `manifest.json`.

## Screenshot Inputs

This repository intentionally does not ship real screenshot images. The intended
evaluation inputs may come from internal product references, private captures, or
local test fixtures that are not redistributable with the source tree.

To run this suite, provide your own screenshots matching the manifest paths:

- `evals/screenshot_cases/screenshots/combat-hud.png`
- `evals/screenshot_cases/screenshots/inventory-panel.png`
- `evals/screenshot_cases/screenshots/quest-tracker.png`
- `evals/screenshot_cases/screenshots/nameplate-cluster.png`
- `evals/screenshot_cases/screenshots/chat-and-social.png`

Do not check proprietary screenshots into this repository. If we need a public
benchmark later, use only redistributable captures or purpose-built mock images.

## Manual Run

Check the manifest without calling the LLM or Unreal bridge:

```powershell
python .\evals\run_screenshot_eval.py --dry-run
```

Dry-run mode reports missing image files as `image_missing`; that is expected
until local screenshots are provided.

Run the full manifest after placing screenshots in `screenshots/`:

```powershell
$env:UESPEC_LLM_PROVIDER = "claude"
$env:UESPEC_BRIDGE_MODE = "mock"
python .\evals\run_screenshot_eval.py --provider claude --max-attempts 3
```

Use `UESPEC_BRIDGE_MODE=commandlet` instead of `mock` when validating against a
real Unreal project.

Human score uses three labels:
- `usable`: compiles, smoke test passes, and needs only minor tuning.
- `needs_small_edit`: compiles or is close, but needs manual fixes within 10 minutes.
- `not_usable`: fails to produce a workable UISpec or requires major rebuild.
