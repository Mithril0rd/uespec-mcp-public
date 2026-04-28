from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EVALS_ROOT = REPO_ROOT / "evals"
if str(EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALS_ROOT))

import run_screenshot_eval
from run_screenshot_eval import (
    ScreenshotEvalCase,
    build_summary,
    collect_token_totals,
    load_cases,
    make_case_slug,
    render_markdown_report,
    run_case,
)


def test_load_cases_reads_manifest_defaults(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "defaultTestSpecPath": "test-specs/smoke.json",
                "cases": [
                    {
                        "caseId": "screenshot/combat-hud",
                        "title": "Combat HUD",
                        "imagePath": "screenshots/combat.png",
                        "targetKind": "HUD",
                        "visualPrompt": "combat HUD",
                        "expectedComponents": ["HealthBar"],
                        "scoringFocus": ["layout"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(manifest_path, root_dir=tmp_path)

    assert len(cases) == 1
    assert cases[0].case_id == "screenshot/combat-hud"
    assert cases[0].image_path == (tmp_path / "screenshots" / "combat.png").resolve()
    assert cases[0].test_spec_path == (tmp_path / "test-specs" / "smoke.json").resolve()
    assert cases[0].expected_components == ["HealthBar"]


def test_make_case_slug_sanitizes_path_like_id() -> None:
    assert make_case_slug("screenshot/combat-hud") == "screenshot-combat-hud"


def test_run_case_short_circuits_missing_image(tmp_path: Path) -> None:
    test_spec_path = tmp_path / "smoke.json"
    test_spec_path.write_text("{}", encoding="utf-8")
    case = ScreenshotEvalCase(
        case_id="screenshot/missing",
        title="Missing",
        image_path=tmp_path / "missing.png",
        test_spec_path=test_spec_path,
        target_kind="HUD",
        visual_prompt="",
        expected_components=[],
        scoring_focus=[],
    )

    result = run_case(
        case,
        provider="mock",
        max_attempts=3,
        max_examples=4,
        output_root=tmp_path / "out",
    )

    assert result["status"] == "image_missing"
    assert result["machineUsable"] is False
    assert result["errors"][0]["code"] == "IMAGE_NOT_FOUND"


def test_run_case_invokes_generation_tool(tmp_path: Path, monkeypatch: Any) -> None:
    image_path = tmp_path / "screen.png"
    image_path.write_bytes(b"fake")
    test_spec_path = tmp_path / "smoke.json"
    test_spec_path.write_text("{}", encoding="utf-8")
    calls: list[dict[str, Any]] = []

    def fake_generate(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "ok": True,
            "status": "success",
            "generation": {"llm": {"inputTokens": 11, "outputTokens": 7}},
            "autofix": {
                "attempts": [
                    {
                        "proposal": {
                            "llmResponse": {
                                "inputTokens": 3,
                                "outputTokens": 2,
                            }
                        }
                    }
                ]
            },
            "errors": [],
        }

    monkeypatch.setattr(run_screenshot_eval, "generate_spec_from_image_with_autofix", fake_generate)
    case = ScreenshotEvalCase(
        case_id="screenshot/combat-hud",
        title="Combat HUD",
        image_path=image_path,
        test_spec_path=test_spec_path,
        target_kind="HUD",
        visual_prompt="combat HUD",
        expected_components=["HealthBar"],
        scoring_focus=["layout"],
    )

    result = run_case(
        case,
        provider="mock",
        max_attempts=2,
        max_examples=5,
        output_root=tmp_path / "out",
    )

    assert result["status"] == "success"
    assert result["machineUsable"] is True
    assert result["inputTokens"] == 14
    assert result["outputTokens"] == 9
    assert calls[0]["image_path"] == str(image_path)
    assert calls[0]["max_attempts"] == 2
    assert calls[0]["max_examples"] == 5


def test_collect_token_totals_handles_missing_sections() -> None:
    assert collect_token_totals({}) == {"inputTokens": 0, "outputTokens": 0}


def test_build_summary_and_markdown_report() -> None:
    summary = build_summary(
        [
            {
                "caseId": "screenshot/a",
                "title": "A",
                "targetKind": "HUD",
                "status": "success",
                "machineUsable": True,
                "inputTokens": 10,
                "outputTokens": 5,
                "generatedSpecPath": "out/a.json",
                "expectedComponents": ["HealthBar"],
                "scoringFocus": ["layout"],
                "humanScore": None,
            },
            {
                "caseId": "screenshot/b",
                "title": "B",
                "targetKind": "Menu",
                "status": "image_missing",
                "machineUsable": False,
                "inputTokens": 0,
                "outputTokens": 0,
                "generatedSpecPath": "out/b.json",
                "expectedComponents": [],
                "scoringFocus": [],
                "humanScore": None,
            },
        ],
        provider="mock",
        dry_run=False,
    )

    markdown = render_markdown_report(summary)

    assert summary["totalCases"] == 2
    assert summary["completedCases"] == 1
    assert summary["missingInputCases"] == 1
    assert summary["machineUsableRate"] == 100.0
    assert "| screenshot/a | HUD | success | true | unscored | `out/a.json` |" in markdown
    assert "Human Review Rubric" in markdown

