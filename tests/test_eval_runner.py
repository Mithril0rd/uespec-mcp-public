from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

EVALS_ROOT = REPO_ROOT / "evals"
if str(EVALS_ROOT) not in sys.path:
    sys.path.insert(0, str(EVALS_ROOT))

import pytest

import run_eval
from run_eval import EvalCase, build_eval_summary, load_cases, make_temp_prefix, read_json_file, render_markdown_report, run_case, sanitize_paths


def test_load_cases_reads_case_metadata(tmp_path: Path) -> None:
    case_dir = tmp_path / "compile_error" / "case-01"
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text(
        json.dumps({"category": "COMPILE_ERROR", "caseId": "compile_error/case-01", "title": "Unknown converter"}),
        encoding="utf-8",
    )
    for file_name in ("broken-spec.json", "test-spec.json", "expected-fix.json"):
        (case_dir / file_name).write_text("{}", encoding="utf-8")

    cases = load_cases(tmp_path)

    assert len(cases) == 1
    assert cases[0].category == "COMPILE_ERROR"
    assert cases[0].case_id == "compile_error/case-01"


def test_read_json_file_accepts_utf8_bom(tmp_path: Path) -> None:
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("\ufeff{\"category\":\"COMPILE_ERROR\"}", encoding="utf-8")

    payload = read_json_file(payload_path)

    assert payload["category"] == "COMPILE_ERROR"


def test_make_temp_prefix_sanitizes_case_id() -> None:
    prefix = make_temp_prefix("compile_error/case-01-unknown-converter")

    assert prefix == "uespec-eval-compile_error-case-01-unknown-converter-"


def test_build_eval_summary_aggregates_rates_and_tokens() -> None:
    summary = build_eval_summary(
        [
            {
                "caseId": "compile_error/case-01",
                "title": "Unknown converter",
                "category": "COMPILE_ERROR",
                "status": "success",
                "fixed": True,
                "attemptCount": 1,
                "historyDir": "A",
                "inputTokens": 100,
                "outputTokens": 20,
                "estimatedUsd": 0.12,
                "result": {},
            },
            {
                "caseId": "timeout/case-01",
                "title": "Wait for state",
                "category": "TIMEOUT",
                "status": "max_attempts_exceeded",
                "fixed": False,
                "attemptCount": 3,
                "historyDir": "B",
                "inputTokens": 300,
                "outputTokens": 40,
                "estimatedUsd": 0.25,
                "result": {},
            },
        ],
        provider="mock",
    )

    assert summary["totalCases"] == 2
    assert summary["fixedCases"] == 1
    assert summary["successRate"] == 50.0
    assert summary["avgAttempts"] == 2.0
    assert summary["inputTokens"] == 400
    assert summary["outputTokens"] == 60
    assert summary["eligibleCases"] == 2
    assert summary["invalidFixtureCases"] == 0


def test_build_eval_summary_excludes_invalid_fixture_cases_from_success_rate() -> None:
    summary = build_eval_summary(
        [
            {
                "caseId": "assertion_failed/case-01",
                "title": "Broken spec already passes",
                "category": "ASSERTION_FAILED",
                "status": "invalid_fixture",
                "fixed": False,
                "attemptCount": 0,
                "historyDir": "A",
                "inputTokens": 0,
                "outputTokens": 0,
                "estimatedUsd": None,
                "fixtureValid": False,
                "fixtureStatus": "broken_spec_already_passes",
                "result": {},
            },
            {
                "caseId": "compile_error/case-01",
                "title": "Unknown converter",
                "category": "COMPILE_ERROR",
                "status": "success",
                "fixed": True,
                "attemptCount": 1,
                "historyDir": "B",
                "inputTokens": 100,
                "outputTokens": 20,
                "estimatedUsd": None,
                "fixtureValid": True,
                "fixtureStatus": "valid",
                "result": {},
            },
        ],
        provider="mock",
    )

    assert summary["totalCases"] == 2
    assert summary["eligibleCases"] == 1
    assert summary["invalidFixtureCases"] == 1
    assert summary["fixedCases"] == 1
    assert summary["successRate"] == 100.0
    assert summary["avgAttempts"] == 1.0
    categories = {entry["category"]: entry for entry in summary["categorySummary"]}
    assert categories["ASSERTION_FAILED"]["invalidFixtures"] == 1
    assert categories["COMPILE_ERROR"]["invalidFixtures"] == 0


def test_sanitize_paths_redacts_home_and_temp_prefixes() -> None:
    payload = {
        "homePath": str(Path.home() / "projects" / "uespec-mcp" / "Saved"),
        "tempPath": str(Path(tempfile.gettempdir()) / "uespec-eval-case" / "broken-spec.json"),
    }

    sanitized = sanitize_paths(payload)

    assert sanitized["homePath"].startswith("<HOME>")
    assert sanitized["tempPath"].startswith("<TEMP>")


def test_render_markdown_report_inlines_summary_values(tmp_path: Path) -> None:
    template_path = tmp_path / "report_template.md"
    template_path.write_text(
        "# Report\n{{ provider }}\n{{ total_cases }}\n{{ category_table }}\n{{ best_case }}\n{{ notes }}\n",
        encoding="utf-8",
    )

    markdown = render_markdown_report(
        {
            "provider": "mock",
            "totalCases": 1,
            "eligibleCases": 1,
            "invalidFixtureCases": 0,
            "fixedCases": 1,
            "successRate": 100.0,
            "avgAttempts": 1.0,
            "inputTokens": 10,
            "outputTokens": 2,
            "estimatedUsd": None,
            "categorySummary": [{"category": "COMPILE_ERROR", "total": 1, "fixed": 1, "rate": 100.0, "avgAttempts": 1.0, "estimatedUsd": None}],
            "cases": [
                {
                    "caseId": "compile_error/case-01",
                    "status": "success",
                    "fixed": True,
                    "attemptCount": 1,
                    "inputTokens": 10,
                    "outputTokens": 2,
                    "result": {"raw": "not rendered"},
                }
            ],
        },
        template_path,
    )

    assert "mock" in markdown
    assert "| COMPILE_ERROR | 1 | 1 | 100.0% | 1.0 | n/a |" in markdown
    assert "Eligible cases: 1/1." in markdown
    assert '"caseId": "compile_error/case-01"' in markdown
    assert '"raw": "not rendered"' not in markdown


def test_run_case_short_circuits_invalid_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    case_dir = tmp_path / "assertion_failed" / "case-01"
    case_dir.mkdir(parents=True)
    payload = {"apiVersion": "0.2", "kind": "HUD", "name": "FixtureCase", "root": {"id": "root", "type": "CanvasPanel"}}
    for file_name in ("broken-spec.json", "test-spec.json", "expected-fix.json"):
        (case_dir / file_name).write_text(json.dumps(payload), encoding="utf-8")

    case = EvalCase(
        category="ASSERTION_FAILED",
        case_id="assertion_failed/case-01",
        title="Fixture drift",
        root=case_dir,
        broken_spec_path=case_dir / "broken-spec.json",
        test_spec_path=case_dir / "test-spec.json",
        expected_fix_path=case_dir / "expected-fix.json",
    )

    monkeypatch.setattr(
        run_eval,
        "validate_case_fixture",
        lambda *args, **kwargs: {
            "valid": False,
            "status": "expected_fix_does_not_pass",
            "issues": [{"code": "EXPECTED_FIX_DOES_NOT_PASS", "message": "fixture drift"}],
            "brokenSpec": {"ok": False, "compileOk": False, "testOk": False, "generatedAsset": None, "failureMessage": ""},
            "expectedFix": {"ok": False, "compileOk": True, "testOk": False, "generatedAsset": None, "failureMessage": "fixture drift"},
        },
    )

    class FailIfConstructed:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("Orchestrator should not be constructed for invalid fixtures")

    monkeypatch.setattr(run_eval, "Orchestrator", FailIfConstructed)

    result = run_case(
        case,
        bridge=object(),
        llm_client=object(),
        max_attempts=3,
        history_root=tmp_path / "history",
    )

    assert result["status"] == "invalid_fixture"
    assert result["attemptCount"] == 0
    assert result["fixtureValid"] is False
    assert result["fixtureStatus"] == "expected_fix_does_not_pass"
