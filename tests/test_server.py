from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp import server
from uespec_mcp.tools import compile as compile_tools
from uespec_mcp.tools import context, surface, test as test_tools, validate


@pytest.fixture(autouse=True)
def reset_tool_state() -> None:
    context._ACTIVE_SPEC = None
    surface._SUPPORT_SURFACE_CACHE = None


def test_get_support_surface_no_ue(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBridge:
        def get_support_surface(self, format: str = "markdown") -> dict[str, object]:
            return {"ok": True, "format": format, "content": {"Signals": [], "Converters": []}, "command": []}

    monkeypatch.setattr(surface, "get_bridge", lambda: FakeBridge())
    mcp = server.build_server()
    tools = server.list_registered_tools(mcp)

    assert "get_support_surface" in tools
    result = tools["get_support_surface"]("json")

    assert result["ok"] is True
    assert result["format"] == "json"
    assert result["content"] == {"Signals": [], "Converters": []}


def test_validate_bad_json_schema_layer() -> None:
    result = validate.validate_spec(json.dumps({"apiVersion": "0.2", "kind": "HUD"}))

    assert result["ok"] is False
    assert result["stage"] == "schema"
    assert any(error["code"] == "SCHEMA_VALIDATION_FAILED" for error in result["errors"])


def test_list_existing_specs(tmp_path: Path) -> None:
    valid_spec = tmp_path / "hud-skill-bar.json"
    valid_spec.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "description": "Example HUD spec.",
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "not-a-spec.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")

    result = compile_tools.list_existing_specs(str(tmp_path))

    assert result["ok"] is True
    assert result["count"] == 1
    assert result["items"][0]["name"] == "SkillBarHUD"
    assert result["items"][0]["kind"] == "HUD"


def test_session_active_spec(tmp_path: Path) -> None:
    spec_path = tmp_path / "active-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "ActiveHUD",
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )

    set_result = context.set_active_spec(str(spec_path))

    assert set_result["ok"] is True
    assert context.get_active_spec() == str(spec_path.resolve())


def test_compile_spec_to_wbp_uses_active_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_path = tmp_path / "compile-me.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "CompileMeHUD",
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )
    context.set_active_spec(str(spec_path))

    captured: dict[str, str] = {}

    class FakeBridge:
        def compile_spec(self, spec_path_arg: Path, output_dir_arg: str) -> dict[str, object]:
            captured["spec_path"] = str(spec_path_arg)
            captured["output_dir"] = output_dir_arg
            return {"ok": True, "generatedAssets": ["/Game/UI/WBP_CompileMeHUD"], "errors": []}

    monkeypatch.setattr(compile_tools, "get_bridge", lambda: FakeBridge())
    result = compile_tools.compile_spec_to_wbp(output_dir="/Game/UI/Generated")

    assert result["ok"] is True
    assert captured["spec_path"] == str(spec_path.resolve())
    assert captured["output_dir"] == "/Game/UI/Generated"


def test_run_test_spec_uses_default_report_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    spec_path = tmp_path / "skill-bar-test.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "Test",
                "name": "SkillBarTest",
                "testSpec": {"target": "SkillBarHUD", "phases": {"assert": []}},
            }
        ),
        encoding="utf-8",
    )
    context.set_active_spec(str(spec_path))

    captured: dict[str, str] = {}

    class FakeBridge:
        def run_test_spec(self, spec_path_arg: Path, output_report_arg: Path, failure_report_arg: Path) -> dict[str, object]:
            captured["spec_path"] = str(spec_path_arg)
            captured["output_report"] = str(output_report_arg)
            captured["failure_report"] = str(failure_report_arg)
            return {"ok": True, "summary": {"passed": 1, "failed": 0}, "errors": [], "failureReport": {}}

    monkeypatch.setattr(test_tools, "get_bridge", lambda: FakeBridge())
    result = test_tools.run_test_spec()

    assert result["ok"] is True
    assert captured["spec_path"] == str(spec_path.resolve())
    expected_report = Path("Saved") / "UESpec" / "TestReports" / "skill-bar-test.junit.xml"
    expected_failure = Path("Saved") / "UESpec" / "TestReports" / "skill-bar-test.failure-report.json"
    assert captured["output_report"].endswith(str(expected_report))
    assert captured["failure_report"].endswith(str(expected_failure))


def test_bridge_parses_failure_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from uespec_mcp.bridge import BridgeClient, BridgeConfig

    spec_path = tmp_path / "skill-bar-test.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "Test",
                "name": "SkillBarTest",
                "testSpec": {"target": "SkillBarHUD", "phases": {"assert": []}},
            }
        ),
        encoding="utf-8",
    )

    report_path = tmp_path / "Saved" / "UESpec" / "TestReports" / "skill-bar-test.junit.xml"
    failure_report_path = tmp_path / "Saved" / "UESpec" / "TestReports" / "skill-bar-test.failure-report.json"

    client = BridgeClient(BridgeConfig(uproject_path=tmp_path / "Fake.uproject", ue_engine_path=tmp_path / "Engine"))

    def fake_run_commandlet(*args: object, **kwargs: object) -> object:
        failure_report_path.parent.mkdir(parents=True, exist_ok=True)
        failure_report_path.write_text(
            json.dumps(
                {
                    "summary": {"total": 1, "pass": 0, "fail": 1, "skip": 0, "duration": 0.1},
                    "failures": [{"testName": "Case1", "category": "WIDGET_MISSING"}],
                }
            ),
            encoding="utf-8",
        )

        output_path = Path(kwargs["output_path"])
        output_path.write_text(
            json.dumps(
                {
                    "ok": False,
                    "summary": {"total": 1, "passed": 0, "failed": 1, "skipped": 0},
                    "errors": [],
                }
            ),
            encoding="utf-8",
        )

        from uespec_mcp.bridge import CommandletInvocation

        return CommandletInvocation(
            ok=False,
            command=["UnrealEditor-Cmd.exe", "-run=UESpecRunTests"],
            outputPath=str(output_path),
            payload=json.loads(output_path.read_text(encoding="utf-8")),
        )

    monkeypatch.setattr(client, "_run_commandlet", fake_run_commandlet)
    result = client.run_test_spec(spec_path, report_path, failure_report_path)

    assert result["ok"] is False
    assert result["failureReportPath"] == str(failure_report_path.resolve())
    assert result["failureReport"]["failures"][0]["category"] == "WIDGET_MISSING"


def test_get_failure_report_reads_json_file(tmp_path: Path) -> None:
    report_path = tmp_path / "failure-report.json"
    report_payload = {
        "summary": {"total": 1, "pass": 0, "fail": 1, "skip": 0, "duration": 0.1},
        "failures": [{"testName": "Case1", "category": "ASSERTION_FAILED"}],
    }
    report_path.write_text(json.dumps(report_payload), encoding="utf-8")

    result = test_tools.get_failure_report(str(report_path))

    assert result["ok"] is True
    assert result["failureReport"] == report_payload


def test_suggest_nearest_uses_support_surface_cache() -> None:
    surface._SUPPORT_SURFACE_CACHE = {
        "BuiltInUMGWidgets": [{"Type": "CanvasPanel"}, {"Type": "Button"}],
        "CommonUIWidgets": [{"Type": "CommonButtonBase"}],
        "RegisteredComponents": [{"Type": "SkillSlot"}],
        "Converters": [{"Name": "Percent"}, {"Name": "BoolToVisibility"}],
        "StyleTokens": [{"Name": "color.primary"}],
        "Signals": [{"Name": "onSkillUsed"}],
    }

    result = surface.suggest_nearest("SkillSlo", "widget")

    assert result["ok"] is True
    assert result["suggestions"][0] == "SkillSlot"
