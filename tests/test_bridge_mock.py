from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp import bridge
from uespec_mcp.bridge import MockBridge


def test_get_bridge_uses_mock_mode(monkeypatch) -> None:
    bridge.reset_bridge()
    monkeypatch.setenv("UESPEC_BRIDGE_MODE", "mock")

    client = bridge.get_bridge()

    assert isinstance(client, MockBridge)
    bridge.reset_bridge()


def test_mock_bridge_detects_compile_error_then_accepts_fix(tmp_path: Path) -> None:
    broken_path = tmp_path / "broken.json"
    fixed_path = tmp_path / "fixed.json"
    broken_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "Broken",
                "root": {
                    "id": "rootPanel",
                    "type": "CanvasPanel",
                    "children": [
                        {
                            "id": "auraBadge",
                            "type": "Image",
                            "visibility": {"bind": "vm.isAuraVisible", "convert": "BoolToVisiblity"},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    fixed_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "Fixed",
                "root": {
                    "id": "rootPanel",
                    "type": "CanvasPanel",
                    "children": [
                        {
                            "id": "auraBadge",
                            "type": "Image",
                            "visibility": {"bind": "vm.isAuraVisible", "convert": "BoolToVisibility"},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    client = MockBridge()

    broken_result = client.compile_spec(broken_path, "/Game/UESpec/Generated")
    fixed_result = client.compile_spec(fixed_path, "/Game/UESpec/Generated")

    assert broken_result["ok"] is False
    assert broken_result["errors"][0]["code"] == "UNKNOWN_CONVERTER"
    assert fixed_result["ok"] is True
    assert fixed_result["generatedAssets"] == ["/Game/UESpec/Generated/WBP_Fixed"]


def test_mock_bridge_reads_mock_test_result(tmp_path: Path) -> None:
    test_path = tmp_path / "test.json"
    report_path = tmp_path / "report.xml"
    failure_path = tmp_path / "failure.json"
    test_path.write_text("{}", encoding="utf-8")
    (tmp_path / "mock-test-result.json").write_text(
        json.dumps(
            {
                "ok": False,
                "summary": {"total": 1, "pass": 0, "fail": 1, "skip": 0},
                "errors": [],
                "failureReport": {
                    "summary": {"total": 1, "pass": 0, "fail": 1, "skip": 0},
                    "failures": [{"category": "ASSERTION_FAILED", "failureMessage": "Expected text."}],
                },
            }
        ),
        encoding="utf-8",
    )

    result = MockBridge().run_test_spec(test_path, report_path, failure_path)

    assert result["ok"] is False
    assert result["failureReport"]["failures"][0]["category"] == "ASSERTION_FAILED"
    assert failure_path.is_file()

