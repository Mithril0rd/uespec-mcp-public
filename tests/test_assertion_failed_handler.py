from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp.llm.adapters.mock import MockLLMClient
from uespec_mcp.llm.types import LLMResponse
from uespec_mcp.orchestrator.handlers.assertion_failed import (
    _classify_assertion_kind,
    _extract_style_property_from_message,
    _extract_widget_id_from_message,
    propose_patch_for_assertion_failed,
)


def test_assertion_failed_handler_text_branch_returns_patch(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-hud.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "viewModel": {"class": "VM_SkillBar"},
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "TextBlock",
                            "id": "cooldownLabel",
                            "content": {"bind": "vm.cooldown", "convert": "RawText"},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "cooldownLabel": LLMResponse(
                content='[{"op":"replace","path":"/root/children/0/content/convert","value":"Percent"}]',
                parsed_output=[{"op": "replace", "path": "/root/children/0/content/convert", "value": "Percent"}],
                raw_response={"provider": "mock"},
                input_tokens=11,
                output_tokens=9,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "ASSERTION_FAILED",
        "testStep": "assert[0] cooldown label shows percent",
        "failureMessage": "Widget 'cooldownLabel' text was '0.5', expected '50%' (Exact).",
        "expectedValue": "50%",
        "actualValue": "0.5",
    }
    support_surface = {"Converters": [{"Name": "Percent"}, {"Name": "RawText"}]}

    result = propose_patch_for_assertion_failed(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface=support_surface,
    )

    assert result["ok"] is True
    assert result["context"]["assertionKind"] == "text"
    assert result["context"]["widgetId"] == "cooldownLabel"
    assert result["context"]["supportSurface"] == support_surface
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/content/convert", "value": "Percent"}]


def test_assertion_failed_handler_state_branch_returns_patch(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-state.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "viewModel": {"class": "VM_SkillBar", "methods": [{"name": "IsCooldownReady"}]},
                "stateMachine": {
                    "initial": "Normal",
                    "states": {
                        "Normal": {"transitions": [{"to": "Cooldown", "when": "vm.IsCooldownReady"}]},
                        "Cooldown": {}
                    },
                },
                "testSpec": {"phases": {"assert": [{"assertState": {"expected": "Cooldown"}}]}},
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "Widget state was": LLMResponse(
                content='[{"op":"replace","path":"/stateMachine/states/Normal/transitions/0/when","value":"vm.ShouldEnterCooldown"}]',
                parsed_output=[{"op": "replace", "path": "/stateMachine/states/Normal/transitions/0/when", "value": "vm.ShouldEnterCooldown"}],
                raw_response={"provider": "mock"},
                input_tokens=10,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "ASSERTION_FAILED",
        "testStep": "assert[0] transitions to cooldown",
        "failureMessage": "Widget state was 'Normal', expected 'Cooldown'.",
        "expectedValue": "Cooldown",
        "actualValue": "Normal",
    }

    result = propose_patch_for_assertion_failed(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={},
    )

    assert result["ok"] is True
    assert result["context"]["assertionKind"] == "state"
    assert result["context"]["stateMachineContext"]["initial"] == "Normal"
    assert result["patch"][0]["path"] == "/stateMachine/states/Normal/transitions/0/when"


def test_assertion_failed_handler_style_branch_returns_patch(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-style.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "stateMachine": {"initial": "Normal", "states": {"Normal": {}, "Cooldown": {}}},
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "Button",
                            "id": "slotQ",
                            "stateBindings": {
                                "Cooldown": {"opacity": 1.0}
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "style 'opacity'": LLMResponse(
                content='[{"op":"replace","path":"/root/children/0/stateBindings/Cooldown/opacity","value":0.6}]',
                parsed_output=[{"op": "replace", "path": "/root/children/0/stateBindings/Cooldown/opacity", "value": 0.6}],
                raw_response={"provider": "mock"},
                input_tokens=10,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "ASSERTION_FAILED",
        "testStep": "assert[1] icon is dimmed",
        "failureMessage": "Widget 'slotQ' style 'opacity' was '1.0', expected '0.6'.",
        "expectedValue": "0.6",
        "actualValue": "1.0",
    }
    support_surface = {"StyleTokens": [{"Name": "opacity.cooldown"}]}

    result = propose_patch_for_assertion_failed(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface=support_surface,
    )

    assert result["ok"] is True
    assert result["context"]["assertionKind"] == "style"
    assert result["context"]["widgetId"] == "slotQ"
    assert result["context"]["styleProperty"] == "opacity"
    assert result["context"]["supportSurface"] == support_surface


def test_assertion_failed_handler_anchors_relative_widget_patch_path(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-visibility.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "AuraHUD",
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "Image",
                            "id": "auraBadge",
                            "visibility": {"bind": "vm.isAuraVisible", "convert": "BoolToVisibility"},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "visibility was": LLMResponse(
                content='[{"op":"replace","path":"/visibility/bind","value":"vm.hasAura"}]',
                parsed_output=[{"op": "replace", "path": "/visibility/bind", "value": "vm.hasAura"}],
                raw_response={"provider": "mock"},
                input_tokens=8,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )

    result = propose_patch_for_assertion_failed(
        spec_path=spec_path,
        failure={
            "category": "ASSERTION_FAILED",
            "testStep": "badge appears",
            "failureMessage": "Widget 'auraBadge' visibility was false, expected true.",
            "expectedValue": "true",
            "actualValue": "false",
        },
        llm_client=llm_client,
        support_surface={"StyleTokens": []},
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/visibility/bind", "value": "vm.hasAura"}]
    assert result["context"]["widgetPointer"] == "/root/children/0"


def test_assertion_failed_handler_anchors_widget_id_prefixed_patch_path(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-text.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "PercentHUD",
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "TextBlock",
                            "id": "cooldownLabel",
                            "content": "0%",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "cooldownLabel": LLMResponse(
                content='[{"op":"replace","path":"/cooldownLabel/content","value":"50%"}]',
                parsed_output=[{"op": "replace", "path": "/cooldownLabel/content", "value": "50%"}],
                raw_response={"provider": "mock"},
                input_tokens=8,
                output_tokens=7,
                model="mock-llm",
            )
        }
    )

    result = propose_patch_for_assertion_failed(
        spec_path=spec_path,
        failure={
            "category": "ASSERTION_FAILED",
            "testStep": "shows percent",
            "failureMessage": "Widget 'cooldownLabel' text was '0%', expected '50%' (Exact).",
            "expectedValue": "50%",
            "actualValue": "0%",
        },
        llm_client=llm_client,
        support_surface={"Converters": []},
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/content", "value": "50%"}]


def test_assertion_failed_handler_rejects_non_assertion_failure(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken.json"
    spec_path.write_text("{}", encoding="utf-8")
    llm_client = MockLLMClient({"*": LLMResponse(content="[]", parsed_output=[], raw_response={}, model="mock")})

    with pytest.raises(ValueError, match="ASSERTION_FAILED"):
        propose_patch_for_assertion_failed(
            spec_path=spec_path,
            failure={"category": "TIMEOUT"},
            llm_client=llm_client,
            support_surface={},
        )


def test_classify_assertion_kind_handles_core_cases() -> None:
    assert _classify_assertion_kind("Widget 'cooldownLabel' text was '0.5', expected '50%' (Exact).") == "text"
    assert _classify_assertion_kind("Widget state was 'Normal', expected 'Cooldown'.") == "state"
    assert _classify_assertion_kind("Widget 'slotQ' style 'opacity' was '1.0', expected '0.6'.") == "style"
    assert _classify_assertion_kind("Focused widget was 'slotW', expected 'slotQ'.") == "generic"


def test_extract_widget_id_and_style_property_from_message() -> None:
    message = "Widget 'slotQ' style 'opacity' was '1.0', expected '0.6'."

    assert _extract_widget_id_from_message(message) == "slotQ"
    assert _extract_style_property_from_message(message) == "opacity"
