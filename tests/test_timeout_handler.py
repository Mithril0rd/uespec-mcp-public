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
from uespec_mcp.orchestrator.handlers.timeout import (
    _extract_widget_id,
    _infer_timeout_kind,
    propose_patch_for_timeout,
)


def test_timeout_handler_state_branch_returns_patch(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-timeout-state.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "viewModel": {
                    "class": "VM_SkillBar",
                    "methods": [{"name": "ShouldEnterCooldown"}],
                },
                "stateMachine": {
                    "initial": "Normal",
                    "states": [
                        {
                            "name": "Normal",
                            "transitions": [{"to": "Cooldown", "when": "vm.IsCooldownReady"}],
                        },
                        {"name": "Cooldown"},
                    ],
                },
                "testSpec": {
                    "target": "SkillBarHUD",
                    "parameters": [{"name": "WhenReady", "vmState": {"isReady": True}}],
                    "phases": {
                        "arrange": [{"step": "seed ready state", "setVMField": {"field": "isReady", "value": True}}],
                        "assert": [{"step": "wait cooldown", "waitForState": {"state": "Cooldown", "timeout": 1}}],
                    },
                },
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "Expected state: Cooldown": LLMResponse(
                content='[{"op":"replace","path":"/stateMachine/states/0/transitions/0/when","value":"vm.ShouldEnterCooldown"}]',
                parsed_output=[{"op": "replace", "path": "/stateMachine/states/0/transitions/0/when", "value": "vm.ShouldEnterCooldown"}],
                raw_response={"provider": "mock"},
                input_tokens=14,
                output_tokens=11,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "TIMEOUT",
        "testStep": "wait cooldown",
        "failureMessage": "Widget did not reach state 'Cooldown' within 1.000s.",
        "expectedValue": "Cooldown",
        "actualValue": "Normal",
        "stateMachineSnapshot": {
            "currentState": "Normal",
            "availableStates": ["Normal", "Cooldown"],
        },
    }

    result = propose_patch_for_timeout(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={"Signals": [{"Name": "onSkillUsed"}], "ActionEvents": ["Clicked"]},
    )

    assert result["ok"] is True
    assert result["context"]["timeoutKind"] == "state"
    assert result["context"]["stateContext"]["currentState"] == "Normal"
    assert result["context"]["stateContext"]["currentStateTransitions"][0]["transition"]["when"] == "vm.IsCooldownReady"
    assert result["patch"][0]["path"] == "/stateMachine/states/0/transitions/0/when"


def test_timeout_handler_visibility_branch_returns_patch(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-timeout-visibility.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "AuraHUD",
                "viewModel": {
                    "class": "VM_Aura",
                    "fields": [{"name": "isAuraVisible", "type": "bool"}],
                },
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "Image",
                            "id": "auraBadge",
                            "visibility": {"bind": "vm.hasAura", "convert": "BoolToVisibility"},
                        }
                    ],
                },
                "testSpec": {
                    "target": "AuraHUD",
                    "parameters": [{"name": "AuraOn", "vmState": {"isAuraVisible": True}}],
                    "phases": {
                        "assert": [{"step": "wait aura badge", "waitForVisible": {"id": "auraBadge", "timeout": 1, "expected": True}}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "Widget id: auraBadge": LLMResponse(
                content='[{"op":"replace","path":"/root/children/0/visibility/bind","value":"vm.isAuraVisible"}]',
                parsed_output=[{"op": "replace", "path": "/root/children/0/visibility/bind", "value": "vm.isAuraVisible"}],
                raw_response={"provider": "mock"},
                input_tokens=14,
                output_tokens=11,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "TIMEOUT",
        "testStep": "wait aura badge",
        "failureMessage": "Widget 'auraBadge' did not reach visible=true within 1.000s.",
        "expectedValue": "true",
        "actualValue": "not-reached",
        "missingIds": ["auraBadge"],
        "widgetTreeSnapshot": {
            "root": {
                "id": "rootPanel",
                "type": "CanvasPanel",
                "children": [
                    {"id": "auraBadge", "type": "Image", "visible": False, "visibility": "Collapsed"},
                ],
            }
        },
    }

    result = propose_patch_for_timeout(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={"Converters": [{"Name": "BoolToVisibility"}], "StyleTokens": []},
    )

    assert result["ok"] is True
    assert result["context"]["timeoutKind"] == "visibility"
    assert result["context"]["widgetId"] == "auraBadge"
    assert result["context"]["visibilityContext"]["visibilityBinding"]["bind"] == "vm.hasAura"
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/visibility/bind", "value": "vm.isAuraVisible"}]


def test_timeout_handler_uses_test_spec_signal_candidates(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-timeout-signal.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SignalHUD",
                "stateMachine": {
                    "initial": "Normal",
                    "states": [
                        {
                            "name": "Normal",
                            "transitions": [{"to": "Cooldown", "when": "onSkillUseed"}],
                        },
                        {"name": "Cooldown"},
                    ],
                },
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )
    test_spec_path = tmp_path / "test-spec.json"
    test_spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "Test",
                "name": "SignalHUDTest",
                "testSpec": {
                    "target": "SignalHUD",
                    "phases": {
                        "act": [{"step": "emit signal", "emitSignal": "onSkillUsed"}],
                        "assert": [{"step": "wait cooldown", "waitForState": {"state": "Cooldown", "timeout": 1}}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "onSkillUsed": LLMResponse(
                content='[{"op":"replace","path":"/stateMachine/states/0/transitions/0/when","value":"onSkillUsed"}]',
                parsed_output=[{"op": "replace", "path": "/stateMachine/states/0/transitions/0/when", "value": "onSkillUsed"}],
                raw_response={"provider": "mock"},
                input_tokens=12,
                output_tokens=10,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "TIMEOUT",
        "specPath": str(test_spec_path),
        "testStep": "wait cooldown",
        "failureMessage": "Widget did not reach state 'Cooldown' within 1.000s.",
        "expectedValue": "Cooldown",
        "actualValue": "Normal",
        "stateMachineSnapshot": {
            "currentState": "Normal",
            "availableStates": ["Normal", "Cooldown"],
        },
    }

    result = propose_patch_for_timeout(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={"Signals": [], "ActionEvents": ["Clicked"]},
    )

    assert result["ok"] is True
    assert "onSkillUsed" in result["context"]["supportSurface"]["Signals"]
    assert result["context"]["arrangeContext"] == []
    assert result["patch"] == [{"op": "replace", "path": "/stateMachine/states/0/transitions/0/when", "value": "onSkillUsed"}]


def test_timeout_handler_drops_test_spec_only_patch_ops(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-timeout-filter.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SignalHUD",
                "stateMachine": {
                    "initial": "Normal",
                    "states": [
                        {
                            "name": "Normal",
                            "transitions": [{"to": "Cooldown", "when": "onSkillUseed"}],
                        },
                        {"name": "Cooldown"},
                    ],
                },
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )
    test_spec_path = tmp_path / "test-spec.json"
    test_spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "Test",
                "name": "SignalHUDTest",
                "testSpec": {
                    "target": "SignalHUD",
                    "phases": {
                        "act": [{"step": "emit signal", "emitSignal": "onSkillUsed"}],
                        "assert": [{"step": "wait cooldown", "waitForState": {"state": "Cooldown", "timeout": 1}}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "Signals": LLMResponse(
                content='[{"op":"replace","path":"/testSpec/phases/assert/0/waitForState/timeout","value":2},{"op":"replace","path":"/stateMachine/states/0/transitions/0/when","value":"onSkillUsed"}]',
                parsed_output=[
                    {"op": "replace", "path": "/testSpec/phases/assert/0/waitForState/timeout", "value": 2},
                    {"op": "replace", "path": "/stateMachine/states/0/transitions/0/when", "value": "onSkillUsed"},
                ],
                raw_response={"provider": "mock"},
                input_tokens=16,
                output_tokens=12,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "TIMEOUT",
        "specPath": str(test_spec_path),
        "testStep": "wait cooldown",
        "failureMessage": "Widget did not reach state 'Cooldown' within 1.000s.",
        "expectedValue": "Cooldown",
        "actualValue": "Normal",
        "stateMachineSnapshot": {
            "currentState": "Normal",
            "availableStates": ["Normal", "Cooldown"],
        },
    }

    result = propose_patch_for_timeout(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={"Signals": [], "ActionEvents": ["Clicked"]},
    )

    assert result["patch"] == [{"op": "replace", "path": "/stateMachine/states/0/transitions/0/when", "value": "onSkillUsed"}]


def test_timeout_handler_rejects_wrong_category(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken.json"
    spec_path.write_text("{}", encoding="utf-8")
    llm_client = MockLLMClient({"*": LLMResponse(content="[]", parsed_output=[], raw_response={}, model="mock")})

    with pytest.raises(ValueError, match="TIMEOUT"):
        propose_patch_for_timeout(
            spec_path=spec_path,
            failure={"category": "ASSERTION_FAILED"},
            llm_client=llm_client,
            support_surface={},
        )


def test_infer_timeout_kind_and_widget_id_cover_state_and_visibility() -> None:
    assert _infer_timeout_kind(
        failure_message="Widget did not reach state 'Cooldown' within 1.000s.",
        state_machine_snapshot={"currentState": "Normal"},
        missing_ids=[],
    ) == "state"
    assert _infer_timeout_kind(
        failure_message="Widget 'auraBadge' did not reach visible=true within 1.000s.",
        state_machine_snapshot={},
        missing_ids=["auraBadge"],
    ) == "visibility"
    assert _extract_widget_id("Widget 'auraBadge' did not reach visible=true within 1.000s.", ["auraBadge"]) == "auraBadge"


def test_infer_timeout_kind_prefers_visibility_over_state_snapshot() -> None:
    assert _infer_timeout_kind(
        failure_message="Widget 'readyBadge' did not reach visible=true within 1.000s.",
        state_machine_snapshot={"currentState": "None"},
        missing_ids=[],
    ) == "visibility"


def test_timeout_handler_uses_literal_visibility_fallback_without_llm(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-literal-visibility.json"
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
                            "visibility": "Hidden",
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient({})
    failure = {
        "category": "TIMEOUT",
        "testStep": "wait aura badge",
        "failureMessage": "Widget 'auraBadge' did not reach visible=true within 1.000s.",
        "expectedValue": "true",
        "actualValue": "not-reached",
        "widgetTreeSnapshot": {
            "root": {
                "id": "rootPanel",
                "type": "CanvasPanel",
                "children": [
                    {"id": "auraBadge", "type": "Image", "visible": False, "visibility": "Hidden"},
                ],
            }
        },
    }

    result = propose_patch_for_timeout(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={"Converters": [], "StyleTokens": []},
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/visibility", "value": "Visible"}]
    assert result["context"]["fallback"] is True


def test_timeout_handler_uses_missing_state_fallback_without_llm(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-missing-state.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "StateHUD",
                "stateMachine": {
                    "initial": "Normal",
                    "states": [
                        {
                            "name": "Normal",
                            "transitions": [{"to": "Cooldown", "when": "onSkillUsed"}],
                        }
                    ],
                },
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient({})
    failure = {
        "category": "TIMEOUT",
        "testStep": "wait cooldown",
        "failureMessage": "Widget did not reach state 'Cooldown' within 1.000s.",
        "expectedValue": "Cooldown",
        "actualValue": "Normal",
        "stateMachineSnapshot": {
            "currentState": "Normal",
            "availableStates": ["Normal"],
        },
    }

    result = propose_patch_for_timeout(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={"Signals": [{"Name": "onSkillUsed"}], "ActionEvents": ["Clicked"]},
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "add", "path": "/stateMachine/states/1", "value": {"name": "Cooldown"}}]
    assert result["context"]["fallback"] is True
