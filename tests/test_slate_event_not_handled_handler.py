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
from uespec_mcp.orchestrator.handlers.slate_event_not_handled import (
    _extract_widget_id_from_message,
    _infer_event_name,
    propose_patch_for_slate_event_not_handled,
)


def test_slate_event_not_handled_handler_fixes_vm_typo_in_string_action(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-click-typo.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "viewModel": {
                    "class": "VM_SkillBar",
                    "methods": [{"name": "UseSkill"}],
                },
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "Button",
                            "id": "slotQ",
                            "actions": [{"on": "Clicked", "do": "vm.UseSkil(0)"}],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "vm_method_typo_or_missing": LLMResponse(
                content='[{"op":"replace","path":"/root/children/0/actions/0/do","value":"vm.UseSkill(0)"}]',
                parsed_output=[{"op": "replace", "path": "/root/children/0/actions/0/do", "value": "vm.UseSkill(0)"}],
                raw_response={"provider": "mock"},
                input_tokens=14,
                output_tokens=10,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "SLATE_EVENT_NOT_HANDLED",
        "testStep": "act[0] click Q slot",
        "failureMessage": "Slate click on widget 'slotQ' was not handled.",
    }
    support_surface = {
        "ActionEvents": ["Clicked", "Hovered"],
        "ActionPrimitives": ["callVM", "emitSignal"],
        "Signals": [],
        "BuiltInUMGWidgets": [{"Type": "Button"}, {"Type": "Image"}],
    }

    result = propose_patch_for_slate_event_not_handled(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface=support_surface,
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/actions/0/do", "value": "vm.UseSkill(0)"}]
    assert result["context"]["eventName"] == "Clicked"
    assert result["context"]["missingVmMethods"] == ["UseSkil"]
    assert result["context"]["primaryDiagnosis"] == "vm_method_typo_or_missing"
    assert result["context"]["methodGuidance"][0]["nearestDeclared"][0]["candidate"] == "UseSkill"


def test_slate_event_not_handled_handler_adds_missing_vm_method_declaration(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-click-missing-method.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "viewModel": {
                    "class": "VM_SkillBar",
                    "methods": [],
                },
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "Button",
                            "id": "slotQ",
                            "actions": [{"on": "Clicked", "do": {"callVM": "vm.UseSkill(0)"}}],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "Widget id: slotQ": LLMResponse(
                content='[{"op":"add","path":"/viewModel/methods/0","value":{"name":"UseSkill"}}]',
                parsed_output=[{"op": "add", "path": "/viewModel/methods/0", "value": {"name": "UseSkill"}}],
                raw_response={"provider": "mock"},
                input_tokens=13,
                output_tokens=10,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "SLATE_EVENT_NOT_HANDLED",
        "testStep": "act[0] click Q slot",
        "failureMessage": "Slate click on widget 'slotQ' was not handled.",
    }

    result = propose_patch_for_slate_event_not_handled(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={
            "ActionEvents": ["Clicked"],
            "ActionPrimitives": ["callVM"],
            "Signals": [],
            "BuiltInUMGWidgets": [{"Type": "Button"}],
        },
    )

    assert result["ok"] is True
    assert result["patch"][0]["path"] == "/viewModel/methods/0"
    assert result["context"]["missingVmMethods"] == ["UseSkill"]
    assert result["context"]["matchingActions"][0]["callVM"]["method"] == "UseSkill"


def test_slate_event_not_handled_handler_flags_unsupported_widget_event(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-click-image.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "viewModel": {
                    "class": "VM_SkillBar",
                    "methods": [{"name": "UseSkill"}],
                },
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "Image",
                            "id": "slotQ",
                            "actions": [{"on": "Clicked", "do": "vm.UseSkill(0)"}],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "unsupported_widget_event": LLMResponse(
                content='[{"op":"replace","path":"/root/children/0/type","value":"Button"}]',
                parsed_output=[{"op": "replace", "path": "/root/children/0/type", "value": "Button"}],
                raw_response={"provider": "mock"},
                input_tokens=12,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "SLATE_EVENT_NOT_HANDLED",
        "testStep": "act[0] click Q slot",
        "failureMessage": "Slate click on widget 'slotQ' was not handled.",
    }

    result = propose_patch_for_slate_event_not_handled(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={
            "ActionEvents": ["Clicked"],
            "ActionPrimitives": ["callVM"],
            "Signals": [],
            "BuiltInUMGWidgets": [{"Type": "Button"}, {"Type": "Image"}],
        },
    )

    assert result["ok"] is True
    assert result["context"]["widgetEventSupport"]["likelySupported"] is False
    assert result["context"]["primaryDiagnosis"] == "unsupported_widget_event"
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/type", "value": "Button"}]
    assert result["context"]["fallback"] is True


def test_slate_event_not_handled_handler_adds_missing_clicked_action_via_fallback(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-hover-only.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "viewModel": {
                    "class": "VM_SkillBar",
                    "methods": [{"name": "UseSkill"}],
                },
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "Button",
                            "id": "slotQ",
                            "actions": [{"on": "Hovered", "do": "vm.UseSkill(0)"}],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    result = propose_patch_for_slate_event_not_handled(
        spec_path=spec_path,
        failure={
            "category": "SLATE_EVENT_NOT_HANDLED",
            "testStep": "act[0] click Q slot",
            "failureMessage": "Slate click on widget 'slotQ' was not handled.",
        },
        llm_client=MockLLMClient({}),
        support_surface={
            "ActionEvents": ["Clicked", "Hovered"],
            "ActionPrimitives": ["callVM"],
            "Signals": [],
            "BuiltInUMGWidgets": [{"Type": "Button"}],
        },
    )

    assert result["ok"] is True
    assert result["patch"] == [
        {
            "op": "add",
            "path": "/root/children/0/actions/1",
            "value": {"on": "Clicked", "do": "vm.UseSkill(0)"},
        }
    ]
    assert result["context"]["primaryDiagnosis"] == "missing_action_for_event"
    assert result["context"]["fallback"] is True


def test_slate_event_not_handled_handler_rejects_wrong_category(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken.json"
    spec_path.write_text("{}", encoding="utf-8")
    llm_client = MockLLMClient({"*": LLMResponse(content="[]", parsed_output=[], raw_response={}, model="mock")})

    with pytest.raises(ValueError, match="SLATE_EVENT_NOT_HANDLED"):
        propose_patch_for_slate_event_not_handled(
            spec_path=spec_path,
            failure={"category": "ASSERTION_FAILED"},
            llm_client=llm_client,
            support_surface={},
        )


def test_extract_widget_id_and_infer_event_name_cover_click_and_hover() -> None:
    message = "Slate click on widget 'slotQ' was not handled."

    assert _extract_widget_id_from_message(message) == "slotQ"
    assert _infer_event_name(message, "act[0] click Q slot", None) == "Clicked"
    assert _infer_event_name("Slate hover on widget 'slotQ' was not handled.", "act[0] hover Q slot", None) == "Hovered"
