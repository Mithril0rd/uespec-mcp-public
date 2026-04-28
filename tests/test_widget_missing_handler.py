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
from uespec_mcp.orchestrator.handlers.widget_missing import (
    _build_typo_guidance,
    _compress_widget_tree_snapshot,
    _extract_missing_id_from_message,
    _find_missing_id_reference_sites,
    propose_patch_for_widget_missing,
    should_target_test_spec_for_widget_missing,
)


def test_widget_missing_handler_prefers_replacing_typoed_reference(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-test.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "Test",
                "name": "SkillBarTest",
                "testSpec": {
                    "target": "SkillBarHUD",
                    "phases": {
                        "act": [{"step": "click typo slot", "click": "slotQq"}]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "slotQq": LLMResponse(
                content='[{"op":"replace","path":"/testSpec/phases/act/0/click","value":"slotQ"}]',
                parsed_output=[{"op": "replace", "path": "/testSpec/phases/act/0/click", "value": "slotQ"}],
                raw_response={"provider": "mock"},
                input_tokens=10,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )

    failure = {
        "category": "WIDGET_MISSING",
        "testStep": "act[0] click typo slot",
        "failureMessage": "Widget 'slotQq' was not found for click.",
        "missingIds": ["slotQq"],
        "widgetTreeSnapshot": {
            "root": {
                "id": "skillRow",
                "type": "HorizontalBox",
                "children": [
                    {"id": "slotQ", "type": "Button"},
                    {"id": "slotW", "type": "Button"},
                ],
            }
        },
    }
    support_surface = {
        "BuiltInUMGWidgets": [{"Type": "Button"}, {"Type": "TextBlock"}],
        "CommonUIWidgets": [],
        "RegisteredComponents": [{"Type": "SkillSlot"}],
    }

    result = propose_patch_for_widget_missing(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface=support_surface,
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "replace", "path": "/testSpec/phases/act/0/click", "value": "slotQ"}]
    assert result["context"]["typoGuidance"][0]["strategy"] == "replace_reference"
    assert result["context"]["typoGuidance"][0]["nearestExisting"][0]["candidate"] == "slotQ"
    assert "slotQq" in result["request"]["user_prompt"]


def test_widget_missing_handler_supports_add_widget_strategy(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-hud.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [{"type": "Button", "id": "slotQ"}],
                },
                "testSpec": {
                    "target": "SkillBarHUD",
                    "phases": {
                        "assert": [{"step": "show aura", "assertVisible": {"id": "auraBadge", "expected": True}}]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "auraBadge": LLMResponse(
                content='[{"op":"add","path":"/root/children/1","value":{"type":"Image","id":"auraBadge"}}]',
                parsed_output=[{"op": "add", "path": "/root/children/1", "value": {"type": "Image", "id": "auraBadge"}}],
                raw_response={"provider": "mock"},
                input_tokens=10,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )

    failure = {
        "category": "WIDGET_MISSING",
        "testStep": "assert[0] show aura",
        "failureMessage": "Widget 'auraBadge' was not found for assertVisible.",
        "missingIds": ["auraBadge"],
        "widgetTreeSnapshot": {
            "root": {
                "id": "rootPanel",
                "type": "CanvasPanel",
                "children": [{"id": "slotQ", "type": "Button"}],
            }
        },
    }
    support_surface = {
        "BuiltInUMGWidgets": [{"Type": "Button"}, {"Type": "Image"}],
        "CommonUIWidgets": [],
        "RegisteredComponents": [],
    }

    result = propose_patch_for_widget_missing(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface=support_surface,
    )

    assert result["ok"] is True
    assert result["patch"][0]["op"] == "add"
    assert result["context"]["typoGuidance"][0]["strategy"] == "add_widget"
    assert result["context"]["repairMode"] == "add_widget"
    assert result["context"]["insertionGuidance"][0]["suggestedType"] == "Image"
    assert "do not modify `/testSpec` paths" in result["request"]["user_prompt"]


def test_widget_missing_handler_hydrates_suggested_text_content(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-keybind-hud.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "KeybindHUD",
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [{"type": "Button", "id": "slotQ"}],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "slotQ_keybind": LLMResponse(
                content='[{"op":"add","path":"/root/children/-","value":{"type":"TextBlock","id":"slotQ_keybind"}}]',
                parsed_output=[{"op": "add", "path": "/root/children/-", "value": {"type": "TextBlock", "id": "slotQ_keybind"}}],
                raw_response={"provider": "mock"},
                input_tokens=10,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )

    failure = {
        "category": "WIDGET_MISSING",
        "testStep": "assert keybind visible",
        "failureMessage": "Widget 'slotQ_keybind' was not found for assertVisible.",
        "missingIds": ["slotQ_keybind"],
        "widgetTreeSnapshot": {
            "root": {
                "id": "rootPanel",
                "type": "CanvasPanel",
                "children": [{"id": "slotQ", "type": "Button"}],
            }
        },
    }

    result = propose_patch_for_widget_missing(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface={},
    )

    assert result["patch"] == [
        {
            "op": "add",
            "path": "/root/children/-",
            "value": {"type": "TextBlock", "id": "slotQ_keybind", "content": "Q"},
        }
    ]
    assert result["context"]["insertionGuidance"][0]["suggestedContent"] == "Q"


def test_widget_missing_handler_rejects_non_widget_missing_failure(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken.json"
    spec_path.write_text("{}", encoding="utf-8")
    llm_client = MockLLMClient({"*": LLMResponse(content="[]", parsed_output=[], raw_response={}, model="mock")})

    with pytest.raises(ValueError, match="WIDGET_MISSING"):
        propose_patch_for_widget_missing(
            spec_path=spec_path,
            failure={"category": "COMPILE_ERROR"},
            llm_client=llm_client,
            support_surface={},
        )


def test_compress_widget_tree_snapshot_flattens_ids_and_types() -> None:
    compressed = _compress_widget_tree_snapshot(
        {
            "root": {
                "id": "rootPanel",
                "type": "CanvasPanel",
                "children": [
                    {"id": "slotQ", "type": "Button"},
                    {"id": "slotW", "type": "Button"},
                ],
            }
        }
    )

    assert compressed == [
        {"id": "rootPanel", "type": "CanvasPanel"},
        {"id": "slotQ", "type": "Button"},
        {"id": "slotW", "type": "Button"},
    ]


def test_build_typo_guidance_prefers_replace_for_close_match() -> None:
    guidance = _build_typo_guidance(
        ["slotQq", "auraBadge"],
        [
            {"id": "slotQ", "type": "Button"},
            {"id": "slotW", "type": "Button"},
        ],
    )

    assert guidance[0]["strategy"] == "replace_reference"
    assert guidance[1]["strategy"] == "add_widget"


def test_find_missing_id_reference_sites_returns_json_pointer_contexts() -> None:
    spec_document = {
        "testSpec": {
            "phases": {
                "act": [{"click": "slotQ_typo"}],
                "assert": [{"assertVisible": {"id": "slotQ_typo", "expected": True}}],
            }
        }
    }

    references = _find_missing_id_reference_sites(spec_document, ["slotQ_typo"])

    assert {"missingId": "slotQ_typo", "pointer": "/testSpec/phases/act/0/click", "container": {"click": "slotQ_typo"}} in references
    assert any(entry["pointer"] == "/testSpec/phases/assert/0/assertVisible/id" for entry in references)


def test_extract_missing_id_from_message_parses_uespec_failure_message() -> None:
    assert _extract_missing_id_from_message("Widget 'slotQ_missing' was not found for click.") == "slotQ_missing"


def test_widget_missing_target_selection_prefers_test_spec_for_close_typo() -> None:
    failure = {
        "category": "WIDGET_MISSING",
        "failureMessage": "Widget 'slotQq' was not found for click.",
        "missingIds": ["slotQq"],
        "widgetTreeSnapshot": {
            "root": {
                "id": "rootPanel",
                "type": "CanvasPanel",
                "children": [{"id": "slotQ", "type": "Button"}],
            }
        },
    }
    test_spec_document = {
        "testSpec": {
            "target": "SkillBarHUD",
            "phases": {"act": [{"click": "slotQq"}]},
        }
    }

    assert should_target_test_spec_for_widget_missing(failure, test_spec_document) is True


def test_widget_missing_target_selection_prefers_ui_spec_for_add_widget() -> None:
    failure = {
        "category": "WIDGET_MISSING",
        "failureMessage": "Widget 'auraBadge' was not found for assertVisible.",
        "missingIds": ["auraBadge"],
        "widgetTreeSnapshot": {
            "root": {
                "id": "rootPanel",
                "type": "CanvasPanel",
                "children": [{"id": "slotQ", "type": "Button"}],
            }
        },
    }
    test_spec_document = {
        "testSpec": {
            "target": "SkillBarHUD",
            "phases": {"assert": [{"assertVisible": {"id": "auraBadge", "expected": True}}]},
        }
    }

    assert should_target_test_spec_for_widget_missing(failure, test_spec_document) is False
