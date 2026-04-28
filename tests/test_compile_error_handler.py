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
from uespec_mcp.orchestrator.handlers.compile_error import (
    _extract_relevant_support_surface_section,
    _extract_spec_snippet,
    _parse_json_path,
    propose_patch_for_compile_error,
)
from uespec_mcp.orchestrator.handlers._json_patch import parse_patch_content, validate_patch_array


def test_compile_error_handler_returns_valid_patch(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "root": {
                    "type": "CanvasPanel",
                    "id": "rootPanel",
                    "children": [
                        {
                            "type": "TextBlock",
                            "id": "cooldownLabel",
                            "content": {"bind": "vm.cooldown", "convert": "Percnt"},
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "UNKNOWN_CONVERTER": LLMResponse(
                content='[{"op":"replace","path":"/root/children/0/content/convert","value":"Percent"}]',
                parsed_output=[
                    {"op": "replace", "path": "/root/children/0/content/convert", "value": "Percent"}
                ],
                raw_response={"provider": "mock"},
                input_tokens=10,
                output_tokens=8,
                model="mock-llm",
            )
        }
    )
    failure = {
        "category": "COMPILE_ERROR",
        "compilerErrorCode": "UNKNOWN_CONVERTER",
        "testStep": "$.root.children[0].content.convert",
        "failureMessage": "Converter 'Percnt' is not registered.",
        "nearestDeclared": ["Percent"],
    }
    support_surface = {"Converters": [{"Name": "Percent"}, {"Name": "BoolToVisibility"}]}

    result = propose_patch_for_compile_error(
        spec_path=spec_path,
        failure=failure,
        llm_client=llm_client,
        support_surface=support_surface,
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/content/convert", "value": "Percent"}]
    assert result["context"]["supportSurface"] == support_surface
    assert "UNKNOWN_CONVERTER" in result["request"]["user_prompt"]


def test_compile_error_handler_rejects_non_compile_failure(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-spec.json"
    spec_path.write_text("{}", encoding="utf-8")
    llm_client = MockLLMClient({"*": LLMResponse(content="[]", parsed_output=[], raw_response={}, model="mock")})

    with pytest.raises(ValueError, match="COMPILE_ERROR"):
        propose_patch_for_compile_error(
            spec_path=spec_path,
            failure={"category": "TIMEOUT"},
            llm_client=llm_client,
            support_surface={},
        )


def test_compile_error_handler_accepts_utf8_bom(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-spec.json"
    spec_path.write_text(
        "\ufeff"
        + json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "SkillBarHUD",
                "root": {"type": "CanvasPanel", "id": "rootPanel"},
            }
        ),
        encoding="utf-8",
    )

    llm_client = MockLLMClient(
        {
            "UNKNOWN_CONVERTER": LLMResponse(
                content='[{"op":"add","path":"/meta","value":{"fixed":true}}]',
                parsed_output=[{"op": "add", "path": "/meta", "value": {"fixed": True}}],
                raw_response={"provider": "mock"},
                input_tokens=4,
                output_tokens=4,
                model="mock-llm",
            )
        }
    )

    result = propose_patch_for_compile_error(
        spec_path=spec_path,
        failure={
            "category": "COMPILE_ERROR",
            "compilerErrorCode": "UNKNOWN_CONVERTER",
            "testStep": "$.root",
            "failureMessage": "Converter is unknown.",
        },
        llm_client=llm_client,
        support_surface={"Converters": []},
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "add", "path": "/meta", "value": {"fixed": True}}]


def test_compile_error_handler_anchors_relative_patch_path(tmp_path: Path) -> None:
    spec_path = tmp_path / "broken-relative-path.json"
    spec_path.write_text(
        json.dumps(
            {
                "apiVersion": "0.2",
                "kind": "HUD",
                "name": "CompileVmTypoHUD",
                "viewModel": {"class": "VM_SkillBar", "methods": [{"name": "UseSkill"}]},
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
            "VM_METHOD_NOT_DECLARED": LLMResponse(
                content='[{"op":"replace","path":"/children/0/actions/0/do","value":"vm.UseSkill(0)"}]',
                parsed_output=[{"op": "replace", "path": "/children/0/actions/0/do", "value": "vm.UseSkill(0)"}],
                raw_response={"provider": "mock"},
                input_tokens=6,
                output_tokens=6,
                model="mock-llm",
            )
        }
    )

    result = propose_patch_for_compile_error(
        spec_path=spec_path,
        failure={
            "category": "COMPILE_ERROR",
            "compilerErrorCode": "VM_METHOD_NOT_DECLARED",
            "testStep": "root.children[0].actions[0].do",
            "failureMessage": "vm.UseSkil referenced but not declared in viewModel.methods.",
            "nearestDeclared": ["UseSkill"],
        },
        llm_client=llm_client,
        support_surface={},
    )

    assert result["ok"] is True
    assert result["patch"] == [{"op": "replace", "path": "/root/children/0/actions/0/do", "value": "vm.UseSkill(0)"}]


def test_extract_spec_snippet_uses_json_path() -> None:
    spec_document = {
        "root": {
            "children": [
                {"id": "slotQ", "content": {"bind": "vm.skill", "convert": "Percent"}}
            ]
        }
    }

    snippet = _extract_spec_snippet(spec_document, "$.root.children[0].content")

    assert snippet == {"bind": "vm.skill", "convert": "Percent"}


def test_parse_json_path_handles_arrays() -> None:
    assert _parse_json_path("$.root.children[0].content.convert") == ["root", "children", 0, "content", "convert"]


def test_extract_relevant_support_surface_section_filters_converter_section() -> None:
    support_surface = {
        "Converters": [{"Name": "Percent"}],
        "StyleTokens": [{"Name": "color.primary"}],
        "Signals": [{"Name": "onSkillUsed"}],
    }

    filtered = _extract_relevant_support_surface_section("UNKNOWN_CONVERTER", support_surface)

    assert filtered == {"Converters": [{"Name": "Percent"}]}


def test_validate_patch_array_normalizes_json_path_fields() -> None:
    patch = [
        {
            "op": "replace",
            "path": "$.root.children[0].visibility.convert",
            "value": "BoolToVisibility",
        }
    ]

    validate_patch_array(patch)

    assert patch[0]["path"] == "/root/children/0/visibility/convert"


def test_parse_patch_content_accepts_fenced_json() -> None:
    parsed = parse_patch_content(
        """```json
[
  {"op":"replace","path":"/stateMachine/currentState","value":"Cooldown"}
]
```"""
    )

    assert parsed == [{"op": "replace", "path": "/stateMachine/currentState", "value": "Cooldown"}]
