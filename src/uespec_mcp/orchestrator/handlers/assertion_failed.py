from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ...llm.base import LLMClient
from ...llm.prompts import render as render_llm_prompt
from ...llm.types import LLMRequest
from ...tools.surface import _ensure_surface_json
from .._json_io import read_json_file
from ._json_patch import anchor_patch_paths, json_patch_schema, parse_patch_content, validate_patch_array


def propose_patch_for_assertion_failed(
    *,
    spec_path: str | Path,
    failure: dict[str, Any],
    llm_client: LLMClient,
    support_surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (failure.get("category") or "").strip().upper() != "ASSERTION_FAILED":
        raise ValueError("assertion_failed handler only accepts ASSERTION_FAILED failures.")

    resolved_spec_path = Path(spec_path).expanduser().resolve()
    spec_document = read_json_file(resolved_spec_path)
    failure_message = str(failure.get("failureMessage") or "")
    expected_value = str(failure.get("expectedValue") or "")
    actual_value = str(failure.get("actualValue") or "")

    assertion_kind = _classify_assertion_kind(failure_message)
    widget_id = _extract_widget_id_from_message(failure_message)
    style_property = _extract_style_property_from_message(failure_message)
    widget_definition, widget_pointer = _find_widget_with_pointer(spec_document.get("root"), widget_id) if widget_id else (None, None)
    assertion_reference_sites = _find_assertion_reference_sites(spec_document, widget_id, expected_value)
    state_machine_context = spec_document.get("stateMachine") if isinstance(spec_document, dict) else None
    view_model_context = spec_document.get("viewModel") if isinstance(spec_document, dict) else None

    support_surface_snapshot = support_surface if isinstance(support_surface, dict) else _ensure_surface_json() or {}
    relevant_support_surface = _extract_relevant_support_surface(assertion_kind, support_surface_snapshot)

    template_name = _template_name_for_assertion_kind(assertion_kind)
    prompt = render_llm_prompt(
        template_name,
        test_step=str(failure.get("testStep") or ""),
        failure_message=failure_message,
        widget_id=widget_id or "",
        style_property=style_property or "",
        expected_value=expected_value,
        actual_value=actual_value,
        widget_definition=json.dumps(widget_definition, ensure_ascii=False, indent=2) if widget_definition is not None else "null",
        assertion_reference_sites=json.dumps(assertion_reference_sites, ensure_ascii=False, indent=2),
        state_machine_context=json.dumps(state_machine_context, ensure_ascii=False, indent=2) if state_machine_context is not None else "null",
        view_model_context=json.dumps(view_model_context, ensure_ascii=False, indent=2) if view_model_context is not None else "null",
        relevant_support_surface=json.dumps(relevant_support_surface, ensure_ascii=False, indent=2),
    )

    request = LLMRequest(
        system_prompt="You are a UISpec autofix worker. Output valid RFC 6902 JSON Patch only.",
        user_prompt=prompt,
        expected_output_schema=json_patch_schema(),
        temperature=0.1,
        max_tokens=2400,
    )
    response = llm_client.complete(request)

    patch = response.parsed_output
    if patch is None:
        patch = parse_patch_content(response.content)
    if not isinstance(patch, list):
        raise ValueError("LLM output did not produce a JSON Patch array.")

    validate_patch_array(patch)
    anchor_patch_paths(
        patch,
        document=spec_document,
        anchor_pointers=_build_anchor_pointers(
            widget_pointer=widget_pointer,
            assertion_kind=assertion_kind,
            assertion_reference_sites=assertion_reference_sites,
            has_state_machine=state_machine_context is not None,
            has_view_model=view_model_context is not None,
        ),
    )
    _normalize_widget_id_anchored_paths(patch, widget_id=widget_id, widget_pointer=widget_pointer)

    return {
        "ok": True,
        "handler": "assertion_failed",
        "category": "ASSERTION_FAILED",
        "specPath": str(resolved_spec_path),
        "patch": patch,
        "request": request.model_dump(),
        "llmResponse": response.model_dump(),
        "context": {
            "assertionKind": assertion_kind,
            "widgetId": widget_id,
            "styleProperty": style_property,
            "expectedValue": expected_value,
            "actualValue": actual_value,
            "widgetDefinition": widget_definition,
            "widgetPointer": widget_pointer,
            "assertionReferenceSites": assertion_reference_sites,
            "stateMachineContext": state_machine_context,
            "viewModelContext": view_model_context,
            "supportSurface": relevant_support_surface,
        },
    }


def _classify_assertion_kind(failure_message: str) -> str:
    normalized_message = failure_message.lower()
    if " text was " in normalized_message or "does not expose text" in normalized_message:
        return "text"
    if "widget state was" in normalized_message or "state apis" in normalized_message:
        return "state"
    if " style '" in normalized_message or " visibility was " in normalized_message:
        return "style"
    return "generic"


def _template_name_for_assertion_kind(assertion_kind: str) -> str:
    return {
        "text": "assertion_text.j2",
        "state": "assertion_state.j2",
        "style": "assertion_style.j2",
    }.get(assertion_kind, "assertion_generic.j2")


def _extract_widget_id_from_message(message: str) -> str | None:
    match = re.search(r"Widget '([^']+)'", message)
    return match.group(1) if match else None


def _extract_style_property_from_message(message: str) -> str | None:
    style_match = re.search(r"style '([^']+)'", message)
    if style_match:
        return style_match.group(1)
    if " visibility was " in message.lower():
        return "visibility"
    return None


def _find_widget_with_pointer(node: Any, widget_id: str | None, pointer: str = "/root") -> tuple[dict[str, Any] | None, str | None]:
    if not widget_id or not isinstance(node, dict):
        return None, None
    if node.get("id") == widget_id:
        return node, pointer
    for index, child in enumerate(node.get("children") or []):
        child_pointer = f"{pointer}/children/{index}"
        result, result_pointer = _find_widget_with_pointer(child, widget_id, child_pointer)
        if result is not None:
            return result, result_pointer
    return None, None


def _find_assertion_reference_sites(spec_document: Any, widget_id: str | None, expected_value: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    candidate_keys = {"assertText", "assertStyle", "assertState", "assertVisible", "assertFocus"}

    def walk(node: Any, pointer: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_pointer = f"{pointer}/{_escape_json_pointer_token(key)}"
                if key in candidate_keys and isinstance(value, dict):
                    if _matches_assertion_reference(value, widget_id, expected_value):
                        matches.append({"pointer": child_pointer or "/", "assertion": value})
                walk(value, child_pointer)
            return
        if isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{pointer}/{index}")

    walk(spec_document)
    return matches


def _matches_assertion_reference(assertion_object: dict[str, Any], widget_id: str | None, expected_value: str) -> bool:
    if widget_id and assertion_object.get("id") == widget_id:
        return True
    expected = assertion_object.get("expected")
    if expected_value and expected == expected_value:
        return True
    return False


def _extract_relevant_support_surface(assertion_kind: str, support_surface: dict[str, Any]) -> dict[str, Any]:
    if assertion_kind == "text":
        return {"Converters": support_surface.get("Converters", [])}
    if assertion_kind == "style":
        return {"StyleTokens": support_surface.get("StyleTokens", [])}
    return {}


def _escape_json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _normalize_widget_id_anchored_paths(
    patch: list[dict[str, Any]],
    *,
    widget_id: str | None,
    widget_pointer: str | None,
) -> None:
    if not widget_id or not widget_pointer:
        return

    widget_id_prefix = f"/{_escape_json_pointer_token(widget_id)}"
    for operation in patch:
        for key in ("path", "from"):
            raw_value = operation.get(key)
            if not isinstance(raw_value, str):
                continue
            if raw_value == widget_id_prefix:
                operation[key] = widget_pointer
                continue
            if raw_value.startswith(widget_id_prefix + "/"):
                operation[key] = widget_pointer + raw_value[len(widget_id_prefix):]


def _build_anchor_pointers(
    *,
    widget_pointer: str | None,
    assertion_kind: str,
    assertion_reference_sites: list[dict[str, Any]],
    has_state_machine: bool,
    has_view_model: bool,
) -> list[str]:
    anchors: list[str] = []
    if widget_pointer:
        anchors.append(widget_pointer)
    if assertion_kind == "state" and has_state_machine:
        anchors.append("/stateMachine")
    if assertion_kind == "text" and has_view_model:
        anchors.append("/viewModel")
    for entry in assertion_reference_sites:
        pointer = entry.get("pointer")
        if isinstance(pointer, str):
            anchors.append(pointer)
    return anchors
