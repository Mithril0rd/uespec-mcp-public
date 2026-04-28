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
from ._json_patch import json_patch_schema, parse_patch_content, validate_patch_array


def propose_patch_for_slate_event_not_handled(
    *,
    spec_path: str | Path,
    failure: dict[str, Any],
    llm_client: LLMClient,
    support_surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (failure.get("category") or "").strip().upper() != "SLATE_EVENT_NOT_HANDLED":
        raise ValueError("slate_event_not_handled handler only accepts SLATE_EVENT_NOT_HANDLED failures.")

    resolved_spec_path = Path(spec_path).expanduser().resolve()
    spec_document = read_json_file(resolved_spec_path)
    failure_message = str(failure.get("failureMessage") or "")
    test_step = str(failure.get("testStep") or "")
    widget_id = _extract_widget_id_from_message(failure_message)
    widget_definition, widget_pointer = _find_widget_with_pointer(spec_document.get("root"), widget_id)
    event_name = _infer_event_name(failure_message, test_step, widget_definition)

    all_actions = _summarize_widget_actions(widget_definition, widget_pointer)
    matching_actions = [entry for entry in all_actions if not event_name or entry.get("on") == event_name]

    declared_vm_methods = _extract_declared_vm_methods(spec_document)
    method_guidance = _build_method_guidance(
        [
            action["callVM"]["method"]
            for action in matching_actions
            if isinstance(action.get("callVM"), dict) and action["callVM"].get("method")
        ],
        declared_vm_methods,
    )
    missing_vm_methods = [entry["method"] for entry in method_guidance if not entry["declared"]]

    support_surface_snapshot = support_surface if isinstance(support_surface, dict) else _ensure_surface_json() or {}
    relevant_support_surface = _extract_relevant_support_surface(support_surface_snapshot, widget_definition)
    missing_signals = _find_missing_signals(matching_actions, relevant_support_surface)
    widget_event_support = _evaluate_widget_event_support(
        widget_type=str(widget_definition.get("type") or "") if isinstance(widget_definition, dict) else "",
        event_name=event_name,
        widget_support_entry=relevant_support_surface.get("WidgetSupportEntry"),
    )
    primary_diagnosis = _classify_primary_diagnosis(
        widget_event_support=widget_event_support,
        matching_actions=matching_actions,
        missing_vm_methods=missing_vm_methods,
        missing_signals=missing_signals,
    )

    fallback_patch = _build_slate_event_fallback_patch(
        primary_diagnosis=primary_diagnosis,
        event_name=event_name,
        widget_pointer=widget_pointer,
        widget_definition=widget_definition,
        all_actions=all_actions,
    )
    if fallback_patch is not None:
        return {
            "ok": True,
            "handler": "slate_event_not_handled",
            "category": "SLATE_EVENT_NOT_HANDLED",
            "specPath": str(resolved_spec_path),
            "patch": fallback_patch,
            "request": None,
            "llmResponse": None,
            "context": {
                "widgetId": widget_id,
                "eventName": event_name,
                "widgetPointer": widget_pointer,
                "widgetDefinition": widget_definition,
                "widgetActions": all_actions,
                "matchingActions": matching_actions,
                "declaredVmMethods": declared_vm_methods,
                "methodGuidance": method_guidance,
                "missingVmMethods": missing_vm_methods,
                "missingSignals": missing_signals,
                "widgetEventSupport": widget_event_support,
                "supportSurface": relevant_support_surface,
                "primaryDiagnosis": primary_diagnosis,
                "fallback": True,
            },
        }

    prompt = render_llm_prompt(
        "slate_event_not_handled.j2",
        test_step=test_step,
        failure_message=failure_message,
        widget_id=widget_id or "",
        event_name=event_name or "",
        primary_diagnosis=primary_diagnosis,
        widget_pointer=widget_pointer or "",
        widget_definition=json.dumps(widget_definition, ensure_ascii=False, indent=2) if widget_definition is not None else "null",
        widget_actions=json.dumps(all_actions, ensure_ascii=False, indent=2),
        matching_actions=json.dumps(matching_actions, ensure_ascii=False, indent=2),
        declared_vm_methods=json.dumps(declared_vm_methods, ensure_ascii=False, indent=2),
        method_guidance=json.dumps(method_guidance, ensure_ascii=False, indent=2),
        missing_signals=json.dumps(missing_signals, ensure_ascii=False, indent=2),
        widget_event_support=json.dumps(widget_event_support, ensure_ascii=False, indent=2),
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

    return {
        "ok": True,
        "handler": "slate_event_not_handled",
        "category": "SLATE_EVENT_NOT_HANDLED",
        "specPath": str(resolved_spec_path),
        "patch": patch,
        "request": request.model_dump(),
        "llmResponse": response.model_dump(),
        "context": {
            "widgetId": widget_id,
            "eventName": event_name,
            "widgetPointer": widget_pointer,
            "widgetDefinition": widget_definition,
            "widgetActions": all_actions,
            "matchingActions": matching_actions,
            "declaredVmMethods": declared_vm_methods,
            "methodGuidance": method_guidance,
            "missingVmMethods": missing_vm_methods,
            "missingSignals": missing_signals,
            "widgetEventSupport": widget_event_support,
            "supportSurface": relevant_support_surface,
            "primaryDiagnosis": primary_diagnosis,
        },
    }


def _extract_widget_id_from_message(message: str) -> str | None:
    match = re.search(r"widget '([^']+)'", message, flags=re.IGNORECASE)
    return match.group(1) if match else None


def _infer_event_name(failure_message: str, test_step: str, widget_definition: dict[str, Any] | None) -> str | None:
    combined_text = f"{failure_message} {test_step}".lower()
    if "double click" in combined_text or "double-click" in combined_text:
        return "DoubleClicked"
    if "long press" in combined_text or "long-press" in combined_text:
        return "LongPressed"
    if "unhover" in combined_text:
        return "Unhovered"
    if "hover" in combined_text:
        return "Hovered"
    if "press" in combined_text:
        return "Pressed"
    if "release" in combined_text:
        return "Released"
    if "click" in combined_text:
        return "Clicked"

    actions = widget_definition.get("actions") if isinstance(widget_definition, dict) else None
    if isinstance(actions, list) and len(actions) == 1 and isinstance(actions[0], dict):
        event_name = actions[0].get("on")
        if isinstance(event_name, str) and event_name:
            return event_name
    return None


def _find_widget_with_pointer(node: Any, widget_id: str | None, pointer: str = "/root") -> tuple[dict[str, Any] | None, str | None]:
    if not widget_id or not isinstance(node, dict):
        return None, None
    if node.get("id") == widget_id:
        return node, pointer

    for index, child in enumerate(node.get("children") or []):
        child_pointer = f"{pointer}/children/{index}"
        result_node, result_pointer = _find_widget_with_pointer(child, widget_id, child_pointer)
        if result_node is not None:
            return result_node, result_pointer

    return None, None


def _summarize_widget_actions(widget_definition: dict[str, Any] | None, widget_pointer: str | None) -> list[dict[str, Any]]:
    if not isinstance(widget_definition, dict):
        return []

    widget_actions = widget_definition.get("actions")
    if not isinstance(widget_actions, list):
        return []

    summaries: list[dict[str, Any]] = []
    for index, action in enumerate(widget_actions):
        if not isinstance(action, dict):
            continue
        action_pointer = f"{widget_pointer}/actions/{index}" if widget_pointer else None
        summary: dict[str, Any] = {
            "pointer": action_pointer,
            "on": action.get("on"),
            "guard": action.get("guard"),
            "raw": action,
        }

        do_value = action.get("do")
        if isinstance(do_value, str):
            summary["doKind"] = "string"
            summary["doValue"] = do_value
            call_vm = _extract_vm_call(do_value)
            if call_vm is not None:
                summary["callVM"] = call_vm
        elif isinstance(do_value, dict):
            summary["doKind"] = "block"
            summary["doValue"] = do_value
            call_vm_value = do_value.get("callVM")
            if isinstance(call_vm_value, str):
                summary["callVM"] = _extract_vm_call(call_vm_value)
            emit_signal_value = do_value.get("emitSignal")
            if isinstance(emit_signal_value, str) and emit_signal_value:
                summary["emitSignal"] = emit_signal_value
            set_state_value = do_value.get("setState")
            if isinstance(set_state_value, str) and set_state_value:
                summary["setState"] = set_state_value
        else:
            summary["doKind"] = type(do_value).__name__
            summary["doValue"] = do_value

        summaries.append(summary)

    return summaries


def _extract_vm_call(raw_expression: str) -> dict[str, Any] | None:
    expression = raw_expression.strip()
    if not expression.lower().startswith("vm."):
        return None

    method_portion = expression[3:]
    method_name = method_portion.split("(", 1)[0].strip()
    if not method_name:
        return None

    return {
        "expression": expression,
        "method": method_name,
    }


def _extract_declared_vm_methods(spec_document: Any) -> list[str]:
    if not isinstance(spec_document, dict):
        return []
    view_model = spec_document.get("viewModel")
    if not isinstance(view_model, dict):
        return []

    declared_methods: list[str] = []
    for entry in view_model.get("methods") or []:
        if isinstance(entry, dict):
            name = entry.get("name")
            if isinstance(name, str) and name:
                declared_methods.append(name)
    return declared_methods


def _build_method_guidance(referenced_methods: list[str], declared_methods: list[str]) -> list[dict[str, Any]]:
    guidance: list[dict[str, Any]] = []
    unique_referenced_methods: list[str] = []
    for method_name in referenced_methods:
        if method_name not in unique_referenced_methods:
            unique_referenced_methods.append(method_name)

    for method_name in unique_referenced_methods:
        declared = method_name in declared_methods
        ranked = sorted(
            (
                {
                    "candidate": declared_method,
                    "distance": _levenshtein_distance(method_name.lower(), declared_method.lower()),
                }
                for declared_method in declared_methods
            ),
            key=lambda item: (item["distance"], item["candidate"].lower()),
        )
        guidance.append(
            {
                "method": method_name,
                "declared": declared,
                "nearestDeclared": ranked[:3],
            }
        )

    return guidance


def _extract_relevant_support_surface(
    support_surface: dict[str, Any],
    widget_definition: dict[str, Any] | None,
) -> dict[str, Any]:
    widget_type = str(widget_definition.get("type") or "") if isinstance(widget_definition, dict) else ""
    widget_entry = _find_widget_support_entry(support_surface, widget_type)
    return {
        "ActionEvents": _get_support_value(support_surface, "ActionEvents", "actionEvents", default=[]),
        "ActionPrimitives": _get_support_value(support_surface, "ActionPrimitives", "actionPrimitives", default=[]),
        "Signals": _get_support_value(support_surface, "Signals", "signals", default=[]),
        "WidgetSupportEntry": widget_entry,
    }


def _find_widget_support_entry(support_surface: dict[str, Any], widget_type: str) -> dict[str, Any] | None:
    if not widget_type:
        return None

    for collection_key in (
        ("BuiltInUMGWidgets", "builtInUMGWidgets"),
        ("CommonUIWidgets", "commonUIWidgets"),
        ("RegisteredComponents", "registeredComponents"),
    ):
        entries = _get_support_value(support_surface, *collection_key, default=[])
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            entry_type = entry.get("Type") or entry.get("type")
            if isinstance(entry_type, str) and entry_type == widget_type:
                return entry
    return None


def _get_support_value(support_surface: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in support_surface:
            return support_surface[key]
    return default


def _find_missing_signals(matching_actions: list[dict[str, Any]], relevant_support_surface: dict[str, Any]) -> list[str]:
    declared_signals = {
        str(entry.get("Name") or entry.get("name"))
        for entry in relevant_support_surface.get("Signals") or []
        if isinstance(entry, dict) and (entry.get("Name") or entry.get("name"))
    }
    missing_signals: list[str] = []
    for action in matching_actions:
        signal_name = action.get("emitSignal")
        if isinstance(signal_name, str) and signal_name and signal_name not in declared_signals:
            missing_signals.append(signal_name)
    return missing_signals


def _evaluate_widget_event_support(
    *,
    widget_type: str,
    event_name: str | None,
    widget_support_entry: dict[str, Any] | None,
) -> dict[str, Any]:
    if not widget_type or not event_name:
        return {
            "likelySupported": None,
            "reason": "Widget type or event name is unavailable.",
        }

    event_support_map = {
        "Clicked": {"Button", "CheckBox", "CommonButtonBase"},
        "Hovered": {"Button", "CommonButtonBase"},
        "Unhovered": {"Button", "CommonButtonBase"},
        "Pressed": {"Button", "CommonButtonBase"},
        "Released": {"Button", "CommonButtonBase"},
        "DoubleClicked": {"Button", "CommonButtonBase"},
        "LongPressed": {"Button", "CommonButtonBase"},
    }
    supported_types = event_support_map.get(event_name, set())
    if widget_type in supported_types:
        return {
            "likelySupported": True,
            "reason": f"{widget_type} is a known {event_name} host in the current compiler/runtime pipeline.",
        }

    if "Button" in widget_type:
        return {
            "likelySupported": True,
            "reason": f"{widget_type} looks button-like, so {event_name} is likely supported.",
        }

    if isinstance(widget_support_entry, dict):
        class_path = str(widget_support_entry.get("classPath") or "")
        if "Button" in class_path:
            return {
                "likelySupported": True,
                "reason": f"{class_path} looks button-like, so {event_name} is likely supported.",
            }

    return {
        "likelySupported": False,
        "reason": f"{widget_type} is not a known {event_name} host in the current compiler/runtime pipeline.",
    }


def _classify_primary_diagnosis(
    *,
    widget_event_support: dict[str, Any],
    matching_actions: list[dict[str, Any]],
    missing_vm_methods: list[str],
    missing_signals: list[str],
) -> str:
    if widget_event_support.get("likelySupported") is False:
        return "unsupported_widget_event"
    if not matching_actions:
        return "missing_action_for_event"
    if missing_vm_methods:
        return "vm_method_typo_or_missing"
    if missing_signals:
        return "missing_signal"
    return "unknown_action_wiring_issue"


def _build_slate_event_fallback_patch(
    *,
    primary_diagnosis: str,
    event_name: str | None,
    widget_pointer: str | None,
    widget_definition: dict[str, Any] | None,
    all_actions: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if primary_diagnosis == "unsupported_widget_event":
        return _build_unsupported_widget_fallback_patch(event_name=event_name, widget_pointer=widget_pointer)
    if primary_diagnosis == "missing_action_for_event":
        return _build_missing_action_for_event_patch(
            event_name=event_name,
            widget_pointer=widget_pointer,
            widget_definition=widget_definition,
            all_actions=all_actions,
        )
    return None


def _build_unsupported_widget_fallback_patch(*, event_name: str | None, widget_pointer: str | None) -> list[dict[str, Any]] | None:
    if not widget_pointer or event_name not in {"Clicked", "Hovered", "Unhovered", "Pressed", "Released", "DoubleClicked", "LongPressed"}:
        return None
    return [
        {
            "op": "replace",
            "path": f"{widget_pointer}/type",
            "value": "Button",
        }
    ]


def _build_missing_action_for_event_patch(
    *,
    event_name: str | None,
    widget_pointer: str | None,
    widget_definition: dict[str, Any] | None,
    all_actions: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if not event_name or not widget_pointer or not isinstance(widget_definition, dict) or not all_actions:
        return None

    raw_actions = widget_definition.get("actions")
    if not isinstance(raw_actions, list):
        return None

    source_action = all_actions[0].get("raw")
    if not isinstance(source_action, dict):
        return None

    new_action = dict(source_action)
    new_action["on"] = event_name
    return [
        {
            "op": "add",
            "path": f"{widget_pointer}/actions/{len(raw_actions)}",
            "value": new_action,
        }
    ]


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            substitution_cost = 0 if left_char == right_char else 1
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]
