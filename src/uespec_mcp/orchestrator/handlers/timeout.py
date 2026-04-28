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


def propose_patch_for_timeout(
    *,
    spec_path: str | Path,
    failure: dict[str, Any],
    llm_client: LLMClient,
    support_surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (failure.get("category") or "").strip().upper() != "TIMEOUT":
        raise ValueError("timeout handler only accepts TIMEOUT failures.")

    resolved_spec_path = Path(spec_path).expanduser().resolve()
    spec_document = read_json_file(resolved_spec_path)
    test_spec_document = _load_test_spec_document(failure.get("specPath"), resolved_spec_path)
    failure_message = str(failure.get("failureMessage") or "")
    test_step = str(failure.get("testStep") or "")
    expected_value = str(failure.get("expectedValue") or "")
    actual_value = str(failure.get("actualValue") or "")
    state_machine_snapshot = _normalize_snapshot(failure.get("stateMachineSnapshot"))
    widget_tree_snapshot = _normalize_snapshot(failure.get("widgetTreeSnapshot"))
    missing_ids = [str(value) for value in failure.get("missingIds") or []]

    timeout_kind = _infer_timeout_kind(
        failure_message=failure_message,
        state_machine_snapshot=state_machine_snapshot,
        missing_ids=missing_ids,
    )
    widget_id = _extract_widget_id(failure_message, missing_ids)
    widget_definition, widget_pointer = _find_widget_with_pointer(spec_document.get("root"), widget_id)
    wait_reference_sites = _find_wait_reference_sites(test_spec_document or spec_document, timeout_kind, widget_id, expected_value)
    parameter_context = _extract_parameter_context(test_spec_document or spec_document)
    arrange_context = _extract_arrange_context(test_spec_document or spec_document)
    signal_candidates = _extract_signal_candidates(test_spec_document or spec_document)

    state_context = _build_state_context(spec_document, expected_value, actual_value, state_machine_snapshot)
    visibility_context = _build_visibility_context(widget_definition, widget_pointer, widget_tree_snapshot)

    support_surface_snapshot = support_surface if isinstance(support_surface, dict) else _ensure_surface_json() or {}
    relevant_support_surface = _extract_relevant_support_surface(timeout_kind, support_surface_snapshot, signal_candidates)
    primary_diagnosis = _classify_timeout_diagnosis(
        timeout_kind=timeout_kind,
        expected_value=expected_value,
        widget_definition=widget_definition,
        state_context=state_context,
    )

    fallback_patch = _build_timeout_fallback_patch(
        timeout_kind=timeout_kind,
        expected_value=expected_value,
        widget_pointer=widget_pointer,
        widget_definition=widget_definition,
        primary_diagnosis=primary_diagnosis,
        state_context=state_context,
    )
    if fallback_patch is not None:
        return {
            "ok": True,
            "handler": "timeout",
            "category": "TIMEOUT",
            "specPath": str(resolved_spec_path),
            "patch": fallback_patch,
            "request": None,
            "llmResponse": None,
            "context": {
                "timeoutKind": timeout_kind,
                "widgetId": widget_id,
                "waitReferenceSites": wait_reference_sites,
                "parameterContext": parameter_context,
                "arrangeContext": arrange_context,
                "stateContext": state_context,
                "visibilityContext": visibility_context,
                "supportSurface": relevant_support_surface,
                "primaryDiagnosis": primary_diagnosis,
                "fallback": True,
            },
        }

    prompt = render_llm_prompt(
        _template_name_for_timeout_kind(timeout_kind),
        timeout_kind=timeout_kind,
        test_step=test_step,
        failure_message=failure_message,
        expected_value=expected_value,
        actual_value=actual_value,
        widget_id=widget_id or "",
        wait_reference_sites=json.dumps(wait_reference_sites, ensure_ascii=False, indent=2),
        parameter_context=json.dumps(parameter_context, ensure_ascii=False, indent=2),
        arrange_context=json.dumps(arrange_context, ensure_ascii=False, indent=2),
        state_context=json.dumps(state_context, ensure_ascii=False, indent=2),
        visibility_context=json.dumps(visibility_context, ensure_ascii=False, indent=2),
        relevant_support_surface=json.dumps(relevant_support_surface, ensure_ascii=False, indent=2),
        primary_diagnosis=primary_diagnosis,
    )

    request = LLMRequest(
        system_prompt="You are a UISpec autofix worker. Output valid RFC 6902 JSON Patch only.",
        user_prompt=prompt,
        expected_output_schema=json_patch_schema(),
        temperature=0.1,
        max_tokens=2600,
    )
    response = llm_client.complete(request)

    patch = response.parsed_output
    if patch is None:
        patch = parse_patch_content(response.content)
    if not isinstance(patch, list):
        raise ValueError("LLM output did not produce a JSON Patch array.")

    validate_patch_array(patch)
    patch = _drop_non_target_patch_ops(patch, spec_document)
    if not patch:
        raise ValueError("LLM patch did not contain any operations applicable to the target UISpec document.")

    return {
        "ok": True,
        "handler": "timeout",
        "category": "TIMEOUT",
        "specPath": str(resolved_spec_path),
        "patch": patch,
        "request": request.model_dump(),
        "llmResponse": response.model_dump(),
        "context": {
            "timeoutKind": timeout_kind,
            "widgetId": widget_id,
            "waitReferenceSites": wait_reference_sites,
            "parameterContext": parameter_context,
            "arrangeContext": arrange_context,
            "stateContext": state_context,
            "visibilityContext": visibility_context,
            "supportSurface": relevant_support_surface,
            "primaryDiagnosis": primary_diagnosis,
        },
    }


def _template_name_for_timeout_kind(timeout_kind: str) -> str:
    return {
        "state": "timeout_state.j2",
        "visibility": "timeout_visibility.j2",
    }.get(timeout_kind, "timeout_generic.j2")


def _load_test_spec_document(failure_spec_path: Any, resolved_spec_path: Path) -> dict[str, Any] | None:
    if not failure_spec_path:
        return None

    candidate_path = Path(str(failure_spec_path)).expanduser().resolve()
    if candidate_path == resolved_spec_path or not candidate_path.is_file():
        return None

    document = read_json_file(candidate_path)
    return document if isinstance(document, dict) else None


def _normalize_snapshot(snapshot: Any) -> dict[str, Any]:
    if isinstance(snapshot, dict):
        return snapshot
    if isinstance(snapshot, str):
        try:
            parsed = json.loads(snapshot)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _infer_timeout_kind(*, failure_message: str, state_machine_snapshot: dict[str, Any], missing_ids: list[str]) -> str:
    normalized_message = failure_message.lower()
    if "did not reach visible=" in normalized_message or missing_ids:
        return "visibility"
    if "did not reach state" in normalized_message:
        return "state"
    if state_machine_snapshot.get("currentState"):
        return "state"
    return "generic"


def _extract_widget_id(failure_message: str, missing_ids: list[str]) -> str | None:
    if missing_ids:
        return missing_ids[0]
    match = re.search(r"Widget '([^']+)'", failure_message)
    return match.group(1) if match else None


def _find_wait_reference_sites(
    spec_document: Any,
    timeout_kind: str,
    widget_id: str | None,
    expected_value: str,
) -> list[dict[str, Any]]:
    candidate_keys = {
        "state": {"waitForState"},
        "visibility": {"waitForVisible"},
        "generic": {"waitForState", "waitForVisible"},
    }.get(timeout_kind, {"waitForState", "waitForVisible"})

    matches: list[dict[str, Any]] = []

    def walk(node: Any, pointer: str = "") -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_pointer = f"{pointer}/{_escape_json_pointer_token(key)}"
                if key in candidate_keys and isinstance(value, dict) and _matches_wait_reference(value, timeout_kind, widget_id, expected_value):
                    matches.append({"pointer": child_pointer or "/", "waitStep": value})
                walk(value, child_pointer)
            return
        if isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{pointer}/{index}")

    walk(spec_document)
    return matches


def _matches_wait_reference(wait_object: dict[str, Any], timeout_kind: str, widget_id: str | None, expected_value: str) -> bool:
    if timeout_kind == "state":
        return str(wait_object.get("state") or "") == expected_value
    if timeout_kind == "visibility":
        return widget_id is not None and str(wait_object.get("id") or "") == widget_id
    if widget_id and str(wait_object.get("id") or "") == widget_id:
        return True
    return str(wait_object.get("state") or "") == expected_value


def _extract_parameter_context(spec_document: Any) -> list[dict[str, Any]]:
    if not isinstance(spec_document, dict):
        return []
    test_spec = spec_document.get("testSpec")
    if not isinstance(test_spec, dict):
        return []

    parameters = test_spec.get("parameters")
    return parameters if isinstance(parameters, list) else []


def _extract_arrange_context(spec_document: Any) -> list[dict[str, Any]]:
    if not isinstance(spec_document, dict):
        return []
    test_spec = spec_document.get("testSpec")
    if not isinstance(test_spec, dict):
        return []
    phases = test_spec.get("phases")
    if not isinstance(phases, dict):
        return []

    arrange = phases.get("arrange")
    return arrange if isinstance(arrange, list) else []


def _extract_signal_candidates(spec_document: Any) -> list[str]:
    if not isinstance(spec_document, dict):
        return []
    test_spec = spec_document.get("testSpec")
    if not isinstance(test_spec, dict):
        return []
    phases = test_spec.get("phases")
    if not isinstance(phases, dict):
        return []

    candidates: list[str] = []
    for phase_name in ("arrange", "act"):
        steps = phases.get(phase_name)
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            signal_name = step.get("emitSignal")
            if isinstance(signal_name, str) and signal_name and signal_name not in candidates:
                candidates.append(signal_name)
    return candidates


def _drop_non_target_patch_ops(patch: list[dict[str, Any]], spec_document: Any) -> list[dict[str, Any]]:
    if not isinstance(spec_document, dict):
        return patch

    foreign_roots = {"testSpec", "assume", "arrange", "act", "assert", "parameters"}
    filtered_patch: list[dict[str, Any]] = []
    for operation in patch:
        path = operation.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            filtered_patch.append(operation)
            continue

        top_level_key = path.split("/", 2)[1]
        if top_level_key in foreign_roots and top_level_key not in spec_document:
            continue

        filtered_patch.append(operation)

    return filtered_patch


def _build_state_context(
    spec_document: Any,
    expected_value: str,
    actual_value: str,
    state_machine_snapshot: dict[str, Any],
) -> dict[str, Any]:
    state_machine = spec_document.get("stateMachine") if isinstance(spec_document, dict) else None
    state_entries = _extract_state_entries(state_machine)
    current_state = str(state_machine_snapshot.get("currentState") or actual_value or "")
    expected_state = expected_value

    current_state_entry = next((entry for entry in state_entries if entry["name"] == current_state), None)
    transitions_to_expected = [
        {
            "from": entry["name"],
            "pointer": f"{entry['pointer']}/transitions/{index}",
            "transition": transition,
        }
        for entry in state_entries
        for index, transition in enumerate(entry["definition"].get("transitions") or [])
        if isinstance(transition, dict) and str(transition.get("to") or "") == expected_state
    ]
    current_state_transitions = [
        {
            "pointer": f"{current_state_entry['pointer']}/transitions/{index}",
            "transition": transition,
        }
        for index, transition in enumerate((current_state_entry or {}).get("definition", {}).get("transitions") or [])
        if isinstance(transition, dict)
    ]

    return {
        "currentState": current_state,
        "expectedState": expected_state,
        "snapshot": state_machine_snapshot,
        "stateMachine": state_machine,
        "stateEntries": state_entries,
        "currentStateEntry": current_state_entry,
        "currentStateTransitions": current_state_transitions,
        "transitionsToExpected": transitions_to_expected,
    }


def _extract_state_entries(state_machine: Any) -> list[dict[str, Any]]:
    if not isinstance(state_machine, dict):
        return []

    states = state_machine.get("states")
    entries: list[dict[str, Any]] = []
    if isinstance(states, list):
        for index, state_definition in enumerate(states):
            if not isinstance(state_definition, dict):
                continue
            state_name = str(state_definition.get("name") or "")
            if not state_name:
                continue
            entries.append(
                {
                    "name": state_name,
                    "pointer": f"/stateMachine/states/{index}",
                    "definition": state_definition,
                }
            )
        return entries

    if isinstance(states, dict):
        for state_name, state_definition in states.items():
            if not isinstance(state_definition, dict):
                continue
            normalized_definition = dict(state_definition)
            normalized_definition.setdefault("name", state_name)
            entries.append(
                {
                    "name": str(state_name),
                    "pointer": f"/stateMachine/states/{_escape_json_pointer_token(str(state_name))}",
                    "definition": normalized_definition,
                }
            )
    return entries


def _build_visibility_context(
    widget_definition: dict[str, Any] | None,
    widget_pointer: str | None,
    widget_tree_snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "widgetPointer": widget_pointer,
        "widgetDefinition": widget_definition,
        "visibilityBinding": widget_definition.get("visibility") if isinstance(widget_definition, dict) else None,
        "widgetTreeSnapshot": widget_tree_snapshot,
        "widgetTreeCompact": _compress_widget_tree_snapshot(widget_tree_snapshot),
    }


def _compress_widget_tree_snapshot(widget_tree_snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    root = widget_tree_snapshot.get("root") if isinstance(widget_tree_snapshot, dict) else None
    result: list[dict[str, Any]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        result.append(
            {
                "id": str(node.get("id") or ""),
                "type": str(node.get("type") or ""),
                "visible": node.get("visible"),
                "visibility": node.get("visibility"),
            }
        )
        for child in node.get("children") or []:
            walk(child)

    walk(root)
    return result


def _extract_relevant_support_surface(timeout_kind: str, support_surface: dict[str, Any], signal_candidates: list[str]) -> dict[str, Any]:
    if timeout_kind == "state":
        signals = _get_support_value(support_surface, "Signals", "signals", default=[])
        normalized_signals = list(signals) if isinstance(signals, list) else []
        for signal_name in signal_candidates:
            if signal_name not in normalized_signals:
                normalized_signals.append(signal_name)
        return {
            "Signals": normalized_signals,
            "ActionEvents": _get_support_value(support_surface, "ActionEvents", "actionEvents", default=[]),
        }
    if timeout_kind == "visibility":
        return {
            "Converters": _get_support_value(support_surface, "Converters", "converters", default=[]),
            "StyleTokens": _get_support_value(support_surface, "StyleTokens", "styleTokens", default=[]),
        }
    return support_surface


def _classify_timeout_diagnosis(
    *,
    timeout_kind: str,
    expected_value: str,
    widget_definition: dict[str, Any] | None,
    state_context: dict[str, Any],
) -> str:
    if timeout_kind == "state":
        state_entries = state_context.get("stateEntries") or []
        state_names = {entry["name"] for entry in state_entries if isinstance(entry, dict) and entry.get("name")}
        if expected_value and expected_value not in state_names:
            return "missing_expected_state"
        if state_context.get("currentState") and not state_context.get("currentStateTransitions"):
            return "missing_transition_from_current_state"
        if expected_value and not state_context.get("transitionsToExpected"):
            return "missing_transition_to_expected_state"
        return "stuck_transition_condition"

    if timeout_kind == "visibility":
        if widget_definition is None:
            return "missing_widget_definition"
        if not widget_definition.get("visibility"):
            return "no_visibility_binding"
        return "visibility_condition_never_satisfied"

    return "generic_timeout"


def _build_timeout_fallback_patch(
    *,
    timeout_kind: str,
    expected_value: str,
    widget_pointer: str | None,
    widget_definition: dict[str, Any] | None,
    primary_diagnosis: str,
    state_context: dict[str, Any],
) -> list[dict[str, Any]] | None:
    if timeout_kind == "visibility":
        return _build_visibility_literal_fallback_patch(
            expected_value=expected_value,
            widget_pointer=widget_pointer,
            widget_definition=widget_definition,
        )
    if timeout_kind == "state" and primary_diagnosis == "missing_expected_state":
        return _build_missing_expected_state_patch(state_context)
    return None


def _build_visibility_literal_fallback_patch(
    *,
    expected_value: str,
    widget_pointer: str | None,
    widget_definition: dict[str, Any] | None,
) -> list[dict[str, Any]] | None:
    if not widget_pointer or not isinstance(widget_definition, dict):
        return None

    visibility_value = widget_definition.get("visibility")
    if not isinstance(visibility_value, str):
        return None

    normalized_expected = expected_value.strip().lower()
    desired_visibility = None
    if normalized_expected == "true":
        desired_visibility = "Visible"
    elif normalized_expected == "false":
        desired_visibility = "Hidden"

    if not desired_visibility or visibility_value == desired_visibility:
        return None

    return [
        {
            "op": "replace",
            "path": f"{widget_pointer}/visibility",
            "value": desired_visibility,
        }
    ]


def _build_missing_expected_state_patch(state_context: dict[str, Any]) -> list[dict[str, Any]] | None:
    if not isinstance(state_context, dict):
        return None

    expected_state = str(state_context.get("expectedState") or "")
    state_machine = state_context.get("stateMachine")
    if not expected_state or not isinstance(state_machine, dict):
        return None

    states = state_machine.get("states")
    if isinstance(states, list):
        for state_definition in states:
            if isinstance(state_definition, dict) and str(state_definition.get("name") or "") == expected_state:
                return None
        return [
            {
                "op": "add",
                "path": f"/stateMachine/states/{len(states)}",
                "value": {
                    "name": expected_state,
                },
            }
        ]

    if isinstance(states, dict):
        if expected_state in states:
            return None
        return [
            {
                "op": "add",
                "path": f"/stateMachine/states/{_escape_json_pointer_token(expected_state)}",
                "value": {},
            }
        ]

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


def _get_support_value(support_surface: dict[str, Any], *keys: str, default: Any) -> Any:
    for key in keys:
        if key in support_surface:
            return support_surface[key]
    return default


def _escape_json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
