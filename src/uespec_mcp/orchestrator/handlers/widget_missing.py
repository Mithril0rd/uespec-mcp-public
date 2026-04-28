from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...llm.base import LLMClient
from ...llm.prompts import render as render_llm_prompt
from ...llm.types import LLMRequest
from ...tools.surface import _ensure_surface_json
from .._json_io import read_json_file
from ._json_patch import json_patch_schema, parse_patch_content, validate_patch_array


def propose_patch_for_widget_missing(
    *,
    spec_path: str | Path,
    failure: dict[str, Any],
    llm_client: LLMClient,
    support_surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (failure.get("category") or "").strip().upper() != "WIDGET_MISSING":
        raise ValueError("widget_missing handler only accepts WIDGET_MISSING failures.")

    resolved_spec_path = Path(spec_path).expanduser().resolve()
    spec_document = read_json_file(resolved_spec_path)

    missing_ids = _collect_missing_ids(failure)

    widget_tree_snapshot = _normalize_widget_tree_snapshot(failure.get("widgetTreeSnapshot"))
    compressed_tree = _compress_widget_tree_snapshot(widget_tree_snapshot)
    typo_guidance = _build_typo_guidance(missing_ids, compressed_tree)
    repair_mode = _resolve_widget_missing_repair_mode(typo_guidance)
    reference_sites = _find_missing_id_reference_sites(spec_document, missing_ids)
    insertion_guidance = _build_widget_insertion_guidance(spec_document, missing_ids)
    support_surface_snapshot = support_surface if isinstance(support_surface, dict) else _ensure_surface_json() or {}
    widget_support_surface = _extract_widget_support_surface(support_surface_snapshot)

    prompt = render_llm_prompt(
        "widget_missing.j2",
        missing_ids=missing_ids,
        test_step=str(failure.get("testStep") or ""),
        failure_message=str(failure.get("failureMessage") or ""),
        reference_sites=json.dumps(reference_sites, ensure_ascii=False, indent=2),
        widget_tree_compact=json.dumps(compressed_tree, ensure_ascii=False, indent=2),
        widget_support_surface=json.dumps(widget_support_surface, ensure_ascii=False, indent=2),
        typo_guidance=json.dumps(typo_guidance, ensure_ascii=False, indent=2),
        repair_mode=repair_mode,
        insertion_guidance=json.dumps(insertion_guidance, ensure_ascii=False, indent=2),
    )

    request = LLMRequest(
        system_prompt="You are a UISpec autofix worker. Output valid RFC 6902 JSON Patch only.",
        user_prompt=prompt,
        expected_output_schema=json_patch_schema(),
        temperature=0.1,
        max_tokens=500,
    )
    response = llm_client.complete(request)

    patch = response.parsed_output
    if patch is None:
        patch = parse_patch_content(response.content)
    if not isinstance(patch, list):
        raise ValueError("LLM output did not produce a JSON Patch array.")

    patch = _hydrate_patch_from_insertion_guidance(patch, insertion_guidance)
    validate_patch_array(patch)

    return {
        "ok": True,
        "handler": "widget_missing",
        "category": "WIDGET_MISSING",
        "specPath": str(resolved_spec_path),
        "patch": patch,
        "request": request.model_dump(),
        "llmResponse": response.model_dump(),
        "context": {
            "missingIds": missing_ids,
            "referenceSites": reference_sites,
            "widgetTreeCompact": compressed_tree,
            "widgetSupportSurface": widget_support_surface,
            "typoGuidance": typo_guidance,
            "repairMode": repair_mode,
            "insertionGuidance": insertion_guidance,
        },
    }


def should_target_test_spec_for_widget_missing(
    failure: dict[str, Any],
    test_spec_document: dict[str, Any] | None,
) -> bool:
    if not isinstance(test_spec_document, dict):
        return False

    missing_ids = _collect_missing_ids(failure)
    if not missing_ids:
        return False

    reference_sites = _find_missing_id_reference_sites(test_spec_document, missing_ids)
    if not reference_sites:
        return False

    widget_tree_snapshot = _normalize_widget_tree_snapshot(failure.get("widgetTreeSnapshot"))
    compressed_tree = _compress_widget_tree_snapshot(widget_tree_snapshot)
    typo_guidance = _build_typo_guidance(missing_ids, compressed_tree)
    repair_mode = _resolve_widget_missing_repair_mode(typo_guidance)
    return repair_mode == "replace_reference"


def _normalize_widget_tree_snapshot(snapshot: Any) -> dict[str, Any]:
    if isinstance(snapshot, dict):
        return snapshot
    if isinstance(snapshot, str):
        try:
            parsed = json.loads(snapshot)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _compress_widget_tree_snapshot(widget_tree_snapshot: dict[str, Any]) -> list[dict[str, str]]:
    root = widget_tree_snapshot.get("root") if isinstance(widget_tree_snapshot, dict) else None
    result: list[dict[str, str]] = []

    def walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        node_id = str(node.get("id") or "")
        node_type = str(node.get("type") or "")
        if node_id or node_type:
            result.append({"id": node_id, "type": node_type})
        for child in node.get("children") or []:
            walk(child)

    walk(root)
    return result


def _collect_missing_ids(failure: dict[str, Any]) -> list[str]:
    missing_ids = [str(value) for value in failure.get("missingIds") or [] if str(value)]
    if missing_ids:
        return missing_ids

    inferred_missing_id = _extract_missing_id_from_message(str(failure.get("failureMessage") or ""))
    if inferred_missing_id:
        return [inferred_missing_id]
    return []


def _build_typo_guidance(missing_ids: list[str], compressed_tree: list[dict[str, str]], max_distance: int = 3) -> list[dict[str, Any]]:
    declared_ids = [entry["id"] for entry in compressed_tree if entry.get("id")]
    guidance: list[dict[str, Any]] = []

    for missing_id in missing_ids:
        ranked = sorted(
            (
                {"candidate": declared_id, "distance": _levenshtein_distance(missing_id.lower(), declared_id.lower())}
                for declared_id in declared_ids
            ),
            key=lambda item: (item["distance"], item["candidate"].lower()),
        )
        close_matches = [item for item in ranked[:3] if item["distance"] <= max_distance]
        if close_matches:
            guidance.append(
                {
                    "missingId": missing_id,
                    "strategy": "replace_reference",
                    "nearestExisting": close_matches,
                }
            )
        else:
            guidance.append(
                {
                    "missingId": missing_id,
                    "strategy": "add_widget",
                    "nearestExisting": ranked[:3],
                }
            )

    return guidance


def _resolve_widget_missing_repair_mode(typo_guidance: list[dict[str, Any]]) -> str:
    strategies = {str(entry.get("strategy") or "") for entry in typo_guidance if entry.get("strategy")}
    if strategies == {"replace_reference"}:
        return "replace_reference"
    if strategies == {"add_widget"}:
        return "add_widget"
    return "mixed"


def _build_widget_insertion_guidance(spec_document: Any, missing_ids: list[str]) -> list[dict[str, Any]]:
    root = spec_document.get("root") if isinstance(spec_document, dict) else None
    root_children = root.get("children") if isinstance(root, dict) else None
    children_pointer = "/root/children/-" if isinstance(root_children, list) else "/root/children"
    guidance: list[dict[str, Any]] = []

    for missing_id in missing_ids:
        suggested_content = _infer_widget_content_from_id(missing_id)
        guidance.append(
            {
                "missingId": missing_id,
                "suggestedType": _infer_widget_type_from_id(missing_id),
                "suggestedParentPointer": "/root",
                "suggestedInsertPointer": children_pointer,
                "suggestedContent": suggested_content,
            }
        )

    return guidance


def _hydrate_patch_from_insertion_guidance(patch: list[dict[str, Any]], insertion_guidance: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guidance_by_id = {
        str(entry.get("missingId") or ""): entry
        for entry in insertion_guidance
        if isinstance(entry, dict) and entry.get("missingId")
    }

    hydrated_patch: list[dict[str, Any]] = []
    for operation in patch:
        if not isinstance(operation, dict):
            hydrated_patch.append(operation)
            continue

        value = operation.get("value")
        if not isinstance(value, dict):
            hydrated_patch.append(operation)
            continue

        missing_id = str(value.get("id") or "")
        guidance = guidance_by_id.get(missing_id)
        if not guidance:
            hydrated_patch.append(operation)
            continue

        suggested_content = guidance.get("suggestedContent")
        widget_type = str(value.get("type") or guidance.get("suggestedType") or "")
        if (
            suggested_content is not None
            and widget_type == "TextBlock"
            and "content" not in value
        ):
            updated_operation = dict(operation)
            updated_value = dict(value)
            updated_value["content"] = suggested_content
            updated_operation["value"] = updated_value
            hydrated_patch.append(updated_operation)
            continue

        hydrated_patch.append(operation)

    return hydrated_patch


def _find_missing_id_reference_sites(spec_document: Any, missing_ids: list[str]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    missing_id_set = set(missing_ids)

    def walk(node: Any, pointer: str = "", parent: Any = None) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                child_pointer = f"{pointer}/{_escape_json_pointer_token(key)}"
                if isinstance(value, str) and value in missing_id_set:
                    result.append(
                        {
                            "missingId": value,
                            "pointer": child_pointer or "/",
                            "parent": parent if isinstance(parent, (dict, list)) else node,
                            "container": node,
                        }
                    )
                walk(value, child_pointer, node)
            return

        if isinstance(node, list):
            for index, value in enumerate(node):
                child_pointer = f"{pointer}/{index}"
                if isinstance(value, str) and value in missing_id_set:
                    result.append(
                        {
                            "missingId": value,
                            "pointer": child_pointer or "/",
                            "parent": parent if isinstance(parent, (dict, list)) else node,
                            "container": node,
                        }
                    )
                walk(value, child_pointer, node)

    walk(spec_document)

    normalized_result: list[dict[str, Any]] = []
    for entry in result:
        normalized_result.append(
            {
                "missingId": entry["missingId"],
                "pointer": entry["pointer"],
                "container": entry["container"],
            }
        )
    return normalized_result


def _extract_widget_support_surface(support_surface: dict[str, Any]) -> dict[str, Any]:
    return {
        "BuiltInUMGWidgets": support_surface.get("BuiltInUMGWidgets", []),
        "CommonUIWidgets": support_surface.get("CommonUIWidgets", []),
        "RegisteredComponents": support_surface.get("RegisteredComponents", []),
    }


def _extract_missing_id_from_message(message: str) -> str | None:
    marker = "Widget '"
    start_index = message.find(marker)
    if start_index == -1:
        return None
    end_index = message.find("'", start_index + len(marker))
    if end_index == -1:
        return None
    return message[start_index + len(marker) : end_index]


def _infer_widget_type_from_id(missing_id: str) -> str:
    lowered = missing_id.lower()
    if "progress" in lowered or "cooldown" in lowered or lowered.endswith("bar") or "_bar" in lowered:
        return "ProgressBar"
    if "label" in lowered or "text" in lowered or "keybind" in lowered:
        return "TextBlock"
    if "anchor" in lowered or "container" in lowered or "frame" in lowered:
        return "Border"
    if "badge" in lowered or "icon" in lowered or "image" in lowered:
        return "Image"
    return "Border"


def _infer_widget_content_from_id(missing_id: str) -> str | None:
    lowered = missing_id.lower()
    if "keybind" in lowered:
        prefix = missing_id.split("_", 1)[0]
        if prefix.startswith("slot") and len(prefix) > 4:
            return prefix[4:]
    return None


def _escape_json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


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
