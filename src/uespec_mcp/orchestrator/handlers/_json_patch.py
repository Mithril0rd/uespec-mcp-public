from __future__ import annotations

import json
import re
from typing import Any

import jsonpatch
import jsonpointer


def parse_patch_content(content: str) -> list[dict[str, Any]] | None:
    stripped = content.strip()
    if not stripped:
        return None
    candidates = [stripped]

    fenced_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        candidates.append(fenced_match.group(1).strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, list) else None
    return None


def json_patch_schema() -> dict[str, Any]:
    value_schema: dict[str, Any] = {
        "type": ["string", "number", "integer", "boolean", "object", "array", "null"]
    }
    return {
        "type": "array",
        "items": {
            "type": "object",
            "required": ["op", "path"],
            "properties": {
                "op": {
                    "type": "string",
                    "enum": ["add", "remove", "replace", "move", "copy", "test"],
                },
                "path": {"type": "string"},
                "from": {"type": "string"},
                "value": value_schema,
            },
            "additionalProperties": False,
        },
    }


def validate_patch_array(patch: list[dict[str, Any]]) -> None:
    for operation in patch:
        _normalize_patch_path_field(operation, "path")
        _normalize_patch_path_field(operation, "from")
    jsonpatch.JsonPatch(patch)


def anchor_patch_paths(
    patch: list[dict[str, Any]],
    *,
    document: Any,
    anchor_pointers: list[str] | None = None,
) -> None:
    normalized_anchors = _normalize_anchor_pointers(anchor_pointers or [])
    if not normalized_anchors:
        return

    for operation in patch:
        operation_name = str(operation.get("op") or "").lower()
        for field_name in ("from", "path"):
            value = operation.get(field_name)
            if not isinstance(value, str) or not value.startswith("/"):
                continue
            if _pointer_applies(document, value, operation_name=operation_name, field_name=field_name):
                continue

            for anchor_pointer in normalized_anchors:
                candidate_pointer = _join_json_pointer(anchor_pointer, value)
                if candidate_pointer == value:
                    continue
                if _pointer_applies(document, candidate_pointer, operation_name=operation_name, field_name=field_name):
                    operation[field_name] = candidate_pointer
                    break


def _normalize_patch_path_field(operation: dict[str, Any], field_name: str) -> None:
    value = operation.get(field_name)
    if not isinstance(value, str):
        return
    normalized = _json_path_to_json_pointer(value)
    if normalized is not None:
        operation[field_name] = normalized


def _json_path_to_json_pointer(path: str) -> str | None:
    normalized = path.strip()
    if not normalized or normalized == "$":
        return ""
    if normalized.startswith("/"):
        return None
    if normalized.startswith("$."):
        normalized = normalized[2:]
    elif normalized.startswith("$"):
        normalized = normalized[1:]
    elif normalized.startswith("."):
        normalized = normalized[1:]
    elif "." not in normalized and "[" not in normalized:
        return None

    segments: list[str] = []
    buffer = ""
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char == ".":
            if buffer:
                segments.append(buffer)
                buffer = ""
            index += 1
            continue
        if char == "[":
            if buffer:
                segments.append(buffer)
                buffer = ""
            end_index = normalized.find("]", index)
            if end_index == -1:
                return None
            token = normalized[index + 1 : end_index]
            if (token.startswith("'") and token.endswith("'")) or (token.startswith('"') and token.endswith('"')):
                token = token[1:-1]
            segments.append(token)
            index = end_index + 1
            continue
        buffer += char
        index += 1

    if buffer:
        segments.append(buffer)

    escaped = [segment.replace("~", "~0").replace("/", "~1") for segment in segments]
    return "/" + "/".join(escaped)


def _normalize_anchor_pointers(anchor_pointers: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for anchor_pointer in anchor_pointers:
        if not isinstance(anchor_pointer, str):
            continue
        value = anchor_pointer.strip()
        if not value:
            continue
        pointer = _json_path_to_json_pointer(value) or value
        if not pointer.startswith("/"):
            continue
        if pointer in seen:
            continue
        seen.add(pointer)
        normalized.append(pointer.rstrip("/") or "/")
    return normalized


def _pointer_applies(document: Any, pointer: str, *, operation_name: str, field_name: str) -> bool:
    if field_name == "from":
        return _pointer_exists(document, pointer)
    if operation_name in {"replace", "remove", "test"}:
        return _pointer_exists(document, pointer)
    if operation_name in {"add", "copy", "move"}:
        return _pointer_parent_exists(document, pointer)
    return _pointer_exists(document, pointer)


def _pointer_exists(document: Any, pointer: str) -> bool:
    if pointer in {"", "/"}:
        return True
    try:
        jsonpointer.JsonPointer(pointer).resolve(document)
    except jsonpointer.JsonPointerException:
        return False
    return True


def _pointer_parent_exists(document: Any, pointer: str) -> bool:
    if pointer in {"", "/"}:
        return True
    parent_pointer = _split_parent_pointer(pointer)
    return _pointer_exists(document, parent_pointer)


def _split_parent_pointer(pointer: str) -> str:
    if not pointer or pointer == "/":
        return ""
    head, _, _ = pointer.rpartition("/")
    return head


def _join_json_pointer(anchor_pointer: str, pointer: str) -> str:
    if not anchor_pointer or anchor_pointer == "/":
        return pointer
    if not pointer or pointer == "/":
        return anchor_pointer
    return f"{anchor_pointer.rstrip('/')}{pointer}"
