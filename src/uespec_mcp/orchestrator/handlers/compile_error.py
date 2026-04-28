from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ...llm.types import LLMRequest
from ...llm.base import LLMClient
from ...llm.prompts import render as render_llm_prompt
from ...tools.surface import _ensure_surface_json
from .._json_io import read_json_file
from ._json_patch import anchor_patch_paths, json_patch_schema, parse_patch_content, validate_patch_array


JSONPath = list[str | int]


def propose_patch_for_compile_error(
    *,
    spec_path: str | Path,
    failure: dict[str, Any],
    llm_client: LLMClient,
    support_surface: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (failure.get("category") or "").strip().upper() != "COMPILE_ERROR":
        raise ValueError("compile_error handler only accepts COMPILE_ERROR failures.")

    resolved_spec_path = Path(spec_path).expanduser().resolve()
    spec_document = read_json_file(resolved_spec_path)

    error_code = str(failure.get("compilerErrorCode") or "UNKNOWN_COMPILE_ERROR")
    error_path = str(failure.get("testStep") or "$")
    error_message = str(failure.get("failureMessage") or "Unknown compile error.")
    nearest_declared = [str(value) for value in failure.get("nearestDeclared") or []]

    support_surface_snapshot = support_surface if isinstance(support_surface, dict) else _ensure_surface_json() or {}
    relevant_support_surface = _extract_relevant_support_surface_section(error_code, support_surface_snapshot)
    spec_snippet = _extract_spec_snippet(spec_document, error_path)
    prompt = render_llm_prompt(
        "compile_error.j2",
        error_code=error_code,
        error_path=error_path,
        error_message=error_message,
        error_hint=str(failure.get("hint") or ""),
        nearest_declared=nearest_declared,
        spec_snippet_around_path=json.dumps(spec_snippet, ensure_ascii=False, indent=2),
        support_surface_relevant_section=json.dumps(relevant_support_surface, ensure_ascii=False, indent=2),
    )

    request = LLMRequest(
        system_prompt="You are a UISpec autofix worker. Output valid RFC 6902 JSON Patch only.",
        user_prompt=prompt,
        expected_output_schema=json_patch_schema(),
        temperature=0.1,
        max_tokens=2000,
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
        anchor_pointers=_build_anchor_pointers_for_error_path(error_path),
    )

    return {
        "ok": True,
        "handler": "compile_error",
        "category": "COMPILE_ERROR",
        "specPath": str(resolved_spec_path),
        "patch": patch,
        "request": request.model_dump(),
        "llmResponse": response.model_dump(),
        "context": {
            "errorCode": error_code,
            "errorPath": error_path,
            "nearestDeclared": nearest_declared,
            "specSnippet": spec_snippet,
            "supportSurface": relevant_support_surface,
        },
    }
def _extract_relevant_support_surface_section(error_code: str, support_surface: dict[str, Any]) -> dict[str, Any]:
    code = error_code.upper()
    if "CONVERTER" in code:
        return {"Converters": support_surface.get("Converters", [])}
    if "TOKEN" in code or "STYLE" in code:
        return {"StyleTokens": support_surface.get("StyleTokens", [])}
    if "SIGNAL" in code:
        return {"Signals": support_surface.get("Signals", [])}
    if "WIDGET" in code or "COMPONENT" in code or "NAV" in code or "TARGET" in code or "STATE" in code:
        return {
            "BuiltInUMGWidgets": support_surface.get("BuiltInUMGWidgets", []),
            "CommonUIWidgets": support_surface.get("CommonUIWidgets", []),
            "RegisteredComponents": support_surface.get("RegisteredComponents", []),
        }
    return support_surface


def _extract_spec_snippet(spec_document: dict[str, Any], error_path: str) -> Any:
    path_segments = _parse_json_path(error_path)
    current: Any = spec_document
    for segment in path_segments:
        if isinstance(segment, int):
            if not isinstance(current, list) or segment >= len(current):
                return spec_document
            current = current[segment]
            continue
        if not isinstance(current, dict) or segment not in current:
            return spec_document
        current = current[segment]
    return current


def _parse_json_path(path: str) -> JSONPath:
    normalized = path.strip()
    if not normalized or normalized == "$":
        return []
    if normalized.startswith("$."):
        normalized = normalized[2:]
    elif normalized.startswith("$"):
        normalized = normalized[1:]

    result: JSONPath = []
    buffer = ""
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char == ".":
            if buffer:
                result.append(buffer)
                buffer = ""
            index += 1
            continue
        if char == "[":
            if buffer:
                result.append(buffer)
                buffer = ""
            end_index = normalized.find("]", index)
            if end_index == -1:
                break
            token = normalized[index + 1 : end_index]
            if token.isdigit():
                result.append(int(token))
            elif token.startswith("'") and token.endswith("'"):
                result.append(token[1:-1])
            elif token.startswith('"') and token.endswith('"'):
                result.append(token[1:-1])
            else:
                result.append(token)
            index = end_index + 1
            continue
        buffer += char
        index += 1

    if buffer:
        result.append(buffer)
    return result


def _build_anchor_pointers_for_error_path(error_path: str) -> list[str]:
    segments = _parse_json_path(error_path)
    if not segments:
        return []

    anchors: list[str] = []
    current = ""
    for segment in segments[:-1]:
        token = str(segment).replace("~", "~0").replace("/", "~1")
        current = f"{current}/{token}" if current else f"/{token}"
        anchors.append(current)
    return anchors
