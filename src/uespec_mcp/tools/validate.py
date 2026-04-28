from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator

from ..bridge import BridgeError, get_bridge, structured_error


_SCHEMA: dict[str, Any] | None = None
_VALIDATOR: Draft202012Validator | None = None


def validate_spec(spec_content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(spec_content)
    except json.JSONDecodeError as exc:
        return {
            "ok": False,
            "stage": "json",
            "normalizedJson": None,
            "errors": [
                structured_error(
                    "INVALID_JSON",
                    exc.msg,
                    path=f"$[{exc.lineno}:{exc.colno}]",
                    hint="Provide valid JSON text before attempting UE semantic validation.",
                )
            ],
            "semanticValidation": {"status": "skipped", "reason": "JSON parsing failed."},
        }

    validator = _get_validator()
    schema_errors = sorted(validator.iter_errors(parsed), key=lambda error: _jsonschema_path(error))
    if schema_errors:
        return {
            "ok": False,
            "stage": "schema",
            "normalizedJson": None,
            "errors": [_schema_error_to_structured_error(error) for error in schema_errors],
            "semanticValidation": {"status": "skipped", "reason": "Schema validation failed."},
        }

    normalized_json = json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)
    try:
        semantic = get_bridge().validate_spec_content(normalized_json)
    except BridgeError as exc:
        return {
            "ok": True,
            "stage": "schema",
            "normalizedJson": normalized_json,
            "errors": [],
            "semanticValidation": {"status": "skipped", "reason": str(exc)},
        }

    semantic["stage"] = "semantic"
    semantic.setdefault("semanticValidation", {"status": "completed"})
    if semantic.get("normalizedJson") is None:
        semantic["normalizedJson"] = normalized_json
    return semantic


def validate_text(spec_content: str) -> dict[str, Any]:
    return validate_spec(spec_content)


def _load_schema() -> dict[str, Any]:
    global _SCHEMA
    if _SCHEMA is None:
        schema_text = files("uespec_mcp").joinpath("schemas/uispec.schema.json").read_text(encoding="utf-8")
        _SCHEMA = json.loads(schema_text)
    return _SCHEMA


def _get_validator() -> Draft202012Validator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = Draft202012Validator(_load_schema())
    return _VALIDATOR


def _jsonschema_path(error: Any) -> str:
    parts = ["$"]
    for segment in error.absolute_path:
        if isinstance(segment, int):
            parts.append(f"[{segment}]")
        else:
            parts.append(f".{segment}")
    return "".join(parts)


def _schema_error_to_structured_error(error: Any) -> dict[str, Any]:
    hint = ""
    if error.validator == "required":
        missing_key = list(error.validator_value) if isinstance(error.validator_value, list) else []
        if missing_key:
            hint = f"Add the missing required key(s): {', '.join(missing_key)}."
        else:
            hint = "Add the missing required key."
    elif error.validator == "enum":
        hint = f"Allowed values: {', '.join(str(item) for item in error.validator_value)}."
    elif error.validator == "oneOf":
        hint = "Match one supported shape for this field."
    elif error.validator == "type":
        hint = f"Expected type: {error.validator_value}."

    return structured_error(
        "SCHEMA_VALIDATION_FAILED",
        error.message,
        path=_jsonschema_path(error),
        hint=hint,
    )


def register(mcp: Any) -> None:
    mcp.tool(name="validate_spec", description="Validate raw UISpec JSON against schema first, then Unreal semantic validation if available.")(validate_spec)
