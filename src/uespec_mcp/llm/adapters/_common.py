from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel


def maybe_parse_json(content: str) -> dict[str, Any] | list[Any] | None:
    stripped = content.strip()
    if not stripped:
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, (dict, list)):
        return parsed
    return None


def normalize_model_output(value: Any) -> dict[str, Any] | list[Any] | None:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        return maybe_parse_json(value)
    return None


def model_dump_compat(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        return value.model_dump()
    if hasattr(value, "model_dump") and callable(value.model_dump):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for attribute_name in dir(value):
        if attribute_name.startswith("_"):
            continue
        try:
            attribute_value = getattr(value, attribute_name)
        except Exception:
            continue
        if callable(attribute_value):
            continue
        if isinstance(attribute_value, (str, int, float, bool, dict, list, type(None))):
            result[attribute_name] = attribute_value
    return result
