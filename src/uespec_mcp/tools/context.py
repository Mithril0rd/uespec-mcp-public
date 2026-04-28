from __future__ import annotations

from pathlib import Path
from typing import Any


_ACTIVE_SPEC: str | None = None


def set_active_spec(spec_path: str) -> dict[str, Any]:
    global _ACTIVE_SPEC
    resolved = Path(spec_path).expanduser().resolve()
    if not resolved.is_file():
        return {
            "ok": False,
            "errors": [
                {
                    "code": "SPEC_NOT_FOUND",
                    "path": "$.spec_path",
                    "message": f"Spec file '{resolved}' does not exist.",
                    "hint": "Pass an existing JSON spec path.",
                    "nearestDeclared": [],
                }
            ],
        }
    previous = _ACTIVE_SPEC
    _ACTIVE_SPEC = str(resolved)
    return {"ok": True, "activeSpec": _ACTIVE_SPEC, "previousActiveSpec": previous}


def get_active_spec() -> str | None:
    return _ACTIVE_SPEC


def resolve_spec_path(spec_path: str | None, *, must_exist: bool = True) -> Path:
    candidate = spec_path or _ACTIVE_SPEC
    if not candidate:
        raise ValueError("No spec path was provided and no active spec is set.")
    resolved = Path(candidate).expanduser().resolve()
    if must_exist and not resolved.is_file():
        raise FileNotFoundError(f"Spec file '{resolved}' does not exist.")
    return resolved


def resolve_base_dir(base_dir: str | None) -> Path:
    if base_dir:
        resolved = Path(base_dir).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Base directory '{resolved}' does not exist.")
        if not resolved.is_dir():
            raise NotADirectoryError(f"Base directory '{resolved}' is not a directory.")
        return resolved
    if _ACTIVE_SPEC:
        return Path(_ACTIVE_SPEC).parent
    return Path.cwd()


def register(mcp: Any) -> None:
    mcp.tool(name="set_active_spec", description="Set the session-scoped active UISpec path.")(set_active_spec)
    mcp.tool(name="get_active_spec", description="Return the current session-scoped active UISpec path.")(get_active_spec)
