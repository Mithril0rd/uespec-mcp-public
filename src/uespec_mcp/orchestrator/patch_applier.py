from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonpatch

from ._json_io import read_json_file, read_text_file


def apply_patch_to_spec_file(spec_path: str | Path, patch: list[dict[str, Any]]) -> dict[str, Any]:
    resolved_path = Path(spec_path).expanduser().resolve()
    original_text = read_text_file(resolved_path)
    original_document = read_json_file(resolved_path)
    patched_document = apply_patch_to_document(original_document, patch)
    patched_text = _serialize_document(patched_document)
    resolved_path.write_text(patched_text, encoding="utf-8")
    return {
        "specPath": str(resolved_path),
        "originalText": original_text,
        "patchedText": patched_text,
        "originalDocument": original_document,
        "patchedDocument": patched_document,
        "patch": patch,
    }


def apply_patch_to_document(document: Any, patch: list[dict[str, Any]]) -> Any:
    patch_object = jsonpatch.JsonPatch(patch)
    return patch_object.apply(document, in_place=False)


def restore_spec_file(spec_path: str | Path, text: str) -> None:
    Path(spec_path).expanduser().resolve().write_text(text, encoding="utf-8")


def _serialize_document(document: Any) -> str:
    return json.dumps(document, ensure_ascii=False, indent=2) + "\n"
