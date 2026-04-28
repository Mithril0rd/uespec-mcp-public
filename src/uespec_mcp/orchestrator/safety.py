from __future__ import annotations

import difflib
import json
from typing import Any


def patch_edit_distance(previous_patch: list[dict[str, Any]], current_patch: list[dict[str, Any]]) -> int:
    previous_text = json.dumps(previous_patch, ensure_ascii=False, sort_keys=True)
    current_text = json.dumps(current_patch, ensure_ascii=False, sort_keys=True)
    return _levenshtein_distance(previous_text, current_text)


def patch_removes_more_than_n_lines(original_text: str, patched_text: str, limit: int) -> bool:
    removed_line_count = 0
    diff = difflib.unified_diff(
        _canonicalize_diff_lines(original_text),
        _canonicalize_diff_lines(patched_text),
        fromfile="before",
        tofile="after",
        lineterm="",
    )
    for line in diff:
        if line.startswith("---") or line.startswith("+++"):
            continue
        if line.startswith("-"):
            removed_line_count += 1
    return removed_line_count > limit


def is_regression(previous_failure_count: int, current_failure_count: int) -> bool:
    return current_failure_count > previous_failure_count


def _canonicalize_diff_lines(text: str) -> list[str]:
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text.splitlines()
    canonical = json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)
    return canonical.splitlines()


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
