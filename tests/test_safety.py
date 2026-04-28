from __future__ import annotations

import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp.orchestrator.safety import patch_removes_more_than_n_lines


def test_patch_removal_limit_ignores_formatting_only_rewrites() -> None:
    original_text = json.dumps(
        {
            "apiVersion": "0.2",
            "kind": "HUD",
            "root": {
                "children": [
                    {
                        "id": "slotQ",
                        "actions": [{"on": "Clicked", "do": "vm.UseSkil(0)"}],
                    }
                ]
            },
        },
        ensure_ascii=False,
        indent="\t",
    )
    patched_text = json.dumps(
        {
            "apiVersion": "0.2",
            "kind": "HUD",
            "root": {
                "children": [
                    {
                        "id": "slotQ",
                        "actions": [{"on": "Clicked", "do": "vm.UseSkill(0)"}],
                    }
                ]
            },
        },
        ensure_ascii=False,
        indent=2,
    )

    assert patch_removes_more_than_n_lines(original_text, patched_text, limit=5) is False


def test_patch_removal_limit_detects_large_semantic_delete() -> None:
    original_text = json.dumps(
        {"items": [{"id": f"slot-{index}"} for index in range(30)]},
        ensure_ascii=False,
        indent=2,
    )
    patched_text = json.dumps(
        {"items": [{"id": "slot-0"}]},
        ensure_ascii=False,
        indent=2,
    )

    assert patch_removes_more_than_n_lines(original_text, patched_text, limit=5) is True
