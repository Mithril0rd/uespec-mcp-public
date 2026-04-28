from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class PatchHistoryStore:
    def __init__(self, root_dir: str | Path, spec_path: str | Path) -> None:
        resolved_root = Path(root_dir).expanduser().resolve()
        spec_name = Path(spec_path).stem
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = resolved_root / spec_name / timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def write_attempt_json(self, attempt_index: int, file_name: str, payload: Any) -> Path:
        attempt_dir = self._attempt_dir(attempt_index)
        attempt_dir.mkdir(parents=True, exist_ok=True)
        path = attempt_dir / file_name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def write_final_result(self, payload: Any) -> Path:
        path = self.run_dir / "final-result.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def _attempt_dir(self, attempt_index: int) -> Path:
        return self.run_dir / f"attempt-{attempt_index:02d}"
