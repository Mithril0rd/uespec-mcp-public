from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_text_file(path: str | Path) -> str:
    return Path(path).expanduser().resolve().read_text(encoding="utf-8-sig")


def read_json_file(path: str | Path) -> Any:
    return json.loads(read_text_file(path))
