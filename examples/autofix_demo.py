from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp.bridge import get_bridge
from uespec_mcp.llm.factory import create_llm_client
from uespec_mcp.orchestrator.core import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one manual UESpec autofix demo.")
    parser.add_argument("--spec", required=True, help="Path to the broken UI spec JSON file.")
    parser.add_argument("--test-spec", required=True, help="Path to the test spec JSON file.")
    parser.add_argument("--provider", default=None, help="LLM provider override, for example 'claude', 'openai', or 'mock'.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum autofix attempts.")
    parser.add_argument("--history-dir", default="Saved/UESpec/PatchHistory", help="Patch history output directory.")
    args = parser.parse_args()

    orchestrator = Orchestrator(
        get_bridge(),
        create_llm_client(args.provider),
        max_attempts=args.max_attempts,
        history_dir=Path(args.history_dir),
    )
    result = orchestrator.run_with_autofix(Path(args.spec), Path(args.test_spec))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
