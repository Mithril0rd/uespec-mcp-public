from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from ..base import LLMClient
from ..types import LLMRequest, LLMResponse


class MockLLMNoMatchError(RuntimeError):
    """Raised when no canned mock response matches the prompt."""


class MockLLMClient(LLMClient):
    def __init__(self, canned_responses: dict[str, LLMResponse] | None = None) -> None:
        self.canned_responses = canned_responses or {}
        self.call_history: list[LLMRequest] = []

    @classmethod
    def from_fixtures(cls, fixture_path: Path | None = None) -> "MockLLMClient":
        resolved_path = fixture_path or resources.files("uespec_mcp.llm.fixtures").joinpath("mock_responses.json")
        payload = json.loads(Path(resolved_path).read_text(encoding="utf-8"))
        canned_responses = {
            keyword: LLMResponse.model_validate(response_payload)
            for keyword, response_payload in payload.items()
        }
        return cls(canned_responses)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_history.append(request)
        prompt = request.user_prompt
        for keyword, response in self.canned_responses.items():
            if keyword == "*" or keyword in prompt:
                return response.model_copy(deep=True)
        raise MockLLMNoMatchError(f"No canned response for prompt: {prompt[:200]}")

    def complete_with_tools(self, request: LLMRequest, tools: list[dict[str, Any]]) -> LLMResponse:
        del tools
        return self.complete(request)
