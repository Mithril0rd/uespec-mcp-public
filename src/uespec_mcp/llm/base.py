from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import LLMRequest, LLMResponse


class LLMConfigurationError(RuntimeError):
    """Raised when the selected LLM provider is not configured correctly."""


class LLMClient(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def complete_with_tools(self, request: LLMRequest, tools: list[dict[str, Any]]) -> LLMResponse:
        raise NotImplementedError
