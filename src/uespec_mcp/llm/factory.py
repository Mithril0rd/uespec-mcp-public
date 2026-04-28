from __future__ import annotations

import os
from pathlib import Path

from .adapters.claude import ClaudeClient
from .adapters.mock import MockLLMClient
from .adapters.openai import OpenAIClient
from .base import LLMClient, LLMConfigurationError


def create_llm_client(provider: str | None = None) -> LLMClient:
    normalized_provider = (provider or os.getenv("UESPEC_LLM_PROVIDER", "claude")).strip().lower()
    if normalized_provider == "claude":
        return ClaudeClient()
    if normalized_provider == "openai":
        return OpenAIClient()
    if normalized_provider == "mock":
        fixture_path = os.getenv("UESPEC_LLM_MOCK_FIXTURES")
        return MockLLMClient.from_fixtures(Path(fixture_path).expanduser() if fixture_path else None)
    raise LLMConfigurationError(f"Unknown provider: {normalized_provider}")
