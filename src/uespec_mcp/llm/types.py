from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class LLMImageAttachment(BaseModel):
    media_type: str
    data_base64: str


class LLMRequest(BaseModel):
    system_prompt: str
    user_prompt: str
    images: list[LLMImageAttachment] = Field(default_factory=list)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    max_tokens: int = 4000
    temperature: float = 0.1


class LLMResponse(BaseModel):
    content: str
    parsed_output: dict[str, Any] | list[Any] | None = None
    raw_response: dict[str, Any] = Field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str
