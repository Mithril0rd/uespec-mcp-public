from __future__ import annotations

import os
from typing import Any

from ..base import LLMClient, LLMConfigurationError
from ..types import LLMRequest, LLMResponse
from ._common import maybe_parse_json, model_dump_compat, normalize_model_output


class OpenAIClient(LLMClient):
    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self.model = model or os.getenv("UESPEC_OPENAI_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-5"
        self._client = client or self._create_default_client()

    def _create_default_client(self) -> Any:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise LLMConfigurationError("OPENAI_API_KEY is required for OpenAI provider.")
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - exercised in integration environments
            raise LLMConfigurationError("openai is not installed.") from exc
        return OpenAI(api_key=api_key)

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._client.responses.create(**self._build_request_kwargs(request))
        return self._to_response(response)

    def complete_with_tools(self, request: LLMRequest, tools: list[dict[str, Any]]) -> LLMResponse:
        response = self._client.responses.create(**self._build_request_kwargs(request, tools=tools))
        return self._to_response(response)

    def _build_request_kwargs(self, request: LLMRequest, tools: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": request.system_prompt}],
                },
                {
                    "role": "user",
                    "content": self._build_user_content(request),
                },
            ],
            "temperature": request.temperature,
            "max_output_tokens": request.max_tokens,
        }
        if request.expected_output_schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "uespec_llm_output",
                    "schema": request.expected_output_schema,
                    "strict": True,
                }
            }
        if tools:
            kwargs["tools"] = tools
        return kwargs

    def _build_user_content(self, request: LLMRequest) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": request.user_prompt}]
        for image in request.images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image.media_type};base64,{image.data_base64}",
                }
            )
        return content

    def _to_response(self, response: Any) -> LLMResponse:
        content = getattr(response, "output_text", "") or ""
        parsed_output = normalize_model_output(getattr(response, "output_parsed", None))

        if parsed_output is None:
            for item in getattr(response, "output", []):
                item_type = getattr(item, "type", "")
                if item_type in {"function_call", "custom_tool_call"}:
                    parsed_output = normalize_model_output(getattr(item, "arguments", None) or getattr(item, "input", None))
                    break

        if parsed_output is None:
            parsed_output = maybe_parse_json(content)

        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        return LLMResponse(
            content=content,
            parsed_output=parsed_output,
            raw_response=model_dump_compat(response),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=getattr(response, "model", self.model),
        )
