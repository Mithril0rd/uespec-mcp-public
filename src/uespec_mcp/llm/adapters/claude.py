from __future__ import annotations

import os
import time
from typing import Any

from ..base import LLMClient, LLMConfigurationError
from ..types import LLMRequest, LLMResponse
from ._common import maybe_parse_json, model_dump_compat, normalize_model_output


class ClaudeClient(LLMClient):
    _STRUCTURED_OUTPUT_TOOL_NAME = "uespec_structured_output"

    def __init__(self, client: Any | None = None, model: str | None = None) -> None:
        self.model = model or os.getenv("UESPEC_CLAUDE_MODEL", "claude-opus-4-1")
        self._client = client or self._create_default_client()
        self.max_retries = max(int(os.getenv("UESPEC_CLAUDE_RETRIES", "1") or 1), 0)
        self.retry_delay_seconds = max(float(os.getenv("UESPEC_CLAUDE_RETRY_DELAY_MS", "250") or 250.0) / 1000.0, 0.0)

    def _create_default_client(self) -> Any:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMConfigurationError("ANTHROPIC_API_KEY is required for Claude provider.")
        base_url = os.getenv("ANTHROPIC_BASE_URL") or os.getenv("UESPEC_CLAUDE_BASE_URL")
        try:
            import anthropic
        except ImportError as exc:  # pragma: no cover - exercised in integration environments
            raise LLMConfigurationError("anthropic is not installed.") from exc
        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        return anthropic.Anthropic(**client_kwargs)

    def complete(self, request: LLMRequest) -> LLMResponse:
        messages = [{"role": "user", "content": self._build_user_content(request)}]
        if request.expected_output_schema:
            tool = self._build_structured_output_tool(request.expected_output_schema)
            response = self._invoke_messages_create(
                model=self.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                system=request.system_prompt,
                messages=messages,
                tools=[tool],
                tool_choice={"type": "tool", "name": tool["name"]},
            )
            return self._to_response(response, structured_output_schema=request.expected_output_schema)
        response = self._invoke_messages_create(
            model=self.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system_prompt,
            messages=messages,
        )
        return self._to_response(response)

    def complete_with_tools(self, request: LLMRequest, tools: list[dict[str, Any]]) -> LLMResponse:
        response = self._invoke_messages_create(
            model=self.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system_prompt,
            messages=[{"role": "user", "content": self._build_user_content(request)}],
            tools=tools,
        )
        return self._to_response(response)

    def _build_user_content(self, request: LLMRequest) -> str | list[dict[str, Any]]:
        if not request.images:
            return request.user_prompt

        content: list[dict[str, Any]] = [{"type": "text", "text": request.user_prompt}]
        for image in request.images:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image.media_type,
                        "data": image.data_base64,
                    },
                }
            )
        return content

    def _invoke_messages_create(self, **kwargs: Any) -> Any:
        attempts = self.max_retries + 1
        last_error: Exception | None = None
        for attempt_index in range(attempts):
            try:
                return self._client.messages.create(**kwargs)
            except Exception as exc:
                last_error = exc
                if attempt_index >= self.max_retries or not self._should_retry_exception(exc):
                    raise
                if self.retry_delay_seconds > 0:
                    time.sleep(self.retry_delay_seconds)
        if last_error is not None:
            raise last_error
        raise RuntimeError("Claude request failed without raising a concrete exception.")

    def _should_retry_exception(self, exc: Exception) -> bool:
        message = str(exc).upper()
        return "INVALID_MODEL_ID" in message or "RATE LIMIT" in message or "TIMEOUT" in message

    def _to_response(
        self,
        response: Any,
        *,
        structured_output_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        text_fragments: list[str] = []
        parsed_output: dict[str, Any] | list[Any] | None = None

        for block in getattr(response, "content", []):
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_fragments.append(getattr(block, "text", ""))
            elif block_type == "tool_use":
                parsed_output = normalize_model_output(getattr(block, "input", None))

        if structured_output_schema and isinstance(parsed_output, dict) and "result" in parsed_output:
            parsed_output = normalize_model_output(parsed_output.get("result"))

        content = "".join(text_fragments).strip()
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

    def _build_structured_output_tool(self, schema: dict[str, Any]) -> dict[str, Any]:
        normalized_schema = schema if schema.get("type") == "object" else {
            "type": "object",
            "properties": {
                "result": schema,
            },
            "required": ["result"],
            "additionalProperties": False,
        }
        return {
            "name": self._STRUCTURED_OUTPUT_TOOL_NAME,
            "description": "Return structured output that matches the requested JSON schema exactly.",
            "input_schema": normalized_schema,
        }
