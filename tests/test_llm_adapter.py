from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from uespec_mcp.llm.adapters.claude import ClaudeClient
from uespec_mcp.llm.adapters.mock import MockLLMClient
from uespec_mcp.llm.adapters.openai import OpenAIClient
from uespec_mcp.llm.factory import create_llm_client
from uespec_mcp.llm.prompts import render
from uespec_mcp.llm.types import LLMImageAttachment, LLMRequest, LLMResponse


class FakeAnthropicUsage:
    def __init__(self, input_tokens: int = 11, output_tokens: int = 7) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeAnthropicTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class FakeAnthropicToolBlock:
    def __init__(self, payload: dict[str, object]) -> None:
        self.type = "tool_use"
        self.input = payload


class FakeAnthropicResponse:
    def __init__(self, model: str, content: list[object]) -> None:
        self.model = model
        self.content = content
        self.usage = FakeAnthropicUsage()


class FakeAnthropicMessages:
    def __init__(self, response: FakeAnthropicResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeAnthropicResponse:
        self.calls.append(dict(kwargs))
        return self.response


class FlakyAnthropicMessages:
    def __init__(self, responses: list[FakeAnthropicResponse], errors: list[Exception]) -> None:
        self.responses = list(responses)
        self.errors = list(errors)
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeAnthropicResponse:
        self.calls.append(dict(kwargs))
        if self.errors:
            raise self.errors.pop(0)
        if not self.responses:
            raise RuntimeError("No more fake anthropic responses available.")
        return self.responses.pop(0)


class FakeAnthropicClient:
    def __init__(self, response: FakeAnthropicResponse) -> None:
        self.messages = FakeAnthropicMessages(response)


class FlakyAnthropicClient:
    def __init__(self, responses: list[FakeAnthropicResponse], errors: list[Exception]) -> None:
        self.messages = FlakyAnthropicMessages(responses, errors)


class FakeOpenAIUsage:
    def __init__(self, input_tokens: int = 13, output_tokens: int = 5) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class FakeOpenAIResponse:
    def __init__(self, model: str, output_text: str, output_parsed: object = None) -> None:
        self.model = model
        self.output_text = output_text
        self.output_parsed = output_parsed
        self.output: list[object] = []
        self.usage = FakeOpenAIUsage()


class FakeOpenAIResponses:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeOpenAIResponse:
        self.calls.append(dict(kwargs))
        return self.response


class FakeOpenAIClient:
    def __init__(self, response: FakeOpenAIResponse) -> None:
        self.responses = FakeOpenAIResponses(response)


def test_mock_client_matches_canned_response() -> None:
    client = MockLLMClient(
        {
            "compile error": LLMResponse(
                content='{"patch": []}',
                parsed_output={"patch": []},
                raw_response={"provider": "mock"},
                input_tokens=1,
                output_tokens=1,
                model="mock-llm",
            )
        }
    )
    request = LLMRequest(system_prompt="sys", user_prompt="please fix compile error", expected_output_schema={})

    response = client.complete(request)

    assert response.parsed_output == {"patch": []}


def test_mock_client_records_call_history() -> None:
    client = MockLLMClient(
        {
            "*": LLMResponse(
                content="{}",
                parsed_output={},
                raw_response={},
                input_tokens=0,
                output_tokens=0,
                model="mock-llm",
            )
        }
    )
    request = LLMRequest(system_prompt="sys", user_prompt="hello", expected_output_schema={})

    client.complete(request)

    assert client.call_history == [request]


def test_claude_adapter_parses_response() -> None:
    fake_response = FakeAnthropicResponse(
        "claude-opus-4-1",
        [FakeAnthropicTextBlock('{"patch": []}'), FakeAnthropicToolBlock({"tool": "noop"})],
    )
    fake_client = FakeAnthropicClient(fake_response)
    client = ClaudeClient(client=fake_client, model="claude-opus-4-1")
    request = LLMRequest(system_prompt="sys", user_prompt="fix", expected_output_schema={})

    response = client.complete_with_tools(request, [{"name": "apply_patch"}])

    assert fake_client.messages.calls[0]["tools"] == [{"name": "apply_patch"}]
    assert response.parsed_output == {"tool": "noop"}
    assert response.input_tokens == 11
    assert response.output_tokens == 7


def test_claude_adapter_uses_tools_for_structured_output() -> None:
    fake_response = FakeAnthropicResponse(
        "claude-opus-4-1",
        [
            FakeAnthropicToolBlock(
                {
                    "result": [
                        {"op": "replace", "path": "/root/content/convert", "value": "Percent"},
                    ]
                }
            )
        ],
    )
    fake_client = FakeAnthropicClient(fake_response)
    client = ClaudeClient(client=fake_client, model="claude-opus-4-1")
    request = LLMRequest(
        system_prompt="sys",
        user_prompt="fix",
        expected_output_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string"},
                    "path": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["op", "path"],
            },
        },
    )

    response = client.complete(request)

    call = fake_client.messages.calls[0]
    assert call["tool_choice"] == {"type": "tool", "name": "uespec_structured_output"}
    assert call["tools"][0]["input_schema"]["type"] == "object"
    assert call["tools"][0]["input_schema"]["required"] == ["result"]
    assert response.parsed_output == [{"op": "replace", "path": "/root/content/convert", "value": "Percent"}]


def test_claude_adapter_sends_image_content() -> None:
    fake_response = FakeAnthropicResponse("claude-opus-4-1", [FakeAnthropicTextBlock("{}")])
    fake_client = FakeAnthropicClient(fake_response)
    client = ClaudeClient(client=fake_client, model="claude-opus-4-1")
    request = LLMRequest(
        system_prompt="sys",
        user_prompt="generate",
        images=[LLMImageAttachment(media_type="image/png", data_base64="aGVsbG8=")],
        expected_output_schema={},
    )

    client.complete(request)

    content = fake_client.messages.calls[0]["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "generate"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"] == "aGVsbG8="


def test_openai_adapter_parses_response() -> None:
    fake_response = FakeOpenAIResponse("gpt-5", '{"patch":[{"op":"replace","path":"/x","value":1}]}')
    fake_client = FakeOpenAIClient(fake_response)
    client = OpenAIClient(client=fake_client, model="gpt-5")
    request = LLMRequest(
        system_prompt="sys",
        user_prompt="fix",
        expected_output_schema={"type": "object", "properties": {"patch": {"type": "array"}}},
    )

    response = client.complete(request)

    assert fake_client.responses.calls[0]["text"]["format"]["type"] == "json_schema"
    assert response.parsed_output == {"patch": [{"op": "replace", "path": "/x", "value": 1}]}
    assert response.model == "gpt-5"


def test_openai_adapter_sends_image_content() -> None:
    fake_response = FakeOpenAIResponse("gpt-5", "{}")
    fake_client = FakeOpenAIClient(fake_response)
    client = OpenAIClient(client=fake_client, model="gpt-5")
    request = LLMRequest(
        system_prompt="sys",
        user_prompt="generate",
        images=[LLMImageAttachment(media_type="image/png", data_base64="aGVsbG8=")],
        expected_output_schema={},
    )

    client.complete(request)

    content = fake_client.responses.calls[0]["input"][1]["content"]
    assert content[0] == {"type": "input_text", "text": "generate"}
    assert content[1] == {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="}


def test_factory_respects_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UESPEC_LLM_PROVIDER", "mock")
    monkeypatch.delenv("UESPEC_LLM_MOCK_FIXTURES", raising=False)

    client = create_llm_client()

    assert isinstance(client, MockLLMClient)


def test_mock_factory_uses_fixture_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture_path = tmp_path / "mock_responses.json"
    fixture_path.write_text(
        json.dumps(
            {
                "special": {
                    "content": "{\"status\":\"ok\"}",
                    "parsed_output": {"status": "ok"},
                    "raw_response": {"provider": "mock"},
                    "input_tokens": 2,
                    "output_tokens": 3,
                    "model": "mock-override"
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("UESPEC_LLM_PROVIDER", "mock")
    monkeypatch.setenv("UESPEC_LLM_MOCK_FIXTURES", str(fixture_path))

    client = create_llm_client()
    response = client.complete(LLMRequest(system_prompt="sys", user_prompt="special case", expected_output_schema={}))

    assert response.model == "mock-override"
    assert response.parsed_output == {"status": "ok"}


def test_template_render_smoke() -> None:
    rendered = render("response_schema.j2", failure_category="COMPILE_ERROR", task_summary="Fix one field")

    assert "COMPILE_ERROR" in rendered
    assert "Fix one field" in rendered


def test_claude_client_uses_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeAnthropicCtor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://relay.example.com")
    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeAnthropicCtor))

    ClaudeClient()

    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://relay.example.com"


def test_claude_adapter_retries_invalid_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("UESPEC_CLAUDE_RETRIES", "1")
    monkeypatch.setenv("UESPEC_CLAUDE_RETRY_DELAY_MS", "0")

    fake_response = FakeAnthropicResponse(
        "claude-haiku-4-5-20251001",
        [FakeAnthropicToolBlock({"result": [{"op": "replace", "path": "/root/id", "value": "rootPanel"}]})],
    )
    fake_client = FlakyAnthropicClient(
        [fake_response],
        [RuntimeError("INVALID_MODEL_ID from relay")],
    )
    client = ClaudeClient(client=fake_client, model="claude-haiku-4-5-20251001")
    request = LLMRequest(
        system_prompt="sys",
        user_prompt="fix",
        expected_output_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "op": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["op", "path"],
            },
        },
    )

    response = client.complete(request)

    assert len(fake_client.messages.calls) == 2
    assert response.parsed_output == [{"op": "replace", "path": "/root/id", "value": "rootPanel"}]
