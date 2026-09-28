"""Adapter tests using a stub OpenAI-compatible client (no network access)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orchestrator_worker.config import Settings
from orchestrator_worker.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    available_providers,
    build_llm_client,
)
from orchestrator_worker.llm.base import LLMClient, LLMError
from orchestrator_worker.llm.deepseek import DeepSeekClient


class StubCompletions:
    """Mimics client.chat.completions and records the outgoing payload."""

    def __init__(self, response: object) -> None:
        self._response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class StubOpenAI:
    def __init__(self, response: object) -> None:
        self.completions = StubCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def make_response(content: str = "hello", model: str = "deepseek-chat"):
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7, total_tokens=18),
    )
def test_deepseek_translates_request_and_response():
    stub = StubOpenAI(make_response("the answer is 42"))
    client = DeepSeekClient(client=stub)

    response = client.complete(
        LLMRequest(
            messages=[Message.system("be brief"), Message.user("meaning of life?")],
            temperature=0.2,
            max_tokens=128,
        )
    )

    sent = stub.completions.calls[0]
    assert sent["model"] == "deepseek-chat"
    assert sent["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "meaning of life?"},
    ]
    assert sent["temperature"] == 0.2
    assert sent["max_tokens"] == 128
    assert "response_format" not in sent

    assert isinstance(response, LLMResponse)
    assert response.content == "the answer is 42"
    assert response.finish_reason == "stop"
    assert response.usage.total_tokens == 18


def test_json_mode_sets_response_format():
    stub = StubOpenAI(make_response("{}"))
    DeepSeekClient(client=stub).complete(
        LLMRequest(messages=[Message.user("return json")], json_mode=True)
    )
    assert stub.completions.calls[0]["response_format"] == {"type": "json_object"}


def test_optional_parameters_are_omitted_when_unset():
    stub = StubOpenAI(make_response())
    DeepSeekClient(client=stub).complete(LLMRequest(messages=[Message.user("hi")]))
    sent = stub.completions.calls[0]
    assert "temperature" not in sent
    assert "max_tokens" not in sent
    assert "stop" not in sent


def test_missing_api_key_is_rejected_before_any_call():
    with pytest.raises(LLMError, match="DEEPSEEK_API_KEY"):
        DeepSeekClient(api_key=None)


def test_sdk_failure_is_wrapped_in_llm_error():
    class ExplodingCompletions:
        def create(self, **kwargs):
            raise RuntimeError("connection reset")

    broken = SimpleNamespace(chat=SimpleNamespace(completions=ExplodingCompletions()))
    client = DeepSeekClient(client=broken)
    with pytest.raises(LLMError, match="connection reset"):
        client.complete(LLMRequest(messages=[Message.user("hi")]))


def test_empty_choices_raise_llm_error():
    empty = SimpleNamespace(model="m", choices=[], usage=None)
    client = DeepSeekClient(client=StubOpenAI(empty))
    with pytest.raises(LLMError, match="no choices"):
        client.complete(LLMRequest(messages=[Message.user("hi")]))


def test_deepseek_is_registered_and_buildable():
    assert "deepseek" in available_providers()
    client = build_llm_client(Settings(api_key="sk-test"))
    assert isinstance(client, DeepSeekClient)
    assert isinstance(client, LLMClient)


def test_unknown_provider_lists_alternatives():
    with pytest.raises(LLMError, match="deepseek"):
        build_llm_client(Settings(llm_provider="claude"))


def test_async_wrapper_delegates_to_complete():
    import asyncio

    stub = StubOpenAI(make_response("async ok"))
    client = DeepSeekClient(client=stub)
    response = asyncio.run(client.acomplete(LLMRequest(messages=[Message.user("hi")])))
    assert response.content == "async ok"