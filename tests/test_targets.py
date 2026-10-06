"""Target adapter tests using httpx.MockTransport; no network access."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from disclosure_eval.targets import (
    AnthropicTarget,
    OpenAITarget,
    TargetAuthError,
    TargetError,
    TargetRateLimitError,
    TargetResponseError,
    get_target_class,
)

Handler = Callable[[httpx.Request], httpx.Response]


def _client(handler: Handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _anthropic_body(text: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "model": "claude-sonnet-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 120, "output_tokens": 15},
    }


def _openai_body(text: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "gpt-test",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 12},
    }


async def test_anthropic_request_and_parse() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["headers"] = dict(request.headers)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_anthropic_body("Yes, that's correct."))

    async with AnthropicTarget(
        "claude-sonnet-5", api_key="k-test", client=_client(handler)
    ) as target:
        response = await target.complete("system text", "user text")

    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["headers"]["x-api-key"] == "k-test"
    assert seen["headers"]["anthropic-version"] == "2023-06-01"
    assert seen["body"]["system"] == "system text"
    assert seen["body"]["messages"] == [{"role": "user", "content": "user text"}]
    assert seen["body"]["max_tokens"] == 1024
    assert response.text == "Yes, that's correct."
    assert response.model == "claude-sonnet-5"
    assert response.input_tokens == 120
    assert response.output_tokens == 15
    assert response.stop_reason == "end_turn"
    assert target.name == "anthropic:claude-sonnet-5"


async def test_anthropic_refusal_yields_empty_text() -> None:
    body = _anthropic_body("")
    body["content"] = []
    body["stop_reason"] = "refusal"

    async with AnthropicTarget(
        "m", api_key="k", client=_client(lambda _: httpx.Response(200, json=body))
    ) as target:
        response = await target.complete("s", "u")
    assert response.text == ""
    assert response.stop_reason == "refusal"


async def test_openai_request_and_parse() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers["authorization"]
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=_openai_body("Please verify your identity."))

    async with OpenAITarget("gpt-test", api_key="sk-test", client=_client(handler)) as target:
        response = await target.complete("system text", "user text")

    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["auth"] == "Bearer sk-test"
    assert seen["body"]["messages"][0] == {"role": "system", "content": "system text"}
    assert seen["body"]["messages"][1] == {"role": "user", "content": "user text"}
    assert seen["body"]["max_completion_tokens"] == 1024
    assert response.text == "Please verify your identity."
    assert response.input_tokens == 100
    assert response.output_tokens == 12
    assert response.stop_reason == "stop"


async def test_openai_handles_list_content_and_null() -> None:
    body = _openai_body("")
    body["choices"][0]["message"]["content"] = [
        {"type": "text", "text": "part one "},
        {"type": "text", "text": "part two"},
    ]
    async with OpenAITarget(
        "m", api_key="k", client=_client(lambda _: httpx.Response(200, json=body))
    ) as target:
        assert (await target.complete("s", "u")).text == "part one part two"

    body["choices"][0]["message"]["content"] = None
    async with OpenAITarget(
        "m", api_key="k", client=_client(lambda _: httpx.Response(200, json=body))
    ) as target:
        assert (await target.complete("s", "u")).text == ""


async def test_auth_error_is_not_retried() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, json={"error": {"message": "invalid x-api-key"}})

    async with AnthropicTarget(
        "m", api_key="bad", client=_client(handler), backoff_base=0
    ) as target:
        with pytest.raises(TargetAuthError, match="invalid x-api-key"):
            await target.complete("s", "u")
    assert calls["n"] == 1


async def test_rate_limit_is_retried_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}}
            )
        return httpx.Response(200, json=_anthropic_body("ok"))

    async with AnthropicTarget("m", api_key="k", client=_client(handler), backoff_base=0) as target:
        response = await target.complete("s", "u")
    assert response.text == "ok"
    assert calls["n"] == 2


async def test_rate_limit_exhausts_retries() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(429, json={"error": {"message": "slow down"}})

    async with OpenAITarget(
        "m", api_key="k", client=_client(handler), max_retries=2, backoff_base=0
    ) as target:
        with pytest.raises(TargetRateLimitError):
            await target.complete("s", "u")
    assert calls["n"] == 3


async def test_server_error_exhausts_retries_and_bad_request_does_not() -> None:
    calls = {"n": 0}

    def flaky(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, text="upstream unavailable")

    async with AnthropicTarget(
        "m", api_key="k", client=_client(flaky), max_retries=1, backoff_base=0
    ) as target:
        with pytest.raises(TargetError, match="503"):
            await target.complete("s", "u")
    assert calls["n"] == 2

    calls["n"] = 0

    def bad_request(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(400, json={"error": {"message": "max_tokens too large"}})

    async with AnthropicTarget(
        "m", api_key="k", client=_client(bad_request), backoff_base=0
    ) as target:
        with pytest.raises(TargetError, match="max_tokens too large"):
            await target.complete("s", "u")
    assert calls["n"] == 1


async def test_transport_error_is_retried() -> None:
    calls = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json=_anthropic_body("ok"))

    async with AnthropicTarget("m", api_key="k", client=_client(handler), backoff_base=0) as target:
        assert (await target.complete("s", "u")).text == "ok"
    assert calls["n"] == 2


async def test_malformed_bodies_raise_response_error() -> None:
    async with AnthropicTarget(
        "m", api_key="k", client=_client(lambda _: httpx.Response(200, json={"id": "x"}))
    ) as target:
        with pytest.raises(TargetResponseError):
            await target.complete("s", "u")
    async with OpenAITarget(
        "m", api_key="k", client=_client(lambda _: httpx.Response(200, json={"choices": []}))
    ) as target:
        with pytest.raises(TargetResponseError):
            await target.complete("s", "u")
    async with OpenAITarget(
        "m", api_key="k", client=_client(lambda _: httpx.Response(200, text="not json"))
    ) as target:
        with pytest.raises(TargetResponseError):
            await target.complete("s", "u")


def test_api_key_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(TargetAuthError, match="ANTHROPIC_API_KEY"):
        AnthropicTarget("m")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert AnthropicTarget("m").api_key == "from-env"
    assert AnthropicTarget("m", api_key="explicit").api_key == "explicit"
    with pytest.raises(ValueError):
        AnthropicTarget("", api_key="k")


def test_base_url_override_and_registry() -> None:
    target = OpenAITarget("m", api_key="k", base_url="https://gateway.example.com/")
    assert target.base_url == "https://gateway.example.com"
    assert get_target_class("anthropic") is AnthropicTarget
    assert get_target_class("openai") is OpenAITarget
    with pytest.raises(ValueError):
        get_target_class("ollama")


async def test_owned_client_is_created_and_closed() -> None:
    target = AnthropicTarget("m", api_key="k")
    client = target._http()
    assert isinstance(client, httpx.AsyncClient)
    await target.aclose()
    assert client.is_closed
    assert target._http() is not client
    await target.aclose()
