"""Anthropic Messages API target."""

from __future__ import annotations

from typing import Any, ClassVar

from .base import Target, TargetResponse, TargetResponseError

_API_VERSION = "2023-06-01"


class AnthropicTarget(Target):
    """Calls ``POST /v1/messages`` on the Anthropic API.

    The system prompt is sent in the top-level ``system`` field and the
    probe text as a single user turn. Only ``text`` content blocks are
    read from the reply; a safety refusal (``stop_reason == "refusal"``)
    yields empty text, which the classifier treats as no disclosure.
    """

    provider: ClassVar[str] = "anthropic"
    api_key_env: ClassVar[str | None] = "ANTHROPIC_API_KEY"
    default_base_url: ClassVar[str] = "https://api.anthropic.com"

    async def complete(self, system: str, user: str) -> TargetResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        headers = {
            "x-api-key": self.api_key or "",
            "anthropic-version": _API_VERSION,
            "content-type": "application/json",
        }
        body, elapsed_ms = await self._post("/v1/messages", payload, headers)
        return _parse_message(body, elapsed_ms, fallback_model=self.model)


def _parse_message(body: dict[str, Any], elapsed_ms: float, fallback_model: str) -> TargetResponse:
    content = body.get("content")
    if not isinstance(content, list):
        raise TargetResponseError("anthropic: response has no 'content' list")
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return TargetResponse(
        text="".join(parts).strip(),
        model=str(body.get("model") or fallback_model),
        latency_ms=round(elapsed_ms, 2),
        input_tokens=_int_or_none(usage.get("input_tokens")),
        output_tokens=_int_or_none(usage.get("output_tokens")),
        stop_reason=body.get("stop_reason"),
    )


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None
