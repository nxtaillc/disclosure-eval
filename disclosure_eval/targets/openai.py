"""OpenAI Chat Completions API target."""

from __future__ import annotations

from typing import Any, ClassVar

from .base import Target, TargetResponse, TargetResponseError


class OpenAITarget(Target):
    """Calls ``POST /v1/chat/completions`` on the OpenAI API.

    The system prompt is sent as a ``system`` message followed by the probe
    text as a ``user`` message. Only the first choice is read.
    """

    provider: ClassVar[str] = "openai"
    api_key_env: ClassVar[str | None] = "OPENAI_API_KEY"
    default_base_url: ClassVar[str] = "https://api.openai.com"

    async def complete(self, system: str, user: str) -> TargetResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        headers = {
            "authorization": f"Bearer {self.api_key or ''}",
            "content-type": "application/json",
        }
        body, elapsed_ms = await self._post("/v1/chat/completions", payload, headers)
        return _parse_completion(body, elapsed_ms, fallback_model=self.model)


def _parse_completion(
    body: dict[str, Any], elapsed_ms: float, fallback_model: str
) -> TargetResponse:
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise TargetResponseError("openai: response has no 'choices'")
    first = choices[0]
    if not isinstance(first, dict):
        raise TargetResponseError("openai: malformed choice entry")
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        text = "".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    else:
        text = str(content) if content is not None else ""
    usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
    return TargetResponse(
        text=text.strip(),
        model=str(body.get("model") or fallback_model),
        latency_ms=round(elapsed_ms, 2),
        input_tokens=_int_or_none(usage.get("prompt_tokens")),
        output_tokens=_int_or_none(usage.get("completion_tokens")),
        stop_reason=first.get("finish_reason"),
    )


def _int_or_none(value: Any) -> int | None:
    return int(value) if isinstance(value, int) else None
