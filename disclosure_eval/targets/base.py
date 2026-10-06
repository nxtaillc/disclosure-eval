"""Base class and shared HTTP plumbing for model targets.

A target is anything that accepts a system prompt plus a single user message
and returns text. Provider adapters subclass :class:`Target`, set the class
attributes that describe their API, and implement :meth:`Target.complete`.

The base class owns the ``httpx.AsyncClient`` lifecycle, credential
resolution, and a retry loop for transient failures (429, 5xx, transport
errors). Non-transient failures are surfaced as typed exceptions so the
evaluator can decide whether to abort the run or record a per-trial error.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

_RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 429, 500, 502, 503, 504})


class TargetError(Exception):
    """Base class for all target failures."""


class TargetAuthError(TargetError):
    """Missing or rejected credentials."""


class TargetRateLimitError(TargetError):
    """The provider kept rate limiting after all retries were exhausted."""


class TargetResponseError(TargetError):
    """The provider returned a body the adapter could not interpret."""


@dataclass(frozen=True)
class TargetResponse:
    """A single completion returned by a target.

    Attributes:
        text: Concatenated text output. Empty if the model produced none.
        model: Model identifier reported by the provider.
        latency_ms: Wall-clock time for the request, including retries.
        input_tokens: Prompt tokens billed, when reported.
        output_tokens: Completion tokens billed, when reported.
        stop_reason: Provider-specific stop reason, when reported.
    """

    text: str
    model: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "text": self.text,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetResponse:
        """Rebuild a response from :meth:`to_dict` output."""
        return cls(
            text=str(data["text"]),
            model=str(data["model"]),
            latency_ms=float(data["latency_ms"]),
            input_tokens=data.get("input_tokens"),
            output_tokens=data.get("output_tokens"),
            stop_reason=data.get("stop_reason"),
        )


class Target(ABC):
    """Abstract async model target.

    Subclasses must set :attr:`provider` and :attr:`default_base_url`, and
    should set :attr:`api_key_env` when the provider needs a credential.

    Args:
        model: Provider model identifier.
        api_key: Explicit credential. If omitted, the environment variable
            named by :attr:`api_key_env` is consulted.
        base_url: Override for the provider's API root.
        timeout: Per-request timeout in seconds.
        max_retries: Number of retries for transient failures.
        max_tokens: Completion length cap sent to the provider.
        backoff_base: Seconds for the first retry delay; doubles each retry.
        client: Optional pre-built ``httpx.AsyncClient`` (used in tests).

    Raises:
        TargetAuthError: If a credential is required and none is available.
    """

    provider: ClassVar[str] = "abstract"
    api_key_env: ClassVar[str | None] = None
    default_base_url: ClassVar[str] = ""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
        max_tokens: int = 1024,
        backoff_base: float = 0.5,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not model:
            raise ValueError("model must be a non-empty string")
        self.model = model
        self.base_url = (base_url or self.default_base_url).rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)
        self.max_tokens = max_tokens
        self.backoff_base = max(0.0, backoff_base)
        self.api_key = self._resolve_api_key(api_key)
        self._client = client
        self._owns_client = client is None

    @property
    def name(self) -> str:
        """Provider-qualified target name, for example ``anthropic:claude-sonnet-5``."""
        return f"{self.provider}:{self.model}"

    def _resolve_api_key(self, explicit: str | None) -> str | None:
        if explicit:
            return explicit
        if self.api_key_env is None:
            return None
        from_env = os.environ.get(self.api_key_env, "").strip()
        if not from_env:
            raise TargetAuthError(
                f"{self.provider} target needs an API key: pass api_key= or set "
                f"the {self.api_key_env} environment variable"
            )
        return from_env

    @abstractmethod
    async def complete(self, system: str, user: str) -> TargetResponse:
        """Send one system prompt and one user message; return the reply.

        Raises:
            TargetError: On any provider failure.
        """

    async def aclose(self) -> None:
        """Release the underlying HTTP client if this target created it."""
        if self._client is not None and self._owns_client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> Target:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
            self._owns_client = True
        return self._client

    async def _post(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> tuple[dict[str, Any], float]:
        """POST JSON with retries; return the decoded body and elapsed ms.

        Raises:
            TargetAuthError: On 401/403.
            TargetRateLimitError: If 429 persists past ``max_retries``.
            TargetResponseError: If the body is not a JSON object.
            TargetError: On other non-retryable HTTP or transport errors.
        """
        url = f"{self.base_url}{path}"
        started = time.perf_counter()
        last_error: TargetError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._http().post(
                    url, json=dict(payload), headers=dict(headers), timeout=self.timeout
                )
            except httpx.TransportError as exc:
                last_error = TargetError(f"{self.provider}: transport error: {exc}")
                await self._sleep_before_retry(attempt, None)
                continue

            if response.status_code in (401, 403):
                raise TargetAuthError(
                    f"{self.provider}: authentication failed ({response.status_code}): "
                    f"{_error_detail(response)}"
                )
            if response.status_code in _RETRYABLE_STATUS:
                detail = _error_detail(response)
                if response.status_code == 429:
                    last_error = TargetRateLimitError(f"{self.provider}: rate limited: {detail}")
                else:
                    last_error = TargetError(
                        f"{self.provider}: server error {response.status_code}: {detail}"
                    )
                await self._sleep_before_retry(attempt, response.headers.get("retry-after"))
                continue
            if response.status_code >= 400:
                raise TargetError(
                    f"{self.provider}: request failed ({response.status_code}): "
                    f"{_error_detail(response)}"
                )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            try:
                body = response.json()
            except ValueError as exc:
                raise TargetResponseError(f"{self.provider}: response was not JSON") from exc
            if not isinstance(body, dict):
                raise TargetResponseError(f"{self.provider}: expected a JSON object body")
            return body, elapsed_ms

        assert last_error is not None
        raise last_error

    async def _sleep_before_retry(self, attempt: int, retry_after: str | None) -> None:
        if attempt >= self.max_retries:
            return
        delay = self.backoff_base * (2**attempt)
        if retry_after:
            with contextlib.suppress(ValueError):
                delay = max(delay, float(retry_after))
        if delay > 0:
            await asyncio.sleep(delay)


def _error_detail(response: httpx.Response) -> str:
    """Extract a short error message from a provider error body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200].strip() or response.reason_phrase
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if isinstance(error, str):
            return error
        if body.get("message"):
            return str(body["message"])
    return response.text[:200].strip() or response.reason_phrase
