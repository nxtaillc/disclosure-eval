"""Model target adapters."""

from .anthropic import AnthropicTarget
from .base import (
    Target,
    TargetAuthError,
    TargetError,
    TargetRateLimitError,
    TargetResponse,
    TargetResponseError,
)
from .openai import OpenAITarget

TARGET_PROVIDERS: dict[str, type[Target]] = {
    AnthropicTarget.provider: AnthropicTarget,
    OpenAITarget.provider: OpenAITarget,
}
"""Registry of provider name to target class, used by the config loader."""


def get_target_class(provider: str) -> type[Target]:
    """Look up a target class by provider name.

    Raises:
        ValueError: If the provider is not registered.
    """
    try:
        return TARGET_PROVIDERS[provider]
    except KeyError as exc:
        supported = ", ".join(sorted(TARGET_PROVIDERS))
        raise ValueError(f"Unknown target provider {provider!r}. Supported: {supported}") from exc


__all__ = [
    "TARGET_PROVIDERS",
    "AnthropicTarget",
    "OpenAITarget",
    "Target",
    "TargetAuthError",
    "TargetError",
    "TargetRateLimitError",
    "TargetResponse",
    "TargetResponseError",
    "get_target_class",
]
