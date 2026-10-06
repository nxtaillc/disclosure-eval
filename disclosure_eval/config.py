"""YAML configuration for the command-line interface.

The config file describes the target, where the records live, how many
trials to run, and the agent's system prompt. :func:`load_config` parses
and validates it; :func:`build_target` instantiates the adapter.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .evaluator import DEFAULT_SYSTEM_PROMPT
from .probes import PROBES
from .targets import Target, TargetAuthError, get_target_class


class ConfigError(ValueError):
    """Raised when the configuration file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class TargetConfig:
    """Target section of the config file."""

    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 3


@dataclass(frozen=True)
class EvalConfig:
    """Fully parsed evaluation configuration.

    Paths are resolved relative to the directory containing the config file.
    """

    target: TargetConfig
    records_path: Path
    output_path: Path
    trials_per_suite: int = 10
    suites: tuple[str, ...] = ("confirmation", "oracle")
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    inject_records: bool = True
    concurrency: int = 4
    seed: int | None = None
    config_dir: Path = field(default_factory=Path.cwd)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{context} must be a mapping")
    return value


def _positive_int(value: Any, key: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"'{key}' must be a positive integer, got {value!r}")
    return value


def _optional_int(value: Any, key: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{key}' must be an integer or null, got {value!r}")
    return value


def _parse_target(raw: Any) -> TargetConfig:
    data = _require_mapping(raw, "'target'")
    provider = str(data.get("provider", "")).strip().lower()
    model = str(data.get("model", "")).strip()
    if not provider:
        raise ConfigError("'target.provider' is required")
    if not model:
        raise ConfigError("'target.model' is required")
    try:
        get_target_class(provider)
    except ValueError as exc:
        raise ConfigError(str(exc)) from exc
    timeout = data.get("timeout", 60.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ConfigError(f"'target.timeout' must be a positive number, got {timeout!r}")
    api_key_env = data.get("api_key_env")
    base_url = data.get("base_url")
    return TargetConfig(
        provider=provider,
        model=model,
        api_key_env=str(api_key_env).strip() if api_key_env else None,
        base_url=str(base_url).strip() if base_url else None,
        max_tokens=_positive_int(data.get("max_tokens"), "target.max_tokens", 1024),
        timeout=float(timeout),
        max_retries=_positive_int(data.get("max_retries"), "target.max_retries", 3),
    )


def parse_config(data: Mapping[str, Any], config_dir: Path) -> EvalConfig:
    """Validate a parsed YAML mapping and return an :class:`EvalConfig`.

    Raises:
        ConfigError: On any invalid or missing value.
    """
    if "target" not in data:
        raise ConfigError("'target' section is required")
    target = _parse_target(data["target"])

    records_value = data.get("records", "records.yaml")
    if not isinstance(records_value, str) or not records_value.strip():
        raise ConfigError("'records' must be a file path")
    output_value = data.get("output", "results.json")
    if not isinstance(output_value, str) or not output_value.strip():
        raise ConfigError("'output' must be a file path")

    raw_suites = data.get("suites", list(PROBES))
    if not isinstance(raw_suites, list) or not raw_suites:
        raise ConfigError("'suites' must be a non-empty list")
    suites: list[str] = []
    for item in raw_suites:
        name = str(item).strip().lower()
        if name not in PROBES:
            raise ConfigError(f"Unknown suite {name!r}. Supported: {', '.join(sorted(PROBES))}")
        if name not in suites:
            suites.append(name)

    system_prompt = data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ConfigError("'system_prompt' must be a non-empty string")
    inject_records = data.get("inject_records", True)
    if not isinstance(inject_records, bool):
        raise ConfigError("'inject_records' must be true or false")

    return EvalConfig(
        target=target,
        records_path=(config_dir / records_value).resolve(),
        output_path=(config_dir / output_value).resolve(),
        trials_per_suite=_positive_int(data.get("trials_per_suite"), "trials_per_suite", 10),
        suites=tuple(suites),
        system_prompt=system_prompt,
        inject_records=inject_records,
        concurrency=_positive_int(data.get("concurrency"), "concurrency", 4),
        seed=_optional_int(data.get("seed"), "seed"),
        config_dir=config_dir,
    )


def load_config(path: str | Path) -> EvalConfig:
    """Read and validate a YAML config file.

    Raises:
        ConfigError: If the file is missing, unparsable, or invalid.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_path}: invalid YAML: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ConfigError(f"{config_path}: expected a mapping at the top level")
    return parse_config(data, config_path.resolve().parent)


def build_target(config: TargetConfig) -> Target:
    """Instantiate the target adapter described by ``config``.

    Raises:
        ConfigError: If the credential named by ``api_key_env`` is unset or
            the provider rejects the configuration.
    """
    target_class = get_target_class(config.provider)
    api_key: str | None = None
    if config.api_key_env:
        api_key = os.environ.get(config.api_key_env, "").strip() or None
        if api_key is None:
            raise ConfigError(
                f"Environment variable {config.api_key_env} is not set "
                f"(required for the {config.provider} target)"
            )
    try:
        return target_class(
            config.model,
            api_key=api_key,
            base_url=config.base_url,
            timeout=config.timeout,
            max_retries=config.max_retries,
            max_tokens=config.max_tokens,
        )
    except TargetAuthError as exc:
        raise ConfigError(str(exc)) from exc
