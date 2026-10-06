"""Probe suites that generate prompts from records and score replies."""

from .base import Combination, Probe, ProbePrompt, PromptTemplate, sample_combinations
from .confirmation import CONFIRMATION_TEMPLATES, ConfirmationProbe
from .oracle import ORACLE_TEMPLATES, OracleProbe

PROBES: dict[str, type[Probe]] = {
    ConfirmationProbe.name: ConfirmationProbe,
    OracleProbe.name: OracleProbe,
}
"""Registry of suite name to probe class."""


def get_probe(name: str) -> Probe:
    """Instantiate a probe by suite name.

    Raises:
        ValueError: If the name is not registered.
    """
    try:
        return PROBES[name]()
    except KeyError as exc:
        supported = ", ".join(sorted(PROBES))
        raise ValueError(f"Unknown probe suite {name!r}. Supported: {supported}") from exc


__all__ = [
    "CONFIRMATION_TEMPLATES",
    "ORACLE_TEMPLATES",
    "PROBES",
    "Combination",
    "ConfirmationProbe",
    "OracleProbe",
    "Probe",
    "ProbePrompt",
    "PromptTemplate",
    "get_probe",
    "sample_combinations",
]
