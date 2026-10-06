"""Deterministic scoring: per-trial labels, oracle score, aggregation."""

from .aggregator import (
    FieldDisclosure,
    RiskLevel,
    SuiteResult,
    Trial,
    TrialError,
    assess_risk,
    build_suite_result,
    write_summary,
)
from .classifier import (
    Classification,
    Label,
    classify,
    detect_affirmation,
    detect_denial,
    detect_verification_request,
    find_leaks,
)
from .oracle import OracleObservation, OracleResult, OracleStrength, oracle_score
from .wilson import WilsonInterval, wilson_interval

__all__ = [
    "Classification",
    "FieldDisclosure",
    "Label",
    "OracleObservation",
    "OracleResult",
    "OracleStrength",
    "RiskLevel",
    "SuiteResult",
    "Trial",
    "TrialError",
    "WilsonInterval",
    "assess_risk",
    "build_suite_result",
    "classify",
    "detect_affirmation",
    "detect_denial",
    "detect_verification_request",
    "find_leaks",
    "oracle_score",
    "wilson_interval",
    "write_summary",
]
