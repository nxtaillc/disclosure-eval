"""disclosure-eval: test AI customer service agents for confirmation-based PII disclosure.

Quick start::

    from disclosure_eval import AnthropicTarget, Evaluator, SyntheticRecords

    records = SyntheticRecords.generate(count=5, seed=42)
    evaluator = Evaluator(
        target=AnthropicTarget(model="claude-sonnet-5"),
        records=records,
        trials_per_suite=10,
    )
    report = evaluator.run()
    report.print_summary()
    report.save_json("results.json")
"""

__version__ = "0.1.0"

from .evaluator import DEFAULT_SYSTEM_PROMPT, EvalReport, Evaluator
from .probes import ConfirmationProbe, OracleProbe, Probe, ProbePrompt
from .records import CustomerRecord, RecordSet, RecordValidationError, SyntheticRecords
from .scoring import Label, RiskLevel, SuiteResult, Trial
from .targets import (
    AnthropicTarget,
    OpenAITarget,
    Target,
    TargetAuthError,
    TargetError,
    TargetResponse,
)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "AnthropicTarget",
    "ConfirmationProbe",
    "CustomerRecord",
    "EvalReport",
    "Evaluator",
    "Label",
    "OpenAITarget",
    "OracleProbe",
    "Probe",
    "ProbePrompt",
    "RecordSet",
    "RecordValidationError",
    "RiskLevel",
    "SuiteResult",
    "SyntheticRecords",
    "Target",
    "TargetAuthError",
    "TargetError",
    "TargetResponse",
    "Trial",
    "__version__",
]
