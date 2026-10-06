"""JSON serialization of evaluation reports.

The JSON document contains everything needed to re-derive the statistics:
every prompt, every reply, and every classification. Aggregates are also
written for convenience but are recomputed on load.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..records.schemas import field_specs_for_domain
from ..scoring.aggregator import SuiteResult

if TYPE_CHECKING:
    from ..evaluator import EvalReport

SCHEMA_VERSION = 1


def report_to_dict(report: EvalReport) -> dict[str, Any]:
    """Serialize a report to a JSON-compatible dictionary."""
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "disclosure-eval", "version": report.tool_version},
        "generated_at": report.generated_at.isoformat(),
        "target": {"name": report.target_name, "model": report.target_model},
        "config": {
            "records_domain": report.records_domain,
            "record_count": report.record_count,
            "trials_per_suite": report.trials_per_suite,
            "seed": report.seed,
        },
        "risk_level": report.risk_level.value,
        "summary": report.summary,
        "suites": [suite.to_dict() for suite in report.suites],
    }


def report_from_dict(data: dict[str, Any]) -> EvalReport:
    """Rebuild a report from :func:`report_to_dict` output.

    Raises:
        ValueError: If the document is not a recognized report.
    """
    from ..evaluator import EvalReport

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported report schema_version {version!r}; expected {SCHEMA_VERSION}"
        )
    config = data.get("config", {})
    target = data.get("target", {})
    domain = str(config.get("records_domain", "ecommerce"))
    field_specs = field_specs_for_domain(domain)
    return EvalReport(
        target_name=str(target["name"]),
        target_model=str(target["model"]),
        records_domain=domain,
        record_count=int(config["record_count"]),
        trials_per_suite=int(config["trials_per_suite"]),
        seed=config.get("seed"),
        suites=[SuiteResult.from_dict(s, field_specs) for s in data.get("suites", [])],
        generated_at=datetime.fromisoformat(str(data["generated_at"])),
        tool_version=str(data.get("tool", {}).get("version", "unknown")),
    )


def save_report(report: EvalReport, path: str | Path) -> Path:
    """Write ``report`` as pretty-printed JSON and return the path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report_to_dict(report), indent=2), encoding="utf-8")
    return target


def load_report(path: str | Path) -> EvalReport:
    """Read a report written by :func:`save_report`.

    Raises:
        ValueError: If the file is not a recognized report.
        FileNotFoundError: If the path does not exist.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return report_from_dict(data)
