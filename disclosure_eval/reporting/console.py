"""Terminal report rendering with rich."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ..scoring.aggregator import RiskLevel, SuiteResult
from ..scoring.classifier import Label

if TYPE_CHECKING:
    from ..evaluator import EvalReport

_RISK_STYLES: dict[RiskLevel, str] = {
    RiskLevel.VULNERABLE: "bold white on red",
    RiskLevel.PARTIAL: "bold black on yellow",
    RiskLevel.DEFENDED: "bold white on green",
}

_LABEL_STYLES: dict[Label, str] = {
    Label.DISCLOSED: "red",
    Label.PARTIAL: "yellow",
    Label.DEFENDED: "green",
    Label.HALLUCINATED: "magenta",
}


def print_report(
    report: EvalReport,
    *,
    console: Console | None = None,
    show_trials: bool = False,
) -> None:
    """Print the report header, per-suite tables, and summary.

    Args:
        report: The evaluation report to render.
        console: Console to write to; defaults to a new stdout console.
        show_trials: Also print every trial's prompt, reply, and label.
    """
    out = console or Console()
    out.print(_header(report))
    for suite in report.suites:
        out.print(_suite_table(suite))
        fields = suite.fields_disclosed
        if fields:
            listed = ", ".join(f"{d.field} ({d.sensitivity.value}) x{d.count}" for d in fields)
            out.print(f"  Fields disclosed: {escape(listed)}")
        oracle = suite.oracle
        if oracle is not None:
            out.print(
                f"  Oracle score: [bold]{oracle.score:.2f}[/bold] "
                f"(TPR {oracle.tpr:.2f}, FPR {oracle.fpr:.2f}) - "
                f"{oracle.strength.value} oracle"
            )
        if suite.errors:
            out.print(
                f"  [yellow]{len(suite.errors)} request(s) failed and were not scored[/yellow]"
            )
        if show_trials:
            _print_trials(out, suite)
        out.print()
    out.print(Panel(report.summary, title="Summary", border_style="dim"))


def _header(report: EvalReport) -> Panel:
    risk = report.risk_level
    body = Text()
    body.append("Target: ", style="dim")
    body.append(f"{report.target_name}\n")
    body.append("Records: ", style="dim")
    body.append(f"{report.record_count} ({report.records_domain})    ")
    body.append("Trials/suite: ", style="dim")
    body.append(f"{report.trials_per_suite}    ")
    body.append("Seed: ", style="dim")
    body.append(f"{report.seed if report.seed is not None else 'none'}\n")
    body.append("Risk level: ", style="dim")
    body.append(f" {risk.value} ", style=_RISK_STYLES[risk])
    return Panel(body, title="disclosure-eval", border_style="blue")


_LABEL_HEADERS: dict[Label, str] = {
    Label.DISCLOSED: "Disclosed",
    Label.PARTIAL: "Partial",
    Label.DEFENDED: "Defended",
    Label.HALLUCINATED: "Halluc.",
}


def _suite_table(suite: SuiteResult) -> Table:
    table = Table(
        title=f"{suite.name} suite",
        title_justify="left",
        box=box.SIMPLE_HEAD,
        show_edge=False,
        pad_edge=False,
        collapse_padding=True,
    )
    table.add_column("Trials", justify="right")
    for label in Label:
        table.add_column(_LABEL_HEADERS[label], justify="right", style=_LABEL_STYLES[label])
    table.add_column("Rate", justify="right", no_wrap=True)
    table.add_column("95% CI", no_wrap=True)
    table.add_column("Verify req.", justify="right", no_wrap=True)

    counts = suite.counts
    rate = suite.disclosure_rate
    if suite.n:
        rate_text = f"{rate.point * 100:.0f}%"
        ci_text = f"{rate.lower * 100:.1f}-{rate.upper * 100:.1f}%"
    else:
        rate_text, ci_text = "n/a", "n/a"
    table.add_row(
        str(suite.n),
        *(str(counts[label]) for label in Label),
        rate_text,
        ci_text,
        f"{suite.verification_requested_count}/{suite.n}",
    )
    return table


def _print_trials(out: Console, suite: SuiteResult) -> None:
    for index, trial in enumerate(suite.trials, start=1):
        label = trial.classification.label
        correctness = "correct" if trial.prompt.value_is_correct else "incorrect"
        style = _LABEL_STYLES[label]
        out.print(
            f"  {index:02d}. [{style}]{label.value}[/{style}] "
            + escape(
                f"{trial.prompt.template_id} / {trial.prompt.record_id} / {trial.prompt.field} "
                f"({correctness}) - {trial.classification.reason}"
            )
        )
        out.print(f"      > {escape(trial.prompt.user_message)}", style="dim")
        reply = trial.response.text.replace("\n", " ")
        out.print(f"      < {escape(reply)}", style="dim")
