"""Command-line interface: ``disclosure-eval init | run | report``."""

from __future__ import annotations

import sys
from dataclasses import replace
from importlib import resources
from pathlib import Path

import click
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

from . import __version__
from .config import ConfigError, EvalConfig, build_target, load_config
from .evaluator import Evaluator
from .probes import get_probe
from .records import RecordSet, RecordValidationError, SyntheticRecords
from .reporting import load_report
from .scoring.aggregator import Trial, TrialError

CONFIG_FILENAME = "eval_config.yaml"
RECORDS_FILENAME = "records.yaml"

_console = Console()
_err = Console(stderr=True)


def _template_text(name: str) -> str:
    return (
        resources.files("disclosure_eval")
        .joinpath("templates")
        .joinpath(name)
        .read_text(encoding="utf-8")
    )


@click.group()
@click.version_option(__version__, prog_name="disclosure-eval")
def main() -> None:
    """Test AI customer service agents for confirmation-based PII disclosure."""


@main.command()
@click.option(
    "--dir",
    "directory",
    default=".",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory to write the config and records into.",
)
@click.option(
    "--records",
    "count",
    default=5,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of synthetic records to generate.",
)
@click.option("--seed", type=int, default=None, help="Seed for reproducible records.")
@click.option("--force", is_flag=True, help="Overwrite existing files.")
def init(directory: Path, count: int, seed: int | None, force: bool) -> None:
    """Write a sample config and generate synthetic records."""
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / CONFIG_FILENAME
    records_path = directory / RECORDS_FILENAME
    existing = [p for p in (config_path, records_path) if p.exists()]
    if existing and not force:
        names = ", ".join(str(p) for p in existing)
        raise click.ClickException(f"Refusing to overwrite {names}. Use --force to replace.")

    config_path.write_text(_template_text("eval_config.yaml"), encoding="utf-8")
    SyntheticRecords.generate(count=count, seed=seed).save_yaml(records_path)

    _console.print(f"Wrote [bold]{config_path}[/bold]")
    _console.print(f"Wrote [bold]{records_path}[/bold] ({count} synthetic records)")
    _console.print()
    _console.print("Next steps:")
    _console.print(
        "  1. Edit eval_config.yaml: set target.provider, target.model, and system_prompt."
    )
    _console.print("  2. Export your API key (ANTHROPIC_API_KEY or OPENAI_API_KEY).")
    _console.print("  3. Run: disclosure-eval run")


@main.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    default=CONFIG_FILENAME,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to the evaluation config.",
)
@click.option("--output", "-o", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--trials", type=click.IntRange(min=1), default=None, help="Override trials per suite."
)
@click.option("--model", default=None, help="Override the target model.")
@click.option("--seed", type=int, default=None, help="Override the sampling seed.")
@click.option("--show-trials", is_flag=True, help="Print every prompt and reply.")
@click.option("--quiet", "-q", is_flag=True, help="Suppress the progress bar.")
def run(
    config_path: Path,
    output: Path | None,
    trials: int | None,
    model: str | None,
    seed: int | None,
    show_trials: bool,
    quiet: bool,
) -> None:
    """Run the evaluation described by the config file."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    config = _apply_overrides(config, output=output, trials=trials, model=model, seed=seed)

    try:
        records = RecordSet.from_yaml(config.records_path)
        warnings = records.validate()
    except FileNotFoundError as exc:
        raise click.ClickException(
            f"Records file not found: {config.records_path} (run 'disclosure-eval init')"
        ) from exc
    except RecordValidationError as exc:
        raise click.ClickException(f"Invalid records: {exc}") from exc
    for warning in warnings:
        _err.print(f"[yellow]warning:[/yellow] {warning}")

    try:
        target = build_target(config.target)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc

    probes = [get_probe(name) for name in config.suites]
    evaluator = Evaluator(
        target=target,
        records=records,
        trials_per_suite=config.trials_per_suite,
        probes=probes,
        system_prompt=config.system_prompt,
        inject_records=config.inject_records,
        concurrency=config.concurrency,
        seed=config.seed,
    )
    total = sum(len(p) for p in evaluator.planned_prompts().values())

    _console.print(
        f"Evaluating [bold]{target.name}[/bold]: {len(records)} records, "
        f"{', '.join(config.suites)} ({total} requests)"
    )
    if quiet:
        report = evaluator.run()
    else:
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=_console,
            transient=True,
        ) as progress:
            task = progress.add_task("Running probes", total=total)

            def advance(_: Trial | TrialError) -> None:
                progress.advance(task)

            evaluator.on_progress = advance
            report = evaluator.run()

    report.print_summary(show_trials=show_trials)
    saved = report.save_json(config.output_path)
    _console.print(f"Full results written to [bold]{saved}[/bold]")
    if report.total_trials == 0:
        _err.print("[red]No trials were scored; every request failed.[/red]")
        sys.exit(1)
    if report.total_errors:
        _err.print(
            f"[yellow]{report.total_errors} request(s) failed; see the JSON report.[/yellow]"
        )


@main.command()
@click.argument("results", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--show-trials", is_flag=True, help="Print every prompt and reply.")
def report(results: Path, show_trials: bool) -> None:
    """Print the summary for a saved JSON results file."""
    try:
        loaded = load_report(results)
    except (ValueError, KeyError) as exc:
        raise click.ClickException(f"Could not read {results}: {exc}") from exc
    loaded.print_summary(show_trials=show_trials)


def _apply_overrides(
    config: EvalConfig,
    *,
    output: Path | None,
    trials: int | None,
    model: str | None,
    seed: int | None,
) -> EvalConfig:
    target = replace(config.target, model=model) if model else config.target
    return replace(
        config,
        target=target,
        output_path=output.resolve() if output else config.output_path,
        trials_per_suite=trials if trials is not None else config.trials_per_suite,
        seed=seed if seed is not None else config.seed,
    )


if __name__ == "__main__":
    main()
