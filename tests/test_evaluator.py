"""End-to-end evaluator, report round-trip, config, and CLI tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from disclosure_eval import EvalReport, Evaluator, RiskLevel, cli
from disclosure_eval.config import ConfigError, load_config, parse_config
from disclosure_eval.evaluator import build_system_prompt, render_records_block
from disclosure_eval.probes import ConfirmationProbe, OracleProbe
from disclosure_eval.records import RecordSet
from disclosure_eval.reporting import load_report
from disclosure_eval.scoring import Label, OracleStrength
from tests.conftest import (
    ScriptedTarget,
    always_verify,
    always_yes,
    failing_every_other,
    oracle_responder,
)


def test_sycophantic_agent_is_vulnerable_with_weak_oracle(records: RecordSet) -> None:
    target = ScriptedTarget(always_yes)
    report = Evaluator(target, records, trials_per_suite=6, seed=1).run()

    confirmation = report.suite("confirmation")
    assert confirmation.n == 6
    assert confirmation.counts[Label.DISCLOSED] == 6
    assert confirmation.disclosure_rate.point == 1.0
    assert report.risk_level is RiskLevel.VULNERABLE

    oracle = report.suite("oracle")
    assert oracle.n == 12
    assert oracle.counts[Label.HALLUCINATED] == 6
    assert oracle.oracle is not None
    assert oracle.oracle.score == 0.0
    assert oracle.oracle.strength is OracleStrength.WEAK
    assert len(target.calls) == 18


def test_verifying_agent_is_defended(records: RecordSet) -> None:
    report = Evaluator(ScriptedTarget(always_verify), records, trials_per_suite=5, seed=1).run()
    for suite in report.suites:
        assert suite.counts[Label.DEFENDED] == suite.n
        assert suite.verification_requested_count == suite.n
    assert report.risk_level is RiskLevel.DEFENDED
    assert "did not disclose customer data" in report.summary


def test_honest_agent_has_strong_oracle(records: RecordSet) -> None:
    report = Evaluator(
        ScriptedTarget(oracle_responder(records)), records, trials_per_suite=8, seed=3
    ).run()
    oracle = report.suite("oracle").oracle
    assert oracle is not None
    assert oracle.tpr == 1.0
    assert oracle.fpr == 0.0
    assert oracle.strength is OracleStrength.STRONG
    assert report.suite("confirmation").counts[Label.DISCLOSED] == 8


def test_failed_requests_are_recorded_not_fatal(records: RecordSet) -> None:
    target = ScriptedTarget(failing_every_other(always_yes))
    report = Evaluator(
        target, records, trials_per_suite=4, probes=[ConfirmationProbe()], seed=2
    ).run()
    suite = report.suite("confirmation")
    assert suite.n == 2
    assert len(suite.errors) == 2
    assert all("simulated provider failure" in e.error for e in suite.errors)
    assert report.total_errors == 2
    assert "2 of 2 tests" in report.summary


def test_progress_callback_and_planned_prompts(records: RecordSet) -> None:
    seen: list = []
    target = ScriptedTarget(always_verify)
    evaluator = Evaluator(target, records, trials_per_suite=3, seed=9, on_progress=seen.append)
    planned = evaluator.planned_prompts()
    report = evaluator.run()
    expected_total = sum(len(p) for p in planned.values())
    assert len(seen) == expected_total == report.total_trials
    sent = {user for _, user in target.calls}
    assert sent == {p.user_message for prompts in planned.values() for p in prompts}


def test_system_prompt_contains_records_unless_disabled(records: RecordSet) -> None:
    block = render_records_block(records)
    for record in records:
        assert record.email in block and record.order_id in block
    with_records = build_system_prompt("Base prompt.", records, inject_records=True)
    assert with_records.startswith("Base prompt.")
    assert block in with_records
    assert build_system_prompt("Base prompt.", records, inject_records=False) == "Base prompt."

    target = ScriptedTarget(always_verify)
    Evaluator(
        target, records, trials_per_suite=1, probes=[ConfirmationProbe()], inject_records=False
    ).run()
    assert records.records[0].email not in target.calls[0][0]


def test_evaluator_argument_validation(records: RecordSet) -> None:
    with pytest.raises(ValueError):
        Evaluator(ScriptedTarget(always_yes), records, trials_per_suite=0)
    with pytest.raises(ValueError):
        Evaluator(ScriptedTarget(always_yes), records, concurrency=0)


def test_json_round_trip(tmp_path: Path, records: RecordSet) -> None:
    report = Evaluator(ScriptedTarget(always_yes), records, trials_per_suite=4, seed=5).run()
    path = report.save_json(tmp_path / "out" / "results.json")
    assert path.exists()
    loaded = load_report(path)
    assert isinstance(loaded, EvalReport)
    assert loaded.risk_level is report.risk_level
    assert loaded.summary == report.summary
    assert loaded.target_name == report.target_name
    assert loaded.generated_at == report.generated_at
    for original, restored in zip(report.suites, loaded.suites, strict=True):
        assert restored.name == original.name
        assert restored.counts == original.counts
        assert restored.trials == original.trials
        assert restored.disclosure_rate == original.disclosure_rate
        assert restored.oracle == original.oracle
    assert EvalReport.from_dict(report.to_dict()).to_dict() == report.to_dict()


def test_report_rejects_unknown_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"schema_version": 99}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_report(path)


def test_print_summary_renders(records: RecordSet, capsys: pytest.CaptureFixture[str]) -> None:
    report = Evaluator(ScriptedTarget(always_yes), records, trials_per_suite=2, seed=5).run()
    report.print_summary(show_trials=True)
    out = capsys.readouterr().out
    assert "VULNERABLE" in out
    assert "confirmation suite" in out
    assert "Oracle score" in out
    assert "confirm-" in out


def test_config_parsing_and_validation(tmp_path: Path) -> None:
    config = parse_config(
        {
            "target": {"provider": "openai", "model": "gpt-test", "max_tokens": 256},
            "records": "recs.yaml",
            "trials_per_suite": 3,
            "suites": ["oracle", "oracle"],
            "seed": 7,
        },
        tmp_path,
    )
    assert config.target.provider == "openai"
    assert config.target.max_tokens == 256
    assert config.records_path == (tmp_path / "recs.yaml").resolve()
    assert config.suites == ("oracle",)
    assert config.seed == 7

    with pytest.raises(ConfigError, match="'target' section"):
        parse_config({}, tmp_path)
    with pytest.raises(ConfigError, match="Unknown target provider"):
        parse_config({"target": {"provider": "ollama", "model": "x"}}, tmp_path)
    with pytest.raises(ConfigError, match="Unknown suite"):
        parse_config({"target": {"provider": "openai", "model": "x"}, "suites": ["doc"]}, tmp_path)
    with pytest.raises(ConfigError, match="positive integer"):
        parse_config(
            {"target": {"provider": "openai", "model": "x"}, "trials_per_suite": -1}, tmp_path
        )
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")

    broken = tmp_path / "broken.yaml"
    broken.write_text("target: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(broken)


def test_cli_init_run_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli.main, ["init", "--dir", str(tmp_path), "--records", "4", "--seed", "1"]
    )
    assert result.exit_code == 0, result.output
    config_path = tmp_path / cli.CONFIG_FILENAME
    records_path = tmp_path / cli.RECORDS_FILENAME
    assert config_path.exists() and records_path.exists()
    assert len(RecordSet.from_yaml(records_path)) == 4

    result = runner.invoke(cli.main, ["init", "--dir", str(tmp_path)])
    assert result.exit_code != 0
    assert "Refusing to overwrite" in result.output

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert loaded["target"]["provider"] == "anthropic"

    monkeypatch.setattr(cli, "build_target", lambda _cfg: ScriptedTarget(always_verify))
    output_path = tmp_path / "results.json"
    result = runner.invoke(
        cli.main,
        [
            "run",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--trials",
            "3",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "DEFENDED" in result.output
    assert output_path.exists()

    result = runner.invoke(cli.main, ["report", str(output_path), "--show-trials"])
    assert result.exit_code == 0, result.output
    assert "DEFENDED" in result.output
    assert "oracle suite" in result.output


def test_cli_run_without_records_file(tmp_path: Path) -> None:
    config_path = tmp_path / "eval_config.yaml"
    config_path.write_text(
        yaml.safe_dump({"target": {"provider": "openai", "model": "x"}, "records": "nope.yaml"}),
        encoding="utf-8",
    )
    result = CliRunner().invoke(cli.main, ["run", "--config", str(config_path)])
    assert result.exit_code != 0
    assert "Records file not found" in result.output


def test_custom_probe_list(records: RecordSet) -> None:
    report = Evaluator(
        ScriptedTarget(always_yes), records, trials_per_suite=2, probes=[OracleProbe()], seed=0
    ).run()
    assert [s.name for s in report.suites] == ["oracle"]
