from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.execution_adapters import (
    InternalPaperExecutionAdapter,
)
from src.jobs.cli import main
from src.jobs.runtime import (
    build_runtime,
    load_runtime_settings,
    make_release_gate_lookup,
)
from src.paper import PaperRepository



RUNTIME_ENVIRONMENT_KEYS = (
    "PAPER_DATABASE_PATH",
    "PAPER_ACCOUNT_ID",
    "PAPER_UNIVERSE_PATH",
    "PAPER_RELEASE_ELIGIBLE_STRATEGIES",
    "PAPER_APP_VERSION",
    "PAPER_THRESHOLD_VERSION",
    "PAPER_TELEGRAM_BOT_TOKEN",
    "PAPER_TELEGRAM_CHAT_ID",
    "PAPER_TELEGRAM_TIMEOUT",
    "PAPER_SMTP_HOST",
    "PAPER_SMTP_PORT",
    "PAPER_SMTP_USERNAME",
    "PAPER_SMTP_PASSWORD",
    "PAPER_SMTP_STARTTLS",
    "PAPER_SMTP_SSL",
    "PAPER_SMTP_TIMEOUT",
    "PAPER_EMAIL_FROM",
    "PAPER_EMAIL_TO",
)


@pytest.fixture(autouse=True)
def isolate_runtime_environment(
    monkeypatch,
) -> None:
    """Prevent developer shell settings from changing tests."""

    for key in RUNTIME_ENVIRONMENT_KEYS:
        monkeypatch.delenv(
            key,
            raising=False,
        )


def create_account(
    tmp_path,
):
    database_path = (
        tmp_path / "cli.db"
    )

    repository = PaperRepository(
        database_path
    )

    account = repository.create_account(
        name="CLI Test",
        base_currency="USD",
        starting_balance="10000",
    )

    return database_path, account


def test_runtime_settings_require_account() -> None:
    with pytest.raises(
        ValueError,
        match="account ID is required",
    ):
        load_runtime_settings({})


def test_runtime_settings_read_environment(
    tmp_path,
) -> None:
    database_path = (
        tmp_path / "runtime.db"
    )

    settings = load_runtime_settings(
        {
            "PAPER_DATABASE_PATH":
            str(database_path),
            "PAPER_ACCOUNT_ID":
            "ACC-001",
            "PAPER_UNIVERSE_PATH":
            "config/custom.json",
            "PAPER_RELEASE_ELIGIBLE_STRATEGIES":
            "trend_pullback, mean_reversion",
            "PAPER_APP_VERSION":
            "test-version",
            "PAPER_THRESHOLD_VERSION":
            "test-thresholds",
        }
    )

    assert (
        settings.database_path
        == database_path
    )

    assert settings.account_id == "ACC-001"

    assert (
        settings.universe_path
        == Path("config/custom.json")
    )

    assert (
        settings
        .release_eligible_strategies
        == (
            "trend_pullback",
            "mean_reversion",
        )
    )


def test_release_gate_denies_by_default() -> None:
    lookup = make_release_gate_lookup(())

    assert lookup(
        "trend_pullback"
    ) is None


def test_release_gate_allows_only_configured_strategy() -> None:
    lookup = make_release_gate_lookup(
        ("trend_pullback",)
    )

    approved = lookup(
        "trend_pullback"
    )

    rejected = lookup(
        "mean_reversion"
    )

    assert (
        approved
        .alert_scheduling_eligible
        is True
    )

    assert (
        rejected
        .alert_scheduling_eligible
        is False
    )


def test_runtime_without_delivery_configuration(
    tmp_path,
) -> None:
    database_path, account = (
        create_account(tmp_path)
    )

    settings = load_runtime_settings(
        {
            "PAPER_DATABASE_PATH":
            str(database_path),
            "PAPER_ACCOUNT_ID":
            account.account_id,
        }
    )

    runtime = build_runtime(
        settings,
        environ={
            "PAPER_DATABASE_PATH":
            str(database_path),
            "PAPER_ACCOUNT_ID":
            account.account_id,
        },
    )

    assert (
        runtime.notification_channels
        == ()
    )

    assert isinstance(
        runtime.execution_adapter,
        InternalPaperExecutionAdapter,
    )

    assert (
        runtime.execution_adapter
        .descriptor
        .live_trading_enabled
        is False
    )


def test_status_command_outputs_account_state(
    tmp_path,
    capsys,
) -> None:
    database_path, account = (
        create_account(tmp_path)
    )

    result = main(
        [
            "status",
            "--database",
            str(database_path),
            "--account-id",
            account.account_id,
        ]
    )

    captured = capsys.readouterr()

    payload = json.loads(
        captured.out
    )

    assert result == 0

    assert (
        payload["account"]["account_id"]
        == account.account_id
    )

    assert (
        payload["portfolio"][
            "reconciled"
        ]
        is True
    )

    assert (
        payload[
            "release_configuration"
        ]["deny_by_default"]
        is True
    )


def test_closed_day_cli_cycle_is_idempotent(
    tmp_path,
    capsys,
) -> None:
    database_path, account = (
        create_account(tmp_path)
    )

    arguments = [
        "market-cycle",
        "--database",
        str(database_path),
        "--account-id",
        account.account_id,
        "--at",
        "2026-08-01T20:30:00+00:00",
    ]

    first_result = main(arguments)

    first_output = json.loads(
        capsys.readouterr().out
    )

    second_result = main(arguments)

    second_output = json.loads(
        capsys.readouterr().out
    )

    assert first_result == 0
    assert second_result == 0

    assert (
        first_output["status"]
        == "SKIPPED"
    )

    assert (
        first_output["duplicate"]
        is False
    )

    assert (
        second_output["duplicate"]
        is True
    )

    assert (
        first_output["job_run_id"]
        == second_output["job_run_id"]
    )


def test_empty_dispatch_command_succeeds(
    tmp_path,
    capsys,
) -> None:
    database_path, account = (
        create_account(tmp_path)
    )

    result = main(
        [
            "dispatch",
            "--database",
            str(database_path),
            "--account-id",
            account.account_id,
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert result == 0
    assert payload["processed"] == 0
    assert payload["sent"] == 0
    assert payload["failed"] == 0
