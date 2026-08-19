"""P4.9 per-strategy pause operator controls."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from src.automation import AutomationRepository
from src.jobs.cli import main
from src.paper import PaperRepository


NOW = datetime(2026, 8, 19, 8, tzinfo=timezone.utc)


def _account(tmp_path):
    database_path = tmp_path / "strategy-pause.db"
    paper = PaperRepository(database_path)
    account = paper.create_account(
        name="Strategy Pause Test",
        base_currency="EUR",
        starting_balance="100000",
    )
    return database_path, paper, account


def test_pause_is_persistent_idempotent_and_audited(tmp_path) -> None:
    database_path, paper, account = _account(tmp_path)
    repository = AutomationRepository(database_path)

    first = repository.set_strategy_pause(
        account.account_id,
        strategy=" Trend_Pullback ",
        active=True,
        reason="Review required",
        changed_by="salih",
        changed_at=NOW,
    )
    duplicate = repository.set_strategy_pause(
        account.account_id,
        strategy="trend_pullback",
        active=True,
        reason="Must not replace the first reason",
        changed_by="salih",
        changed_at=NOW,
    )

    assert first.strategy == "trend_pullback"
    assert duplicate.reason == "Review required"
    assert AutomationRepository(database_path).get_strategy_pause(
        account.account_id,
        "trend_pullback",
    ).active is True
    events = tuple(
        event
        for event in paper.list_system_events(account.account_id)
        if event.event_type.startswith("STRATEGY_PAUSE_")
    )
    assert len(events) == 1
    assert events[0].event_type == "STRATEGY_PAUSE_ACTIVATED"
    assert events[0].metadata["changed_by"] == "salih"


def test_strategy_pause_cli_lifecycle(tmp_path, capsys) -> None:
    database_path, _, account = _account(tmp_path)
    common = [
        "--database",
        str(database_path),
        "--account-id",
        account.account_id,
    ]

    result = main(
        [
            "strategy-pause",
            *common,
            "activate",
            "trend_pullback",
            "--reason",
            "Controlled review",
            "--operator",
            "salih",
        ]
    )
    activated = json.loads(capsys.readouterr().out)
    assert result == 0
    assert activated["active"] is True
    assert activated["new_entries_allowed"] is False

    assert main(["strategy-pause", *common, "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["active_strategies"] == ["trend_pullback"]

    assert main(
        [
            "strategy-pause",
            *common,
            "deactivate",
            "trend_pullback",
            "--reason",
            "Review completed",
            "--operator",
            "salih",
        ]
    ) == 0
    deactivated = json.loads(capsys.readouterr().out)
    assert deactivated["active"] is False
    assert deactivated["new_entries_allowed"] is True
