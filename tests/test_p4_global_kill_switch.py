"""P4.9 global kill-switch acceptance tests."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from src.automation import AutomationRepository
from src.jobs.cli import main
from src.paper import PaperRepository


def _account(tmp_path):
    database_path = tmp_path / "kill-switch.db"
    paper = PaperRepository(database_path)
    account = paper.create_account(
        name="Kill Switch Test",
        base_currency="EUR",
        starting_balance="100000",
    )
    return database_path, paper, account


def test_kill_switch_is_inactive_by_default(tmp_path) -> None:
    database_path, _, account = _account(tmp_path)
    state = AutomationRepository(database_path).get_control(
        account.account_id,
        at=datetime.now(timezone.utc),
    )

    assert state.kill_switch_active is False
    assert state.kill_switch_reason is None


def test_operator_changes_are_persistent_idempotent_and_audited(
    tmp_path,
) -> None:
    database_path, paper, account = _account(tmp_path)
    controls = AutomationRepository(database_path)
    at = datetime.now(timezone.utc)

    first = controls.set_kill_switch(
        account.account_id,
        active=True,
        reason="Unexpected provider behaviour",
        changed_by="operator@example.com",
        updated_at=at,
    )
    duplicate = controls.set_kill_switch(
        account.account_id,
        active=True,
        reason="This must not replace the original reason",
        changed_by="operator@example.com",
        updated_at=at,
    )

    assert first.kill_switch_active is True
    assert duplicate.kill_switch_reason == (
        "Unexpected provider behaviour"
    )
    assert AutomationRepository(database_path).get_control(
        account.account_id,
        at=at,
    ).kill_switch_active is True

    events = tuple(
        event
        for event in paper.list_system_events(account.account_id)
        if event.event_type.startswith("GLOBAL_KILL_SWITCH_")
    )
    assert [event.event_type for event in events] == [
        "GLOBAL_KILL_SWITCH_ACTIVATED"
    ]
    assert events[0].severity == "CRITICAL"


def test_cli_activation_status_and_deactivation(tmp_path, capsys) -> None:
    database_path, _, account = _account(tmp_path)
    common = [
        "--database",
        str(database_path),
        "--account-id",
        account.account_id,
    ]

    assert main(
        [
            "kill-switch",
            *common,
            "activate",
            "--reason",
            "Operator drill",
            "--operator",
            "salih",
        ]
    ) == 0
    activated = json.loads(capsys.readouterr().out)
    assert activated["active"] is True
    assert activated["new_orders_allowed"] is False

    assert main(["kill-switch", *common, "status"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["active"] is True
    assert status["changed"] is False

    assert main(
        [
            "kill-switch",
            *common,
            "deactivate",
            "--reason",
            "Drill completed and reviewed",
            "--operator",
            "salih",
        ]
    ) == 0
    deactivated = json.loads(capsys.readouterr().out)
    assert deactivated["active"] is False
    assert deactivated["new_orders_allowed"] is True
