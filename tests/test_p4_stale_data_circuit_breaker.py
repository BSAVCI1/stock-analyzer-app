"""P4.9 stale-data circuit-breaker persistence and status."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from src.automation import AutomationRepository
from src.jobs.cli import main
from src.paper import PaperRepository


NOW = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)


def _account(tmp_path):
    database_path = tmp_path / "stale-breaker.db"
    paper = PaperRepository(database_path)
    account = paper.create_account(
        name="Stale Breaker Test",
        base_currency="EUR",
        starting_balance="100000",
    )
    return database_path, paper, account


def test_breaker_trip_is_idempotent_and_audited(tmp_path) -> None:
    database_path, paper, account = _account(tmp_path)
    repository = AutomationRepository(database_path)

    first = repository.trip_circuit_breaker(
        account.account_id,
        breaker_type="stale_data",
        reason="AAPL data is stale",
        tripped_at=NOW,
        metadata={"symbols": ["AAPL"]},
    )
    duplicate = repository.trip_circuit_breaker(
        account.account_id,
        breaker_type="STALE_DATA",
        reason="Must not replace the original incident",
        tripped_at=NOW,
    )

    assert first.active is True
    assert duplicate.reason == "AAPL data is stale"
    events = tuple(
        event
        for event in paper.list_system_events(account.account_id)
        if event.event_type == "CIRCUIT_BREAKER_TRIPPED"
    )
    assert len(events) == 1
    assert events[0].severity == "ERROR"


def test_circuit_breaker_cli_reports_persisted_state(
    tmp_path,
    capsys,
) -> None:
    database_path, _, account = _account(tmp_path)
    AutomationRepository(database_path).trip_circuit_breaker(
        account.account_id,
        breaker_type="STALE_DATA",
        reason="Market history exceeded freshness policy",
        tripped_at=NOW,
    )

    result = main(
        [
            "circuit-breaker",
            "--database",
            str(database_path),
            "--account-id",
            account.account_id,
            "status",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload["active_breaker_types"] == ["STALE_DATA"]
    assert payload["circuit_breakers"][0]["active"] is True
    assert (
        payload["circuit_breakers"][0]["new_entries_allowed"]
        is False
    )
