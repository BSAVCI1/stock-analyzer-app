"""P4.9 operational incident-log acceptance tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from src.jobs.cli import main
from src.paper import (
    IncidentSeverity,
    IncidentStatus,
    PaperRepository,
    SCHEMA_VERSION,
)


NOW = datetime(2026, 8, 19, 12, tzinfo=timezone.utc)


def _account(tmp_path):
    database_path = tmp_path / "operational-incidents.db"
    paper = PaperRepository(database_path)
    account = paper.create_account(
        name="Incident Test",
        base_currency="EUR",
        starting_balance="100000",
    )
    return database_path, paper, account


def test_schema_version_fifteen_adds_incident_log(tmp_path) -> None:
    database_path, _, _ = _account(tmp_path)
    connection = sqlite3.connect(database_path)
    try:
        version = connection.execute(
            "PRAGMA user_version"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()

    assert version == SCHEMA_VERSION == 16
    assert "paper_operational_incidents" in tables


def test_incident_lifecycle_is_persistent_and_audited(tmp_path) -> None:
    database_path, paper, account = _account(tmp_path)
    incident = paper.open_incident(
        account_id=account.account_id,
        title="Provider outage during cycle",
        severity=IncidentSeverity.HIGH,
        summary="All requested symbols failed at provider boundary.",
        opened_by="salih",
        opened_at=NOW,
        reference_type="JOB_RUN",
        reference_id="JOB-123",
    )
    monitoring = paper.update_incident(
        incident.incident_id,
        changed_by="salih",
        note="Provider recovered; monitoring next cycle.",
        status=IncidentStatus.MONITORING,
        changed_at=NOW + timedelta(minutes=15),
    )
    resolved = PaperRepository(database_path).resolve_incident(
        incident.incident_id,
        root_cause="Provider authentication endpoint was unavailable.",
        resolution="Credentials verified and clean cycle completed.",
        resolved_by="salih",
        resolved_at=NOW + timedelta(minutes=30),
    )

    assert monitoring.status is IncidentStatus.MONITORING
    assert resolved.status is IncidentStatus.RESOLVED
    assert resolved.root_cause.startswith("Provider authentication")
    assert resolved.resolved_by == "salih"
    assert PaperRepository(database_path).get_incident(
        incident.incident_id
    ) == resolved

    events = tuple(
        event
        for event in paper.list_system_events(account.account_id)
        if event.reference_id == incident.incident_id
    )
    assert [event.event_type for event in events] == [
        "OPERATIONAL_INCIDENT_OPENED",
        "OPERATIONAL_INCIDENT_UPDATED",
        "OPERATIONAL_INCIDENT_RESOLVED",
    ]

    duplicate = paper.resolve_incident(
        incident.incident_id,
        root_cause="Different text must not replace evidence.",
        resolution="Different resolution.",
        resolved_by="other",
        resolved_at=NOW + timedelta(hours=1),
    )
    assert duplicate == resolved
    assert len(
        tuple(
            event
            for event in paper.list_system_events(account.account_id)
            if event.reference_id == incident.incident_id
        )
    ) == 3

    with pytest.raises(ValueError, match="resolved incident"):
        paper.update_incident(
            incident.incident_id,
            changed_by="salih",
            note="Must not reopen.",
            status=IncidentStatus.OPEN,
        )


def test_incident_cli_open_review_and_resolve(tmp_path, capsys) -> None:
    database_path, _, account = _account(tmp_path)
    common = [
        "--database",
        str(database_path),
        "--account-id",
        account.account_id,
    ]

    assert main(
        [
            "incident",
            *common,
            "open",
            "--title",
            "Restart recovery required",
            "--severity",
            "CRITICAL",
            "--summary",
            "A scheduled job remained running after interruption.",
            "--operator",
            "salih",
        ]
    ) == 0
    opened = json.loads(capsys.readouterr().out)
    incident_id = opened["incident_id"]
    assert opened["status"] == "OPEN"

    assert main(
        [
            "incident",
            *common,
            "update",
            incident_id,
            "--status",
            "MONITORING",
            "--note",
            "Recovery replay completed without duplicates.",
            "--operator",
            "salih",
        ]
    ) == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["status"] == "MONITORING"

    assert main(
        [
            "incident",
            *common,
            "show",
            incident_id,
        ]
    ) == 0
    shown = json.loads(capsys.readouterr().out)
    assert len(shown["timeline"]) == 2

    assert main(
        [
            "incident",
            *common,
            "resolve",
            incident_id,
            "--root-cause",
            "Running jobs were previously classified as duplicates.",
            "--resolution",
            "Resume running keys through the managed recovery path.",
            "--operator",
            "salih",
        ]
    ) == 0
    resolved = json.loads(capsys.readouterr().out)
    assert resolved["status"] == "RESOLVED"
    assert resolved["root_cause"].startswith("Running jobs")

    assert main(
        [
            "incident",
            *common,
            "list",
            "--status",
            "RESOLVED",
        ]
    ) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed["total"] == 1
    assert listed["incidents"][0]["incident_id"] == incident_id

    assert main(["status", *common]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["incidents"] == {
        "total": 1,
        "open": 0,
        "critical_open": 0,
    }
