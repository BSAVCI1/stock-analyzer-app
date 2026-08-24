from datetime import (
    datetime,
    timezone,
)
from decimal import Decimal
import importlib
from types import SimpleNamespace

import pytest

from src.deployment.bootstrap import (
    ensure_local_paper_account,
)


cycle_module = importlib.import_module(
    "src.deployment.paper_cycle"
)
NOW = datetime(
    2026,
    8,
    18,
    20,
    0,
    tzinfo=timezone.utc,
)


def test_paper_cycle_is_disabled_by_default():
    with pytest.raises(
        RuntimeError,
        match="disabled",
    ):
        cycle_module.paper_cycle(
            run_at=NOW,
            run_key="managed:1",
            environ={},
        )


@pytest.mark.parametrize(
    "name",
    [
        "PAPER_BROKER_ENABLED",
        "PAPER_BROKER_LIVE_TRADING",
    ],
)
def test_paper_cycle_rejects_broker_modes(
    name,
):
    with pytest.raises(
        RuntimeError,
        match="prohibited",
    ):
        cycle_module.paper_cycle(
            run_at=NOW,
            run_key="managed:1",
            environ={
                "BSAVCI_PAPER_CYCLE_ENABLED":
                    "true",
                name: "true",
            },
        )


def test_paper_cycle_runs_persistent_service(
    monkeypatch,
):
    report = SimpleNamespace(
        cycle=SimpleNamespace(
            failed_count=0,
        )
    )
    captured = {}

    def fake_settings(
        environ,
        *,
        database_path,
    ):
        captured["database_path"] = (
            database_path
        )
        return "settings"

    class Service:
        def run(self, *, now):
            captured["now"] = now
            return report

    def fake_runtime(
        settings,
        *,
        environ,
    ):
        captured["settings"] = settings
        captured["environ"] = environ
        return SimpleNamespace(
            orchestration_service=Service()
        )

    monkeypatch.setattr(
        cycle_module,
        "load_runtime_settings",
        fake_settings,
    )
    monkeypatch.setattr(
        cycle_module,
        "build_runtime",
        fake_runtime,
    )

    result = cycle_module.paper_cycle(
        run_at=NOW,
        run_key="managed:1",
        environ={
            "BSAVCI_PAPER_CYCLE_ENABLED":
                "true",
            "BSAVCI_DATABASE_PATH":
                "/app/data/paper.db",
        },
    )

    assert result is report
    assert captured["database_path"] == (
        "/app/data/paper.db"
    )
    assert captured["settings"] == "settings"
    assert captured["now"] == NOW


def test_paper_cycle_surfaces_persisted_failures(
    monkeypatch,
):
    report = SimpleNamespace(
        cycle=SimpleNamespace(
            failed_count=2,
        )
    )
    runtime = SimpleNamespace(
        orchestration_service=(
            SimpleNamespace(
                run=lambda **kwargs: report
            )
        )
    )
    monkeypatch.setattr(
        cycle_module,
        "load_runtime_settings",
        lambda *args, **kwargs: "settings",
    )
    monkeypatch.setattr(
        cycle_module,
        "build_runtime",
        lambda *args, **kwargs: runtime,
    )

    with pytest.raises(
        RuntimeError,
        match="2 failed",
    ):
        cycle_module.paper_cycle(
            run_at=NOW,
            run_key="managed:1",
            environ={
                "BSAVCI_PAPER_CYCLE_ENABLED":
                    "true",
                "PAPER_ACCOUNT_ID":
                    "ACC-LOCAL",
            },
        )


def test_local_bootstrap_is_idempotent(
    tmp_path,
):
    values = {
        "BSAVCI_LOCAL_PAPER_BOOTSTRAP":
            "true",
        "BSAVCI_DATABASE_PATH":
            str(tmp_path / "paper.db"),
        "PAPER_ACCOUNT_ID":
            "ACC-LOCAL",
        "PAPER_BROKER_ENABLED":
            "false",
        "PAPER_BROKER_LIVE_TRADING":
            "false",
    }

    first = ensure_local_paper_account(
        values
    )
    second = ensure_local_paper_account(
        values
    )

    assert first.account_id == "ACC-LOCAL"
    assert second.account_id == "ACC-LOCAL"
    assert first.starting_balance == (
        second.starting_balance
    )


def test_local_bootstrap_uses_configured_operational_policy(tmp_path) -> None:
    account = ensure_local_paper_account({
        "BSAVCI_LOCAL_PAPER_BOOTSTRAP": "true",
        "BSAVCI_DATABASE_PATH": str(tmp_path / "paper.db"),
        "PAPER_ACCOUNT_ID": "ACC-P4-EUR-2000",
        "PAPER_BROKER_ENABLED": "false",
        "PAPER_BROKER_LIVE_TRADING": "false",
        "BSAVCI_LOCAL_PAPER_ACCOUNT_NAME": "P4 Operational Paper Account",
        "BSAVCI_LOCAL_PAPER_BASE_CURRENCY": "EUR",
        "BSAVCI_LOCAL_PAPER_STARTING_BALANCE": "2000",
    })
    assert account.account_id == "ACC-P4-EUR-2000"
    assert account.base_currency == "EUR"
    assert account.starting_balance == Decimal("2000.00000000")


def test_local_bootstrap_rejects_existing_policy_mismatch(tmp_path) -> None:
    values = {
        "BSAVCI_LOCAL_PAPER_BOOTSTRAP": "true",
        "BSAVCI_DATABASE_PATH": str(tmp_path / "paper.db"),
        "PAPER_ACCOUNT_ID": "ACC-P4",
        "PAPER_BROKER_ENABLED": "false",
        "PAPER_BROKER_LIVE_TRADING": "false",
        "BSAVCI_LOCAL_PAPER_STARTING_BALANCE": "2000",
    }
    ensure_local_paper_account(values)
    values["BSAVCI_LOCAL_PAPER_STARTING_BALANCE"] = "3000"
    with pytest.raises(RuntimeError, match="does not match its configured policy"):
        ensure_local_paper_account(values)


def test_local_bootstrap_requires_gate():
    with pytest.raises(
        RuntimeError,
        match="disabled",
    ):
        ensure_local_paper_account({})
