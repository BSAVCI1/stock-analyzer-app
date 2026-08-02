from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from types import SimpleNamespace

import src.jobs.cli as cli


T0 = datetime(
    2026,
    8,
    2,
    21,
    0,
    tzinfo=timezone.utc,
)


def metric(
    name: str,
    *,
    total: int = 1,
    successful: int = 1,
    failed: int = 0,
    pending_or_other: int = 0,
):
    return SimpleNamespace(
        name=name,
        total=total,
        successful=successful,
        failed=failed,
        pending_or_other=(
            pending_or_other
        ),
        provenance=SimpleNamespace(
            source_tables=(
                f"paper_{name}",
            ),
        ),
    )


def make_snapshot(
    *,
    notification_total: int = 1,
):
    return SimpleNamespace(
        generated_at=T0,
        account=SimpleNamespace(
            account_id="ACC-P3-CLI"
        ),
        reconciliation=SimpleNamespace(
            reconciled=True,
            difference=Decimal("0"),
        ),
        broker_reconciliation_run=(
            SimpleNamespace(
                reconciliation_run_id=(
                    "BRR-P3-CLI"
                ),
                provider=(
                    "deterministic-paper"
                ),
                status=SimpleNamespace(
                    value="MATCHED"
                ),
                reconciled=True,
                unresolved_item_count=0,
                account_item_count=1,
                order_item_count=1,
                position_item_count=1,
                error_message=None,
            )
        ),
        reliability=SimpleNamespace(
            scans=metric("scans"),
            execution_runs=metric(
                "execution_runs"
            ),
            scheduled_jobs=metric(
                "scheduled_jobs"
            ),
            notifications=metric(
                "notifications",
                total=notification_total,
                successful=(
                    notification_total
                ),
            ),
            system_events=metric(
                "system_events"
            ),
        ),
    )


def make_runtime(
    *,
    live: bool = False,
):
    return SimpleNamespace(
        settings=SimpleNamespace(
            database_path="ignored.db",
            account_id="ACC-P3-CLI",
        ),
        execution_adapter=(
            SimpleNamespace(
                descriptor=SimpleNamespace(
                    live_trading_enabled=live
                )
            )
        ),
    )


class FakeDashboardService:
    snapshot = make_snapshot()

    def __init__(
        self,
        repository,
    ) -> None:
        self.repository = repository

    def build_snapshot(
        self,
        account_id,
        *,
        generated_at,
    ):
        assert account_id == "ACC-P3-CLI"
        assert generated_at == T0

        return self.snapshot


def install_fakes(
    monkeypatch,
    *,
    runtime=None,
    snapshot=None,
) -> None:
    selected_runtime = (
        runtime or make_runtime()
    )

    FakeDashboardService.snapshot = (
        snapshot or make_snapshot()
    )

    monkeypatch.setattr(
        cli,
        "_runtime_from_args",
        lambda args: selected_runtime,
    )

    monkeypatch.setattr(
        cli,
        "PortfolioDashboardRepository",
        lambda database_path:
        SimpleNamespace(
            database_path=database_path
        ),
    )

    monkeypatch.setattr(
        cli,
        "PortfolioDashboardService",
        FakeDashboardService,
    )


def command_arguments(
    *,
    regression_passed: bool = True,
):
    arguments = [
        "p3-release-status",
        "--database",
        "ignored.db",
        "--account-id",
        "ACC-P3-CLI",
        "--test-count",
        "380",
        "--workflow",
        "Automated tests #43",
        "--at",
        T0.isoformat(),
    ]

    if regression_passed:
        arguments.append(
            "--regression-passed"
        )

    return arguments


def test_parser_registers_p3_release_status():
    args = cli.build_parser().parse_args(
        command_arguments()
    )

    assert (
        args.command
        == "p3-release-status"
    )

    assert args.test_count == 380
    assert args.regression_passed is True


def test_ready_release_returns_zero(
    monkeypatch,
    capsys,
) -> None:
    install_fakes(monkeypatch)

    result = cli.main(
        command_arguments()
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert result == 0
    assert payload["status"] == "READY"
    assert payload["release_ready"] is True

    assert (
        payload[
            "operational_reliability"
        ]["live_trading_enabled"]
        is False
    )


def test_missing_regression_attestation_blocks(
    monkeypatch,
    capsys,
) -> None:
    install_fakes(monkeypatch)

    result = cli.main(
        command_arguments(
            regression_passed=False
        )
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert result == 1
    assert payload["status"] == "BLOCKED"

    assert any(
        "regression" in reason.lower()
        for reason in payload["reasons"]
    )


def test_missing_operational_evidence_blocks(
    monkeypatch,
    capsys,
) -> None:
    install_fakes(
        monkeypatch,
        snapshot=make_snapshot(
            notification_total=0
        ),
    )

    result = cli.main(
        command_arguments()
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert result == 1
    assert payload["status"] == "BLOCKED"

    notification = next(
        check
        for check
        in payload[
            "operational_reliability"
        ]["checks"]
        if check["name"] == "notifications"
    )

    assert (
        notification["status"]
        == "NOT_OBSERVED"
    )


def test_live_descriptor_blocks_release(
    monkeypatch,
    capsys,
) -> None:
    install_fakes(
        monkeypatch,
        runtime=make_runtime(
            live=True
        ),
    )

    result = cli.main(
        command_arguments()
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert result == 1
    assert payload["status"] == "BLOCKED"

    assert (
        payload[
            "operational_reliability"
        ]["live_trading_enabled"]
        is True
    )
