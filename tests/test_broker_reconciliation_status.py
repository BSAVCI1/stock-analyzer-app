from __future__ import annotations

from datetime import (
    datetime,
    timezone,
)
import json

from src.execution_adapters import (
    BrokerReconciliationCategory,
    BrokerReconciliationItem,
    BrokerReconciliationItemStatus,
    BrokerReconciliationRepository,
)
from src.jobs.cli import main
from src.paper import (
    PaperRepository,
    PaperTradingService,
)
from src.portfolio_dashboard import (
    PortfolioDashboardRepository,
    PortfolioDashboardService,
    broker_reconciliation_item_rows,
    broker_reconciliation_summary_rows,
)


T0 = datetime(
    2026,
    8,
    3,
    20,
    0,
    tzinfo=timezone.utc,
)


def make_environment(tmp_path):
    database_path = (
        tmp_path / "status.db"
    )

    paper_repository = PaperRepository(
        database_path
    )

    paper_service = PaperTradingService(
        paper_repository
    )

    account = paper_service.create_account(
        name="Reconciliation status test",
        created_at=T0,
    )

    reconciliation_repository = (
        BrokerReconciliationRepository(
            database_path
        )
    )

    dashboard_repository = (
        PortfolioDashboardRepository(
            database_path
        )
    )

    dashboard_service = (
        PortfolioDashboardService(
            dashboard_repository
        )
    )

    return (
        database_path,
        account,
        reconciliation_repository,
        dashboard_service,
    )


def start_run(
    repository,
    account_id,
    *,
    key,
):
    run, created = repository.start_run(
        account_id=account_id,
        reconciliation_key=key,
        provider="Example",
        broker_account_id=(
            "BROKER-PAPER-1"
        ),
        started_at=T0,
        metadata={
            "paper_only": True,
            "read_only": True,
        },
    )

    assert created is True

    return run


def test_cli_reports_missing_run(
    tmp_path,
    capsys,
) -> None:
    (
        database_path,
        account,
        _,
        _,
    ) = make_environment(tmp_path)

    result = main(
        [
            "broker-reconciliation-status",
            "--database",
            str(database_path),
            "--account-id",
            account.account_id,
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert result == 2
    assert payload["status"] == "NOT_RUN"
    assert payload["latest_run"] is None


def test_cli_reports_matched_run(
    tmp_path,
    capsys,
) -> None:
    (
        database_path,
        account,
        repository,
        _,
    ) = make_environment(tmp_path)

    run = start_run(
        repository,
        account.account_id,
        key="MATCHED-RUN",
    )

    repository.complete_run(
        run.reconciliation_run_id,
        items=(),
        completed_at=T0,
    )

    result = main(
        [
            "broker-reconciliation-status",
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
    assert payload["status"] == "MATCHED"

    assert (
        payload["latest_run"][
            "unresolved_item_count"
        ]
        == 0
    )


def test_cli_reports_unresolved_difference(
    tmp_path,
    capsys,
) -> None:
    (
        database_path,
        account,
        repository,
        _,
    ) = make_environment(tmp_path)

    run = start_run(
        repository,
        account.account_id,
        key="DIFFERENCE-RUN",
    )

    item = BrokerReconciliationItem(
        reconciliation_item_id="BRI-1",
        reconciliation_run_id=(
            run.reconciliation_run_id
        ),
        account_id=account.account_id,
        category=(
            BrokerReconciliationCategory
            .ACCOUNT
        ),
        comparison_key=(
            account.account_id
        ),
        status=(
            BrokerReconciliationItemStatus
            .MISMATCH
        ),
        internal_reference_ids=(
            account.account_id,
        ),
        broker_reference_ids=(
            "BROKER-PAPER-1",
        ),
        differences={
            "cash": {
                "internal": "10000",
                "broker": "9990",
            }
        },
        message="Cash differs.",
        created_at=T0,
        metadata={},
    )

    repository.complete_run(
        run.reconciliation_run_id,
        items=(item,),
        completed_at=T0,
    )

    result = main(
        [
            "broker-reconciliation-status",
            "--database",
            str(database_path),
            "--account-id",
            account.account_id,
        ]
    )

    payload = json.loads(
        capsys.readouterr().out
    )

    assert result == 1

    assert (
        payload["status"]
        == "DIFFERENCES"
    )

    assert len(
        payload["unresolved_items"]
    ) == 1


def test_dashboard_has_empty_broker_state(
    tmp_path,
) -> None:
    (
        _,
        account,
        _,
        dashboard_service,
    ) = make_environment(tmp_path)

    snapshot = (
        dashboard_service.build_snapshot(
            account.account_id,
            generated_at=T0,
        )
    )

    assert (
        snapshot
        .broker_reconciliation_run
        is None
    )

    assert (
        snapshot
        .broker_reconciliation_items
        == ()
    )

    assert (
        snapshot.metadata[
            "broker_reconciliation_available"
        ]
        is False
    )


def test_dashboard_loads_latest_difference(
    tmp_path,
) -> None:
    (
        _,
        account,
        repository,
        dashboard_service,
    ) = make_environment(tmp_path)

    run = start_run(
        repository,
        account.account_id,
        key="DASHBOARD-RUN",
    )

    item = BrokerReconciliationItem(
        reconciliation_item_id="BRI-DASH",
        reconciliation_run_id=(
            run.reconciliation_run_id
        ),
        account_id=account.account_id,
        category=(
            BrokerReconciliationCategory
            .POSITION
        ),
        comparison_key="AAPL:LONG",
        status=(
            BrokerReconciliationItemStatus
            .MISSING_BROKER
        ),
        internal_reference_ids=(
            "POSITION-1",
        ),
        broker_reference_ids=(),
        differences={},
        message=(
            "Internal position has no "
            "broker-paper position."
        ),
        created_at=T0,
        metadata={},
    )

    repository.complete_run(
        run.reconciliation_run_id,
        items=(item,),
        completed_at=T0,
    )

    snapshot = (
        dashboard_service.build_snapshot(
            account.account_id,
            generated_at=T0,
        )
    )

    assert (
        snapshot
        .broker_reconciliation_run
        .status.value
        == "DIFFERENCES"
    )

    assert len(
        snapshot
        .broker_reconciliation_items
    ) == 1

    provenance = snapshot.provenance_for(
        "broker_reconciliation"
    )

    assert (
        "paper_broker_"
        "reconciliation_runs"
        in provenance.source_tables
    )


def test_dashboard_reconciliation_rows(
    tmp_path,
) -> None:
    (
        _,
        account,
        repository,
        dashboard_service,
    ) = make_environment(tmp_path)

    run = start_run(
        repository,
        account.account_id,
        key="ROW-RUN",
    )

    repository.complete_run(
        run.reconciliation_run_id,
        items=(),
        completed_at=T0,
    )

    snapshot = (
        dashboard_service.build_snapshot(
            account.account_id,
            generated_at=T0,
        )
    )

    summary = (
        broker_reconciliation_summary_rows(
            snapshot
        )
    )

    unresolved = (
        broker_reconciliation_item_rows(
            snapshot
        )
    )

    assert summary[0]["status"] == (
        "MATCHED"
    )

    assert summary[0]["reconciled"] is True
    assert unresolved == ()
