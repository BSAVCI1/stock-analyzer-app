"""Dispatch autonomous invocations into paper services."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.automation import (
    AutomatedPaperExecutionEngine,
)
from src.notifications import NotificationService
from src.paper import (
    NotificationChannel,
    PaperRepository,
)
from src.scanner import (
    AutomaticMarketScanner,
    StockUniverse,
)
from src.strategy import StrategyHorizon

from .schedule import (
    AutonomousJobKind,
    ScheduledInvocation,
)
from .service import ScheduledJobService


UniverseLoader = Callable[[], StockUniverse]


class AutonomousInvocationDispatcher:
    """Concrete paper-only invocation boundary."""

    def __init__(
        self,
        *,
        account_id: str,
        paper_repository: PaperRepository,
        scanner: AutomaticMarketScanner,
        execution_engine:
        AutomatedPaperExecutionEngine,
        notification_service:
        NotificationService,
        job_service: ScheduledJobService,
        universe_loader: UniverseLoader,
        notification_channels: tuple[
            NotificationChannel,
            ...,
        ] = (),
        strategy_versions: Mapping[
            StrategyHorizon,
            str,
        ] | None = None,
    ) -> None:
        value = str(account_id).strip()

        if not value:
            raise ValueError(
                "account_id is required."
            )

        self.account_id = value
        self.paper_repository = (
            paper_repository
        )
        self.scanner = scanner
        self.execution_engine = (
            execution_engine
        )
        self.notification_service = (
            notification_service
        )
        self.job_service = job_service
        self.universe_loader = universe_loader
        self.notification_channels = tuple(
            dict.fromkeys(
                channel
                for channel
                in notification_channels
                if channel
                is not NotificationChannel.INTERNAL
            )
        )
        self.strategy_versions = dict(
            strategy_versions
            or {
                StrategyHorizon.SWING:
                "p4.3-swing-v1",
                StrategyHorizon.MEDIUM_TERM:
                "p4.3-medium-term-v1",
            }
        )

        if set(self.strategy_versions) != {
            StrategyHorizon.SWING,
            StrategyHorizon.MEDIUM_TERM,
        }:
            raise ValueError(
                "strategy_versions must contain "
                "SWING and MEDIUM_TERM."
            )

        self._universe: StockUniverse | None = (
            None
        )

    def _load_universe(self) -> StockUniverse:
        universe = self.universe_loader()

        if not isinstance(
            universe,
            StockUniverse,
        ):
            raise ValueError(
                "universe_loader must return "
                "StockUniverse."
            )

        self._universe = universe
        return universe

    def _run_scan(
        self,
        invocation: ScheduledInvocation,
    ) -> None:
        horizon = invocation.strategy_horizon

        if horizon not in {
            StrategyHorizon.SWING,
            StrategyHorizon.MEDIUM_TERM,
        }:
            raise ValueError(
                "Scan invocation requires an "
                "approved strategy horizon."
            )

        universe = (
            self._universe
            or self._load_universe()
        )
        self.scanner.run_scan(
            account_id=self.account_id,
            universe=universe,
            started_at=(
                invocation.scheduled_for
            ),
            scan_key=(
                invocation.idempotency_key
            ),
            strategy_horizon=horizon,
            strategy_version=(
                self.strategy_versions[horizon]
            ),
        )

    def _run_position_monitor(
        self,
        invocation: ScheduledInvocation,
    ) -> None:
        self.execution_engine.run(
            account_id=self.account_id,
            run_key=(
                invocation.idempotency_key
            ),
            run_at=invocation.scheduled_for,
        )

    def _dispatch_notifications(
        self,
        invocation: ScheduledInvocation,
    ) -> None:
        self.notification_service.fan_out_internal(
            self.account_id,
            channels=(
                self.notification_channels
            ),
            created_at=(
                invocation.scheduled_for
            ),
        )
        self.notification_service.dispatch_pending(
            self.account_id,
            attempted_at=(
                invocation.scheduled_for
            ),
        )

    def _post_close_report(
        self,
        invocation: ScheduledInvocation,
    ) -> None:
        account = (
            self.paper_repository.get_account(
                self.account_id
            )
        )
        reconciliation = (
            self.paper_repository
            .reconcile_account(
                self.account_id
            )
        )

        if not reconciliation.reconciled:
            raise RuntimeError(
                "Post-close reconciliation "
                "failed; the daily report is "
                "blocking."
            )

        positions = (
            self.paper_repository
            .list_open_positions(
                self.account_id
            )
        )
        orders = (
            self.paper_repository
            .list_pending_orders(
                self.account_id
            )
        )
        subject = (
            "Autonomous paper close — "
            f"{invocation.scheduled_for.date()}"
        )
        text = "\n".join(
            (
                "Autonomous paper post-close report",
                (
                    "Cash balance: "
                    f"{account.cash_balance} "
                    f"{account.base_currency}"
                ),
                (
                    "Reserved cash: "
                    f"{account.reserved_cash} "
                    f"{account.base_currency}"
                ),
                (
                    "Open positions: "
                    f"{len(positions)}"
                ),
                (
                    "Pending orders: "
                    f"{len(orders)}"
                ),
                "Account reconciled: True",
            )
        )

        self.paper_repository.queue_notification(
            account_id=self.account_id,
            event_type=(
                "AUTONOMOUS_POST_CLOSE_REPORT"
            ),
            reference_type=(
                "AUTONOMOUS_INVOCATION"
            ),
            reference_id=(
                invocation.idempotency_key
            ),
            channel=(
                NotificationChannel.INTERNAL
            ),
            payload={
                "subject": subject,
                "text": text,
            },
            created_at=(
                invocation.scheduled_for
            ),
        )
        self._dispatch_notifications(
            invocation
        )

    def __call__(
        self,
        invocation: ScheduledInvocation,
    ) -> bool:
        if not isinstance(
            invocation,
            ScheduledInvocation,
        ):
            raise ValueError(
                "invocation must be a "
                "ScheduledInvocation."
            )

        kind = invocation.job_kind

        if (
            kind
            is AutonomousJobKind.UNIVERSE_REFRESH
        ):
            self._load_universe()
        elif kind in {
            AutonomousJobKind.SWING_SCAN,
            AutonomousJobKind.MEDIUM_TERM_SCAN,
        }:
            self._run_scan(invocation)
        elif (
            kind
            is AutonomousJobKind.POSITION_MONITOR
        ):
            self._run_position_monitor(
                invocation
            )
        elif (
            kind
            is AutonomousJobKind
            .NOTIFICATION_DISPATCH
        ):
            self._dispatch_notifications(
                invocation
            )
        elif (
            kind
            is AutonomousJobKind
            .POST_CLOSE_REPORT
        ):
            self._post_close_report(
                invocation
            )
        elif (
            kind
            is AutonomousJobKind.WEEKLY_REPORT
        ):
            report = (
                self.job_service
                .run_weekly_report(
                    account_id=(
                        self.account_id
                    ),
                    scheduled_for=(
                        invocation.scheduled_for
                    ),
                )
            )
            return not report.duplicate
        else:
            raise ValueError(
                "Unsupported autonomous job kind."
            )

        return True
