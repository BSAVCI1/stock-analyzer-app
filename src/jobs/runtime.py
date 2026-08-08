"""Environment-driven scheduled-job runtime."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Mapping

from src.automation import (
    AutomatedPaperExecutionEngine,
    AutomationRepository,
)
from src.execution_adapters import (
    ExecutionAdapter,
    InternalPaperExecutionAdapter,
)
from src.notifications import (
    EmailNotificationSender,
    NotificationService,
    TelegramNotificationSender,
    load_email_config,
    load_telegram_config,
)
from src.paper import (
    DEFAULT_DATABASE_PATH,
    NotificationChannel,
    PaperRepository,
    PaperTradingService,
    YahooFXRateProvider,
)
from src.scanner import (
    AutomaticMarketScanner,
    ScannerRepository,
    load_stock_universe,
)

from .calendar import ExchangeCalendar
from .repository import JobRepository
from .service import ScheduledJobService


@dataclass(frozen=True, slots=True)
class RuntimeReleaseGateReport:
    """Minimal scanner-compatible runtime release result."""

    alert_scheduling_eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    database_path: Path
    account_id: str
    universe_path: Path

    release_eligible_strategies: tuple[
        str,
        ...,
    ]

    app_version: str
    threshold_version: str


@dataclass(frozen=True, slots=True)
class PaperJobRuntime:
    settings: RuntimeSettings

    paper_repository: PaperRepository
    scanner_repository: ScannerRepository
    automation_repository: AutomationRepository
    job_repository: JobRepository

    paper_service: PaperTradingService
    execution_adapter: ExecutionAdapter
    scanner: AutomaticMarketScanner
    execution_engine: (
        AutomatedPaperExecutionEngine
    )
    notification_service: (
        NotificationService
    )
    job_service: ScheduledJobService

    notification_channels: tuple[
        NotificationChannel,
        ...,
    ]


def _split_csv(
    value: str | None,
) -> tuple[str, ...]:
    if not value:
        return ()

    return tuple(
        dict.fromkeys(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )
    )


def load_runtime_settings(
    environ: Mapping[
        str,
        str,
    ] | None = None,
    *,
    database_path: str | Path | None = None,
    account_id: str | None = None,
) -> RuntimeSettings:
    values = (
        os.environ
        if environ is None
        else environ
    )

    resolved_database = Path(
        database_path
        or values.get(
            "PAPER_DATABASE_PATH",
            str(DEFAULT_DATABASE_PATH),
        )
    )

    resolved_account = str(
        account_id
        or values.get(
            "PAPER_ACCOUNT_ID",
            "",
        )
    ).strip()

    if not resolved_account:
        raise ValueError(
            "Paper account ID is required. "
            "Set PAPER_ACCOUNT_ID or pass "
            "--account-id."
        )

    universe_path = Path(
        values.get(
            "PAPER_UNIVERSE_PATH",
            "config/stock_universe.json",
        )
    )

    approved_strategies = tuple(
        item.lower()
        for item in _split_csv(
            values.get(
                "PAPER_RELEASE_ELIGIBLE_STRATEGIES"
            )
        )
    )

    return RuntimeSettings(
        database_path=resolved_database,
        account_id=resolved_account,
        universe_path=universe_path,
        release_eligible_strategies=(
            approved_strategies
        ),
        app_version=values.get(
            "PAPER_APP_VERSION",
            "v0.3.4-p3.4",
        ),
        threshold_version=values.get(
            "PAPER_THRESHOLD_VERSION",
            "schema-1",
        ),
    )


def make_release_gate_lookup(
    approved_strategies: tuple[
        str,
        ...,
    ],
):
    approved = {
        strategy.strip().lower()
        for strategy in approved_strategies
        if strategy.strip()
    }

    def lookup(
        strategy: str,
    ) -> RuntimeReleaseGateReport | None:
        normalised = (
            str(strategy)
            .strip()
            .lower()
        )

        if not approved:
            return None

        if normalised in approved:
            return RuntimeReleaseGateReport(
                alert_scheduling_eligible=True,
                reasons=(
                    "Strategy is explicitly "
                    "enabled by the runtime "
                    "release allowlist.",
                ),
            )

        return RuntimeReleaseGateReport(
            alert_scheduling_eligible=False,
            reasons=(
                f"Strategy {strategy!r} is "
                "not present in "
                "PAPER_RELEASE_ELIGIBLE_STRATEGIES.",
            ),
        )

    return lookup


def build_runtime(
    settings: RuntimeSettings,
    *,
    environ: Mapping[
        str,
        str,
    ] | None = None,
) -> PaperJobRuntime:
    values = (
        os.environ
        if environ is None
        else environ
    )

    paper_repository = PaperRepository(
        settings.database_path
    )

    # Fail before running a scheduled cycle when
    # the configured account does not exist.
    paper_repository.get_account(
        settings.account_id
    )

    scanner_repository = ScannerRepository(
        settings.database_path
    )

    automation_repository = (
        AutomationRepository(
            settings.database_path
        )
    )

    job_repository = JobRepository(
        settings.database_path
    )

    fx_rate_provider = (
        YahooFXRateProvider()
    )

    paper_service = PaperTradingService(
        paper_repository,
        fx_rate_provider=(
            fx_rate_provider
        ),
        app_version=settings.app_version,
        threshold_version=(
            settings.threshold_version
        ),
    )

    execution_adapter = (
        InternalPaperExecutionAdapter(
            paper_repository=paper_repository,
            paper_service=paper_service,
        )
    )

    release_gate_lookup = (
        make_release_gate_lookup(
            settings
            .release_eligible_strategies
        )
    )

    scanner = AutomaticMarketScanner(
        scanner_repository=(
            scanner_repository
        ),
        paper_service=paper_service,
        release_gate_lookup=(
            release_gate_lookup
        ),
        app_version=settings.app_version,
    )

    execution_engine = (
        AutomatedPaperExecutionEngine(
            paper_repository=(
                paper_repository
            ),
            paper_service=paper_service,
            execution_adapter=execution_adapter,
            scanner_repository=(
                scanner_repository
            ),
            automation_repository=(
                automation_repository
            ),
            app_version=(
                settings.app_version
            ),
        )
    )

    senders = {}
    channels: list[
        NotificationChannel
    ] = []

    email_config = load_email_config(
        values
    )

    if email_config is not None:
        senders[
            NotificationChannel.EMAIL
        ] = EmailNotificationSender(
            email_config
        )

        channels.append(
            NotificationChannel.EMAIL
        )

    telegram_config = (
        load_telegram_config(values)
    )

    if telegram_config is not None:
        senders[
            NotificationChannel.TELEGRAM
        ] = TelegramNotificationSender(
            telegram_config
        )

        channels.append(
            NotificationChannel.TELEGRAM
        )

    notification_channels = tuple(
        channels
    )

    notification_service = (
        NotificationService(
            paper_repository,
            senders=senders,
        )
    )

    calendar = ExchangeCalendar()

    job_service = ScheduledJobService(
        job_repository=job_repository,
        paper_repository=paper_repository,
        scanner=scanner,
        execution_engine=(
            execution_engine
        ),
        notification_service=(
            notification_service
        ),
        calendar=calendar,
        universe_loader=lambda: (
            load_stock_universe(
                settings.universe_path
            )
        ),
        notification_channels=(
            notification_channels
        ),
    )

    return PaperJobRuntime(
        settings=settings,
        paper_repository=paper_repository,
        scanner_repository=(
            scanner_repository
        ),
        automation_repository=(
            automation_repository
        ),
        job_repository=job_repository,
        paper_service=paper_service,
        execution_adapter=execution_adapter,
        scanner=scanner,
        execution_engine=(
            execution_engine
        ),
        notification_service=(
            notification_service
        ),
        job_service=job_service,
        notification_channels=(
            notification_channels
        ),
    )
