"""Automated, paper-only portfolio execution and monitoring."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from decimal import (
    Decimal,
    ROUND_FLOOR,
)
from typing import Callable

import pandas as pd

from src.analysis import Signal
from src.backtest import (
    ExecutionStatus,
    OrderRecord,
    SignalRecord,
    apply_entry_slippage,
    apply_exit_slippage,
    calculate_fee,
    execute_next_session,
)
from src.execution_adapters import (
    ExecutionAdapter,
    InternalPaperExecutionAdapter,
)
from src.costs import (
    IBKREconomicDecision,
    IBKRTradeSide,
    calculate_us_long_trade_economics,
    calculate_us_stock_reference_fees,
)

from src.data import (
    MarketSnapshot,
    load_market_snapshot,
)
from src.paper import (
    OrderStatus,
    PaperExitReason,
    PaperPositionRecord,
    PaperRepository,
    PaperTradingService,
    PositionStatus,
    money,
)
from src.paper.sizing import (
    FixedNotionalSizingPolicy,
    FixedNotionalSizingRequest,
    PositionSizingMode,
    calculate_fixed_notional_size,
)
from src.scanner import (
    ScanResultStatus,
    ScannerAnalysisOutcome,
    ScannerRepository,
    run_deterministic_scanner_analysis,
)

from .models import (
    AutomatedExecutionConfig,
    ExecutionRunReport,
    ExecutionRunStatus,
    ExitRequestStatus,
    PortfolioControl,
)
from .repository import AutomationRepository


SnapshotLoader = Callable[
    [str],
    MarketSnapshot,
]

AnalysisRunner = Callable[
    [MarketSnapshot],
    ScannerAnalysisOutcome,
]


class StaleMarketDataError(RuntimeError):
    """Market history is not fresh enough for automation."""


class IBKRCostGateRejected(RuntimeError):
    """Raised when P4.2 rejects a candidate after costs."""



class AutomatedPaperExecutionEngine:
    """Execute and monitor a persistent paper portfolio."""

    def __init__(
        self,
        *,
        paper_repository: PaperRepository,
        paper_service: PaperTradingService,
        execution_adapter: ExecutionAdapter | None = None,
        scanner_repository: ScannerRepository,
        automation_repository: AutomationRepository,
        config: AutomatedExecutionConfig | None = None,
        snapshot_loader: SnapshotLoader | None = None,
        analysis_runner: AnalysisRunner = (
            run_deterministic_scanner_analysis
        ),
        app_version: str = "v0.3.2-p3.2",
    ) -> None:
        self.paper_repository = (
            paper_repository
        )

        self.paper_service = paper_service

        self.execution_adapter = (
            execution_adapter
            or InternalPaperExecutionAdapter(
                paper_repository=paper_repository,
                paper_service=paper_service,
            )
        )

        self.scanner_repository = (
            scanner_repository
        )

        self.automation_repository = (
            automation_repository
        )

        self.config = (
            config
            or AutomatedExecutionConfig()
        )

        self.snapshot_loader = (
            snapshot_loader
            or (
                lambda symbol:
                load_market_snapshot(
                    symbol,
                    min_rows=200,
                )
            )
        )

        self.analysis_runner = (
            analysis_runner
        )

        self.app_version = app_version

    @staticmethod
    def _validate_run_time(
        value: datetime,
    ) -> datetime:
        if (
            value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "run_at must be timezone-aware."
            )

        return value.astimezone(
            timezone.utc
        )

    def _load_market(
        self,
        symbol: str,
        *,
        run_at: datetime,
        control: PortfolioControl,
        cache: dict[
            str,
            tuple[
                MarketSnapshot,
                pd.DataFrame,
            ],
        ],
    ) -> tuple[
        MarketSnapshot,
        pd.DataFrame,
    ]:
        if symbol in cache:
            return cache[symbol]

        snapshot = self.snapshot_loader(
            symbol
        )

        history = snapshot.history.copy()

        if not isinstance(
            history,
            pd.DataFrame,
        ) or history.empty:
            raise ValueError(
                f"{symbol} history is empty."
            )

        required = {
            "Open",
            "High",
            "Low",
            "Close",
        }

        missing = sorted(
            required.difference(
                history.columns
            )
        )

        if missing:
            raise ValueError(
                f"{symbol} history is missing: "
                + ", ".join(missing)
                + "."
            )

        history.index = pd.to_datetime(
            history.index,
            errors="coerce",
            utc=True,
        )

        history = history.loc[
            ~history.index.isna()
        ]

        history = history.loc[
            ~history.index.duplicated(
                keep="last"
            )
        ].sort_index()

        for column in required:
            history[column] = pd.to_numeric(
                history[column],
                errors="coerce",
            )

        history = history.dropna(
            subset=list(required)
        )

        history = history.loc[
            history.index
            <= pd.Timestamp(run_at)
        ]

        if history.empty:
            raise ValueError(
                f"{symbol} has no history "
                "available by the run time."
            )

        invalid_geometry = (
            (history["High"] < history["Low"])
            | (
                history["Open"]
                > history["High"]
            )
            | (
                history["Open"]
                < history["Low"]
            )
            | (
                history["Close"]
                > history["High"]
            )
            | (
                history["Close"]
                < history["Low"]
            )
        )

        if bool(invalid_geometry.any()):
            raise ValueError(
                f"{symbol} contains invalid "
                "OHLC geometry."
            )

        latest_at = (
            history.index[-1]
            .to_pydatetime()
        )

        staleness_days = (
            run_at.date()
            - latest_at.date()
        ).days

        if (
            staleness_days
            > control
            .maximum_stale_market_days
        ):
            raise StaleMarketDataError(
                f"{symbol} market data is "
                f"{staleness_days} days old."
            )

        sliced_snapshot = replace(
            snapshot,
            history=history,
            fetched_at_utc=run_at,
        )

        cache[symbol] = (
            sliced_snapshot,
            history,
        )

        return cache[symbol]

    @staticmethod
    def _fixed_notional_policy(
        *,
        control: PortfolioControl,
        account_currency: str,
    ) -> FixedNotionalSizingPolicy:
        if (
            control.sizing_mode
            is not PositionSizingMode
            .FIXED_NOTIONAL_WITH_RISK_CAP
        ):
            raise ValueError(
                "Portfolio control is not using "
                "fixed-notional sizing."
            )

        required = (
            control.portfolio_currency,
            control.target_order_value,
            control.maximum_order_value,
            control.maximum_planned_loss,
            control.maximum_open_positions,
            control.maximum_invested_exposure,
        )

        if any(
            value is None
            for value in required
        ):
            raise ValueError(
                "Fixed-notional sizing control "
                "is incomplete."
            )

        if (
            control.portfolio_currency
            != account_currency
        ):
            raise ValueError(
                "Sizing portfolio currency "
                "does not match account base "
                "currency."
            )

        return FixedNotionalSizingPolicy(
            portfolio_currency=(
                control.portfolio_currency
            ),
            target_order_value=(
                control.target_order_value
            ),
            maximum_order_value=(
                control.maximum_order_value
            ),
            maximum_planned_loss=(
                control.maximum_planned_loss
            ),
            maximum_open_positions=(
                control.maximum_open_positions
            ),
            maximum_invested_exposure=(
                control
                .maximum_invested_exposure
            ),
        )

    def _committed_exposure_portfolio(
        self,
        *,
        account_id: str,
    ) -> tuple[Decimal, int]:
        account = (
            self.paper_repository
            .get_account(account_id)
        )

        positions = (
            self.paper_repository
            .list_open_positions(account_id)
        )

        pending_orders = (
            self.paper_repository
            .list_pending_orders(account_id)
        )

        exposure = Decimal("0")

        for position in positions:
            quote_currency = (
                position.quote_currency
                or account.base_currency
            )

            portfolio_currency = (
                position.portfolio_currency
                or account.base_currency
            )

            if (
                portfolio_currency
                != account.base_currency
            ):
                raise ValueError(
                    "Position portfolio currency "
                    "does not match account base "
                    "currency."
                )

            if (
                position.entry_fx_rate
                is not None
            ):
                fx_rate = (
                    position.entry_fx_rate
                )

            elif (
                quote_currency
                == account.base_currency
            ):
                fx_rate = Decimal("1")

            else:
                raise ValueError(
                    "Cross-currency position "
                    "is missing entry FX "
                    "provenance."
                )

            exposure = money(
                exposure
                + (
                    position.entry_price
                    * position.quantity
                    * fx_rate
                )
            )

        for order in pending_orders:
            quote_currency = (
                order.quote_currency
                or account.base_currency
            )

            portfolio_currency = (
                order.portfolio_currency
                or account.base_currency
            )

            if (
                portfolio_currency
                != account.base_currency
            ):
                raise ValueError(
                    "Pending-order portfolio "
                    "currency does not match "
                    "account base currency."
                )

            if (
                order.reservation_fx_rate
                is not None
            ):
                fx_rate = (
                    order.reservation_fx_rate
                )

            elif (
                quote_currency
                == account.base_currency
            ):
                fx_rate = Decimal("1")

            else:
                raise ValueError(
                    "Cross-currency pending "
                    "order is missing "
                    "reservation FX provenance."
                )

            exposure = money(
                exposure
                + (
                    order.entry_high
                    * order.quantity
                    * fx_rate
                )
            )

        committed_count = (
            len(positions)
            + len(pending_orders)
        )

        return (
            money(exposure),
            committed_count,
        )

    def _calculate_legacy_quantity(
        self,
        *,
        account_id: str,
        signal_id: str,
    ) -> Decimal:
        account = (
            self.paper_repository
            .get_account(account_id)
        )

        signal = (
            self.paper_repository
            .get_signal(signal_id)
        )

        risk_per_share = money(
            signal.entry_high
            - signal.stop_price
        )

        if risk_per_share <= 0:
            raise ValueError(
                "Signal risk per share "
                "must be positive."
            )

        risk_budget = money(
            account.cash_balance
            * self.paper_service
            .config
            .risk_fraction_per_trade
        )

        allocation_budget = money(
            account.cash_balance
            * self.paper_service
            .config
            .maximum_allocation_fraction
        )

        cash_budget = min(
            account.available_cash,
            allocation_budget,
        )

        risk_quantity = (
            risk_budget
            / risk_per_share
        )

        cash_quantity = (
            cash_budget
            / signal.entry_high
        )

        quantity = min(
            risk_quantity,
            cash_quantity,
        ).to_integral_value(
            rounding=ROUND_FLOOR
        )

        while quantity > 0:
            notional = money(
                signal.entry_high
                * quantity
            )

            fee = self._calculate_lifecycle_fee_quote(
                      quote_currency=(
                          signal.quote_currency
                          or account.base_currency
                      ),
                      quantity=quantity,
                      trade_value_quote=notional,
                      side=IBKRTradeSide.BUY,
                      require_complete=False,
                  )

            if money(
                notional + fee
            ) <= account.available_cash:
                break

            quantity -= 1

        if quantity <= 0:
            raise ValueError(
                "No whole-share quantity fits "
                "the configured capital and "
                "risk constraints."
            )

        return money(quantity)

    def _calculate_quantity(
        self,
        *,
        account_id: str,
        signal_id: str,
        control: PortfolioControl | None = None,
        run_at: datetime | None = None,
    ) -> Decimal:
        if (
            control is None
            or control.sizing_mode is None
        ):
            return (
                self._calculate_legacy_quantity(
                    account_id=account_id,
                    signal_id=signal_id,
                )
            )

        if (
            control.sizing_mode
            is not PositionSizingMode
            .FIXED_NOTIONAL_WITH_RISK_CAP
        ):
            raise ValueError(
                "Unsupported portfolio sizing "
                f"mode: {control.sizing_mode}."
            )

        if run_at is None:
            raise ValueError(
                "run_at is required for "
                "FX-aware fixed-notional "
                "sizing."
            )

        account = (
            self.paper_repository
            .get_account(account_id)
        )

        signal = (
            self.paper_repository
            .get_signal(signal_id)
        )

        policy = self._fixed_notional_policy(
            control=control,
            account_currency=(
                account.base_currency
            ),
        )

        quote_currency = (
            signal.quote_currency
            or account.base_currency
        )

        fx_rate = (
            self.paper_service
            ._resolve_fx_rate(
                quote_currency=quote_currency,
                portfolio_currency=(
                    account.base_currency
                ),
                as_of=run_at,
            )
        )

        (
            invested_exposure,
            committed_position_count,
        ) = (
            self
            ._committed_exposure_portfolio(
                account_id=account_id,
            )
        )

        fee_reference_portfolio = min(
            policy.target_order_value,
            policy.maximum_order_value,
        )

        fee_reference_quote = money(
            fee_reference_portfolio
            / fx_rate.rate
        )

        entry_fee_quote = self._calculate_lifecycle_fee_quote(
                              quote_currency=quote_currency,
                              quantity=max(
                                  Decimal("1"),
                                  (
                                      fee_reference_quote
                                      // signal.entry_high
                                  ),
                              ),
                              trade_value_quote=(
                                  fee_reference_quote
                              ),
                              side=IBKRTradeSide.BUY,
                              require_complete=False,
                          )

        exit_fee_quote = self._calculate_lifecycle_fee_quote(
                             quote_currency=quote_currency,
                             quantity=max(
                                 Decimal("1"),
                                 (
                                     fee_reference_quote
                                     // signal.entry_high
                                 ),
                             ),
                             trade_value_quote=(
                                 fee_reference_quote
                             ),
                             side=IBKRTradeSide.SELL,
                             require_complete=False,
                         )

        entry_fee_portfolio = (
            fx_rate
            .convert_quote_to_portfolio(
                entry_fee_quote
            )
        )

        exit_fee_portfolio = (
            fx_rate
            .convert_quote_to_portfolio(
                exit_fee_quote
            )
        )

        decision = (
            calculate_fixed_notional_size(
                FixedNotionalSizingRequest(
                    quote_currency=(
                        quote_currency
                    ),
                    entry_price_quote=(
                        signal.entry_high
                    ),
                    stop_price_quote=(
                        signal.stop_price
                    ),
                    quote_to_portfolio_rate=(
                        fx_rate.rate
                    ),
                    available_cash_portfolio=(
                        account.available_cash
                    ),
                    invested_exposure_portfolio=(
                        invested_exposure
                    ),
                    current_position_count=(
                        committed_position_count
                    ),
                    estimated_entry_fee_portfolio=(
                        entry_fee_portfolio
                    ),
                    estimated_exit_fee_portfolio=(
                        exit_fee_portfolio
                    ),
                    quantity_step=Decimal("1"),
                ),
                policy=policy,
            )
        )

        return decision.quantity

    @staticmethod
    def _execution_records(
        *,
        signal,
        order,
    ) -> tuple[
        SignalRecord,
        OrderRecord,
    ]:
        signal_record = SignalRecord(
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            strategy=signal.strategy,
            signal=Signal.BUY,
            generated_at=(
                signal.generated_at
            ),
            expires_at=signal.expires_at,
            score=signal.score,
            confidence=signal.confidence,
        )

        order_record = OrderRecord(
            order_id=order.order_id,
            signal_id=order.signal_id,
            symbol=order.symbol,
            side=order.side,
            created_at=order.created_at,
            expires_at=order.expires_at,
            entry_low=float(
                order.entry_low
            ),
            entry_high=float(
                order.entry_high
            ),
            stop_price=float(
                order.stop_price
            ),
            targets=tuple(
                float(value)
                for value in order.targets
            ),
            quantity=float(
                order.quantity
            ),
            paper_only=True,
        )

        return (
            signal_record,
            order_record,
        )

    def _close_position(
        self,
        *,
        position: PaperPositionRecord,
        raw_exit_price: object,
        reason: PaperExitReason,
        closed_at: datetime,
    ) -> None:
        adjusted_price = (
            apply_exit_slippage(
                raw_exit_price,
                position.side,
                self.config.costs,
            )
        )

        raw_price = money(
            raw_exit_price
        )

        slippage = money(
            abs(
                adjusted_price
                - raw_price
            )
            * position.quantity
        )

        fee = self._calculate_lifecycle_fee_quote(
                  quote_currency=(
                      position.quote_currency
                  ),
                  quantity=position.quantity,
                  trade_value_quote=money(
                      adjusted_price
                      * position.quantity
                  ),
                  side=IBKRTradeSide.SELL,
              )

        self.execution_adapter.close_position(
            position_id=(
                position.position_id
            ),
            exit_price=adjusted_price,
            exit_reason=reason,
            exit_fees=fee,
            exit_slippage=slippage,
            closed_at=closed_at,
        )

    def _monitor_price_exits(
        self,
        *,
        account_id: str,
        run_at: datetime,
        control: PortfolioControl,
        cache,
    ) -> tuple[int, int]:
        closed_count = 0
        error_count = 0

        positions = (
            self.paper_repository
            .list_open_positions(account_id)
        )

        for position in positions:
            try:
                _, history = self._load_market(
                    position.symbol,
                    run_at=run_at,
                    control=control,
                    cache=cache,
                )

                eligible = history.loc[
                    history.index
                    > pd.Timestamp(
                        position.opened_at
                    )
                ]

                for timestamp, row in (
                    eligible.iterrows()
                ):
                    bar_at = (
                        timestamp.to_pydatetime()
                    )

                    open_price = money(
                        row["Open"]
                    )

                    high_price = money(
                        row["High"]
                    )

                    low_price = money(
                        row["Low"]
                    )

                    raw_exit = None
                    reason = None

                    # Conservative ordering:
                    # stop wins if stop and target
                    # occur in the same bar.
                    if (
                        open_price
                        <= position.stop_price
                    ):
                        raw_exit = open_price
                        reason = (
                            PaperExitReason
                            .STOP_LOSS
                        )
                    elif (
                        low_price
                        <= position.stop_price
                    ):
                        raw_exit = (
                            position.stop_price
                        )
                        reason = (
                            PaperExitReason
                            .STOP_LOSS
                        )
                    else:
                        primary_target = (
                            position.targets[0]
                        )

                        if (
                            open_price
                            >= primary_target
                        ):
                            raw_exit = open_price
                            reason = (
                                PaperExitReason
                                .TARGET
                            )
                        elif (
                            high_price
                            >= primary_target
                        ):
                            raw_exit = (
                                primary_target
                            )
                            reason = (
                                PaperExitReason
                                .TARGET
                            )

                    if (
                        raw_exit is None
                        and bar_at
                        >= position.expires_at
                    ):
                        raw_exit = open_price
                        reason = (
                            PaperExitReason
                            .TIME_EXIT
                        )

                    if raw_exit is None:
                        continue

                    self._close_position(
                        position=position,
                        raw_exit_price=raw_exit,
                        reason=reason,
                        closed_at=bar_at,
                    )

                    closed_count += 1
                    break

            except Exception as exc:
                error_count += 1

                self.paper_repository.record_system_event(
                    account_id=account_id,
                    event_type=(
                        "POSITION_MONITOR_ERROR"
                    ),
                    severity="ERROR",
                    reference_type="POSITION",
                    reference_id=(
                        position.position_id
                    ),
                    message=(
                        f"{position.symbol} position "
                        "monitoring failed."
                    ),
                    metadata={
                        "error_type":
                        type(exc).__name__,
                        "error_message":
                        str(exc),
                    },
                    created_at=run_at,
                )

        return closed_count, error_count

    def _execute_exit_requests(
        self,
        *,
        account_id: str,
        run_at: datetime,
        control: PortfolioControl,
        cache,
    ) -> tuple[int, int]:
        closed_count = 0
        error_count = 0

        requests = (
            self.automation_repository
            .list_pending_exit_requests(
                account_id
            )
        )

        for request in requests:
            try:
                position = (
                    self.paper_repository
                    .get_position(
                        request.position_id
                    )
                )

                if (
                    position.status
                    is not PositionStatus.OPEN
                ):
                    self.automation_repository.update_exit_request(
                        request.request_id,
                        status=(
                            ExitRequestStatus
                            .CANCELLED
                        ),
                        executed_at=run_at,
                        error_message=(
                            "Position was already "
                            "closed."
                        ),
                    )

                    continue

                _, history = self._load_market(
                    position.symbol,
                    run_at=run_at,
                    control=control,
                    cache=cache,
                )

                eligible = history.loc[
                    history.index
                    > pd.Timestamp(
                        request.triggered_at
                    )
                ]

                if eligible.empty:
                    continue

                timestamp = eligible.index[0]
                raw_exit = money(
                    eligible.iloc[0]["Open"]
                )

                self._close_position(
                    position=position,
                    raw_exit_price=raw_exit,
                    reason=request.reason,
                    closed_at=(
                        timestamp.to_pydatetime()
                    ),
                )

                self.automation_repository.update_exit_request(
                    request.request_id,
                    status=(
                        ExitRequestStatus
                        .EXECUTED
                    ),
                    executed_at=(
                        timestamp.to_pydatetime()
                    ),
                )

                closed_count += 1

            except Exception as exc:
                error_count += 1

                self.automation_repository.update_exit_request(
                    request.request_id,
                    status=(
                        ExitRequestStatus
                        .PENDING
                    ),
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                )

        return closed_count, error_count

    def _detect_signal_reversals(
        self,
        *,
        account_id: str,
        run_at: datetime,
        control: PortfolioControl,
        cache,
    ) -> int:
        if not self.config.enable_signal_reversal:
            return 0

        created_count = 0

        positions = (
            self.paper_repository
            .list_open_positions(account_id)
        )

        for position in positions:
            try:
                snapshot, _ = self._load_market(
                    position.symbol,
                    run_at=run_at,
                    control=control,
                    cache=cache,
                )

                outcome = self.analysis_runner(
                    snapshot
                )

                if (
                    outcome.recommendation
                    not in {
                        Signal.REDUCE,
                        Signal.SELL,
                    }
                ):
                    continue

                if (
                    outcome.generated_at
                    <= position.opened_at
                ):
                    continue

                _, created = (
                    self.automation_repository
                    .create_exit_request(
                        account_id=(
                            account_id
                        ),
                        position_id=(
                            position
                            .position_id
                        ),
                        reason=(
                            PaperExitReason
                            .SIGNAL_REVERSAL
                        ),
                        triggered_at=(
                            outcome.generated_at
                        ),
                        created_at=run_at,
                    )
                )

                if created:
                    created_count += 1

                    self.paper_repository.record_system_event(
                        account_id=account_id,
                        event_type=(
                            "SIGNAL_REVERSAL_EXIT_REQUESTED"
                        ),
                        reference_type=(
                            "POSITION"
                        ),
                        reference_id=(
                            position
                            .position_id
                        ),
                        message=(
                            f"{position.symbol} "
                            "signal-reversal exit "
                            "requested."
                        ),
                        metadata={
                            "recommendation":
                            outcome
                            .recommendation
                            .value,
                            "strategy":
                            outcome.strategy,
                            "score":
                            outcome.score,
                            "confidence":
                            outcome.confidence,
                        },
                        created_at=run_at,
                    )

            except Exception as exc:
                self.paper_repository.record_system_event(
                    account_id=account_id,
                    event_type=(
                        "SIGNAL_REVERSAL_CHECK_ERROR"
                    ),
                    severity="ERROR",
                    reference_type="POSITION",
                    reference_id=(
                        position.position_id
                    ),
                    message=(
                        f"{position.symbol} signal "
                        "reversal check failed."
                    ),
                    metadata={
                        "error_type":
                        type(exc).__name__,
                        "error_message":
                        str(exc),
                    },
                    created_at=run_at,
                )

        return created_count

    def _portfolio_market_value(
        self,
        *,
        account_id: str,
        run_at: datetime,
        control: PortfolioControl,
        cache,
    ) -> Decimal:
        account = (
            self.paper_repository
            .get_account(account_id)
        )

        value = Decimal("0")

        for position in (
            self.paper_repository
            .list_open_positions(account_id)
        ):
            _, history = self._load_market(
                position.symbol,
                run_at=run_at,
                control=control,
                cache=cache,
            )

            latest_close = money(
                history["Close"].iloc[-1]
            )

            quote_currency = (
                position.quote_currency
                or account.base_currency
            )

            portfolio_currency = (
                position.portfolio_currency
                or account.base_currency
            )

            if (
                portfolio_currency
                != account.base_currency
            ):
                raise ValueError(
                    "Position portfolio currency "
                    "does not match account base "
                    "currency."
                )

            fx_rate = (
                self.paper_service
                ._resolve_fx_rate(
                    quote_currency=(
                        quote_currency
                    ),
                    portfolio_currency=(
                        account.base_currency
                    ),
                    as_of=run_at,
                )
            )

            market_value_quote = money(
                latest_close
                * position.quantity
            )

            value = money(
                value
                + fx_rate
                .convert_quote_to_portfolio(
                    market_value_quote
                )
            )

        return money(value)

    def _entry_block_reasons(
        self,
        *,
        account_id: str,
        control: PortfolioControl,
        current_equity: Decimal,
        run_at: datetime,
    ) -> tuple[str, ...]:
        reasons: list[str] = []

        account = (
            self.paper_repository
            .get_account(account_id)
        )

        if control.kill_switch_active:
            reasons.append(
                control.kill_switch_reason
                or "Portfolio kill switch is active."
            )

        trades = (
            self.paper_repository
            .list_closed_trades(account_id)
        )

        daily_net_pnl = money(
            sum(
                (
                    trade.net_pnl
                    for trade in trades
                    if (
                        trade.exit_time
                        .astimezone(timezone.utc)
                        .date()
                        == run_at.date()
                    )
                ),
                Decimal("0"),
            )
        )

        daily_loss_limit = money(
            account.starting_balance
            * control
            .maximum_daily_loss_fraction
        )

        if daily_net_pnl <= -daily_loss_limit:
            reasons.append(
                "Daily realised-loss circuit "
                "breaker is active."
            )

        stored_peak = (
            self.automation_repository
            .peak_equity(account_id)
        )

        peak_equity = max(
            account.starting_balance,
            current_equity,
            (
                stored_peak
                if stored_peak is not None
                else account.starting_balance
            ),
        )

        drawdown = (
            money(
                peak_equity
                - current_equity
            )
            / peak_equity
            if peak_equity > 0
            else Decimal("1")
        )

        if (
            drawdown
            >= control
            .maximum_drawdown_fraction
        ):
            reasons.append(
                "Portfolio drawdown circuit "
                "breaker is active."
            )

        return tuple(reasons)

    def _cancel_pending_entries(
        self,
        *,
        account_id: str,
        run_at: datetime,
        reasons: tuple[str, ...],
    ) -> int:
        count = 0

        reason = (
            "New entries disabled: "
            + "; ".join(reasons)
        )

        for order in (
            self.paper_repository
            .list_pending_orders(account_id)
        ):
            self.execution_adapter.cancel_order(
                order_id=order.order_id,
                reason=reason,
                cancelled_at=run_at,
            )

            count += 1

        return count

    def _process_pending_entries(
        self,
        *,
        account_id: str,
        run_at: datetime,
        control: PortfolioControl,
        cache,
    ) -> tuple[int, int, int]:
        filled_count = 0
        expired_count = 0
        error_count = 0

        orders = (
            self.paper_repository
            .list_pending_orders(account_id)
        )

        for order in orders:
            try:
                signal = (
                    self.paper_repository
                    .get_signal(
                        order.signal_id
                    )
                )

                _, history = self._load_market(
                    order.symbol,
                    run_at=run_at,
                    control=control,
                    cache=cache,
                )

                signal_record, order_record = (
                    self._execution_records(
                        signal=signal,
                        order=order,
                    )
                )

                result = execute_next_session(
                    signal_record,
                    order_record,
                    history,
                    fill_rule=(
                        self.config.fill_rule
                    ),
                )

                if (
                    result.status
                    is ExecutionStatus.FILLED
                ):
                    fill = result.lifecycle.fill

                    adjusted_price = (
                        apply_entry_slippage(
                            fill.fill_price,
                            order.side,
                            self.config.costs,
                        )
                    )

                    raw_price = money(
                        fill.fill_price
                    )

                    slippage = money(
                        abs(
                            adjusted_price
                            - raw_price
                        )
                        * order.quantity
                    )

                    fee = self._calculate_lifecycle_fee_quote(
                              quote_currency=(
                                  order.quote_currency
                              ),
                              quantity=order.quantity,
                              trade_value_quote=money(
                                  adjusted_price
                                  * order.quantity
                              ),
                              side=IBKRTradeSide.BUY,
                          )

                    self.execution_adapter.record_buy_fill(
                        order_id=order.order_id,
                        fill_price=(
                            adjusted_price
                        ),
                        fees=fee,
                        slippage=slippage,
                        filled_at=(
                            fill.filled_at
                        ),
                    )

                    filled_count += 1

                elif (
                    result.status
                    is ExecutionStatus.EXPIRED
                ):
                    self.execution_adapter.expire_order(
                        order.order_id,
                        expired_at=(
                            result.decision_at
                            or run_at
                        ),
                        reason=result.reason,
                    )

                    expired_count += 1

                elif (
                    result.status
                    is ExecutionStatus.NOT_FILLED
                    and self.config.fill_rule
                    .value == "NEXT_OPEN"
                ):
                    self.execution_adapter.cancel_order(
                        order_id=order.order_id,
                        reason=result.reason,
                        cancelled_at=(
                            result.decision_at
                            or run_at
                        ),
                    )

            except Exception as exc:
                error_count += 1

                self.paper_repository.record_system_event(
                    account_id=account_id,
                    event_type=(
                        "ENTRY_EXECUTION_ERROR"
                    ),
                    severity="ERROR",
                    reference_type="ORDER",
                    reference_id=(
                        order.order_id
                    ),
                    message=(
                        f"{order.symbol} paper "
                        "entry processing failed."
                    ),
                    metadata={
                        "error_type":
                        type(exc).__name__,
                        "error_message":
                        str(exc),
                    },
                    created_at=run_at,
                )

        return (
            filled_count,
            expired_count,
            error_count,
        )

    def _calculate_lifecycle_fee_quote(
        self,
        *,
        quote_currency: str | None,
        quantity: object,
        trade_value_quote: object,
        side: IBKRTradeSide | str,
        require_complete: bool = True,
    ) -> Decimal:
        """Return the authoritative paper-lifecycle fee."""

        trade_value = money(
            trade_value_quote
        )

        if (
            self.config.ibkr_pricing_plan
            is None
        ):
            return calculate_fee(
                trade_value,
                self.config.costs,
            )

        currency = str(
            quote_currency or ""
        ).strip().upper()

        if currency != "USD":
            raise RuntimeError(
                "INCOMPLETE_COST_ESTIMATE: "
                "P4.2 authoritative lifecycle "
                "fees currently support only "
                "USD-quoted securities."
            )

        estimate = (
            calculate_us_stock_reference_fees(
                quantity=quantity,
                trade_value_usd=trade_value,
                pricing_plan=(
                    self.config
                    .ibkr_pricing_plan
                ),
                side=side,
                fractional=False,
            )
        )

        if (
            not estimate.complete
            and require_complete
        ):
            raise RuntimeError(
                "INCOMPLETE_COST_ESTIMATE: "
                "IBKR lifecycle cost estimate "
                "is incomplete."
            )

        return money(
            estimate.total_known_cost
        )

    def _apply_ibkr_cost_gate(
        self,
        *,
        account_id: str,
        signal_id: str,
        quantity: Decimal,
        run_at: datetime,
    ) -> None:
        """Reject a candidate when configured IBKR economics fail."""

        if not self.config.ibkr_cost_gate_enabled:
            return

        pricing_plan = (
            self.config.ibkr_pricing_plan
        )

        if pricing_plan is None:
            raise IBKRCostGateRejected(
                "INCOMPLETE_COST_ESTIMATE: "
                "IBKR pricing plan is not configured."
            )

        account = (
            self.paper_repository
            .get_account(account_id)
        )

        signal = (
            self.paper_repository
            .get_signal(signal_id)
        )

        quote_currency = (
            signal.quote_currency
            or account.base_currency
        ).strip().upper()

        if quote_currency != "USD":
            raise IBKRCostGateRejected(
                "INCOMPLETE_COST_ESTIMATE: "
                "P4.2 engine gate currently "
                "supports USD-quoted instruments only; "
                f"received {quote_currency}."
            )

        if not signal.targets:
            raise IBKRCostGateRejected(
                "INCOMPLETE_COST_ESTIMATE: "
                "signal has no primary target."
            )

        fx_rate = (
            self.paper_service
            ._resolve_fx_rate(
                quote_currency="USD",
                portfolio_currency=(
                    account.base_currency
                ),
                as_of=run_at,
            )
        )

        economics = (
            calculate_us_long_trade_economics(
                quantity=quantity,
                entry_price_usd=(
                    signal.entry_high
                ),
                stop_price_usd=(
                    signal.stop_price
                ),
                target_price_usd=(
                    signal.targets[0]
                ),
                usd_to_portfolio_rate=(
                    fx_rate.rate
                ),
                pricing_plan=pricing_plan,
                minimum_net_reward_to_risk=(
                    self.paper_service
                    .config
                    .minimum_reward_to_risk
                ),
                fractional=False,
                fx_mode=(
                    self.config
                    .ibkr_fx_mode
                ),
                include_entry_fx_conversion=(
                    self.config
                    .ibkr_include_entry_fx_conversion
                ),
                include_exit_fx_conversion=(
                    self.config
                    .ibkr_include_exit_fx_conversion
                ),
            )
        )

        if (
            economics.decision
            is IBKREconomicDecision.ACCEPT
        ):
            return

        raise IBKRCostGateRejected(
            f"{economics.decision.value}: "
            f"gross_rr="
            f"{economics.gross_reward_to_risk}; "
            f"net_rr="
            f"{economics.net_reward_to_risk}; "
            f"minimum_rr="
            f"{economics.minimum_net_reward_to_risk}; "
            f"complete={economics.complete}; "
            f"pricing_plan="
            f"{economics.pricing_plan.value}; "
            f"entry_notional_usd="
            f"{economics.entry_notional_usd}; "
            f"reward_path_cost="
            f"{economics.reward_path_cost_portfolio}; "
            f"risk_path_cost="
            f"{economics.risk_path_cost_portfolio}."
        )

    def _create_candidate_orders(
        self,
        *,
        account_id: str,
        scan_id: str,
        run_at: datetime,
        control: PortfolioControl,
    ) -> tuple[int, int]:
        report = (
            self.scanner_repository
            .get_report(scan_id)
        )

        if report.scan.account_id != account_id:
            raise ValueError(
                "Scan belongs to another account."
            )

        created_count = 0
        rejected_count = 0

        candidates = sorted(
            (
                result
                for result in report.results
                if (
                    result.status
                    is ScanResultStatus
                    .ORDER_CANDIDATE
                    and result.release_eligible
                    and result.signal_id
                    is not None
                )
            ),
            key=lambda result: (
                result.rank_position
                if result.rank_position
                is not None
                else 10**9,
                result.symbol,
            ),
        )

        for candidate in candidates:
            if (
                created_count
                >= control
                .maximum_new_orders_per_run
            ):
                break

            try:
                if candidate.data_as_of is None:
                    raise ValueError(
                        "Candidate has no "
                        "market-data timestamp."
                    )

                data_age = (
                    run_at.date()
                    - candidate.data_as_of
                    .astimezone(timezone.utc)
                    .date()
                ).days

                if (
                    data_age
                    > control
                    .maximum_stale_market_days
                ):
                    raise StaleMarketDataError(
                        f"Candidate data is "
                        f"{data_age} days old."
                    )

                quantity = (
                    self._calculate_quantity(
                        account_id=account_id,
                        signal_id=(
                            candidate.signal_id
                        ),
                        control=control,
                        run_at=run_at,
                    )
                )

                signal = (
                    self.paper_repository
                    .get_signal(
                        candidate.signal_id
                    )
                )

                self._apply_ibkr_cost_gate(
                    account_id=account_id,
                    signal_id=(
                        candidate.signal_id
                    ),
                    quantity=quantity,
                    run_at=run_at,
                )

                fee = self._calculate_lifecycle_fee_quote(
                          quote_currency=(
                              signal.quote_currency
                          ),
                          quantity=quantity,
                          trade_value_quote=money(
                              signal.entry_high
                              * quantity
                          ),
                          side=IBKRTradeSide.BUY,
                      )

                _, created = (
                    self.paper_service
                    .create_automatic_buy(
                        account_id=account_id,
                        signal_id=(
                            candidate.signal_id
                        ),
                        quantity=quantity,
                        idempotency_key=(
                            "AUTO-ENTRY:"
                            f"{candidate.signal_id}"
                        ),
                        estimated_fees=fee,
                        created_at=run_at,
                    )
                )

                if created:
                    created_count += 1

            except Exception as exc:
                rejected_count += 1

                self.paper_repository.record_system_event(
                    account_id=account_id,
                    event_type=(
                        "AUTOMATIC_ENTRY_REJECTED"
                    ),
                    severity="WARNING",
                    reference_type=(
                        "SCAN_RESULT"
                    ),
                    reference_id=(
                        candidate.result_id
                    ),
                    message=(
                        f"{candidate.symbol} "
                        "automatic entry was "
                        "rejected."
                    ),
                    metadata={
                        "error_type":
                        type(exc).__name__,
                        "reason": str(exc),
                        "signal_id":
                        candidate.signal_id,
                        "rank_position":
                        candidate.rank_position,
                    },
                    created_at=run_at,
                )

        self.automation_repository.refresh_scan_order_count(
            scan_id
        )

        return (
            created_count,
            rejected_count,
        )

    def run(
        self,
        *,
        account_id: str,
        run_key: str,
        scan_id: str | None = None,
        run_at: datetime | None = None,
    ) -> ExecutionRunReport:
        at = self._validate_run_time(
            run_at
            or datetime.now(timezone.utc)
        )

        configuration = {
            "fill_rule":
            self.config.fill_rule.value,
            "enable_signal_reversal":
            self.config.enable_signal_reversal,
            "costs": {
                key: str(value)
                for key, value
                in asdict(
                    self.config.costs
                ).items()
            },
        }

        run, created = (
            self.automation_repository
            .start_run(
                account_id=account_id,
                run_key=run_key,
                scan_id=scan_id,
                configuration=configuration,
                app_version=(
                    self.app_version
                ),
                started_at=at,
            )
        )

        if (
            not created
            and run.status
            is not ExecutionRunStatus.RUNNING
        ):
            reconciliation = (
                self.paper_repository
                .reconcile_account(
                    account_id
                )
            )

            equity = (
                self.automation_repository
                .get_equity_for_run(
                    run.run_id,
                    account_id,
                )
            )

            return ExecutionRunReport(
                run=run,
                entries_enabled=(
                    not bool(
                        run
                        .entry_block_reasons
                    )
                ),
                entry_block_reasons=(
                    run
                    .entry_block_reasons
                ),
                reconciliation=(
                    reconciliation
                ),
                equity_snapshot=equity,
            )

        counters = {
            "created_orders": 0,
            "filled_orders": 0,
            "expired_orders": 0,
            "cancelled_orders": 0,
            "closed_positions": 0,
            "rejected_entries": 0,
            "error_count": 0,
        }

        cache = {}

        try:
            reconciliation = (
                self.paper_repository
                .reconcile_account(
                    account_id
                )
            )

            if not reconciliation.reconciled:
                raise RuntimeError(
                    "Account did not reconcile "
                    "before execution."
                )

            control = (
                self.automation_repository
                .get_control(
                    account_id,
                    at=at,
                )
            )

            closed, errors = (
                self._monitor_price_exits(
                    account_id=account_id,
                    run_at=at,
                    control=control,
                    cache=cache,
                )
            )

            counters[
                "closed_positions"
            ] += closed

            counters[
                "error_count"
            ] += errors

            closed, errors = (
                self._execute_exit_requests(
                    account_id=account_id,
                    run_at=at,
                    control=control,
                    cache=cache,
                )
            )

            counters[
                "closed_positions"
            ] += closed

            counters[
                "error_count"
            ] += errors

            self._detect_signal_reversals(
                account_id=account_id,
                run_at=at,
                control=control,
                cache=cache,
            )

            valuation_error = None

            try:
                market_value = (
                    self._portfolio_market_value(
                        account_id=(
                            account_id
                        ),
                        run_at=at,
                        control=control,
                        cache=cache,
                    )
                )
            except Exception as exc:
                market_value = Decimal("0")
                valuation_error = (
                    "Portfolio valuation failed: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            account = (
                self.paper_repository
                .get_account(account_id)
            )

            current_equity = money(
                account.cash_balance
                + market_value
            )

            block_reasons = list(
                self._entry_block_reasons(
                    account_id=account_id,
                    control=control,
                    current_equity=(
                        current_equity
                    ),
                    run_at=at,
                )
            )

            if valuation_error:
                block_reasons.append(
                    valuation_error
                )

            entry_block_reasons = tuple(
                block_reasons
            )

            entries_enabled = not bool(
                entry_block_reasons
            )

            if not entries_enabled:
                counters[
                    "cancelled_orders"
                ] += (
                    self._cancel_pending_entries(
                        account_id=account_id,
                        run_at=at,
                        reasons=(
                            entry_block_reasons
                        ),
                    )
                )
            else:
                (
                    filled,
                    expired,
                    errors,
                ) = (
                    self._process_pending_entries(
                        account_id=account_id,
                        run_at=at,
                        control=control,
                        cache=cache,
                    )
                )

                counters[
                    "filled_orders"
                ] += filled

                counters[
                    "expired_orders"
                ] += expired

                counters[
                    "error_count"
                ] += errors

                if scan_id is not None:
                    (
                        created_orders,
                        rejected_entries,
                    ) = (
                        self._create_candidate_orders(
                            account_id=(
                                account_id
                            ),
                            scan_id=scan_id,
                            run_at=at,
                            control=control,
                        )
                    )

                    counters[
                        "created_orders"
                    ] += created_orders

                    counters[
                        "rejected_entries"
                    ] += rejected_entries

            account = (
                self.paper_repository
                .get_account(account_id)
            )

            try:
                market_value = (
                    self._portfolio_market_value(
                        account_id=(
                            account_id
                        ),
                        run_at=at,
                        control=control,
                        cache=cache,
                    )
                )
            except Exception:
                market_value = Decimal("0")

            equity_snapshot = (
                self.automation_repository
                .save_equity_snapshot(
                    run_id=run.run_id,
                    account_id=account_id,
                    captured_at=at,
                    cash_balance=(
                        account.cash_balance
                    ),
                    reserved_cash=(
                        account.reserved_cash
                    ),
                    market_value=(
                        market_value
                    ),
                )
            )

            reconciliation = (
                self.paper_repository
                .reconcile_account(
                    account_id
                )
            )

            if not reconciliation.reconciled:
                raise RuntimeError(
                    "Account did not reconcile "
                    "after execution."
                )

            status = (
                ExecutionRunStatus
                .COMPLETED_WITH_ERRORS
                if counters["error_count"]
                else ExecutionRunStatus
                .COMPLETED
            )

            completed_run = (
                self.automation_repository
                .complete_run(
                    run.run_id,
                    status=status,
                    completed_at=at,
                    entry_block_reasons=(
                        entry_block_reasons
                    ),
                    **counters,
                )
            )

            return ExecutionRunReport(
                run=completed_run,
                entries_enabled=(
                    entries_enabled
                ),
                entry_block_reasons=(
                    entry_block_reasons
                ),
                reconciliation=(
                    reconciliation
                ),
                equity_snapshot=(
                    equity_snapshot
                ),
            )

        except Exception as exc:
            failed_run = (
                self.automation_repository
                .complete_run(
                    run.run_id,
                    status=(
                        ExecutionRunStatus
                        .FAILED
                    ),
                    completed_at=at,
                    entry_block_reasons=(),
                    error_message=(
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                    **counters,
                )
            )

            self.paper_repository.record_system_event(
                account_id=account_id,
                event_type=(
                    "AUTOMATED_EXECUTION_FAILED"
                ),
                severity="ERROR",
                reference_type="EXECUTION_RUN",
                reference_id=run.run_id,
                message=(
                    "Automated paper execution "
                    "run failed."
                ),
                metadata={
                    "error_type":
                    type(exc).__name__,
                    "error_message":
                    str(exc),
                },
                created_at=at,
            )

            raise RuntimeError(
                failed_run.error_message
                or "Execution run failed."
            ) from exc
