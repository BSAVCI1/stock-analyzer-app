"""Deterministic backtest performance metrics and benchmark comparison.

Metrics are calculated from validated closed trades, exact trade settlements
and the same market-history window used by the strategy.

The module contains no Streamlit, provider, broker or live-execution code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from math import isfinite, sqrt
from typing import Iterable, Sequence

import pandas as pd

from .economics import TradeSettlement
from .model import ClosedTradeRecord


_MONEY_QUANTUM = Decimal("0.00000001")
_CALENDAR_DAYS_PER_YEAR = 365.0
_TRADING_PERIODS_PER_YEAR = 252


def _decimal(name: str, value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")

    try:
        result = (
            value
            if isinstance(value, Decimal)
            else Decimal(str(value))
        )
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number."
        ) from exc

    if not result.is_finite():
        raise ValueError(
            f"{name} must be a finite number."
        )

    return result


def _positive_decimal(name: str, value: object) -> Decimal:
    result = _decimal(name, value)

    if result <= 0:
        raise ValueError(
            f"{name} must be greater than zero."
        )

    return result


def _money(value: object) -> Decimal:
    return _decimal("money", value).quantize(
        _MONEY_QUANTUM,
        rounding=ROUND_HALF_EVEN,
    )


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite.") from exc

    if not isfinite(result):
        raise ValueError(f"{name} must be finite.")

    return result


def _aware_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{name} must be timezone-aware."
        )

    return value


def _utc_timestamp(value: datetime) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        raise ValueError(
            "Performance timestamps must be timezone-aware."
        )

    return timestamp.tz_convert("UTC")


def _annualised_return(
    starting_balance: Decimal,
    ending_balance: Decimal,
    elapsed_days: float,
) -> float:
    if elapsed_days <= 0:
        raise ValueError(
            "Performance period must be longer than zero."
        )

    if starting_balance <= 0 or ending_balance <= 0:
        raise ValueError(
            "Annualised return requires positive balances."
        )

    growth_ratio = float(
        ending_balance / starting_balance
    )

    return (
        growth_ratio
        ** (
            _CALENDAR_DAYS_PER_YEAR
            / elapsed_days
        )
        - 1.0
    )


def calculate_max_drawdown(
    values: Iterable[object],
) -> float:
    """Return maximum peak-to-trough drawdown as a positive ratio."""

    series = pd.Series(
        [
            _finite_float("equity value", value)
            for value in values
        ],
        dtype="float64",
    )

    if series.empty:
        raise ValueError(
            "At least one equity value is required."
        )

    if (series <= 0).any():
        raise ValueError(
            "Equity values must be greater than zero."
        )

    running_peak = series.cummax()
    drawdowns = (
        series / running_peak
    ) - 1.0

    return abs(float(drawdowns.min()))


def calculate_sharpe_ratio(
    returns: Iterable[object],
    *,
    annual_risk_free_rate: float = 0.0,
    periods_per_year: int = _TRADING_PERIODS_PER_YEAR,
) -> float:
    """Return annualised Sharpe ratio using sample standard deviation."""

    risk_free_rate = _finite_float(
        "annual_risk_free_rate",
        annual_risk_free_rate,
    )

    if risk_free_rate <= -1:
        raise ValueError(
            "annual_risk_free_rate must be greater than -1."
        )

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year < 1
    ):
        raise ValueError(
            "periods_per_year must be a positive integer."
        )

    series = pd.Series(
        [
            _finite_float("return", value)
            for value in returns
        ],
        dtype="float64",
    )

    if len(series) < 2:
        return 0.0

    periodic_risk_free_rate = (
        (1.0 + risk_free_rate)
        ** (1.0 / periods_per_year)
        - 1.0
    )

    excess_returns = (
        series
        - periodic_risk_free_rate
    )

    standard_deviation = float(
        excess_returns.std(ddof=1)
    )

    if (
        not isfinite(standard_deviation)
        or standard_deviation == 0
    ):
        return 0.0

    return (
        sqrt(periods_per_year)
        * float(excess_returns.mean())
        / standard_deviation
    )


@dataclass(frozen=True, slots=True)
class SettledTrade:
    """A closed trade paired with its exact economic settlement."""

    trade: ClosedTradeRecord
    settlement: TradeSettlement

    def __post_init__(self) -> None:
        if not isinstance(
            self.trade,
            ClosedTradeRecord,
        ):
            raise ValueError(
                "trade must be a ClosedTradeRecord."
            )

        if not isinstance(
            self.settlement,
            TradeSettlement,
        ):
            raise ValueError(
                "settlement must be a TradeSettlement."
            )

        if (
            self.settlement.raw_entry_price
            != _money(self.trade.entry_price)
        ):
            raise ValueError(
                "Settlement entry price must match the trade."
            )

        if (
            self.settlement.raw_exit_price
            != _money(self.trade.exit_price)
        ):
            raise ValueError(
                "Settlement exit price must match the trade."
            )

        if (
            self.settlement.quantity
            != _decimal(
                "trade.quantity",
                self.trade.quantity,
            )
        ):
            raise ValueError(
                "Settlement quantity must match the trade."
            )

        if (
            self.settlement.market_gross_pnl
            != _money(self.trade.gross_pnl)
        ):
            raise ValueError(
                "Settlement gross P&L must match the trade."
            )

        if self.settlement.ending_balance <= 0:
            raise ValueError(
                "Settlement ending balance must remain positive."
            )


@dataclass(frozen=True, slots=True)
class EquityPoint:
    """One strategy-account valuation on a market session."""

    timestamp: datetime
    balance: Decimal

    def __post_init__(self) -> None:
        timestamp = _aware_datetime(
            "timestamp",
            self.timestamp,
        )
        balance = _positive_decimal(
            "balance",
            self.balance,
        )

        object.__setattr__(
            self,
            "timestamp",
            timestamp,
        )
        object.__setattr__(
            self,
            "balance",
            balance,
        )


@dataclass(frozen=True, slots=True)
class BuyAndHoldComparison:
    """Frictionless buy-and-hold result over the same test window."""

    first_session: datetime
    last_session: datetime

    starting_price: float
    ending_price: float

    total_return: float
    annualised_return: float
    max_drawdown: float

    hypothetical_ending_balance: Decimal
    strategy_excess_return: float

    def __post_init__(self) -> None:
        first_session = _aware_datetime(
            "first_session",
            self.first_session,
        )
        last_session = _aware_datetime(
            "last_session",
            self.last_session,
        )

        if last_session <= first_session:
            raise ValueError(
                "Benchmark requires at least two sessions."
            )

        starting_price = _finite_float(
            "starting_price",
            self.starting_price,
        )
        ending_price = _finite_float(
            "ending_price",
            self.ending_price,
        )

        if starting_price <= 0 or ending_price <= 0:
            raise ValueError(
                "Benchmark prices must be positive."
            )

        for name in (
            "total_return",
            "annualised_return",
            "max_drawdown",
            "strategy_excess_return",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(
                    name,
                    getattr(self, name),
                ),
            )

        if not 0 <= self.max_drawdown <= 1:
            raise ValueError(
                "max_drawdown must be between 0 and 1."
            )

        hypothetical_ending_balance = (
            _positive_decimal(
                "hypothetical_ending_balance",
                self.hypothetical_ending_balance,
            )
        )

        object.__setattr__(
            self,
            "first_session",
            first_session,
        )
        object.__setattr__(
            self,
            "last_session",
            last_session,
        )
        object.__setattr__(
            self,
            "starting_price",
            starting_price,
        )
        object.__setattr__(
            self,
            "ending_price",
            ending_price,
        )
        object.__setattr__(
            self,
            "hypothetical_ending_balance",
            hypothetical_ending_balance,
        )


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Complete deterministic P2.4 performance result."""

    period_start: datetime
    period_end: datetime

    starting_balance: Decimal
    ending_balance: Decimal
    total_net_pnl: Decimal

    total_return: float
    annualised_return: float
    max_drawdown: float
    sharpe_ratio: float

    trade_count: int
    winning_trades: int
    losing_trades: int
    breakeven_trades: int

    win_rate: float
    gross_profit: Decimal
    gross_loss: Decimal
    profit_factor: float | None

    exposure: float
    average_holding_period_days: float

    equity_curve: tuple[EquityPoint, ...]
    benchmark: BuyAndHoldComparison

    def __post_init__(self) -> None:
        period_start = _aware_datetime(
            "period_start",
            self.period_start,
        )
        period_end = _aware_datetime(
            "period_end",
            self.period_end,
        )

        if period_end <= period_start:
            raise ValueError(
                "period_end must be later than period_start."
            )

        starting_balance = _positive_decimal(
            "starting_balance",
            self.starting_balance,
        )
        ending_balance = _positive_decimal(
            "ending_balance",
            self.ending_balance,
        )

        if (
            isinstance(self.trade_count, bool)
            or not isinstance(self.trade_count, int)
            or self.trade_count < 0
        ):
            raise ValueError(
                "trade_count must be non-negative."
            )

        counts = (
            self.winning_trades
            + self.losing_trades
            + self.breakeven_trades
        )

        if counts != self.trade_count:
            raise ValueError(
                "Trade counts do not reconcile."
            )

        if not 0 <= self.win_rate <= 1:
            raise ValueError(
                "win_rate must be between 0 and 1."
            )

        if not 0 <= self.max_drawdown <= 1:
            raise ValueError(
                "max_drawdown must be between 0 and 1."
            )

        if not 0 <= self.exposure <= 1:
            raise ValueError(
                "exposure must be between 0 and 1."
            )

        if self.average_holding_period_days < 0:
            raise ValueError(
                "Average holding period cannot be negative."
            )

        equity_curve = tuple(self.equity_curve)

        if not equity_curve:
            raise ValueError(
                "equity_curve cannot be empty."
            )

        if not all(
            isinstance(point, EquityPoint)
            for point in equity_curve
        ):
            raise ValueError(
                "equity_curve must contain EquityPoint objects."
            )

        if not isinstance(
            self.benchmark,
            BuyAndHoldComparison,
        ):
            raise ValueError(
                "benchmark must be a BuyAndHoldComparison."
            )

        object.__setattr__(
            self,
            "period_start",
            period_start,
        )
        object.__setattr__(
            self,
            "period_end",
            period_end,
        )
        object.__setattr__(
            self,
            "starting_balance",
            starting_balance,
        )
        object.__setattr__(
            self,
            "ending_balance",
            ending_balance,
        )
        object.__setattr__(
            self,
            "equity_curve",
            equity_curve,
        )


def _normalise_market_history(
    market_history: pd.DataFrame | pd.Series,
    period_start: datetime,
    period_end: datetime,
) -> pd.Series:
    if isinstance(market_history, pd.DataFrame):
        if "Close" not in market_history.columns:
            raise ValueError(
                "market_history is missing the Close column."
            )

        close = market_history["Close"].copy()

    elif isinstance(market_history, pd.Series):
        close = market_history.copy()

    else:
        raise ValueError(
            "market_history must be a DataFrame or Series."
        )

    if close.empty:
        raise ValueError(
            "market_history cannot be empty."
        )

    index = pd.to_datetime(
        close.index,
        errors="coerce",
        utc=True,
    )

    if index.isna().any():
        raise ValueError(
            "market_history contains invalid timestamps."
        )

    if index.duplicated().any():
        raise ValueError(
            "market_history contains duplicate timestamps."
        )

    close.index = index
    close = close.sort_index()
    close = pd.to_numeric(
        close,
        errors="coerce",
    )

    if close.isna().any():
        raise ValueError(
            "market_history contains invalid closing prices."
        )

    if (close <= 0).any():
        raise ValueError(
            "Closing prices must be greater than zero."
        )

    start_timestamp = _utc_timestamp(
        period_start
    )
    end_timestamp = _utc_timestamp(
        period_end
    )

    close = close.loc[
        (close.index >= start_timestamp)
        & (close.index <= end_timestamp)
    ]

    if len(close) < 2:
        raise ValueError(
            "At least two market sessions are required "
            "inside the performance period."
        )

    return close.astype("float64")


def _validate_and_order_trades(
    settled_trades: Sequence[SettledTrade],
    starting_balance: Decimal,
    period_start: datetime,
    period_end: datetime,
) -> tuple[SettledTrade, ...]:
    records = tuple(settled_trades)

    if not all(
        isinstance(record, SettledTrade)
        for record in records
    ):
        raise ValueError(
            "settled_trades must contain SettledTrade objects."
        )

    ordered = tuple(
        sorted(
            records,
            key=lambda record: (
                record.trade.closed_at,
                record.trade.trade_id,
            ),
        )
    )

    trade_ids = [
        record.trade.trade_id
        for record in ordered
    ]

    if len(trade_ids) != len(set(trade_ids)):
        raise ValueError(
            "Trade IDs must be unique."
        )

    current_balance = _money(
        starting_balance
    )

    for record in ordered:
        trade = record.trade

        if trade.opened_at < period_start:
            raise ValueError(
                "A trade opens before the performance period."
            )

        if trade.closed_at > period_end:
            raise ValueError(
                "A trade closes after the performance period."
            )

        if (
            record.settlement.opening_balance
            != current_balance
        ):
            raise ValueError(
                "Trade settlement balances do not form "
                "a continuous account sequence."
            )

        current_balance = (
            record.settlement.ending_balance
        )

    return ordered


def _build_equity_curve(
    close_history: pd.Series,
    settled_trades: Sequence[SettledTrade],
    starting_balance: Decimal,
) -> tuple[EquityPoint, ...]:
    balances = [
        _money(starting_balance)
        for _ in range(len(close_history))
    ]

    for record in settled_trades:
        closed_at = _utc_timestamp(
            record.trade.closed_at
        )

        position = int(
            close_history.index.searchsorted(
                closed_at,
                side="right",
            )
        ) - 1

        if position < 0:
            raise ValueError(
                "A trade closes before the first market session."
            )

        ending_balance = (
            record.settlement.ending_balance
        )

        for index in range(
            position,
            len(balances),
        ):
            balances[index] = ending_balance

    return tuple(
        EquityPoint(
            timestamp=timestamp.to_pydatetime(),
            balance=balance,
        )
        for timestamp, balance in zip(
            close_history.index,
            balances,
        )
    )


def _calculate_exposure(
    settled_trades: Sequence[SettledTrade],
    period_start: datetime,
    period_end: datetime,
) -> float:
    if not settled_trades:
        return 0.0

    intervals = sorted(
        (
            record.trade.opened_at,
            record.trade.closed_at,
        )
        for record in settled_trades
    )

    merged: list[list[datetime]] = []

    for start, end in intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(
                merged[-1][1],
                end,
            )

    exposed_seconds = sum(
        (
            end - start
        ).total_seconds()
        for start, end in merged
    )

    total_seconds = (
        period_end - period_start
    ).total_seconds()

    return min(
        1.0,
        exposed_seconds / total_seconds,
    )


def calculate_performance(
    settled_trades: Sequence[SettledTrade],
    *,
    starting_balance: object,
    period_start: datetime,
    period_end: datetime,
    market_history: pd.DataFrame | pd.Series,
    annual_risk_free_rate: float = 0.0,
) -> PerformanceReport:
    """Calculate P2.4 strategy and buy-and-hold performance metrics."""

    start = _aware_datetime(
        "period_start",
        period_start,
    )
    end = _aware_datetime(
        "period_end",
        period_end,
    )

    if end <= start:
        raise ValueError(
            "period_end must be later than period_start."
        )

    initial_balance = _money(
        _positive_decimal(
            "starting_balance",
            starting_balance,
        )
    )

    close_history = _normalise_market_history(
        market_history,
        start,
        end,
    )

    ordered_trades = _validate_and_order_trades(
        settled_trades,
        initial_balance,
        start,
        end,
    )

    ending_balance = (
        ordered_trades[-1]
        .settlement
        .ending_balance
        if ordered_trades
        else initial_balance
    )

    total_net_pnl = _money(
        ending_balance
        - initial_balance
    )

    elapsed_days = (
        end - start
    ).total_seconds() / 86400.0

    total_return = float(
        ending_balance
        / initial_balance
        - Decimal("1")
    )

    annualised_return = _annualised_return(
        initial_balance,
        ending_balance,
        elapsed_days,
    )

    equity_curve = _build_equity_curve(
        close_history,
        ordered_trades,
        initial_balance,
    )

    equity_values = [
        float(point.balance)
        for point in equity_curve
    ]

    maximum_drawdown = calculate_max_drawdown(
        equity_values
    )

    equity_series = pd.Series(
        equity_values,
        index=close_history.index,
        dtype="float64",
    )

    strategy_returns = (
        equity_series
        .pct_change()
        .dropna()
    )

    sharpe_ratio = calculate_sharpe_ratio(
        strategy_returns,
        annual_risk_free_rate=(
            annual_risk_free_rate
        ),
    )

    trade_results = [
        record.settlement.net_pnl
        for record in ordered_trades
    ]

    winning_trades = sum(
        result > 0
        for result in trade_results
    )

    losing_trades = sum(
        result < 0
        for result in trade_results
    )

    breakeven_trades = sum(
        result == 0
        for result in trade_results
    )

    trade_count = len(trade_results)

    win_rate = (
        winning_trades / trade_count
        if trade_count
        else 0.0
    )

    gross_profit = _money(
        sum(
            (
                result
                for result in trade_results
                if result > 0
            ),
            Decimal("0"),
        )
    )

    gross_loss = _money(
        abs(
            sum(
                (
                    result
                    for result in trade_results
                    if result < 0
                ),
                Decimal("0"),
            )
        )
    )

    profit_factor = (
        float(gross_profit / gross_loss)
        if gross_loss > 0
        else None
    )

    exposure = _calculate_exposure(
        ordered_trades,
        start,
        end,
    )

    average_holding_period_days = (
        sum(
            (
                record.trade.closed_at
                - record.trade.opened_at
            ).total_seconds()
            for record in ordered_trades
        )
        / trade_count
        / 86400.0
        if trade_count
        else 0.0
    )

    benchmark_start_price = float(
        close_history.iloc[0]
    )
    benchmark_end_price = float(
        close_history.iloc[-1]
    )

    benchmark_total_return = (
        benchmark_end_price
        / benchmark_start_price
        - 1.0
    )

    benchmark_ending_balance = _money(
        initial_balance
        * Decimal(
            str(
                1.0
                + benchmark_total_return
            )
        )
    )

    benchmark_annualised_return = (
        _annualised_return(
            initial_balance,
            benchmark_ending_balance,
            elapsed_days,
        )
    )

    benchmark = BuyAndHoldComparison(
        first_session=(
            close_history
            .index[0]
            .to_pydatetime()
        ),
        last_session=(
            close_history
            .index[-1]
            .to_pydatetime()
        ),
        starting_price=benchmark_start_price,
        ending_price=benchmark_end_price,
        total_return=benchmark_total_return,
        annualised_return=(
            benchmark_annualised_return
        ),
        max_drawdown=calculate_max_drawdown(
            close_history
        ),
        hypothetical_ending_balance=(
            benchmark_ending_balance
        ),
        strategy_excess_return=(
            total_return
            - benchmark_total_return
        ),
    )

    return PerformanceReport(
        period_start=start,
        period_end=end,
        starting_balance=initial_balance,
        ending_balance=ending_balance,
        total_net_pnl=total_net_pnl,
        total_return=total_return,
        annualised_return=annualised_return,
        max_drawdown=maximum_drawdown,
        sharpe_ratio=sharpe_ratio,
        trade_count=trade_count,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=breakeven_trades,
        win_rate=win_rate,
        gross_profit=gross_profit,
        gross_loss=gross_loss,
        profit_factor=profit_factor,
        exposure=exposure,
        average_holding_period_days=(
            average_holding_period_days
        ),
        equity_curve=equity_curve,
        benchmark=benchmark,
    )
