"""Immutable backtest lifecycle and trade records.

P2.1 defines records only. It does not simulate execution, inspect future
prices, connect to a broker or mutate portfolio state. P2.2 will consume these
records to implement next-session execution.
"""

from __future__ import annotations

from src.strategy import StrategyHorizon

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from math import isfinite

from src.analysis import PaperOrder, Signal, StrategyResult


class PositionSide(str, Enum):
    """Backtest position direction."""

    LONG = "LONG"
    SHORT = "SHORT"


class ExitReason(str, Enum):
    """Reason an open simulated position was closed."""

    STOP = "STOP"
    TARGET = "TARGET"
    EXPIRY = "EXPIRY"
    MANUAL = "MANUAL"
    SYSTEM = "SYSTEM"


class LifecycleEventType(str, Enum):
    """Auditable events in a backtest lifecycle."""

    SIGNAL_CREATED = "SIGNAL_CREATED"
    SIGNAL_EXPIRED = "SIGNAL_EXPIRED"
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_EXPIRED = "ORDER_EXPIRED"
    POSITION_OPENED = "POSITION_OPENED"
    STOP_HIT = "STOP_HIT"
    TARGET_HIT = "TARGET_HIT"
    POSITION_EXPIRED = "POSITION_EXPIRED"
    POSITION_CLOSED = "POSITION_CLOSED"


def _required_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")

    return value.strip()


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc

    if not isfinite(result):
        raise ValueError(f"{name} must be a finite number.")

    return result


def _positive_number(name: str, value: object) -> float:
    result = _finite_number(name, value)

    if result <= 0:
        raise ValueError(f"{name} must be greater than zero.")

    return result


def _aware_datetime(name: str, value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime.")

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware.")

    return value


def _validated_targets(
    targets: object,
    side: PositionSide,
    reference_price: float,
) -> tuple[float, ...]:
    try:
        result = tuple(
            _positive_number("target", target)
            for target in targets
        )
    except TypeError as exc:
        raise ValueError("targets must be an iterable of prices.") from exc

    if not result:
        raise ValueError("At least one target is required.")

    if side is PositionSide.LONG:
        if not all(target > reference_price for target in result):
            raise ValueError(
                "LONG targets must be above the reference price."
            )

        if any(
            later <= earlier
            for earlier, later in zip(result, result[1:])
        ):
            raise ValueError(
                "LONG targets must be strictly ascending."
            )

    if side is PositionSide.SHORT:
        if not all(target < reference_price for target in result):
            raise ValueError(
                "SHORT targets must be below the reference price."
            )

        if any(
            later >= earlier
            for earlier, later in zip(result, result[1:])
        ):
            raise ValueError(
                "SHORT targets must be strictly descending."
            )

    return result


def side_from_signal(signal: Signal) -> PositionSide:
    """Convert an actionable recommendation to a position side."""

    if signal is Signal.BUY:
        return PositionSide.LONG

    if signal is Signal.SELL:
        return PositionSide.SHORT

    raise ValueError("Only BUY or SELL can create a backtest side.")


@dataclass(frozen=True, slots=True)
class SignalRecord:
    """One actionable strategy signal entering the backtest lifecycle."""

    signal_id: str
    symbol: str
    strategy: str
    signal: Signal
    generated_at: datetime
    expires_at: datetime
    score: float
    confidence: float
    strategy_horizon: StrategyHorizon | None = None
    strategy_version: str | None = None

    def __post_init__(self) -> None:
        signal_id = _required_text("signal_id", self.signal_id)
        symbol = _required_text("symbol", self.symbol).upper()
        strategy = _required_text("strategy", self.strategy)

        if self.signal not in {Signal.BUY, Signal.SELL}:
            raise ValueError(
                "A backtest signal must be BUY or SELL."
            )

        generated_at = _aware_datetime(
            "generated_at",
            self.generated_at,
        )
        expires_at = _aware_datetime(
            "expires_at",
            self.expires_at,
        )

        if expires_at <= generated_at:
            raise ValueError(
                "expires_at must be later than generated_at."
            )

        score = _finite_number("score", self.score)

        if not -100 <= score <= 100:
            raise ValueError(
                "score must be between -100 and 100."
            )

        if self.signal is Signal.BUY and score <= 0:
            raise ValueError("BUY requires a positive score.")

        if self.signal is Signal.SELL and score >= 0:
            raise ValueError("SELL requires a negative score.")

        confidence = _finite_number(
            "confidence",
            self.confidence,
        )

        if not 0 <= confidence <= 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        object.__setattr__(self, "signal_id", signal_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "strategy", strategy)
        object.__setattr__(self, "generated_at", generated_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def from_strategy_result(
        cls,
        *,
        signal_id: str,
        symbol: str,
        generated_at: datetime,
        expires_at: datetime,
        result: StrategyResult,
    ) -> "SignalRecord":
        """Create a record from a non-vetoed actionable recommendation."""

        if not isinstance(result, StrategyResult):
            raise ValueError(
                "result must be a StrategyResult."
            )

        if result.vetoed:
            raise ValueError(
                "A vetoed recommendation cannot create a signal record."
            )

        return cls(
            signal_id=signal_id,
            symbol=symbol,
            strategy=result.strategy,
            signal=result.signal,
            generated_at=generated_at,
            expires_at=expires_at,
            score=result.score,
            confidence=result.confidence,
        )


@dataclass(frozen=True, slots=True)
class OrderRecord:
    """Paper order awaiting simulated execution."""

    order_id: str
    signal_id: str
    symbol: str
    side: PositionSide

    created_at: datetime
    expires_at: datetime

    entry_low: float
    entry_high: float
    stop_price: float
    targets: tuple[float, ...]
    quantity: float

    paper_only: bool = True

    def __post_init__(self) -> None:
        order_id = _required_text("order_id", self.order_id)
        signal_id = _required_text("signal_id", self.signal_id)
        symbol = _required_text("symbol", self.symbol).upper()

        if not isinstance(self.side, PositionSide):
            raise ValueError(
                "side must be a PositionSide value."
            )

        created_at = _aware_datetime(
            "created_at",
            self.created_at,
        )
        expires_at = _aware_datetime(
            "expires_at",
            self.expires_at,
        )

        if expires_at <= created_at:
            raise ValueError(
                "expires_at must be later than created_at."
            )

        entry_low = _positive_number(
            "entry_low",
            self.entry_low,
        )
        entry_high = _positive_number(
            "entry_high",
            self.entry_high,
        )
        stop_price = _positive_number(
            "stop_price",
            self.stop_price,
        )
        quantity = _positive_number(
            "quantity",
            self.quantity,
        )

        if entry_low > entry_high:
            raise ValueError(
                "entry_low cannot exceed entry_high."
            )

        if self.side is PositionSide.LONG:
            if stop_price >= entry_low:
                raise ValueError(
                    "LONG stop must be below the entry zone."
                )

            reference_price = entry_high

        else:
            if stop_price <= entry_high:
                raise ValueError(
                    "SHORT stop must be above the entry zone."
                )

            reference_price = entry_low

        targets = _validated_targets(
            self.targets,
            self.side,
            reference_price,
        )

        if self.paper_only is not True:
            raise ValueError(
                "Backtest orders must remain paper-only."
            )

        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "signal_id", signal_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "entry_low", entry_low)
        object.__setattr__(self, "entry_high", entry_high)
        object.__setattr__(self, "stop_price", stop_price)
        object.__setattr__(self, "targets", targets)
        object.__setattr__(self, "quantity", quantity)

    @classmethod
    def from_paper_order(
        cls,
        *,
        order_id: str,
        signal_id: str,
        paper_order: PaperOrder,
        quantity: float = 1.0,
    ) -> "OrderRecord":
        """Convert a P1 paper order to a P2 backtest order."""

        if not isinstance(paper_order, PaperOrder):
            raise ValueError(
                "paper_order must be a PaperOrder."
            )

        return cls(
            order_id=order_id,
            signal_id=signal_id,
            symbol=paper_order.symbol,
            side=side_from_signal(paper_order.signal),
            created_at=paper_order.created_at,
            expires_at=paper_order.expires_at,
            entry_low=paper_order.entry_low,
            entry_high=paper_order.entry_high,
            stop_price=paper_order.stop_price,
            targets=paper_order.targets,
            quantity=quantity,
            paper_only=True,
        )


@dataclass(frozen=True, slots=True)
class FillRecord:
    """Simulated fill produced by a future execution engine."""

    fill_id: str
    order_id: str
    signal_id: str
    symbol: str
    side: PositionSide

    filled_at: datetime
    fill_price: float
    quantity: float

    def __post_init__(self) -> None:
        fill_id = _required_text("fill_id", self.fill_id)
        order_id = _required_text("order_id", self.order_id)
        signal_id = _required_text("signal_id", self.signal_id)
        symbol = _required_text("symbol", self.symbol).upper()

        if not isinstance(self.side, PositionSide):
            raise ValueError(
                "side must be a PositionSide value."
            )

        filled_at = _aware_datetime(
            "filled_at",
            self.filled_at,
        )
        fill_price = _positive_number(
            "fill_price",
            self.fill_price,
        )
        quantity = _positive_number(
            "quantity",
            self.quantity,
        )

        object.__setattr__(self, "fill_id", fill_id)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "signal_id", signal_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "filled_at", filled_at)
        object.__setattr__(self, "fill_price", fill_price)
        object.__setattr__(self, "quantity", quantity)


@dataclass(frozen=True, slots=True)
class PositionRecord:
    """Open simulated position created from a fill."""

    position_id: str
    order_id: str
    fill_id: str
    signal_id: str
    symbol: str
    side: PositionSide

    opened_at: datetime
    expires_at: datetime

    entry_price: float
    quantity: float
    stop_price: float
    targets: tuple[float, ...]

    def __post_init__(self) -> None:
        position_id = _required_text(
            "position_id",
            self.position_id,
        )
        order_id = _required_text("order_id", self.order_id)
        fill_id = _required_text("fill_id", self.fill_id)
        signal_id = _required_text("signal_id", self.signal_id)
        symbol = _required_text("symbol", self.symbol).upper()

        if not isinstance(self.side, PositionSide):
            raise ValueError(
                "side must be a PositionSide value."
            )

        opened_at = _aware_datetime(
            "opened_at",
            self.opened_at,
        )
        expires_at = _aware_datetime(
            "expires_at",
            self.expires_at,
        )

        if expires_at <= opened_at:
            raise ValueError(
                "expires_at must be later than opened_at."
            )

        entry_price = _positive_number(
            "entry_price",
            self.entry_price,
        )
        quantity = _positive_number(
            "quantity",
            self.quantity,
        )
        stop_price = _positive_number(
            "stop_price",
            self.stop_price,
        )

        if self.side is PositionSide.LONG:
            if stop_price >= entry_price:
                raise ValueError(
                    "LONG stop must be below entry_price."
                )
        else:
            if stop_price <= entry_price:
                raise ValueError(
                    "SHORT stop must be above entry_price."
                )

        targets = _validated_targets(
            self.targets,
            self.side,
            entry_price,
        )

        object.__setattr__(self, "position_id", position_id)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "fill_id", fill_id)
        object.__setattr__(self, "signal_id", signal_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "opened_at", opened_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "entry_price", entry_price)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "stop_price", stop_price)
        object.__setattr__(self, "targets", targets)


@dataclass(frozen=True, slots=True)
class ClosedTradeRecord:
    """Completed simulated trade with a validated exit reason."""

    trade_id: str
    position_id: str
    order_id: str
    fill_id: str
    signal_id: str
    symbol: str
    side: PositionSide

    opened_at: datetime
    closed_at: datetime
    expires_at: datetime

    entry_price: float
    exit_price: float
    quantity: float
    stop_price: float
    targets: tuple[float, ...]

    exit_reason: ExitReason
    target_index: int | None = None

    def __post_init__(self) -> None:
        trade_id = _required_text("trade_id", self.trade_id)
        position_id = _required_text(
            "position_id",
            self.position_id,
        )
        order_id = _required_text("order_id", self.order_id)
        fill_id = _required_text("fill_id", self.fill_id)
        signal_id = _required_text("signal_id", self.signal_id)
        symbol = _required_text("symbol", self.symbol).upper()

        if not isinstance(self.side, PositionSide):
            raise ValueError(
                "side must be a PositionSide value."
            )

        if not isinstance(self.exit_reason, ExitReason):
            raise ValueError(
                "exit_reason must be an ExitReason value."
            )

        opened_at = _aware_datetime(
            "opened_at",
            self.opened_at,
        )
        closed_at = _aware_datetime(
            "closed_at",
            self.closed_at,
        )
        expires_at = _aware_datetime(
            "expires_at",
            self.expires_at,
        )

        if closed_at <= opened_at:
            raise ValueError(
                "closed_at must be later than opened_at."
            )

        entry_price = _positive_number(
            "entry_price",
            self.entry_price,
        )
        exit_price = _positive_number(
            "exit_price",
            self.exit_price,
        )
        quantity = _positive_number(
            "quantity",
            self.quantity,
        )
        stop_price = _positive_number(
            "stop_price",
            self.stop_price,
        )

        if self.side is PositionSide.LONG:
            if stop_price >= entry_price:
                raise ValueError(
                    "LONG stop must be below entry_price."
                )
        else:
            if stop_price <= entry_price:
                raise ValueError(
                    "SHORT stop must be above entry_price."
                )

        targets = _validated_targets(
            self.targets,
            self.side,
            entry_price,
        )

        target_index = self.target_index

        if self.exit_reason is ExitReason.STOP:
            if target_index is not None:
                raise ValueError(
                    "STOP exits cannot have target_index."
                )

            if (
                self.side is PositionSide.LONG
                and exit_price > stop_price
            ):
                raise ValueError(
                    "LONG STOP exit must be at or below stop_price."
                )

            if (
                self.side is PositionSide.SHORT
                and exit_price < stop_price
            ):
                raise ValueError(
                    "SHORT STOP exit must be at or above stop_price."
                )

        elif self.exit_reason is ExitReason.TARGET:
            if (
                isinstance(target_index, bool)
                or not isinstance(target_index, int)
            ):
                raise ValueError(
                    "TARGET exits require an integer target_index."
                )

            if not 0 <= target_index < len(targets):
                raise ValueError(
                    "target_index is outside the target list."
                )

            selected_target = targets[target_index]

            if (
                self.side is PositionSide.LONG
                and exit_price < selected_target
            ):
                raise ValueError(
                    "LONG TARGET exit must reach the selected target."
                )

            if (
                self.side is PositionSide.SHORT
                and exit_price > selected_target
            ):
                raise ValueError(
                    "SHORT TARGET exit must reach the selected target."
                )

        else:
            if target_index is not None:
                raise ValueError(
                    "target_index is allowed only for TARGET exits."
                )

        if (
            self.exit_reason is ExitReason.EXPIRY
            and closed_at < expires_at
        ):
            raise ValueError(
                "EXPIRY exit cannot occur before expires_at."
            )

        object.__setattr__(self, "trade_id", trade_id)
        object.__setattr__(self, "position_id", position_id)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "fill_id", fill_id)
        object.__setattr__(self, "signal_id", signal_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "opened_at", opened_at)
        object.__setattr__(self, "closed_at", closed_at)
        object.__setattr__(self, "expires_at", expires_at)
        object.__setattr__(self, "entry_price", entry_price)
        object.__setattr__(self, "exit_price", exit_price)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "stop_price", stop_price)
        object.__setattr__(self, "targets", targets)

    @property
    def gross_pnl(self) -> float:
        """Return simulated gross profit or loss before costs."""

        price_change = self.exit_price - self.entry_price

        if self.side is PositionSide.SHORT:
            price_change *= -1

        return round(price_change * self.quantity, 8)

    @property
    def return_pct(self) -> float:
        """Return direction-adjusted percentage return."""

        price_change = self.exit_price - self.entry_price

        if self.side is PositionSide.SHORT:
            price_change *= -1

        return round(
            price_change / self.entry_price * 100,
            8,
        )

    @property
    def target_number(self) -> int | None:
        """Return a human-readable one-based target number."""

        if self.target_index is None:
            return None

        return self.target_index + 1


@dataclass(frozen=True, slots=True)
class LifecycleEvent:
    """Auditable event linked to one lifecycle record."""

    event_id: str
    event_type: LifecycleEventType
    occurred_at: datetime
    symbol: str
    reference_id: str
    message: str

    def __post_init__(self) -> None:
        event_id = _required_text("event_id", self.event_id)
        reference_id = _required_text(
            "reference_id",
            self.reference_id,
        )
        symbol = _required_text("symbol", self.symbol).upper()
        message = _required_text("message", self.message)

        if not isinstance(
            self.event_type,
            LifecycleEventType,
        ):
            raise ValueError(
                "event_type must be a LifecycleEventType value."
            )

        occurred_at = _aware_datetime(
            "occurred_at",
            self.occurred_at,
        )

        object.__setattr__(self, "event_id", event_id)
        object.__setattr__(self, "reference_id", reference_id)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "occurred_at", occurred_at)


@dataclass(frozen=True, slots=True)
class BacktestLifecycle:
    """Validated linked records for one simulated trade lifecycle."""

    signal: SignalRecord
    order: OrderRecord
    fill: FillRecord | None = None
    position: PositionRecord | None = None
    closed_trade: ClosedTradeRecord | None = None
    events: tuple[LifecycleEvent, ...] = field(
        default_factory=tuple,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.signal, SignalRecord):
            raise ValueError(
                "signal must be a SignalRecord."
            )

        if not isinstance(self.order, OrderRecord):
            raise ValueError(
                "order must be an OrderRecord."
            )

        if self.order.signal_id != self.signal.signal_id:
            raise ValueError(
                "order.signal_id must match signal.signal_id."
            )

        if self.order.symbol != self.signal.symbol:
            raise ValueError(
                "Order and signal symbols must match."
            )

        if self.order.side is not side_from_signal(
            self.signal.signal
        ):
            raise ValueError(
                "Order side must match signal direction."
            )

        if self.order.created_at < self.signal.generated_at:
            raise ValueError(
                "Order cannot be created before the signal."
            )

        if self.order.expires_at > self.signal.expires_at:
            raise ValueError(
                "Order cannot outlive the signal."
            )

        if self.fill is None:
            if self.position is not None:
                raise ValueError(
                    "A position requires a fill."
                )

            if self.closed_trade is not None:
                raise ValueError(
                    "A closed trade requires a position."
                )

        else:
            if not isinstance(self.fill, FillRecord):
                raise ValueError(
                    "fill must be a FillRecord or None."
                )

            if self.fill.order_id != self.order.order_id:
                raise ValueError(
                    "fill.order_id must match order.order_id."
                )

            if self.fill.signal_id != self.signal.signal_id:
                raise ValueError(
                    "fill.signal_id must match signal.signal_id."
                )

            if (
                self.fill.symbol != self.order.symbol
                or self.fill.side is not self.order.side
            ):
                raise ValueError(
                    "Fill symbol and side must match the order."
                )

            if not (
                self.order.created_at
                <= self.fill.filled_at
                <= self.order.expires_at
            ):
                raise ValueError(
                    "Fill must occur while the order is valid."
                )

        if self.position is not None:
            if not isinstance(self.position, PositionRecord):
                raise ValueError(
                    "position must be a PositionRecord or None."
                )

            if self.fill is None:
                raise ValueError(
                    "A position requires a fill."
                )

            if self.position.fill_id != self.fill.fill_id:
                raise ValueError(
                    "position.fill_id must match fill.fill_id."
                )

            if self.position.order_id != self.order.order_id:
                raise ValueError(
                    "position.order_id must match order.order_id."
                )

            if self.position.signal_id != self.signal.signal_id:
                raise ValueError(
                    "position.signal_id must match signal.signal_id."
                )

            if (
                self.position.symbol != self.fill.symbol
                or self.position.side is not self.fill.side
            ):
                raise ValueError(
                    "Position symbol and side must match the fill."
                )

            if self.position.opened_at < self.fill.filled_at:
                raise ValueError(
                    "Position cannot open before its fill."
                )

            if self.position.expires_at != self.order.expires_at:
                raise ValueError(
                    "Position expiry must match order expiry."
                )

        if self.closed_trade is not None:
            if not isinstance(
                self.closed_trade,
                ClosedTradeRecord,
            ):
                raise ValueError(
                    "closed_trade must be a ClosedTradeRecord or None."
                )

            if self.position is None:
                raise ValueError(
                    "A closed trade requires a position."
                )

            trade = self.closed_trade
            position = self.position

            if trade.position_id != position.position_id:
                raise ValueError(
                    "Trade position_id must match the position."
                )

            for name in (
                "order_id",
                "fill_id",
                "signal_id",
                "symbol",
                "side",
                "opened_at",
                "expires_at",
                "entry_price",
                "quantity",
                "stop_price",
                "targets",
            ):
                if getattr(trade, name) != getattr(position, name):
                    raise ValueError(
                        f"Trade {name} must match the position."
                    )

        events = tuple(self.events)

        if not all(
            isinstance(event, LifecycleEvent)
            for event in events
        ):
            raise ValueError(
                "events must contain only LifecycleEvent objects."
            )

        event_ids = [event.event_id for event in events]

        if len(event_ids) != len(set(event_ids)):
            raise ValueError(
                "Lifecycle event IDs must be unique."
            )

        if any(
            event.symbol != self.signal.symbol
            for event in events
        ):
            raise ValueError(
                "Lifecycle event symbols must match the signal."
            )

        object.__setattr__(self, "events", events)

    @property
    def is_filled(self) -> bool:
        return self.fill is not None

    @property
    def is_open(self) -> bool:
        return (
            self.position is not None
            and self.closed_trade is None
        )

    @property
    def is_closed(self) -> bool:
        return self.closed_trade is not None
