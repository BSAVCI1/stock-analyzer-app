"""Deterministic next-session paper-execution engine.

The engine:

- never executes on the signal/order creation bar
- evaluates sessions strictly after signal and order creation
- supports next-open and persistent limit-zone rules
- never fills at or after order expiry
- creates linked FillRecord and PositionRecord objects
- has no broker, network or live-execution integration
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import pandas as pd

from .model import (
    BacktestLifecycle,
    FillRecord,
    LifecycleEvent,
    LifecycleEventType,
    OrderRecord,
    PositionRecord,
    SignalRecord,
)


_REQUIRED_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
)


class FillRule(str, Enum):
    """Supported deterministic entry rules."""

    NEXT_OPEN = "NEXT_OPEN"
    LIMIT = "LIMIT"


class ExecutionStatus(str, Enum):
    """Outcome of one execution-engine evaluation."""

    FILLED = "FILLED"
    NOT_FILLED = "NOT_FILLED"
    EXPIRED = "EXPIRED"
    PENDING = "PENDING"


def _required_text(
    name: str,
    value: object,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value.strip()


def _aware_datetime(
    name: str,
    value: object,
) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(
            f"{name} must be a datetime."
        )

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            f"{name} must be timezone-aware."
        )

    return value


def _utc_timestamp(
    value: datetime,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        raise ValueError(
            "Execution timestamps must be timezone-aware."
        )

    return timestamp.tz_convert("UTC")


def _normalise_history(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Return validated, UTC-normalised OHLC history."""

    if not isinstance(history, pd.DataFrame):
        raise ValueError(
            "history must be a pandas DataFrame."
        )

    if history.empty:
        raise ValueError(
            "history cannot be empty."
        )

    missing = [
        column
        for column in _REQUIRED_COLUMNS
        if column not in history.columns
    ]

    if missing:
        raise ValueError(
            "history is missing required columns: "
            + ", ".join(missing)
        )

    frame = history.loc[
        :,
        list(_REQUIRED_COLUMNS),
    ].copy()

    index = pd.to_datetime(
        frame.index,
        errors="coerce",
        utc=True,
    )

    if index.isna().any():
        raise ValueError(
            "history contains invalid session timestamps."
        )

    if index.duplicated().any():
        raise ValueError(
            "history contains duplicate session timestamps."
        )

    frame.index = index
    frame = frame.sort_index()

    for column in _REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        )

    if frame[list(_REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(
            "history contains non-numeric OHLC values."
        )

    if (
        frame[list(_REQUIRED_COLUMNS)]
        <= 0
    ).any().any():
        raise ValueError(
            "OHLC prices must be greater than zero."
        )

    maximum_required = frame[
        ["Open", "Low", "Close"]
    ].max(axis=1)

    minimum_required = frame[
        ["Open", "High", "Close"]
    ].min(axis=1)

    if (
        frame["High"] < maximum_required
    ).any():
        raise ValueError(
            "Session High cannot be below Open, Low or Close."
        )

    if (
        frame["Low"] > minimum_required
    ).any():
        raise ValueError(
            "Session Low cannot be above Open, High or Close."
        )

    return frame


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Traceable result returned by the execution engine."""

    lifecycle: BacktestLifecycle
    status: ExecutionStatus
    fill_rule: FillRule
    evaluated_sessions: int
    reason: str
    decision_at: datetime | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.lifecycle,
            BacktestLifecycle,
        ):
            raise ValueError(
                "lifecycle must be a BacktestLifecycle."
            )

        if not isinstance(
            self.status,
            ExecutionStatus,
        ):
            raise ValueError(
                "status must be an ExecutionStatus."
            )

        if not isinstance(
            self.fill_rule,
            FillRule,
        ):
            raise ValueError(
                "fill_rule must be a FillRule."
            )

        if (
            isinstance(self.evaluated_sessions, bool)
            or not isinstance(
                self.evaluated_sessions,
                int,
            )
            or self.evaluated_sessions < 0
        ):
            raise ValueError(
                "evaluated_sessions must be a "
                "non-negative integer."
            )

        reason = _required_text(
            "reason",
            self.reason,
        )

        decision_at = self.decision_at

        if decision_at is not None:
            decision_at = _aware_datetime(
                "decision_at",
                decision_at,
            )

        if self.status is ExecutionStatus.FILLED:
            if (
                self.lifecycle.fill is None
                or self.lifecycle.position is None
            ):
                raise ValueError(
                    "FILLED requires a fill and position."
                )
        elif (
            self.lifecycle.fill is not None
            or self.lifecycle.position is not None
        ):
            raise ValueError(
                "Non-filled results cannot contain "
                "a fill or position."
            )

        object.__setattr__(
            self,
            "reason",
            reason,
        )
        object.__setattr__(
            self,
            "decision_at",
            decision_at,
        )

    @property
    def fill(self) -> FillRecord | None:
        return self.lifecycle.fill

    @property
    def position(self) -> PositionRecord | None:
        return self.lifecycle.position


def _next_open_fill_price(
    order: OrderRecord,
    session: pd.Series,
) -> float | None:
    """Fill only when the first eligible open is inside the entry zone."""

    opening_price = float(session["Open"])

    if (
        order.entry_low
        <= opening_price
        <= order.entry_high
    ):
        return opening_price

    return None


def _limit_fill_price(
    order: OrderRecord,
    session: pd.Series,
) -> float | None:
    """Determine the first tradable price inside the entry zone.

    Gap handling is deterministic:

    - opening inside the zone fills at the opening price
    - opening below the zone fills at entry_low if price reaches it
    - opening above the zone fills at entry_high if price reaches it
    """

    opening_price = float(session["Open"])
    high = float(session["High"])
    low = float(session["Low"])

    intersects_zone = not (
        high < order.entry_low
        or low > order.entry_high
    )

    if not intersects_zone:
        return None

    if (
        order.entry_low
        <= opening_price
        <= order.entry_high
    ):
        return opening_price

    if opening_price < order.entry_low:
        return order.entry_low

    return order.entry_high


def _filled_result(
    *,
    signal: SignalRecord,
    order: OrderRecord,
    fill_rule: FillRule,
    filled_at: datetime,
    fill_price: float,
    evaluated_sessions: int,
) -> ExecutionResult:
    fill_id = f"{order.order_id}:FILL"
    position_id = f"{order.order_id}:POSITION"

    fill = FillRecord(
        fill_id=fill_id,
        order_id=order.order_id,
        signal_id=signal.signal_id,
        symbol=order.symbol,
        side=order.side,
        filled_at=filled_at,
        fill_price=fill_price,
        quantity=order.quantity,
    )

    position = PositionRecord(
        position_id=position_id,
        order_id=order.order_id,
        fill_id=fill.fill_id,
        signal_id=signal.signal_id,
        symbol=order.symbol,
        side=order.side,
        opened_at=filled_at,
        expires_at=order.expires_at,
        entry_price=fill_price,
        quantity=order.quantity,
        stop_price=order.stop_price,
        targets=order.targets,
    )

    events = (
        LifecycleEvent(
            event_id=(
                f"{order.order_id}:ORDER_FILLED"
            ),
            event_type=(
                LifecycleEventType.ORDER_FILLED
            ),
            occurred_at=filled_at,
            symbol=order.symbol,
            reference_id=fill.fill_id,
            message=(
                f"{fill_rule.value} order filled "
                f"at {fill_price:.6f}."
            ),
        ),
        LifecycleEvent(
            event_id=(
                f"{order.order_id}:POSITION_OPENED"
            ),
            event_type=(
                LifecycleEventType.POSITION_OPENED
            ),
            occurred_at=filled_at,
            symbol=order.symbol,
            reference_id=position.position_id,
            message=(
                "Simulated paper position opened."
            ),
        ),
    )

    lifecycle = BacktestLifecycle(
        signal=signal,
        order=order,
        fill=fill,
        position=position,
        events=events,
    )

    return ExecutionResult(
        lifecycle=lifecycle,
        status=ExecutionStatus.FILLED,
        fill_rule=fill_rule,
        evaluated_sessions=evaluated_sessions,
        decision_at=filled_at,
        reason=(
            f"Order filled using the "
            f"{fill_rule.value} rule."
        ),
    )


def _expired_result(
    *,
    signal: SignalRecord,
    order: OrderRecord,
    fill_rule: FillRule,
    evaluated_sessions: int,
) -> ExecutionResult:
    event = LifecycleEvent(
        event_id=(
            f"{order.order_id}:ORDER_EXPIRED"
        ),
        event_type=LifecycleEventType.ORDER_EXPIRED,
        occurred_at=order.expires_at,
        symbol=order.symbol,
        reference_id=order.order_id,
        message=(
            "Paper order expired without a fill."
        ),
    )

    lifecycle = BacktestLifecycle(
        signal=signal,
        order=order,
        events=(event,),
    )

    return ExecutionResult(
        lifecycle=lifecycle,
        status=ExecutionStatus.EXPIRED,
        fill_rule=fill_rule,
        evaluated_sessions=evaluated_sessions,
        decision_at=order.expires_at,
        reason=(
            "No valid fill occurred before order expiry."
        ),
    )


def _unfilled_result(
    *,
    signal: SignalRecord,
    order: OrderRecord,
    fill_rule: FillRule,
    status: ExecutionStatus,
    evaluated_sessions: int,
    reason: str,
    decision_at: datetime | None,
) -> ExecutionResult:
    return ExecutionResult(
        lifecycle=BacktestLifecycle(
            signal=signal,
            order=order,
        ),
        status=status,
        fill_rule=fill_rule,
        evaluated_sessions=evaluated_sessions,
        decision_at=decision_at,
        reason=reason,
    )


def execute_next_session(
    signal: SignalRecord,
    order: OrderRecord,
    history: pd.DataFrame,
    *,
    fill_rule: FillRule = FillRule.LIMIT,
) -> ExecutionResult:
    """Evaluate a paper order using only sessions after signal creation.

    NEXT_OPEN:
        Evaluates only the first eligible session. It fills when that
        session opens inside the predefined entry zone.

    LIMIT:
        Evaluates eligible sessions chronologically until the entry zone
        is touched or the order expires.

    A session is eligible only when its timestamp is strictly later than
    both signal generation and order creation, and strictly earlier than
    order expiry.
    """

    if not isinstance(signal, SignalRecord):
        raise ValueError(
            "signal must be a SignalRecord."
        )

    if not isinstance(order, OrderRecord):
        raise ValueError(
            "order must be an OrderRecord."
        )

    if not isinstance(fill_rule, FillRule):
        raise ValueError(
            "fill_rule must be a FillRule."
        )

    # Validate the signal/order relationship before reading market data.
    BacktestLifecycle(
        signal=signal,
        order=order,
    )

    frame = _normalise_history(history)

    signal_time = _utc_timestamp(
        signal.generated_at
    )
    order_time = _utc_timestamp(
        order.created_at
    )
    expiry_time = _utc_timestamp(
        order.expires_at
    )

    execution_cutoff = max(
        signal_time,
        order_time,
    )

    # Strictly later than creation prevents same-bar execution.
    # Strictly earlier than expiry prevents creation of an already-expired
    # PositionRecord.
    eligible = frame.loc[
        (frame.index > execution_cutoff)
        & (frame.index < expiry_time)
    ]

    if fill_rule is FillRule.NEXT_OPEN:
        if not eligible.empty:
            session_time = eligible.index[0]
            session = eligible.iloc[0]

            fill_price = _next_open_fill_price(
                order,
                session,
            )

            decision_at = (
                session_time.to_pydatetime()
            )

            if fill_price is not None:
                return _filled_result(
                    signal=signal,
                    order=order,
                    fill_rule=fill_rule,
                    filled_at=decision_at,
                    fill_price=fill_price,
                    evaluated_sessions=1,
                )

            return _unfilled_result(
                signal=signal,
                order=order,
                fill_rule=fill_rule,
                status=ExecutionStatus.NOT_FILLED,
                evaluated_sessions=1,
                decision_at=decision_at,
                reason=(
                    "The first eligible session opened "
                    "outside the entry zone."
                ),
            )

    else:
        for evaluated_sessions, (
            session_time,
            session,
        ) in enumerate(
            eligible.iterrows(),
            start=1,
        ):
            fill_price = _limit_fill_price(
                order,
                session,
            )

            if fill_price is None:
                continue

            return _filled_result(
                signal=signal,
                order=order,
                fill_rule=fill_rule,
                filled_at=(
                    session_time.to_pydatetime()
                ),
                fill_price=fill_price,
                evaluated_sessions=(
                    evaluated_sessions
                ),
            )

    latest_history_time = frame.index[-1]

    if latest_history_time >= expiry_time:
        return _expired_result(
            signal=signal,
            order=order,
            fill_rule=fill_rule,
            evaluated_sessions=len(eligible),
        )

    last_evaluated_at = (
        eligible.index[-1].to_pydatetime()
        if not eligible.empty
        else None
    )

    return _unfilled_result(
        signal=signal,
        order=order,
        fill_rule=fill_rule,
        status=ExecutionStatus.PENDING,
        evaluated_sessions=len(eligible),
        decision_at=last_evaluated_at,
        reason=(
            "No fill has occurred and the available "
            "history does not yet reach order expiry."
        ),
    )
