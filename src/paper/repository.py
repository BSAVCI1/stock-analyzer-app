"""Persistent SQLite repository for the paper portfolio."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import sqlite3
from typing import Mapping, Sequence
from uuid import uuid4

from src.backtest import PositionSide

from .database import (
    DEFAULT_DATABASE_PATH,
    connect_database,
    transaction,
)
from .ledger import (
    calculate_entry_cash,
    calculate_long_trade,
)
from .migrations import initialize_database
from .models import (
    AccountReconciliation,
    AccountStatus,
    ClosedPaperTrade,
    NotificationChannel,
    NotificationRecord,
    NotificationStatus,
    OrderStatus,
    PaperAccount,
    PaperExitReason,
    PaperFillRecord,
    PaperOrderRecord,
    PaperPositionRecord,
    PersistedSignal,
    PositionStatus,
    SystemEventRecord,
    money,
)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(
            "Timestamp must be timezone-aware."
        )

    return value.astimezone(
        timezone.utc
    ).isoformat()


def _datetime(value: str | None) -> datetime | None:
    if value is None:
        return None

    return datetime.fromisoformat(value)


def _json_dump(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decimal_targets(
    raw_value: str,
) -> tuple[Decimal, ...]:
    return tuple(
        money(value)
        for value in json.loads(raw_value)
    )


def _insert_event(
    connection: sqlite3.Connection,
    *,
    account_id: str | None,
    event_type: str,
    severity: str,
    reference_type: str | None,
    reference_id: str | None,
    message: str,
    metadata: Mapping[str, object] | None,
    created_at: datetime,
) -> None:
    connection.execute(
        """
        INSERT INTO paper_system_events(
            event_id,
            account_id,
            event_type,
            severity,
            reference_type,
            reference_id,
            message,
            metadata_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _new_id("EVT"),
            account_id,
            event_type,
            severity,
            reference_type,
            reference_id,
            message,
            _json_dump(dict(metadata or {})),
            _timestamp(created_at),
        ),
    )


class PaperRepository:
    """SQLite-backed persistent paper portfolio."""

    def __init__(
        self,
        database_path: str | Path = DEFAULT_DATABASE_PATH,
    ) -> None:
        self.database_path = Path(database_path)
        initialize_database(self.database_path)

    def _read_one(
        self,
        query: str,
        parameters: Sequence[object],
    ) -> sqlite3.Row | None:
        connection = connect_database(
            self.database_path
        )

        try:
            return connection.execute(
                query,
                tuple(parameters),
            ).fetchone()
        finally:
            connection.close()

    @staticmethod
    def _account_from_row(
        row: sqlite3.Row,
    ) -> PaperAccount:
        return PaperAccount(
            account_id=row["account_id"],
            name=row["name"],
            base_currency=row["base_currency"],
            starting_balance=money(
                row["starting_balance"]
            ),
            cash_balance=money(
                row["cash_balance"]
            ),
            reserved_cash=money(
                row["reserved_cash"]
            ),
            status=AccountStatus(row["status"]),
            created_at=_datetime(
                row["created_at"]
            ),
            updated_at=_datetime(
                row["updated_at"]
            ),
        )

    @staticmethod
    def _signal_from_row(
        row: sqlite3.Row,
    ) -> PersistedSignal:
        return PersistedSignal(
            signal_id=row["signal_id"],
            account_id=row["account_id"],
            scan_id=row["scan_id"],
            symbol=row["symbol"],
            quote_currency=(
                row["quote_currency"]
            ),
            generated_at=_datetime(
                row["generated_at"]
            ),
            expires_at=_datetime(
                row["expires_at"]
            ),
            strategy=row["strategy"],
            recommendation=row["recommendation"],
            market_regime=row["market_regime"],
            score=float(row["score"]),
            confidence=float(row["confidence"]),
            reward_to_risk=float(
                row["reward_to_risk"]
            ),
            entry_low=money(row["entry_low"]),
            entry_high=money(row["entry_high"]),
            stop_price=money(row["stop_price"]),
            targets=_decimal_targets(
                row["targets_json"]
            ),
            evidence=tuple(
                json.loads(row["evidence_json"])
            ),
            conflicts=tuple(
                json.loads(row["conflicts_json"])
            ),
            threshold_version=(
                row["threshold_version"]
            ),
            app_version=row["app_version"],
            created_at=_datetime(
                row["created_at"]
            ),
        )

    @staticmethod
    def _order_from_row(
        row: sqlite3.Row,
    ) -> PaperOrderRecord:
        return PaperOrderRecord(
            order_id=row["order_id"],
            account_id=row["account_id"],
            signal_id=row["signal_id"],
            idempotency_key=(
                row["idempotency_key"]
            ),
            symbol=row["symbol"],
            quote_currency=(
                row["quote_currency"]
            ),
            portfolio_currency=(
                row["portfolio_currency"]
            ),
            side=PositionSide(row["side"]),
            quantity=money(row["quantity"]),
            entry_low=money(row["entry_low"]),
            entry_high=money(row["entry_high"]),
            stop_price=money(row["stop_price"]),
            targets=_decimal_targets(
                row["targets_json"]
            ),
            estimated_cash_required=money(
                row["estimated_cash_required"]
            ),
            reserved_cash=money(
                row["reserved_cash"]
            ),
            reservation_fx_rate=(
                money(
                    row["reservation_fx_rate"]
                )
                if row["reservation_fx_rate"]
                is not None
                else None
            ),
            reservation_fx_as_of=(
                _datetime(
                    row["reservation_fx_as_of"]
                )
                if row["reservation_fx_as_of"]
                is not None
                else None
            ),
            reservation_fx_source=(
                row["reservation_fx_source"]
            ),
            status=OrderStatus(row["status"]),
            created_at=_datetime(
                row["created_at"]
            ),
            expires_at=_datetime(
                row["expires_at"]
            ),
            filled_at=_datetime(
                row["filled_at"]
            ),
            closed_at=_datetime(
                row["closed_at"]
            ),
        )

    @staticmethod
    def _fill_from_row(
        row: sqlite3.Row,
    ) -> PaperFillRecord:
        return PaperFillRecord(
            fill_id=row["fill_id"],
            order_id=row["order_id"],
            quote_currency=(
                row["quote_currency"]
            ),
            portfolio_currency=(
                row["portfolio_currency"]
            ),
            price=money(row["price"]),
            quantity=money(row["quantity"]),
            fees=money(row["fees"]),
            slippage=money(row["slippage"]),
            entry_fx_rate=(
                money(row["entry_fx_rate"])
                if row["entry_fx_rate"]
                is not None
                else None
            ),
            entry_fx_as_of=(
                _datetime(row["entry_fx_as_of"])
                if row["entry_fx_as_of"]
                is not None
                else None
            ),
            entry_fx_source=(
                row["entry_fx_source"]
            ),
            cash_required_portfolio=(
                money(
                    row[
                        "cash_required_portfolio"
                    ]
                )
                if row[
                    "cash_required_portfolio"
                ]
                is not None
                else None
            ),
            filled_at=_datetime(
                row["filled_at"]
            ),
        )

    @staticmethod
    def _position_from_row(
        row: sqlite3.Row,
    ) -> PaperPositionRecord:
        return PaperPositionRecord(
            position_id=row["position_id"],
            account_id=row["account_id"],
            order_id=row["order_id"],
            fill_id=row["fill_id"],
            symbol=row["symbol"],
            quote_currency=(
                row["quote_currency"]
            ),
            portfolio_currency=(
                row["portfolio_currency"]
            ),
            side=PositionSide(row["side"]),
            quantity=money(row["quantity"]),
            entry_price=money(
                row["entry_price"]
            ),
            stop_price=money(
                row["stop_price"]
            ),
            targets=_decimal_targets(
                row["targets_json"]
            ),
            entry_fx_rate=(
                money(row["entry_fx_rate"])
                if row["entry_fx_rate"]
                is not None
                else None
            ),
            entry_fx_as_of=(
                _datetime(row["entry_fx_as_of"])
                if row["entry_fx_as_of"]
                is not None
                else None
            ),
            entry_fx_source=(
                row["entry_fx_source"]
            ),
            entry_cash_portfolio=(
                money(
                    row["entry_cash_portfolio"]
                )
                if row["entry_cash_portfolio"]
                is not None
                else None
            ),
            opened_at=_datetime(
                row["opened_at"]
            ),
            expires_at=_datetime(
                row["expires_at"]
            ),
            status=PositionStatus(
                row["status"]
            ),
            closed_at=_datetime(
                row["closed_at"]
            ),
        )

    @staticmethod
    def _trade_from_row(
        row: sqlite3.Row,
    ) -> ClosedPaperTrade:
        return ClosedPaperTrade(
            trade_id=row["trade_id"],
            position_id=row["position_id"],
            account_id=row["account_id"],
            order_id=row["order_id"],
            fill_id=row["fill_id"],
            signal_id=row["signal_id"],
            symbol=row["symbol"],
            quote_currency=(
                row["quote_currency"]
            ),
            portfolio_currency=(
                row["portfolio_currency"]
            ),
            strategy=row["strategy"],
            market_regime=row["market_regime"],
            entry_time=_datetime(
                row["entry_time"]
            ),
            entry_price=money(
                row["entry_price"]
            ),
            exit_time=_datetime(
                row["exit_time"]
            ),
            exit_price=money(
                row["exit_price"]
            ),
            exit_reason=PaperExitReason(
                row["exit_reason"]
            ),
            quantity=money(row["quantity"]),
            entry_fx_rate=(
                money(row["entry_fx_rate"])
                if row["entry_fx_rate"]
                is not None
                else None
            ),
            entry_fx_as_of=(
                _datetime(row["entry_fx_as_of"])
                if row["entry_fx_as_of"]
                is not None
                else None
            ),
            entry_fx_source=(
                row["entry_fx_source"]
            ),
            exit_fx_rate=(
                money(row["exit_fx_rate"])
                if row["exit_fx_rate"]
                is not None
                else None
            ),
            exit_fx_as_of=(
                _datetime(row["exit_fx_as_of"])
                if row["exit_fx_as_of"]
                is not None
                else None
            ),
            exit_fx_source=(
                row["exit_fx_source"]
            ),
            gross_pnl=money(
                row["gross_pnl"]
            ),
            fees=money(row["fees"]),
            slippage=money(row["slippage"]),
            net_pnl=money(row["net_pnl"]),
            return_pct=float(
                row["return_pct"]
            ),
            holding_seconds=int(
                row["holding_seconds"]
            ),
        )

    def create_account(
        self,
        *,
        name: str,
        base_currency: str,
        starting_balance: object,
        created_at: datetime | None = None,
        account_id: str | None = None,
    ) -> PaperAccount:
        at = created_at or _utc_now()
        account_id = account_id or _new_id(
            "ACC"
        )
        balance = money(starting_balance)

        if balance <= 0:
            raise ValueError(
                "starting_balance must be positive."
            )

        with transaction(
            self.database_path
        ) as connection:
            connection.execute(
                """
                INSERT INTO paper_accounts(
                    account_id,
                    name,
                    base_currency,
                    starting_balance,
                    cash_balance,
                    reserved_cash,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    account_id,
                    name.strip(),
                    base_currency.strip().upper(),
                    str(balance),
                    str(balance),
                    str(money(0)),
                    AccountStatus.ACTIVE.value,
                    _timestamp(at),
                    _timestamp(at),
                ),
            )

            connection.execute(
                """
                INSERT INTO paper_ledger_entries(
                    ledger_id,
                    account_id,
                    event_type,
                    amount,
                    balance_after,
                    reference_type,
                    reference_id,
                    description,
                    occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_id("LED"),
                    account_id,
                    "INITIAL_DEPOSIT",
                    str(balance),
                    str(balance),
                    "ACCOUNT",
                    account_id,
                    "Initial paper-account deposit.",
                    _timestamp(at),
                ),
            )

            _insert_event(
                connection,
                account_id=account_id,
                event_type="ACCOUNT_CREATED",
                severity="INFO",
                reference_type="ACCOUNT",
                reference_id=account_id,
                message="Paper account created.",
                metadata={
                    "base_currency":
                    base_currency.upper(),
                    "starting_balance":
                    str(balance),
                },
                created_at=at,
            )

        return self.get_account(account_id)

    def get_account(
        self,
        account_id: str,
    ) -> PaperAccount:
        row = self._read_one(
            """
            SELECT *
            FROM paper_accounts
            WHERE account_id = ?
            """,
            (account_id,),
        )

        if row is None:
            raise ValueError(
                f"Unknown paper account: {account_id}."
            )

        return self._account_from_row(row)

    def save_signal(
        self,
        *,
        account_id: str,
        symbol: str,
        generated_at: datetime,
        expires_at: datetime,
        strategy: str,
        recommendation: str,
        market_regime: str,
        score: float,
        confidence: float,
        reward_to_risk: float,
        entry_low: object,
        entry_high: object,
        stop_price: object,
        targets: Sequence[object],
        evidence: Sequence[str],
        conflicts: Sequence[str],
        threshold_version: str,
        app_version: str,
        scan_id: str | None = None,
        signal_id: str | None = None,
        created_at: datetime | None = None,
        quote_currency: str | None = None,
    ) -> PersistedSignal:
        at = created_at or _utc_now()

        currency_value = None

        if quote_currency is not None:
            currency_value = str(
                quote_currency
            ).strip().upper()

            if (
                len(currency_value) != 3
                or not currency_value.isalpha()
            ):
                raise ValueError(
                    "quote_currency must be a "
                    "three-letter currency code."
                )

        signal_id = signal_id or _new_id(
            "SIG"
        )

        existing = self._read_one(
            """
            SELECT *
            FROM paper_signals
            WHERE signal_id = ?
            """,
            (signal_id,),
        )

        if existing is not None:
            return self._signal_from_row(existing)

        target_values = tuple(
            money(value)
            for value in targets
        )

        if not target_values:
            raise ValueError(
                "Signal requires at least one target."
            )

        with transaction(
            self.database_path
        ) as connection:
            connection.execute(
                """
                INSERT INTO paper_signals(
                    signal_id,
                    account_id,
                    scan_id,
                    symbol,
                    quote_currency,
                    generated_at,
                    expires_at,
                    strategy,
                    recommendation,
                    market_regime,
                    score,
                    confidence,
                    reward_to_risk,
                    entry_low,
                    entry_high,
                    stop_price,
                    targets_json,
                    evidence_json,
                    conflicts_json,
                    threshold_version,
                    app_version,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    signal_id,
                    account_id,
                    scan_id,
                    symbol.strip().upper(),
                    currency_value,
                    _timestamp(generated_at),
                    _timestamp(expires_at),
                    strategy.strip(),
                    recommendation.strip().upper(),
                    market_regime.strip().upper(),
                    float(score),
                    float(confidence),
                    float(reward_to_risk),
                    str(money(entry_low)),
                    str(money(entry_high)),
                    str(money(stop_price)),
                    _json_dump(
                        [
                            str(value)
                            for value
                            in target_values
                        ]
                    ),
                    _json_dump(list(evidence)),
                    _json_dump(list(conflicts)),
                    threshold_version.strip(),
                    app_version.strip(),
                    _timestamp(at),
                ),
            )

            _insert_event(
                connection,
                account_id=account_id,
                event_type="SIGNAL_PERSISTED",
                severity="INFO",
                reference_type="SIGNAL",
                reference_id=signal_id,
                message=(
                    f"{symbol.upper()} "
                    f"{recommendation.upper()} "
                    "signal persisted."
                ),
                metadata={
                    "strategy": strategy,
                    "score": float(score),
                    "confidence":
                    float(confidence),
                    "reward_to_risk":
                    float(reward_to_risk),
                    "quote_currency":
                    currency_value,
                },
                created_at=at,
            )

        return self.get_signal(signal_id)

    def get_signal(
        self,
        signal_id: str,
    ) -> PersistedSignal:
        row = self._read_one(
            """
            SELECT *
            FROM paper_signals
            WHERE signal_id = ?
            """,
            (signal_id,),
        )

        if row is None:
            raise ValueError(
                f"Unknown signal: {signal_id}."
            )

        return self._signal_from_row(row)

    def create_order(
        self,
        *,
        account_id: str,
        signal_id: str,
        idempotency_key: str,
        symbol: str,
        side: PositionSide,
        quantity: object,
        entry_low: object,
        entry_high: object,
        stop_price: object,
        targets: Sequence[object],
        estimated_cash_required: object,
        reserved_cash: object,
        created_at: datetime,
        expires_at: datetime,
        order_id: str | None = None,
    ) -> tuple[PaperOrderRecord, bool]:
        existing = self._read_one(
            """
            SELECT *
            FROM paper_orders
            WHERE account_id = ?
              AND idempotency_key = ?
            """,
            (
                account_id,
                idempotency_key,
            ),
        )

        if existing is not None:
            return (
                self._order_from_row(existing),
                False,
            )

        order_id = order_id or _new_id("ORD")
        quantity_value = money(quantity)
        reserve_value = money(reserved_cash)
        target_values = tuple(
            money(value)
            for value in targets
        )

        with transaction(
            self.database_path
        ) as connection:
            account_row = connection.execute(
                """
                SELECT *
                FROM paper_accounts
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchone()

            if account_row is None:
                raise ValueError(
                    f"Unknown account: {account_id}."
                )

            account = self._account_from_row(
                account_row
            )

            if account.status is not AccountStatus.ACTIVE:
                raise ValueError(
                    "Paper account is not active."
                )

            if reserve_value > account.available_cash:
                raise ValueError(
                    "Insufficient available cash "
                    "for reservation."
                )

            connection.execute(
                """
                INSERT INTO paper_orders(
                    order_id,
                    account_id,
                    signal_id,
                    idempotency_key,
                    symbol,
                    side,
                    quantity,
                    entry_low,
                    entry_high,
                    stop_price,
                    targets_json,
                    estimated_cash_required,
                    reserved_cash,
                    status,
                    created_at,
                    expires_at,
                    filled_at,
                    closed_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, NULL, NULL
                )
                """,
                (
                    order_id,
                    account_id,
                    signal_id,
                    idempotency_key,
                    symbol.upper(),
                    side.value,
                    str(quantity_value),
                    str(money(entry_low)),
                    str(money(entry_high)),
                    str(money(stop_price)),
                    _json_dump(
                        [
                            str(value)
                            for value
                            in target_values
                        ]
                    ),
                    str(
                        money(
                            estimated_cash_required
                        )
                    ),
                    str(reserve_value),
                    OrderStatus.PENDING.value,
                    _timestamp(created_at),
                    _timestamp(expires_at),
                ),
            )

            new_reserved = money(
                account.reserved_cash
                + reserve_value
            )

            connection.execute(
                """
                UPDATE paper_accounts
                SET reserved_cash = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    str(new_reserved),
                    _timestamp(created_at),
                    account_id,
                ),
            )

            _insert_event(
                connection,
                account_id=account_id,
                event_type="ORDER_CREATED",
                severity="INFO",
                reference_type="ORDER",
                reference_id=order_id,
                message=(
                    f"Pending paper BUY order created "
                    f"for {symbol.upper()}."
                ),
                metadata={
                    "quantity":
                    str(quantity_value),
                    "reserved_cash":
                    str(reserve_value),
                    "idempotency_key":
                    idempotency_key,
                },
                created_at=created_at,
            )

        return self.get_order(order_id), True

    def get_order(
        self,
        order_id: str,
    ) -> PaperOrderRecord:
        row = self._read_one(
            """
            SELECT *
            FROM paper_orders
            WHERE order_id = ?
            """,
            (order_id,),
        )

        if row is None:
            raise ValueError(
                f"Unknown order: {order_id}."
            )

        return self._order_from_row(row)

    def cancel_order(
        self,
        order_id: str,
        *,
        cancelled_at: datetime,
        reason: str,
    ) -> PaperOrderRecord:
        with transaction(
            self.database_path
        ) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM paper_orders
                WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()

            if row is None:
                raise ValueError(
                    f"Unknown order: {order_id}."
                )

            order = self._order_from_row(row)

            if order.status is OrderStatus.CANCELLED:
                return order

            if order.status is not OrderStatus.PENDING:
                raise ValueError(
                    "Only pending orders can be cancelled."
                )

            account = self._account_from_row(
                connection.execute(
                    """
                    SELECT *
                    FROM paper_accounts
                    WHERE account_id = ?
                    """,
                    (order.account_id,),
                ).fetchone()
            )

            new_reserved = money(
                account.reserved_cash
                - order.reserved_cash
            )

            if new_reserved < 0:
                raise RuntimeError(
                    "Reserved cash would become negative."
                )

            connection.execute(
                """
                UPDATE paper_orders
                SET status = ?,
                    closed_at = ?
                WHERE order_id = ?
                """,
                (
                    OrderStatus.CANCELLED.value,
                    _timestamp(cancelled_at),
                    order_id,
                ),
            )

            connection.execute(
                """
                UPDATE paper_accounts
                SET reserved_cash = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    str(new_reserved),
                    _timestamp(cancelled_at),
                    order.account_id,
                ),
            )

            _insert_event(
                connection,
                account_id=order.account_id,
                event_type="ORDER_CANCELLED",
                severity="INFO",
                reference_type="ORDER",
                reference_id=order_id,
                message=reason,
                metadata={},
                created_at=cancelled_at,
            )

        return self.get_order(order_id)

    def record_fill_and_open_position(
        self,
        order_id: str,
        *,
        fill_price: object,
        fees: object,
        slippage: object,
        filled_at: datetime,
        fill_id: str | None = None,
        position_id: str | None = None,
    ) -> tuple[
        PaperFillRecord,
        PaperPositionRecord,
    ]:
        with transaction(
            self.database_path
        ) as connection:
            order_row = connection.execute(
                """
                SELECT *
                FROM paper_orders
                WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()

            if order_row is None:
                raise ValueError(
                    f"Unknown order: {order_id}."
                )

            order = self._order_from_row(
                order_row
            )

            existing_fill = connection.execute(
                """
                SELECT *
                FROM paper_fills
                WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()

            existing_position = connection.execute(
                """
                SELECT *
                FROM paper_positions
                WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()

            if (
                existing_fill is not None
                and existing_position is not None
            ):
                return (
                    self._fill_from_row(
                        existing_fill
                    ),
                    self._position_from_row(
                        existing_position
                    ),
                )

            if order.status is not OrderStatus.PENDING:
                raise ValueError(
                    "Only pending orders can be filled."
                )

            if not (
                filled_at > order.created_at
                and filled_at < order.expires_at
            ):
                raise ValueError(
                    "Fill must occur after order creation "
                    "and before expiry."
                )

            if order.side is not PositionSide.LONG:
                raise ValueError(
                    "P3.1 supports long paper positions only."
                )

            price = money(fill_price)
            fee_value = money(fees)
            slippage_value = money(slippage)

            cash_required = calculate_entry_cash(
                price,
                order.quantity,
                fee_value,
            )

            account = self._account_from_row(
                connection.execute(
                    """
                    SELECT *
                    FROM paper_accounts
                    WHERE account_id = ?
                    """,
                    (order.account_id,),
                ).fetchone()
            )

            if cash_required > account.cash_balance:
                raise ValueError(
                    "Insufficient cash for fill."
                )

            new_cash = money(
                account.cash_balance
                - cash_required
            )

            new_reserved = money(
                account.reserved_cash
                - order.reserved_cash
            )

            if new_reserved < 0:
                raise RuntimeError(
                    "Reserved cash would become negative."
                )

            fill_id = fill_id or _new_id("FILL")
            position_id = (
                position_id
                or _new_id("POS")
            )

            connection.execute(
                """
                INSERT INTO paper_fills(
                    fill_id,
                    order_id,
                    price,
                    quantity,
                    fees,
                    slippage,
                    filled_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    fill_id,
                    order_id,
                    str(price),
                    str(order.quantity),
                    str(fee_value),
                    str(slippage_value),
                    _timestamp(filled_at),
                ),
            )

            connection.execute(
                """
                INSERT INTO paper_positions(
                    position_id,
                    account_id,
                    order_id,
                    fill_id,
                    symbol,
                    side,
                    quantity,
                    entry_price,
                    stop_price,
                    targets_json,
                    opened_at,
                    expires_at,
                    status,
                    closed_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, NULL
                )
                """,
                (
                    position_id,
                    order.account_id,
                    order.order_id,
                    fill_id,
                    order.symbol,
                    order.side.value,
                    str(order.quantity),
                    str(price),
                    str(order.stop_price),
                    _json_dump(
                        [
                            str(value)
                            for value
                            in order.targets
                        ]
                    ),
                    _timestamp(filled_at),
                    _timestamp(order.expires_at),
                    PositionStatus.OPEN.value,
                ),
            )

            connection.execute(
                """
                UPDATE paper_orders
                SET status = ?,
                    filled_at = ?
                WHERE order_id = ?
                """,
                (
                    OrderStatus.FILLED.value,
                    _timestamp(filled_at),
                    order_id,
                ),
            )

            connection.execute(
                """
                UPDATE paper_accounts
                SET cash_balance = ?,
                    reserved_cash = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    str(new_cash),
                    str(new_reserved),
                    _timestamp(filled_at),
                    order.account_id,
                ),
            )

            connection.execute(
                """
                INSERT INTO paper_ledger_entries(
                    ledger_id,
                    account_id,
                    event_type,
                    amount,
                    balance_after,
                    reference_type,
                    reference_id,
                    description,
                    occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_id("LED"),
                    order.account_id,
                    "PAPER_BUY",
                    str(-cash_required),
                    str(new_cash),
                    "FILL",
                    fill_id,
                    (
                        f"Paper purchase of "
                        f"{order.quantity} {order.symbol}."
                    ),
                    _timestamp(filled_at),
                ),
            )

            _insert_event(
                connection,
                account_id=order.account_id,
                event_type="ORDER_FILLED",
                severity="INFO",
                reference_type="FILL",
                reference_id=fill_id,
                message=(
                    f"{order.symbol} paper BUY filled."
                ),
                metadata={
                    "price": str(price),
                    "quantity":
                    str(order.quantity),
                    "fees": str(fee_value),
                    "slippage":
                    str(slippage_value),
                    "position_id":
                    position_id,
                },
                created_at=filled_at,
            )

            _insert_event(
                connection,
                account_id=order.account_id,
                event_type="POSITION_OPENED",
                severity="INFO",
                reference_type="POSITION",
                reference_id=position_id,
                message=(
                    f"{order.symbol} paper position opened."
                ),
                metadata={
                    "order_id": order_id,
                    "fill_id": fill_id,
                },
                created_at=filled_at,
            )

        return (
            self.get_fill_for_order(order_id),
            self.get_position_by_order(order_id),
        )

    def get_fill_for_order(
        self,
        order_id: str,
    ) -> PaperFillRecord:
        row = self._read_one(
            """
            SELECT *
            FROM paper_fills
            WHERE order_id = ?
            """,
            (order_id,),
        )

        if row is None:
            raise ValueError(
                f"No fill exists for order: {order_id}."
            )

        return self._fill_from_row(row)

    def get_position(
        self,
        position_id: str,
    ) -> PaperPositionRecord:
        row = self._read_one(
            """
            SELECT *
            FROM paper_positions
            WHERE position_id = ?
            """,
            (position_id,),
        )

        if row is None:
            raise ValueError(
                f"Unknown position: {position_id}."
            )

        return self._position_from_row(row)

    def get_position_by_order(
        self,
        order_id: str,
    ) -> PaperPositionRecord:
        row = self._read_one(
            """
            SELECT *
            FROM paper_positions
            WHERE order_id = ?
            """,
            (order_id,),
        )

        if row is None:
            raise ValueError(
                f"No position exists for order: {order_id}."
            )

        return self._position_from_row(row)

    def list_open_positions(
        self,
        account_id: str,
    ) -> tuple[PaperPositionRecord, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM paper_positions
                WHERE account_id = ?
                  AND status = ?
                ORDER BY opened_at, position_id
                """,
                (
                    account_id,
                    PositionStatus.OPEN.value,
                ),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._position_from_row(row)
            for row in rows
        )

    def close_position(
        self,
        position_id: str,
        *,
        exit_price: object,
        exit_fees: object,
        exit_slippage: object,
        exit_reason: PaperExitReason,
        closed_at: datetime,
        trade_id: str | None = None,
    ) -> ClosedPaperTrade:
        with transaction(
            self.database_path
        ) as connection:
            position_row = connection.execute(
                """
                SELECT *
                FROM paper_positions
                WHERE position_id = ?
                """,
                (position_id,),
            ).fetchone()

            if position_row is None:
                raise ValueError(
                    f"Unknown position: {position_id}."
                )

            position = self._position_from_row(
                position_row
            )

            existing_trade = connection.execute(
                """
                SELECT *
                FROM paper_closed_trades
                WHERE position_id = ?
                """,
                (position_id,),
            ).fetchone()

            if existing_trade is not None:
                return self._trade_from_row(
                    existing_trade
                )

            if position.status is not PositionStatus.OPEN:
                raise ValueError(
                    "Only open positions can be closed."
                )

            if closed_at <= position.opened_at:
                raise ValueError(
                    "Position must close after it opens."
                )

            if position.side is not PositionSide.LONG:
                raise ValueError(
                    "P3.1 supports long paper positions only."
                )

            order = self._order_from_row(
                connection.execute(
                    """
                    SELECT *
                    FROM paper_orders
                    WHERE order_id = ?
                    """,
                    (position.order_id,),
                ).fetchone()
            )

            fill = self._fill_from_row(
                connection.execute(
                    """
                    SELECT *
                    FROM paper_fills
                    WHERE fill_id = ?
                    """,
                    (position.fill_id,),
                ).fetchone()
            )

            signal = self._signal_from_row(
                connection.execute(
                    """
                    SELECT *
                    FROM paper_signals
                    WHERE signal_id = ?
                    """,
                    (order.signal_id,),
                ).fetchone()
            )

            account = self._account_from_row(
                connection.execute(
                    """
                    SELECT *
                    FROM paper_accounts
                    WHERE account_id = ?
                    """,
                    (position.account_id,),
                ).fetchone()
            )

            calculation = calculate_long_trade(
                entry_price=position.entry_price,
                exit_price=exit_price,
                quantity=position.quantity,
                entry_fees=fill.fees,
                exit_fees=exit_fees,
                entry_slippage=fill.slippage,
                exit_slippage=exit_slippage,
            )

            new_cash = money(
                account.cash_balance
                + calculation.cash_proceeds
            )

            trade_id = trade_id or _new_id(
                "TRD"
            )

            holding_seconds = int(
                (
                    closed_at
                    - position.opened_at
                ).total_seconds()
            )

            connection.execute(
                """
                INSERT INTO paper_closed_trades(
                    trade_id,
                    position_id,
                    account_id,
                    order_id,
                    fill_id,
                    signal_id,
                    symbol,
                    strategy,
                    market_regime,
                    entry_time,
                    entry_price,
                    exit_time,
                    exit_price,
                    exit_reason,
                    quantity,
                    gross_pnl,
                    fees,
                    slippage,
                    net_pnl,
                    return_pct,
                    holding_seconds
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    trade_id,
                    position.position_id,
                    position.account_id,
                    position.order_id,
                    position.fill_id,
                    order.signal_id,
                    position.symbol,
                    signal.strategy,
                    signal.market_regime,
                    _timestamp(position.opened_at),
                    str(position.entry_price),
                    _timestamp(closed_at),
                    str(money(exit_price)),
                    exit_reason.value,
                    str(position.quantity),
                    str(calculation.gross_pnl),
                    str(calculation.total_fees),
                    str(
                        calculation.total_slippage
                    ),
                    str(calculation.net_pnl),
                    calculation.return_pct,
                    holding_seconds,
                ),
            )

            connection.execute(
                """
                UPDATE paper_positions
                SET status = ?,
                    closed_at = ?
                WHERE position_id = ?
                """,
                (
                    PositionStatus.CLOSED.value,
                    _timestamp(closed_at),
                    position_id,
                ),
            )

            connection.execute(
                """
                UPDATE paper_orders
                SET status = ?,
                    closed_at = ?
                WHERE order_id = ?
                """,
                (
                    OrderStatus.CLOSED.value,
                    _timestamp(closed_at),
                    order.order_id,
                ),
            )

            connection.execute(
                """
                UPDATE paper_accounts
                SET cash_balance = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    str(new_cash),
                    _timestamp(closed_at),
                    position.account_id,
                ),
            )

            connection.execute(
                """
                INSERT INTO paper_ledger_entries(
                    ledger_id,
                    account_id,
                    event_type,
                    amount,
                    balance_after,
                    reference_type,
                    reference_id,
                    description,
                    occurred_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _new_id("LED"),
                    position.account_id,
                    "PAPER_SELL",
                    str(
                        calculation.cash_proceeds
                    ),
                    str(new_cash),
                    "TRADE",
                    trade_id,
                    (
                        f"Paper sale of "
                        f"{position.quantity} "
                        f"{position.symbol}."
                    ),
                    _timestamp(closed_at),
                ),
            )

            _insert_event(
                connection,
                account_id=position.account_id,
                event_type="POSITION_CLOSED",
                severity="INFO",
                reference_type="TRADE",
                reference_id=trade_id,
                message=(
                    f"{position.symbol} paper position "
                    f"closed with {exit_reason.value}."
                ),
                metadata={
                    "gross_pnl":
                    str(calculation.gross_pnl),
                    "net_pnl":
                    str(calculation.net_pnl),
                    "return_pct":
                    calculation.return_pct,
                },
                created_at=closed_at,
            )

        return self.get_closed_trade_by_position(
            position_id
        )

    def get_closed_trade_by_position(
        self,
        position_id: str,
    ) -> ClosedPaperTrade:
        row = self._read_one(
            """
            SELECT *
            FROM paper_closed_trades
            WHERE position_id = ?
            """,
            (position_id,),
        )

        if row is None:
            raise ValueError(
                f"No closed trade for position: "
                f"{position_id}."
            )

        return self._trade_from_row(row)

    def list_closed_trades(
        self,
        account_id: str,
    ) -> tuple[ClosedPaperTrade, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM paper_closed_trades
                WHERE account_id = ?
                ORDER BY exit_time, trade_id
                """,
                (account_id,),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._trade_from_row(row)
            for row in rows
        )

    def queue_notification(
        self,
        *,
        account_id: str,
        event_type: str,
        reference_type: str,
        reference_id: str,
        channel: NotificationChannel,
        payload: Mapping[str, object],
        created_at: datetime | None = None,
    ) -> NotificationRecord:
        at = created_at or _utc_now()

        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT *
                FROM paper_notifications
                WHERE account_id = ?
                  AND event_type = ?
                  AND reference_type = ?
                  AND reference_id = ?
                  AND channel = ?
                """,
                (
                    account_id,
                    event_type,
                    reference_type,
                    reference_id,
                    channel.value,
                ),
            ).fetchone()

            if existing is None:
                notification_id = _new_id(
                    "NOT"
                )

                connection.execute(
                    """
                    INSERT INTO paper_notifications(
                        notification_id,
                        account_id,
                        event_type,
                        reference_type,
                        reference_id,
                        channel,
                        status,
                        payload_json,
                        created_at,
                        sent_at,
                        error_message
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        NULL, NULL
                    )
                    """,
                    (
                        notification_id,
                        account_id,
                        event_type,
                        reference_type,
                        reference_id,
                        channel.value,
                        NotificationStatus.PENDING.value,
                        _json_dump(dict(payload)),
                        _timestamp(at),
                    ),
                )
            else:
                notification_id = (
                    existing["notification_id"]
                )

        return self.get_notification(
            notification_id
        )

    def get_notification(
        self,
        notification_id: str,
    ) -> NotificationRecord:
        row = self._read_one(
            """
            SELECT *
            FROM paper_notifications
            WHERE notification_id = ?
            """,
            (notification_id,),
        )

        if row is None:
            raise ValueError(
                f"Unknown notification: "
                f"{notification_id}."
            )

        return NotificationRecord(
            notification_id=(
                row["notification_id"]
            ),
            account_id=row["account_id"],
            event_type=row["event_type"],
            reference_type=(
                row["reference_type"]
            ),
            reference_id=row["reference_id"],
            channel=NotificationChannel(
                row["channel"]
            ),
            status=NotificationStatus(
                row["status"]
            ),
            payload=json.loads(
                row["payload_json"]
            ),
            created_at=_datetime(
                row["created_at"]
            ),
            sent_at=_datetime(
                row["sent_at"]
            ),
            error_message=(
                row["error_message"]
            ),
        )

    def list_notifications(
        self,
        account_id: str,
    ) -> tuple[NotificationRecord, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT notification_id
                FROM paper_notifications
                WHERE account_id = ?
                ORDER BY created_at, notification_id
                """,
                (account_id,),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self.get_notification(
                row["notification_id"]
            )
            for row in rows
        )

    def list_pending_notifications(
        self,
        account_id: str,
        *,
        channels: tuple[
            NotificationChannel,
            ...,
        ] | None = None,
        include_failed: bool = False,
    ) -> tuple[NotificationRecord, ...]:
        allowed_statuses = {
            NotificationStatus.PENDING,
        }

        if include_failed:
            allowed_statuses.add(
                NotificationStatus.FAILED
            )

        allowed_channels = (
            set(channels)
            if channels is not None
            else None
        )

        return tuple(
            notification
            for notification
            in self.list_notifications(
                account_id
            )
            if (
                notification.status
                in allowed_statuses
                and (
                    allowed_channels is None
                    or notification.channel
                    in allowed_channels
                )
            )
        )

    def mark_notification_sent(
        self,
        notification_id: str,
        *,
        sent_at: datetime,
        provider_message_id: str | None = None,
        delivery_metadata: Mapping[
            str,
            object,
        ] | None = None,
    ) -> NotificationRecord:
        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT notification_id
                FROM paper_notifications
                WHERE notification_id = ?
                """,
                (notification_id,),
            ).fetchone()

            if existing is None:
                raise ValueError(
                    "Unknown notification: "
                    f"{notification_id}."
                )

            connection.execute(
                """
                UPDATE paper_notifications
                SET status = ?,
                    sent_at = ?,
                    error_message = NULL,
                    attempt_count =
                        attempt_count + 1,
                    last_attempt_at = ?,
                    provider_message_id = ?,
                    delivery_metadata_json = ?
                WHERE notification_id = ?
                """,
                (
                    NotificationStatus.SENT.value,
                    _timestamp(sent_at),
                    _timestamp(sent_at),
                    (
                        str(
                            provider_message_id
                        ).strip()
                        if provider_message_id
                        else None
                    ),
                    _json_dump(
                        dict(
                            delivery_metadata
                            or {}
                        )
                    ),
                    notification_id,
                ),
            )

        return self.get_notification(
            notification_id
        )

    def mark_notification_failed(
        self,
        notification_id: str,
        *,
        attempted_at: datetime,
        error_message: str,
        delivery_metadata: Mapping[
            str,
            object,
        ] | None = None,
    ) -> NotificationRecord:
        message = str(error_message).strip()

        if not message:
            raise ValueError(
                "error_message cannot be empty."
            )

        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT notification_id
                FROM paper_notifications
                WHERE notification_id = ?
                """,
                (notification_id,),
            ).fetchone()

            if existing is None:
                raise ValueError(
                    "Unknown notification: "
                    f"{notification_id}."
                )

            connection.execute(
                """
                UPDATE paper_notifications
                SET status = ?,
                    sent_at = NULL,
                    error_message = ?,
                    attempt_count =
                        attempt_count + 1,
                    last_attempt_at = ?,
                    provider_message_id = NULL,
                    delivery_metadata_json = ?
                WHERE notification_id = ?
                """,
                (
                    NotificationStatus
                                       .FAILED.value,
                    message,
                    _timestamp(attempted_at),
                    _json_dump(
                        dict(
                            delivery_metadata
                            or {}
                        )
                    ),
                    notification_id,
                ),
            )

        return self.get_notification(
            notification_id
        )

    def requeue_notification(
        self,
        notification_id: str,
    ) -> NotificationRecord:
        with transaction(
            self.database_path
        ) as connection:
            existing = connection.execute(
                """
                SELECT status
                FROM paper_notifications
                WHERE notification_id = ?
                """,
                (notification_id,),
            ).fetchone()

            if existing is None:
                raise ValueError(
                    "Unknown notification: "
                    f"{notification_id}."
                )

            if (
                existing["status"]
                == NotificationStatus.SENT.value
            ):
                raise ValueError(
                    "A sent notification cannot "
                    "be requeued."
                )

            connection.execute(
                """
                UPDATE paper_notifications
                SET status = ?,
                    sent_at = NULL,
                    error_message = NULL
                WHERE notification_id = ?
                """,
                (
                    NotificationStatus
                    .PENDING.value,
                    notification_id,
                ),
            )

        return self.get_notification(
            notification_id
        )

    def list_system_events(
        self,
        account_id: str,
    ) -> tuple[SystemEventRecord, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM paper_system_events
                WHERE account_id = ?
                ORDER BY created_at, event_id
                """,
                (account_id,),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            SystemEventRecord(
                event_id=row["event_id"],
                account_id=row["account_id"],
                event_type=row["event_type"],
                severity=row["severity"],
                reference_type=(
                    row["reference_type"]
                ),
                reference_id=(
                    row["reference_id"]
                ),
                message=row["message"],
                metadata=json.loads(
                    row["metadata_json"]
                ),
                created_at=_datetime(
                    row["created_at"]
                ),
            )
            for row in rows
        )

    def list_pending_orders(
        self,
        account_id: str,
    ) -> tuple[PaperOrderRecord, ...]:
        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM paper_orders
                WHERE account_id = ?
                  AND status = ?
                ORDER BY created_at, order_id
                """,
                (
                    account_id,
                    OrderStatus.PENDING.value,
                ),
            ).fetchall()
        finally:
            connection.close()

        return tuple(
            self._order_from_row(row)
            for row in rows
        )

    def expire_order(
        self,
        order_id: str,
        *,
        expired_at: datetime,
        reason: str,
    ) -> PaperOrderRecord:
        with transaction(
            self.database_path
        ) as connection:
            row = connection.execute(
                """
                SELECT *
                FROM paper_orders
                WHERE order_id = ?
                """,
                (order_id,),
            ).fetchone()

            if row is None:
                raise ValueError(
                    f"Unknown order: {order_id}."
                )

            order = self._order_from_row(row)

            if order.status is OrderStatus.EXPIRED:
                return order

            if order.status is not OrderStatus.PENDING:
                raise ValueError(
                    "Only pending orders can expire."
                )

            account = self._account_from_row(
                connection.execute(
                    """
                    SELECT *
                    FROM paper_accounts
                    WHERE account_id = ?
                    """,
                    (order.account_id,),
                ).fetchone()
            )

            new_reserved = money(
                account.reserved_cash
                - order.reserved_cash
            )

            if new_reserved < 0:
                raise RuntimeError(
                    "Reserved cash would become negative."
                )

            connection.execute(
                """
                UPDATE paper_orders
                SET status = ?,
                    closed_at = ?
                WHERE order_id = ?
                """,
                (
                    OrderStatus.EXPIRED.value,
                    _timestamp(expired_at),
                    order_id,
                ),
            )

            connection.execute(
                """
                UPDATE paper_accounts
                SET reserved_cash = ?,
                    updated_at = ?
                WHERE account_id = ?
                """,
                (
                    str(new_reserved),
                    _timestamp(expired_at),
                    order.account_id,
                ),
            )

            _insert_event(
                connection,
                account_id=order.account_id,
                event_type="ORDER_EXPIRED",
                severity="INFO",
                reference_type="ORDER",
                reference_id=order_id,
                message=reason,
                metadata={},
                created_at=expired_at,
            )

        return self.get_order(order_id)

    def record_system_event(
        self,
        *,
        event_type: str,
        message: str,
        severity: str = "INFO",
        account_id: str | None = None,
        reference_type: str | None = None,
        reference_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        at = created_at or _utc_now()

        with transaction(
            self.database_path
        ) as connection:
            _insert_event(
                connection,
                account_id=account_id,
                event_type=event_type,
                severity=severity,
                reference_type=reference_type,
                reference_id=reference_id,
                message=message,
                metadata=metadata,
                created_at=at,
            )

    def reconcile_account(
        self,
        account_id: str,
    ) -> AccountReconciliation:
        account = self.get_account(
            account_id
        )

        connection = connect_database(
            self.database_path
        )

        try:
            rows = connection.execute(
                """
                SELECT amount
                FROM paper_ledger_entries
                WHERE account_id = ?
                """,
                (account_id,),
            ).fetchall()
        finally:
            connection.close()

        ledger_balance = money(
            sum(
                (
                    money(row["amount"])
                    for row in rows
                ),
                Decimal("0"),
            )
        )

        difference = money(
            account.cash_balance
            - ledger_balance
        )

        return AccountReconciliation(
            account_id=account_id,
            stored_cash_balance=(
                account.cash_balance
            ),
            ledger_cash_balance=(
                ledger_balance
            ),
            difference=difference,
        )
