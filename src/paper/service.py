"""Portfolio-level paper-trading service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from math import isfinite
from typing import Sequence

from src.backtest import PositionSide

from .models import (
    NotificationChannel,
    PaperAccount,
    PaperExitReason,
    PaperFillRecord,
    PaperOrderRecord,
    PaperPositionRecord,
    PersistedSignal,
    ClosedPaperTrade,
    money,
)
from .repository import PaperRepository


@dataclass(frozen=True, slots=True)
class PaperPortfolioConfig:
    starting_balance: Decimal = Decimal("10000")
    base_currency: str = "USD"

    maximum_open_positions: int = 5
    maximum_allocation_fraction: Decimal = Decimal("0.20")
    risk_fraction_per_trade: Decimal = Decimal("0.01")
    maximum_open_risk_fraction: Decimal = Decimal("0.04")
    minimum_reward_to_risk: Decimal = Decimal("2.0")

    def __post_init__(self) -> None:
        if (
            isinstance(
                self.maximum_open_positions,
                bool,
            )
            or not isinstance(
                self.maximum_open_positions,
                int,
            )
            or self.maximum_open_positions < 1
        ):
            raise ValueError(
                "maximum_open_positions must be positive."
            )

        for name in (
            "maximum_allocation_fraction",
            "risk_fraction_per_trade",
            "maximum_open_risk_fraction",
        ):
            value = money(
                getattr(self, name)
            )

            if not 0 < value <= 1:
                raise ValueError(
                    f"{name} must be between 0 and 1."
                )

            object.__setattr__(
                self,
                name,
                value,
            )

        minimum_reward_to_risk = money(
            self.minimum_reward_to_risk
        )

        if minimum_reward_to_risk <= 0:
            raise ValueError(
                "minimum_reward_to_risk must be positive."
            )

        object.__setattr__(
            self,
            "starting_balance",
            money(self.starting_balance),
        )

        object.__setattr__(
            self,
            "minimum_reward_to_risk",
            minimum_reward_to_risk,
        )

        object.__setattr__(
            self,
            "base_currency",
            self.base_currency.strip().upper(),
        )


class PaperTradingService:
    """Coordinates persistent, long-only automated paper trading."""

    def __init__(
        self,
        repository: PaperRepository,
        *,
        config: PaperPortfolioConfig | None = None,
        app_version: str = "v0.2.0-p2",
        threshold_version: str = "schema-1",
    ) -> None:
        self.repository = repository
        self.config = (
            config or PaperPortfolioConfig()
        )
        self.app_version = app_version
        self.threshold_version = (
            threshold_version
        )

    @staticmethod
    def _now() -> datetime:
        return datetime.now(timezone.utc)

    def create_account(
        self,
        name: str = "Personal Paper Portfolio",
        *,
        created_at: datetime | None = None,
    ) -> PaperAccount:
        return self.repository.create_account(
            name=name,
            base_currency=(
                self.config.base_currency
            ),
            starting_balance=(
                self.config.starting_balance
            ),
            created_at=created_at,
        )

    def persist_signal(
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
        conflicts: Sequence[str] = (),
        signal_id: str | None = None,
        scan_id: str | None = None,
    ) -> PersistedSignal:
        if not isfinite(float(score)):
            raise ValueError("score must be finite.")

        if not 0 <= float(confidence) <= 1:
            raise ValueError(
                "confidence must be between 0 and 1."
            )

        return self.repository.save_signal(
            account_id=account_id,
            scan_id=scan_id,
            symbol=symbol,
            generated_at=generated_at,
            expires_at=expires_at,
            strategy=strategy,
            recommendation=recommendation,
            market_regime=market_regime,
            score=score,
            confidence=confidence,
            reward_to_risk=reward_to_risk,
            entry_low=entry_low,
            entry_high=entry_high,
            stop_price=stop_price,
            targets=targets,
            evidence=evidence,
            conflicts=conflicts,
            threshold_version=(
                self.threshold_version
            ),
            app_version=self.app_version,
            signal_id=signal_id,
            created_at=generated_at,
        )

    def create_automatic_buy(
        self,
        *,
        account_id: str,
        signal_id: str,
        quantity: object,
        idempotency_key: str,
        estimated_fees: object = 0,
        created_at: datetime | None = None,
    ) -> tuple[PaperOrderRecord, bool]:
        at = created_at or self._now()

        account = self.repository.get_account(
            account_id
        )

        signal = self.repository.get_signal(
            signal_id
        )

        if signal.account_id != account_id:
            raise ValueError(
                "Signal belongs to another account."
            )

        if signal.recommendation != "BUY":
            raise ValueError(
                "Only BUY signals can create "
                "automatic paper purchases."
            )

        if at >= signal.expires_at:
            raise ValueError(
                "Signal has already expired."
            )

        reward_to_risk = money(
            signal.reward_to_risk
        )

        if (
            reward_to_risk
            < self.config.minimum_reward_to_risk
        ):
            raise ValueError(
                "Signal reward-to-risk is below "
                "the configured minimum."
            )

        quantity_value = money(quantity)

        if quantity_value <= 0:
            raise ValueError(
                "quantity must be positive."
            )

        open_positions = (
            self.repository.list_open_positions(
                account_id
            )
        )

        if (
            len(open_positions)
            >= self.config.maximum_open_positions
        ):
            raise ValueError(
                "Maximum open-position limit reached."
            )

        if any(
            position.symbol == signal.symbol
            for position in open_positions
        ):
            raise ValueError(
                "An open position already exists "
                f"for {signal.symbol}."
            )

        entry_notional = money(
            signal.entry_high
            * quantity_value
        )

        estimated_cash = money(
            entry_notional
            + money(estimated_fees)
        )

        allocation_limit = money(
            account.cash_balance
            * self.config
            .maximum_allocation_fraction
        )

        if estimated_cash > allocation_limit:
            raise ValueError(
                "Order exceeds maximum allocation "
                "per stock."
            )

        risk_per_unit = money(
            signal.entry_high
            - signal.stop_price
        )

        if risk_per_unit <= 0:
            raise ValueError(
                "BUY signal requires a stop "
                "below the entry zone."
            )

        new_trade_risk = money(
            risk_per_unit
            * quantity_value
        )

        trade_risk_limit = money(
            account.cash_balance
            * self.config
            .risk_fraction_per_trade
        )

        if new_trade_risk > trade_risk_limit:
            raise ValueError(
                "Order exceeds maximum risk per trade."
            )

        existing_open_risk = money(
            sum(
                (
                    money(
                        position.entry_price
                        - position.stop_price
                    )
                    * position.quantity
                    for position in open_positions
                ),
                Decimal("0"),
            )
        )

        combined_open_risk = money(
            existing_open_risk
            + new_trade_risk
        )

        combined_risk_limit = money(
            account.cash_balance
            * self.config
            .maximum_open_risk_fraction
        )

        if combined_open_risk > combined_risk_limit:
            raise ValueError(
                "Order exceeds maximum combined "
                "open portfolio risk."
            )

        return self.repository.create_order(
            account_id=account_id,
            signal_id=signal_id,
            idempotency_key=idempotency_key,
            symbol=signal.symbol,
            side=PositionSide.LONG,
            quantity=quantity_value,
            entry_low=signal.entry_low,
            entry_high=signal.entry_high,
            stop_price=signal.stop_price,
            targets=signal.targets,
            estimated_cash_required=(
                estimated_cash
            ),
            reserved_cash=estimated_cash,
            created_at=at,
            expires_at=signal.expires_at,
        )

    def record_automatic_buy_fill(
        self,
        *,
        order_id: str,
        fill_price: object,
        fees: object = 0,
        slippage: object = 0,
        filled_at: datetime | None = None,
    ) -> tuple[
        PaperFillRecord,
        PaperPositionRecord,
    ]:
        at = filled_at or self._now()

        fill, position = (
            self.repository
            .record_fill_and_open_position(
                order_id,
                fill_price=fill_price,
                fees=fees,
                slippage=slippage,
                filled_at=at,
            )
        )

        order = self.repository.get_order(
            order_id
        )

        self.repository.queue_notification(
            account_id=order.account_id,
            event_type="PAPER_BUY_EXECUTED",
            reference_type="POSITION",
            reference_id=position.position_id,
            channel=NotificationChannel.INTERNAL,
            payload={
                "symbol": position.symbol,
                "quantity": str(
                    position.quantity
                ),
                "fill_price": str(fill.price),
                "filled_at":
                fill.filled_at.isoformat(),
                "stop_price": str(
                    position.stop_price
                ),
                "targets": [
                    str(target)
                    for target in position.targets
                ],
                "order_id": order_id,
                "position_id":
                position.position_id,
            },
            created_at=at,
        )

        return fill, position

    def close_automatic_position(
        self,
        *,
        position_id: str,
        exit_price: object,
        exit_reason: PaperExitReason,
        exit_fees: object = 0,
        exit_slippage: object = 0,
        closed_at: datetime | None = None,
    ) -> ClosedPaperTrade:
        at = closed_at or self._now()

        trade = self.repository.close_position(
            position_id,
            exit_price=exit_price,
            exit_fees=exit_fees,
            exit_slippage=exit_slippage,
            exit_reason=exit_reason,
            closed_at=at,
        )

        self.repository.queue_notification(
            account_id=trade.account_id,
            event_type="PAPER_SELL_EXECUTED",
            reference_type="TRADE",
            reference_id=trade.trade_id,
            channel=NotificationChannel.INTERNAL,
            payload={
                "symbol": trade.symbol,
                "quantity": str(trade.quantity),
                "entry_price": str(
                    trade.entry_price
                ),
                "exit_price": str(
                    trade.exit_price
                ),
                "exit_reason":
                trade.exit_reason.value,
                "gross_pnl": str(
                    trade.gross_pnl
                ),
                "fees": str(trade.fees),
                "slippage": str(
                    trade.slippage
                ),
                "net_pnl": str(
                    trade.net_pnl
                ),
                "return_pct":
                trade.return_pct,
                "entry_time":
                trade.entry_time.isoformat(),
                "exit_time":
                trade.exit_time.isoformat(),
                "trade_id": trade.trade_id,
            },
            created_at=at,
        )

        return trade

    def cancel_pending_order(
        self,
        *,
        order_id: str,
        reason: str,
        cancelled_at: datetime | None = None,
    ) -> PaperOrderRecord:
        return self.repository.cancel_order(
            order_id,
            cancelled_at=(
                cancelled_at
                or self._now()
            ),
            reason=reason,
        )
