"""Portfolio-level paper-trading service."""

from __future__ import annotations

from src.product_config import load_product_policy
from src.strategy import (
    StrategyHorizon,
    horizon_policies_from_product_policy,
)

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
from .fx import (
    FXRateError,
    FXRateProvider,
    QuoteToPortfolioFXRate,
    identity_fx_rate,
)
from .repository import PaperRepository
from .valuation import (
    calculate_entry_cash_portfolio,
)


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
        fx_rate_provider: FXRateProvider | None = None,
        app_version: str = "v0.2.0-p2",
        threshold_version: str = "schema-1",
    ) -> None:
        self.repository = repository
        self.config = (
            config or PaperPortfolioConfig()
        )
        self.fx_rate_provider = (
            fx_rate_provider
        )
        self.app_version = app_version
        self.threshold_version = (
            threshold_version
        )

    def _resolve_fx_rate(
        self,
        *,
        quote_currency: str,
        portfolio_currency: str,
        as_of: datetime,
    ) -> QuoteToPortfolioFXRate:
        quote = (
            quote_currency
            .strip()
            .upper()
        )

        portfolio = (
            portfolio_currency
            .strip()
            .upper()
        )

        if quote == portfolio:
            return identity_fx_rate(
                quote,
                as_of=as_of,
            )

        if self.fx_rate_provider is None:
            raise FXRateError(
                "No FX rate provider is "
                f"configured for {quote}/"
                f"{portfolio}."
            )

        rate = (
            self.fx_rate_provider
            .get_rate(
                quote_currency=quote,
                portfolio_currency=(
                    portfolio
                ),
                as_of=as_of,
            )
        )

        if (
            rate.quote_currency != quote
            or rate.portfolio_currency
            != portfolio
        ):
            raise FXRateError(
                "FX provider returned a rate "
                "for the wrong currency pair."
            )

        return rate

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
        quote_currency: str | None = None,
        strategy_horizon: (
            StrategyHorizon | str | None
        ) = None,
        strategy_version: str | None = None,
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
            quote_currency=quote_currency,
            generated_at=generated_at,
            expires_at=expires_at,
            strategy=strategy,
            strategy_horizon=(
                strategy_horizon
            ),
            strategy_version=(
                strategy_version
            ),
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

        quote_currency = (
            signal.quote_currency
            or account.base_currency
        )

        reservation_fx = (
            self._resolve_fx_rate(
                quote_currency=quote_currency,
                portfolio_currency=(
                    account.base_currency
                ),
                as_of=at,
            )
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

        entry_cash = (
            calculate_entry_cash_portfolio(
                price_quote=signal.entry_high,
                quantity=quantity_value,
                fee_quote=estimated_fees,
                fx_rate=reservation_fx,
            )
        )

        estimated_cash = (
            entry_cash.cash_required_portfolio
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

        risk_per_unit_quote = money(
            signal.entry_high
            - signal.stop_price
        )

        if risk_per_unit_quote <= 0:
            raise ValueError(
                "BUY signal requires a stop "
                "below the entry zone."
            )

        new_trade_risk = (
            reservation_fx
            .convert_quote_to_portfolio(
                money(
                    risk_per_unit_quote
                    * quantity_value
                )
            )
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

        existing_open_risk = Decimal("0")

        for position in open_positions:
            position_risk_quote = money(
                (
                    position.entry_price
                    - position.stop_price
                )
                * position.quantity
            )

            if position.entry_fx_rate is not None:
                position_risk = money(
                    position_risk_quote
                    * position.entry_fx_rate
                )

            elif (
                position.quote_currency
                in {
                    None,
                    account.base_currency,
                }
                and position.portfolio_currency
                in {
                    None,
                    account.base_currency,
                }
            ):
                position_risk = (
                    position_risk_quote
                )

            else:
                raise ValueError(
                    "Open position is missing "
                    "entry FX provenance."
                )

            existing_open_risk = money(
                existing_open_risk
                + position_risk
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
            strategy_horizon=(
                signal.strategy_horizon
            ),
            strategy_version=(
                signal.strategy_version
            ),
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
            reservation_fx_rate=(
                reservation_fx
            ),
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
        maximum_holding_sessions: int | None = None,
    ) -> tuple[
        PaperFillRecord,
        PaperPositionRecord,
    ]:
        at = filled_at or self._now()

        order = self.repository.get_order(
            order_id
        )

        account = self.repository.get_account(
            order.account_id
        )

        holding_sessions = (
            maximum_holding_sessions
        )

        if (
            holding_sessions is None
            and order.strategy_horizon is not None
        ):
            policies = (
                horizon_policies_from_product_policy(
                    load_product_policy()
                )
            )
            holding_sessions = (
                policies[
                    order.strategy_horizon
                ].maximum_holding_sessions
            )

        quote_currency = (
            order.quote_currency
            or account.base_currency
        )

        entry_fx = self._resolve_fx_rate(
            quote_currency=quote_currency,
            portfolio_currency=(
                account.base_currency
            ),
            as_of=at,
        )

        fill, position = (
            self.repository
            .record_fill_and_open_position(
                order_id,
                fill_price=fill_price,
                fees=fees,
                slippage=slippage,
                entry_fx_rate=entry_fx,
                filled_at=at,
                maximum_holding_sessions=(
                    holding_sessions
                ),
            )
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
                "quote_currency":
                fill.quote_currency,
                "portfolio_currency":
                fill.portfolio_currency,
                "entry_fx_rate": (
                    str(fill.entry_fx_rate)
                    if fill.entry_fx_rate
                    is not None
                    else None
                ),
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

        position = self.repository.get_position(
            position_id
        )

        account = self.repository.get_account(
            position.account_id
        )

        quote_currency = (
            position.quote_currency
            or account.base_currency
        )

        exit_fx = self._resolve_fx_rate(
            quote_currency=quote_currency,
            portfolio_currency=(
                account.base_currency
            ),
            as_of=at,
        )

        trade = self.repository.close_position(
            position_id,
            exit_price=exit_price,
            exit_fees=exit_fees,
            exit_slippage=exit_slippage,
            exit_fx_rate=exit_fx,
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
                "quote_currency":
                trade.quote_currency,
                "portfolio_currency":
                trade.portfolio_currency,
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
