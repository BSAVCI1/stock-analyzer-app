"""Persistent automated paper-portfolio package."""

from .database import (
    DEFAULT_DATABASE_PATH,
    connect_database,
    transaction,
)
from .fx import (
    FXRateError,
    FXRateProvider,
    QuoteToPortfolioFXRate,
    StaticFXRateProvider,
    identity_fx_rate,
)
from .yahoo_fx import (
    YahooFXRateProvider,
)
from .ledger import (
    LongTradeCalculation,
    calculate_entry_cash,
    calculate_long_trade,
)
from .migrations import (
    SCHEMA_VERSION,
    apply_migrations,
    initialize_database,
)
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
from .repository import PaperRepository
from .service import (
    PaperPortfolioConfig,
    PaperTradingService,
)
from .sizing import (
    FixedNotionalSizingDecision,
    FixedNotionalSizingPolicy,
    FixedNotionalSizingRequest,
    PositionSizingConstraint,
    PositionSizingMode,
    PositionSizingRejected,
    calculate_fixed_notional_size,
    fixed_notional_policy_from_product_policy,
)

__all__ = [
    "DEFAULT_DATABASE_PATH",
    "SCHEMA_VERSION",
    "AccountReconciliation",
    "AccountStatus",
    "ClosedPaperTrade",
    "FixedNotionalSizingDecision",
    "FixedNotionalSizingPolicy",
    "FixedNotionalSizingRequest",
    "FXRateError",
    "FXRateProvider",
    "LongTradeCalculation",
    "NotificationChannel",
    "NotificationRecord",
    "NotificationStatus",
    "OrderStatus",
    "PaperAccount",
    "PaperExitReason",
    "PaperFillRecord",
    "PaperOrderRecord",
    "PaperPortfolioConfig",
    "PaperPositionRecord",
    "PaperRepository",
    "PaperTradingService",
    "PersistedSignal",
    "QuoteToPortfolioFXRate",
    "PositionSizingConstraint",
    "PositionSizingMode",
    "PositionSizingRejected",
    "PositionStatus",
    "StaticFXRateProvider",
    "SystemEventRecord",
    "YahooFXRateProvider",
    "apply_migrations",
    "calculate_entry_cash",
    "calculate_fixed_notional_size",
    "calculate_long_trade",
    "connect_database",
    "fixed_notional_policy_from_product_policy",
    "identity_fx_rate",
    "initialize_database",
    "money",
    "transaction",
]
