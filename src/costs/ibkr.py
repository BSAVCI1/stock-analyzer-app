"""Broker-disconnected IBKR reference-cost economics.

This module is a versioned reference model only.  It does
not authenticate with, connect to, query, or place orders
with Interactive Brokers.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_HALF_UP,
)
from enum import Enum
import json
from pathlib import Path
from typing import Mapping


MONEY_QUANTUM = Decimal("0.00000001")
BASIS_POINTS = Decimal("10000")

DEFAULT_IBKR_REFERENCE_PROFILE = (
    Path(__file__).resolve().parents[2]
    / "config"
    / "ibkr_reference_costs_v2.json"
)


class IBKRCostProfileError(ValueError):
    """Raised when the reference profile is invalid."""


class IBKRPricingPlan(str, Enum):
    TIERED = "TIERED"
    FIXED = "FIXED"


class IBKRTradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class IBKRFXMode(str, Enum):
    SPOT_FX = "SPOT_FX"
    AUTO_CONVERSION = "AUTO_CONVERSION"


@dataclass(frozen=True, slots=True)
class IBKRStockCostEstimate:
    currency: str

    commission: Decimal
    regulatory_fees: Decimal
    clearing_fees: Decimal
    route_dependent_fees: Decimal

    total_known_cost: Decimal
    complete: bool

    def __post_init__(self) -> None:
        currency = str(
            self.currency
        ).strip().upper()

        if (
            len(currency) != 3
            or not currency.isalpha()
        ):
            raise ValueError(
                "currency must be a "
                "three-letter code."
            )

        object.__setattr__(
            self,
            "currency",
            currency,
        )

        for name in (
            "commission",
            "regulatory_fees",
            "clearing_fees",
            "route_dependent_fees",
            "total_known_cost",
        ):
            object.__setattr__(
                self,
                name,
                _money(
                    _non_negative(
                        name,
                        getattr(self, name),
                    )
                ),
            )


@dataclass(frozen=True, slots=True)
class IBKRFXCostEstimate:
    mode: IBKRFXMode
    reference_currency: str
    trade_value: Decimal
    estimated_cost: Decimal
    separate_commission: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.mode,
            IBKRFXMode,
        ):
            raise ValueError(
                "mode must be an IBKRFXMode."
            )

        currency = str(
            self.reference_currency
        ).strip().upper()

        if (
            len(currency) != 3
            or not currency.isalpha()
        ):
            raise ValueError(
                "reference_currency must "
                "be a three-letter code."
            )

        object.__setattr__(
            self,
            "reference_currency",
            currency,
        )

        object.__setattr__(
            self,
            "trade_value",
            _money(
                _positive(
                    "trade_value",
                    self.trade_value,
                )
            ),
        )

        object.__setattr__(
            self,
            "estimated_cost",
            _money(
                _non_negative(
                    "estimated_cost",
                    self.estimated_cost,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class IBKRNetRewardToRisk:
    gross_reward: Decimal
    gross_risk: Decimal
    round_trip_cost: Decimal

    net_reward: Decimal
    cost_adjusted_risk: Decimal
    net_reward_to_risk: Decimal

    def __post_init__(self) -> None:
        for name in (
            "gross_reward",
            "gross_risk",
            "cost_adjusted_risk",
        ):
            object.__setattr__(
                self,
                name,
                _money(
                    _positive(
                        name,
                        getattr(self, name),
                    )
                ),
            )

        for name in (
            "round_trip_cost",
            "net_reward",
            "net_reward_to_risk",
        ):
            value = _non_negative(
                name,
                getattr(self, name),
            )

            object.__setattr__(
                self,
                name,
                _money(value),
            )


def _decimal(
    name: str,
    value: object,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(
            f"{name} must be numeric."
        )

    try:
        result = Decimal(
            str(value)
        )
    except (
        InvalidOperation,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            f"{name} must be numeric."
        ) from exc

    if not result.is_finite():
        raise ValueError(
            f"{name} must be finite."
        )

    return result


def _positive(
    name: str,
    value: object,
) -> Decimal:
    result = _decimal(
        name,
        value,
    )

    if result <= 0:
        raise ValueError(
            f"{name} must be positive."
        )

    return result


def _non_negative(
    name: str,
    value: object,
) -> Decimal:
    result = _decimal(
        name,
        value,
    )

    if result < 0:
        raise ValueError(
            f"{name} must be non-negative."
        )

    return result


def _money(
    value: object,
) -> Decimal:
    return _decimal(
        "money",
        value,
    ).quantize(
        MONEY_QUANTUM,
        rounding=ROUND_HALF_UP,
    )


def _enum(
    enum_type,
    value,
):
    if isinstance(
        value,
        enum_type,
    ):
        return value

    try:
        return enum_type(
            str(value).strip().upper()
        )
    except ValueError as exc:
        raise ValueError(
            f"Unsupported {enum_type.__name__}: "
            f"{value}."
        ) from exc


def _mapping(
    value: object,
    path: str,
) -> Mapping[str, object]:
    if not isinstance(
        value,
        Mapping,
    ):
        raise IBKRCostProfileError(
            f"{path} must be an object."
        )

    return value


def _expect_keys(
    mapping: Mapping[str, object],
    expected: set[str],
    path: str,
) -> None:
    actual = set(mapping)

    missing = expected - actual
    unexpected = actual - expected

    if missing or unexpected:
        parts = []

        if missing:
            parts.append(
                "missing="
                + ",".join(
                    sorted(missing)
                )
            )

        if unexpected:
            parts.append(
                "unexpected="
                + ",".join(
                    sorted(unexpected)
                )
            )

        raise IBKRCostProfileError(
            f"{path}: "
            + "; ".join(parts)
        )


def _reject_sensitive_keys(
    value: object,
    path: str = "$",
) -> None:
    forbidden = (
        "password",
        "secret",
        "credential",
        "api_key",
        "api_token",
        "private_key",
        "access_token",
        "refresh_token",
    )

    if isinstance(
        value,
        Mapping,
    ):
        for key, nested in value.items():
            normalised = (
                str(key)
                .strip()
                .lower()
            )

            if any(
                token in normalised
                for token in forbidden
            ):
                raise IBKRCostProfileError(
                    "Sensitive configuration "
                    f"key at {path}.{key}."
                )

            _reject_sensitive_keys(
                nested,
                f"{path}.{key}",
            )

    elif isinstance(
        value,
        list,
    ):
        for index, nested in enumerate(
            value
        ):
            _reject_sensitive_keys(
                nested,
                f"{path}[{index}]",
            )


def validate_ibkr_reference_profile(
    profile: Mapping[str, object],
) -> None:
    profile = _mapping(
        profile,
        "$",
    )

    _reject_sensitive_keys(profile)

    schema_version = profile.get(
        "schema_version"
    )

    if schema_version not in (1, 2):
        raise IBKRCostProfileError(
            "schema_version must be 1 or 2."
        )

    expected_keys = {
        "schema_version",
        "profile_version",
        "provider",
        "verified_at",
        "api_connection_enabled",
        "active_pricing_plan",
        "active_fx_mode",
        "us_stocks",
        "us_fractional",
        "us_regulatory",
        "europe_eur_reference",
        "fx",
        "sources",
    }

    if schema_version == 2:
        expected_keys.add(
            "operational_assumptions"
        )

    _expect_keys(
        profile,
        expected_keys,
        "$",
    )

    if profile["provider"] != "IBKR":
        raise IBKRCostProfileError(
            "provider must be IBKR."
        )

    if (
        profile["api_connection_enabled"]
        is not False
    ):
        raise IBKRCostProfileError(
            "IBKR API connectivity must "
            "remain disabled."
        )

    # Historical schema v1 remains
    # reference-only and inactive.
    if schema_version == 1:
        for key in (
            "active_pricing_plan",
            "active_fx_mode",
        ):
            if profile[key] is not None:
                raise IBKRCostProfileError(
                    f"{key} must remain null "
                    "for schema v1."
                )

    # Schema v2 contains the confirmed
    # operational pricing plan. FX remains a
    # manual portfolio-funding event.
    if schema_version == 2:
        if (
            profile["active_pricing_plan"]
            != "FIXED"
        ):
            raise IBKRCostProfileError(
                "active_pricing_plan must be "
                "FIXED for schema v2."
            )

        if (
            profile["active_fx_mode"]
            is not None
        ):
            raise IBKRCostProfileError(
                "active_fx_mode must remain "
                "null for manual FX funding."
            )

    if schema_version == 2:
        assumptions = _mapping(
            profile[
                "operational_assumptions"
            ],
            "$.operational_assumptions",
        )

        _expect_keys(
            assumptions,
            {
                "confirmed_pricing_plan",
                "intended_routing",
                "maximum_modeled_order_eur",
                "eur_fixed_minimum_order_eur",
                "entry_fx_conversion_per_trade",
                "exit_fx_conversion_per_trade",
                "fx_conversion_control",
                "usd_sale_proceeds_policy",
                "confirmation_basis",
            },
            "$.operational_assumptions",
        )

        expected_values = {
            "confirmed_pricing_plan":
                "FIXED",
            "intended_routing":
                "IBKR_SMARTROUTING",
            "fx_conversion_control":
                "MANUAL_PORTFOLIO_FUNDING_EVENT",
            "usd_sale_proceeds_policy":
                "RETAIN_USD",
            "confirmation_basis":
                (
                    "USER_CONFIRMED_ACCOUNT_HISTORY_"
                    "AND_OFFICIAL_IBKR"
                ),
        }

        for key, expected in (
            expected_values.items()
        ):
            if assumptions[key] != expected:
                raise IBKRCostProfileError(
                    f"{key} must be "
                    f"{expected!r}."
                )

        maximum_order = _positive(
            "maximum_modeled_order_eur",
            assumptions[
                "maximum_modeled_order_eur"
            ],
        )

        if maximum_order != Decimal("100.00"):
            raise IBKRCostProfileError(
                "maximum_modeled_order_eur "
                "must be 100.00."
            )

        fixed_minimum = _positive(
            "eur_fixed_minimum_order_eur",
            assumptions[
                "eur_fixed_minimum_order_eur"
            ],
        )

        if fixed_minimum != Decimal("3.00"):
            raise IBKRCostProfileError(
                "eur_fixed_minimum_order_eur "
                "must be 3.00."
            )

        for key in (
            "entry_fx_conversion_per_trade",
            "exit_fx_conversion_per_trade",
        ):
            if assumptions[key] is not False:
                raise IBKRCostProfileError(
                    f"{key} must remain false."
                )

    us = _mapping(
        profile["us_stocks"],
        "$.us_stocks",
    )

    _expect_keys(
        us,
        {"tiered", "fixed"},
        "$.us_stocks",
    )

    for name in (
        "tiered",
        "fixed",
    ):
        rule = _mapping(
            us[name],
            f"$.us_stocks.{name}",
        )

        _expect_keys(
            rule,
            {
                "per_share_usd",
                "minimum_order_usd",
                "maximum_trade_value_fraction",
                "route_dependent_fees",
            },
            f"$.us_stocks.{name}",
        )

        _positive(
            f"{name}.per_share_usd",
            rule["per_share_usd"],
        )

        _positive(
            f"{name}.minimum_order_usd",
            rule["minimum_order_usd"],
        )

        cap = _positive(
            f"{name}.maximum_trade_value_fraction",
            rule[
                "maximum_trade_value_fraction"
            ],
        )

        if cap > 1:
            raise IBKRCostProfileError(
                "commission cap cannot "
                "exceed trade value."
            )

    fractional = _mapping(
        profile["us_fractional"],
        "$.us_fractional",
    )

    _expect_keys(
        fractional,
        {
            "commission_trade_value_fraction",
            "minimum_commission_usd",
            "eligible_instruments_only",
            "published_minimum_order_value_usd",
        },
        "$.us_fractional",
    )

    _positive(
        "fractional commission rate",
        fractional[
            "commission_trade_value_fraction"
        ],
    )

    _positive(
        "fractional minimum",
        fractional[
            "minimum_commission_usd"
        ],
    )

    if (
        fractional[
            "eligible_instruments_only"
        ]
        is not True
    ):
        raise IBKRCostProfileError(
            "fractional trading must be "
            "eligibility-gated."
        )

    regulatory = _mapping(
        profile["us_regulatory"],
        "$.us_regulatory",
    )

    _expect_keys(
        regulatory,
        {
            "sec_sale_value_rate",
            "finra_taf_per_share_sale",
            "finra_taf_maximum_usd",
            "cat_per_share",
            "tiered_clearing_per_share",
            "tiered_clearing_maximum_trade_value_fraction",
        },
        "$.us_regulatory",
    )

    for key in regulatory:
        _non_negative(
            key,
            regulatory[key],
        )

    europe = _mapping(
        profile[
            "europe_eur_reference"
        ],
        "$.europe_eur_reference",
    )

    _expect_keys(
        europe,
        {
            "scope",
            "tiered",
            "fixed_smartrouting",
        },
        "$.europe_eur_reference",
    )

    if (
        europe["scope"]
        != "REFERENCE_ONLY_MARKET_SPECIFIC"
    ):
        raise IBKRCostProfileError(
            "Europe profile must remain "
            "explicitly reference-only."
        )

    fx = _mapping(
        profile["fx"],
        "$.fx",
    )

    _expect_keys(
        fx,
        {
            "spot_fx",
            "auto_conversion",
        },
        "$.fx",
    )

    spot = _mapping(
        fx["spot_fx"],
        "$.fx.spot_fx",
    )

    _expect_keys(
        spot,
        {
            "commission_bps",
            "minimum_commission_usd",
        },
        "$.fx.spot_fx",
    )

    _positive(
        "spot FX commission",
        spot["commission_bps"],
    )

    _positive(
        "spot FX minimum",
        spot[
            "minimum_commission_usd"
        ],
    )

    auto = _mapping(
        fx["auto_conversion"],
        "$.fx.auto_conversion",
    )

    _expect_keys(
        auto,
        {
            "rate_adjustment_fraction",
            "separate_commission",
        },
        "$.fx.auto_conversion",
    )

    _positive(
        "auto FX adjustment",
        auto[
            "rate_adjustment_fraction"
        ],
    )

    if (
        auto["separate_commission"]
        is not False
    ):
        raise IBKRCostProfileError(
            "Reference auto-conversion "
            "profile expects no separate "
            "commission."
        )


def load_ibkr_reference_profile(
    path: str | Path = (
        DEFAULT_IBKR_REFERENCE_PROFILE
    ),
) -> dict[str, object]:
    profile_path = Path(path)

    payload = json.loads(
        profile_path.read_text(
            encoding="utf-8"
        )
    )

    validate_ibkr_reference_profile(
        payload
    )

    return payload


def _profile_or_default(
    profile: (
        Mapping[str, object] | None
    ),
) -> Mapping[str, object]:
    if profile is None:
        return (
            load_ibkr_reference_profile()
        )

    validate_ibkr_reference_profile(
        profile
    )

    return profile


def calculate_us_stock_commission(
    *,
    quantity: object,
    trade_value_usd: object,
    pricing_plan: (
        IBKRPricingPlan | str
    ),
    fractional: bool = False,
    profile: (
        Mapping[str, object] | None
    ) = None,
) -> Decimal:
    quantity_value = _positive(
        "quantity",
        quantity,
    )

    trade_value = _positive(
        "trade_value_usd",
        trade_value_usd,
    )

    plan = _enum(
        IBKRPricingPlan,
        pricing_plan,
    )

    reference = _profile_or_default(
        profile
    )

    if not isinstance(
        fractional,
        bool,
    ):
        raise ValueError(
            "fractional must be boolean."
        )

    if fractional:
        rule = _mapping(
            reference["us_fractional"],
            "$.us_fractional",
        )

        calculated = (
            trade_value
            * _positive(
                "fractional rate",
                rule[
                    "commission_trade_value_fraction"
                ],
            )
        )

        return _money(
            max(
                calculated,
                _positive(
                    "fractional minimum",
                    rule[
                        "minimum_commission_usd"
                    ],
                ),
            )
        )

    rules = _mapping(
        reference["us_stocks"],
        "$.us_stocks",
    )

    key = (
        "tiered"
        if plan
        is IBKRPricingPlan.TIERED
        else "fixed"
    )

    rule = _mapping(
        rules[key],
        f"$.us_stocks.{key}",
    )

    calculated = (
        quantity_value
        * _positive(
            "per_share_usd",
            rule["per_share_usd"],
        )
    )

    minimum = _positive(
        "minimum_order_usd",
        rule["minimum_order_usd"],
    )

    maximum = (
        trade_value
        * _positive(
            "maximum_trade_value_fraction",
            rule[
                "maximum_trade_value_fraction"
            ],
        )
    )

    return _money(
        min(
            max(
                calculated,
                minimum,
            ),
            maximum,
        )
    )


def calculate_us_stock_reference_fees(
    *,
    quantity: object,
    trade_value_usd: object,
    pricing_plan: (
        IBKRPricingPlan | str
    ),
    side: IBKRTradeSide | str,
    fractional: bool = False,
    route_dependent_fee_usd: (
        object | None
    ) = None,
    profile: (
        Mapping[str, object] | None
    ) = None,
) -> IBKRStockCostEstimate:
    quantity_value = _positive(
        "quantity",
        quantity,
    )

    trade_value = _positive(
        "trade_value_usd",
        trade_value_usd,
    )

    plan = _enum(
        IBKRPricingPlan,
        pricing_plan,
    )

    trade_side = _enum(
        IBKRTradeSide,
        side,
    )

    reference = _profile_or_default(
        profile
    )

    commission = (
        calculate_us_stock_commission(
            quantity=quantity_value,
            trade_value_usd=trade_value,
            pricing_plan=plan,
            fractional=fractional,
            profile=reference,
        )
    )

    regulatory = _mapping(
        reference["us_regulatory"],
        "$.us_regulatory",
    )

    cat_fee = (
        quantity_value
        * _non_negative(
            "cat_per_share",
            regulatory["cat_per_share"],
        )
    )

    sale_fees = Decimal("0")

    if (
        trade_side
        is IBKRTradeSide.SELL
    ):
        sec_fee = (
            trade_value
            * _non_negative(
                "sec_sale_value_rate",
                regulatory[
                    "sec_sale_value_rate"
                ],
            )
        )

        taf_fee = min(
            quantity_value
            * _non_negative(
                "finra_taf_per_share_sale",
                regulatory[
                    "finra_taf_per_share_sale"
                ],
            ),
            _non_negative(
                "finra_taf_maximum_usd",
                regulatory[
                    "finra_taf_maximum_usd"
                ],
            ),
        )

        sale_fees = (
            sec_fee
            + taf_fee
        )

    regulatory_fees = _money(
        cat_fee
        + sale_fees
    )

    clearing_fees = Decimal("0")

    if (
        plan
        is IBKRPricingPlan.TIERED
    ):
        clearing_fees = _money(
            min(
                quantity_value
                * _non_negative(
                    "tiered_clearing_per_share",
                    regulatory[
                        "tiered_clearing_per_share"
                    ],
                ),
                trade_value
                * _non_negative(
                    "tiered clearing cap",
                    regulatory[
                        "tiered_clearing_maximum_trade_value_fraction"
                    ],
                ),
            )
        )

    route_fee_supplied = (
        route_dependent_fee_usd
        is not None
    )

    route_fee = (
        Decimal("0")
        if route_dependent_fee_usd
        is None
        else _non_negative(
            "route_dependent_fee_usd",
            route_dependent_fee_usd,
        )
    )

    # Tiered US pricing has venue/
    # routing-dependent external costs.
    complete = (
        route_fee_supplied
        if plan
        is IBKRPricingPlan.TIERED
        else True
    )

    total = _money(
        commission
        + regulatory_fees
        + clearing_fees
        + route_fee
    )

    return IBKRStockCostEstimate(
        currency="USD",
        commission=commission,
        regulatory_fees=(
            regulatory_fees
        ),
        clearing_fees=clearing_fees,
        route_dependent_fees=(
            route_fee
        ),
        total_known_cost=total,
        complete=complete,
    )


def calculate_europe_eur_reference_fees(
    *,
    trade_value_eur: object,
    pricing_plan: (
        IBKRPricingPlan | str
    ),
    route_dependent_fee_eur: (
        object | None
    ) = None,
    profile: (
        Mapping[str, object] | None
    ) = None,
) -> IBKRStockCostEstimate:
    trade_value = _positive(
        "trade_value_eur",
        trade_value_eur,
    )

    plan = _enum(
        IBKRPricingPlan,
        pricing_plan,
    )

    reference = _profile_or_default(
        profile
    )

    europe = _mapping(
        reference[
            "europe_eur_reference"
        ],
        "$.europe_eur_reference",
    )

    key = (
        "tiered"
        if plan
        is IBKRPricingPlan.TIERED
        else "fixed_smartrouting"
    )

    rule = _mapping(
        europe[key],
        (
            "$.europe_eur_reference."
            + key
        ),
    )

    calculated = (
        trade_value
        * _positive(
            "trade_value_fraction",
            rule[
                "trade_value_fraction"
            ],
        )
    )

    minimum = _positive(
        "minimum_order_eur",
        rule["minimum_order_eur"],
    )

    commission = max(
        calculated,
        minimum,
    )

    if (
        plan
        is IBKRPricingPlan.TIERED
    ):
        maximum = _positive(
            "maximum_order_eur",
            rule["maximum_order_eur"],
        )

        commission = min(
            commission,
            maximum,
        )

    route_supplied = (
        route_dependent_fee_eur
        is not None
    )

    route_fee = (
        Decimal("0")
        if route_dependent_fee_eur
        is None
        else _non_negative(
            "route_dependent_fee_eur",
            route_dependent_fee_eur,
        )
    )

    # Europe remains explicitly
    # market-specific in this first slice.
    # Even Fixed is not marked complete
    # until a concrete venue profile exists.
    complete = False

    return IBKRStockCostEstimate(
        currency="EUR",
        commission=_money(commission),
        regulatory_fees=Decimal("0"),
        clearing_fees=Decimal("0"),
        route_dependent_fees=(
            _money(route_fee)
        ),
        total_known_cost=_money(
            commission
            + route_fee
        ),
        complete=(
            complete
            and route_supplied
        ),
    )


def calculate_fx_reference_cost(
    *,
    trade_value_usd: object,
    mode: IBKRFXMode | str,
    profile: (
        Mapping[str, object] | None
    ) = None,
) -> IBKRFXCostEstimate:
    trade_value = _positive(
        "trade_value_usd",
        trade_value_usd,
    )

    fx_mode = _enum(
        IBKRFXMode,
        mode,
    )

    reference = _profile_or_default(
        profile
    )

    fx = _mapping(
        reference["fx"],
        "$.fx",
    )

    if (
        fx_mode
        is IBKRFXMode.SPOT_FX
    ):
        rule = _mapping(
            fx["spot_fx"],
            "$.fx.spot_fx",
        )

        rate = (
            _positive(
                "commission_bps",
                rule["commission_bps"],
            )
            / BASIS_POINTS
        )

        cost = max(
            trade_value * rate,
            _positive(
                "minimum_commission_usd",
                rule[
                    "minimum_commission_usd"
                ],
            ),
        )

        return IBKRFXCostEstimate(
            mode=fx_mode,
            reference_currency="USD",
            trade_value=trade_value,
            estimated_cost=_money(cost),
            separate_commission=True,
        )

    rule = _mapping(
        fx["auto_conversion"],
        "$.fx.auto_conversion",
    )

    adjustment = _positive(
        "rate_adjustment_fraction",
        rule[
            "rate_adjustment_fraction"
        ],
    )

    return IBKRFXCostEstimate(
        mode=fx_mode,
        reference_currency="USD",
        trade_value=trade_value,
        estimated_cost=_money(
            trade_value
            * adjustment
        ),
        separate_commission=False,
    )


def calculate_net_reward_to_risk(
    *,
    gross_reward_portfolio: object,
    gross_risk_portfolio: object,
    round_trip_cost_portfolio: object,
) -> IBKRNetRewardToRisk:
    reward = _positive(
        "gross_reward_portfolio",
        gross_reward_portfolio,
    )

    risk = _positive(
        "gross_risk_portfolio",
        gross_risk_portfolio,
    )

    cost = _non_negative(
        "round_trip_cost_portfolio",
        round_trip_cost_portfolio,
    )

    net_reward = max(
        Decimal("0"),
        reward - cost,
    )

    adjusted_risk = (
        risk + cost
    )

    ratio = (
        net_reward
        / adjusted_risk
    )

    return IBKRNetRewardToRisk(
        gross_reward=reward,
        gross_risk=risk,
        round_trip_cost=cost,
        net_reward=_money(
            net_reward
        ),
        cost_adjusted_risk=_money(
            adjusted_risk
        ),
        net_reward_to_risk=_money(
            ratio
        ),
    )

class IBKREconomicDecision(str, Enum):
    """Cost-aware P4.2 trade acceptance outcome."""

    ACCEPT = "ACCEPT"

    UNECONOMIC_AFTER_COSTS = (
        "UNECONOMIC_AFTER_COSTS"
    )

    INCOMPLETE_COST_ESTIMATE = (
        "INCOMPLETE_COST_ESTIMATE"
    )


@dataclass(frozen=True, slots=True)
class IBKRLongTradeEconomics:
    """Auditable economics for one proposed long trade."""

    pricing_plan: IBKRPricingPlan
    fx_mode: IBKRFXMode | None
    decision: IBKREconomicDecision

    quantity: Decimal
    usd_to_portfolio_rate: Decimal

    entry_notional_usd: Decimal
    stop_notional_usd: Decimal
    target_notional_usd: Decimal

    entry_stock_cost_usd: Decimal
    stop_exit_stock_cost_usd: Decimal
    target_exit_stock_cost_usd: Decimal

    entry_fx_cost_usd: Decimal
    stop_exit_fx_cost_usd: Decimal
    target_exit_fx_cost_usd: Decimal

    gross_reward_portfolio: Decimal
    gross_risk_portfolio: Decimal

    reward_path_cost_portfolio: Decimal
    risk_path_cost_portfolio: Decimal

    net_reward_portfolio: Decimal
    cost_adjusted_risk_portfolio: Decimal

    gross_reward_to_risk: Decimal
    net_reward_to_risk: Decimal
    minimum_net_reward_to_risk: Decimal

    complete: bool

    def __post_init__(self) -> None:
        if not isinstance(
            self.pricing_plan,
            IBKRPricingPlan,
        ):
            raise ValueError(
                "pricing_plan must be an "
                "IBKRPricingPlan."
            )

        if (
            self.fx_mode is not None
            and not isinstance(
                self.fx_mode,
                IBKRFXMode,
            )
        ):
            raise ValueError(
                "fx_mode must be an "
                "IBKRFXMode or None."
            )

        if not isinstance(
            self.decision,
            IBKREconomicDecision,
        ):
            raise ValueError(
                "decision must be an "
                "IBKREconomicDecision."
            )

        if not isinstance(
            self.complete,
            bool,
        ):
            raise ValueError(
                "complete must be boolean."
            )

        for name in (
            "quantity",
            "usd_to_portfolio_rate",
            "entry_notional_usd",
            "stop_notional_usd",
            "target_notional_usd",
            "gross_reward_portfolio",
            "gross_risk_portfolio",
            "cost_adjusted_risk_portfolio",
            "gross_reward_to_risk",
            "minimum_net_reward_to_risk",
        ):
            object.__setattr__(
                self,
                name,
                _money(
                    _positive(
                        name,
                        getattr(self, name),
                    )
                ),
            )

        for name in (
            "entry_stock_cost_usd",
            "stop_exit_stock_cost_usd",
            "target_exit_stock_cost_usd",
            "entry_fx_cost_usd",
            "stop_exit_fx_cost_usd",
            "target_exit_fx_cost_usd",
            "reward_path_cost_portfolio",
            "risk_path_cost_portfolio",
            "net_reward_portfolio",
            "net_reward_to_risk",
        ):
            object.__setattr__(
                self,
                name,
                _money(
                    _non_negative(
                        name,
                        getattr(self, name),
                    )
                ),
            )


def calculate_us_long_trade_economics(
    *,
    quantity: object,
    entry_price_usd: object,
    stop_price_usd: object,
    target_price_usd: object,
    usd_to_portfolio_rate: object,
    pricing_plan: IBKRPricingPlan | str,
    minimum_net_reward_to_risk: object,
    fractional: bool = False,
    fx_mode: IBKRFXMode | str | None = None,
    include_entry_fx_conversion: bool = False,
    include_exit_fx_conversion: bool = False,
    entry_route_dependent_fee_usd: (
        object | None
    ) = None,
    stop_exit_route_dependent_fee_usd: (
        object | None
    ) = None,
    target_exit_route_dependent_fee_usd: (
        object | None
    ) = None,
    profile: (
        Mapping[str, object] | None
    ) = None,
) -> IBKRLongTradeEconomics:
    """Evaluate a US long trade after realistic IBKR costs.

    FX conversion is deliberately explicit.  The function
    never assumes whether an IBKR account converts currency
    on entry, on exit, on both sides, or not at all.
    """

    quantity_value = _positive(
        "quantity",
        quantity,
    )

    entry_price = _positive(
        "entry_price_usd",
        entry_price_usd,
    )

    stop_price = _positive(
        "stop_price_usd",
        stop_price_usd,
    )

    target_price = _positive(
        "target_price_usd",
        target_price_usd,
    )

    rate = _positive(
        "usd_to_portfolio_rate",
        usd_to_portfolio_rate,
    )

    minimum_ratio = _positive(
        "minimum_net_reward_to_risk",
        minimum_net_reward_to_risk,
    )

    if not stop_price < entry_price:
        raise ValueError(
            "stop_price_usd must be below "
            "entry_price_usd."
        )

    if not target_price > entry_price:
        raise ValueError(
            "target_price_usd must be above "
            "entry_price_usd."
        )

    if not isinstance(
        fractional,
        bool,
    ):
        raise ValueError(
            "fractional must be boolean."
        )

    for name, value in (
        (
            "include_entry_fx_conversion",
            include_entry_fx_conversion,
        ),
        (
            "include_exit_fx_conversion",
            include_exit_fx_conversion,
        ),
    ):
        if not isinstance(
            value,
            bool,
        ):
            raise ValueError(
                f"{name} must be boolean."
            )

    plan = _enum(
        IBKRPricingPlan,
        pricing_plan,
    )

    resolved_fx_mode = (
        None
        if fx_mode is None
        else _enum(
            IBKRFXMode,
            fx_mode,
        )
    )

    if (
        (
            include_entry_fx_conversion
            or include_exit_fx_conversion
        )
        and resolved_fx_mode is None
    ):
        raise ValueError(
            "fx_mode is required when "
            "FX conversion costs are included."
        )

    reference = _profile_or_default(
        profile
    )

    entry_notional = _money(
        entry_price
        * quantity_value
    )

    stop_notional = _money(
        stop_price
        * quantity_value
    )

    target_notional = _money(
        target_price
        * quantity_value
    )

    entry_stock = (
        calculate_us_stock_reference_fees(
            quantity=quantity_value,
            trade_value_usd=entry_notional,
            pricing_plan=plan,
            side=IBKRTradeSide.BUY,
            fractional=fractional,
            route_dependent_fee_usd=(
                entry_route_dependent_fee_usd
            ),
            profile=reference,
        )
    )

    stop_stock = (
        calculate_us_stock_reference_fees(
            quantity=quantity_value,
            trade_value_usd=stop_notional,
            pricing_plan=plan,
            side=IBKRTradeSide.SELL,
            fractional=fractional,
            route_dependent_fee_usd=(
                stop_exit_route_dependent_fee_usd
            ),
            profile=reference,
        )
    )

    target_stock = (
        calculate_us_stock_reference_fees(
            quantity=quantity_value,
            trade_value_usd=target_notional,
            pricing_plan=plan,
            side=IBKRTradeSide.SELL,
            fractional=fractional,
            route_dependent_fee_usd=(
                target_exit_route_dependent_fee_usd
            ),
            profile=reference,
        )
    )

    entry_fx_cost = Decimal("0")
    stop_fx_cost = Decimal("0")
    target_fx_cost = Decimal("0")

    if include_entry_fx_conversion:
        assert resolved_fx_mode is not None

        entry_fx_cost = (
            calculate_fx_reference_cost(
                trade_value_usd=entry_notional,
                mode=resolved_fx_mode,
                profile=reference,
            )
            .estimated_cost
        )

    if include_exit_fx_conversion:
        assert resolved_fx_mode is not None

        stop_fx_cost = (
            calculate_fx_reference_cost(
                trade_value_usd=stop_notional,
                mode=resolved_fx_mode,
                profile=reference,
            )
            .estimated_cost
        )

        target_fx_cost = (
            calculate_fx_reference_cost(
                trade_value_usd=target_notional,
                mode=resolved_fx_mode,
                profile=reference,
            )
            .estimated_cost
        )

    gross_reward = _money(
        (
            target_price
            - entry_price
        )
        * quantity_value
        * rate
    )

    gross_risk = _money(
        (
            entry_price
            - stop_price
        )
        * quantity_value
        * rate
    )

    reward_path_cost = _money(
        (
            entry_stock.total_known_cost
            + target_stock.total_known_cost
            + entry_fx_cost
            + target_fx_cost
        )
        * rate
    )

    risk_path_cost = _money(
        (
            entry_stock.total_known_cost
            + stop_stock.total_known_cost
            + entry_fx_cost
            + stop_fx_cost
        )
        * rate
    )

    net_reward = _money(
        max(
            Decimal("0"),
            gross_reward
            - reward_path_cost,
        )
    )

    cost_adjusted_risk = _money(
        gross_risk
        + risk_path_cost
    )

    gross_ratio = _money(
        gross_reward
        / gross_risk
    )

    net_ratio = _money(
        net_reward
        / cost_adjusted_risk
    )

    complete = (
        entry_stock.complete
        and stop_stock.complete
        and target_stock.complete
    )

    if not complete:
        decision = (
            IBKREconomicDecision
            .INCOMPLETE_COST_ESTIMATE
        )

    elif net_ratio < minimum_ratio:
        decision = (
            IBKREconomicDecision
            .UNECONOMIC_AFTER_COSTS
        )

    else:
        decision = (
            IBKREconomicDecision.ACCEPT
        )

    return IBKRLongTradeEconomics(
        pricing_plan=plan,
        fx_mode=resolved_fx_mode,
        decision=decision,
        quantity=quantity_value,
        usd_to_portfolio_rate=rate,
        entry_notional_usd=entry_notional,
        stop_notional_usd=stop_notional,
        target_notional_usd=target_notional,
        entry_stock_cost_usd=(
            entry_stock.total_known_cost
        ),
        stop_exit_stock_cost_usd=(
            stop_stock.total_known_cost
        ),
        target_exit_stock_cost_usd=(
            target_stock.total_known_cost
        ),
        entry_fx_cost_usd=entry_fx_cost,
        stop_exit_fx_cost_usd=stop_fx_cost,
        target_exit_fx_cost_usd=(
            target_fx_cost
        ),
        gross_reward_portfolio=(
            gross_reward
        ),
        gross_risk_portfolio=(
            gross_risk
        ),
        reward_path_cost_portfolio=(
            reward_path_cost
        ),
        risk_path_cost_portfolio=(
            risk_path_cost
        ),
        net_reward_portfolio=(
            net_reward
        ),
        cost_adjusted_risk_portfolio=(
            cost_adjusted_risk
        ),
        gross_reward_to_risk=(
            gross_ratio
        ),
        net_reward_to_risk=(
            net_ratio
        ),
        minimum_net_reward_to_risk=(
            minimum_ratio
        ),
        complete=complete,
    )
