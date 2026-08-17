"""Strategy-horizon provenance shared across project layers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class StrategyHorizon(str, Enum):
    """Approved non-intraday strategy horizons."""

    SWING = "SWING"
    MEDIUM_TERM = "MEDIUM_TERM"


def coerce_strategy_horizon(
    value: object | None,
) -> StrategyHorizon | None:
    """Normalize and validate optional horizon provenance."""

    if value is None:
        return None

    if isinstance(
        value,
        StrategyHorizon,
    ):
        return value

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "strategy_horizon must be "
            "SWING, MEDIUM_TERM or None."
        )

    normalized = (
        value.strip()
        .upper()
        .replace("-", "_")
    )

    try:
        return StrategyHorizon(
            normalized
        )
    except ValueError as exc:
        raise ValueError(
            "strategy_horizon must be "
            "SWING or MEDIUM_TERM."
        ) from exc


def strategy_horizon_value(
    value: object | None,
) -> str | None:
    """Return canonical storage representation."""

    horizon = coerce_strategy_horizon(
        value
    )

    return (
        horizon.value
        if horizon is not None
        else None
    )


def normalise_strategy_version(
    value: object | None,
) -> str | None:
    """Normalize optional strategy-version provenance."""

    if value is None:
        return None

    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            "strategy_version must be "
            "a string or None."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            "strategy_version cannot be blank."
        )

    return result

HORIZON_POLICY_VERSION = "horizon-policy-v1"


class StrategyConfirmationPolicy(
    str,
    Enum,
):
    """Confirmation contract for a strategy horizon."""

    STRATEGY_CONFIRMATION = (
        "STRATEGY_CONFIRMATION"
    )

    WEEKLY_CLOSE_PLUS_STRATEGY_CONFIRMATION = (
        "WEEKLY_CLOSE_PLUS_"
        "STRATEGY_CONFIRMATION"
    )


class StrategyEntryTiming(
    str,
    Enum,
):
    """Permitted entry timing."""

    NEXT_ELIGIBLE_SESSION = (
        "NEXT_ELIGIBLE_SESSION"
    )


class StrategyExitPolicy(
    str,
    Enum,
):
    """Strategy-driven exit mechanisms."""

    STOP_LOSS = "STOP_LOSS"
    TARGET = "TARGET"
    TIME_EXIT = "TIME_EXIT"
    SIGNAL_REVERSAL = "SIGNAL_REVERSAL"
    REGIME_INVALIDATION = (
        "REGIME_INVALIDATION"
    )
    PORTFOLIO_RISK = "PORTFOLIO_RISK"


def _required_policy_text(
    name: str,
    value: object,
) -> str:
    if not isinstance(
        value,
        str,
    ):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    result = value.strip()

    if not result:
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return result


def _positive_sessions(
    name: str,
    value: object,
) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(
            f"{name} must be a positive integer."
        )

    return value


@dataclass(
    frozen=True,
    slots=True,
)
class HorizonPolicy:
    """Immutable strategy-horizon policy contract."""

    policy_version: str
    horizon: StrategyHorizon
    strategy_version: str
    market_data_period: str
    market_data_interval: str
    signal_validity_sessions: int
    maximum_holding_sessions: int
    confirmation_policy: (
        StrategyConfirmationPolicy
    )
    entry_timing: StrategyEntryTiming
    intraday_entries_allowed: bool
    exit_policies: tuple[
        StrategyExitPolicy,
        ...,
    ]

    def __post_init__(
        self,
    ) -> None:
        policy_version = (
            _required_policy_text(
                "policy_version",
                self.policy_version,
            )
        )

        if (
            policy_version
            != HORIZON_POLICY_VERSION
        ):
            raise ValueError(
                "Unsupported horizon "
                "policy_version."
            )

        horizon = (
            coerce_strategy_horizon(
                self.horizon
            )
        )

        if horizon is None:
            raise ValueError(
                "horizon is required."
            )

        strategy_version = (
            _required_policy_text(
                "strategy_version",
                self.strategy_version,
            )
        )

        period = _required_policy_text(
            "market_data_period",
            self.market_data_period,
        )

        interval = _required_policy_text(
            "market_data_interval",
            self.market_data_interval,
        )

        if interval not in {
            "1d",
            "1wk",
        }:
            raise ValueError(
                "Only end-of-day or weekly "
                "market-data intervals are "
                "permitted."
            )

        signal_sessions = (
            _positive_sessions(
                "signal_validity_sessions",
                self.signal_validity_sessions,
            )
        )

        holding_sessions = (
            _positive_sessions(
                "maximum_holding_sessions",
                self.maximum_holding_sessions,
            )
        )

        if (
            holding_sessions
            <= signal_sessions
        ):
            raise ValueError(
                "maximum_holding_sessions "
                "must exceed "
                "signal_validity_sessions."
            )

        try:
            confirmation = (
                self.confirmation_policy
                if isinstance(
                    self.confirmation_policy,
                    StrategyConfirmationPolicy,
                )
                else StrategyConfirmationPolicy(
                    self.confirmation_policy
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Unsupported "
                "confirmation_policy."
            ) from exc

        try:
            entry_timing = (
                self.entry_timing
                if isinstance(
                    self.entry_timing,
                    StrategyEntryTiming,
                )
                else StrategyEntryTiming(
                    self.entry_timing
                )
            )
        except ValueError as exc:
            raise ValueError(
                "Unsupported entry_timing."
            ) from exc

        if not isinstance(
            self.intraday_entries_allowed,
            bool,
        ):
            raise ValueError(
                "intraday_entries_allowed "
                "must be boolean."
            )

        if self.intraday_entries_allowed:
            raise ValueError(
                "Intraday/day-trading entries "
                "are prohibited."
            )

        try:
            exits = tuple(
                value
                if isinstance(
                    value,
                    StrategyExitPolicy,
                )
                else StrategyExitPolicy(
                    value
                )
                for value in self.exit_policies
            )
        except ValueError as exc:
            raise ValueError(
                "Unsupported exit policy."
            ) from exc

        if len(exits) != len(
            set(exits)
        ):
            raise ValueError(
                "exit_policies must be unique."
            )

        mandatory = {
            StrategyExitPolicy.STOP_LOSS,
            StrategyExitPolicy.TARGET,
            StrategyExitPolicy.TIME_EXIT,
        }

        if not mandatory.issubset(
            set(exits)
        ):
            raise ValueError(
                "Horizon policy requires "
                "stop, target and time exits."
            )

        object.__setattr__(
            self,
            "policy_version",
            policy_version,
        )

        object.__setattr__(
            self,
            "horizon",
            horizon,
        )

        object.__setattr__(
            self,
            "strategy_version",
            strategy_version,
        )

        object.__setattr__(
            self,
            "market_data_period",
            period,
        )

        object.__setattr__(
            self,
            "market_data_interval",
            interval,
        )

        object.__setattr__(
            self,
            "signal_validity_sessions",
            signal_sessions,
        )

        object.__setattr__(
            self,
            "maximum_holding_sessions",
            holding_sessions,
        )

        object.__setattr__(
            self,
            "confirmation_policy",
            confirmation,
        )

        object.__setattr__(
            self,
            "entry_timing",
            entry_timing,
        )

        object.__setattr__(
            self,
            "exit_policies",
            exits,
        )


def horizon_policies_from_product_policy(
    policy: Mapping[str, object],
) -> dict[
    StrategyHorizon,
    HorizonPolicy,
]:
    """Parse immutable horizon policies from product policy."""

    if not isinstance(
        policy,
        Mapping,
    ):
        raise ValueError(
            "policy must be a mapping."
        )

    strategies = policy.get(
        "strategies"
    )

    if not isinstance(
        strategies,
        Mapping,
    ):
        raise ValueError(
            "strategies must be a mapping."
        )

    policy_version = (
        _required_policy_text(
            "horizon_policy_version",
            strategies.get(
                "horizon_policy_version"
            ),
        )
    )

    if (
        policy_version
        != HORIZON_POLICY_VERSION
    ):
        raise ValueError(
            "Unsupported horizon policy version."
        )

    raw_policies = strategies.get(
        "horizon_policies"
    )

    if not isinstance(
        raw_policies,
        Mapping,
    ):
        raise ValueError(
            "horizon_policies must be a mapping."
        )

    expected_keys = {
        "swing",
        "medium_term",
    }

    if set(raw_policies) != expected_keys:
        raise ValueError(
            "horizon_policies must contain "
            "exactly swing and medium_term."
        )

    result: dict[
        StrategyHorizon,
        HorizonPolicy,
    ] = {}

    for (
        key,
        horizon,
    ) in (
        (
            "swing",
            StrategyHorizon.SWING,
        ),
        (
            "medium_term",
            StrategyHorizon.MEDIUM_TERM,
        ),
    ):
        raw = raw_policies[key]

        if not isinstance(
            raw,
            Mapping,
        ):
            raise ValueError(
                f"{key} horizon policy "
                "must be a mapping."
            )

        expected_fields = {
            "strategy_version",
            "market_data_period",
            "market_data_interval",
            "signal_validity_sessions",
            "maximum_holding_sessions",
            "confirmation_policy",
            "entry_timing",
            "intraday_entries_allowed",
            "exit_policies",
        }

        if set(raw) != expected_fields:
            raise ValueError(
                f"{key} horizon policy has "
                "unexpected or missing fields."
            )

        raw_exits = raw.get(
            "exit_policies"
        )

        if (
            not isinstance(
                raw_exits,
                (list, tuple),
            )
            or isinstance(
                raw_exits,
                (str, bytes),
            )
        ):
            raise ValueError(
                "exit_policies must be "
                "a sequence."
            )

        result[horizon] = HorizonPolicy(
            policy_version=policy_version,
            horizon=horizon,
            strategy_version=raw[
                "strategy_version"
            ],
            market_data_period=raw[
                "market_data_period"
            ],
            market_data_interval=raw[
                "market_data_interval"
            ],
            signal_validity_sessions=raw[
                "signal_validity_sessions"
            ],
            maximum_holding_sessions=raw[
                "maximum_holding_sessions"
            ],
            confirmation_policy=raw[
                "confirmation_policy"
            ],
            entry_timing=raw[
                "entry_timing"
            ],
            intraday_entries_allowed=raw[
                "intraday_entries_allowed"
            ],
            exit_policies=tuple(
                raw_exits
            ),
        )

    return result
