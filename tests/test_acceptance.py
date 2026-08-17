from __future__ import annotations

from dataclasses import replace
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from decimal import Decimal

import pandas as pd
import pytest

from src.backtest import (
    AcceptanceStatus,
    BuyAndHoldComparison,
    CandidateEvaluation,
    EquityPoint,
    FoldValidationResult,
    HorizonAcceptanceEvidence,
    IndependentHorizonAcceptanceReport,
    ParameterStabilityEntry,
    ParameterStabilityReport,
    PerformanceReport,
    PerformanceScope,
    PromotionDecision,
    StrategyAcceptanceThresholds,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardValidationReport,
    build_strategy_acceptance_report,
    build_independent_horizon_acceptance_report,
)
from src.strategy import StrategyHorizon


T0 = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def make_performance(
    *,
    total_return: float = 0.10,
    max_drawdown: float = 0.10,
    trade_count: int = 30,
) -> PerformanceReport:
    starting_balance = Decimal("10000")

    ending_balance = (
        starting_balance
        * Decimal(
            str(1 + total_return)
        )
    )

    total_net_pnl = (
        ending_balance
        - starting_balance
    )

    winning_trades = (
        trade_count // 2
    )

    losing_trades = (
        trade_count
        - winning_trades
    )

    benchmark_return = 0.04

    benchmark = BuyAndHoldComparison(
        first_session=T0,
        last_session=(
            T0
            + timedelta(days=30)
        ),
        starting_price=100.0,
        ending_price=104.0,
        total_return=benchmark_return,
        annualised_return=benchmark_return,
        max_drawdown=0.08,
        hypothetical_ending_balance=(
            Decimal("10400")
        ),
        strategy_excess_return=(
            total_return
            - benchmark_return
        ),
    )

    return PerformanceReport(
        period_start=T0,
        period_end=(
            T0
            + timedelta(days=30)
        ),
        starting_balance=starting_balance,
        ending_balance=ending_balance,
        total_net_pnl=total_net_pnl,
        total_return=total_return,
        annualised_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=1.2,
        trade_count=trade_count,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        breakeven_trades=0,
        win_rate=(
            winning_trades / trade_count
            if trade_count
            else 0.0
        ),
        gross_profit=(
            Decimal("1000")
            if trade_count
            else Decimal("0")
        ),
        gross_loss=(
            Decimal("500")
            if trade_count
            else Decimal("0")
        ),
        profit_factor=(
            2.0
            if trade_count
            else None
        ),
        exposure=0.50,
        average_holding_period_days=2.0,
        equity_curve=(
            EquityPoint(
                timestamp=T0,
                balance=starting_balance,
            ),
            EquityPoint(
                timestamp=(
                    T0
                    + timedelta(days=30)
                ),
                balance=ending_balance,
            ),
        ),
        benchmark=benchmark,
    )


def make_validation_report(
    *,
    stability: float = 1.0,
) -> WalkForwardValidationReport:
    config = WalkForwardConfig(
        train_size=10,
        test_size=2,
        max_folds=1,
    )

    fold = WalkForwardFold(
        fold_number=1,
        train_start_position=0,
        train_end_position=10,
        test_start_position=10,
        test_end_position=12,
        train_start=pd.Timestamp(
            T0
        ),
        train_end=pd.Timestamp(
            T0
            + timedelta(days=9)
        ),
        test_start=pd.Timestamp(
            T0
            + timedelta(days=10)
        ),
        test_end=pd.Timestamp(
            T0
            + timedelta(days=11)
        ),
    )

    candidate = CandidateEvaluation(
        parameters={"threshold": 1},
        metrics={"score": 1.0},
    )

    fold_result = FoldValidationResult(
        fold=fold,
        selected_parameters={
            "threshold": 1,
        },
        in_sample_metrics={
            "score": 1.0,
        },
        out_of_sample_metrics={
            "score": 0.8,
        },
        candidate_evaluations=(
            candidate,
        ),
    )

    stability_entry = ParameterStabilityEntry(
        parameter="threshold",
        selected_values=(1,),
        most_common_value=1,
        most_common_share=1.0,
        unique_value_count=1,
        change_count=0,
        stability_score=stability,
    )

    stability_report = ParameterStabilityReport(
        entries=(
            stability_entry,
        ),
        overall_stability_score=stability,
    )

    return WalkForwardValidationReport(
        config=config,
        fold_results=(
            fold_result,
        ),
        mean_in_sample_metric=1.0,
        mean_out_of_sample_metric=0.8,
        generalisation_gap=0.2,
        parameter_stability=(
            stability_report
        ),
    )


def accepted_promotion() -> PromotionDecision:
    return PromotionDecision(
        promoted=True,
        reasons=(
            "Out-of-sample promotion approved.",
        ),
    )


def thresholds() -> StrategyAcceptanceThresholds:
    return StrategyAcceptanceThresholds(
        minimum_total_return=0.05,
        maximum_drawdown=0.20,
        minimum_trade_count=20,
        minimum_parameter_stability=0.75,
    )


def test_strategy_is_accepted_when_all_checks_pass() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert (
        report.status
        is AcceptanceStatus.ACCEPT
    )

    assert report.accepted is True

    assert all(
        check.passed
        for check in report.checks
    )


def test_weak_instrument_return_is_rejected() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(
                total_return=0.01,
            ),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert (
        report.status
        is AcceptanceStatus.REJECT
    )

    assert any(
        "total return"
        in reason.lower()
        for reason in report.reasons
    )


def test_high_regime_drawdown_is_rejected() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(),
        },
        regime_performance={
            "VOLATILE": make_performance(
                max_drawdown=0.30,
            ),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert report.accepted is False

    assert any(
        "drawdown"
        in reason.lower()
        for reason in report.reasons
    )


def test_low_trade_count_is_rejected() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(
                trade_count=5,
            ),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert report.accepted is False

    assert any(
        "trade count"
        in reason.lower()
        for reason in report.reasons
    )


def test_low_parameter_stability_is_rejected() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report(
                stability=0.50,
            )
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert report.accepted is False

    assert any(
        "stability"
        in reason.lower()
        for reason in report.reasons
    )


def test_failed_oos_promotion_is_rejected() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            PromotionDecision(
                promoted=False,
                reasons=(
                    "Out-of-sample metric was weak.",
                ),
            )
        ),
        thresholds=thresholds(),
    )

    assert report.accepted is False

    assert (
        "Out-of-sample metric was weak."
        in report.reasons
    )


def test_missing_instrument_coverage_is_rejected() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={},
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert report.accepted is False

    assert any(
        "instrument"
        in reason.lower()
        for reason in report.reasons
    )


def test_missing_regime_coverage_is_rejected() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(),
        },
        regime_performance={},
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert report.accepted is False

    assert any(
        "market regime"
        in reason.lower()
        for reason in report.reasons
    )


def test_performance_slices_are_sorted_deterministically() -> None:
    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "TSLA": make_performance(),
            "AAPL": make_performance(),
            "MSFT": make_performance(),
        },
        regime_performance={
            "VOLATILE": make_performance(),
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    assert [
        item.name
        for item in (
            report.instrument_performance
        )
    ] == [
        "AAPL",
        "MSFT",
        "TSLA",
    ]

    assert [
        item.name
        for item in (
            report.regime_performance
        )
    ] == [
        "BULLISH",
        "VOLATILE",
    ]


def test_source_performance_values_are_not_recalculated() -> None:
    source = make_performance(
        total_return=0.1234,
        max_drawdown=0.0876,
        trade_count=42,
    )

    report = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": source,
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    copied = report.instrument_performance[0]

    assert (
        copied.scope
        is PerformanceScope.INSTRUMENT
    )

    assert (
        copied.total_return
        == source.total_return
    )

    assert (
        copied.maximum_drawdown
        == source.max_drawdown
    )

    assert (
        copied.trade_count
        == source.trade_count
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"maximum_drawdown": -0.01},
        {"maximum_drawdown": 1.01},
        {"minimum_trade_count": -1},
        {
            "minimum_parameter_stability":
            -0.01
        },
        {
            "minimum_parameter_stability":
            1.01
        },
    ],
)
def test_invalid_acceptance_thresholds_are_rejected(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        StrategyAcceptanceThresholds(
            **kwargs
        )

def test_horizon_acceptance_decisions_are_independent() -> None:
    swing = build_strategy_acceptance_report(
        "trend-pullback-swing",
        instrument_performance={
            "AAPL": make_performance(),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    medium = build_strategy_acceptance_report(
        "trend-pullback-medium",
        instrument_performance={
            "AAPL": make_performance(
                trade_count=5,
            ),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    report = (
        build_independent_horizon_acceptance_report(
            swing_report=swing,
            swing_strategy_version=(
                "p4.3-swing-v1"
            ),
            medium_term_report=medium,
            medium_term_strategy_version=(
                "p4.3-medium-term-v1"
            ),
        )
    )

    assert report.accepted_horizons == (
        StrategyHorizon.SWING,
    )
    assert report.rejected_horizons == (
        StrategyHorizon.MEDIUM_TERM,
    )
    assert (
        report.for_horizon(
            StrategyHorizon.SWING
        ).strategy_version
        == "p4.3-swing-v1"
    )
    assert (
        report.for_horizon(
            StrategyHorizon.MEDIUM_TERM
        ).accepted
        is False
    )


def test_independent_evidence_rejects_duplicate_horizons(
) -> None:
    accepted = build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": make_performance(),
        },
        regime_performance={
            "BULLISH": make_performance(),
        },
        validation_report=(
            make_validation_report()
        ),
        promotion_decision=(
            accepted_promotion()
        ),
        thresholds=thresholds(),
    )

    duplicate = HorizonAcceptanceEvidence(
        horizon=StrategyHorizon.SWING,
        strategy_version="p4.3-swing-v1",
        acceptance_report=accepted,
    )

    with pytest.raises(
        ValueError,
        match="exactly one SWING",
    ):
        IndependentHorizonAcceptanceReport(
            evidence=(
                duplicate,
                duplicate,
            )
        )

