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
    ApprovedThresholdManifest,
    BuyAndHoldComparison,
    CandidateEvaluation,
    EquityPoint,
    FoldValidationResult,
    ParameterStabilityEntry,
    ParameterStabilityReport,
    PerformanceReport,
    PromotionDecision,
    RegressionEvidence,
    ReleaseGateStatus,
    StrategyAcceptanceThresholds,
    WalkForwardConfig,
    WalkForwardFold,
    WalkForwardValidationReport,
    build_strategy_acceptance_report,
    evaluate_p2_release_gate,
    load_approved_threshold_manifest,
)


T0 = datetime(
    2026,
    1,
    1,
    tzinfo=timezone.utc,
)


def make_performance(
    *,
    total_return: float = 0.10,
) -> PerformanceReport:
    starting_balance = Decimal("10000")
    ending_balance = Decimal("11000")

    benchmark = BuyAndHoldComparison(
        first_session=T0,
        last_session=T0 + timedelta(days=30),
        starting_price=100.0,
        ending_price=104.0,
        total_return=0.04,
        annualised_return=0.04,
        max_drawdown=0.05,
        hypothetical_ending_balance=Decimal("10400"),
        strategy_excess_return=(
            total_return - 0.04
        ),
    )

    return PerformanceReport(
        period_start=T0,
        period_end=T0 + timedelta(days=30),
        starting_balance=starting_balance,
        ending_balance=ending_balance,
        total_net_pnl=Decimal("1000"),
        total_return=total_return,
        annualised_return=total_return,
        max_drawdown=0.10,
        sharpe_ratio=1.0,
        trade_count=30,
        winning_trades=18,
        losing_trades=12,
        breakeven_trades=0,
        win_rate=0.60,
        gross_profit=Decimal("1500"),
        gross_loss=Decimal("500"),
        profit_factor=3.0,
        exposure=0.50,
        average_holding_period_days=2.0,
        equity_curve=(
            EquityPoint(
                timestamp=T0,
                balance=starting_balance,
            ),
            EquityPoint(
                timestamp=T0 + timedelta(days=30),
                balance=ending_balance,
            ),
        ),
        benchmark=benchmark,
    )


def make_validation_report() -> WalkForwardValidationReport:
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
        train_start=pd.Timestamp(T0),
        train_end=pd.Timestamp(
            T0 + timedelta(days=9)
        ),
        test_start=pd.Timestamp(
            T0 + timedelta(days=10)
        ),
        test_end=pd.Timestamp(
            T0 + timedelta(days=11)
        ),
    )

    candidate = CandidateEvaluation(
        parameters={"threshold": 1},
        metrics={"score": 1.0},
    )

    fold_result = FoldValidationResult(
        fold=fold,
        selected_parameters={"threshold": 1},
        in_sample_metrics={"score": 1.0},
        out_of_sample_metrics={"score": 0.8},
        candidate_evaluations=(candidate,),
    )

    stability_entry = ParameterStabilityEntry(
        parameter="threshold",
        selected_values=(1,),
        most_common_value=1,
        most_common_share=1.0,
        unique_value_count=1,
        change_count=0,
        stability_score=1.0,
    )

    stability = ParameterStabilityReport(
        entries=(stability_entry,),
        overall_stability_score=1.0,
    )

    return WalkForwardValidationReport(
        config=config,
        fold_results=(fold_result,),
        mean_in_sample_metric=1.0,
        mean_out_of_sample_metric=0.8,
        generalisation_gap=0.2,
        parameter_stability=stability,
    )


def make_acceptance_report(
    *,
    accepted: bool = True,
):
    performance = make_performance(
        total_return=(
            0.10
            if accepted
            else -0.10
        )
    )

    return build_strategy_acceptance_report(
        "trend-pullback",
        instrument_performance={
            "AAPL": performance,
        },
        regime_performance={
            "BULLISH": performance,
        },
        validation_report=make_validation_report(),
        promotion_decision=PromotionDecision(
            promoted=True,
            reasons=(
                "Out-of-sample promotion approved.",
            ),
        ),
        thresholds=StrategyAcceptanceThresholds(
            minimum_total_return=0.05,
            maximum_drawdown=0.20,
            minimum_trade_count=20,
            minimum_parameter_stability=0.75,
        ),
    )


def make_manifest(
    *,
    approved: bool = True,
) -> ApprovedThresholdManifest:
    return ApprovedThresholdManifest(
        schema_version=1,
        approval_status=(
            "APPROVED_FOR_P2_RELEASE"
            if approved
            else "DRAFT"
        ),
        profiles={
            "trend_pullback": {
                "class": "TrendPullbackThresholds",
                "values": {
                    "minimum_score": 1.0,
                },
            },
        },
    )


def make_regression(
    *,
    passed: bool = True,
    phases=("P0", "P1", "P2"),
) -> RegressionEvidence:
    return RegressionEvidence(
        passed=passed,
        test_count=231,
        covered_phases=tuple(phases),
    )


def test_validated_strategy_is_eligible() -> None:
    report = evaluate_p2_release_gate(
        make_acceptance_report(),
        regression_evidence=make_regression(),
        threshold_manifest=make_manifest(),
    )

    assert (
        report.status
        is ReleaseGateStatus.ELIGIBLE
    )
    assert report.alert_scheduling_eligible is True
    assert (
        report.acceptance_report.status
        is AcceptanceStatus.ACCEPT
    )


def test_rejected_strategy_is_ineligible() -> None:
    report = evaluate_p2_release_gate(
        make_acceptance_report(
            accepted=False
        ),
        regression_evidence=make_regression(),
        threshold_manifest=make_manifest(),
    )

    assert (
        report.status
        is ReleaseGateStatus.INELIGIBLE
    )
    assert report.alert_scheduling_eligible is False


def test_failed_regression_is_ineligible() -> None:
    report = evaluate_p2_release_gate(
        make_acceptance_report(),
        regression_evidence=make_regression(
            passed=False
        ),
        threshold_manifest=make_manifest(),
    )

    assert report.alert_scheduling_eligible is False
    assert any(
        "regression suite"
        in reason.lower()
        for reason in report.reasons
    )


def test_missing_regression_phase_is_ineligible() -> None:
    report = evaluate_p2_release_gate(
        make_acceptance_report(),
        regression_evidence=make_regression(
            phases=("P0", "P2")
        ),
        threshold_manifest=make_manifest(),
    )

    assert report.alert_scheduling_eligible is False
    assert any(
        "P1" in reason
        for reason in report.reasons
    )


def test_unapproved_thresholds_are_ineligible() -> None:
    report = evaluate_p2_release_gate(
        make_acceptance_report(),
        regression_evidence=make_regression(),
        threshold_manifest=make_manifest(
            approved=False
        ),
    )

    assert report.alert_scheduling_eligible is False
    assert any(
        "threshold"
        in reason.lower()
        for reason in report.reasons
    )


def test_committed_threshold_manifest_loads() -> None:
    manifest = load_approved_threshold_manifest(
        "config/approved_signal_thresholds.json"
    )

    assert manifest.approved is True
    assert manifest.schema_version == 1
    assert "trend_pullback" in manifest.profiles
    assert "risk_management" in manifest.profiles


def test_limitations_cannot_be_empty() -> None:
    with pytest.raises(ValueError):
        evaluate_p2_release_gate(
            make_acceptance_report(),
            regression_evidence=make_regression(),
            threshold_manifest=make_manifest(),
            documented_limitations=(),
        )


def test_release_gate_is_deterministic() -> None:
    arguments = {
        "acceptance_report":
        make_acceptance_report(),
        "regression_evidence":
        make_regression(),
        "threshold_manifest":
        make_manifest(),
    }

    first = evaluate_p2_release_gate(
        **arguments
    )
    second = evaluate_p2_release_gate(
        **arguments
    )

    assert first == second
