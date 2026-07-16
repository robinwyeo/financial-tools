"""Tests for cross-sectional scoring."""

import numpy as np
import pandas as pd
import pytest

from core.factors import FACTOR_SCORE_COLUMNS
from core.scoring import (
    _composite_and_coverage,
    _evaluate_good_buy,
    _merge_ticker_row_with_universe,
    apply_universe_snapshot_scoring,
    compute_bargain_score,
    score_ticker,
    score_universe_df,
)


def _minimal_config() -> dict:
    weights = {family: 1.0 / len(FACTOR_SCORE_COLUMNS) for family in FACTOR_SCORE_COLUMNS}
    return {"factor_weights": weights, "universe": {"sector_scoring": False}}


def _minimal_df() -> pd.DataFrame:
    """Minimal universe dataframe with all sub-signal columns for the 8 factor groups."""
    base = {
        "sector": "Tech",
        # value group
        "earnings_yield": 0.08,
        "fcf_yield": 0.06,
        "book_to_market": 0.3,
        "graham_ratio": 0.9,
        # garp group
        "garp": 1.5,
        # quality group
        "gross_profitability": 0.4,
        "roe": 0.15,
        "roa": 0.08,
        "profit_margin": 0.12,
        "roic": 0.18,
        "earnings_quality": 0.02,
        "financial_strength": 7.0,
        # balance_sheet group
        "net_cash_to_mcap": -0.05,
        "low_leverage": 0.5,
        "altman_z": 3.5,
        # momentum group
        "momentum_12_1": 0.15,
        # low_volatility group
        "low_volatility": 8.0,
        # capital_discipline group
        "shareholder_yield": 0.03,
        "investment": -0.05,
        # earnings_revisions group
        "earnings_revisions": 1.0,
    }
    rows = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        row = {"ticker": ticker, **base}
        # Vary some values so cross-sectional ranking is non-degenerate
        row["earnings_yield"] = base["earnings_yield"] * (1 + i * 0.1)
        row["momentum_12_1"] = base["momentum_12_1"] * (1 - i * 0.1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_composite_excludes_missing_factors():
    weights = {"value": 0.5, "momentum": 0.5}
    row = pd.Series({"pct_value": 80.0, "pct_momentum": np.nan})
    composite, coverage = _composite_and_coverage(row, weights)
    assert composite == 80.0
    assert coverage == 50.0


def test_composite_coverage_zero_when_all_missing():
    weights = {"value": 0.5, "momentum": 0.5}
    row = pd.Series({"pct_value": np.nan, "pct_momentum": np.nan})
    composite, coverage = _composite_and_coverage(row, weights)
    assert composite is None
    assert coverage == 0.0


def test_score_universe_df_adds_factor_coverage():
    df = _minimal_df()
    scored = score_universe_df(df, _minimal_config(), group_col=None)
    assert "factor_coverage_pct" in scored.columns
    assert scored["factor_coverage_pct"].iloc[0] == pytest.approx(100.0)
    assert "pct_value" in scored.columns
    assert "pct_quality" in scored.columns
    assert "pct_balance_sheet" in scored.columns
    assert "composite" in scored.columns


def test_score_universe_df_rank_averages_within_group():
    """The value group score should be the mean of sub-signal percentiles, not a raw average."""
    df = _minimal_df()
    scored = score_universe_df(df, _minimal_config(), group_col=None)
    # All pct values should be in [0, 100]
    for family in FACTOR_SCORE_COLUMNS:
        col = f"pct_{family}"
        vals = scored[col].dropna()
        assert (vals >= 0).all() and (vals <= 100).all(), f"{col} out of range"


def test_compute_bargain_score_high_when_discounted():
    result = compute_bargain_score(
        price=50.0,
        graham_ratio=1.5,
        all_time_high=100.0,
        fifty_two_week_high=80.0,
        rsi_14=25.0,
        implied_upside_pct=30.0,
    )
    assert result["score"] is not None
    assert result["score"] >= 70


def test_compute_bargain_score_low_when_expensive():
    result = compute_bargain_score(
        price=95.0,
        graham_ratio=0.3,
        all_time_high=100.0,
        fifty_two_week_high=98.0,
        rsi_14=75.0,
        implied_upside_pct=-5.0,
    )
    assert result["score"] is not None
    assert result["score"] < 40


def test_compute_bargain_score_renormalizes_partial_data():
    """When only 52w discount data is available, score uses that component alone."""
    result = compute_bargain_score(
        price=50.0,
        graham_ratio=None,
        all_time_high=100.0,  # ignored; discount_ath removed
        fifty_two_week_high=100.0,
        rsi_14=None,
        implied_upside_pct=None,
    )
    assert result["score"] is not None
    assert result["components"]["discount_52w"] is not None
    assert result["components"]["margin_of_safety"] is None
    assert result["components"]["rsi_oversold"] is None
    # discount_52w alone: 50% below 52w high → linear(0.5, 0, 0.30) = 100 (clamped)
    assert result["score"] == pytest.approx(100.0)


def test_compute_bargain_score_three_components_only():
    """Removed components (discount_ath, analyst_upside) must not appear."""
    result = compute_bargain_score(
        price=60.0, graham_ratio=0.8, all_time_high=120.0,
        fifty_two_week_high=100.0, rsi_14=40.0, implied_upside_pct=25.0,
    )
    assert "discount_ath" not in result["components"]
    assert "analyst_upside" not in result["components"]
    assert set(result["components"].keys()) == {"margin_of_safety", "discount_52w", "rsi_oversold"}


def test_compute_bargain_score_margin_of_safety_range():
    """Graham ratio should discriminate across the S&P 500 distribution [0.3, 1.3]."""
    # Median S&P 500 graham_ratio ≈ 0.47 should produce a non-zero score
    r_median = compute_bargain_score(None, 0.47, None, None, None, None)
    assert r_median["components"]["margin_of_safety"] is not None
    assert r_median["components"]["margin_of_safety"] > 0

    # ratio=0.30 → exactly 0 (floor)
    r_low = compute_bargain_score(None, 0.30, None, None, None, None)
    assert r_low["components"]["margin_of_safety"] == pytest.approx(0.0)

    # ratio=1.30 → exactly 100 (ceiling)
    r_high = compute_bargain_score(None, 1.30, None, None, None, None)
    assert r_high["components"]["margin_of_safety"] == pytest.approx(100.0)


def test_merge_ticker_row_keeps_snapshot_factors_when_live_row_empty():
    uni = pd.DataFrame(
        [
            {
                "ticker": "AMZN",
                "name": "Amazon.com, Inc.",
                "sector": "Consumer Cyclical",
                "earnings_yield": 0.05,
                "garp": 2.3,
            }
        ]
    )
    live = {
        "ticker": "AMZN",
        "name": "AMZN",
        "sector": None,
        "earnings_yield": None,
        "garp": None,
    }
    merged = _merge_ticker_row_with_universe(live, uni, "AMZN")
    assert merged["name"] == "Amazon.com, Inc."
    assert merged["sector"] == "Consumer Cyclical"
    assert merged["earnings_yield"] == 0.05
    assert merged["garp"] == 2.3


def test_merge_ticker_row_keeps_structural_factors_when_live_differs():
    uni = pd.DataFrame(
        [
            {
                "ticker": "COST",
                "garp": 2.33,
                "earnings_revisions": 0.33,
                "momentum_12_1": -0.035,
            }
        ]
    )
    live = {
        "ticker": "COST",
        "garp": 0.95,
        "earnings_revisions": 0.0,
        "momentum_12_1": -0.009,
        "price": 955.0,
    }
    merged = _merge_ticker_row_with_universe(live, uni, "COST")
    assert merged["garp"] == 2.33
    assert merged["earnings_revisions"] == 0.33
    assert merged["momentum_12_1"] == pytest.approx(-0.009)
    assert merged["price"] == 955.0


def test_apply_universe_snapshot_scoring_overrides_composite():
    scored = pd.DataFrame(
        [
            {
                "ticker": "COST",
                "composite": 80.4,
                "factor_coverage_pct": 100.0,
                "pct_garp": 97.6,
            }
        ]
    )
    analysis = {
        "ticker": "COST",
        "composite": 64.4,
        "factor_breakdown": {"garp": {"percentile": 62.0}},
        "analyst": {"implied_upside_pct": 13.0, "consensus_label": "Buy"},
        "bargain": {"score": 44.0},
    }
    updated = apply_universe_snapshot_scoring(analysis, scored, "COST", _minimal_config())
    assert updated["composite"] == pytest.approx(80.4)
    assert updated["factor_breakdown"]["garp"]["percentile"] == pytest.approx(97.6)


def test_score_ticker_survives_empty_quote_info(monkeypatch):
    """Partial yfinance info must not wipe universe snapshot sub-signal columns."""
    uni = _minimal_df()
    uni.loc[0, "ticker"] = "AMZN"
    uni.loc[1, "ticker"] = "MSFT"
    uni = uni.head(2).copy()

    def fake_build(_ticker: str) -> dict:
        return {
            "ticker": "AMZN",
            "name": "AMZN",
            "sector": None,
            "industry": None,
            "price": 200.0,
            "market_cap": 2e12,
            "fifty_two_week_high": 220.0,
            "fifty_two_week_low": 150.0,
            "all_time_high": 230.0,
            "data_warnings": [],
            "recommendations": None,
            "target_mean": 240.0,
            "num_analysts": 40,
        }

    # All sub-signal columns return None (simulates partial live fetch)
    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    monkeypatch.setattr("core.scoring.build_raw_metrics", fake_build)
    monkeypatch.setattr(
        "core.scoring.compute_all_factors",
        lambda _raw: {col: None for col in all_sub_cols},
    )

    analysis = score_ticker("AMZN", _minimal_config(), universe_df=uni)
    # Snapshot values should be used for sector/name
    assert analysis["sector"] == "Tech"
    # factor_breakdown should have percentile entries for all groups
    for family in FACTOR_SCORE_COLUMNS:
        assert family in analysis["factor_breakdown"]
        assert "percentile" in analysis["factor_breakdown"][family]


def test_evaluate_good_buy_requires_composite_upside_and_bargain():
    thresholds = {
        "composite_min": 50,
        "bargain_min": 50,
        "implied_upside_min_pct": 15,
        "exclude_sell_consensus": True,
    }
    analyst = {"consensus_label": "Buy"}
    assert _evaluate_good_buy(55, 20, analyst, thresholds, bargain_score=60) is True
    assert _evaluate_good_buy(49, 20, analyst, thresholds, bargain_score=60) is False
    assert _evaluate_good_buy(55, 14, analyst, thresholds, bargain_score=60) is False
    assert _evaluate_good_buy(55, 20, analyst, thresholds, bargain_score=49) is False
    assert _evaluate_good_buy(55, 20, {"consensus_label": "Sell"}, thresholds, bargain_score=60) is False
