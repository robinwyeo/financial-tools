"""Tests for cross-sectional scoring."""

import numpy as np
import pandas as pd
import pytest

from core.factors import FACTOR_SCORE_COLUMNS, FACTOR_SUB_BUCKETS
from core.scoring import (
    _composite_and_coverage,
    _evaluate_good_buy,
    _merge_ticker_row_with_universe,
    apply_universe_snapshot_scoring,
    compute_bargain_score,
    compute_family_percentile,
    compute_fund_bargain_score,
    is_distressed,
    rank_percentile,
    score_ticker,
    score_universe_df,
)


def _minimal_config() -> dict:
    weights = {family: 1.0 / len(FACTOR_SCORE_COLUMNS) for family in FACTOR_SCORE_COLUMNS}
    return {
        "factor_weights": weights,
        "universe": {"sector_scoring": False},
        "thresholds": {
            "composite_min": 50,
            "bargain_min": 50,
            "require_implied_upside": False,
            "exclude_sell_consensus": True,
        },
        "decision": {"mode": "legacy"},
    }


def _minimal_df() -> pd.DataFrame:
    """Minimal universe dataframe with all sub-signal columns for the factor groups."""
    base = {
        "sector": "Tech",
        "earnings_yield": 0.08,
        "fcf_yield": 0.06,
        "book_to_market": 0.3,
        "graham_ratio": 0.9,  # kept for the bargain score; not a composite sub-signal
        "garp": 1.5,
        "gross_profitability": 0.4,
        "roe": 0.15,
        "roa": 0.08,
        "profit_margin": 0.12,
        "roic": 0.18,
        "earnings_quality": 0.02,
        "financial_strength": 7.0,
        "net_cash_to_mcap": -0.05,
        "low_leverage": 0.5,
        "altman_z": 3.5,
        "momentum_12_1": 0.15,
        "low_volatility": 8.0,
        "shareholder_yield": 0.03,
        "investment": -0.05,
        "revision_agreement": 0.6,
        "revision_magnitude": 0.02,
        "earnings_surprise": 0.04,
        "insider_buying": 0.001,
    }
    rows = []
    for i, ticker in enumerate(["AAA", "BBB", "CCC"]):
        row = {"ticker": ticker, **base}
        row["earnings_yield"] = base["earnings_yield"] * (1 + i * 0.1)
        row["momentum_12_1"] = base["momentum_12_1"] * (1 - i * 0.1)
        rows.append(row)
    return pd.DataFrame(rows)


def test_rank_percentile_is_empirical():
    """Percentiles are (rank - 0.5) / n * 100 with average ranks for ties."""
    pct = rank_percentile(pd.Series([1.0, 2.0, 3.0, 4.0]))
    assert pct.tolist() == pytest.approx([12.5, 37.5, 62.5, 87.5])

    # Extreme outliers cannot distort spacing — ranks are distribution-free.
    pct_outlier = rank_percentile(pd.Series([1.0, 2.0, 3.0, 1e9]))
    assert pct_outlier.tolist() == pytest.approx([12.5, 37.5, 62.5, 87.5])

    constant = rank_percentile(pd.Series([5.0, 5.0, 5.0]))
    assert constant.tolist() == pytest.approx([50.0, 50.0, 50.0])

    too_few = rank_percentile(pd.Series([1.0, 2.0]))
    assert too_few.isna().all()


def test_compute_family_percentile_buckets_decorrelate():
    """Bucketed signals contribute per-bucket, not per-column."""
    n = 5
    df = pd.DataFrame(
        {
            # Two perfectly correlated signals in one bucket...
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [10.0, 20.0, 30.0, 40.0, 50.0],
            # ...and one anti-correlated signal in its own bucket.
            "c": [5.0, 4.0, 3.0, 2.0, 1.0],
        }
    )
    flat = compute_family_percentile(df, ["a", "b", "c"])
    bucketed = compute_family_percentile(df, ["a", "b", "c"], buckets=[["a", "b"], ["c"]])

    pct_a = rank_percentile(df["a"])
    pct_c = rank_percentile(df["c"])
    # Flat: (a + b + c) / 3 = (2*pct_a + pct_c) / 3; bucketed: (pct_a + pct_c) / 2 = 50.
    assert flat.tolist() == pytest.approx(((2 * pct_a + pct_c) / 3).tolist())
    assert bucketed.tolist() == pytest.approx([50.0] * n)


def test_quality_group_uses_sub_buckets():
    """A profitability juggernaut with poor accruals/strength lands mid-pack on quality."""
    assert "quality" in FACTOR_SUB_BUCKETS
    covered = {c for bucket in FACTOR_SUB_BUCKETS["quality"] for c in bucket}
    assert covered == set(FACTOR_SCORE_COLUMNS["quality"])


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
    for family in FACTOR_SCORE_COLUMNS:
        col = f"pct_{family}"
        vals = scored[col].dropna()
        assert (vals >= 0).all() and (vals <= 100).all(), f"{col} out of range"


def test_compute_bargain_score_high_when_discounted():
    result = compute_bargain_score(
        price=50.0,
        graham_ratio=1.5,
        fifty_two_week_high=80.0,
        valuation_vs_history=90.0,
    )
    assert result["score"] is not None
    assert result["score"] >= 70


def test_compute_bargain_score_low_when_expensive():
    result = compute_bargain_score(
        price=95.0,
        graham_ratio=0.3,
        fifty_two_week_high=98.0,
        valuation_vs_history=10.0,
    )
    assert result["score"] is not None
    assert result["score"] < 40


def test_compute_bargain_score_renormalizes_partial_data():
    """When only 52w discount data is available, score uses that component alone."""
    result = compute_bargain_score(
        price=50.0,
        graham_ratio=None,
        fifty_two_week_high=100.0,
        valuation_vs_history=None,
    )
    assert result["score"] is not None
    assert result["components"]["discount_52w"] is not None
    assert result["components"]["margin_of_safety"] is None
    assert result["components"]["valuation_vs_history"] is None
    # discount_52w alone: 50% below 52w high → linear(0.5, 0, 0.30) = 100 (clamped)
    assert result["score"] == pytest.approx(100.0)


def test_compute_bargain_score_three_components_only():
    """RSI / ATH / analyst upside must not appear in bargain components."""
    result = compute_bargain_score(
        price=60.0,
        graham_ratio=0.8,
        fifty_two_week_high=100.0,
        valuation_vs_history=55.0,
        rsi_14=40.0,
        all_time_high=120.0,
        implied_upside_pct=25.0,
    )
    assert "discount_ath" not in result["components"]
    assert "analyst_upside" not in result["components"]
    assert "rsi_oversold" not in result["components"]
    assert set(result["components"].keys()) == {
        "margin_of_safety",
        "valuation_vs_history",
        "discount_52w",
    }


def test_compute_bargain_score_margin_of_safety_range():
    """Graham ratio should discriminate across the S&P 500 distribution [0.3, 1.3]."""
    r_median = compute_bargain_score(None, 0.47, None, None)
    assert r_median["components"]["margin_of_safety"] is not None
    assert r_median["components"]["margin_of_safety"] > 0

    r_low = compute_bargain_score(None, 0.30, None, None)
    assert r_low["components"]["margin_of_safety"] == pytest.approx(0.0)

    r_high = compute_bargain_score(None, 1.30, None, None)
    assert r_high["components"]["margin_of_safety"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Fund bargain score
# ---------------------------------------------------------------------------

def test_compute_fund_bargain_score_high_when_discounted_and_oversold():
    """Deep 52W discount + oversold RSI → high bargain score."""
    result = compute_fund_bargain_score(
        price=70.0,
        fifty_two_week_high=100.0,
        rsi_14=25.0,
    )
    assert result["score"] is not None
    assert result["score"] >= 80


def test_compute_fund_bargain_score_low_when_near_high_and_overbought():
    """Near 52W high + overbought RSI → low bargain score."""
    result = compute_fund_bargain_score(
        price=98.0,
        fifty_two_week_high=100.0,
        rsi_14=75.0,
    )
    assert result["score"] is not None
    assert result["score"] < 20


def test_compute_fund_bargain_score_renormalizes_missing_rsi():
    """When RSI is missing, only 52W discount drives the score."""
    result = compute_fund_bargain_score(
        price=70.0,
        fifty_two_week_high=100.0,
        rsi_14=None,
    )
    assert result["score"] is not None
    # 30% off 52W high → discount_52w = 100 (clamped), only component available
    assert result["score"] == pytest.approx(100.0)


def test_compute_fund_bargain_score_none_when_no_data():
    result = compute_fund_bargain_score(
        price=None,
        fifty_two_week_high=None,
        rsi_14=None,
    )
    assert result["score"] is None


def test_compute_fund_bargain_score_components_keys():
    result = compute_fund_bargain_score(price=80.0, fifty_two_week_high=100.0, rsi_14=50.0)
    assert set(result["components"]) == {"discount_52w", "rsi_oversold"}


def test_compute_fund_bargain_score_rsi_boundary():
    """RSI exactly 30 → rsi_oversold = 100; RSI exactly 70 → rsi_oversold = 0."""
    r_oversold = compute_fund_bargain_score(price=None, fifty_two_week_high=None, rsi_14=30.0)
    assert r_oversold["components"]["rsi_oversold"] == pytest.approx(100.0)

    r_overbought = compute_fund_bargain_score(price=None, fifty_two_week_high=None, rsi_14=70.0)
    assert r_overbought["components"]["rsi_oversold"] == pytest.approx(0.0)


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
                "roe": 0.33,
                "momentum_12_1": -0.035,
            }
        ]
    )
    live = {
        "ticker": "COST",
        "garp": 0.95,
        "roe": 0.0,
        "momentum_12_1": -0.009,
        "price": 955.0,
    }
    merged = _merge_ticker_row_with_universe(live, uni, "COST")
    assert merged["garp"] == 2.33
    assert merged["roe"] == 0.33
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
        "bargain": {"score": 55.0},
    }
    updated = apply_universe_snapshot_scoring(analysis, scored, "COST", _minimal_config())
    assert updated["composite"] == pytest.approx(80.4)
    assert updated["factor_breakdown"]["garp"]["percentile"] == pytest.approx(97.6)
    # Upside below 15% must not block good-buy when require_implied_upside is false.
    assert updated["is_good_buy"] is True


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
            "valuation_vs_history": 60.0,
            "data_warnings": [],
            "recommendations": None,
            "target_mean": 240.0,
            "num_analysts": 40,
        }

    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    monkeypatch.setattr("core.scoring.build_raw_metrics", fake_build)
    monkeypatch.setattr(
        "core.scoring.compute_all_factors",
        lambda _raw: {col: None for col in all_sub_cols},
    )

    analysis = score_ticker("AMZN", _minimal_config(), universe_df=uni)
    assert analysis["sector"] == "Tech"
    assert "data_quality" in analysis
    for family in FACTOR_SCORE_COLUMNS:
        assert family in analysis["factor_breakdown"]
        assert "percentile" in analysis["factor_breakdown"][family]


def test_score_ticker_propagates_listing_currency(monkeypatch):
    uni = _minimal_df()
    uni.loc[0, "ticker"] = "SHOP.TO"
    uni = uni.head(1).copy()

    def fake_build(_ticker: str) -> dict:
        return {
            "ticker": "SHOP.TO",
            "name": "Shopify",
            "sector": "Technology",
            "industry": None,
            "price": 73.0,
            "price_listing": 100.0,
            "market_cap": 73e9,
            "market_cap_listing": 100e9,
            "currency": "CAD",
            "financial_currency": "USD",
            "fifty_two_week_high": 80.0,
            "fifty_two_week_high_listing": 110.0,
            "fifty_two_week_low": 50.0,
            "data_warnings": [],
            "data_quality": {"grade": "B"},
            "recommendations": None,
            "target_mean": 90.0,
            "target_mean_listing": 123.0,
            "num_analysts": 10,
        }

    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    monkeypatch.setattr("core.scoring.is_fund", lambda t: False)
    monkeypatch.setattr("core.scoring.build_raw_metrics", fake_build)
    monkeypatch.setattr(
        "core.scoring.compute_all_factors",
        lambda _raw: {col: None for col in all_sub_cols},
    )
    analysis = score_ticker("SHOP.TO", _minimal_config(), universe_df=uni)
    assert analysis["currency"] == "CAD"
    assert analysis["price"] == pytest.approx(100.0)
    assert analysis["analyst"]["target_mean"] == pytest.approx(123.0)


def test_evaluate_good_buy_requires_composite_and_bargain_not_upside():
    thresholds = {
        "composite_min": 50,
        "bargain_min": 50,
        "implied_upside_min_pct": 15,
        "require_implied_upside": False,
        "exclude_sell_consensus": True,
    }
    analyst = {"consensus_label": "Buy"}
    assert _evaluate_good_buy(55, 20, analyst, thresholds, bargain_score=60) is True
    assert _evaluate_good_buy(49, 20, analyst, thresholds, bargain_score=60) is False
    # Low upside no longer blocks.
    assert _evaluate_good_buy(55, 5, analyst, thresholds, bargain_score=60) is True
    assert _evaluate_good_buy(55, None, analyst, thresholds, bargain_score=60) is True
    assert _evaluate_good_buy(55, 20, analyst, thresholds, bargain_score=49) is False
    assert _evaluate_good_buy(55, 20, {"consensus_label": "Sell"}, thresholds, bargain_score=60) is False


def test_evaluate_good_buy_coverage_gate():
    thresholds = {
        "composite_min": 50,
        "bargain_min": 50,
        "coverage_min_pct": 70,
        "require_implied_upside": False,
        "exclude_sell_consensus": True,
    }
    analyst = {"consensus_label": "Buy"}
    common = dict(bargain_score=60)
    assert _evaluate_good_buy(55, 20, analyst, thresholds, factor_coverage_pct=100.0, **common) is True
    assert _evaluate_good_buy(55, 20, analyst, thresholds, factor_coverage_pct=69.9, **common) is False
    # Rows without a coverage figure are not blocked (e.g. legacy snapshots).
    assert _evaluate_good_buy(55, 20, analyst, thresholds, factor_coverage_pct=None, **common) is True


def test_evaluate_good_buy_altman_distress_gate():
    thresholds = {
        "composite_min": 50,
        "bargain_min": 50,
        "altman_zpp_min": 1.1,
        "require_implied_upside": False,
        "exclude_sell_consensus": True,
    }
    analyst = {"consensus_label": "Buy"}
    common = dict(bargain_score=60)
    # Healthy Z'' passes; distress-zone Z'' blocks regardless of scores.
    assert _evaluate_good_buy(80, 20, analyst, thresholds, altman_z_pp=3.0, **common) is True
    assert _evaluate_good_buy(80, 20, analyst, thresholds, altman_z_pp=0.8, **common) is False
    # Missing Z never blocks (e.g. financials where Z is undefined).
    assert _evaluate_good_buy(80, 20, analyst, thresholds, altman_z_pp=None, **common) is True
    assert _evaluate_good_buy(80, 20, analyst, thresholds, altman_z_pp=float("nan"), **common) is True


def test_is_distressed():
    assert is_distressed(0.8) is True
    assert is_distressed(1.1) is False
    assert is_distressed(3.0) is False
    assert is_distressed(None) is False
    assert is_distressed(float("nan")) is False
    # Configurable cutoff (Z'').
    assert is_distressed(1.5, {"altman_zpp_min": 2.0}) is True
    # Financials remain exempt; utilities and RE are scored with Z''.
    assert is_distressed(0.8, None, "Financial Services") is False
    assert is_distressed(0.8, None, "Real Estate") is True
    assert is_distressed(0.8, None, "Utilities") is True
    assert is_distressed(0.8, None, "Technology") is True
    # Explicit Z'' wins over a legacy classic-Z argument.
    assert is_distressed(3.0, None, "Technology", altman_z_pp=0.5) is True


def test_evaluate_good_buy_uncertainty_widens_hurdles():
    thresholds = {
        "composite_min": 50,
        "bargain_min": 50,
        "require_implied_upside": False,
        "exclude_sell_consensus": True,
    }
    analyst = {"consensus_label": "Buy"}
    # 52 composite clears 50 but not 50+6.
    assert _evaluate_good_buy(52, 20, analyst, thresholds, bargain_score=60) is True
    assert _evaluate_good_buy(
        52, 20, analyst, thresholds, bargain_score=60, uncertainty_bump=6.0
    ) is False
    assert _evaluate_good_buy(
        57, 20, analyst, thresholds, bargain_score=57, uncertainty_bump=6.0
    ) is True


def test_evaluate_good_buy_optional_upside_gate():
    thresholds = {
        "composite_min": 50,
        "bargain_min": 50,
        "implied_upside_min_pct": 15,
        "require_implied_upside": True,
        "exclude_sell_consensus": True,
    }
    analyst = {"consensus_label": "Buy"}
    assert _evaluate_good_buy(55, 20, analyst, thresholds, bargain_score=60) is True
    assert _evaluate_good_buy(55, 14, analyst, thresholds, bargain_score=60) is False


def test_evaluate_good_buy_excludes_underperform():
    thresholds = {
        "composite_min": 50,
        "bargain_min": 50,
        "require_implied_upside": False,
        "exclude_sell_consensus": True,
        "exclude_underperform": True,
    }
    assert _evaluate_good_buy(
        55, 20, {"consensus_label": "Underperform"}, thresholds, bargain_score=60
    ) is False
    thresholds["exclude_underperform"] = False
    assert _evaluate_good_buy(
        55, 20, {"consensus_label": "Underperform"}, thresholds, bargain_score=60
    ) is True
    assert _evaluate_good_buy(
        55, 20, {"consensus_label": "Sell"}, thresholds, bargain_score=60
    ) is False
    thresholds["exclude_sell_consensus"] = False
    assert _evaluate_good_buy(
        55, 20, {"consensus_label": "Sell"}, thresholds, bargain_score=60
    ) is True


def test_composite_coverage_uses_sub_signal_fraction():
    """A quality group with 1 of 7 sub-signals is not 100% covered."""
    weights = {"quality": 1.0}
    families = {
        "quality": [
            "gross_profitability",
            "roe",
            "roa",
            "profit_margin",
            "roic",
            "earnings_quality",
            "financial_strength",
        ]
    }
    row = pd.Series({"pct_quality": 80.0, "gross_profitability": 0.4})
    composite, coverage = _composite_and_coverage(row, weights, families)
    assert composite == 80.0
    assert coverage == pytest.approx(100.0 / 7)


def test_vectorized_composite_matches_rowwise():
    df = _minimal_df()
    cfg = _minimal_config()
    scored = score_universe_df(df, cfg)
    weights = cfg["factor_weights"]
    for _, row in scored.iterrows():
        c, cov = _composite_and_coverage(row, weights)
        assert scored.loc[row.name, "composite"] == pytest.approx(c)
        assert scored.loc[row.name, "factor_coverage_pct"] == pytest.approx(cov)


def test_sector_fallback_when_group_too_small():
    """A 4-name sector falls back to universe-wide percentiles."""
    rows = []
    for i in range(8):
        row = {
            "ticker": f"T{i}",
            "sector": "Micro" if i < 4 else "Large",
            "earnings_yield": float(i + 1),
            "fcf_yield": 0.05,
            "book_to_market": 0.3,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    from core.scoring import _score_column

    pct = _score_column(df, "earnings_yield", "sector", min_group_size=5)
    # Micro names still get a percentile (universe fallback), not NaN.
    assert pct.notna().all()
    assert pct.iloc[0] < pct.iloc[-1]


def test_apply_universe_snapshot_scoring_keeps_decision_mode(monkeypatch):
    cfg = _minimal_config()
    uni = score_universe_df(_minimal_df(), cfg)
    analysis = {
        "ticker": "AAA",
        "composite": 10.0,
        "bargain": {"score": 10.0},
        "analyst": {"consensus_label": "Buy", "implied_upside_pct": 20},
        "factors_raw": {},
        "factor_coverage_pct": 90.0,
        "is_good_buy": False,
    }
    updated = apply_universe_snapshot_scoring(analysis, uni, "AAA", cfg)
    assert "composite" in updated
    assert updated["ticker"] == "AAA"
