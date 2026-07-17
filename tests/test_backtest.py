"""Unit tests for backtest harness."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.constants import BACKTEST_FACTOR_FAMILIES, BARGAIN_BACKTEST_COMPONENTS
from backtest.engine import (
    bootstrap_mean_ci,
    precompute_multi_horizon_returns,
    run_backtest,
    score_factor_panel,
)
from backtest.factors import enrich_factor_panel
from backtest.thresholds import calibrate_thresholds
from backtest.weights import (
    current_baseline_factor_weights,
    named_weight_candidates,
    normalize_backtest_weights,
    theme_weights_to_factor_weights,
)
from core.data import percentile_rank_in_history
from core.factors import FACTOR_SCORE_COLUMNS


def _synthetic_panel(n_tickers: int = 20, n_quarters: int = 8) -> pd.DataFrame:
    """Synthetic panel with all sub-signal columns for the backtestable groups."""
    rng = np.random.default_rng(0)
    qends = [date(2012, 3, 31), date(2012, 6, 30), date(2012, 9, 30), date(2012, 12, 31)]
    qends += [date(2013, 3, 31), date(2013, 6, 30), date(2013, 9, 30), date(2013, 12, 31)]
    qends = qends[:n_quarters]
    rows = []
    backtest_sub_cols = [
        col
        for family in BACKTEST_FACTOR_FAMILIES
        for col in FACTOR_SCORE_COLUMNS.get(family, [])
    ]
    for q in qends:
        for i in range(n_tickers):
            row = {
                "quarter_end": q,
                "ticker": f"T{i:02d}",
                "price": 100 + i,
                "sector": "Tech" if i % 2 == 0 else "Health",
                "bargain_score": rng.random() * 100,
                "graham_ratio": 0.5 + rng.random(),
                "bargain_discount_52w": rng.random() * 100,
            }
            for col in backtest_sub_cols:
                row[col] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_prices(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tickers = panel["ticker"].unique()
    for ticker in tickers:
        for d in pd.date_range("2011-01-01", "2014-12-31", freq="B"):
            rows.append({"Date": d, "ticker": ticker, "Close": 100.0})
    # Mild upward SPY path for excess-return math.
    for d in pd.date_range("2011-01-01", "2014-12-31", freq="B"):
        rows.append({"Date": d, "ticker": "SPY", "Close": 100.0 + (d - pd.Timestamp("2011-01-01")).days * 0.01})
    return pd.DataFrame(rows)


def test_normalize_backtest_weights_excludes_analyst_factor():
    full = current_baseline_factor_weights()
    assert "earnings_revisions" not in full
    assert 0.80 < sum(full.values()) < 1.05


def test_theme_weights_expand_to_factors():
    theme = {k: 1 / len(BACKTEST_FACTOR_FAMILIES) for k in BACKTEST_FACTOR_FAMILIES}
    fw = theme_weights_to_factor_weights(theme)
    for family in BACKTEST_FACTOR_FAMILIES:
        assert family in fw
    assert sum(fw.values()) == pytest.approx(1.0, rel=1e-6)


def test_named_weight_candidates_include_evidence_based():
    cands = named_weight_candidates()
    assert set(cands.keys()) == {"evidence_based", "legacy_tuned", "equal"}
    assert cands["evidence_based"]["quality"] > cands["legacy_tuned"]["quality"]


def test_score_factor_panel_adds_composite():
    panel = _synthetic_panel()
    weights = normalize_backtest_weights(current_baseline_factor_weights())
    scored = score_factor_panel(panel, weights)
    assert "composite" in scored.columns
    assert scored["composite"].notna().any()


def test_score_factor_panel_pct_columns_in_range():
    """All group percentile columns should be in [0, 100]."""
    panel = _synthetic_panel()
    weights = normalize_backtest_weights(current_baseline_factor_weights())
    scored = score_factor_panel(panel, weights)
    for family in BACKTEST_FACTOR_FAMILIES:
        col = f"pct_{family}"
        assert col in scored.columns, f"Missing {col}"
        vals = scored[col].dropna()
        assert (vals >= 0).all() and (vals <= 100).all(), f"{col} out of [0,100]"


def test_run_backtest_returns_metrics():
    panel = _synthetic_panel()
    prices = _synthetic_prices(panel)
    weights = current_baseline_factor_weights()
    result = run_backtest(
        weights,
        panel=panel,
        prices=prices,
        start=date(2012, 3, 31),
        end=date(2013, 12, 31),
        skip_ic=True,
    )
    assert 0.0 <= result.rolling_win_rate <= 1.0
    assert isinstance(result.cagr, float)


def test_multi_horizon_returns_have_expected_columns():
    panel = _synthetic_panel(n_quarters=8)
    prices = _synthetic_prices(panel)
    multi = precompute_multi_horizon_returns(panel, prices)
    assert not multi.empty
    for col in ("fwd_1q", "fwd_1y", "fwd_3y", "excess_1q"):
        assert col in multi.columns


def test_bootstrap_mean_ci_bounds():
    ci = bootstrap_mean_ci([0.1, 0.2, 0.15, 0.05, 0.12], n_boot=200, seed=1)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_percentile_rank_in_history():
    assert percentile_rank_in_history(0.10, [0.05, 0.08, 0.10, 0.12, 0.15]) == pytest.approx(60.0)
    assert percentile_rank_in_history(None, [0.1, 0.2, 0.3, 0.4]) is None
    assert percentile_rank_in_history(0.1, [0.1, 0.2]) is None  # need >= 4


def test_enrich_factor_panel_adds_valuation_bargain():
    panel = _synthetic_panel(n_tickers=5, n_quarters=8)
    # Ensure earnings_yield varies so history percentiles are defined.
    panel["earnings_yield"] = np.linspace(0.02, 0.12, len(panel))
    enriched = enrich_factor_panel(panel)
    assert "valuation_vs_history" in enriched.columns
    assert "bargain_valuation_vs_history" in enriched.columns
    assert "bargain_rsi_oversold" not in enriched.columns
    for comp in BARGAIN_BACKTEST_COMPONENTS:
        assert f"bargain_{comp}" in enriched.columns


def test_bargain_validation_uses_forward_horizons():
    """validate_bargain_weights should run and return a primary IC float."""
    from backtest.tune import validate_bargain_weights

    rng = np.random.default_rng(42)
    qends = [date(2012, 3, 31), date(2012, 6, 30), date(2012, 9, 30), date(2012, 12, 31)]
    n_tickers = 30
    rows = []
    for q in qends:
        for i in range(n_tickers):
            rows.append({
                "quarter_end": q,
                "ticker": f"T{i:02d}",
                "price": 100 + i,
                "bargain_score": rng.random() * 100,
                "bargain_margin_of_safety": rng.random() * 100,
                "bargain_discount_52w": rng.random() * 100,
                "bargain_valuation_vs_history": rng.random() * 100,
                "earnings_yield": 0.05 + rng.random() * 0.05,
                "graham_ratio": 0.5 + rng.random(),
            })
    panel = pd.DataFrame(rows)

    price_rows = []
    for ticker in [f"T{i:02d}" for i in range(n_tickers)] + ["SPY"]:
        for d in pd.date_range("2011-01-01", "2014-06-30", freq="B"):
            price_rows.append({"Date": d, "ticker": ticker, "Close": 100.0})
    prices = pd.DataFrame(price_rows)

    # Clear engine caches so synthetic prices are used.
    import backtest.engine as eng
    eng._FORWARD_RETURNS = None
    eng._MULTI_HORIZON_RETURNS = None
    eng._MONTHLY_RETURNS = None
    eng._QUARTER_END_PRICES = None

    result = validate_bargain_weights(panel=panel, prices=prices)
    assert isinstance(result["winner_mean_ic"], float)
    assert "valuation_vs_history" in result["winner_weights"]
    assert "rsi_oversold" not in result["winner_weights"]


def test_calibrate_thresholds_returns_bounds():
    panel = _synthetic_panel()
    prices = _synthetic_prices(panel)
    weights = current_baseline_factor_weights()

    import backtest.engine as eng
    eng._FORWARD_RETURNS = None
    eng._MULTI_HORIZON_RETURNS = None
    eng._MONTHLY_RETURNS = None
    eng._QUARTER_END_PRICES = None

    out = calibrate_thresholds(weights, panel=panel, prices=prices, horizon="1q")
    assert 30.0 <= out["composite_min"] <= 80.0
    assert 30.0 <= out["bargain_min"] <= 80.0
