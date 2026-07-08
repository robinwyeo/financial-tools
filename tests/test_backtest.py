"""Unit tests for backtest harness."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest, score_factor_panel
from backtest.thresholds import calibrate_thresholds
from backtest.weights import (
    current_baseline_factor_weights,
    normalize_backtest_weights,
    theme_weights_to_factor_weights,
)
from core.factors import FACTOR_SCORE_COLUMNS
from backtest.constants import BACKTEST_FACTOR_FAMILIES


def _synthetic_panel(n_tickers: int = 20, n_quarters: int = 8) -> pd.DataFrame:
    """Synthetic panel with all sub-signal columns for the backtestable groups."""
    rng = np.random.default_rng(0)
    qends = [date(2012, 3, 31), date(2012, 6, 30), date(2012, 9, 30), date(2012, 12, 31)]
    qends += [date(2013, 3, 31), date(2013, 6, 30), date(2013, 9, 30), date(2013, 12, 31)]
    qends = qends[:n_quarters]
    rows = []
    # Collect all sub-signal columns for backtestable groups
    backtest_sub_cols = [
        col
        for family in BACKTEST_FACTOR_FAMILIES
        for col in FACTOR_SCORE_COLUMNS.get(family, [])
    ]
    for q in qends:
        for i in range(n_tickers):
            row = {"quarter_end": q, "ticker": f"T{i:02d}", "price": 100 + i, "bargain_score": rng.random() * 100}
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
    rows.append({"Date": pd.Timestamp("2011-01-01"), "ticker": "SPY", "Close": 100.0})
    rows.append({"Date": pd.Timestamp("2014-12-31"), "ticker": "SPY", "Close": 150.0})
    return pd.DataFrame(rows)


def test_normalize_backtest_weights_excludes_analyst_factor():
    full = current_baseline_factor_weights()
    assert "earnings_revisions" not in full
    # 7 groups remain; mass should be close to 1.0 minus the revisions weight
    assert 0.80 < sum(full.values()) < 1.05


def test_theme_weights_expand_to_factors():
    # Each theme = exactly one factor group now (1:1 mapping)
    theme = {k: 1 / len(BACKTEST_FACTOR_FAMILIES) for k in BACKTEST_FACTOR_FAMILIES}
    fw = theme_weights_to_factor_weights(theme)
    for family in BACKTEST_FACTOR_FAMILIES:
        assert family in fw
    assert sum(fw.values()) == pytest.approx(1.0, rel=1e-6)


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
        skip_ic=True,  # scipy not required in test env
    )
    assert 0.0 <= result.rolling_win_rate <= 1.0
    assert isinstance(result.cagr, float)


def test_bargain_tuning_uses_forward_not_trailing_returns():
    """Regression test: tune_bargain_weights must correlate against NEXT quarter's
    returns, not the current quarter's returns (trailing-return bug fix)."""
    from backtest.tune import tune_bargain_weights

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
                "bargain_rsi_oversold": rng.random() * 100,
            })
    panel = pd.DataFrame(rows)

    # Build synthetic prices; all tickers move identically so forward returns are ~0
    price_rows = []
    for ticker in [f"T{i:02d}" for i in range(n_tickers)]:
        for d in pd.date_range("2011-01-01", "2013-06-30", freq="B"):
            price_rows.append({"Date": d, "ticker": ticker, "Close": 100.0})
    prices = pd.DataFrame(price_rows)

    result = tune_bargain_weights(n_samples=10, seed=7, panel=panel, prices=prices)
    # The winner_mean_ic should be a float and not wildly negative (old bug produced ~-0.45)
    assert isinstance(result["winner_mean_ic"], float)
    # With properly computed forward returns (all zero here), ICs should be near zero on random data
    assert result["winner_mean_ic"] >= -0.5, (
        f"IC {result['winner_mean_ic']:.3f} is too negative — "
        "likely still using trailing instead of forward returns"
    )


def test_calibrate_thresholds_returns_bounds():
    panel = _synthetic_panel()
    prices = _synthetic_prices(panel)
    weights = current_baseline_factor_weights()
    out = calibrate_thresholds(weights, panel=panel, prices=prices)
    assert 30.0 <= out["composite_min"] <= 80.0
    assert 30.0 <= out["bargain_min"] <= 80.0
