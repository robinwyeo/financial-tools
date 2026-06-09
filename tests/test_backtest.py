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


def _synthetic_panel(n_tickers: int = 20, n_quarters: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    qends = [date(2012, 3, 31), date(2012, 6, 30), date(2012, 9, 30), date(2012, 12, 31)]
    qends += [date(2013, 3, 31), date(2013, 6, 30), date(2013, 9, 30), date(2013, 12, 31)]
    qends = qends[:n_quarters]
    rows = []
    for q in qends:
        for i in range(n_tickers):
            row = {"quarter_end": q, "ticker": f"T{i:02d}", "price": 100 + i, "bargain_score": rng.random() * 100}
            for col in FACTOR_SCORE_COLUMNS.values():
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
    # Renormalized mass excludes the 7% earnings_revisions slot from production weights.
    assert 0.85 < sum(full.values()) < 1.05


def test_theme_weights_expand_to_factors():
    theme = {k: 1 / 6 for k in ["value", "quality", "trend", "risk", "solvency", "capital_allocation"]}
    fw = theme_weights_to_factor_weights(theme)
    assert "momentum" in fw
    assert "value" in fw
    assert sum(fw.values()) == pytest.approx(1.0, rel=1e-6)


def test_score_factor_panel_adds_composite():
    panel = _synthetic_panel()
    weights = normalize_backtest_weights(current_baseline_factor_weights())
    scored = score_factor_panel(panel, weights)
    assert "composite" in scored.columns
    assert scored["composite"].notna().any()


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
    )
    assert 0.0 <= result.rolling_win_rate <= 1.0
    assert isinstance(result.cagr, float)


def test_calibrate_thresholds_returns_bounds():
    panel = _synthetic_panel()
    prices = _synthetic_prices(panel)
    weights = current_baseline_factor_weights()
    # Monkeypatch via direct call with scored internals using synthetic panel only.
    out = calibrate_thresholds(weights, panel=panel, prices=prices)
    assert 30.0 <= out["composite_min"] <= 80.0
    assert 30.0 <= out["bargain_min"] <= 80.0
