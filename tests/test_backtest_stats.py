"""Tests for backtest statistics helpers."""

import numpy as np
import pandas as pd
import pytest

from backtest.stats import (
    block_bootstrap_ci,
    decile_spread,
    n_independent_windows,
    newey_west_tstat,
)


def test_newey_west_tstat_known_series():
    series = [1.0, 2.0, 3.0]
    tstat = newey_west_tstat(series, lag=1)
    expected = 2.0 / np.sqrt((2.0 / 3.0) / 3.0)
    assert tstat == pytest.approx(expected)


def test_newey_west_ar1_mean_tstat_finite():
    rng = np.random.default_rng(0)
    n = 200
    phi = 0.5
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.2 + phi * x[t - 1] + rng.normal() * 0.5
    tstat = newey_west_tstat(x, lag=4)
    assert np.isfinite(tstat)
    # Positive-mean AR(1) should produce a clearly positive t-stat.
    assert tstat > 2.0


def test_block_bootstrap_reproducible():
    vals = list(np.linspace(-0.02, 0.03, 40))
    a = block_bootstrap_ci(vals, block=12, n_boot=200, seed=42)
    b = block_bootstrap_ci(vals, block=12, n_boot=200, seed=42)
    assert a == b
    assert a["ci_low"] <= a["mean"] <= a["ci_high"]


def test_decile_spread_monotone():
    qends = pd.date_range("2015-03-31", periods=8, freq="QE")
    rows = []
    fwd_rows = []
    for q in qends:
        for i in range(30):
            score = float(i)
            fwd = float(i) / 100.0
            rows.append({"quarter_end": q, "ticker": f"T{i:02d}", "score": score})
            fwd_rows.append({"as_of_quarter": q, "ticker": f"T{i:02d}", "fwd_3y": fwd})
    scored = pd.DataFrame(rows)
    fwd = pd.DataFrame(fwd_rows)
    out = decile_spread(scored, fwd, "score", "3y")
    assert out["spread"] > 0
    assert out["n_quarters"] >= 1


def test_n_independent_windows():
    assert n_independent_windows(60, 12) == 5.0
    assert n_independent_windows(20, 4) == 5.0
