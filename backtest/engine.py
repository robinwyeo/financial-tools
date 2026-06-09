"""Backtest engine: scoring, portfolio simulation, objective metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from backtest.constants import (
    BACKTEST_END,
    BACKTEST_FACTOR_FAMILIES,
    BACKTEST_START,
    DEFAULT_DELIST_RETURN,
    ROLLING_WINDOW_MONTHS,
    TOP_QUINTILE_FRAC,
    TRAIN_END,
    VALID_END,
)
from backtest.data.prices import benchmark_monthly_returns, load_prices, portfolio_monthly_returns
from backtest.factors import load_factor_panel
from backtest.weights import normalize_backtest_weights
from core.factors import FACTOR_SCORE_COLUMNS
from core.scoring import (
    _composite_and_coverage,
    cross_sectional_zscore,
    winsorize,
    zscore_to_percentile,
)


@dataclass
class BacktestResult:
    factor_weights: dict[str, float]
    monthly_returns: pd.Series
    benchmark_returns: pd.Series
    rolling_win_rate: float
    median_excess: float
    mean_ic: float
    max_drawdown: float
    cagr: float
    benchmark_cagr: float
    period_start: date
    period_end: date


def _score_panel_quarter(
    quarter_df: pd.DataFrame,
    factor_weights: dict[str, float],
) -> pd.DataFrame:
    """Score one quarter cross-sectionally (universe-wide)."""
    result = quarter_df.copy()
    weights = normalize_backtest_weights(factor_weights)

    for family in BACKTEST_FACTOR_FAMILIES:
        col = FACTOR_SCORE_COLUMNS[family]
        if col not in result.columns:
            result[f"pct_{family}"] = np.nan
            continue
        z = cross_sectional_zscore(winsorize(result[col]))
        result[f"pct_{family}"] = z.apply(zscore_to_percentile)

    composites = []
    for _, row in result.iterrows():
        composite, _ = _composite_and_coverage(row, weights)
        composites.append(composite)
    result["composite"] = composites
    return result


def score_factor_panel(
    panel: pd.DataFrame,
    factor_weights: dict[str, float],
) -> pd.DataFrame:
    """Score full factor panel quarter by quarter."""
    frames = []
    for qend, grp in panel.groupby("quarter_end"):
        frames.append(_score_panel_quarter(grp, factor_weights))
    return pd.concat(frames, ignore_index=True)


def _top_quintile_tickers(scored_q: pd.DataFrame) -> list[str]:
    valid = scored_q.dropna(subset=["composite"]).sort_values("composite", ascending=False)
    if valid.empty:
        return []
    n = max(1, int(np.ceil(len(valid) * TOP_QUINTILE_FRAC)))
    return valid.head(n)["ticker"].astype(str).tolist()


def _holdings_from_scored(
    scored: pd.DataFrame,
    start: date,
    end: date,
) -> dict[pd.Timestamp, list[str]]:
    holdings: dict[pd.Timestamp, list[str]] = {}
    for qend, grp in scored.groupby("quarter_end"):
        qdate = pd.Timestamp(qend).date() if not isinstance(qend, date) else qend
        if qdate < start or qdate > end:
            continue
        month = pd.Timestamp(qdate).to_period("M").to_timestamp("M")
        holdings[month] = _top_quintile_tickers(grp)
    return holdings


def _compute_ic(scored: pd.DataFrame, forward_returns: pd.DataFrame) -> float:
    """Mean Spearman IC between composite and next-quarter return."""
    ics: list[float] = []
    qends = sorted(scored["quarter_end"].unique())
    for i, qend in enumerate(qends[:-1]):
        next_q = qends[i + 1]
        q_df = scored[scored["quarter_end"] == qend][["ticker", "composite"]].dropna()
        fwd = forward_returns[forward_returns["quarter_end"] == next_q][["ticker", "fwd_return"]]
        merged = q_df.merge(fwd, on="ticker", how="inner")
        if len(merged) < 10:
            continue
        ic = merged["composite"].corr(merged["fwd_return"], method="spearman")
        if ic is not None and not np.isnan(ic):
            ics.append(float(ic))
    return float(np.mean(ics)) if ics else 0.0


def _forward_quarter_returns(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    qends = sorted(panel["quarter_end"].unique())
    for i, qend in enumerate(qends[:-1]):
        next_q = qends[i + 1]
        q_ts = pd.Timestamp(qend)
        n_ts = pd.Timestamp(next_q)
        tickers = panel[panel["quarter_end"] == qend]["ticker"].astype(str).unique()
        for ticker in tickers:
            p0 = prices[
                (prices["ticker"] == ticker) & (prices["Date"] <= q_ts)
            ].sort_values("Date")
            p1 = prices[
                (prices["ticker"] == ticker) & (prices["Date"] <= n_ts)
            ].sort_values("Date")
            if p0.empty or p1.empty:
                continue
            r = float(p1.iloc[-1]["Close"] / p0.iloc[-1]["Close"] - 1.0)
            rows.append({"quarter_end": next_q, "ticker": ticker, "fwd_return": r})
    return pd.DataFrame(rows)


def _rolling_win_rate(
    port_rets: pd.Series,
    bench_rets: pd.Series,
    window: int = ROLLING_WINDOW_MONTHS,
) -> tuple[float, float]:
    aligned = pd.concat([port_rets, bench_rets], axis=1, join="inner")
    aligned.columns = ["port", "bench"]
    if len(aligned) < window:
        return 0.0, 0.0
    port_cum = (1 + aligned["port"]).rolling(window).apply(np.prod, raw=True) - 1
    bench_cum = (1 + aligned["bench"]).rolling(window).apply(np.prod, raw=True) - 1
    excess = port_cum - bench_cum
    valid = excess.dropna()
    if valid.empty:
        return 0.0, 0.0
    win_rate = float((valid > 0).mean())
    median_excess = float(valid.median())
    return win_rate, median_excess


def _cagr(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    total = float((1 + returns).prod() - 1)
    years = len(returns) / 12.0
    if years <= 0:
        return 0.0
    return float((1 + total) ** (1 / years) - 1)


def _max_drawdown(returns: pd.Series) -> float:
    if returns.empty:
        return 0.0
    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def run_backtest(
    factor_weights: dict[str, float],
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    start: date = BACKTEST_START,
    end: date = BACKTEST_END,
    delist_return: float = DEFAULT_DELIST_RETURN,
) -> BacktestResult:
    """Run quarterly top-quintile backtest for a weight configuration."""
    panel = panel if panel is not None else load_factor_panel()
    prices = prices if prices is not None else load_prices()
    scored = score_factor_panel(panel, factor_weights)
    holdings = _holdings_from_scored(scored, start, end)

    port_rets = portfolio_monthly_returns(holdings, prices, delist_return)
    bench_rets = benchmark_monthly_returns(prices, start, end)

    win_rate, median_excess = _rolling_win_rate(port_rets, bench_rets)
    fwd = _forward_quarter_returns(panel, prices)
    mean_ic = _compute_ic(scored, fwd)

    return BacktestResult(
        factor_weights=normalize_backtest_weights(factor_weights),
        monthly_returns=port_rets,
        benchmark_returns=bench_rets,
        rolling_win_rate=win_rate,
        median_excess=median_excess,
        mean_ic=mean_ic,
        max_drawdown=_max_drawdown(port_rets),
        cagr=_cagr(port_rets),
        benchmark_cagr=_cagr(bench_rets),
        period_start=start,
        period_end=end,
    )


def split_period(name: str) -> tuple[date, date]:
    if name == "train":
        return BACKTEST_START, TRAIN_END
    if name == "valid":
        return date(2019, 3, 31), VALID_END
    if name == "test":
        return date(2023, 3, 31), BACKTEST_END
    raise ValueError(f"Unknown period: {name}")


def objective_tuple(result: BacktestResult) -> tuple[float, float, float]:
    """Higher is better: win rate, median excess, mean IC."""
    return (result.rolling_win_rate, result.median_excess, result.mean_ic)
