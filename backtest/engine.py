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
    DCA_INVESTMENT_USD,
    DEFAULT_DELIST_RETURN,
    ROLLING_WINDOW_MONTHS,
    TOP_QUINTILE_FRAC,
    TRAIN_END,
    VALID_END,
)
from backtest.data.prices import BENCHMARK_TICKER, load_delisted_catalog, load_prices
from backtest.factors import load_factor_panel
from backtest.weights import normalize_backtest_weights
from core.factors import FACTOR_SCORE_COLUMNS
from core.scoring import (
    _composite_and_coverage,
    cross_sectional_zscore,
    winsorize,
    zscore_to_percentile,
)

# Shared caches reused across tuning iterations.
_FORWARD_RETURNS: pd.DataFrame | None = None
_MONTHLY_RETURNS: pd.DataFrame | None = None
_QUARTER_END_PRICES: dict[tuple[pd.Timestamp, str], float] | None = None


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


def _forward_month_ends(qend: date, n_months: int = 3) -> list[pd.Timestamp]:
    """Month-end timestamps for the holding period following a quarter-end.

    A quarter-end selection made on (e.g.) 2010-03-31 earns the monthly returns
    labelled 2010-04-30, 2010-05-31, 2010-06-30 (the realized path until the next
    rebalance), so we map the picks onto those forward months.
    """
    base = pd.Timestamp(qend).to_period("M").to_timestamp("M")
    return [base + pd.offsets.MonthEnd(k) for k in range(1, n_months + 1)]


def _holdings_from_scored(
    scored: pd.DataFrame,
    start: date,
    end: date,
    holding_months: int = 3,
) -> dict[pd.Timestamp, list[str]]:
    """Map each holding month to the top-quintile names picked at the prior quarter-end.

    The top quintile selected at a quarter-end is held through the next
    ``holding_months`` months, producing a monthly (not quarterly) return path so
    that the rolling 36-month win-rate window can actually be filled.
    """
    holdings: dict[pd.Timestamp, list[str]] = {}
    for qend, grp in scored.groupby("quarter_end"):
        qdate = pd.Timestamp(qend).date() if not isinstance(qend, date) else qend
        if qdate < start or qdate > end:
            continue
        picks = _top_quintile_tickers(grp)
        if not picks:
            continue
        for month in _forward_month_ends(qdate, holding_months):
            holdings[month] = picks
    return holdings


def precompute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Monthly total returns for all tickers (cached for tuning)."""
    global _MONTHLY_RETURNS
    if _MONTHLY_RETURNS is not None:
        return _MONTHLY_RETURNS
    sub = prices.sort_values(["ticker", "Date"])
    rows: list[dict] = []
    for ticker, grp in sub.groupby("ticker"):
        series = grp.set_index("Date")["Close"].resample("ME").last().pct_change()
        for dt, ret in series.items():
            if pd.notna(ret):
                rows.append({"month": pd.Timestamp(dt), "ticker": ticker, "return": float(ret)})
    _MONTHLY_RETURNS = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["month", "ticker", "return"])
    return _MONTHLY_RETURNS


def precompute_forward_returns(panel: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Quarter-over-quarter forward returns using last close on or before each quarter-end."""
    global _FORWARD_RETURNS
    if _FORWARD_RETURNS is not None:
        return _FORWARD_RETURNS
    tickers = panel["ticker"].astype(str).unique()
    sub = prices[prices["ticker"].isin(tickers)].sort_values(["ticker", "Date"])
    qends = sorted(panel["quarter_end"].unique())
    rows: list[dict] = []
    for i, qend in enumerate(qends[:-1]):
        next_q = qends[i + 1]
        q_ts = pd.Timestamp(qend)
        n_ts = pd.Timestamp(next_q)
        p0 = sub[sub["Date"] <= q_ts].groupby("ticker")["Close"].last()
        p1 = sub[sub["Date"] <= n_ts].groupby("ticker")["Close"].last()
        for ticker in p0.index.intersection(p1.index):
            if p0[ticker] and p0[ticker] > 0:
                rows.append(
                    {
                        "quarter_end": next_q,
                        "ticker": ticker,
                        "fwd_return": float(p1[ticker] / p0[ticker] - 1.0),
                    }
                )
    _FORWARD_RETURNS = pd.DataFrame(rows)
    return _FORWARD_RETURNS


def precompute_quarter_end_prices(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[tuple[pd.Timestamp, str], float]:
    """Last close on or before each quarter-end, per ticker (for DCA simulation).

    Returns a dict keyed by ``(quarter_end_timestamp, ticker)`` so the DCA-CV
    objective can look up buy/mark prices in O(1) instead of filtering the full
    price panel on every quarter.
    """
    global _QUARTER_END_PRICES
    if _QUARTER_END_PRICES is not None:
        return _QUARTER_END_PRICES
    qends = [pd.Timestamp(q) for q in sorted(panel["quarter_end"].unique())]
    q_arr = np.array([np.datetime64(q) for q in qends])
    out: dict[tuple[pd.Timestamp, str], float] = {}
    sub = prices.sort_values(["ticker", "Date"])
    for ticker, grp in sub.groupby("ticker"):
        dates = grp["Date"].values.astype("datetime64[ns]")
        closes = grp["Close"].values
        idx = np.searchsorted(dates, q_arr, side="right") - 1
        tkr = str(ticker)
        for qi, q in enumerate(qends):
            j = idx[qi]
            if j >= 0:
                out[(q, tkr)] = float(closes[j])
    _QUARTER_END_PRICES = out
    return out


def make_quarter_folds(
    quarters,
    k_folds: int = 5,
) -> list[tuple[list[pd.Timestamp], pd.Timestamp]]:
    """Split the quarter timeline into k contiguous folds.

    Each fold is ``(quarters_in_fold, fold_end_quarter)``; the fold end is the
    quarter at which a fold's DCA campaign is marked to market.
    """
    qs = [pd.Timestamp(q) for q in sorted(quarters)]
    folds: list[tuple[list[pd.Timestamp], pd.Timestamp]] = []
    for chunk in np.array_split(np.array(qs), k_folds):
        members = [pd.Timestamp(q) for q in chunk.tolist()]
        if members:
            folds.append((members, members[-1]))
    return folds


def dca_fold_excess_roi(
    picks_by_quarter: dict[pd.Timestamp, list[str]],
    quarter_end_prices: dict[tuple[pd.Timestamp, str], float],
    fold_quarters: list[pd.Timestamp],
    fold_end: pd.Timestamp,
    *,
    delisted: set[str],
    delist_return: float = DEFAULT_DELIST_RETURN,
) -> float | None:
    """Excess ROI of a $20k/quarter top-N DCA campaign vs SPY within one fold.

    Each quarter in the fold, $20k is split equally across that quarter's picks
    and held (no rebalance) until ``fold_end``, where positions are marked. The
    same schedule is run for SPY. Returns ``strategy_ROI - spy_ROI`` on deployed
    capital, or ``None`` if no capital was deployed.
    """
    positions: dict[str, dict[str, float]] = {}
    invested = 0.0
    for q in fold_quarters:
        picks = picks_by_quarter.get(q, [])
        if not picks:
            continue
        priced = [t for t in picks if quarter_end_prices.get((q, t), 0.0) > 0]
        if not priced:
            continue
        per_stock = DCA_INVESTMENT_USD / len(priced)
        invested += per_stock * len(priced)
        for t in priced:
            px = quarter_end_prices[(q, t)]
            pos = positions.setdefault(t, {"shares": 0.0, "cost": 0.0})
            pos["shares"] += per_stock / px
            pos["cost"] += per_stock
    if invested <= 0:
        return None

    terminal = 0.0
    for t, pos in positions.items():
        end_px = quarter_end_prices.get((fold_end, t))
        if end_px is not None and end_px > 0:
            terminal += pos["shares"] * end_px
        elif t in delisted:
            terminal += pos["cost"] * (1.0 + delist_return)
        # else: no terminal price and not flagged delisted -> treat as zero.
    strat_roi = terminal / invested - 1.0

    spy_shares = 0.0
    spy_invested = 0.0
    for q in fold_quarters:
        px = quarter_end_prices.get((q, BENCHMARK_TICKER))
        if px and px > 0:
            spy_shares += DCA_INVESTMENT_USD / px
            spy_invested += DCA_INVESTMENT_USD
    spy_end = quarter_end_prices.get((fold_end, BENCHMARK_TICKER))
    if spy_invested <= 0 or not spy_end:
        return None
    spy_roi = spy_shares * spy_end / spy_invested - 1.0

    return strat_roi - spy_roi


def init_backtest_cache(
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load panel/prices and warm monthly + forward return caches."""
    global _FORWARD_RETURNS, _MONTHLY_RETURNS, _QUARTER_END_PRICES
    _FORWARD_RETURNS = None
    _MONTHLY_RETURNS = None
    _QUARTER_END_PRICES = None
    panel = panel if panel is not None else load_factor_panel()
    prices = prices if prices is not None else load_prices()
    monthly = precompute_monthly_returns(prices)
    forward = precompute_forward_returns(panel, prices)
    return panel, monthly, forward


def _portfolio_monthly_returns_from_cache(
    holdings_by_month: dict[pd.Timestamp, list[str]],
    monthly_returns: pd.DataFrame,
    delist_return: float = DEFAULT_DELIST_RETURN,
) -> pd.Series:
    if not holdings_by_month or monthly_returns.empty:
        return pd.Series(dtype=float)
    delisted = load_delisted_catalog()
    rets: list[tuple[pd.Timestamp, float]] = []
    for month, tickers in sorted(holdings_by_month.items()):
        if not tickers:
            continue
        sub = monthly_returns[
            (monthly_returns["month"] == month) & (monthly_returns["ticker"].isin(tickers))
        ]
        if sub.empty:
            continue
        values = sub.set_index("ticker")["return"].copy()
        for t in tickers:
            if t not in values.index and t in delisted:
                values.loc[t] = delist_return
        if values.empty:
            continue
        rets.append((month, float(values.mean())))
    if not rets:
        return pd.Series(dtype=float)
    return pd.Series({m: r for m, r in rets}).sort_index()


def _benchmark_monthly_returns_from_cache(
    monthly_returns: pd.DataFrame,
    start: date,
    end: date,
    ticker: str = BENCHMARK_TICKER,
) -> pd.Series:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sub = monthly_returns[
        (monthly_returns["ticker"] == ticker)
        & (monthly_returns["month"] >= start_ts.to_period("M").to_timestamp("M"))
        & (monthly_returns["month"] <= end_ts.to_period("M").to_timestamp("M"))
    ]
    if sub.empty:
        return pd.Series(dtype=float)
    return sub.set_index("month")["return"].sort_index()


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
    return precompute_forward_returns(panel, prices)


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
    *,
    skip_ic: bool = False,
    holding_months: int = 3,
    monthly_returns: pd.DataFrame | None = None,
    forward_returns: pd.DataFrame | None = None,
    scored: pd.DataFrame | None = None,
) -> BacktestResult:
    """Run quarterly top-quintile backtest for a weight configuration."""
    panel = panel if panel is not None else load_factor_panel()
    if monthly_returns is None or forward_returns is None:
        prices = prices if prices is not None else load_prices()
        monthly_returns = monthly_returns or precompute_monthly_returns(prices)
        forward_returns = forward_returns or precompute_forward_returns(panel, prices)
    # Scoring is split-independent, so callers may pass a pre-scored panel to
    # avoid re-scoring the full panel once per train/valid/test split.
    if scored is None:
        scored = score_factor_panel(panel, factor_weights)
    holdings = _holdings_from_scored(scored, start, end, holding_months)

    # Benchmark must cover the portfolio's forward-held tail months so the rolling
    # window is not truncated by an inner join against a shorter benchmark series.
    bench_end = (pd.Timestamp(end) + pd.offsets.MonthEnd(holding_months)).date()
    port_rets = _portfolio_monthly_returns_from_cache(holdings, monthly_returns, delist_return)
    bench_rets = _benchmark_monthly_returns_from_cache(monthly_returns, start, bench_end)

    win_rate, median_excess = _rolling_win_rate(port_rets, bench_rets)
    if skip_ic:
        mean_ic = 0.0
    else:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        scored_window = scored[
            (pd.to_datetime(scored["quarter_end"]) >= start_ts)
            & (pd.to_datetime(scored["quarter_end"]) <= end_ts)
        ]
        mean_ic = _compute_ic(scored_window, forward_returns)

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
