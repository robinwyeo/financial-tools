"""Backtest engine: scoring, portfolio simulation, objective metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from backtest.constants import (
    BACKTEST_END,
    BACKTEST_FACTOR_FAMILIES,
    BACKTEST_START,
    BOOTSTRAP_CI,
    BOOTSTRAP_N,
    DCA_INVESTMENT_USD,
    DEFAULT_DELIST_RETURN,
    FORWARD_HORIZON_QUARTERS,
    PRIMARY_EVAL_HORIZON,
    ROLLING_WINDOW_MONTHS,
    TOP_QUINTILE_FRAC,
    TRAIN_END,
    TRANSACTION_COST_BPS,
    VALID_END,
)
from backtest.data.prices import BENCHMARK_TICKER, load_delisted_catalog, load_prices
from backtest.factors import load_factor_panel
from backtest.weights import normalize_backtest_weights
from core.factors import FACTOR_SCORE_COLUMNS
from core.scoring import (
    _composite_and_coverage,
    _score_column,
    cross_sectional_zscore,
    winsorize,
    zscore_to_percentile,
)

# Shared caches reused across tuning iterations.
_FORWARD_RETURNS: pd.DataFrame | None = None
_MULTI_HORIZON_RETURNS: pd.DataFrame | None = None
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
    horizon_ics: dict[str, float] = field(default_factory=dict)


_SECTOR_MAP: pd.Series | None = None


def _load_sector_map() -> pd.Series | None:
    """Cached ticker→sector map from the live universe snapshot."""
    global _SECTOR_MAP
    if _SECTOR_MAP is not None:
        return _SECTOR_MAP if not _SECTOR_MAP.empty else None
    try:
        from core.universe import load_universe_snapshot

        snap = load_universe_snapshot()
    except Exception:
        _SECTOR_MAP = pd.Series(dtype=object)
        return None
    if snap is None or snap.empty or "sector" not in snap.columns:
        _SECTOR_MAP = pd.Series(dtype=object)
        return None
    _SECTOR_MAP = (
        snap.assign(ticker=snap["ticker"].astype(str).str.upper())
        .drop_duplicates("ticker")
        .set_index("ticker")["sector"]
    )
    return _SECTOR_MAP


def _attach_sector_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Attach sector from the live universe snapshot when the panel lacks it."""
    if "sector" in df.columns and df["sector"].notna().any():
        return df
    sector_map = _load_sector_map()
    if sector_map is None or sector_map.empty:
        return df
    out = df.copy()
    out["sector"] = out["ticker"].astype(str).str.upper().map(sector_map)
    return out


def _score_panel_quarter(
    quarter_df: pd.DataFrame,
    factor_weights: dict[str, float],
    *,
    use_sector: bool = True,
) -> pd.DataFrame:
    """Score one quarter cross-sectionally, matching live sector-adjusted scoring."""
    result = quarter_df
    weights = normalize_backtest_weights(factor_weights)
    group_col = "sector" if use_sector and "sector" in result.columns else None

    for family in BACKTEST_FACTOR_FAMILIES:
        cols = FACTOR_SCORE_COLUMNS.get(family, [])
        sub_series: list[pd.Series] = []
        for col in cols:
            if col not in result.columns:
                continue
            if group_col:
                z = _score_column(result, col, group_col)
            else:
                z = cross_sectional_zscore(winsorize(result[col]))
            sub_series.append(z.apply(zscore_to_percentile))
        if sub_series:
            result[f"pct_{family}"] = pd.concat(sub_series, axis=1).mean(axis=1, skipna=True)
        else:
            result[f"pct_{family}"] = np.nan

    composites = []
    for _, row in result.iterrows():
        composite, _ = _composite_and_coverage(row, weights)
        composites.append(composite)
    result["composite"] = composites
    return result


def score_factor_panel(
    panel: pd.DataFrame,
    factor_weights: dict[str, float],
    *,
    use_sector: bool = True,
) -> pd.DataFrame:
    """Score full factor panel quarter by quarter."""
    panel = _attach_sector_if_missing(panel) if use_sector else panel
    frames = []
    for qend, grp in panel.groupby("quarter_end"):
        frames.append(_score_panel_quarter(grp, factor_weights, use_sector=use_sector))
    return pd.concat(frames, ignore_index=True)


def _top_quintile_tickers(scored_q: pd.DataFrame) -> list[str]:
    valid = scored_q.dropna(subset=["composite"]).sort_values("composite", ascending=False)
    if valid.empty:
        return []
    n = max(1, int(np.ceil(len(valid) * TOP_QUINTILE_FRAC)))
    return valid.head(n)["ticker"].astype(str).tolist()


def _forward_month_ends(qend: date, n_months: int = 3) -> list[pd.Timestamp]:
    """Month-end timestamps for the holding period following a quarter-end."""
    base = pd.Timestamp(qend).to_period("M").to_timestamp("M")
    return [base + pd.offsets.MonthEnd(k) for k in range(1, n_months + 1)]


def _holdings_from_scored(
    scored: pd.DataFrame,
    start: date,
    end: date,
    holding_months: int = 3,
) -> dict[pd.Timestamp, list[str]]:
    """Map each holding month to the top-quintile names picked at the prior quarter-end."""
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
    """Next-quarter forward returns (legacy 1q horizon)."""
    global _FORWARD_RETURNS
    if _FORWARD_RETURNS is not None:
        return _FORWARD_RETURNS
    multi = precompute_multi_horizon_returns(panel, prices)
    if multi.empty:
        _FORWARD_RETURNS = pd.DataFrame(columns=["quarter_end", "ticker", "fwd_return"])
        return _FORWARD_RETURNS
    out = multi[["as_of_quarter", "ticker", "fwd_1q"]].rename(
        columns={"as_of_quarter": "quarter_end", "fwd_1q": "fwd_return"}
    )
    # Match legacy labeling: return row tagged with the quarter it ENDS at.
    qends = sorted(panel["quarter_end"].unique())
    next_map = {qends[i]: qends[i + 1] for i in range(len(qends) - 1)}
    out = out.copy()
    out["quarter_end"] = out["quarter_end"].map(next_map)
    out = out.dropna(subset=["quarter_end", "fwd_return"])
    _FORWARD_RETURNS = out
    return _FORWARD_RETURNS


def precompute_multi_horizon_returns(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Forward returns at 1q / 1y / 3y / 5y horizons, labeled by the selection quarter.

    Columns: as_of_quarter, ticker, fwd_1q, fwd_1y, fwd_3y, fwd_5y,
             excess_1q, excess_1y, excess_3y, excess_5y (vs SPY).
    """
    global _MULTI_HORIZON_RETURNS
    if _MULTI_HORIZON_RETURNS is not None:
        return _MULTI_HORIZON_RETURNS

    tickers = panel["ticker"].astype(str).unique().tolist()
    if BENCHMARK_TICKER not in tickers:
        tickers = list(tickers) + [BENCHMARK_TICKER]
    sub = prices[prices["ticker"].isin(tickers)].sort_values(["ticker", "Date"])
    qends = sorted(pd.to_datetime(panel["quarter_end"].unique()))
    q_arr = np.array([np.datetime64(q) for q in qends])

    # Last close on/before each quarter-end per ticker.
    px: dict[str, np.ndarray] = {}
    for ticker, grp in sub.groupby("ticker"):
        dates = grp["Date"].values.astype("datetime64[ns]")
        closes = grp["Close"].values.astype(float)
        idx = np.searchsorted(dates, q_arr, side="right") - 1
        arr = np.full(len(qends), np.nan)
        valid = idx >= 0
        arr[valid] = closes[idx[valid]]
        px[str(ticker)] = arr

    spy = px.get(BENCHMARK_TICKER)
    rows: list[dict] = []
    for i, qend in enumerate(qends):
        for ticker, arr in px.items():
            if ticker == BENCHMARK_TICKER:
                continue
            p0 = arr[i]
            if not (p0 and p0 > 0 and not np.isnan(p0)):
                continue
            row: dict[str, Any] = {
                "as_of_quarter": pd.Timestamp(qend),
                "ticker": ticker,
            }
            for name, n_q in FORWARD_HORIZON_QUARTERS.items():
                j = i + n_q
                col = f"fwd_{name}"
                exc = f"excess_{name}"
                if j >= len(qends):
                    row[col] = np.nan
                    row[exc] = np.nan
                    continue
                p1 = arr[j]
                if not (p1 and p1 > 0 and not np.isnan(p1)):
                    row[col] = np.nan
                    row[exc] = np.nan
                    continue
                fwd = float(p1 / p0 - 1.0)
                row[col] = fwd
                if spy is not None and spy[i] > 0 and not np.isnan(spy[i]) and spy[j] > 0 and not np.isnan(spy[j]):
                    spy_fwd = float(spy[j] / spy[i] - 1.0)
                    row[exc] = fwd - spy_fwd
                else:
                    row[exc] = np.nan
            rows.append(row)

    _MULTI_HORIZON_RETURNS = pd.DataFrame(rows) if rows else pd.DataFrame()
    return _MULTI_HORIZON_RETURNS


def precompute_quarter_end_prices(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[tuple[pd.Timestamp, str], float]:
    """Last close on or before each quarter-end, per ticker (for DCA simulation)."""
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
    """Split the quarter timeline into k contiguous folds."""
    qs = [pd.Timestamp(q) for q in sorted(quarters)]
    folds: list[tuple[list[pd.Timestamp], pd.Timestamp]] = []
    for chunk in np.array_split(np.array(qs), k_folds):
        members = [pd.Timestamp(q) for q in chunk.tolist()]
        if members:
            folds.append((members, members[-1]))
    return folds


def make_expanding_window_folds(
    quarters,
    min_train_quarters: int = 20,
) -> list[tuple[list[pd.Timestamp], list[pd.Timestamp]]]:
    """Expanding-window walk-forward: train on [0..t), test on fold block."""
    qs = [pd.Timestamp(q) for q in sorted(quarters)]
    if len(qs) <= min_train_quarters + 4:
        return []
    # Evaluate on successive 8-quarter (2y) blocks after the min train window.
    folds: list[tuple[list[pd.Timestamp], list[pd.Timestamp]]] = []
    start = min_train_quarters
    while start < len(qs):
        end = min(start + 8, len(qs))
        train = qs[:start]
        test = qs[start:end]
        if test:
            folds.append((train, test))
        start = end
    return folds


def bootstrap_mean_ci(
    values: list[float] | np.ndarray,
    *,
    n_boot: int = BOOTSTRAP_N,
    ci: float = BOOTSTRAP_CI,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap confidence interval for the mean of ``values``."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": float(arr.mean()),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }


def dca_fold_excess_roi(
    picks_by_quarter: dict[pd.Timestamp, list[str]],
    quarter_end_prices: dict[tuple[pd.Timestamp, str], float],
    fold_quarters: list[pd.Timestamp],
    fold_end: pd.Timestamp,
    *,
    delisted: set[str],
    delist_return: float = DEFAULT_DELIST_RETURN,
    transaction_cost_bps: float = TRANSACTION_COST_BPS,
) -> float | None:
    """Excess ROI of a $20k/quarter gated DCA campaign vs SPY within one fold."""
    cost_frac = transaction_cost_bps / 10_000.0
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
            # Apply buy-side transaction cost by reducing shares purchased.
            effective_px = px * (1.0 + cost_frac)
            pos = positions.setdefault(t, {"shares": 0.0, "cost": 0.0})
            pos["shares"] += per_stock / effective_px
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
    strat_roi = terminal / invested - 1.0

    spy_shares = 0.0
    spy_invested = 0.0
    for q in fold_quarters:
        px = quarter_end_prices.get((q, BENCHMARK_TICKER))
        if px and px > 0:
            effective_px = px * (1.0 + cost_frac)
            spy_shares += DCA_INVESTMENT_USD / effective_px
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
    global _FORWARD_RETURNS, _MONTHLY_RETURNS, _QUARTER_END_PRICES, _MULTI_HORIZON_RETURNS
    _FORWARD_RETURNS = None
    _MONTHLY_RETURNS = None
    _QUARTER_END_PRICES = None
    _MULTI_HORIZON_RETURNS = None
    panel = panel if panel is not None else load_factor_panel()
    prices = prices if prices is not None else load_prices()
    monthly = precompute_monthly_returns(prices)
    forward = precompute_forward_returns(panel, prices)
    precompute_multi_horizon_returns(panel, prices)
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


def _spearman_ic(a: pd.Series, b: pd.Series) -> float | None:
    if len(a) < 10:
        return None
    ra = a.rank()
    rb = b.rank()
    ic = float(np.corrcoef(ra, rb)[0, 1])
    return None if np.isnan(ic) else ic


def _compute_ic(scored: pd.DataFrame, forward_returns: pd.DataFrame) -> float:
    """Mean Spearman IC between composite and next-quarter return."""
    ics: list[float] = []
    qends = sorted(scored["quarter_end"].unique())
    for i, qend in enumerate(qends[:-1]):
        next_q = qends[i + 1]
        q_df = scored[scored["quarter_end"] == qend][["ticker", "composite"]].dropna()
        fwd = forward_returns[forward_returns["quarter_end"] == next_q][["ticker", "fwd_return"]]
        merged = q_df.merge(fwd, on="ticker", how="inner")
        ic = _spearman_ic(merged["composite"], merged["fwd_return"])
        if ic is not None:
            ics.append(ic)
    return float(np.mean(ics)) if ics else 0.0


def compute_horizon_ics(
    scored: pd.DataFrame,
    multi_horizon: pd.DataFrame,
    score_col: str = "composite",
) -> dict[str, float]:
    """Mean Spearman IC of ``score_col`` vs each forward horizon."""
    if multi_horizon is None or multi_horizon.empty:
        return {}
    out: dict[str, float] = {}
    scored = scored.copy()
    scored["as_of_quarter"] = pd.to_datetime(scored["quarter_end"])
    for name in FORWARD_HORIZON_QUARTERS:
        col = f"fwd_{name}"
        if col not in multi_horizon.columns:
            continue
        ics: list[float] = []
        for qend, grp in scored.groupby("as_of_quarter"):
            s = grp[["ticker", score_col]].dropna()
            fwd = multi_horizon[multi_horizon["as_of_quarter"] == qend][["ticker", col]].dropna()
            merged = s.merge(fwd, on="ticker", how="inner")
            ic = _spearman_ic(merged[score_col], merged[col])
            if ic is not None:
                ics.append(ic)
        out[name] = float(np.mean(ics)) if ics else 0.0
    return out


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
    use_sector: bool = True,
) -> BacktestResult:
    """Run quarterly top-quintile backtest for a weight configuration."""
    panel = panel if panel is not None else load_factor_panel()
    if monthly_returns is None or forward_returns is None:
        prices = prices if prices is not None else load_prices()
        monthly_returns = monthly_returns or precompute_monthly_returns(prices)
        forward_returns = forward_returns or precompute_forward_returns(panel, prices)
    if scored is None:
        scored = score_factor_panel(panel, factor_weights, use_sector=use_sector)
    holdings = _holdings_from_scored(scored, start, end, holding_months)

    bench_end = (pd.Timestamp(end) + pd.offsets.MonthEnd(holding_months)).date()
    port_rets = _portfolio_monthly_returns_from_cache(holdings, monthly_returns, delist_return)
    bench_rets = _benchmark_monthly_returns_from_cache(monthly_returns, start, bench_end)

    win_rate, median_excess = _rolling_win_rate(port_rets, bench_rets)
    horizon_ics: dict[str, float] = {}
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
        if prices is not None or _MULTI_HORIZON_RETURNS is not None:
            prices_for_h = prices if prices is not None else load_prices()
            multi = precompute_multi_horizon_returns(panel, prices_for_h)
            horizon_ics = compute_horizon_ics(scored_window, multi)

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
        horizon_ics=horizon_ics,
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
    primary_ic = result.horizon_ics.get(PRIMARY_EVAL_HORIZON, result.mean_ic)
    return (result.rolling_win_rate, result.median_excess, primary_ic)
