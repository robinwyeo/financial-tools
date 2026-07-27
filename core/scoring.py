"""Cross-sectional scoring: winsorize, z-scores, percentiles, composite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.analysts import aggregate_analyst_data
from core.config import (
    get_bargain_weights,
    get_factor_weights,
    get_fund_factor_weights,
    get_thresholds,
    load_config,
)
from core.data import (
    build_fund_raw_metrics,
    build_raw_metrics,
    compute_valuation_vs_history,
    get_security_type,
    is_etf,
    is_fund,
)
from core.factors import FACTOR_SCORE_COLUMNS, compute_all_factors
from core.fund_factors import FUND_FACTOR_SCORE_COLUMNS, compute_fund_factors
from core.fund_universe import load_fund_universe_snapshot
from core.universe import load_universe_snapshot, snapshot_path
from core.watchlist import load_watchlist

# Long-horizon valuation bargain weights (RSI removed; kept as informational only).
BARGAIN_COMPONENT_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.40,
    "valuation_vs_history": 0.35,
    "discount_52w": 0.25,
}


def _linear_score(value: float, low: float, high: float) -> float:
    """Map value in [low, high] to 0-100, clamped."""
    if high <= low:
        return 0.0
    pct = (value - low) / (high - low)
    return float(max(0.0, min(100.0, pct * 100.0)))


def compute_bargain_score(
    price: float | None,
    graham_ratio: float | None,
    fifty_two_week_high: float | None,
    valuation_vs_history: float | None = None,
    component_weights: dict[str, float] | None = None,
    *,
    # Backward-compatible unused kwargs (removed from the score).
    all_time_high: float | None = None,
    rsi_14: float | None = None,
    implied_upside_pct: float | None = None,
) -> dict[str, Any]:
    """
    Absolute 0-100 bargain score for long-horizon entry (higher = more of a bargain).

    Components:
      margin_of_safety       — Graham ratio scored over [0.30, 1.30].
      valuation_vs_history   — current EBIT/EV percentile vs own 5y history (0-100).
      discount_52w           — % below 52-week high; linear 0%→0, 30%→100.

    RSI is intentionally excluded (short-horizon mean-reversion). Analyst upside
    is informational on the dashboard, not part of this score.
    Renormalizes weights over components with available data.
    """
    del all_time_high, rsi_14, implied_upside_pct  # retained only for call-site compat

    components: dict[str, float | None] = {
        "margin_of_safety": None,
        "valuation_vs_history": None,
        "discount_52w": None,
    }

    if graham_ratio is not None and graham_ratio > 0:
        components["margin_of_safety"] = _linear_score(graham_ratio, 0.30, 1.30)

    if valuation_vs_history is not None and not (
        isinstance(valuation_vs_history, float) and np.isnan(valuation_vs_history)
    ):
        components["valuation_vs_history"] = float(
            max(0.0, min(100.0, valuation_vs_history))
        )

    if (
        price is not None
        and fifty_two_week_high is not None
        and fifty_two_week_high > 0
        and price > 0
    ):
        discount_52w = 1.0 - (price / fifty_two_week_high)
        components["discount_52w"] = _linear_score(discount_52w, 0.0, 0.30)

    weights = component_weights or BARGAIN_COMPONENT_WEIGHTS
    weighted_sum = 0.0
    weight_available = 0.0
    for key, sub_score in components.items():
        if sub_score is None:
            continue
        w = weights.get(key, 0.0)
        weighted_sum += sub_score * w
        weight_available += w

    score = weighted_sum / weight_available if weight_available > 0 else None
    return {"score": score, "components": components}


# Weights for fund-specific bargain score. Fund financials don't exist, so only
# price-based signals are available: 52-week discount and RSI as an oversold proxy.
FUND_BARGAIN_COMPONENT_WEIGHTS: dict[str, float] = {
    "discount_52w": 0.65,
    "rsi_oversold": 0.35,
}


def compute_fund_bargain_score(
    price: float | None,
    fifty_two_week_high: float | None,
    rsi_14: float | None,
) -> dict[str, Any]:
    """
    Price-based bargain score for ETFs and mutual funds (0-100, higher = more of a bargain).

    Fund financial statements don't exist, so only price-relative signals are used:
      discount_52w  — % below 52-week high; linear 0%→0, 30%→100 (matches stock formula).
      rsi_oversold  — inverted RSI(14): RSI 30→100, RSI 70→0. Signals a dip vs recent trend.

    Renormalizes over available components when one signal is missing.
    """
    components: dict[str, float | None] = {
        "discount_52w": None,
        "rsi_oversold": None,
    }

    if (
        price is not None
        and fifty_two_week_high is not None
        and fifty_two_week_high > 0
        and price > 0
    ):
        discount_52w = 1.0 - (price / fifty_two_week_high)
        components["discount_52w"] = _linear_score(discount_52w, 0.0, 0.30)

    if rsi_14 is not None:
        # RSI 30 (oversold) → score 100; RSI 70 (overbought) → score 0.
        components["rsi_oversold"] = float(max(0.0, min(100.0, (70.0 - rsi_14) / 40.0 * 100.0)))

    weighted_sum = 0.0
    weight_available = 0.0
    for key, sub_score in components.items():
        if sub_score is None:
            continue
        w = FUND_BARGAIN_COMPONENT_WEIGHTS.get(key, 0.0)
        weighted_sum += sub_score * w
        weight_available += w

    score = weighted_sum / weight_available if weight_available > 0 else None
    return {"score": score, "components": components}


def _bargain_fields(
    raw: dict,
    factors: dict,
    analyst: dict,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build bargain score and related fields for analysis dict."""
    del analyst  # upside is informational; not used in bargain score
    valuation_vs_history = raw.get("valuation_vs_history")
    if valuation_vs_history is None:
        current_ey = raw.get("earnings_yield_current")
        if current_ey is None:
            current_ey = factors.get("earnings_yield")
        ticker = raw.get("ticker")
        if ticker:
            try:
                valuation_vs_history = compute_valuation_vs_history(
                    str(ticker), current_ey
                )
            except Exception:
                valuation_vs_history = None
    bargain = compute_bargain_score(
        price=raw.get("price"),
        graham_ratio=factors.get("graham_ratio"),
        fifty_two_week_high=raw.get("fifty_two_week_high"),
        valuation_vs_history=valuation_vs_history,
        component_weights=get_bargain_weights(config),
    )
    return {
        "all_time_high": raw.get("all_time_high"),
        "rsi_14": raw.get("rsi_14"),  # informational timing indicator
        "valuation_vs_history": valuation_vs_history,
        "bargain": bargain,
    }


def winsorize(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if series.dropna().empty:
        return series
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def cross_sectional_zscore(series: pd.Series) -> pd.Series:
    s = series.dropna()
    if len(s) < 3:
        return pd.Series(np.nan, index=series.index)
    mean = s.mean()
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def _score_column(
    df: pd.DataFrame,
    col: str,
    group_col: str | None,
    min_group_size: int = 5,
) -> pd.Series:
    """Z-score a column, using sector groups when large enough else universe-wide."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    winsorized = winsorize(df[col])

    if group_col and group_col in df.columns:
        def group_z(s: pd.Series) -> pd.Series:
            if s.dropna().shape[0] >= min_group_size:
                return cross_sectional_zscore(winsorize(s))
            return pd.Series(np.nan, index=s.index)

        z = winsorized.groupby(df[group_col]).transform(group_z)
        # Fallback to universe-wide for small sectors
        missing = z.isna() & winsorized.notna()
        if missing.any():
            universe_z = cross_sectional_zscore(winsorized)
            z = z.where(~missing, universe_z)
        return z

    return cross_sectional_zscore(winsorized)


def zscore_to_percentile(z: float | None) -> float | None:
    """Convert z-score to 0-100 percentile using normal CDF."""
    if z is None or np.isnan(z):
        return None
    from core.analysts import norm_cdf

    return float(norm_cdf(z) * 100)


def _composite_and_coverage(
    row: pd.Series,
    weights: dict[str, float],
    factor_columns: dict[str, list[str]] | None = None,
) -> tuple[float | None, float]:
    """Weighted composite using only groups with data; returns (composite, coverage_pct)."""
    families = factor_columns if factor_columns is not None else FACTOR_SCORE_COLUMNS
    weighted_sum = 0.0
    weight_available = 0.0
    weight_total = sum(weights.get(family, 0) for family in families)

    for family in families:
        pct_col = f"pct_{family}"
        if pct_col not in row.index:
            continue
        pct = row[pct_col]
        w = weights.get(family, 0)
        if pct is not None and not (isinstance(pct, float) and np.isnan(pct)):
            weighted_sum += float(pct) * w
            weight_available += w

    if weight_available == 0:
        return None, 0.0

    composite = weighted_sum / weight_available
    coverage = (weight_available / weight_total * 100.0) if weight_total > 0 else 0.0
    return composite, coverage


METADATA_OVERLAY_COLUMNS = frozenset({
    "name",
    "sector",
    "industry",
    "price",
    "market_cap",
})

LIVE_FACTOR_OVERLAY_COLUMNS = frozenset({
    "momentum_12_1",
    "low_volatility",
    "volatility_12m",
    "max_drawdown",
    "downside_deviation",
})

# Zero means "no signal" for these columns, not a measured value.
ZERO_NEUTRAL_COLUMNS: frozenset[str] = frozenset()


def _is_meaningful_value(val: Any, *, column: str | None = None) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if column in ZERO_NEUTRAL_COLUMNS and isinstance(val, (int, float)) and val == 0:
        return False
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped or stripped.lower() == "unknown":
            return False
    return True


def _should_overlay_live_value(key: str) -> bool:
    return key in METADATA_OVERLAY_COLUMNS or key in LIVE_FACTOR_OVERLAY_COLUMNS


def _merge_ticker_row_with_universe(row: dict, uni: pd.DataFrame, ticker: str) -> dict:
    """
    Overlay a freshly built ticker row onto the universe snapshot row.

    Only price-sensitive factors and display metadata are refreshed live.
    Structural financial factors (GARP, quality, balance sheet, etc.) stay on
    the snapshot so partial or noisy live fetches cannot corrupt scores.
    """
    existing = uni[uni["ticker"].astype(str).str.upper() == ticker]
    if existing.empty:
        return row
    merged = existing.iloc[0].to_dict()
    for key, val in row.items():
        if key == "ticker":
            continue
        if key == "name" and val == ticker:
            continue
        if not _should_overlay_live_value(key):
            continue
        if _is_meaningful_value(val, column=key):
            merged[key] = val
    merged["ticker"] = ticker
    return merged


def _resolved_display_field(
    raw: dict,
    row: dict,
    field: str,
    *,
    ticker: str,
) -> Any:
    """Prefer a meaningful live value, then universe snapshot, with name/ticker guard."""
    val = raw.get(field)
    if field == "name" and val == ticker:
        val = None
    if _is_meaningful_value(val):
        return val
    snap = row.get(field)
    if field == "name" and snap == ticker:
        return ticker
    return snap if _is_meaningful_value(snap) else val


def score_universe_df(
    factors_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    group_col: str | None = "sector",
    *,
    factor_columns: dict[str, list[str]] | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Score all tickers in a factors dataframe cross-sectionally.

    For each factor group: cross-sectionally rank each sub-signal (winsorize →
    sector z-score → normal-CDF percentile), then average available sub-signal
    percentiles into a single group percentile score. Composite = weighted average
    of group scores over groups with data.

    ``factor_columns``/``weights`` default to the stock factor groups; pass the
    fund factor groups (and fund weights) to score a fund universe.
    """
    cfg = config or load_config()
    families = factor_columns if factor_columns is not None else FACTOR_SCORE_COLUMNS
    if weights is None:
        weights = get_factor_weights(cfg)
    use_sector = cfg.get("universe", {}).get("sector_scoring", True) and group_col in factors_df.columns

    result = factors_df.copy()

    for family, cols in families.items():
        pct_col = f"pct_{family}"
        sub_series: list[pd.Series] = []

        for col in cols:
            if col not in result.columns:
                continue
            if use_sector:
                z = _score_column(result, col, group_col)
            else:
                z = cross_sectional_zscore(winsorize(result[col]))
            sub_series.append(z.apply(zscore_to_percentile))

        if sub_series:
            result[pct_col] = pd.concat(sub_series, axis=1).mean(axis=1, skipna=True)
        else:
            result[pct_col] = np.nan

    composites = []
    coverages = []
    for _, row in result.iterrows():
        composite, coverage = _composite_and_coverage(row, weights, families)
        composites.append(composite)
        coverages.append(coverage)
    result["composite"] = composites
    result["factor_coverage_pct"] = coverages

    return result


def score_ticker(
    ticker: str,
    config: dict[str, Any] | None = None,
    universe_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Score a single ticker against the universe snapshot.
    Returns full analysis dict for dashboard display.
    """
    ticker = ticker.upper().strip()
    cfg = config or load_config()

    if is_fund(ticker):
        # ETFs and mutual funds go through the fund pipeline (score_fund).
        return {
            "ticker": ticker,
            "is_etf": True,
            "is_fund": True,
            "security_type": get_security_type(ticker),
        }

    raw = build_raw_metrics(ticker)
    factors = compute_all_factors(raw)
    analyst = aggregate_analyst_data(raw)

    uni = universe_df if universe_df is not None else load_universe_snapshot()
    if uni is None or uni.empty:
        return _score_without_universe(ticker, raw, factors, analyst, cfg)

    # Build row for this ticker; merge with snapshot so partial live fetches do not wipe factors.
    row = {
        "ticker": ticker,
        "name": raw.get("name"),
        "sector": raw.get("sector"),
        "industry": raw.get("industry"),
        **factors,
    }
    row = _merge_ticker_row_with_universe(row, uni, ticker)
    ticker_df = pd.DataFrame([row])

    # Append to universe for cross-section (or replace if exists)
    combined = uni.copy()
    combined = combined[combined["ticker"] != ticker]
    combined = pd.concat([combined, ticker_df], ignore_index=True)

    scored = score_universe_df(combined, cfg)
    scored_row = scored[scored["ticker"] == ticker].iloc[0].to_dict()

    thresholds = get_thresholds(cfg)
    composite = scored_row.get("composite")
    factor_coverage_pct = scored_row.get("factor_coverage_pct")
    implied_upside = analyst.get("implied_upside_pct")
    bargain_data = _bargain_fields(raw, factors, analyst, cfg)
    bargain_score = (bargain_data.get("bargain") or {}).get("score")
    is_good_buy = _evaluate_good_buy(
        composite,
        implied_upside,
        analyst,
        thresholds,
        bargain_score=bargain_score,
        factor_coverage_pct=factor_coverage_pct,
    )

    factor_breakdown: dict[str, dict] = {}
    for family in FACTOR_SCORE_COLUMNS:
        factor_breakdown[family] = {
            "percentile": scored_row.get(f"pct_{family}"),
        }

    # Flatten all sub-signal columns for the raw values expander
    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    factors_raw = {col: row.get(col) for col in all_sub_cols}
    factors_raw["trailing_pe"] = raw.get("trailing_pe")

    return {
        "ticker": ticker,
        "name": _resolved_display_field(raw, row, "name", ticker=ticker),
        "sector": _resolved_display_field(raw, row, "sector", ticker=ticker),
        "industry": _resolved_display_field(raw, row, "industry", ticker=ticker),
        "exchange": raw.get("exchange"),
        "price": raw.get("price"),
        "market_cap": raw.get("market_cap") or row.get("market_cap"),
        "dividend_yield": raw.get("dividend_yield"),
        "fifty_two_week_high": raw.get("fifty_two_week_high"),
        "fifty_two_week_low": raw.get("fifty_two_week_low"),
        "is_etf": False,
        "composite": composite,
        "factor_coverage_pct": factor_coverage_pct,
        "factor_breakdown": factor_breakdown,
        "factors_raw": factors_raw,
        "analyst": analyst,
        "is_good_buy": is_good_buy,
        "data_warnings": raw.get("data_warnings", []),
        "scored_row": scored_row,
        **bargain_data,
    }


def score_universe(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Score entire universe snapshot."""
    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        return pd.DataFrame()
    return score_universe_df(uni, config)


def score_fund_universe(config: dict[str, Any] | None = None) -> pd.DataFrame:
    """Score the entire fund universe snapshot (US + Canadian ETFs and mutual funds)."""
    uni = load_fund_universe_snapshot()
    if uni is None or uni.empty:
        return pd.DataFrame()
    cfg = config or load_config()
    return score_universe_df(
        uni,
        cfg,
        group_col="category",
        factor_columns=FUND_FACTOR_SCORE_COLUMNS,
        weights=get_fund_factor_weights(cfg),
    )


def _fund_display_fields(raw: dict[str, Any]) -> dict[str, Any]:
    price = raw.get("price")
    nav = raw.get("nav_price")
    nav_premium = None
    if price is not None and nav is not None and nav > 0:
        nav_premium = (price / nav) - 1.0
    return {
        "ticker": raw.get("ticker"),
        "name": raw.get("name"),
        "security_type": raw.get("quote_type"),
        "category": raw.get("category"),
        "fund_family": raw.get("fund_family"),
        "currency": raw.get("currency"),
        "exchange": raw.get("exchange"),
        "price": price,
        "nav_price": nav,
        "nav_premium": nav_premium,
        "expense_ratio": raw.get("expense_ratio"),
        "total_assets": raw.get("total_assets"),
        "distribution_yield": raw.get("distribution_yield"),
        "beta_3y": raw.get("beta_3y"),
        "ytd_return": raw.get("ytd_return"),
        "return_1y": raw.get("return_1y"),
        "return_3y": raw.get("return_3y"),
        "return_5y": raw.get("return_5y"),
        "volatility_12m": raw.get("volatility_12m"),
        "max_drawdown": raw.get("max_drawdown"),
        "rsi_14": raw.get("rsi_14"),
        "fifty_two_week_high": raw.get("fifty_two_week_high"),
        "fifty_two_week_low": raw.get("fifty_two_week_low"),
        "all_time_high": raw.get("all_time_high"),
        "description": raw.get("description"),
        "is_etf": raw.get("quote_type") == "ETF",
        "is_fund": True,
    }


def score_fund(
    ticker: str,
    config: dict[str, Any] | None = None,
    fund_universe_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Score an ETF or mutual fund against the fund universe snapshot.

    Mirrors score_ticker but uses fund-appropriate factors (cost, performance,
    risk-adjusted return, volatility, momentum, income). Stock metrics that
    depend on company financials or analyst coverage are not computed.
    """
    ticker = ticker.upper().strip()
    cfg = config or load_config()

    raw = build_fund_raw_metrics(ticker)
    factors = compute_fund_factors(raw)
    analysis = _fund_display_fields(raw)
    analysis["factors_raw"] = {
        col: factors.get(col)
        for cols in FUND_FACTOR_SCORE_COLUMNS.values()
        for col in cols
    }
    analysis["factors_raw"]["expense_ratio"] = raw.get("expense_ratio")

    uni = fund_universe_df if fund_universe_df is not None else load_fund_universe_snapshot()
    if uni is None or uni.empty:
        analysis.update(
            {
                "composite": None,
                "factor_coverage_pct": 0.0,
                "factor_breakdown": {
                    family: {"percentile": None} for family in FUND_FACTOR_SCORE_COLUMNS
                },
                "bargain": compute_fund_bargain_score(
                    price=raw.get("price"),
                    fifty_two_week_high=raw.get("fifty_two_week_high"),
                    rsi_14=raw.get("rsi_14"),
                ),
                "warning": (
                    "Fund universe snapshot missing. Run `python -m core.fund_universe` "
                    "to build it."
                ),
            }
        )
        return analysis

    row = {
        "ticker": ticker,
        "name": raw.get("name"),
        "quote_type": raw.get("quote_type"),
        "category": raw.get("category"),
        "fund_family": raw.get("fund_family"),
        "currency": raw.get("currency"),
        **factors,
    }
    combined = uni[uni["ticker"].astype(str).str.upper() != ticker].copy()
    combined = pd.concat([combined, pd.DataFrame([row])], ignore_index=True)

    scored = score_universe_df(
        combined,
        cfg,
        group_col="category",
        factor_columns=FUND_FACTOR_SCORE_COLUMNS,
        weights=get_fund_factor_weights(cfg),
    )
    scored_row = scored[scored["ticker"] == ticker].iloc[0].to_dict()

    analysis["composite"] = scored_row.get("composite")
    analysis["factor_coverage_pct"] = scored_row.get("factor_coverage_pct")
    analysis["factor_breakdown"] = {
        family: {"percentile": scored_row.get(f"pct_{family}")}
        for family in FUND_FACTOR_SCORE_COLUMNS
    }
    analysis["scored_row"] = scored_row
    analysis["bargain"] = compute_fund_bargain_score(
        price=raw.get("price"),
        fifty_two_week_high=raw.get("fifty_two_week_high"),
        rsi_14=raw.get("rsi_14"),
    )
    return analysis


def apply_fund_snapshot_scoring(
    analysis: dict[str, Any],
    scored_fund_universe: pd.DataFrame,
    ticker: str,
) -> dict[str, Any]:
    """
    Replace fund composite and factor percentiles with snapshot-universe scores
    so the gauge matches the fund rankings table (mirrors the stock behavior).
    """
    ticker = ticker.upper().strip()
    if scored_fund_universe is None or scored_fund_universe.empty:
        return analysis

    mask = scored_fund_universe["ticker"].astype(str).str.upper() == ticker
    if not mask.any():
        return analysis

    snap_row = scored_fund_universe.loc[mask].iloc[0]
    updated = {**analysis}
    updated["composite"] = snap_row.get("composite")
    updated["factor_coverage_pct"] = snap_row.get("factor_coverage_pct")
    updated["factor_breakdown"] = {
        family: {"percentile": snap_row.get(f"pct_{family}")}
        for family in FUND_FACTOR_SCORE_COLUMNS
    }
    return updated


def apply_universe_snapshot_scoring(
    analysis: dict[str, Any],
    scored_universe: pd.DataFrame,
    ticker: str,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Replace composite and factor percentiles with snapshot-universe scores.

    Keeps live price, analyst, and bargain data from score_ticker while ensuring
    the dashboard gauge matches the universe rankings table.
    """
    cfg = config or load_config()
    ticker = ticker.upper().strip()
    if scored_universe is None or scored_universe.empty:
        return analysis

    mask = scored_universe["ticker"].astype(str).str.upper() == ticker
    if not mask.any():
        return analysis

    snap_row = scored_universe.loc[mask].iloc[0]
    updated = {**analysis}
    updated["composite"] = snap_row.get("composite")
    updated["factor_coverage_pct"] = snap_row.get("factor_coverage_pct")

    factor_breakdown: dict[str, dict[str, Any]] = {}
    for family in FACTOR_SCORE_COLUMNS:
        factor_breakdown[family] = {"percentile": snap_row.get(f"pct_{family}")}
    updated["factor_breakdown"] = factor_breakdown

    analyst = updated.get("analyst") or {}
    bargain_score = (updated.get("bargain") or {}).get("score")
    updated["is_good_buy"] = _evaluate_good_buy(
        updated["composite"],
        analyst.get("implied_upside_pct"),
        analyst,
        get_thresholds(cfg),
        bargain_score=bargain_score,
        factor_coverage_pct=updated.get("factor_coverage_pct"),
    )
    return updated


def _score_without_universe(
    ticker: str,
    raw: dict,
    factors: dict,
    analyst: dict,
    cfg: dict,
) -> dict[str, Any]:
    """Fallback when no universe snapshot exists."""
    factor_breakdown = {
        family: {"percentile": None}
        for family in FACTOR_SCORE_COLUMNS
    }
    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    return {
        "ticker": ticker,
        "name": raw.get("name"),
        "sector": raw.get("sector"),
        "industry": raw.get("industry"),
        "exchange": raw.get("exchange"),
        "price": raw.get("price"),
        "market_cap": raw.get("market_cap"),
        "dividend_yield": raw.get("dividend_yield"),
        "fifty_two_week_high": raw.get("fifty_two_week_high"),
        "fifty_two_week_low": raw.get("fifty_two_week_low"),
        "is_etf": False,
        "composite": None,
        "factor_coverage_pct": 0.0,
        "factor_breakdown": factor_breakdown,
        "factors_raw": {**{col: factors.get(col) for col in all_sub_cols}, "trailing_pe": raw.get("trailing_pe")},
        "analyst": analyst,
        "is_good_buy": False,
        "data_warnings": raw.get("data_warnings", []),
        "warning": "Universe snapshot missing. Run jobs/watchlist_weekly.py or core/universe.py to build it.",
        **_bargain_fields(raw, factors, analyst, cfg),
    }


def _evaluate_good_buy(
    composite: float | None,
    implied_upside: float | None,
    analyst: dict,
    thresholds: dict,
    *,
    bargain_score: float | None = None,
    factor_coverage_pct: float | None = None,
) -> bool:
    """
    Good-buy gate: composite + bargain + factor coverage
    (+ optional sell-consensus filter).

    The coverage gate stops thin data from producing confident scores: with
    missing factor groups the composite renormalizes over whatever is left, so
    a stock scored on 2 of 7 groups would otherwise look as trustworthy as one
    scored on all 7. Rows without a coverage figure (None) are not blocked.

    Analyst implied upside is informational by default. Set
    ``require_implied_upside: true`` in config to restore the hard gate.
    """
    composite_min = float(thresholds.get("composite_min", 50))
    bargain_min = float(thresholds.get("bargain_min", 50))
    coverage_min = float(thresholds.get("coverage_min_pct", 70))
    exclude_sell = bool(thresholds.get("exclude_sell_consensus", True))
    require_upside = bool(thresholds.get("require_implied_upside", False))
    upside_min = float(thresholds.get("implied_upside_min_pct", 15))

    if composite is None or composite < composite_min:
        return False
    if bargain_score is None or bargain_score < bargain_min:
        return False
    if factor_coverage_pct is not None and factor_coverage_pct < coverage_min:
        return False
    if require_upside and (implied_upside is None or implied_upside < upside_min):
        return False
    if exclude_sell and analyst.get("consensus_label") == "Sell":
        return False
    return True


def evaluate_watchlist(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Score all watchlist tickers and return those meeting good-buy criteria."""
    cfg = config or load_config()
    watchlist = load_watchlist()
    uni = load_universe_snapshot()
    scored_universe = score_universe_df(uni, cfg) if uni is not None and not uni.empty else None
    results = []
    for ticker in watchlist:
        try:
            analysis = score_ticker(ticker, cfg, uni)
            if scored_universe is not None:
                analysis = apply_universe_snapshot_scoring(
                    analysis, scored_universe, ticker, cfg
                )
            if analysis.get("is_good_buy"):
                results.append(analysis)
        except Exception:
            continue
    return results
