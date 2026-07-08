"""Cross-sectional scoring: winsorize, z-scores, percentiles, composite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.analysts import aggregate_analyst_data
from core.config import get_bargain_weights, get_factor_weights, get_thresholds, load_config
from core.data import build_raw_metrics, is_etf
from core.factors import FACTOR_SCORE_COLUMNS, compute_all_factors
from core.universe import load_universe_snapshot, snapshot_path
from core.watchlist import load_watchlist

# Default bargain component weights from historical IC tuning.
# RSI oversold dominates (mean-reversion); discount_ath and analyst_upside removed.
BARGAIN_COMPONENT_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.2489,
    "discount_52w": 0.0024,
    "rsi_oversold": 0.7488,
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
    all_time_high: float | None,
    fifty_two_week_high: float | None,
    rsi_14: float | None,
    implied_upside_pct: float | None,
    component_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Absolute 0-100 bargain score from fixed thresholds (higher = more of a bargain).

    Components:
      margin_of_safety  — Graham ratio scored over [0.30, 1.30]; discriminates
                          across the full S&P 500 distribution (median ~0.47).
      discount_52w      — % below 52-week high; linear 0%→0, 30%→100.
      rsi_oversold      — RSI 70→0, RSI 30→100 (linear(70-RSI, 0, 40)).

    Renormalizes weights over components with available data.
    analyst_upside and discount_ath are intentionally excluded: upside is a
    standalone good-buy gate; ATH discount is 0.95-correlated with 52w discount.
    """
    components: dict[str, float | None] = {
        "margin_of_safety": None,
        "discount_52w": None,
        "rsi_oversold": None,
    }

    if graham_ratio is not None and graham_ratio > 0:
        # Spans [0.30, 1.30]: covers the full S&P 500 distribution without
        # clamping 90%+ of stocks to zero as the old (graham_ratio - 1) did.
        components["margin_of_safety"] = _linear_score(graham_ratio, 0.30, 1.30)

    if (
        price is not None
        and fifty_two_week_high is not None
        and fifty_two_week_high > 0
        and price > 0
    ):
        discount_52w = 1.0 - (price / fifty_two_week_high)
        components["discount_52w"] = _linear_score(discount_52w, 0.0, 0.30)

    if rsi_14 is not None:
        components["rsi_oversold"] = _linear_score(70.0 - float(rsi_14), 0.0, 40.0)

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


def _bargain_fields(
    raw: dict,
    factors: dict,
    analyst: dict,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build bargain score and related fields for analysis dict."""
    bargain = compute_bargain_score(
        price=raw.get("price"),
        graham_ratio=factors.get("graham_ratio"),
        all_time_high=raw.get("all_time_high"),
        fifty_two_week_high=raw.get("fifty_two_week_high"),
        rsi_14=raw.get("rsi_14"),
        implied_upside_pct=analyst.get("implied_upside_pct"),
        component_weights=get_bargain_weights(config),
    )
    return {
        "all_time_high": raw.get("all_time_high"),
        "rsi_14": raw.get("rsi_14"),
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
) -> tuple[float | None, float]:
    """Weighted composite using only groups with data; returns (composite, coverage_pct)."""
    weighted_sum = 0.0
    weight_available = 0.0
    weight_total = sum(weights.get(family, 0) for family in FACTOR_SCORE_COLUMNS)

    for family in FACTOR_SCORE_COLUMNS:
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


def _is_meaningful_value(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and np.isnan(val):
        return False
    if isinstance(val, str):
        stripped = val.strip()
        if not stripped or stripped.lower() == "unknown":
            return False
    return True


def _merge_ticker_row_with_universe(row: dict, uni: pd.DataFrame, ticker: str) -> dict:
    """
    Overlay a freshly built ticker row onto the universe snapshot row.
    Keeps snapshot factor values when the live fetch is partial (e.g. empty quote info).
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
        if _is_meaningful_value(val):
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
) -> pd.DataFrame:
    """
    Score all tickers in a factors dataframe cross-sectionally.

    For each factor group: cross-sectionally rank each sub-signal (winsorize →
    sector z-score → normal-CDF percentile), then average available sub-signal
    percentiles into a single group percentile score. Composite = weighted average
    of group scores over groups with data.
    """
    cfg = config or load_config()
    weights = get_factor_weights(cfg)
    use_sector = cfg.get("universe", {}).get("sector_scoring", True) and group_col in factors_df.columns

    result = factors_df.copy()

    for family, cols in FACTOR_SCORE_COLUMNS.items():
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
        composite, coverage = _composite_and_coverage(row, weights)
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

    if is_etf(ticker):
        return {"ticker": ticker, "is_etf": True}

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
        "warning": "Universe snapshot missing. Run jobs/daily_check.py or core/universe.py to build it.",
        **_bargain_fields(raw, factors, analyst, cfg),
    }


def _evaluate_good_buy(
    composite: float | None,
    implied_upside: float | None,
    analyst: dict,
    thresholds: dict,
    *,
    bargain_score: float | None = None,
) -> bool:
    composite_min = float(thresholds.get("composite_min", 50))
    bargain_min = float(thresholds.get("bargain_min", 50))
    upside_min = float(thresholds.get("implied_upside_min_pct", 15))
    exclude_sell = bool(thresholds.get("exclude_sell_consensus", True))

    if composite is None or composite < composite_min:
        return False
    if implied_upside is None or implied_upside < upside_min:
        return False
    if bargain_score is None or bargain_score < bargain_min:
        return False
    if exclude_sell and analyst.get("consensus_label") == "Sell":
        return False
    return True


def evaluate_watchlist(config: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Score all watchlist tickers and return those meeting good-buy criteria."""
    cfg = config or load_config()
    watchlist = load_watchlist()
    uni = load_universe_snapshot()
    results = []
    for ticker in watchlist:
        try:
            analysis = score_ticker(ticker, cfg, uni)
            if analysis.get("is_good_buy"):
                results.append(analysis)
        except Exception:
            continue
    return results
