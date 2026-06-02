"""Cross-sectional scoring: winsorize, z-scores, percentiles, composite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.analysts import aggregate_analyst_data
from core.config import get_factor_weights, get_thresholds, load_config
from core.data import build_raw_metrics, is_etf
from core.factors import FACTOR_SCORE_COLUMNS, compute_all_factors
from core.universe import load_universe_snapshot, snapshot_path
from core.watchlist import load_watchlist

BARGAIN_COMPONENT_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.30,
    "discount_ath": 0.25,
    "discount_52w": 0.15,
    "rsi_oversold": 0.15,
    "analyst_upside": 0.15,
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
) -> dict[str, Any]:
    """
    Absolute 0-100 bargain score from fixed thresholds (higher = more of a bargain).
    Renormalizes weights over components with available data.
    """
    components: dict[str, float | None] = {
        "margin_of_safety": None,
        "discount_ath": None,
        "discount_52w": None,
        "rsi_oversold": None,
        "analyst_upside": None,
    }

    if graham_ratio is not None and graham_ratio > 0:
        mos = graham_ratio - 1.0
        components["margin_of_safety"] = _linear_score(mos, 0.0, 0.50)

    if price is not None and all_time_high is not None and all_time_high > 0 and price > 0:
        discount_ath = 1.0 - (price / all_time_high)
        components["discount_ath"] = _linear_score(discount_ath, 0.0, 0.50)

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

    if implied_upside_pct is not None:
        components["analyst_upside"] = _linear_score(float(implied_upside_pct), 0.0, 40.0)

    weighted_sum = 0.0
    weight_available = 0.0
    for key, sub_score in components.items():
        if sub_score is None:
            continue
        w = BARGAIN_COMPONENT_WEIGHTS.get(key, 0.0)
        weighted_sum += sub_score * w
        weight_available += w

    score = weighted_sum / weight_available if weight_available > 0 else None
    return {"score": score, "components": components}


def _bargain_fields(
    raw: dict,
    factors: dict,
    analyst: dict,
) -> dict[str, Any]:
    """Build bargain score and related fields for analysis dict."""
    bargain = compute_bargain_score(
        price=raw.get("price"),
        graham_ratio=factors.get("graham_ratio"),
        all_time_high=raw.get("all_time_high"),
        fifty_two_week_high=raw.get("fifty_two_week_high"),
        rsi_14=raw.get("rsi_14"),
        implied_upside_pct=analyst.get("implied_upside_pct"),
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
    """Weighted composite using only factors with data; returns (composite, coverage_pct)."""
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


def score_universe_df(
    factors_df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    group_col: str | None = "sector",
) -> pd.DataFrame:
    """
    Score all tickers in a factors dataframe cross-sectionally.
    Returns dataframe with z-scores, percentiles, and composite.
    """
    cfg = config or load_config()
    weights = get_factor_weights(cfg)
    use_sector = cfg.get("universe", {}).get("sector_scoring", True) and group_col in factors_df.columns

    result = factors_df.copy()

    for family, col in FACTOR_SCORE_COLUMNS.items():
        if col not in result.columns:
            result[f"z_{family}"] = np.nan
            result[f"pct_{family}"] = np.nan
            continue

        z_col = f"z_{family}"
        pct_col = f"pct_{family}"

        if use_sector:
            result[z_col] = _score_column(result, col, group_col)
        else:
            result[z_col] = cross_sectional_zscore(winsorize(result[col]))

        result[pct_col] = result[z_col].apply(zscore_to_percentile)

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
        # Fallback: score without cross-section (raw percentiles unavailable)
        return _score_without_universe(ticker, raw, factors, analyst, cfg)

    # Build row for this ticker
    row = {
        "ticker": ticker,
        "name": raw.get("name"),
        "sector": raw.get("sector"),
        "industry": raw.get("industry"),
        **factors,
    }
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
    bargain_data = _bargain_fields(raw, factors, analyst)
    bargain_score = (bargain_data.get("bargain") or {}).get("score")
    is_good_buy = _evaluate_good_buy(
        composite,
        implied_upside,
        analyst,
        thresholds,
        bargain_score=bargain_score,
    )

    factor_breakdown = {}
    for family in FACTOR_SCORE_COLUMNS:
        factor_breakdown[family] = {
            "raw": factors.get(FACTOR_SCORE_COLUMNS[family]),
            "percentile": scored_row.get(f"pct_{family}"),
            "z_score": scored_row.get(f"z_{family}"),
        }

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
        "composite": composite,
        "factor_coverage_pct": factor_coverage_pct,
        "factor_breakdown": factor_breakdown,
        "factors_raw": {**factors, "trailing_pe": raw.get("trailing_pe")},
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
        family: {"raw": factors.get(col), "percentile": None, "z_score": None}
        for family, col in FACTOR_SCORE_COLUMNS.items()
    }
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
        "factors_raw": {**factors, "trailing_pe": raw.get("trailing_pe")},
        "analyst": analyst,
        "is_good_buy": False,
        "data_warnings": raw.get("data_warnings", []),
        "warning": "Universe snapshot missing. Run jobs/daily_check.py or core/universe.py to build it.",
        **_bargain_fields(raw, factors, analyst),
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
