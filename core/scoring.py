"""Cross-sectional scoring: empirical rank percentiles, group buckets, composite."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.analysts import aggregate_analyst_data
from core.config import (
    get_bargain_weights,
    get_decision_config,
    get_factor_weights,
    get_fund_factor_weights,
    get_thresholds,
    load_config,
)
from core.data import (
    build_fund_raw_metrics,
    build_raw_metrics,
    get_security_type,
    is_etf,
    is_fund,
    listing_amount,
)
from core.edgar_history import compute_valuation_vs_history_detail
from core.estimates import compute_revision_factors
from core.insiders import compute_insider_factor
from core.signals import compute_short_interest, compute_uncertainty
from core.factors import FACTOR_SCORE_COLUMNS, FACTOR_SUB_BUCKETS, compute_all_factors
from core.fund_factors import FUND_FACTOR_SCORE_COLUMNS, compute_fund_factors
from core.fund_universe import load_fund_universe_snapshot
from core.universe import load_universe_snapshot, snapshot_path
from core.watchlist import load_watchlist

# Long-horizon valuation bargain weights (RSI removed; kept as informational only).
# graham_heavy (0.55/0.30/0.15) is the live default. The old 0.40/0.35/0.25 mix is
# registered as bargain candidate `legacy_040_035_025` for an apples-to-apples
# comparison on the clean companyfacts panel. Do not cite pre-fix IC numbers.
BARGAIN_COMPONENT_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.55,
    "valuation_vs_history": 0.30,
    "discount_52w": 0.15,
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
      valuation_vs_history   — cheapness vs own 10y EDGAR history (best of
                               EBIT/EV, OCF yield, book-to-market).
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
    valuation_detail: dict[str, Any] = {}
    valuation_vs_history = raw.get("valuation_vs_history")
    if valuation_vs_history is None:
        current_ey = raw.get("earnings_yield_current")
        if current_ey is None:
            current_ey = factors.get("earnings_yield")
        ocf = raw.get("operating_cashflow")
        mcap = raw.get("market_cap")
        current_ocf_yield = None
        if ocf is not None and mcap and mcap > 0:
            current_ocf_yield = float(ocf) / float(mcap)
        ticker = raw.get("ticker")
        if ticker:
            try:
                valuation_detail = compute_valuation_vs_history_detail(
                    str(ticker),
                    current_ey,
                    current_ocf_yield=current_ocf_yield,
                    current_book_to_market=factors.get("book_to_market"),
                )
                valuation_vs_history = valuation_detail.get("score")
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
        "valuation_history": valuation_detail,
        "bargain": bargain,
    }


def rank_percentile(series: pd.Series) -> pd.Series:
    """
    Empirical cross-sectional percentile (0-100) using average ranks.

    Rank-based, so robust to outliers without a distributional assumption
    (replaces the previous winsorize -> z-score -> normal-CDF mapping).
    Requires at least 3 non-null values; constant columns map to 50.
    """
    n = series.count()
    if n < 3:
        return pd.Series(np.nan, index=series.index)
    ranks = series.rank(method="average")
    return (ranks - 0.5) / n * 100.0


def _score_column(
    df: pd.DataFrame,
    col: str,
    group_col: str | None,
    min_group_size: int = 5,
) -> pd.Series:
    """Empirical percentile of a column, within sector groups when large enough
    else universe-wide."""
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)

    values = df[col]

    if group_col and group_col in df.columns:
        def group_pct(s: pd.Series) -> pd.Series:
            if s.dropna().shape[0] >= min_group_size:
                return rank_percentile(s)
            return pd.Series(np.nan, index=s.index)

        pct = values.groupby(df[group_col]).transform(group_pct)
        # Fallback to universe-wide for small sectors
        missing = pct.isna() & values.notna()
        if missing.any():
            universe_pct = rank_percentile(values)
            pct = pct.where(~missing, universe_pct)
        return pct

    return rank_percentile(values)


def compute_family_percentile(
    df: pd.DataFrame,
    cols: list[str],
    *,
    group_col: str | None = None,
    buckets: list[list[str]] | None = None,
) -> pd.Series:
    """
    Group percentile score for one factor family.

    Each sub-signal is ranked cross-sectionally (sector-relative when
    ``group_col`` is given), then averaged within buckets, then bucket scores
    are averaged (skipna at both levels). Without explicit ``buckets`` every
    sub-signal is its own bucket, which reduces to a plain mean of sub-signal
    percentiles. Buckets stop correlated sub-signals (e.g. five profitability
    ratios) from silently dominating a family score.
    """
    if buckets:
        covered = {c for bucket in buckets for c in bucket}
        bucket_defs = list(buckets) + [[c] for c in cols if c not in covered]
    else:
        bucket_defs = [[c] for c in cols]

    bucket_scores: list[pd.Series] = []
    for bucket in bucket_defs:
        sub = [
            _score_column(df, col, group_col)
            for col in bucket
            if col in df.columns
        ]
        if sub:
            bucket_scores.append(pd.concat(sub, axis=1).mean(axis=1, skipna=True))

    if not bucket_scores:
        return pd.Series(np.nan, index=df.index)
    return pd.concat(bucket_scores, axis=1).mean(axis=1, skipna=True)


def _composite_and_coverage(
    row: pd.Series,
    weights: dict[str, float],
    factor_columns: dict[str, list[str]] | None = None,
) -> tuple[float | None, float]:
    """Weighted composite; coverage is weight × fraction of sub-signals present."""
    families = factor_columns if factor_columns is not None else FACTOR_SCORE_COLUMNS
    weighted_sum = 0.0
    weight_available = 0.0
    coverage_weight = 0.0
    weight_total = sum(weights.get(family, 0) for family in families)

    for family in families:
        w = weights.get(family, 0)
        pct_col = f"pct_{family}"
        pct = row[pct_col] if pct_col in row.index else None
        if pct is not None and not (isinstance(pct, float) and np.isnan(pct)):
            weighted_sum += float(pct) * w
            weight_available += w

        cols = list(families.get(family, []))
        if cols and any(c in row.index for c in cols):
            present = 0
            for c in cols:
                val = row[c] if c in row.index else None
                if _is_meaningful_value(val, column=c):
                    present += 1
            frac = present / len(cols)
        else:
            frac = 1.0 if pct is not None and not (isinstance(pct, float) and np.isnan(pct)) else 0.0
        coverage_weight += w * frac

    if weight_available == 0:
        return None, 0.0

    composite = weighted_sum / weight_available
    coverage = (coverage_weight / weight_total * 100.0) if weight_total > 0 else 0.0
    return composite, coverage


def _meaningful_mask(series: pd.Series, column: str) -> pd.Series:
    """Vectorized counterpart of ``_is_meaningful_value``."""
    if series is None:
        return pd.Series(dtype=bool)
    mask = series.notna()
    if pd.api.types.is_numeric_dtype(series):
        mask = mask & ~pd.isna(series)
        if column in ZERO_NEUTRAL_COLUMNS:
            mask = mask & (series != 0)
        return mask
    as_str = series.astype(str).str.strip()
    mask = mask & ~as_str.str.lower().isin({"", "none", "nan", "unknown", "<na>"})
    numeric = pd.to_numeric(series, errors="coerce")
    if column in ZERO_NEUTRAL_COLUMNS:
        mask = mask & (numeric.fillna(1) != 0)
    return mask


def _composite_and_coverage_frame(
    df: pd.DataFrame,
    weights: dict[str, float],
    factor_columns: dict[str, list[str]] | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Vectorized composite + coverage; matches ``_composite_and_coverage`` row-wise."""
    families = factor_columns if factor_columns is not None else FACTOR_SCORE_COLUMNS
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_available = pd.Series(0.0, index=df.index)
    coverage_weight = pd.Series(0.0, index=df.index)
    weight_total = sum(weights.get(family, 0) for family in families)

    for family in families:
        w = float(weights.get(family, 0) or 0.0)
        pct_col = f"pct_{family}"
        pct = None
        has_pct = pd.Series(False, index=df.index)
        if pct_col in df.columns:
            pct = pd.to_numeric(df[pct_col], errors="coerce")
            has_pct = pct.notna()
            weighted_sum = weighted_sum + pct.fillna(0.0) * w
            weight_available = weight_available + (w * has_pct.astype(float))

        cols = list(families.get(family, []))
        present_cols = [c for c in cols if c in df.columns]
        if cols and present_cols:
            present = pd.Series(0.0, index=df.index)
            for c in cols:
                if c in df.columns:
                    present = present + _meaningful_mask(df[c], c).astype(float)
            frac = present / float(len(cols))
        else:
            frac = has_pct.astype(float)
        coverage_weight = coverage_weight + (w * frac)

    composite = weighted_sum / weight_available.replace(0.0, np.nan)
    coverage = (
        coverage_weight / weight_total * 100.0
        if weight_total > 0
        else pd.Series(0.0, index=df.index)
    )
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
    "revision_agreement",
    "revision_magnitude",
    "earnings_surprise",
    "insider_buying",
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


def _attach_live_signals(
    analysis: dict[str, Any],
    raw: dict[str, Any],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Attach revision sparkline, insider badge, short-interest flag, uncertainty."""
    revisions = compute_revision_factors(raw)
    insider = compute_insider_factor(raw)
    short = compute_short_interest(raw, thresholds)
    analyst = analysis.get("analyst") or {}
    uncertainty = compute_uncertainty(
        factor_coverage_pct=analysis.get("factor_coverage_pct"),
        volatility_12m=raw.get("volatility_12m") or analysis.get("volatility_12m"),
        target_high=raw.get("target_high", analyst.get("target_high")),
        target_low=raw.get("target_low", analyst.get("target_low")),
        target_mean=raw.get("target_mean", analyst.get("target_mean")),
        thresholds=thresholds,
    )
    analysis["eps_trend_sparkline"] = revisions.get("eps_trend_sparkline") or []
    analysis["revision_agreement"] = revisions.get("revision_agreement")
    analysis["revision_magnitude"] = revisions.get("revision_magnitude")
    analysis["earnings_surprise"] = revisions.get("earnings_surprise")
    analysis["insider"] = {
        "cluster_buy": bool(insider.get("insider_cluster_buy")),
        "buyers_90d": insider.get("insider_buyers_90d") or 0,
        "sellers_90d": insider.get("insider_sellers_90d") or 0,
        "net_value_90d": insider.get("insider_net_value_90d"),
        "buying_yield": insider.get("insider_buying"),
    }
    analysis["short_interest"] = short
    analysis["uncertainty"] = uncertainty
    analysis["volatility_12m"] = raw.get("volatility_12m")
    return analysis


def _attach_intrinsic_layers(
    analysis: dict[str, Any],
    raw: dict[str, Any],
    cfg: dict[str, Any],
    scored_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach valuation, flags, quality percentile, and Decision. Never raises."""
    from core.decision import decide
    from core.fundamentals import get_fundamentals
    from core.quality import flags_to_dicts, value_trap_flags
    from core.valuation import valuation_summary

    scored_row = scored_row or {}
    analysis["quality_score"] = scored_row.get("quality_score", analysis.get("quality_score"))
    analysis["quality_coverage_pct"] = scored_row.get(
        "quality_coverage_pct", analysis.get("quality_coverage_pct")
    )
    analysis["quality_percentile"] = scored_row.get("quality_score", analysis.get("quality_score"))
    analysis["ev_to_ebit"] = scored_row.get("ev_to_ebit", raw.get("ev_to_ebit"))
    analysis["p_to_oe"] = scored_row.get("p_to_oe", raw.get("p_to_oe"))

    fund = raw.get("_fundamentals")
    if fund is None and get_decision_config(cfg).get("mode") != "legacy":
        try:
            fund = get_fundamentals(str(analysis.get("ticker") or raw.get("ticker") or ""))
        except Exception:
            fund = None

    flags: list[dict[str, Any]] = []
    try:
        flags = flags_to_dicts(value_trap_flags(raw, fund, cfg))
    except Exception:
        flags = []
    analysis["value_trap_flags"] = flags

    valuation: dict[str, Any] = {}
    if fund is not None:
        try:
            analyst = analysis.get("analyst") or {}
            valuation = valuation_summary(
                fund,
                price=raw.get("price") or analysis.get("price"),
                shares=raw.get("shares_outstanding") or raw.get("shares_diluted"),
                cash=raw.get("total_cash"),
                debt=raw.get("total_debt"),
                market_cap=raw.get("market_cap"),
                ev=raw.get("enterprise_value"),
                sector=analysis.get("sector") or raw.get("sector"),
                industry=analysis.get("industry") or raw.get("industry"),
                uncertainty_label=(analysis.get("uncertainty") or {}).get("label"),
                analyst_growth=raw.get("earnings_growth"),
                config=cfg,
            )
        except Exception:
            valuation = {}
    analysis["valuation"] = valuation

    # Extra uncertainty points: data-quality B, wide DCF spread.
    try:
        analysis["uncertainty"] = compute_uncertainty(
            factor_coverage_pct=analysis.get("factor_coverage_pct"),
            volatility_12m=analysis.get("volatility_12m"),
            target_high=(analysis.get("analyst") or {}).get("target_high"),
            target_low=(analysis.get("analyst") or {}).get("target_low"),
            target_mean=(analysis.get("analyst") or {}).get("target_mean"),
            thresholds=get_thresholds(cfg),
            data_quality_grade=(analysis.get("data_quality") or {}).get("grade"),
            dcf_base=(valuation.get("dcf") or {}).get("base", {}).get("per_share") if valuation.get("dcf") else None,
            dcf_bear=(valuation.get("dcf") or {}).get("bear", {}).get("per_share") if valuation.get("dcf") else None,
        )
    except TypeError:
        pass

    try:
        decision = decide(analysis, cfg)
        analysis["decision"] = decision.to_dict()
        if get_decision_config(cfg).get("mode") == "legacy":
            analysis["is_good_buy"] = decision.label == "Accumulate"
        else:
            analysis["is_good_buy"] = decision.label == "Accumulate"
    except Exception:
        analysis["decision"] = {
            "label": "Avoid",
            "buy_below_price": None,
            "pct_to_buy": None,
            "gates": [],
            "flags": flags,
            "timing_context": {},
            "mode": get_decision_config(cfg).get("mode"),
        }
        analysis["is_good_buy"] = False
    return analysis


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

    For each factor group: rank each sub-signal cross-sectionally (empirical
    percentile, sector-relative when enabled), average within sub-buckets
    (see FACTOR_SUB_BUCKETS), then average bucket scores into a single group
    percentile. Composite = weighted average of group scores over groups with
    data.

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
        result[f"pct_{family}"] = compute_family_percentile(
            result,
            cols,
            group_col=group_col if use_sector else None,
            buckets=FACTOR_SUB_BUCKETS.get(family),
        )

    composite, coverage = _composite_and_coverage_frame(result, weights, families)
    result["composite"] = composite
    result["factor_coverage_pct"] = coverage

    if families is FACTOR_SCORE_COLUMNS or factor_columns is None:
        try:
            from core.quality import compute_quality_score

            result = compute_quality_score(result, cfg, group_col=group_col if use_sector else None)
        except Exception:
            result["quality_score"] = None
            result["quality_coverage_pct"] = None

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
    altman_z = row.get("altman_z")
    altman_z_pp = row.get("altman_z_pp")
    sector = _resolved_display_field(raw, row, "sector", ticker=ticker)

    factor_breakdown: dict[str, dict] = {}
    for family in FACTOR_SCORE_COLUMNS:
        factor_breakdown[family] = {
            "percentile": scored_row.get(f"pct_{family}"),
        }

    # Flatten all sub-signal columns for the raw values expander
    all_sub_cols = [col for cols in FACTOR_SCORE_COLUMNS.values() for col in cols]
    factors_raw = {col: row.get(col) for col in all_sub_cols}
    factors_raw["trailing_pe"] = raw.get("trailing_pe")

    analysis = {
        "ticker": ticker,
        "name": _resolved_display_field(raw, row, "name", ticker=ticker),
        "sector": sector,
        "industry": _resolved_display_field(raw, row, "industry", ticker=ticker),
        "exchange": raw.get("exchange"),
        "price": listing_amount(raw, "price"),
        "market_cap": listing_amount(raw, "market_cap") or row.get("market_cap"),
        "dividend_yield": raw.get("dividend_yield"),
        "fifty_two_week_high": listing_amount(raw, "fifty_two_week_high"),
        "fifty_two_week_low": listing_amount(raw, "fifty_two_week_low"),
        "is_etf": False,
        "composite": composite,
        "factor_coverage_pct": factor_coverage_pct,
        "factor_breakdown": factor_breakdown,
        "factors_raw": factors_raw,
        "analyst": analyst,
        "altman_z": altman_z,
        "altman_z_pp": altman_z_pp,
        "distress_flag": is_distressed(altman_z, thresholds, sector, altman_z_pp=altman_z_pp),
        "data_warnings": raw.get("data_warnings", []),
        "data_quality": raw.get("data_quality"),
        "currency": raw.get("currency") or "USD",
        "financial_currency": raw.get("financial_currency"),
        "scored_row": scored_row,
        **bargain_data,
    }
    analysis = _attach_live_signals(analysis, raw, thresholds)
    bump = (analysis.get("uncertainty") or {}).get("threshold_bump") or 0.0
    analysis["is_good_buy"] = _evaluate_good_buy(
        composite,
        implied_upside,
        analyst,
        thresholds,
        bargain_score=bargain_score,
        factor_coverage_pct=factor_coverage_pct,
        altman_z=altman_z,
        sector=sector,
        uncertainty_bump=bump,
        altman_z_pp=altman_z_pp,
    )
    return _attach_intrinsic_layers(analysis, raw, cfg, scored_row)


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
    thresholds = get_thresholds(cfg)
    altman_z = snap_row.get("altman_z", updated.get("altman_z"))
    if isinstance(altman_z, float) and np.isnan(altman_z):
        altman_z = updated.get("altman_z")
    altman_z_pp = snap_row.get("altman_z_pp", updated.get("altman_z_pp"))
    if isinstance(altman_z_pp, float) and np.isnan(altman_z_pp):
        altman_z_pp = updated.get("altman_z_pp")
    sector = updated.get("sector") or snap_row.get("sector")
    updated["altman_z"] = altman_z
    updated["altman_z_pp"] = altman_z_pp
    updated["distress_flag"] = is_distressed(
        altman_z, thresholds, sector, altman_z_pp=altman_z_pp
    )
    uncertainty = compute_uncertainty(
        factor_coverage_pct=updated.get("factor_coverage_pct"),
        volatility_12m=updated.get("volatility_12m"),
        target_high=analyst.get("target_high"),
        target_low=analyst.get("target_low"),
        target_mean=analyst.get("target_mean"),
        thresholds=thresholds,
    )
    updated["uncertainty"] = uncertainty
    updated["is_good_buy"] = _evaluate_good_buy(
        updated["composite"],
        analyst.get("implied_upside_pct"),
        analyst,
        thresholds,
        bargain_score=bargain_score,
        factor_coverage_pct=updated.get("factor_coverage_pct"),
        altman_z=altman_z,
        sector=sector,
        uncertainty_bump=uncertainty.get("threshold_bump") or 0.0,
        altman_z_pp=altman_z_pp,
    )
    raw_stub = {
        "ticker": ticker,
        "price": updated.get("price"),
        "market_cap": updated.get("market_cap"),
        "sector": sector,
        "industry": updated.get("industry"),
        "shares_outstanding": updated.get("factors_raw", {}).get("shares_outstanding") if isinstance(updated.get("factors_raw"), dict) else None,
        "total_cash": None,
        "total_debt": None,
        "enterprise_value": None,
        "accruals": (updated.get("factors_raw") or {}).get("accruals"),
        "roic": (updated.get("factors_raw") or {}).get("roic"),
        "net_debt_to_ebitda": snap_row.get("net_debt_to_ebitda"),
        "interest_coverage": snap_row.get("interest_coverage"),
        "revenue_5y_cagr": snap_row.get("revenue_5y_cagr"),
        "gross_margin_5y_delta": snap_row.get("gross_margin_5y_delta"),
        "share_cagr_3y": snap_row.get("share_cagr_3y"),
        "fcf_conversion_3y": snap_row.get("fcf_conversion_3y"),
        "owner_earnings_norm": snap_row.get("owner_earnings_norm"),
        "altman_z_pp": altman_z_pp,
        "data_quality": updated.get("data_quality"),
    }
    return _attach_intrinsic_layers(updated, raw_stub, cfg, snap_row.to_dict())


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
    analysis = {
        "ticker": ticker,
        "name": raw.get("name"),
        "sector": raw.get("sector"),
        "industry": raw.get("industry"),
        "exchange": raw.get("exchange"),
        "price": listing_amount(raw, "price"),
        "market_cap": listing_amount(raw, "market_cap") or raw.get("market_cap"),
        "dividend_yield": raw.get("dividend_yield"),
        "fifty_two_week_high": listing_amount(raw, "fifty_two_week_high"),
        "fifty_two_week_low": listing_amount(raw, "fifty_two_week_low"),
        "is_etf": False,
        "composite": None,
        "factor_coverage_pct": 0.0,
        "factor_breakdown": factor_breakdown,
        "factors_raw": {**{col: factors.get(col) for col in all_sub_cols}, "trailing_pe": raw.get("trailing_pe")},
        "analyst": analyst,
        "is_good_buy": False,
        "altman_z": factors.get("altman_z"),
        "altman_z_pp": factors.get("altman_z_pp"),
        "distress_flag": is_distressed(
            factors.get("altman_z"),
            get_thresholds(cfg),
            raw.get("sector"),
            altman_z_pp=factors.get("altman_z_pp"),
        ),
        "data_warnings": raw.get("data_warnings", []),
        "data_quality": raw.get("data_quality"),
        "currency": raw.get("currency") or "USD",
        "financial_currency": raw.get("financial_currency"),
        "warning": "Universe snapshot missing. Run jobs/watchlist_weekly.py or core/universe.py to build it.",
        **_bargain_fields(raw, factors, analyst, cfg),
    }
    analysis = _attach_live_signals(analysis, raw, get_thresholds(cfg))
    return _attach_intrinsic_layers(analysis, raw, cfg, None)


# Altman Z'' (1995) is valid for non-financials including utilities and REITs.
# Financials still exempt: the model is not designed for deposit-taking balance sheets.
ALTMAN_EXEMPT_SECTORS: frozenset[str] = frozenset({
    "Financial Services",
})


def is_distressed(
    altman_z: float | None,
    thresholds: dict | None = None,
    sector: str | None = None,
    *,
    altman_z_pp: float | None = None,
) -> bool:
    """
    Hard distress disqualifier using Altman Z'' (1995), cutoff 1.1.

    Missing Z'' never blocks. Financials are exempt.
    """
    if sector is not None and str(sector) in ALTMAN_EXEMPT_SECTORS:
        return False
    z = altman_z_pp if altman_z_pp is not None else altman_z
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return False
    z_min = float((thresholds or {}).get("altman_zpp_min", 1.1))
    return float(z) < z_min


def _evaluate_good_buy(
    composite: float | None,
    implied_upside: float | None,
    analyst: dict,
    thresholds: dict,
    *,
    bargain_score: float | None = None,
    factor_coverage_pct: float | None = None,
    altman_z: float | None = None,
    sector: str | None = None,
    uncertainty_bump: float | None = None,
    altman_z_pp: float | None = None,
) -> bool:
    """
    Good-buy gate: composite + bargain + factor coverage + no distress
    (+ optional sell-consensus filter).

    The coverage gate stops thin data from producing confident scores: with
    missing factor groups the composite renormalizes over whatever is left, so
    a stock scored on 2 of 7 groups would otherwise look as trustworthy as one
    scored on all 7. Rows without a coverage figure (None) are not blocked.

    The Altman Z'' distress gate blocks names below 1.1 outright.
    Financials are exempt; missing Z'' never blocks.

    High uncertainty widens the composite/bargain cutoffs (Morningstar-style
    larger required discount when the estimate is noisier).

    Analyst implied upside is informational by default. Set
    ``require_implied_upside: true`` in config to restore the hard gate.
    """
    bump = float(uncertainty_bump or 0.0)
    composite_min = float(thresholds.get("composite_min", 50)) + bump
    bargain_min = float(thresholds.get("bargain_min", 50)) + bump
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
    if is_distressed(altman_z, thresholds, sector, altman_z_pp=altman_z_pp):
        return False
    if require_upside and (implied_upside is None or implied_upside < upside_min):
        return False
    if exclude_sell:
        blocked = {"Sell"}
        if bool(thresholds.get("exclude_underperform", True)):
            blocked.add("Underperform")
        if analyst.get("consensus_label") in blocked:
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
