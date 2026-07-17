"""Historical factor panel reconstruction."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from backtest.constants import (
    BACKTEST_FACTOR_FAMILIES,
    BARGAIN_BACKTEST_COMPONENTS,
    DATA_STORE,
    QUARTER_ENDS,
    VALUATION_HISTORY_QUARTERS,
)
from backtest.data.constituents import load_membership
from backtest.data.edgar import fundamentals_as_of, load_fundamentals
from backtest.data.prices import load_prices, price_history_as_of, price_on_or_before
from core.data import (
    _compute_drawdown_metrics,
    _compute_momentum_12_1,
    _compute_rsi,
    _compute_volatility_12m,
    normalize_debt_to_equity,
    percentile_rank_in_history,
)
from core.factors import FACTOR_SCORE_COLUMNS, compute_all_factors
from core.scoring import compute_bargain_score

logger = logging.getLogger(__name__)

FACTOR_PANEL_PATH = DATA_STORE / "factor_panel.parquet"


def _realized_earnings_growth(fund: dict[str, float]) -> float | None:
    ni = fund.get("net_income")
    ni_prior = fund.get("net_income_prior")
    if ni is None or ni_prior is None or ni_prior == 0:
        return None
    return (ni / ni_prior) - 1.0


def _build_raw_row(
    ticker: str,
    as_of: date,
    fundamentals: pd.DataFrame,
    prices: pd.DataFrame,
) -> dict[str, Any]:
    fund = fundamentals_as_of(as_of, ticker, fundamentals)
    price = price_on_or_before(prices, ticker, as_of)
    hist = price_history_as_of(prices, ticker, as_of)

    shares = fund.get("shares_outstanding")
    market_cap = price * shares if price and shares else None
    book_value = fund.get("book_value")
    book_equity = fund.get("book_equity")
    total_debt = fund.get("total_debt")
    total_cash = fund.get("total_cash")

    debt_to_equity = None
    if total_debt is not None and book_equity and book_equity != 0:
        debt_to_equity = normalize_debt_to_equity(total_debt / book_equity * 100.0)

    trailing_eps = None
    if fund.get("net_income") is not None and shares:
        trailing_eps = fund["net_income"] / shares

    trailing_pe = None
    if trailing_eps and trailing_eps > 0 and price:
        trailing_pe = price / trailing_eps

    earnings_growth = _realized_earnings_growth(fund)
    dividend_yield = None
    if fund.get("dividends_paid") is not None and market_cap and market_cap > 0:
        dividend_yield = abs(fund["dividends_paid"]) / market_cap

    ebit = fund.get("ebit")
    enterprise_value = None
    if market_cap is not None:
        enterprise_value = market_cap + (total_debt or 0.0) - (total_cash or 0.0)

    current_ratio = None
    ca = fund.get("current_assets")
    cl = fund.get("current_liabilities")
    if ca is not None and cl and cl > 0:
        current_ratio = ca / cl

    fifty_two_week_high = None
    all_time_high = None
    if not hist.empty and "Close" in hist.columns:
        closes = hist["Close"].dropna()
        if len(closes) > 0:
            window = closes.tail(min(252, len(closes)))
            fifty_two_week_high = float(window.max())
            all_time_high = float(closes.max())

    raw: dict[str, Any] = {
        "ticker": ticker.upper(),
        "price": price,
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,
        "book_value": book_value,
        "shares_outstanding": shares,
        "shares_prior": fund.get("shares_outstanding_prior"),
        "total_assets": fund.get("total_assets"),
        "total_assets_prior": fund.get("total_assets_prior"),
        "total_liabilities": fund.get("total_liabilities"),
        "current_assets": ca,
        "current_liabilities": cl,
        "current_assets_prior": fund.get("current_assets_prior"),
        "current_liabilities_prior": fund.get("current_liabilities_prior"),
        "long_term_debt": fund.get("long_term_debt"),
        "long_term_debt_prior": fund.get("long_term_debt_prior"),
        "gross_profit": fund.get("gross_profit"),
        "gross_profit_prior": fund.get("gross_profit_prior"),
        "net_income": fund.get("net_income"),
        "net_income_prior": fund.get("net_income_prior"),
        "ebit": ebit,
        "revenue": fund.get("revenue"),
        "revenue_prior": fund.get("revenue_prior"),
        "operating_cashflow": fund.get("operating_cashflow"),
        "free_cashflow": None,
        "dividends_paid": fund.get("dividends_paid"),
        "repurchase_of_stock": fund.get("repurchase_of_stock"),
        "retained_earnings": fund.get("retained_earnings"),
        "total_cash": total_cash,
        "total_debt": total_debt,
        "debt_to_equity": debt_to_equity,
        "current_ratio_info": current_ratio,
        "trailing_pe": trailing_pe,
        "trailing_eps": trailing_eps,
        "earnings_growth": earnings_growth,
        "dividend_yield": dividend_yield,
        "trailing_peg_ratio": None,
        "momentum_12_1": _compute_momentum_12_1(hist) if not hist.empty else None,
        "volatility_12m": _compute_volatility_12m(hist) if not hist.empty else None,
        "max_drawdown": None,
        "downside_deviation": None,
        "fifty_two_week_high": fifty_two_week_high,
        "all_time_high": all_time_high,
        "rsi_14": _compute_rsi(hist) if not hist.empty else None,
        "recommendations": pd.DataFrame(),
        "target_mean": None,
    }
    if not hist.empty:
        dd = _compute_drawdown_metrics(hist)
        raw["max_drawdown"] = dd.get("max_drawdown")
        raw["downside_deviation"] = dd.get("downside_deviation")
    return raw


def compute_historical_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Compute factor sub-signal columns, excluding earnings_revisions."""
    all_factors = compute_all_factors(raw)
    all_factors.pop("earnings_revisions", None)
    return all_factors


def compute_historical_bargain(
    raw: dict[str, Any],
    factors: dict[str, Any],
    *,
    valuation_vs_history: float | None = None,
) -> dict[str, Any]:
    """Bargain score using long-horizon valuation components (no RSI / analyst upside)."""
    return compute_bargain_score(
        price=raw.get("price"),
        graham_ratio=factors.get("graham_ratio"),
        fifty_two_week_high=raw.get("fifty_two_week_high"),
        valuation_vs_history=valuation_vs_history,
    )


def _attach_valuation_vs_history(panel: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time EY percentile vs each ticker's trailing 5y history."""
    if panel.empty or "earnings_yield" not in panel.columns:
        panel = panel.copy()
        panel["valuation_vs_history"] = np.nan
        return panel

    out = panel.copy()
    out["quarter_end"] = pd.to_datetime(out["quarter_end"])
    out = out.sort_values(["ticker", "quarter_end"])
    scores: list[float | None] = []
    for ticker, grp in out.groupby("ticker", sort=False):
        ey = grp["earnings_yield"].tolist()
        for i, current in enumerate(ey):
            start = max(0, i - VALUATION_HISTORY_QUARTERS)
            history = [
                float(v)
                for v in ey[start:i]
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            scores.append(percentile_rank_in_history(
                None if current is None or (isinstance(current, float) and np.isnan(current)) else float(current),
                history,
            ))
    out["valuation_vs_history"] = scores
    return out


def _weighted_bargain_from_components(
    components: dict[str, float | None],
    weights: dict[str, float] | None = None,
) -> float | None:
    """Blend precomputed 0-100 bargain components with default weights."""
    from core.scoring import BARGAIN_COMPONENT_WEIGHTS

    wmap = weights or BARGAIN_COMPONENT_WEIGHTS
    weighted_sum = 0.0
    weight_available = 0.0
    for key, val in components.items():
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        w = float(wmap.get(key, 0.0))
        weighted_sum += float(val) * w
        weight_available += w
    if weight_available <= 0:
        return None
    return weighted_sum / weight_available


def _recompute_panel_bargain(panel: pd.DataFrame) -> pd.DataFrame:
    """Recompute bargain_score / components with the long-horizon formula."""
    from core.scoring import _linear_score

    out = panel.copy()
    if "valuation_vs_history" not in out.columns:
        out = _attach_valuation_vs_history(out)

    scores: list[float | None] = []
    mos_list: list[float | None] = []
    val_list: list[float | None] = []
    d52_list: list[float | None] = []

    for _, row in out.iterrows():
        graham = row.get("graham_ratio")
        mos = None
        if graham is not None and not (isinstance(graham, float) and np.isnan(graham)) and graham > 0:
            mos = _linear_score(float(graham), 0.30, 1.30)

        val = row.get("valuation_vs_history")
        if val is not None and isinstance(val, float) and np.isnan(val):
            val = None

        # Prefer freshly computed discount; fall back to stored component score.
        d52 = None
        price = row.get("price")
        high = row.get("fifty_two_week_high")
        if (
            price is not None
            and high is not None
            and not (isinstance(high, float) and np.isnan(high))
            and high > 0
            and price > 0
        ):
            d52 = _linear_score(1.0 - (float(price) / float(high)), 0.0, 0.30)
        else:
            stored = row.get("bargain_discount_52w")
            if stored is not None and not (isinstance(stored, float) and np.isnan(stored)):
                d52 = float(stored)

        components = {
            "margin_of_safety": mos,
            "valuation_vs_history": None if val is None else float(val),
            "discount_52w": d52,
        }
        scores.append(_weighted_bargain_from_components(components))
        mos_list.append(mos)
        val_list.append(None if val is None else float(val))
        d52_list.append(d52)

    out["bargain_score"] = scores
    out["bargain_margin_of_safety"] = mos_list
    out["bargain_valuation_vs_history"] = val_list
    out["bargain_discount_52w"] = d52_list
    if "bargain_rsi_oversold" in out.columns:
        out = out.drop(columns=["bargain_rsi_oversold"])
    return out


def enrich_factor_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach valuation-vs-history and recompute long-horizon bargain scores."""
    panel = _attach_valuation_vs_history(panel)
    return _recompute_panel_bargain(panel)


def build_factor_panel(
    quarter_ends: list[date] | None = None,
    force: bool = False,
    max_quarters: int | None = None,
) -> pd.DataFrame:
    """Build quarterly factor panel for historical backtesting.

    Stores all sub-signal columns for each factor group so the backtest engine
    can apply rank-then-average aggregation, mirroring core.scoring.score_universe_df.
    """
    if FACTOR_PANEL_PATH.exists() and not force:
        return pd.read_parquet(FACTOR_PANEL_PATH)

    DATA_STORE.mkdir(parents=True, exist_ok=True)
    membership = load_membership()
    fundamentals = load_fundamentals()
    prices = load_prices()

    qends = quarter_ends or QUARTER_ENDS
    if max_quarters is not None:
        qends = qends[:max_quarters]

    # Collect all sub-signal column names for the backtestable groups
    backtest_sub_cols: list[str] = [
        col
        for family in BACKTEST_FACTOR_FAMILIES
        for col in FACTOR_SCORE_COLUMNS.get(family, [])
    ]

    rows: list[dict[str, Any]] = []
    for qi, qend in enumerate(qends, start=1):
        logger.info("Building factors for %s (%d/%d)", qend, qi, len(qends))
        tickers = membership[membership["quarter_end"] == qend]["ticker"].astype(str).tolist()
        for ticker in tickers:
            raw = _build_raw_row(ticker, qend, fundamentals, prices)
            if raw.get("price") is None:
                continue
            factors = compute_historical_factors(raw)
            row: dict[str, Any] = {
                "quarter_end": qend,
                "ticker": ticker.upper(),
                "price": raw.get("price"),
                "market_cap": raw.get("market_cap"),
                "fifty_two_week_high": raw.get("fifty_two_week_high"),
            }
            for col in backtest_sub_cols:
                row[col] = factors.get(col)
            # Placeholder bargain; enrich_factor_panel fills valuation + score.
            row["bargain_score"] = None
            for comp in BARGAIN_BACKTEST_COMPONENTS:
                row[f"bargain_{comp}"] = None
            rows.append(row)

    panel = pd.DataFrame(rows)
    panel = enrich_factor_panel(panel)
    panel.to_parquet(FACTOR_PANEL_PATH, index=False)
    logger.info("Saved factor panel with %d rows", len(panel))
    return panel


def load_factor_panel(*, enrich: bool = True) -> pd.DataFrame:
    """Load factor panel; enrich with valuation-vs-history bargain if needed."""
    if not FACTOR_PANEL_PATH.exists():
        return build_factor_panel()
    panel = pd.read_parquet(FACTOR_PANEL_PATH)
    if not enrich:
        return panel
    needs_enrich = (
        "valuation_vs_history" not in panel.columns
        or "bargain_valuation_vs_history" not in panel.columns
        or "bargain_rsi_oversold" in panel.columns
    )
    if needs_enrich:
        logger.info("Enriching factor panel with long-horizon bargain components")
        panel = enrich_factor_panel(panel)
        panel.to_parquet(FACTOR_PANEL_PATH, index=False)
    return panel
