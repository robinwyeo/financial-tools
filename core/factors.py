"""Empirical factor computations inspired by OpenSourceAP definitions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from core.analysts import recommendation_period_shift


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def compute_value_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Value: earnings yield, book-to-market, FCF yield."""
    ev = raw.get("enterprise_value")
    ebit = raw.get("ebit")
    market_cap = raw.get("market_cap")
    book_value = raw.get("book_value")
    shares = raw.get("shares_outstanding")
    fcf = raw.get("free_cashflow")

    book_equity = None
    if book_value is not None and shares is not None:
        book_equity = book_value * shares

    earnings_yield = _ratio(ebit, ev)
    book_to_market = _ratio(book_equity, market_cap)
    fcf_yield = _ratio(fcf, market_cap)

    return {
        "earnings_yield": earnings_yield,
        "book_to_market": book_to_market,
        "fcf_yield": fcf_yield,
    }


def compute_momentum_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    return {"momentum_12_1": raw.get("momentum_12_1")}


def compute_quality_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Quality: gross profitability, ROE, ROA, profit margin."""
    gross_profit = raw.get("gross_profit")
    total_assets = raw.get("total_assets")
    net_income = raw.get("net_income")
    market_cap = raw.get("market_cap")
    book_value = raw.get("book_value")
    shares = raw.get("shares_outstanding")
    revenue = raw.get("revenue")

    book_equity = book_value * shares if book_value and shares else None

    gross_profitability = _ratio(gross_profit, total_assets)
    roa = _ratio(net_income, total_assets)
    roe = _ratio(net_income, book_equity)
    profit_margin = _ratio(net_income, revenue)

    return {
        "gross_profitability": gross_profitability,
        "roe": roe,
        "roa": roa,
        "profit_margin": profit_margin,
    }


def compute_low_volatility_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    vol = raw.get("volatility_12m")
    inv_vol = 1.0 / vol if vol and vol > 0 else None
    return {"volatility_12m": vol, "low_volatility": inv_vol}


def compute_investment_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    """Investment: asset growth YoY (lower growth = better, inverted at scoring)."""
    ta = raw.get("total_assets")
    ta_prior = raw.get("total_assets_prior")
    if ta is None or ta_prior is None or ta_prior == 0:
        return {"asset_growth": None, "investment": None}
    growth = (ta / ta_prior) - 1.0
    return {"asset_growth": growth, "investment": -growth}


def compute_earnings_revisions(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Analyst recommendation momentum: recent upgrades minus downgrades.
    Uses yfinance recommendation history only (no target-price blend to avoid
    double-counting with the analyst_upside good-buy gate).
    """
    recs: pd.DataFrame = raw.get("recommendations", pd.DataFrame())
    score = None

    if recs is not None and not recs.empty:
        df = recs.copy()
        col_map = {c.lower(): c for c in df.columns}
        action_col = col_map.get("action") or col_map.get("rating")
        if action_col:
            recent = df.tail(20)
            upgrades = recent[action_col].astype(str).str.lower().str.contains("up|raise|buy", na=False).sum()
            downgrades = recent[action_col].astype(str).str.lower().str.contains("down|lower|sell", na=False).sum()
            score = float(upgrades - downgrades)
        else:
            period_score, _, _ = recommendation_period_shift(df)
            if period_score is not None:
                score = period_score * 10.0

    return {"earnings_revisions": score}


def compute_piotroski_f_score(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Piotroski F-Score (0-9) — financial strength.
    """
    score = 0
    checks = 0

    net_income = raw.get("net_income")
    net_income_prior = raw.get("net_income_prior")
    total_assets = raw.get("total_assets")
    total_assets_prior = raw.get("total_assets_prior")
    operating_cashflow = raw.get("operating_cashflow")
    long_term_debt = raw.get("long_term_debt")
    long_term_debt_prior = raw.get("long_term_debt_prior")
    current_assets = raw.get("current_assets")
    current_liabilities = raw.get("current_liabilities")
    current_assets_prior = raw.get("current_assets_prior")
    current_liabilities_prior = raw.get("current_liabilities_prior")
    shares = raw.get("shares_outstanding")
    shares_prior = raw.get("shares_prior")
    gross_profit = raw.get("gross_profit")
    gross_profit_prior = raw.get("gross_profit_prior")
    revenue = raw.get("revenue")
    revenue_prior = raw.get("revenue_prior")

    # 1. Positive ROA
    if net_income is not None and total_assets and total_assets > 0:
        roa = net_income / total_assets
        score += int(roa > 0)
        checks += 1

    # 2. Positive operating cash flow
    if operating_cashflow is not None:
        score += int(operating_cashflow > 0)
        checks += 1

    # 3. ROA increase
    if net_income is not None and net_income_prior is not None and total_assets and total_assets_prior:
        roa = net_income / total_assets
        roa_prior = net_income_prior / total_assets_prior if total_assets_prior else 0
        score += int(roa > roa_prior)
        checks += 1

    # 4. Accruals: OCF > Net Income
    if operating_cashflow is not None and net_income is not None:
        score += int(operating_cashflow > net_income)
        checks += 1

    # 5. Lower leverage
    if long_term_debt is not None and long_term_debt_prior is not None and total_assets and total_assets_prior:
        lev = long_term_debt / total_assets if total_assets else 0
        lev_prior = long_term_debt_prior / total_assets_prior if total_assets_prior else 0
        score += int(lev <= lev_prior)
        checks += 1

    # 6. Higher current ratio
    if current_assets and current_liabilities and current_assets_prior and current_liabilities_prior:
        cr = current_assets / current_liabilities if current_liabilities else 0
        cr_prior = current_assets_prior / current_liabilities_prior if current_liabilities_prior else 0
        score += int(cr >= cr_prior)
        checks += 1

    # 7. No new shares
    if shares is not None and shares_prior is not None:
        score += int(shares <= shares_prior)
        checks += 1

    # 8. Higher gross margin
    if gross_profit and revenue and gross_profit_prior and revenue_prior and revenue > 0 and revenue_prior > 0:
        gm = gross_profit / revenue
        gm_prior = gross_profit_prior / revenue_prior
        score += int(gm >= gm_prior)
        checks += 1

    # 9. Higher asset turnover
    if revenue and total_assets and revenue_prior and total_assets_prior:
        at = revenue / total_assets
        at_prior = revenue_prior / total_assets_prior
        score += int(at >= at_prior)
        checks += 1

    if checks == 0:
        return {"piotroski_f_score": None, "financial_strength": None}

    normalized = (score / checks) * 9.0
    return {"piotroski_f_score": float(score), "financial_strength": float(normalized)}


def _pct_from_decimal(val: float | None) -> float | None:
    """Convert yfinance decimal rates (0.15) to percentage points (15)."""
    if val is None:
        return None
    if abs(val) > 1:
        return val
    return val * 100.0


def compute_garp_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Lynch dividend-adjusted PEG: (growth% + yield%) / P/E.
    Higher = better (more growth/yield per unit of P/E).
    """
    trailing_pe = raw.get("trailing_pe")
    earnings_growth = raw.get("earnings_growth")
    dividend_yield = raw.get("dividend_yield")
    trailing_peg = raw.get("trailing_peg_ratio")

    growth_pct = _pct_from_decimal(earnings_growth)
    yield_pct = _pct_from_decimal(dividend_yield)

    peg_ratio = trailing_peg
    if trailing_pe and trailing_pe > 0 and growth_pct is not None and growth_pct > 0:
        peg_ratio = trailing_pe / growth_pct

    dividend_adjusted_peg = None
    if trailing_pe and trailing_pe > 0:
        numerator_parts = [p for p in [growth_pct, yield_pct] if p is not None and p > 0]
        if numerator_parts:
            dividend_adjusted_peg = sum(numerator_parts) / trailing_pe

    garp_score = dividend_adjusted_peg
    if garp_score is None and peg_ratio is not None and peg_ratio > 0:
        garp_score = 1.0 / peg_ratio

    return {
        "peg_ratio": peg_ratio,
        "dividend_adjusted_peg": dividend_adjusted_peg,
        "garp": garp_score,
    }


def compute_balance_sheet_strength(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Lynch net-cash position and low leverage.
    Higher net cash / lower debt-to-equity = stronger balance sheet.
    """
    total_cash = raw.get("total_cash")
    total_debt = raw.get("total_debt")
    market_cap = raw.get("market_cap")
    debt_to_equity = raw.get("debt_to_equity")

    net_cash = None
    if total_cash is not None and total_debt is not None:
        net_cash = total_cash - total_debt

    net_cash_to_mcap = _ratio(net_cash, market_cap)

    low_leverage = None
    if debt_to_equity is not None:
        if debt_to_equity <= 0:
            low_leverage = 1.0
        else:
            low_leverage = 1.0 / (1.0 + debt_to_equity)

    return {
        "net_cash": net_cash,
        "net_cash_to_mcap": net_cash_to_mcap,
        "low_leverage": low_leverage,
    }


def compute_graham_value(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Graham Number: sqrt(22.5 * EPS * BVPS) / price.
    Ratio > 1 means price below Graham fair value (margin of safety).
    current_ratio retained for display only; not used in composite scoring.
    """
    price = raw.get("price")
    trailing_eps = raw.get("trailing_eps")
    book_value = raw.get("book_value")  # BVPS from yfinance
    current_ratio = raw.get("current_ratio_info")
    current_assets = raw.get("current_assets")
    current_liabilities = raw.get("current_liabilities")

    if current_ratio is None and current_assets and current_liabilities and current_liabilities > 0:
        current_ratio = current_assets / current_liabilities

    graham_fair_value = None
    graham_ratio = None
    if (
        trailing_eps is not None
        and trailing_eps > 0
        and book_value is not None
        and book_value > 0
    ):
        graham_fair_value = math.sqrt(22.5 * trailing_eps * book_value)
        graham_ratio = _ratio(graham_fair_value, price)

    return {
        "graham_fair_value": graham_fair_value,
        "graham_ratio": graham_ratio,
        "current_ratio": current_ratio,
    }


def compute_downside_protection(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Historical downside metrics (retained for display; excluded from composite).
    """
    max_drawdown = raw.get("max_drawdown")
    downside_deviation = raw.get("downside_deviation")

    drawdown_score = -max_drawdown if max_drawdown is not None else None

    downside_score = None
    if downside_deviation is not None:
        downside_score = 1.0 / (1.0 + downside_deviation)

    parts = [v for v in [drawdown_score, downside_score] if v is not None]
    downside_protection = float(np.mean(parts)) if parts else None

    return {
        "max_drawdown": max_drawdown,
        "downside_deviation": downside_deviation,
        "downside_protection": downside_protection,
    }


def compute_earnings_quality(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Sloan (1996) accruals anomaly: earnings backed by cash flow outperform.
    accruals = (net_income - operating_cashflow) / total_assets
    Score = -accruals so higher score means lower accruals (cash-backed earnings).
    """
    net_income = raw.get("net_income")
    operating_cashflow = raw.get("operating_cashflow")
    total_assets = raw.get("total_assets")

    accruals = None
    earnings_quality = None
    if net_income is not None and operating_cashflow is not None and total_assets and total_assets > 0:
        accruals = (net_income - operating_cashflow) / total_assets
        earnings_quality = -accruals

    return {
        "accruals": accruals,
        "earnings_quality": earnings_quality,
    }


def compute_shareholder_yield(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Faber shareholder yield: total cash returned to shareholders as a fraction of market cap.
    shareholder_yield = (dividends_paid + net_buybacks) / market_cap
    Higher = better (more capital handed back to shareholders).
    """
    market_cap = raw.get("market_cap")
    dividends_paid = raw.get("dividends_paid")
    repurchase_of_stock = raw.get("repurchase_of_stock")

    shareholder_yield = None
    net_buybacks = None
    total_returned = None

    if repurchase_of_stock is not None:
        net_buybacks = -repurchase_of_stock

    if market_cap and market_cap > 0:
        parts = []
        if dividends_paid is not None:
            parts.append(abs(dividends_paid))
        if net_buybacks is not None and net_buybacks > 0:
            parts.append(net_buybacks)
        if parts:
            total_returned = sum(parts)
            shareholder_yield = total_returned / market_cap

    return {
        "dividends_paid": dividends_paid,
        "net_buybacks": net_buybacks,
        "total_returned_to_shareholders": total_returned,
        "shareholder_yield": shareholder_yield,
    }


def compute_capital_efficiency(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Greenblatt Magic Formula return on invested capital.
    roic = ebit / invested_capital
    invested_capital = total_debt + book_equity - total_cash (floored at 1 to avoid distortion)
    Higher ROIC = better capital allocation.
    """
    ebit = raw.get("ebit")
    total_debt = raw.get("total_debt")
    book_value = raw.get("book_value")
    shares = raw.get("shares_outstanding")
    total_cash = raw.get("total_cash")

    book_equity = book_value * shares if book_value is not None and shares is not None else None

    invested_capital = None
    roic = None

    if total_debt is not None and book_equity is not None:
        ic = total_debt + book_equity - (total_cash or 0.0)
        floor = max(abs(book_equity) * 0.01, 1.0) if book_equity else 1.0
        invested_capital = max(ic, floor)

    if ebit is not None and invested_capital is not None and invested_capital > 0:
        roic = ebit / invested_capital

    return {
        "invested_capital": invested_capital,
        "roic": roic,
    }


def compute_altman_z(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Altman Z-Score (1968): multi-ratio distress predictor.
    Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
      X1 = working capital / total assets
      X2 = retained earnings / total assets
      X3 = EBIT / total assets
      X4 = market cap / total liabilities
      X5 = revenue / total assets
    Higher Z = lower distress risk.
    """
    current_assets = raw.get("current_assets")
    current_liabilities = raw.get("current_liabilities")
    retained_earnings = raw.get("retained_earnings")
    ebit = raw.get("ebit")
    total_assets = raw.get("total_assets")
    total_liabilities = raw.get("total_liabilities")
    market_cap = raw.get("market_cap")
    revenue = raw.get("revenue")

    if not total_assets or total_assets <= 0:
        return {"altman_z": None}

    x1 = x2 = x3 = x4 = x5 = None

    if current_assets is not None and current_liabilities is not None:
        x1 = (current_assets - current_liabilities) / total_assets

    if retained_earnings is not None:
        x2 = retained_earnings / total_assets

    if ebit is not None:
        x3 = ebit / total_assets

    if market_cap is not None and total_liabilities and total_liabilities > 0:
        x4 = market_cap / total_liabilities

    if revenue is not None:
        x5 = revenue / total_assets

    components = [x1, x2, x3, x4, x5]
    weights = [1.2, 1.4, 3.3, 0.6, 1.0]

    available = [(w, v) for w, v in zip(weights, components) if v is not None]
    if not available:
        return {"altman_z": None}

    total_weight = sum(w for w, _ in available)
    full_weight = sum(weights)
    z_partial = sum(w * v for w, v in available)
    altman_z = z_partial * (full_weight / total_weight) if total_weight > 0 else None

    return {"altman_z": altman_z}


def compute_all_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Compute all raw factor values for a ticker."""
    out: dict[str, float | None] = {}
    out.update(compute_value_factors(raw))
    out.update(compute_momentum_factor(raw))
    out.update(compute_quality_factors(raw))
    out.update(compute_low_volatility_factor(raw))
    out.update(compute_investment_factor(raw))
    out.update(compute_earnings_revisions(raw))
    out.update(compute_piotroski_f_score(raw))
    out.update(compute_garp_factor(raw))
    out.update(compute_balance_sheet_strength(raw))
    out.update(compute_graham_value(raw))
    out.update(compute_downside_protection(raw))
    out.update(compute_earnings_quality(raw))
    out.update(compute_shareholder_yield(raw))
    out.update(compute_capital_efficiency(raw))
    out.update(compute_altman_z(raw))
    return out


# Columns used for cross-sectional scoring.
# Each group is scored by: rank each sub-signal cross-sectionally → average
# available sub-signal percentiles → group percentile score.
# Composite = weighted average of group percentile scores.
FACTOR_SCORE_COLUMNS: dict[str, list[str]] = {
    "value": ["earnings_yield", "fcf_yield", "book_to_market", "graham_ratio"],
    "garp": ["garp"],
    "quality": [
        "gross_profitability", "roe", "roa", "profit_margin",
        "roic", "earnings_quality", "financial_strength",
    ],
    "balance_sheet": ["net_cash_to_mcap", "low_leverage", "altman_z"],
    "momentum": ["momentum_12_1"],
    "low_volatility": ["low_volatility"],
    "capital_discipline": ["shareholder_yield", "investment"],
    "earnings_revisions": ["earnings_revisions"],
}
