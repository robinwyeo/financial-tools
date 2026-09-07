"""Empirical factor computations inspired by OpenSourceAP definitions."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _ratio(num: float | None, den: float | None) -> float | None:
    if num is None or den is None or den == 0:
        return None
    return num / den


def compute_value_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Value: earnings yield, book-to-market, FCF yield."""
    ev = raw.get("enterprise_value")
    ebit = raw.get("ebit")
    market_cap = raw.get("market_cap")
    fcf = raw.get("free_cashflow")
    book_equity = raw.get("book_equity")

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
    revenue = raw.get("revenue")
    book_equity = raw.get("book_equity")

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
    book_equity = raw.get("book_equity")
    debt_to_equity = raw.get("debt_to_equity")

    net_cash = None
    if total_cash is not None and total_debt is not None:
        net_cash = total_cash - total_debt

    net_cash_to_mcap = _ratio(net_cash, market_cap)

    negative_equity = False
    if book_equity is not None and book_equity <= 0:
        negative_equity = True

    if book_equity is not None and book_equity > 0 and total_debt is not None:
        debt_to_equity = total_debt / book_equity

    low_leverage = None
    if negative_equity:
        low_leverage = None
    elif debt_to_equity is not None:
        if debt_to_equity <= 0:
            low_leverage = 1.0
        else:
            low_leverage = 1.0 / (1.0 + debt_to_equity)

    return {
        "net_cash": net_cash,
        "net_cash_to_mcap": net_cash_to_mcap,
        "low_leverage": low_leverage,
        "negative_equity": negative_equity,
        "debt_to_equity": debt_to_equity,
    }


def compute_graham_value(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Graham Number: sqrt(22.5 * EPS * BVPS) / price.
    Ratio > 1 means price below Graham fair value (margin of safety).
    current_ratio retained for display only; not used in composite scoring.
    """
    price = raw.get("price")
    trailing_eps = raw.get("trailing_eps")
    book_value = raw.get("book_value")  # BVPS
    book_equity = raw.get("book_equity")
    shares = raw.get("shares_outstanding")
    if book_equity is not None and shares and shares > 0:
        book_value = book_equity / shares
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
    Return on invested capital.

    Preferred invested capital = NWC ex-cash + net PPE.
    Fallback = equity + debt - cash. Floor IC at 10% of total assets and clip
    ROIC to [-1, 2] so cash-rich / dual-class names cannot dominate the rank.
    """
    ebit = raw.get("ebit")
    total_debt = raw.get("total_debt")
    book_equity = raw.get("book_equity")
    total_cash = raw.get("total_cash")
    current_assets = raw.get("current_assets")
    current_liabilities = raw.get("current_liabilities")
    short_term_debt = raw.get("short_term_debt") or raw.get("debt_st")
    ppe_net = raw.get("ppe_net")
    total_assets = raw.get("total_assets")

    invested_capital = None
    roic_basis = None
    if (
        current_assets is not None
        and current_liabilities is not None
        and ppe_net is not None
        and total_cash is not None
    ):
        st_debt = short_term_debt or 0.0
        nwc_ex_cash = (current_assets - total_cash) - (current_liabilities - st_debt)
        invested_capital = nwc_ex_cash + ppe_net
        roic_basis = "nwc_ppe"
    elif total_debt is not None and book_equity is not None and total_cash is not None:
        invested_capital = total_debt + book_equity - total_cash
        roic_basis = "equity_debt_cash"

    if invested_capital is not None and total_assets and total_assets > 0:
        invested_capital = max(invested_capital, 0.10 * total_assets)

    roic = None
    if ebit is not None and invested_capital is not None and invested_capital > 0:
        roic = ebit / invested_capital
        roic = max(-1.0, min(2.0, roic))

    return {
        "invested_capital": invested_capital,
        "roic": roic,
        "roic_basis": roic_basis,
    }


def compute_leverage_metrics(raw: dict[str, Any]) -> dict[str, float | None]:
    """Net debt / EBITDA and EBIT / interest. Missing interest → coverage None."""
    ebit = raw.get("ebit")
    da = raw.get("depreciation")
    total_debt = raw.get("total_debt")
    total_cash = raw.get("total_cash")
    interest = raw.get("interest_expense")

    ebitda = None
    if ebit is not None:
        ebitda = ebit + (da or 0.0)

    net_debt = None
    if total_debt is not None and total_cash is not None:
        net_debt = total_debt - total_cash
    elif total_debt is not None:
        net_debt = total_debt

    net_debt_to_ebitda = None
    if net_debt is not None and ebitda and ebitda > 0:
        net_debt_to_ebitda = net_debt / ebitda

    interest_coverage = None
    if ebit is not None and interest is not None and interest > 0:
        interest_coverage = ebit / interest

    return {
        "ebitda": ebitda,
        "net_debt": net_debt,
        "net_debt_to_ebitda": net_debt_to_ebitda,
        "interest_coverage": interest_coverage,
        "leverage_quality": None if net_debt_to_ebitda is None else -float(net_debt_to_ebitda),
        "equity_to_assets": _ratio(raw.get("book_equity") or raw.get("equity"), raw.get("total_assets")),
        "ev_to_ebit": _ratio(raw.get("enterprise_value"), raw.get("ebit")),
        "p_to_oe": raw.get("p_to_oe"),
        "owner_earnings_norm": raw.get("owner_earnings_norm"),
    }


def compute_fcf_conversion(raw: dict[str, Any]) -> dict[str, float | None]:
    """FCF margin and 3y FCF/NI conversion (pass-through of history metrics)."""
    revenue = raw.get("revenue")
    fcf = raw.get("free_cashflow")
    ocf = raw.get("operating_cashflow")
    capex = raw.get("capex")
    if fcf is None and ocf is not None:
        fcf = float(ocf) - abs(float(capex or 0.0))
    fcf_margin = _ratio(fcf, revenue)
    fcf_conversion_3y = raw.get("fcf_conversion_3y")
    return {"fcf_margin": fcf_margin, "fcf_conversion_3y": fcf_conversion_3y}


def compute_dilution(raw: dict[str, Any]) -> dict[str, float | None]:
    """Share-count CAGR; anti_dilution is the inverted rank input."""
    share_cagr_3y = raw.get("share_cagr_3y")
    if share_cagr_3y is None:
        shares = raw.get("shares_outstanding")
        prior = raw.get("shares_prior")
        if shares and prior and prior > 0:
            share_cagr_3y = (float(shares) / float(prior)) - 1.0
    anti_dilution = None if share_cagr_3y is None else -float(share_cagr_3y)
    return {"share_cagr_3y": share_cagr_3y, "anti_dilution": anti_dilution}


def compute_stability(raw: dict[str, Any]) -> dict[str, float | None]:
    """5y ROIC mean/std, gross-margin change, revenue CAGR. Lower vol ranks higher."""
    roic_5y_mean = raw.get("roic_5y_mean")
    roic_5y_std = raw.get("roic_5y_std")
    gross_margin_5y_delta = raw.get("gross_margin_5y_delta")
    revenue_5y_cagr = raw.get("revenue_5y_cagr")
    stability_roic = None if roic_5y_std is None else -float(roic_5y_std)
    return {
        "roic_5y_mean": roic_5y_mean,
        "roic_5y_std": roic_5y_std,
        "stability_roic": stability_roic,
        "gross_margin_5y_delta": gross_margin_5y_delta,
        "revenue_5y_cagr": revenue_5y_cagr,
    }


def compute_altman_z(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Altman Z-Score (1968). All five components required; missing any → None
    (no partial-component renormalization).
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
    if current_assets is None or current_liabilities is None:
        return {"altman_z": None}
    if retained_earnings is None or ebit is None:
        return {"altman_z": None}
    if market_cap is None or not total_liabilities or total_liabilities <= 0:
        return {"altman_z": None}
    if revenue is None:
        return {"altman_z": None}

    x1 = (current_assets - current_liabilities) / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = market_cap / total_liabilities
    x5 = revenue / total_assets
    altman_z = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5
    return {"altman_z": altman_z}


def compute_altman_z_double_prime(raw: dict[str, Any]) -> dict[str, float | None]:
    """
    Altman Z'' (1995) for non-manufacturers:
    Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
      X1 = working capital / total assets
      X2 = retained earnings / total assets
      X3 = EBIT / total assets
      X4 = book equity / total liabilities
    All four components required.
    """
    current_assets = raw.get("current_assets")
    current_liabilities = raw.get("current_liabilities")
    retained_earnings = raw.get("retained_earnings")
    ebit = raw.get("ebit")
    total_assets = raw.get("total_assets")
    total_liabilities = raw.get("total_liabilities")
    book_equity = raw.get("book_equity")

    if not total_assets or total_assets <= 0:
        return {"altman_z_pp": None}
    if current_assets is None or current_liabilities is None:
        return {"altman_z_pp": None}
    if retained_earnings is None or ebit is None:
        return {"altman_z_pp": None}
    if book_equity is None or not total_liabilities or total_liabilities <= 0:
        return {"altman_z_pp": None}

    x1 = (current_assets - current_liabilities) / total_assets
    x2 = retained_earnings / total_assets
    x3 = ebit / total_assets
    x4 = book_equity / total_liabilities
    z_pp = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
    return {"altman_z_pp": z_pp}


def compute_all_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Compute all raw factor values for a ticker."""
    from core.estimates import compute_revision_factors
    from core.insiders import compute_insider_factor

    out: dict[str, float | None] = {}
    out.update(compute_value_factors(raw))
    out.update(compute_momentum_factor(raw))
    out.update(compute_quality_factors(raw))
    out.update(compute_low_volatility_factor(raw))
    out.update(compute_investment_factor(raw))
    out.update(compute_piotroski_f_score(raw))
    out.update(compute_garp_factor(raw))
    out.update(compute_balance_sheet_strength(raw))
    out.update(compute_graham_value(raw))
    out.update(compute_downside_protection(raw))
    out.update(compute_earnings_quality(raw))
    out.update(compute_shareholder_yield(raw))
    out.update(compute_capital_efficiency(raw))
    out.update(compute_leverage_metrics(raw))
    out.update(compute_fcf_conversion(raw))
    out.update(compute_dilution(raw))
    out.update(compute_stability(raw))
    out.update(compute_altman_z(raw))
    out.update(compute_altman_z_double_prime(raw))
    revisions = compute_revision_factors(raw)
    out["revision_agreement"] = revisions.get("revision_agreement")
    out["revision_magnitude"] = revisions.get("revision_magnitude")
    out["earnings_surprise"] = revisions.get("earnings_surprise")
    insider = compute_insider_factor(raw)
    out["insider_buying"] = insider.get("insider_buying")
    return out


# Columns used for cross-sectional scoring.
# Each group is scored by: rank each sub-signal cross-sectionally → average
# available sub-signal percentiles → group percentile score.
# Composite = weighted average of group percentile scores.
#
# graham_ratio is intentionally NOT in the value group: it already drives 55%
# of the bargain score, and including it here double-counted the same signal
# across both legs of the Buy gate.
FACTOR_SCORE_COLUMNS: dict[str, list[str]] = {
    "value": ["earnings_yield", "fcf_yield", "book_to_market"],
    "garp": ["garp"],
    "quality": [
        "gross_profitability", "roe", "roa", "profit_margin",
        "roic", "earnings_quality", "financial_strength",
    ],
    "balance_sheet": ["net_cash_to_mcap", "low_leverage", "altman_z"],
    "momentum": ["momentum_12_1"],
    "low_volatility": ["low_volatility"],
    "capital_discipline": ["shareholder_yield", "investment"],
    "estimate_revisions": [
        "revision_agreement",
        "revision_magnitude",
        "earnings_surprise",
    ],
    "insider": ["insider_buying"],
}

# Five-bucket quality screen (Layer A). Higher-is-better rank inputs.
QUALITY_SCORE_COLUMNS: dict[str, list[str]] = {
    "profitability": ["gross_profitability", "roic", "fcf_margin"],
    "earnings_quality": ["earnings_quality", "fcf_conversion_3y"],
    "financial_strength": ["financial_strength", "leverage_quality", "interest_coverage"],
    "stability": ["roic_5y_mean", "stability_roic", "gross_margin_5y_delta", "revenue_5y_cagr"],
    "capital_discipline": ["shareholder_yield", "investment", "anti_dilution"],
}

QUALITY_SUB_BUCKETS: dict[str, list[list[str]]] = {
    "profitability": [["gross_profitability"], ["roic"], ["fcf_margin"]],
    "earnings_quality": [["earnings_quality"], ["fcf_conversion_3y"]],
    "financial_strength": [["financial_strength"], ["leverage_quality"], ["interest_coverage"]],
    "stability": [["roic_5y_mean"], ["stability_roic"], ["gross_margin_5y_delta"], ["revenue_5y_cagr"]],
    "capital_discipline": [["shareholder_yield"], ["investment"], ["anti_dilution"]],
}

FINANCIALS_QUALITY_COLUMNS: dict[str, list[str]] = {
    "profitability": ["roe", "roa"],
    "earnings_quality": ["earnings_quality"],
    "financial_strength": ["equity_to_assets"],
    "stability": ["roic_5y_mean", "stability_roic", "revenue_5y_cagr"],
    "capital_discipline": ["anti_dilution"],
}

# Sub-buckets de-correlate a factor group before averaging: sub-signal
# percentiles are averaged within each bucket, then bucket scores are averaged.
# Without this, the five highly correlated profitability ratios in the quality
# group carried 5/7 of the group score, silently drowning out earnings quality
# (accruals) and financial strength (Piotroski). With buckets each theme
# contributes 1/3. Groups not listed here average their sub-signals equally.
FACTOR_SUB_BUCKETS: dict[str, list[list[str]]] = {
    "quality": [
        ["gross_profitability", "roe", "roa", "profit_margin", "roic"],
        ["earnings_quality"],
        ["financial_strength"],
    ],
}
