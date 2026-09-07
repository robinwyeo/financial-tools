"""Owner earnings, DCF base/bear, EPV, reverse DCF, expected return, relative valuation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.config import get_valuation_config
from core.data import percentile_rank_in_history
from core.fundamentals import (
    Fundamentals,
    cagr,
    margin_series,
    owner_earnings,
    owner_earnings_per_share,
)
from core.rates import hurdle_rate


def _cfg(config: dict[str, Any] | None) -> dict[str, Any]:
    return get_valuation_config(config)


def _is_cyclical(sector: str | None, industry: str | None, cfg: dict[str, Any]) -> bool:
    sectors = {str(s).lower() for s in (cfg.get("cyclical_sectors") or [])}
    industries = {str(s).lower() for s in (cfg.get("cyclical_industries") or [])}
    if sector and str(sector).lower() in sectors:
        return True
    if industry and str(industry).lower() in industries:
        return True
    return False


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def normalized_owner_earnings(
    fund: Fundamentals,
    cfg: dict[str, Any] | None = None,
    *,
    sector: str | None = None,
    industry: str | None = None,
) -> dict[str, Any]:
    """Non-cyclical: mean(TTM, FY, FY-1, FY-2) OE. Cyclical: 7y OE margin × TTM revenue."""
    cfg = _cfg(cfg)
    subtract = bool(cfg.get("subtract_sbc", True))
    ttm = fund.ttm if fund.ttm is not None else pd.Series(dtype="float64")
    annual = fund.annual if fund.annual is not None else pd.DataFrame()
    ttm_oe = owner_earnings(ttm, subtract_sbc=subtract)

    if _is_cyclical(sector, industry, cfg) and not annual.empty and "revenue" in annual.columns:
        years = int(cfg.get("cyclical_normalization_years", 7))
        tail = annual.tail(years)
        margins = []
        for _, row in tail.iterrows():
            oe = owner_earnings(row, subtract_sbc=subtract)
            rev = row.get("revenue")
            if oe is None or rev is None or (isinstance(rev, float) and (np.isnan(rev) or rev == 0)):
                continue
            margins.append(float(oe) / float(rev))
        ttm_rev = ttm.get("revenue") if "revenue" in ttm.index else None
        if not margins or ttm_rev is None:
            value = ttm_oe
            method = "cyclical_fallback_ttm"
        else:
            value = float(np.mean(margins)) * float(ttm_rev)
            method = "cyclical_7y_margin"
        return {
            "value": value,
            "method": method,
            "negative": value is not None and value < 0,
            "inputs": {"margins": margins, "ttm_revenue": ttm_rev, "ttm_oe": ttm_oe},
        }

    values: list[float] = []
    if ttm_oe is not None:
        values.append(float(ttm_oe))
    n_years = int(cfg.get("normalization_years", 3))
    if not annual.empty:
        for _, row in annual.tail(n_years).iterrows():
            oe = owner_earnings(row, subtract_sbc=subtract)
            if oe is not None:
                values.append(float(oe))
    value = float(np.mean(values)) if values else ttm_oe
    return {
        "value": value,
        "method": "mean_ttm_and_fy",
        "negative": value is not None and value < 0,
        "inputs": {"values": values, "ttm_oe": ttm_oe},
    }


def historical_growth(fund: Fundamentals, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """5y CAGR of owner-earnings per diluted share; revenue/share fallback."""
    cfg = _cfg(cfg)
    subtract = bool(cfg.get("subtract_sbc", True))
    annual = fund.annual
    empty = {"g_hist": None, "method": None, "n_years": 0}
    if annual is None or annual.empty or len(annual) < 6:
        return empty
    oeps = []
    rps = []
    for _, row in annual.iterrows():
        oeps.append(owner_earnings_per_share(row, subtract_sbc=subtract))
        shares = row.get("shares_diluted")
        rev = row.get("revenue")
        if shares and shares > 0 and rev is not None:
            rps.append(float(rev) / float(shares))
        else:
            rps.append(np.nan)
    oeps_s = pd.Series(oeps, index=annual.index)
    g = cagr(oeps_s, years=5)
    method = "oe_per_share_5y"
    if g is None:
        g = cagr(pd.Series(rps, index=annual.index), years=5)
        method = "revenue_per_share_5y"
    return {"g_hist": g, "method": method, "n_years": int(len(annual))}


def dcf_value(
    oe: float,
    shares: float,
    cash: float | None,
    debt: float | None,
    r: float,
    g1: float,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """10-year explicit DCF with linear fade to terminal growth after year 5."""
    cfg = _cfg(cfg)
    terminal_g = float(cfg.get("terminal_growth", 0.025))
    n = int(cfg.get("explicit_years", 10))
    fade_start = int(cfg.get("fade_start_year", 5))
    if oe is None or shares is None or shares <= 0 or r is None:
        return {"equity_value": None, "per_share": None, "cash_flows": [], "terminal_value": None}
    if r <= terminal_g:
        terminal_g = r - 0.005
    cf = float(oe)
    cash_flows: list[float] = []
    growths: list[float] = []
    for t in range(1, n + 1):
        if t <= fade_start:
            g = float(g1)
        else:
            g = float(g1) + (terminal_g - float(g1)) * (t - fade_start) / max(n - fade_start, 1)
        cf = cf * (1.0 + g)
        cash_flows.append(cf)
        growths.append(g)
    tv = cash_flows[-1] * (1.0 + terminal_g) / (r - terminal_g)
    pv_cfs = sum(cf_t / ((1.0 + r) ** (i + 1)) for i, cf_t in enumerate(cash_flows))
    pv_tv = tv / ((1.0 + r) ** n)
    firm = pv_cfs + pv_tv
    equity = firm + float(cash or 0.0) - float(debt or 0.0)
    return {
        "equity_value": equity,
        "per_share": equity / float(shares),
        "cash_flows": cash_flows,
        "growths": growths,
        "terminal_value": tv,
        "r": r,
        "g1": g1,
        "terminal_g": terminal_g,
    }


def dcf_scenarios(
    oe: float,
    shares: float,
    cash: float | None,
    debt: float | None,
    r: float,
    g_hist: float | None,
    cfg: dict[str, Any] | None = None,
    *,
    analyst_growth: float | None = None,
) -> dict[str, Any]:
    cfg = _cfg(cfg)
    floor = float(cfg.get("base_growth_floor", 0.0))
    cap = float(cfg.get("base_growth_cap", 0.15))
    g1_base = _clip(float(g_hist or 0.0), floor, cap)
    bear_mult = float(cfg.get("bear_growth_multiplier", 0.5))
    bear_cap = float(cfg.get("bear_growth_cap", 0.04))
    bear_bump = float(cfg.get("bear_rate_bump", 0.01))
    g1_bear = min(bear_mult * g1_base, bear_cap)
    base = dcf_value(oe, shares, cash, debt, r, g1_base, cfg)
    bear = dcf_value(oe, shares, cash, debt, r + bear_bump, g1_bear, cfg)
    analyst = None
    if analyst_growth is not None:
        g1_a = _clip(float(analyst_growth), floor, cap)
        analyst = dcf_value(oe, shares, cash, debt, r, g1_a, cfg)
        analyst["display_only"] = True
    return {"base": base, "bear": bear, "analyst": analyst, "g1_base": g1_base, "g1_bear": g1_bear}


def epv(
    ebit_normalized: float | None,
    tax_rate: float | None,
    r: float,
    cash: float | None,
    debt: float | None,
    shares: float | None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = _cfg(cfg)
    min_tax = float(cfg.get("min_tax_rate", 0.15))
    max_tax = float(cfg.get("max_tax_rate", 0.35))
    if ebit_normalized is None or r is None or r <= 0 or shares is None or shares <= 0:
        return {"per_share": None, "equity_value": None, "tax_rate": None}
    tax = _clip(float(tax_rate if tax_rate is not None else min_tax), min_tax, max_tax)
    nopat = float(ebit_normalized) * (1.0 - tax)
    equity = nopat / r + float(cash or 0.0) - float(debt or 0.0)
    return {"per_share": equity / float(shares), "equity_value": equity, "tax_rate": tax}


def reverse_dcf(
    price: float,
    oe: float,
    shares: float,
    cash: float | None,
    debt: float | None,
    r: float,
    cfg: dict[str, Any] | None = None,
    *,
    lo: float = -0.20,
    hi: float = 0.40,
) -> dict[str, Any]:
    """Bisection on g1 such that DCF(base structure) equals price."""
    cfg = _cfg(cfg)
    if price is None or price <= 0 or oe is None or shares is None or shares <= 0:
        return {"implied_g1": None, "iterations": 0}
    target = float(price)

    def _ps(g: float) -> float:
        return float(dcf_value(oe, shares, cash, debt, r, g, cfg)["per_share"])

    low, high = lo, hi
    f_low, f_high = _ps(low) - target, _ps(high) - target
    if f_low == 0:
        return {"implied_g1": low, "iterations": 0}
    if f_high == 0:
        return {"implied_g1": high, "iterations": 0}
    if f_low * f_high > 0:
        # Price outside the bracket — return the nearer bound.
        nearer = low if abs(f_low) < abs(f_high) else high
        return {"implied_g1": nearer, "iterations": 0, "bracketed": False}
    mid = 0.0
    for i in range(60):
        mid = 0.5 * (low + high)
        f_mid = _ps(mid) - target
        if abs(f_mid) < 1e-6 * max(target, 1.0):
            return {"implied_g1": mid, "iterations": i + 1, "bracketed": True}
        if f_low * f_mid <= 0:
            high, f_high = mid, f_mid
        else:
            low, f_low = mid, f_mid
    return {"implied_g1": mid, "iterations": 60, "bracketed": True}


def expected_return(
    oe: float | None,
    market_cap: float | None,
    g_hist: float | None,
    cfg: dict[str, Any] | None = None,
) -> float | None:
    cfg = _cfg(cfg)
    if oe is None or market_cap is None or market_cap <= 0:
        return None
    g_cap = float(cfg.get("expected_return_growth_cap", 0.06))
    g = 0.0 if g_hist is None else _clip(float(g_hist), 0.0, g_cap)
    return float(oe) / float(market_cap) + g


def relative_valuation(
    fund: Fundamentals,
    price: float | None,
    shares: float | None,
    sector_peers_df: pd.DataFrame | None = None,
    *,
    ev: float | None = None,
    ebit: float | None = None,
    oe: float | None = None,
) -> dict[str, Any]:
    """EV/EBIT and P/OE vs own 10y history and vs sector peer median."""
    annual = fund.annual if fund.annual is not None else pd.DataFrame()
    current_ev_ebit = None
    current_p_oe = None
    if ev and ebit and float(ebit) != 0:
        current_ev_ebit = float(ev) / float(ebit)
    if price and shares and oe and float(oe) != 0:
        current_p_oe = float(price) * float(shares) / float(oe)

    hist_ev, hist_poe = [], []
    if not annual.empty:
        for _, row in annual.iterrows():
            ebit_h = row.get("ebit")
            debt = row.get("debt")
            cash = row.get("cash")
            sh = row.get("shares_diluted")
            # Without a PIT price, use TTM-relative multiples only on current.
            oe_h = owner_earnings(row, subtract_sbc=True)
            if oe_h and sh and sh > 0 and price:
                hist_poe.append(float(price) * float(sh) / float(oe_h))  # last price; last-resort
            if ebit_h and ev and float(ebit_h) != 0:
                hist_ev.append(float(ev) / float(ebit_h))

    # Prefer yield history (higher = cheaper) built from annual OE and EBIT vs last known EV/price.
    ey_hist = []
    oe_yield_hist = []
    if not annual.empty:
        for _, row in annual.iterrows():
            ebit_h = row.get("ebit")
            if ebit_h and ev and float(ev) > 0:
                ey_hist.append(float(ebit_h) / float(ev))
            oe_h = owner_earnings(row, subtract_sbc=True)
            mcap = (float(price) * float(shares)) if price and shares else None
            if oe_h and mcap and mcap > 0:
                oe_yield_hist.append(float(oe_h) / mcap)

    def _block(current_mult, current_yield, yield_hist, label):
        if current_yield is None:
            pct = percentile_rank_in_history(None, yield_hist)
        else:
            pct = percentile_rank_in_history(current_yield, yield_hist)
        median = float(np.median(yield_hist)) if yield_hist else None
        dist = None
        if current_mult is not None and yield_hist:
            # Distance of the multiple from the implied median multiple.
            med_mult = (1.0 / median) if median else None
            if med_mult:
                dist = (current_mult / med_mult - 1.0) * 100.0
        return {
            "current": current_mult,
            "median_10y": (1.0 / median) if median else None,
            "percentile": pct,  # 100 = cheapest vs own history (yield rank)
            "n": len(yield_hist),
            "label": label,
            "distance_from_median_pct": dist,
        }

    ev_block = _block(
        current_ev_ebit,
        (1.0 / current_ev_ebit) if current_ev_ebit else None,
        ey_hist,
        "ev_ebit",
    )
    poe_block = _block(
        current_p_oe,
        (1.0 / current_p_oe) if current_p_oe else None,
        oe_yield_hist,
        "p_oe",
    )
    peer = {"ev_to_ebit_median": None, "p_to_oe_median": None}
    if sector_peers_df is not None and not sector_peers_df.empty:
        if "ev_to_ebit" in sector_peers_df.columns:
            peer["ev_to_ebit_median"] = float(pd.to_numeric(sector_peers_df["ev_to_ebit"], errors="coerce").median())
        if "p_to_oe" in sector_peers_df.columns:
            peer["p_to_oe_median"] = float(pd.to_numeric(sector_peers_df["p_to_oe"], errors="coerce").median())

    scores = [b["percentile"] for b in (ev_block, poe_block) if b["percentile"] is not None]
    score = float(np.mean(scores)) if scores else None
    return {"ev_ebit": ev_block, "p_oe": poe_block, "peer": peer, "score": score}


def valuation_summary(
    fund: Fundamentals,
    *,
    price: float | None,
    shares: float | None,
    cash: float | None,
    debt: float | None,
    market_cap: float | None,
    ev: float | None,
    sector: str | None,
    industry: str | None,
    uncertainty_label: str | None,
    analyst_growth: float | None = None,
    sector_peers_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    rf: float | None = None,
) -> dict[str, Any]:
    cfg = _cfg(config)
    hurdle = hurdle_rate(uncertainty_label, rf=rf, config=config)
    r = float(hurdle["rate"])
    relative_only = str(sector or "") == "Financial Services"

    assumptions = {
        "hurdle": hurdle,
        "subtract_sbc": bool(cfg.get("subtract_sbc", True)),
        "terminal_growth": cfg.get("terminal_growth"),
        "explicit_years": cfg.get("explicit_years"),
        "cyclical": _is_cyclical(sector, industry, cfg),
    }

    if relative_only:
        pb = None
        pe = None
        equity = fund.mrq.get("equity") if fund.mrq is not None else None
        if price and shares and equity and float(shares) > 0 and float(equity) != 0:
            pb = float(price) * float(shares) / float(equity)
        ni = fund.ttm.get("net_income") if fund.ttm is not None else None
        if price and shares and ni and float(ni) != 0:
            pe = float(price) * float(shares) / float(ni)
        rel = relative_valuation(fund, price, shares, sector_peers_df, ev=ev, ebit=fund.ttm.get("ebit") if fund.ttm is not None else None, oe=None)
        return {
            "relative_only": True,
            "owner_earnings": None,
            "dcf": None,
            "epv": None,
            "reverse_dcf": None,
            "expected_return": None,
            "relative": rel,
            "price_to_book": pb,
            "price_to_earnings": pe,
            "hurdle": hurdle,
            "assumptions": assumptions,
        }

    norm = normalized_owner_earnings(fund, cfg, sector=sector, industry=industry)
    growth = historical_growth(fund, cfg)
    g_hist = growth.get("g_hist")
    oe = norm.get("value")
    sh = shares or (fund.ttm.get("shares_diluted") if fund.ttm is not None else None)
    cash_v = cash if cash is not None else (fund.mrq.get("cash") if fund.mrq is not None else None)
    debt_v = debt if debt is not None else (fund.mrq.get("debt") if fund.mrq is not None else None)
    ebit_n = None
    if fund.ttm is not None and fund.ttm.get("ebit") is not None:
        ebit_n = float(fund.ttm.get("ebit"))
    elif fund.annual is not None and not fund.annual.empty:
        ebit_n = float(pd.to_numeric(fund.annual["ebit"], errors="coerce").tail(3).mean())
    tax_rate = None
    if fund.ttm is not None:
        tax = fund.ttm.get("tax_expense")
        pretax = fund.ttm.get("pretax_income")
        if tax is not None and pretax and float(pretax) != 0:
            tax_rate = float(tax) / float(pretax)

    dcf = None
    rev = {"implied_g1": None}
    if oe is not None and sh and r:
        dcf = dcf_scenarios(float(oe), float(sh), cash_v, debt_v, r, g_hist, cfg, analyst_growth=analyst_growth)
        if price:
            rev = reverse_dcf(float(price), float(oe), float(sh), cash_v, debt_v, r, cfg)

    epv_out = epv(ebit_n, tax_rate, r, cash_v, debt_v, sh, cfg)
    mcap = market_cap
    if mcap is None and price and sh:
        mcap = float(price) * float(sh)
    exp_ret = expected_return(oe, mcap, g_hist, cfg)
    rel = relative_valuation(
        fund,
        price,
        sh,
        sector_peers_df,
        ev=ev,
        ebit=ebit_n,
        oe=oe,
    )
    return {
        "relative_only": False,
        "owner_earnings": norm,
        "growth": growth,
        "dcf": dcf,
        "epv": epv_out,
        "reverse_dcf": rev,
        "expected_return": exp_ret,
        "relative": rel,
        "hurdle": hurdle,
        "assumptions": assumptions,
    }
