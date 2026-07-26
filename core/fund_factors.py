"""Factor computations for pooled funds (ETFs and mutual funds).

Stock factors (value, quality, balance sheet, …) rely on company financial
statements and analyst coverage that do not exist for funds. Funds are instead
ranked on fund-appropriate signals with strong empirical support: fees are the
best predictor of long-run relative fund performance, followed by realized
risk-adjusted returns, volatility, and momentum.
"""

from __future__ import annotations

from typing import Any


def compute_cost_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    """Low cost: negated expense ratio so cheaper funds rank higher."""
    expense_ratio = raw.get("expense_ratio")
    low_cost = -expense_ratio if expense_ratio is not None else None
    return {"expense_ratio": expense_ratio, "low_cost": low_cost}


def compute_performance_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Realized annualized total returns over 3 and 5 years."""
    return {
        "return_1y": raw.get("return_1y"),
        "return_3y": raw.get("return_3y"),
        "return_5y": raw.get("return_5y"),
    }


def compute_risk_adjusted_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    """Sharpe-style ratio: trailing 1y return per unit of annualized volatility."""
    ret = raw.get("return_1y")
    vol = raw.get("volatility_12m")
    sharpe = None
    if ret is not None and vol is not None and vol > 0:
        sharpe = ret / vol
    return {"sharpe_1y": sharpe}


def compute_fund_volatility_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Low volatility (1/σ) and drawdown protection (negated max drawdown)."""
    vol = raw.get("volatility_12m")
    low_volatility = 1.0 / vol if vol and vol > 0 else None
    max_drawdown = raw.get("max_drawdown")
    drawdown_protection = -abs(max_drawdown) if max_drawdown is not None else None
    return {
        "volatility_12m": vol,
        "low_volatility": low_volatility,
        "max_drawdown": max_drawdown,
        "drawdown_protection": drawdown_protection,
    }


def compute_fund_momentum_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    return {"momentum_12_1": raw.get("momentum_12_1")}


def compute_income_factor(raw: dict[str, Any]) -> dict[str, float | None]:
    return {"distribution_yield": raw.get("distribution_yield")}


def compute_fund_factors(raw: dict[str, Any]) -> dict[str, float | None]:
    """Compute all raw fund factor values for a single fund."""
    out: dict[str, float | None] = {}
    out.update(compute_cost_factor(raw))
    out.update(compute_performance_factors(raw))
    out.update(compute_risk_adjusted_factor(raw))
    out.update(compute_fund_volatility_factors(raw))
    out.update(compute_fund_momentum_factor(raw))
    out.update(compute_income_factor(raw))
    return out


# Columns used for cross-sectional fund scoring, mirroring FACTOR_SCORE_COLUMNS:
# each group ranks its sub-signals cross-sectionally (within fund category when
# the group is large enough), averages available sub-signal percentiles, and the
# composite is the weighted average of group percentile scores.
FUND_FACTOR_SCORE_COLUMNS: dict[str, list[str]] = {
    "cost": ["low_cost"],
    "performance": ["return_3y", "return_5y"],
    "risk_adjusted": ["sharpe_1y"],
    "low_volatility": ["low_volatility", "drawdown_protection"],
    "momentum": ["momentum_12_1"],
    "income": ["distribution_yield"],
}
