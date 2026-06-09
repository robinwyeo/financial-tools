"""Shared constants for the backtest harness."""

from __future__ import annotations

from datetime import date

from core.config import ROOT

BACKTEST_ROOT = ROOT / "backtest"
DATA_STORE = BACKTEST_ROOT / "data" / "store"
RESULTS_DIR = BACKTEST_ROOT / "results"

# SEC Financial Statement Data Sets start at 2009Q2; first usable quarter-end is 2010Q1.
BACKTEST_START = date(2010, 3, 31)
BACKTEST_END = date(2026, 3, 31)

# Walk-forward splits (quarter-end dates inclusive).
TRAIN_END = date(2018, 12, 31)
VALID_END = date(2022, 12, 31)

# Factors reconstructable without historical analyst data.
BACKTEST_FACTOR_FAMILIES: tuple[str, ...] = (
    "value",
    "garp",
    "graham_value",
    "quality",
    "financial_strength",
    "earnings_quality",
    "capital_efficiency",
    "momentum",
    "low_volatility",
    "downside_protection",
    "balance_sheet_strength",
    "distress_risk",
    "shareholder_yield",
    "investment",
)

# Excluded from historical reconstruction (no free analyst history).
EXCLUDED_COMPOSITE_FACTORS: frozenset[str] = frozenset({"earnings_revisions"})

BARGAIN_BACKTEST_COMPONENTS: tuple[str, ...] = (
    "margin_of_safety",
    "discount_ath",
    "discount_52w",
    "rsi_oversold",
)

EXCLUDED_BARGAIN_COMPONENTS: frozenset[str] = frozenset({"analyst_upside"})

FACTOR_THEMES: dict[str, list[str]] = {
    "value": ["value", "garp", "graham_value"],
    "quality": ["quality", "financial_strength", "earnings_quality", "capital_efficiency"],
    "trend": ["momentum"],
    "risk": ["low_volatility", "downside_protection"],
    "solvency": ["balance_sheet_strength", "distress_risk"],
    "capital_allocation": ["shareholder_yield", "investment"],
}

# Within-theme proportions from config.yaml (used when scaling theme weights).
WITHIN_THEME_PROPORTIONS: dict[str, dict[str, float]] = {
    "value": {"value": 0.07, "garp": 0.07, "graham_value": 0.07},
    "quality": {
        "quality": 0.06,
        "financial_strength": 0.06,
        "earnings_quality": 0.06,
        "capital_efficiency": 0.06,
    },
    "trend": {"momentum": 0.08},
    "risk": {"low_volatility": 0.05, "downside_protection": 0.05},
    "solvency": {"balance_sheet_strength": 0.05, "distress_risk": 0.05},
    "capital_allocation": {"shareholder_yield": 0.10, "investment": 0.10},
}

DEFAULT_BARGAIN_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.30,
    "discount_ath": 0.25,
    "discount_52w": 0.15,
    "rsi_oversold": 0.15,
}

ROLLING_WINDOW_MONTHS = 36
TOP_QUINTILE_FRAC = 0.20
DCA_INVESTMENT_USD = 20_000.0
DCA_TOP_N = 5
DEFAULT_DELIST_RETURN = -0.50

SEC_USER_AGENT = "financial-tools-backtest contact@example.com"


def _quarter_ends(start: date, end: date) -> list[date]:
    """Quarter-end dates from start through end."""
    out: list[date] = []
    year, month = start.year, ((start.month - 1) // 3 + 1) * 3
    while True:
        if month == 3:
            qend = date(year, 3, 31)
        elif month == 6:
            qend = date(year, 6, 30)
        elif month == 9:
            qend = date(year, 9, 30)
        else:
            qend = date(year, 12, 31)
        if qend >= start:
            out.append(qend)
        if qend >= end:
            break
        month += 3
        if month > 12:
            month = 3
            year += 1
    return out


QUARTER_ENDS = _quarter_ends(BACKTEST_START, BACKTEST_END)
