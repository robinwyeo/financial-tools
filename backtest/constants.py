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

# Factor groups reconstructable without historical analyst data (7 of the 8 groups).
BACKTEST_FACTOR_FAMILIES: tuple[str, ...] = (
    "value",
    "garp",
    "quality",
    "balance_sheet",
    "momentum",
    "low_volatility",
    "capital_discipline",
)

# earnings_revisions requires live analyst rec history; excluded from historical tuning.
EXCLUDED_COMPOSITE_FACTORS: frozenset[str] = frozenset({"earnings_revisions"})

# All three bargain components are reconstructable from EDGAR + price data.
BARGAIN_BACKTEST_COMPONENTS: tuple[str, ...] = (
    "margin_of_safety",
    "discount_52w",
    "rsi_oversold",
)

# No backtest bargain components are excluded (discount_ath and analyst_upside
# were permanently removed from the bargain score, not merely excluded for backtest).
EXCLUDED_BARGAIN_COMPONENTS: frozenset[str] = frozenset()

# Themes = factor groups (1:1 mapping; tuning samples a Dirichlet over these).
FACTOR_THEMES: dict[str, list[str]] = {
    "value": ["value"],
    "garp": ["garp"],
    "quality": ["quality"],
    "balance_sheet": ["balance_sheet"],
    "momentum": ["momentum"],
    "low_volatility": ["low_volatility"],
    "capital_discipline": ["capital_discipline"],
}

# Within-theme proportions: trivial (each theme has exactly one factor group).
WITHIN_THEME_PROPORTIONS: dict[str, dict[str, float]] = {
    "value": {"value": 1.0},
    "garp": {"garp": 1.0},
    "quality": {"quality": 1.0},
    "balance_sheet": {"balance_sheet": 1.0},
    "momentum": {"momentum": 1.0},
    "low_volatility": {"low_volatility": 1.0},
    "capital_discipline": {"capital_discipline": 1.0},
}

DEFAULT_BARGAIN_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.2489,
    "discount_52w": 0.0024,
    "rsi_oversold": 0.7488,
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
