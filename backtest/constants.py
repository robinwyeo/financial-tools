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

# Long-horizon valuation bargain components (RSI removed).
BARGAIN_BACKTEST_COMPONENTS: tuple[str, ...] = (
    "margin_of_safety",
    "valuation_vs_history",
    "discount_52w",
)

EXCLUDED_BARGAIN_COMPONENTS: frozenset[str] = frozenset()

# Themes = factor groups (1:1 mapping; legacy Dirichlet search uses these).
FACTOR_THEMES: dict[str, list[str]] = {
    "value": ["value"],
    "garp": ["garp"],
    "quality": ["quality"],
    "balance_sheet": ["balance_sheet"],
    "momentum": ["momentum"],
    "low_volatility": ["low_volatility"],
    "capital_discipline": ["capital_discipline"],
}

WITHIN_THEME_PROPORTIONS: dict[str, dict[str, float]] = {
    "value": {"value": 1.0},
    "garp": {"garp": 1.0},
    "quality": {"quality": 1.0},
    "balance_sheet": {"balance_sheet": 1.0},
    "momentum": {"momentum": 1.0},
    "low_volatility": {"low_volatility": 1.0},
    "capital_discipline": {"capital_discipline": 1.0},
}

# Evidence-based priors for long-horizon buy-and-hold (research-backed).
EVIDENCE_BASED_FACTOR_WEIGHTS: dict[str, float] = {
    "quality": 0.25,
    "value": 0.25,
    "capital_discipline": 0.125,
    "balance_sheet": 0.10,
    "garp": 0.10,
    "momentum": 0.075,
    "low_volatility": 0.05,
    "earnings_revisions": 0.05,
}

# Previous Dirichlet-tuned weights kept as a named comparison candidate.
LEGACY_TUNED_FACTOR_WEIGHTS: dict[str, float] = {
    "value": 0.0286,
    "garp": 0.4131,
    "quality": 0.1042,
    "balance_sheet": 0.1365,
    "momentum": 0.0688,
    "low_volatility": 0.1321,
    "capital_discipline": 0.0667,
    "earnings_revisions": 0.0500,
}

EQUAL_FACTOR_WEIGHTS: dict[str, float] = {
    family: 1.0 / 8.0
    for family in (
        "value",
        "garp",
        "quality",
        "balance_sheet",
        "momentum",
        "low_volatility",
        "capital_discipline",
        "earnings_revisions",
    )
}

DEFAULT_BARGAIN_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.40,
    "valuation_vs_history": 0.35,
    "discount_52w": 0.25,
}

# Forward-return horizons in quarters (1y / 3y / 5y) plus next-quarter.
FORWARD_HORIZON_QUARTERS: dict[str, int] = {
    "1q": 1,
    "1y": 4,
    "3y": 12,
    "5y": 20,
}

PRIMARY_EVAL_HORIZON = "3y"

ROLLING_WINDOW_MONTHS = 36
TOP_QUINTILE_FRAC = 0.20
DCA_INVESTMENT_USD = 20_000.0
DCA_TOP_N = 5
DEFAULT_DELIST_RETURN = -0.50
TRANSACTION_COST_BPS = 10.0  # ~10 bps per buy
VALUATION_HISTORY_QUARTERS = 20  # 5 years of trailing EY history
BOOTSTRAP_N = 1000
BOOTSTRAP_CI = 0.95

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
