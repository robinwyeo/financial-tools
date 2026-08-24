"""Streamlit dashboard for stock metrics and analyst aggregation."""

from __future__ import annotations

import html
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import (
    get_bargain_weights,
    get_factor_weights,
    get_fund_factor_weights,
    get_thresholds,
    load_config,
)
from core.factors import FACTOR_SCORE_COLUMNS
from core.fund_factors import FUND_FACTOR_SCORE_COLUMNS
from core.data import (
    FUND_QUOTE_TYPES,
    fetch_etf_holdings,
    fetch_price_history,
    get_security_type,
)
from core.fund_universe import load_fund_universe_snapshot
from core.scoring import (
    apply_fund_snapshot_scoring,
    apply_universe_snapshot_scoring,
    score_fund,
    score_fund_universe,
    score_ticker,
    score_universe,
)
from core.universe import load_universe_snapshot

st.set_page_config(
    page_title="Stock & Fund Metrics Tool",
    page_icon="📊",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

PRICE_HISTORY_RANGES: dict[str, str] = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "All": "max",
}
DEFAULT_PRICE_RANGE = "2Y"

# Chart heights tuned so row-2 cards align when columns are stretched to equal height.
CHART_HEIGHT_ROW2 = 198
CHART_HEIGHT_PRICE = CHART_HEIGHT_ROW2
CHART_HEIGHT_RADAR = 208
CHART_HEIGHT_ANALYST_PIE = 158
GAUGE_MAX_WIDTH = "132px"

FACTOR_LABELS = {
    "value": "Value (earnings yield · B/M · FCF)",
    "garp": "GARP (Lynch dividend-adjusted PEG)",
    "quality": "Quality / Profitability",
    "balance_sheet": "Balance Sheet Strength",
    "momentum": "Momentum (12-1)",
    "low_volatility": "Low Volatility",
    "capital_discipline": "Capital Discipline (yield + asset growth)",
    "estimate_revisions": "Estimate Revisions (agreement · magnitude · surprise)",
    "insider": "Insider Buying (net Form 4 open-market)",
}

BARGAIN_LABELS = {
    "margin_of_safety": "Margin of Safety (Graham)",
    "valuation_vs_history": "Valuation vs Own History (10y EDGAR)",
    "discount_52w": "Discount to 52-Week High",
}

SHORT_FACTOR_LABELS = {
    "value": "Value",
    "garp": "GARP",
    "quality": "Quality",
    "balance_sheet": "Balance Sheet",
    "momentum": "Momentum",
    "low_volatility": "Low Volatility",
    "capital_discipline": "Capital Discipline",
    "estimate_revisions": "Est. Revisions",
    "insider": "Insider Buying",
}

# Short labels for the radar chart (one spoke per factor group).
RADAR_FACTOR_LABELS: dict[str, str] = {
    "value":              "Value",
    "garp":               "GARP",
    "quality":            "Quality",
    "balance_sheet":      "Bal. Sheet",
    "momentum":           "Momentum",
    "low_volatility":     "Low Vol",
    "capital_discipline": "Cap. Disc.",
    "estimate_revisions": "Revisions",
    "insider":            "Insiders",
}

# Factor Scorecard display: 2 groups per column.
# Each entry: (group_label, accent_color, [factor_keys])
FACTOR_SCORECARD_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Valuation", "#14b8a6", ["value", "garp"]),
    ("Quality & Health", "#8b5cf6", ["quality", "balance_sheet"]),
    ("Capital & Insiders", "#3b82f6", ["capital_discipline", "insider"]),
    ("Market & Estimates", "#f59e0b", ["momentum", "low_volatility", "estimate_revisions"]),
]

FACTOR_COLORS = {
    "value":              "#14b8a6",
    "garp":               "#10b981",
    "quality":            "#8b5cf6",
    "balance_sheet":      "#60a5fa",
    "momentum":           "#3b82f6",
    "low_volatility":     "#f59e0b",
    "capital_discipline": "#34d399",
    "estimate_revisions": "#f97316",
    "insider":            "#0ea5e9",
}

# ── Fund (ETF / mutual fund) factor display ──────────────────────────────────

FUND_FACTOR_LABELS = {
    "cost": "Low Cost (expense ratio)",
    "performance": "Performance (3y · 5y annualized return)",
    "risk_adjusted": "Risk-Adjusted Return (return / volatility)",
    "low_volatility": "Low Volatility & Drawdown",
    "momentum": "Momentum (12-1)",
    "income": "Income (distribution yield)",
}

SHORT_FUND_FACTOR_LABELS = {
    "cost": "Low Fees",
    "performance": "Returns 3-5Y",
    "risk_adjusted": "Risk-Adj. Return",
    "low_volatility": "Low Volatility",
    "momentum": "Momentum",
    "income": "Yield",
}

RADAR_FUND_FACTOR_LABELS: dict[str, str] = {
    "cost":           "Low Fees",
    "performance":    "Returns",
    "risk_adjusted":  "Risk-Adj.",
    "low_volatility": "Low Vol",
    "momentum":       "Momentum",
    "income":         "Yield",
}

# Fund Factor Scorecard display: (group_label, accent_color, [factor_keys]).
FUND_SCORECARD_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Fees & Income", "#14b8a6", ["cost", "income"]),
    ("Performance", "#3b82f6", ["performance", "risk_adjusted"]),
    ("Risk & Trend", "#f59e0b", ["low_volatility", "momentum"]),
]

FUND_FACTOR_HELP: dict[str, str] = {
    "cost": (
        "Expense ratio ranked against the fund universe, inverted so cheaper "
        "funds score higher. Fees are the strongest documented predictor of "
        "long-run relative fund performance."
    ),
    "performance": (
        "Annualized 3-year and 5-year total returns (price/NAV based), each "
        "ranked cross-sectionally vs peer funds then averaged."
    ),
    "risk_adjusted": (
        "Sharpe-style ratio: trailing 12-month return divided by annualized "
        "volatility. Higher = more return per unit of risk taken, vs peers."
    ),
    "low_volatility": (
        "Inverse of annualized 12-month volatility plus max-drawdown "
        "protection, each ranked then averaged. Calmer funds rank higher."
    ),
    "momentum": (
        "Trailing 12-month return, skipping the most recent month to avoid "
        "short-term reversals. Higher rank = stronger persistent uptrend."
    ),
    "income": (
        "Trailing distribution/dividend yield ranked vs peer funds. Higher = "
        "more income paid out per dollar invested."
    ),
}

SECURITY_TYPE_BADGES = {
    "ETF": "ETF",
    "MUTUALFUND": "Mutual Fund",
}

METRIC_HELP = {
    "composite_score": (
        "Single number from 0–100 that blends how this stock ranks on 7 factor groups "
        "(value, GARP, quality, balance sheet, momentum, low volatility, capital "
        "discipline) vs S&P 500 peers. Each group rank-averages its own sub-signals "
        "before weighting, so no single ratio dominates. The good-buy bar is the "
        "configured composite_min. Only groups with available data are included; a Buy "
        "additionally requires Factor Coverage of at least the configured minimum."
    ),
    "price": (
        "What one share costs right now in dollars. This is market price, not a quality score—"
        "use it with the other metrics to judge whether the stock looks expensive or cheap."
    ),
    "analyst_upside": (
        "How far the average Wall Street 12‑month price target sits above or below today's price, "
        "in percent. Positive = analysts on average expect a higher price; negative = targets are "
        "below the current price."
    ),
    "consensus": (
        "The overall label analysts give this stock (e.g. Buy, Hold, Sell), averaged from their "
        "published recommendations. It summarizes professional opinion, not a guarantee of "
        "future performance."
    ),
    "trailing_pe": (
        "Price divided by earnings per share over the last 12 months—the classic P/E multiple. "
        "Lower usually means a cheaper price tag per dollar of past earnings. Shown for context "
        "only: it is not part of the composite score because it varies by sector, is meaningless "
        "when earnings are negative, and ignores growth and balance-sheet quality. See GARP, "
        "the Value group (Graham ratio), and Quality for richer valuation and profitability context."
    ),
    "lynch_peg": (
        "Compares price to expected earnings growth and dividends (Peter Lynch’s “PEG” idea). "
        "Higher values here suggest you may be paying less per unit of growth; Lynch often liked "
        "values above 2. Below 1 can mean growth looks expensive relative to price."
    ),
    "graham_ratio": (
        "Compares Benjamin Graham's estimated fair price to today's price. Above 1.0 means the "
        "stock trades below that estimate (more “margin of safety”); below 1.0 means it trades "
        "above it."
    ),
    "target_low": (
        "The lowest 12‑month price target among analysts covering the stock. The market could "
        "fall toward this level if those bearish views prove right."
    ),
    "target_mean": (
        "The average 12‑month price target across analysts. “Analyst upside” compares this number "
        "to the current share price."
    ),
    "target_high": (
        "The highest 12‑month price target among analysts—a bullish ceiling some professionals "
        "see if things go well."
    ),
    "upgrades_downgrades": (
        "How many analysts recently raised their rating (upgrades) vs lowered it (downgrades). "
        "More upgrades often means improving sentiment; more downgrades the opposite."
    ),
    "etf_price": "Latest market price for one ETF share in dollars.",
    "etf_expense_ratio": (
        "Annual fund fee as a percent of assets. Lower usually means more of the return stays "
        "with you instead of going to the fund manager."
    ),
    "etf_category": "Broad type of fund (e.g. large-cap equity, bond) from the provider's classification.",
    "etf_yield": "Income paid out by the fund, shown as an annual percent of price (dividends/distributions).",
    "fund_composite": (
        "Single number from 0–100 that blends how this fund ranks on 6 fund factor groups "
        "(low fees, 3-5y performance, risk-adjusted return, low volatility, momentum, income) "
        "vs a peer universe of well-known US and Canadian ETFs and mutual funds. Sub-signals "
        "are ranked within fund category when the category is large enough. Stock metrics "
        "like Graham value or balance-sheet strength don't exist for funds and are excluded."
    ),
    "fund_nav_premium": (
        "How far the market price sits above (+) or below (−) the fund's net asset value. "
        "Persistent premiums mean paying more than the underlying holdings are worth."
    ),
    "fund_aum": (
        "Total net assets managed by the fund. Larger funds tend to be more liquid and less "
        "likely to close, but size itself is not a performance signal."
    ),
    "fund_returns": (
        "Annualized total returns computed from price/NAV history (1Y is a simple trailing "
        "return; 3Y and 5Y are CAGRs). Shown in the fund's own trading currency."
    ),
}

# Hover copy for Factor Scorecard: what the metric means and how it's built.
FACTOR_HELP: dict[str, str] = {
    "value": (
        "Composite value rank: each of three sub-signals (earnings yield EBIT/EV, "
        "FCF yield, book-to-market) is ranked cross-sectionally then averaged. "
        "Higher = cheaper vs peers on multiple measures. The Graham ratio is "
        "tracked separately in the bargain score to avoid double-counting."
    ),
    "garp": (
        "Growth at a reasonable price (Peter Lynch): (earnings growth % + dividend yield %) "
        "divided by P/E. Higher = more growth and income per dollar of valuation. "
        "Falls back to 1/PEG when analyst growth estimates are unavailable."
    ),
    "quality": (
        "Composite quality rank in three de-correlated buckets: profitability "
        "(gross profitability, ROE, ROA, margin, ROIC), earnings quality (accruals), "
        "and financial strength (Piotroski). Higher = more profitable and cleaner business."
    ),
    "estimate_revisions": (
        "Zacks-style estimate-revision rank: revision agreement (% of FY1/FY2 "
        "upgrades), revision magnitude (90-day consensus EPS change), and the last "
        "quarterly earnings surprise. Higher = analysts are revising earnings up."
    ),
    "insider": (
        "Net open-market Form 4 buying (purchases minus sales) over the last 90 days "
        "as a fraction of market cap, ranked vs peers. Cluster buys by multiple "
        "officers also show as a badge. Grants and option exercises are excluded."
    ),
    "balance_sheet": (
        "Composite balance-sheet rank: net cash / market cap, low debt-to-equity "
        "(1/(1+D/E)), and Altman Z-Score are each ranked then averaged. "
        "Higher = more financial cushion and lower distress risk."
    ),
    "momentum": (
        "Trailing 12-month price return, skipping the most recent month to avoid "
        "short-term reversals. A higher rank means a stronger, more persistent uptrend."
    ),
    "low_volatility": (
        "Inverse of annualized 12-month return volatility (1/σ). "
        "Calmer, steadier stocks rank higher; high-swing names rank lower."
    ),
    "capital_discipline": (
        "Composite capital-discipline rank: shareholder yield (dividends + buybacks "
        "/ market cap) and investment factor (inverted asset growth) are each ranked "
        "then averaged. Higher = more cash returned, less balance-sheet expansion."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][min(n % 10, 3)]}"


def currency_symbol(currency: str | None) -> str:
    """Display symbol for a trading currency (C$ distinguishes CAD from USD)."""
    cur = (currency or "USD").upper()
    if cur == "CAD":
        return "C$"
    if cur == "USD":
        return "$"
    return f"{cur} "


def fmt_large_number(n: float | None, currency: str | None = None) -> str:
    sym = currency_symbol(currency)
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{sym}{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{sym}{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{sym}{n / 1e6:.1f}M"
    return f"{sym}{n:,.0f}"


def percentile_color(pct: float | None) -> str:
    """Red (0–30), yellow (31–70), green (71–100) by percentile rank."""
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "#d1d5db"
    p = max(0.0, min(100.0, float(pct)))
    if p <= 30:
        return "#ef4444"  # red
    if p <= 70:
        return "#eab308"  # yellow
    return "#22c55e"  # green


def gauge_score_color(score: float | None) -> str:
    """Red ≤25, yellow 25–50, green ≥50 (aligned with scorecard traffic lights)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "#d1d5db"
    s = max(0.0, min(100.0, float(score)))
    if s >= 50:
        return "#22c55e"
    if s <= 25:
        return "#ef4444"
    return "#eab308"


def gauge_score_label(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "N/A"
    s = float(score)
    if s >= 50:
        return "Good"
    if s <= 25:
        return "Weak"
    return "Fair"


def _proximity_color(pct_below: float | None) -> str:
    """Green = discounted, yellow = moderate, red = near highs."""
    if pct_below is None:
        return "#d1d5db"
    if pct_below <= 0.10:
        return "#ef4444"
    if pct_below <= 0.25:
        return "#eab308"
    return "#22c55e"


def _pct_below_high(price: float | None, high: float | None) -> float | None:
    if price is None or high is None or high <= 0 or price <= 0:
        return None
    return max(0.0, 1.0 - (price / high))


def _plotly_chart(fig: go.Figure, *, height: int, extra_css: str = "") -> None:
    """Render a Plotly figure without Streamlit's PlotlyChart JS chunk.

    Streamlit Community Cloud sometimes serves the SPA index.html for
    ``/static/js/PlotlyChart.*.js``, which makes ``st.plotly_chart`` fail with
    ``Failed to fetch dynamically imported module``. Embedding via components
    loads Plotly from its CDN inside an iframe and avoids that path.
    """
    fig.update_layout(autosize=True, height=height)
    style = f"<style>{extra_css}</style>" if extra_css else ""
    chart_html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=False,
        config={"displayModeBar": False, "responsive": True, "scrollZoom": False},
    )
    components.html(
        f'<div style="width:100%;height:{height}px;">{style}{chart_html}</div>',
        height=height,
        scrolling=False,
    )


def consensus_style(label: str) -> tuple[str, str]:
    """Return (text_color, bg_color) for a consensus label."""
    lower = label.lower()
    if "buy" in lower:
        return "#0d9488", "#ccfbf1"
    if "hold" in lower or "neutral" in lower:
        return "#92400e", "#fef3c7"
    if "sell" in lower or "underperform" in lower:
        return "#dc2626", "#fee2e2"
    return "#6b7280", "#f3f4f6"


# ──────────────────────────────────────────────────────────────────────────────
# Global CSS
# ──────────────────────────────────────────────────────────────────────────────

def _dashboard_row_anchor(row: int) -> None:
    """Marker for JS equal-height pass on the following st.columns() row."""
    st.markdown(
        f'<div id="stock-row-{row}-anchor" class="stock-dashboard-row-anchor" '
        f'aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )


@contextmanager
def _card_shell(bordered: bool) -> Iterator[None]:
    """Dashboard cards use column borders; standalone cards keep st.container(border=True)."""
    if bordered:
        with st.container(border=True):
            yield
    else:
        yield


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Page background */
        [data-testid="stAppViewContainer"] { background-color: #f0f4f8; }
        [data-testid="stHeader"] { background-color: #f0f4f8; }
        .main .block-container {
            padding-top: 0.5rem;
            padding-bottom: 1rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 100%;
        }

        /* Cards (st.container with border=True) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.07) !important;
            background: white !important;
            overflow: hidden !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: white !important;
        }

        .dashboard-card-body {
            display: flex;
            flex-direction: column;
            flex: 1 1 auto;
            min-height: 100%;
        }
        .composite-score-card {
            justify-content: flex-start;
        }
        .composite-gauges-row {
            display: flex;
            gap: 0.45rem;
            align-items: flex-start;
            justify-content: center;
            width: 100%;
            padding: 0.15rem 0 0.1rem;
        }
        .composite-gauges-row .gauge-cell {
            flex: 1 1 0;
            min-width: 0;
        }
        .composite-gauges-row .gauge-title {
            font-size: 0.72rem;
            font-weight: 600;
            color: #6b7280;
            text-align: center;
            margin-bottom: 0.15rem;
            line-height: 1.2;
        }
        .factor-scorecard-card .factor-scorecard-grid {
            align-content: start;
        }
        .factor-scorecard-card .factor-scorecard-col {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }
        .analyst-consensus-card {
            justify-content: center;
            gap: 0.25rem;
        }
        .analyst-consensus-card .analyst-header-wrap {
            flex: 0 0 auto;
        }
        .analyst-consensus-card .analyst-targets {
            display: flex;
            gap: 0.35rem;
            margin-top: 0.3rem;
        }
        .analyst-consensus-card .analyst-target-pill {
            flex: 1;
            min-width: 0;
            text-align: center;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.4rem 0.35rem;
        }
        .analyst-consensus-card .analyst-target-pill .lbl {
            font-size: 0.55rem;
            font-weight: 700;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .analyst-consensus-card .analyst-target-pill .val {
            font-size: 0.95rem;
            font-weight: 800;
            color: #1e3a5f;
            margin-top: 0.15rem;
        }
        .analyst-consensus-card .analyst-chart-slot {
            flex: 0 0 auto;
            min-height: unset;
            margin: 0.2rem 0 0.05rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .factor-radar-card {
            justify-content: center;
            gap: 0.2rem;
        }
        .price-history-card {
            padding-bottom: 0.45rem;
        }
        .price-history-card .price-position-strip {
            margin-top: 0.4rem;
            padding: 0.45rem 0 0.55rem;
            border-top: 1px solid #e5e7eb;
        }
        /* Google Finance-style timeframe tabs (widget key prefix ph-range-). */
        div[class*="st-key-ph-range"] {
            width: 100% !important;
            margin-top: -0.15rem;
            margin-bottom: 0.1rem;
        }
        div[class*="st-key-ph-range"] [data-testid="stWidgetLabel"] {
            display: none !important;
        }
        div[class*="st-key-ph-range"] [data-testid="stRadioGroup"],
        div[class*="st-key-ph-range"] div[role="radiogroup"] {
            gap: 0 !important;
            flex-wrap: nowrap !important;
            justify-content: space-between !important;
            width: 100%;
            border-bottom: 1px solid #e8eaed;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"] {
            flex: 1 1 0;
            justify-content: center !important;
            padding: 0.12rem 0.2rem 0.28rem !important;
            margin: 0 !important;
            border-right: 1px solid #e8eaed;
            border-bottom: 3px solid transparent;
            border-radius: 0 !important;
            background: transparent !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:last-child {
            border-right: none;
        }
        /* Hide the Streamlit radio circle (nested inside the option, after the visually-hidden input). */
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"] > div > div > div:first-child {
            display: none !important;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"] p {
            font-size: 0.74rem !important;
            font-weight: 500 !important;
            color: #5f6368 !important;
            letter-spacing: 0.01em;
            text-align: center;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:hover p {
            color: #202124 !important;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:has(input:checked),
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"][data-selected="true"] {
            border-bottom-color: #1a73e8;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:has(input:checked) p,
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"][data-selected="true"] p {
            color: #1a73e8 !important;
            font-weight: 600 !important;
        }
        .factor-radar-card .dashboard-chart-slot {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.15rem 0;
            margin: auto 0;
        }
        .dashboard-chart-slot {
            flex: 1 1 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 0;
        }

        /* Factor scorecard rows: flexible percentile bars (wider on large viewports) */
        .factor-scorecard-card {
            overflow: visible !important;
        }
        .factor-scorecard-grid .factor-row {
            display: grid;
            grid-template-columns: 5px minmax(4.8em, 1.05fr) minmax(56px, 2.85fr) 26px;
            column-gap: 6px;
            align-items: center;
            margin: 2px 0;
        }
        .factor-scorecard-grid .factor-dot {
            grid-column: 1;
            width: 5px;
            height: 5px;
            border-radius: 50%;
        }
        .factor-scorecard-grid .factor-label {
            grid-column: 2;
            font-size: 0.64rem;
            color: #374151;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.15;
            min-width: 0;
        }
        .factor-scorecard-grid .factor-label.factor-has-tip {
            position: relative;
            cursor: help;
            overflow: hidden;
            z-index: 1;
        }
        .factor-scorecard-grid .factor-label.factor-has-tip:hover,
        .factor-scorecard-grid .factor-label.factor-has-tip:focus-within {
            z-index: 200;
            overflow: visible;
        }
        .factor-scorecard-grid .factor-label-text {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            border-bottom: 1px dotted #9ca3af;
        }
        .factor-scorecard-grid .factor-tooltip {
            visibility: hidden;
            opacity: 0;
            pointer-events: none;
            position: absolute;
            left: 0;
            top: calc(100% + 5px);
            width: 11.5rem;
            max-width: min(11.5rem, 70vw);
            padding: 0.4rem 0.5rem;
            background: #1e293b;
            color: #f8fafc;
            font-size: 0.6rem;
            font-weight: 400;
            line-height: 1.3;
            border-radius: 6px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.28);
            z-index: 201;
            text-transform: none;
            letter-spacing: normal;
            white-space: normal;
            transition: opacity 0.12s ease, visibility 0.12s ease;
        }
        .factor-scorecard-grid .factor-tooltip::before {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 10px;
            border: 5px solid transparent;
            border-bottom-color: #1e293b;
        }
        .factor-scorecard-grid .factor-has-tip:hover .factor-tooltip,
        .factor-scorecard-grid .factor-has-tip:focus-within .factor-tooltip {
            visibility: visible;
            opacity: 1;
        }
        .factor-scorecard-grid .factor-bar-track {
            grid-column: 3;
            background: #f3f4f6;
            border-radius: 3px;
            height: 5px;
            overflow: hidden;
            min-width: 0;
        }
        .factor-scorecard-grid .factor-bar-fill {
            height: 5px;
            border-radius: 3px;
        }
        .factor-scorecard-grid .factor-pct {
            grid-column: 4;
            font-size: 0.62rem;
            font-weight: 700;
            text-align: right;
        }
        @media (min-width: 860px) {
            .factor-scorecard-grid .factor-row {
                grid-template-columns: 5px minmax(5em, 0.95fr) minmax(68px, 3.4fr) 26px;
                column-gap: 6px;
            }
        }
        @media (min-width: 1100px) {
            .factor-scorecard-grid .factor-row {
                grid-template-columns: 5px minmax(5.2em, 0.78fr) minmax(110px, 5.5fr) 28px;
                column-gap: 7px;
            }
        }
        @media (min-width: 1400px) {
            .factor-scorecard-grid .factor-row {
                grid-template-columns: 5px minmax(5.5em, 0.62fr) minmax(160px, 7fr) 28px;
                column-gap: 8px;
            }
        }

        /* Columns: prevent overflow in narrow slots */
        [data-testid="stColumn"] { min-width: 0; }

        /* Sidebar */
        [data-testid="stSidebar"] { background-color: white; }

        /* Hide footer */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stock-dashboard-row-anchor {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_equal_height_js() -> None:
    """Equalize dashboard row column heights in the main document (markdown strips <script>)."""
    js = """
(function () {
  const doc = window.parent && window.parent.document ? window.parent.document : document;

  function findRowForAnchor(anchorId) {
    const anchor = doc.getElementById(anchorId);
    if (!anchor) return null;
    let box = anchor.closest(".element-container") || anchor.parentElement;
    while (box) {
      const sibling = box.nextElementSibling;
      if (!sibling) break;
      const row = sibling.querySelector('[data-testid="stHorizontalBlock"]');
      if (row) return row;
      box = sibling;
    }
    return null;
  }

  function columnShell(col) {
    return (
      col.querySelector('[data-testid="stVerticalBlockBorderWrapper"]') ||
      col.querySelector(':scope > div > [data-testid="stVerticalBlock"]') ||
      col
    );
  }

  function equalizeRow(anchorId) {
    const row = findRowForAnchor(anchorId);
    if (!row) return;
    const cols = row.querySelectorAll('[data-testid="stColumn"]');
    if (cols.length < 2) return;

    const shells = Array.from(cols).map(columnShell);
    shells.forEach((el) => {
      el.style.minHeight = "";
    });
    cols.forEach((c) => {
      c.style.minHeight = "";
    });

    let maxH = 0;
    shells.forEach((el) => {
      maxH = Math.max(maxH, el.getBoundingClientRect().height);
    });
    if (maxH < 1) return;

    if (anchorId === "stock-row-2-anchor") {
      maxH = Math.min(maxH, 340);
    }

    const px = Math.ceil(maxH) + "px";
    cols.forEach((c) => {
      c.style.minHeight = px;
    });
    shells.forEach((el) => {
      el.style.minHeight = px;
    });
  }

  function run() {
    equalizeRow("stock-row-1-anchor");
    equalizeRow("stock-row-2-anchor");
  }

  const schedule = () => requestAnimationFrame(() => requestAnimationFrame(run));
  if (!doc.defaultView.__stockRowEqualize) {
    const win = doc.defaultView;
    win.__stockRowEqualize = schedule;
    win.addEventListener("resize", schedule);
    const root = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
    new MutationObserver(schedule).observe(root, { childList: true, subtree: true });
  }
  doc.defaultView.__stockRowEqualize();
})();
"""
    html_fn = getattr(st, "html", None)
    if html_fn is not None:
        try:
            html_fn(f"<script>{js}</script>", unsafe_allow_javascript=True)
            return
        except TypeError:
            pass
    components.html(f"<script>{js}</script>", height=0)


# ──────────────────────────────────────────────────────────────────────────────
# Card components
# ──────────────────────────────────────────────────────────────────────────────

def render_company_header(analysis: dict) -> None:
    ticker = analysis.get("ticker", "")
    name = analysis.get("name") or ticker
    exchange = analysis.get("exchange") or ""
    sector = analysis.get("sector") or ""
    industry = analysis.get("industry") or ""
    if sector.strip().lower() == "unknown":
        sector = ""
    if industry.strip().lower() == "unknown":
        industry = ""
    market_cap = analysis.get("market_cap")
    price = analysis.get("price")

    ticker_e = html.escape(str(ticker))
    name_e = html.escape(str(name))
    exchange_e = html.escape(str(exchange)) if exchange else ""
    sector_e = html.escape(str(sector)) if sector else ""
    industry_e = html.escape(str(industry)) if industry else ""

    price_html = (
        f' <span style="font-size:1.25rem;font-weight:700;color:#1e3a5f;white-space:nowrap;">'
        f"${price:,.2f}</span>"
        if price
        else ""
    )
    exchange_html = (
        f' <span style="color:#d1d5db;">|</span> '
        f'<span style="font-size:0.88rem;color:#9ca3af;">{exchange_e}</span>'
        if exchange_e
        else ""
    )

    left, right = st.columns([3, 2])
    with left:
        # Single-level markup: Streamlit strips nested <div>s and can leak closing tags as text.
        st.markdown(
            f'<div style="padding:0.05rem 0 0.1rem;line-height:1.35;">'
            f'<span style="font-size:1.55rem;font-weight:800;color:#1e3a5f;">{ticker_e}</span>'
            f"{price_html}<br>"
            f'<span style="font-size:0.82rem;color:#6b7280;">{name_e}</span>'
            f"{exchange_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        parts = []
        if sector_e:
            parts.append(
                f'<span style="display:inline-block;margin-left:1.25rem;">'
                f'<span style="display:block;font-size:0.68rem;color:#9ca3af;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;">Sector</span>'
                f'<span style="display:block;font-size:0.88rem;color:#374151;font-weight:500;">'
                f"{sector_e}</span></span>"
            )
        if industry_e:
            parts.append(
                f'<span style="display:inline-block;margin-left:1.25rem;">'
                f'<span style="display:block;font-size:0.68rem;color:#9ca3af;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;">Industry</span>'
                f'<span style="display:block;font-size:0.88rem;color:#374151;font-weight:500;">'
                f"{industry_e}</span></span>"
            )
        if market_cap:
            parts.append(
                f'<span style="display:inline-block;margin-left:1.25rem;">'
                f'<span style="display:block;font-size:0.68rem;color:#9ca3af;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;">Market Cap</span>'
                f'<span style="display:block;font-size:0.88rem;color:#374151;font-weight:700;">'
                f"{html.escape(fmt_large_number(market_cap))}</span></span>"
            )
        if parts:
            st.markdown(
                f'<div style="text-align:right;padding:0.15rem 0 0.25rem;">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )


def _arc_gauge_html(
    score: float | None,
    label: str,
    label_color: str,
    *,
    subtitle: str = "vs. Global Universe",
    aria_label: str = "Score gauge",
    fill_color: str | None = None,
    max_width: str = "130px",
) -> str:
    """
    Pure SVG 3/4-circle arc gauge.
    Starts at the 7:30 clock position, sweeps 270° clockwise to 4:30.
    Uses width="100%" so it scales from a narrow sidebar column to full mobile width.
    """
    cx, cy, r, sw = 80, 80, 60, 12
    start_deg = 135.0   # 7:30 clock position in SVG angle space
    span_deg = 270.0

    def pt(deg: float) -> tuple[float, float]:
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    # Background arc (full 270°)
    s = pt(start_deg)
    e = pt(start_deg + span_deg)
    bg_path = f"M {s[0]:.1f} {s[1]:.1f} A {r} {r} 0 1 1 {e[0]:.1f} {e[1]:.1f}"

    # Fill arc proportional to score
    fill_svg = ""
    stroke = fill_color or "#14b8a6"
    if score is not None and score > 0:
        fd = score * span_deg / 100.0
        fe = pt(start_deg + fd)
        large = 1 if fd > 180 else 0
        fill_svg = (
            f'<path d="M {s[0]:.1f} {s[1]:.1f} A {r} {r} 0 {large} 1 {fe[0]:.1f} {fe[1]:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"/>'
        )

    score_txt = f"{score:.1f}" if score is not None else "N/A"
    compact = max_width != "130px"
    font_sz = 34 if score is not None and compact else (36 if score is not None else 22)

    # viewBox clips the empty gap at the bottom (arc endpoints sit at y≈122, cut at y=135)
    return (
        '<div style="text-align:center;padding:0.2rem 0.15rem 0.1rem;">'
        '<svg width="100%" viewBox="5 5 150 130" '
        f'style="max-width:{max_width};display:block;margin:0 auto;" '
        f'aria-label="{aria_label}">'
        f'<path d="{bg_path}" fill="none" stroke="#e8ecef" '
        f'stroke-width="{sw}" stroke-linecap="round"/>'
        f'{fill_svg}'
        f'<text x="{cx}" y="78" text-anchor="middle" dominant-baseline="middle" '
        f'font-size="{font_sz}" font-weight="800" fill="#1e3a5f" '
        f'font-family="Inter, Arial, sans-serif">{score_txt}</text>'
        f'<text x="{cx}" y="100" text-anchor="middle" font-size="13" fill="#9ca3af" '
        f'font-family="Inter, Arial, sans-serif">/ 100</text>'
        '</svg>'
        f'<div style="margin-top:0.1rem;">'
        f'<div style="font-size:0.88rem;font-weight:700;color:{label_color};">{label}</div>'
        f'<div style="font-size:0.6rem;color:#9ca3af;margin-top:1px;line-height:1.25;">{subtitle}</div>'
        '</div>'
        '</div>'
    )


def _format_snapshot_date(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    return raw_date[:10] if len(raw_date) >= 10 else raw_date


def _sparkline_svg(values: list[float], *, width: int = 128, height: int = 28) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    pts: list[str] = []
    for i, val in enumerate(values):
        x = 1 + i / (len(values) - 1) * (width - 2)
        y = height - 2 - ((val - lo) / span) * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#16a34a" if values[-1] >= values[0] else "#dc2626"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'aria-label="FY1 EPS estimate trend">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.8" '
        f'points="{" ".join(pts)}"/></svg>'
    )


def _signal_badge(label: str, tone: str) -> str:
    colors = {
        "success": ("#166534", "#dcfce7"),
        "warning": ("#92400e", "#fef3c7"),
        "danger": ("#991b1b", "#fee2e2"),
        "info": ("#1e3a5f", "#e0e7ff"),
        "neutral": ("#374151", "#f3f4f6"),
    }
    fg, bg = colors.get(tone, colors["neutral"])
    return (
        f'<span style="display:inline-block;margin:0 6px 6px 0;padding:2px 8px;'
        f'border-radius:999px;font-size:0.72rem;font-weight:700;'
        f'color:{fg};background:{bg};">{html.escape(label)}</span>'
    )


def render_composite_card(
    analysis: dict,
    *,
    bordered: bool = True,
    snapshot_date: str | None = None,
) -> None:
    composite = analysis.get("composite")
    comp_color = gauge_score_color(composite)
    comp_label = gauge_score_label(composite)
    bargain = analysis.get("bargain") or {}
    bargain_score = bargain.get("score")
    bargain_color = gauge_score_color(bargain_score)
    bargain_label = gauge_score_label(bargain_score)

    date_label = _format_snapshot_date(snapshot_date)
    composite_subtitle = (
        f"Composite scored vs. universe snapshot from {date_label}"
        if date_label
        else "vs. Global Universe"
    )

    composite_gauge = _arc_gauge_html(
        composite,
        comp_label,
        comp_color,
        subtitle=composite_subtitle,
        aria_label="Composite score gauge",
        fill_color=comp_color,
        max_width=GAUGE_MAX_WIDTH,
    )
    bargain_gauge = _arc_gauge_html(
        bargain_score,
        bargain_label,
        bargain_color,
        subtitle="Graham · vs own history · 52W discount",
        aria_label="Bargain score gauge",
        fill_color=bargain_color,
        max_width=GAUGE_MAX_WIDTH,
    )

    rsi = analysis.get("rsi_14")
    rsi_note = (
        f'<div style="text-align:center;color:#6b7280;font-size:0.8rem;margin-top:0.25rem;">'
        f"RSI(14): {rsi:.0f} (timing only — not in bargain score)</div>"
        if rsi is not None
        else ""
    )

    uncertainty = analysis.get("uncertainty") or {}
    unc_label = uncertainty.get("label")
    unc_tone = {"Low": "success", "Medium": "warning", "High": "danger"}.get(unc_label, "neutral")
    badges = []
    if unc_label:
        bump = uncertainty.get("threshold_bump") or 0
        bump_txt = f" · +{bump:.0f} buy hurdle" if bump else ""
        badges.append(_signal_badge(f"Uncertainty: {unc_label}{bump_txt}", unc_tone))
    insider = analysis.get("insider") or {}
    if insider.get("cluster_buy"):
        n_buyers = insider.get("buyers_90d") or 0
        badges.append(_signal_badge(f"Insider cluster buy ({n_buyers} officers, 90d)", "success"))
    short = analysis.get("short_interest") or {}
    if short.get("high_short_interest"):
        days = short.get("short_ratio")
        days_txt = f"{days:.1f}d to cover" if days is not None else "elevated"
        badges.append(_signal_badge(f"High short interest · {days_txt}", "warning"))
    hist = analysis.get("valuation_history") or {}
    if hist.get("source") and hist.get("n_points"):
        metric = hist.get("metric") or "yield"
        badges.append(
            _signal_badge(
                f"History: {hist.get('n_points')} pts · {metric} · {hist.get('source')}",
                "info",
            )
        )
    badge_html = (
        f'<div style="text-align:center;margin-top:0.45rem;">{"".join(badges)}</div>'
        if badges
        else ""
    )

    spark = analysis.get("eps_trend_sparkline") or []
    spark_html = ""
    if len(spark) >= 2:
        spark_html = (
            '<div style="text-align:center;margin-top:0.35rem;">'
            '<div style="font-size:0.68rem;color:#6b7280;letter-spacing:0.04em;'
            'text-transform:uppercase;">FY1 consensus EPS (90d → now)</div>'
            + _sparkline_svg([float(v) for v in spark])
            + "</div>"
        )

    with _card_shell(bordered):
        st.markdown(
            '<div class="dashboard-card-body composite-score-card">'
            '<div class="composite-gauges-row">'
            '<div class="gauge-cell">'
            '<div class="gauge-title">Composite Score</div>'
            + composite_gauge
            + '</div>'
            '<div class="gauge-cell">'
            '<div class="gauge-title">Bargain Score</div>'
            + bargain_gauge
            + rsi_note
            + "</div></div>"
            + badge_html
            + spark_html
            + "</div>",
            unsafe_allow_html=True,
        )


def _factor_label_html(
    factor_key: str,
    short_label: str,
    help_texts: dict[str, str] | None = None,
) -> str:
    """Metric name with hover tooltip (meaning + calculation)."""
    label_e = html.escape(short_label)
    help_text = (help_texts if help_texts is not None else FACTOR_HELP).get(factor_key)
    if not help_text:
        return f'<div class="factor-label">{label_e}</div>'

    return (
        f'<div class="factor-label factor-has-tip" tabindex="0">'
        f'<span class="factor-label-text">{label_e}</span>'
        f'<span class="factor-tooltip" role="tooltip">{html.escape(help_text)}</span>'
        f"</div>"
    )


def _factor_group_html(
    group_label: str,
    accent: str,
    factor_keys: list[str],
    breakdown: dict,
    labels: dict[str, str] | None = None,
    help_texts: dict[str, str] | None = None,
) -> str:
    header = (
        f'<div style="font-size:0.58rem;font-weight:700;color:{accent};text-transform:uppercase;'
        f'letter-spacing:0.07em;margin:0 0 4px;padding-bottom:2px;'
        f'border-bottom:1px solid {accent}22;'
        f'overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{group_label}</div>'
    )
    label_map = labels if labels is not None else SHORT_FACTOR_LABELS
    rows = []
    for key in factor_keys:
        short_label = label_map.get(key, key)
        fb = breakdown.get(key, {})
        pct = fb.get("percentile")
        color = percentile_color(pct)

        if pct is None or (isinstance(pct, float) and math.isnan(pct)):
            bar_w, pct_text = 0, "N/A"
        else:
            bar_w = min(max(float(pct), 0), 100)
            pct_text = ordinal(int(round(float(pct))))

        rows.append(
            f'<div class="factor-row">'
            f'<div class="factor-dot" style="background:{color};"></div>'
            f"{_factor_label_html(key, short_label, help_texts)}"
            f'<div class="factor-bar-track">'
            f'<div class="factor-bar-fill" style="width:{bar_w:.0f}%;background:{color};"></div>'
            f"</div>"
            f'<div class="factor-pct" style="color:{color};">{pct_text}</div>'
            f"</div>"
        )
    return f'<div class="factor-scorecard-group">{header}{"".join(rows)}</div>'


def render_factor_scorecard_card(
    analysis: dict,
    *,
    bordered: bool = True,
    groups: list[tuple[str, str, list[str]]] | None = None,
    labels: dict[str, str] | None = None,
    help_texts: dict[str, str] | None = None,
) -> None:
    breakdown = analysis.get("factor_breakdown", {})
    group_list = groups if groups is not None else FACTOR_SCORECARD_GROUPS
    split = (len(group_list) + 1) // 2

    with _card_shell(bordered):
        # 2-column CSS grid of factor groups.
        left_html = "".join(
            _factor_group_html(lbl, acc, keys, breakdown, labels, help_texts)
            for lbl, acc, keys in group_list[:split]
        )
        right_html = "".join(
            _factor_group_html(lbl, acc, keys, breakdown, labels, help_texts)
            for lbl, acc, keys in group_list[split:]
        )
        st.markdown(
            '<div class="dashboard-card-body factor-scorecard-card">'
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'margin-bottom:0.35rem;">'
            '<span style="font-size:0.88rem;font-weight:700;color:#1e3a5f;">Factor Scorecard</span>'
            '<span style="font-size:0.58rem;font-weight:600;color:#9ca3af;'
            'text-transform:uppercase;letter-spacing:0.05em;">Percentile Rank</span>'
            '</div>'
            '<div class="factor-scorecard-grid" '
            'style="display:grid;grid-template-columns:1fr 1fr;column-gap:14px;flex:1;">'
            f'<div class="factor-scorecard-col">{left_html}</div>'
            f'<div class="factor-scorecard-col">{right_html}</div>'
            "</div></div>",
            unsafe_allow_html=True,
        )


def _price_position_strip_html(analysis: dict) -> str:
    """Colored pills: % below 52W high, % below ATH, optional 52wk range bar."""
    price = analysis.get("price")
    high_52 = analysis.get("fifty_two_week_high")
    low_52 = analysis.get("fifty_two_week_low")
    ath = analysis.get("all_time_high")

    def pill(label: str, pct_below: float | None) -> str:
        color = _proximity_color(pct_below)
        if pct_below is None:
            txt = "N/A"
        else:
            txt = f"{pct_below * 100:.1f}% below"
        return (
            f'<div style="flex:1;min-width:0;background:{color}18;border:1px solid {color}55;'
            f'border-radius:8px;padding:0.35rem 0.5rem;text-align:center;">'
            f'<div style="font-size:0.55rem;font-weight:600;color:#6b7280;text-transform:uppercase;'
            f'letter-spacing:0.04em;">{label}</div>'
            f'<div style="font-size:0.82rem;font-weight:700;color:{color};">{txt}</div>'
            "</div>"
        )

    range_bar = ""
    if (
        price is not None
        and low_52 is not None
        and high_52 is not None
        and high_52 > low_52
    ):
        pos = max(0.0, min(1.0, (price - low_52) / (high_52 - low_52)))
        range_bar = (
            '<div style="margin-top:0.45rem;padding-bottom:0.15rem;">'
            '<div style="font-size:0.55rem;color:#9ca3af;margin-bottom:4px;">52W range</div>'
            '<div style="position:relative;height:7px;background:linear-gradient(90deg,#22c55e,#eab308,#ef4444);'
            'border-radius:3px;">'
            f'<div style="position:absolute;left:{pos * 100:.1f}%;top:50%;transform:translate(-50%,-50%);'
            'width:10px;height:10px;background:#1e3a5f;border:2px solid #fff;border-radius:50%;'
            'box-shadow:0 0 0 1px #94a3b8;"></div></div></div>'
        )

    return (
        '<div class="price-position-strip">'
        '<div style="display:flex;gap:0.4rem;">'
        + pill("vs 52W High", _pct_below_high(price, high_52))
        + pill("vs ATH", _pct_below_high(price, ath))
        + "</div>"
        + range_bar
        + "</div>"
    )


_PRICE_UP = "#188038"
_PRICE_DOWN = "#d93025"
_PRICE_CHART_HOVER_CSS = """
.js-plotly-plot .hoverlayer .hovertext {
    filter: drop-shadow(0 1px 4px rgba(60, 64, 67, 0.18));
}
"""


def _price_hover_label(ts, price: float, currency_code: str) -> str:
    """Google Finance-style hover text, e.g. ``183.91 USD Thu, Apr 9``."""
    t = pd.Timestamp(ts)
    return f"{price:.2f} {currency_code} {t.strftime('%a, %b')} {t.day}"


def _price_history_figure(
    hist: pd.DataFrame,
    *,
    currency_code: str,
    range_label: str,
) -> go.Figure:
    """Area chart styled after Google Finance price history."""
    closes = hist["Close"].astype(float)
    x = hist.index
    first = float(closes.iloc[0])
    last = float(closes.iloc[-1])
    up = last >= first
    line_color = _PRICE_UP if up else _PRICE_DOWN
    fill_top = "rgba(24, 128, 56, 0.20)" if up else "rgba(217, 48, 37, 0.20)"
    fill_bottom = "rgba(24, 128, 56, 0.0)" if up else "rgba(217, 48, 37, 0.0)"

    y_min = float(closes.min())
    y_max = float(closes.max())
    pad = (y_max - y_min) * 0.12 if y_max > y_min else max(abs(y_max) * 0.05, 1.0)
    axis_min = y_min - pad
    axis_max = y_max + pad

    if range_label in {"1M", "3M"}:
        tickformat = "%b %d"
    elif range_label in {"5Y", "All"}:
        tickformat = "%Y"
    else:
        tickformat = "%b %Y"

    hover_labels = [
        _price_hover_label(ts, float(price), currency_code)
        for ts, price in zip(x, closes)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[axis_min] * len(closes),
            mode="lines",
            line=dict(width=0, color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=closes,
            mode="lines",
            line=dict(color=line_color, width=1.8, shape="linear"),
            fill="tonexty",
            fillgradient=dict(
                type="vertical",
                colorscale=[[0.0, fill_bottom], [1.0, fill_top]],
            ),
            marker=dict(size=8, color=line_color, line=dict(width=1.5, color="#fff")),
            customdata=hover_labels,
            hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
            name="",
        )
    )
    fig.update_layout(
        template="simple_white",
        height=CHART_HEIGHT_PRICE,
        margin=dict(l=36, r=8, t=12, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial, Helvetica, sans-serif", color="#80868b"),
        showlegend=False,
        hovermode="x",
        spikedistance=-1,
        hoverdistance=40,
        hoverlabel=dict(
            bgcolor="#fff",
            bordercolor="#dadce0",
            font=dict(size=12, color="#202124", family="Arial, Helvetica, sans-serif"),
            align="left",
        ),
        xaxis=dict(
            showgrid=False,
            showline=True,
            linewidth=1,
            linecolor="#dadce0",
            mirror=False,
            ticks="",
            tickformat=tickformat,
            nticks=5,
            tickfont=dict(size=11, color="#80868b"),
            rangeslider=dict(visible=False),
            fixedrange=True,
            showspikes=True,
            spikemode="across+marker",
            spikecolor="#9aa0a6",
            spikethickness=1,
            spikedash="dash",
            spikesnap="hovered data",
        ),
        yaxis=dict(
            range=[axis_min, axis_max],
            showgrid=True,
            gridcolor="#e8eaed",
            gridwidth=1,
            zeroline=False,
            showline=False,
            ticks="",
            nticks=5,
            tickfont=dict(size=11, color="#80868b"),
            fixedrange=True,
            showspikes=False,
        ),
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False, showline=False)
    return fig


def render_price_history_card(
    analysis: dict,
    *,
    bordered: bool = True,
    currency: str | None = None,
) -> None:
    ticker = analysis.get("ticker", "")
    currency_code = (currency or analysis.get("currency") or "USD").upper()
    with _card_shell(bordered):
        st.markdown('<div class="dashboard-card-body price-history-card">', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:0.92rem;font-weight:700;color:#1e3a5f;padding-top:2px;'
            'padding-bottom:0.15rem;">Price History</div>',
            unsafe_allow_html=True,
        )
        selected = st.radio(
            "Timeframe",
            options=list(PRICE_HISTORY_RANGES.keys()),
            index=list(PRICE_HISTORY_RANGES.keys()).index(DEFAULT_PRICE_RANGE),
            horizontal=True,
            label_visibility="collapsed",
            key=f"ph-range-{ticker}",
        )

        period = PRICE_HISTORY_RANGES[selected]
        hist = fetch_price_history(ticker, period=period)
        if hist.empty or "Close" not in hist.columns:
            st.info("No price history available for this timeframe.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        fig = _price_history_figure(
            hist,
            currency_code=currency_code,
            range_label=selected,
        )
        st.markdown('<div class="dashboard-chart-slot">', unsafe_allow_html=True)
        _plotly_chart(
            fig,
            height=CHART_HEIGHT_PRICE,
            extra_css=_PRICE_CHART_HOVER_CSS,
        )
        st.markdown(
            _price_position_strip_html(analysis) + "</div></div>",
            unsafe_allow_html=True,
        )


def _analyst_recommendations_pie(analyst: dict) -> go.Figure | None:
    total = (
        analyst.get("buy_count", 0)
        + analyst.get("hold_count", 0)
        + analyst.get("sell_count", 0)
    )
    if total <= 0:
        return None

    fig = px.pie(
        values=[
            analyst.get("buy_count", 0),
            analyst.get("hold_count", 0),
            analyst.get("sell_count", 0),
        ],
        names=["Buy", "Hold", "Sell"],
        color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"],
        hole=0.58,
    )
    fig.update_traces(textinfo="none", hoverinfo="skip")
    fig.update_layout(
        height=CHART_HEIGHT_ANALYST_PIE,
        margin=dict(l=0, r=0, t=4, b=4),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _analyst_target_range_html(analyst: dict) -> str:
    pills: list[str] = []
    for lbl, key in [("Low", "target_low"), ("Mean", "target_mean"), ("High", "target_high")]:
        val = analyst.get(key)
        if val is not None:
            pills.append(
                f'<div class="analyst-target-pill">'
                f'<div class="lbl">{lbl}</div>'
                f'<div class="val">${val:,.0f}</div>'
                f"</div>"
            )
    if not pills:
        return ""
    return f'<div class="analyst-targets">{"".join(pills)}</div>'


def render_analyst_card(analysis: dict, *, bordered: bool = True) -> None:
    analyst = analysis.get("analyst", {})

    consensus = analyst.get("consensus_label", "N/A")
    implied_upside = analyst.get("implied_upside_pct")
    num_analysts = analyst.get("num_analysts")

    txt_color, bg_color = consensus_style(consensus)
    upside_color = "#10b981" if (implied_upside or 0) >= 0 else "#ef4444"
    upside_arrow = "↗" if (implied_upside or 0) >= 0 else "↘"
    upside_display = f"{implied_upside:+.0f}%" if implied_upside is not None else "—"
    target_range_html = _analyst_target_range_html(analyst)
    upgrades = analyst.get("recent_upgrades", 0)
    downgrades = analyst.get("recent_downgrades", 0)

    with _card_shell(bordered):
        st.markdown(
            f"""
            <div class="dashboard-card-body analyst-consensus-card">
            <div class="analyst-header-wrap">
            <div style="font-size:0.88rem;font-weight:700;color:#1e3a5f;margin-bottom:0.25rem;">
                Analyst Consensus</div>
            <div style="display:flex;align-items:center;gap:0.55rem;flex-wrap:wrap;margin-bottom:0.2rem;">
                <span style="display:inline-block;background:{bg_color};border-radius:999px;
                    padding:0.18rem 0.85rem;">
                    <span style="font-size:1.15rem;font-weight:800;color:{txt_color};">{consensus}</span>
                </span>
                <span style="font-size:1.3rem;font-weight:800;color:{upside_color};line-height:1;">
                    {upside_display}&thinsp;{upside_arrow}
                </span>
                <span style="font-size:0.6rem;font-weight:600;color:{upside_color};
                    text-transform:uppercase;letter-spacing:0.04em;">implied upside</span>
            </div>
            {target_range_html}
            <div style="font-size:0.64rem;color:#374151;margin-top:0.25rem;margin-bottom:0.15rem;">
                Upgrades <b>{upgrades}</b> · Downgrades <b>{downgrades}</b>
            </div>
            </div>
            <div class="dashboard-chart-slot analyst-chart-slot">
            """,
            unsafe_allow_html=True,
        )

        pie_fig = _analyst_recommendations_pie(analyst)
        if pie_fig is not None:
            _plotly_chart(pie_fig, height=CHART_HEIGHT_ANALYST_PIE)

        analysts_html = (
            f'<div style="font-size:0.65rem;color:#9ca3af;text-align:center;margin-top:0.1rem;">'
            f"{int(num_analysts)} analysts</div>"
            if num_analysts
            else ""
        )
        st.markdown(f"</div>{analysts_html}</div>", unsafe_allow_html=True)

        actions = analyst.get("recent_actions", [])
        if actions:
            with st.expander("Recent analyst actions", expanded=False):
                st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)


def render_factor_radar_card(
    analysis: dict,
    ticker: str,
    *,
    bordered: bool = True,
    radar_labels: dict[str, str] | None = None,
) -> None:
    breakdown = analysis.get("factor_breakdown", {})
    label_map = radar_labels if radar_labels is not None else RADAR_FACTOR_LABELS

    with _card_shell(bordered):
        st.markdown(
            '<div class="dashboard-card-body factor-radar-card">'
            '<div style="font-size:0.88rem;font-weight:700;color:#1e3a5f;margin-bottom:0;">'
            "Factor Radar</div>"
            '<div class="dashboard-chart-slot">',
            unsafe_allow_html=True,
        )

        families = list(label_map.keys())
        raw_vals = [breakdown.get(f, {}).get("percentile") for f in families]
        available = [
            float(v)
            for v in raw_vals
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        fill = sum(available) / len(available) if available else 50.0
        values = [
            float(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else fill
            for v in raw_vals
        ]
        theta_labels = [label_map[f] for f in families]

        fig = go.Figure(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=theta_labels + [theta_labels[0]],
                fill="toself",
                fillcolor="rgba(20, 184, 166, 0.2)",
                line=dict(color="#14b8a6", width=2),
                name=ticker,
            )
        )
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=7),
                    tickvals=[25, 50, 75],
                ),
                angularaxis=dict(tickfont=dict(size=8.5)),
            ),
            showlegend=False,
            height=CHART_HEIGHT_RADAR,
            margin=dict(l=32, r=32, t=12, b=12),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        _plotly_chart(fig, height=CHART_HEIGHT_RADAR)
        st.markdown("</div></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main views
# ──────────────────────────────────────────────────────────────────────────────

def render_fund_header(analysis: dict) -> None:
    ticker = analysis.get("ticker", "")
    name = analysis.get("name") or ticker
    exchange = analysis.get("exchange") or ""
    category = analysis.get("category") or ""
    fund_family = analysis.get("fund_family") or ""
    total_assets = analysis.get("total_assets")
    price = analysis.get("price")
    currency = analysis.get("currency")
    badge = SECURITY_TYPE_BADGES.get(analysis.get("security_type") or "", "Fund")

    ticker_e = html.escape(str(ticker))
    name_e = html.escape(str(name))
    exchange_e = html.escape(str(exchange)) if exchange else ""
    sym = currency_symbol(currency)

    price_html = (
        f' <span style="font-size:1.25rem;font-weight:700;color:#1e3a5f;white-space:nowrap;">'
        f"{sym}{price:,.2f}</span>"
        if price
        else ""
    )
    badge_html = (
        f' <span style="font-size:0.62rem;font-weight:700;color:#0d9488;background:#ccfbf1;'
        f'border-radius:999px;padding:0.14rem 0.55rem;vertical-align:middle;'
        f'text-transform:uppercase;letter-spacing:0.05em;">{html.escape(badge)}</span>'
    )
    exchange_html = (
        f' <span style="color:#d1d5db;">|</span> '
        f'<span style="font-size:0.88rem;color:#9ca3af;">{exchange_e}</span>'
        if exchange_e
        else ""
    )

    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f'<div style="padding:0.05rem 0 0.1rem;line-height:1.35;">'
            f'<span style="font-size:1.55rem;font-weight:800;color:#1e3a5f;">{ticker_e}</span>'
            f"{price_html}{badge_html}<br>"
            f'<span style="font-size:0.82rem;color:#6b7280;">{name_e}</span>'
            f"{exchange_html}"
            f"</div>",
            unsafe_allow_html=True,
        )

    with right:
        def _facet(label: str, value: str, bold: bool = False) -> str:
            weight = "700" if bold else "500"
            return (
                f'<span style="display:inline-block;margin-left:1.25rem;">'
                f'<span style="display:block;font-size:0.68rem;color:#9ca3af;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;">{html.escape(label)}</span>'
                f'<span style="display:block;font-size:0.88rem;color:#374151;font-weight:{weight};">'
                f"{html.escape(value)}</span></span>"
            )

        parts = []
        if category:
            parts.append(_facet("Category", str(category)))
        if fund_family:
            parts.append(_facet("Fund Family", str(fund_family)))
        if total_assets:
            parts.append(_facet("Net Assets", fmt_large_number(total_assets, currency), bold=True))
        if parts:
            st.markdown(
                f'<div style="text-align:right;padding:0.15rem 0 0.25rem;">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )


def render_fund_composite_card(
    analysis: dict,
    *,
    bordered: bool = True,
    snapshot_date: str | None = None,
) -> None:
    composite = analysis.get("composite")
    comp_color = gauge_score_color(composite)
    comp_label = gauge_score_label(composite)
    bargain = analysis.get("bargain") or {}
    bargain_score = bargain.get("score")
    bargain_color = gauge_score_color(bargain_score)
    bargain_label = gauge_score_label(bargain_score)

    date_label = _format_snapshot_date(snapshot_date)
    subtitle = (
        f"vs US + Canadian fund universe snapshot from {date_label}"
        if date_label
        else "vs US + Canadian fund universe"
    )

    composite_gauge = _arc_gauge_html(
        composite,
        comp_label,
        comp_color,
        subtitle=subtitle,
        aria_label="Fund composite score gauge",
        fill_color=comp_color,
        max_width=GAUGE_MAX_WIDTH,
    )
    bargain_gauge = _arc_gauge_html(
        bargain_score,
        bargain_label,
        bargain_color,
        subtitle="52W discount · RSI oversold",
        aria_label="Fund bargain score gauge",
        fill_color=bargain_color,
        max_width=GAUGE_MAX_WIDTH,
    )

    rsi = analysis.get("rsi_14")
    rsi_note = (
        f'<div style="text-align:center;color:#6b7280;font-size:0.8rem;margin-top:0.25rem;">'
        f"RSI(14): {rsi:.0f} (included in bargain score)</div>"
        if rsi is not None
        else ""
    )

    with _card_shell(bordered):
        st.markdown(
            '<div class="dashboard-card-body composite-score-card">'
            '<div class="composite-gauges-row">'
            '<div class="gauge-cell">'
            '<div class="gauge-title">Fund Composite Score</div>'
            + composite_gauge
            + '</div>'
            '<div class="gauge-cell">'
            '<div class="gauge-title">Bargain Score</div>'
            + bargain_gauge
            + rsi_note
            + "</div></div></div>",
            unsafe_allow_html=True,
        )


def _fund_fact_pill(label: str, value: str, color: str = "#1e3a5f") -> str:
    return (
        f'<div class="analyst-target-pill">'
        f'<div class="lbl">{html.escape(label)}</div>'
        f'<div class="val" style="color:{color};">{html.escape(value)}</div>'
        f"</div>"
    )


def render_fund_facts_card(analysis: dict, *, bordered: bool = True) -> None:
    expense_ratio = analysis.get("expense_ratio")
    dist_yield = analysis.get("distribution_yield")
    nav_premium = analysis.get("nav_premium")
    beta = analysis.get("beta_3y")

    expense_txt = f"{expense_ratio * 100:.2f}%" if expense_ratio is not None else "N/A"
    yield_txt = f"{dist_yield * 100:.2f}%" if dist_yield is not None else "N/A"

    def _ret_pill(label: str, val: float | None) -> str:
        if val is None:
            return _fund_fact_pill(label, "—", "#9ca3af")
        color = "#10b981" if val >= 0 else "#ef4444"
        return _fund_fact_pill(label, f"{val * 100:+.1f}%", color)

    nav_html = ""
    if nav_premium is not None:
        nav_color = "#10b981" if nav_premium <= 0 else "#ef4444"
        nav_html = (
            f'<div style="font-size:0.64rem;color:#374151;margin-top:0.3rem;">'
            f'NAV premium/discount: <b style="color:{nav_color};">{nav_premium * 100:+.2f}%</b>'
            f"</div>"
        )
    beta_html = (
        f'<div style="font-size:0.64rem;color:#374151;margin-top:0.15rem;">'
        f"Beta (3Y): <b>{beta:.2f}</b></div>"
        if beta is not None
        else ""
    )

    with _card_shell(bordered):
        st.markdown(
            f"""
            <div class="dashboard-card-body analyst-consensus-card">
            <div class="analyst-header-wrap">
            <div style="font-size:0.88rem;font-weight:700;color:#1e3a5f;margin-bottom:0.25rem;">
                Fund Facts</div>
            <div class="analyst-targets">
                {_fund_fact_pill("Expense", expense_txt)}
                {_fund_fact_pill("Yield", yield_txt)}
            </div>
            <div style="font-size:0.58rem;font-weight:600;color:#9ca3af;text-transform:uppercase;
                letter-spacing:0.05em;margin-top:0.5rem;">Annualized returns</div>
            <div class="analyst-targets" style="margin-top:0.2rem;">
                {_ret_pill("1Y", analysis.get("return_1y"))}
                {_ret_pill("3Y", analysis.get("return_3y"))}
                {_ret_pill("5Y", analysis.get("return_5y"))}
            </div>
            {nav_html}
            {beta_html}
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_fund_view(
    ticker: str,
    config: dict,
    scored_fund_universe: pd.DataFrame | None = None,
    snapshot_date: str | None = None,
) -> None:
    with st.spinner(f"Analyzing {ticker}…"):
        analysis = score_fund(ticker, config)
        if scored_fund_universe is not None and not scored_fund_universe.empty:
            analysis = apply_fund_snapshot_scoring(analysis, scored_fund_universe, ticker)

    if analysis.get("warning"):
        st.warning(analysis["warning"])

    currency = analysis.get("currency")

    # Fund header card
    with st.container(border=True):
        render_fund_header(analysis)

    st.markdown("<div style='margin-top:0.35rem;'></div>", unsafe_allow_html=True)

    # Row 1: Fund Composite | Fund Factor Scorecard
    _dashboard_row_anchor(1)
    row1_left, row1_right = st.columns([2.6, 4.7], gap="small", border=True)
    with row1_left:
        render_fund_composite_card(analysis, bordered=False, snapshot_date=snapshot_date)
    with row1_right:
        render_factor_scorecard_card(
            analysis,
            bordered=False,
            groups=FUND_SCORECARD_GROUPS,
            labels=SHORT_FUND_FACTOR_LABELS,
            help_texts=FUND_FACTOR_HELP,
        )

    st.markdown("<div style='margin-top:0.35rem;'></div>", unsafe_allow_html=True)

    # Row 2: Fund Facts | Price History | Fund Factor Radar
    _dashboard_row_anchor(2)
    row2_a, row2_b, row2_c = st.columns([2.2, 3.5, 1.8], gap="small", border=True)
    with row2_a:
        render_fund_facts_card(analysis, bordered=False)
    with row2_b:
        render_price_history_card(
            analysis,
            bordered=False,
            currency=currency,
        )
    with row2_c:
        render_factor_radar_card(
            analysis,
            ticker,
            bordered=False,
            radar_labels=RADAR_FUND_FACTOR_LABELS,
        )

    inject_equal_height_js()

    st.caption(
        "Funds are scored on fund-appropriate factors (fees, realized returns, "
        "risk-adjusted return, volatility, momentum, income) against a peer universe "
        "of US and Canadian ETFs and mutual funds. The bargain score is price-based only "
        "(52-week high discount + RSI oversold signal) since fund financials don't exist. "
        "Stock metrics that rely on company financials or analyst coverage — Graham margin "
        "of safety, balance-sheet strength, analyst consensus — are intentionally excluded "
        "rather than approximated."
    )

    if analysis.get("is_etf"):
        holdings = fetch_etf_holdings(ticker)
        if not holdings.empty:
            with st.expander("Top holdings"):
                st.dataframe(holdings, use_container_width=True, hide_index=True)

    with st.expander("Raw factor values"):
        st.json(analysis.get("factors_raw", {}))

    if analysis.get("description"):
        with st.expander("Description"):
            st.write(analysis["description"])


def render_stock_view(
    ticker: str,
    config: dict,
    scored_universe: pd.DataFrame | None = None,
    snapshot_date: str | None = None,
) -> None:
    with st.spinner(f"Analyzing {ticker}…"):
        analysis = score_ticker(ticker, config)
        if scored_universe is not None and not scored_universe.empty:
            analysis = apply_universe_snapshot_scoring(analysis, scored_universe, ticker, config)

    if analysis.get("warning"):
        st.warning(analysis["warning"])

    data_warnings = analysis.get("data_warnings") or []
    if data_warnings:
        with st.expander("Data warnings", expanded=False):
            for w in data_warnings:
                st.warning(w)

    thresholds = get_thresholds(config)
    if analysis.get("distress_flag"):
        altman_z = analysis.get("altman_z")
        z_txt = f"{altman_z:.2f}" if altman_z is not None else "n/a"
        st.error(
            f"Distress zone: Altman Z = {z_txt} "
            f"(below {thresholds.get('altman_z_min', 1.8)}). "
            "Blocked from Buy regardless of composite/bargain scores."
        )
    if analysis.get("is_good_buy"):
        bump = (analysis.get("uncertainty") or {}).get("threshold_bump") or 0
        hurdle = (
            f" (hurdles +{bump:.0f} for {analysis.get('uncertainty', {}).get('label')} uncertainty)"
            if bump
            else ""
        )
        st.success(
            f"Meets good-buy criteria (composite ≥ {thresholds['composite_min']}, "
            f"bargain ≥ {thresholds.get('bargain_min', 50)}, "
            f"no distress flag, consensus not Sell){hurdle}"
        )

    # Company header card
    with st.container(border=True):
        render_company_header(analysis)

    st.markdown("<div style='margin-top:0.35rem;'></div>", unsafe_allow_html=True)

    # Row 1: Composite Score | Factor Scorecard
    # border=True on columns (not nested containers) — Streamlit's supported equal-height layout.
    _dashboard_row_anchor(1)
    row1_left, row1_right = st.columns([2.6, 4.7], gap="small", border=True)
    with row1_left:
        render_composite_card(analysis, bordered=False, snapshot_date=snapshot_date)
    with row1_right:
        render_factor_scorecard_card(analysis, bordered=False)

    st.markdown("<div style='margin-top:0.35rem;'></div>", unsafe_allow_html=True)

    # Row 2: Analyst Consensus | Price History | Factor Radar
    _dashboard_row_anchor(2)
    row2_a, row2_b, row2_c = st.columns([2.2, 3.5, 1.8], gap="small", border=True)
    with row2_a:
        render_analyst_card(analysis, bordered=False)
    with row2_b:
        render_price_history_card(analysis, bordered=False)
    with row2_c:
        render_factor_radar_card(analysis, ticker, bordered=False)

    inject_equal_height_js()

    with st.expander("Raw factor values"):
        st.json(analysis.get("factors_raw", {}))


def render_universe_rankings(
    config: dict,
    scored_universe: pd.DataFrame | None = None,
) -> None:
    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        st.warning("No universe snapshot found. Run the monthly universe job to build one.")
        return

    scored = scored_universe if scored_universe is not None else score_universe(config)
    if scored.empty or "composite" not in scored.columns:
        st.warning("Unable to score universe.")
        return

    scored["composite"] = pd.to_numeric(scored["composite"], errors="coerce")
    top = scored.nlargest(20, "composite")[["ticker", "name", "sector", "composite"]]
    st.markdown("### Top 20 by Composite Score (Universe)")
    st.dataframe(top, use_container_width=True, hide_index=True)


def render_fund_universe_rankings(
    scored_fund_universe: pd.DataFrame | None = None,
) -> None:
    scored = scored_fund_universe
    if scored is None or scored.empty or "composite" not in scored.columns:
        st.warning(
            "No fund universe snapshot found. Run `python -m core.fund_universe` to build one."
        )
        return

    scored = scored.copy()
    scored["composite"] = pd.to_numeric(scored["composite"], errors="coerce")
    scored["type"] = (
        scored.get("quote_type", pd.Series(dtype=str))
        .map(SECURITY_TYPE_BADGES)
        .fillna("Fund")
    )
    cols = [c for c in ["ticker", "name", "type", "category", "currency", "composite"] if c in scored.columns]
    top = scored.nlargest(20, "composite")[cols]
    st.markdown("### Top 20 Funds by Composite Score (US + Canada)")
    st.dataframe(top, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _render_stock_sidebar_sections(config: dict) -> None:
    st.markdown("**Good-buy criteria**")
    thresholds = get_thresholds(config)
    st.write(f"Composite ≥ {thresholds['composite_min']}")
    st.write(f"Bargain ≥ {thresholds.get('bargain_min', 50)}")
    st.write(f"Altman Z ≥ {thresholds.get('altman_z_min', 1.8)} (ex-financials/RE/utilities)")
    st.write(
        f"High uncertainty adds +{thresholds.get('uncertainty_high_bump', 6):.0f} "
        "to both hurdles"
    )
    if thresholds.get("exclude_sell_consensus"):
        st.write("Excludes sell-consensus names")
    st.caption(
        f"Analyst upside (info only; context ≥ {thresholds['implied_upside_min_pct']}%)"
    )
    st.markdown("---")
    st.markdown("**Composite factor weights**")
    st.caption(
        "Nine factor groups (revisions and insider buying are live-only). "
        "Shown as a share of total; renormalized at runtime over groups with data."
    )
    factor_weights = get_factor_weights(config)
    factor_total = sum(factor_weights.values()) or 1.0
    for family in sorted(FACTOR_SCORE_COLUMNS, key=lambda f: factor_weights.get(f, 0.0), reverse=True):
        weight = factor_weights.get(family, 0.0)
        st.write(f"{FACTOR_LABELS.get(family, family)}: {weight / factor_total:.1%}")
    st.markdown("---")
    st.markdown("**Bargain score weights**")
    bargain_weights = get_bargain_weights(config)
    bargain_total = sum(bargain_weights.values()) or 1.0
    for key in sorted(bargain_weights, key=lambda k: bargain_weights.get(k, 0.0), reverse=True):
        weight = bargain_weights[key]
        st.write(f"{BARGAIN_LABELS.get(key, key)}: {weight / bargain_total:.1%}")

    snapshot = load_universe_snapshot()
    if snapshot is not None and not snapshot.empty:
        date = snapshot["snapshot_date"].iloc[0] if "snapshot_date" in snapshot.columns else "unknown"
        st.caption(f"Universe: {len(snapshot)} tickers (snapshot: {date})")


def _render_fund_sidebar_sections(config: dict) -> None:
    st.markdown("**Fund composite weights**")
    st.caption(
        "Six fund factor groups (fees first — the strongest documented predictor "
        "of relative fund performance). Renormalized at runtime over groups with data."
    )
    fund_weights = get_fund_factor_weights(config)
    fund_total = sum(fund_weights.values()) or 1.0
    for family in sorted(FUND_FACTOR_SCORE_COLUMNS, key=lambda f: fund_weights.get(f, 0.0), reverse=True):
        weight = fund_weights.get(family, 0.0)
        st.write(f"{FUND_FACTOR_LABELS.get(family, family)}: {weight / fund_total:.1%}")

    snapshot = load_fund_universe_snapshot()
    if snapshot is not None and not snapshot.empty:
        date = snapshot["snapshot_date"].iloc[0] if "snapshot_date" in snapshot.columns else "unknown"
        st.caption(f"Fund universe: {len(snapshot)} funds (snapshot: {date})")


def main() -> None:
    inject_css()
    config = load_config()

    with st.sidebar:
        st.header("Settings")
        default_ticker = st.query_params.get("ticker", "AAPL")
        ticker = st.text_input(
            "Ticker",
            value=default_ticker,
            help="Stocks, ETFs, and mutual funds (US and Canadian; use .TO for TSX listings).",
        ).upper().strip()
        viewing_fund = bool(ticker) and get_security_type(ticker) in FUND_QUOTE_TYPES
        st.markdown("---")
        if viewing_fund:
            _render_fund_sidebar_sections(config)
        else:
            _render_stock_sidebar_sections(config)

    if viewing_fund:
        fund_snapshot = load_fund_universe_snapshot()
        fund_snapshot_date = None
        if (
            fund_snapshot is not None
            and not fund_snapshot.empty
            and "snapshot_date" in fund_snapshot.columns
        ):
            fund_snapshot_date = str(fund_snapshot["snapshot_date"].iloc[0])

        scored_fund_universe = score_fund_universe(config)
        render_fund_view(ticker, config, scored_fund_universe, fund_snapshot_date)
        st.markdown("---")
        render_fund_universe_rankings(scored_fund_universe)
        return

    snapshot = load_universe_snapshot()
    snapshot_date = None
    if snapshot is not None and not snapshot.empty and "snapshot_date" in snapshot.columns:
        snapshot_date = str(snapshot["snapshot_date"].iloc[0])

    scored_universe = score_universe(config)

    if not ticker:
        st.markdown("## Stock & Fund Metrics Tool")
        st.caption("Enter a stock, ETF, or mutual fund ticker in the sidebar to get started.")
        render_universe_rankings(config, scored_universe)
        return

    render_stock_view(ticker, config, scored_universe, snapshot_date)

    st.markdown("---")
    render_universe_rankings(config, scored_universe)


if __name__ == "__main__":
    main()
