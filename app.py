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

from core.config import get_bargain_weights, get_factor_weights, get_thresholds, load_config
from core.data import fetch_etf_holdings, fetch_etf_info, fetch_price_history, is_etf
from core.scoring import score_ticker, score_universe
from core.universe import load_universe_snapshot

st.set_page_config(
    page_title="Stock Metrics Tool",
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
    "value": "Value (earnings yield · B/M · FCF · Graham)",
    "garp": "GARP (Lynch dividend-adjusted PEG)",
    "quality": "Quality / Profitability",
    "balance_sheet": "Balance Sheet Strength",
    "momentum": "Momentum (12-1)",
    "low_volatility": "Low Volatility",
    "capital_discipline": "Capital Discipline (yield + asset growth)",
    "earnings_revisions": "Earnings Revisions",
}

BARGAIN_LABELS = {
    "margin_of_safety": "Margin of Safety (Graham)",
    "discount_52w": "Discount to 52-Week High",
    "rsi_oversold": "RSI Oversold",
}

SHORT_FACTOR_LABELS = {
    "value": "Value",
    "garp": "GARP",
    "quality": "Quality",
    "balance_sheet": "Balance Sheet",
    "momentum": "Momentum",
    "low_volatility": "Low Volatility",
    "capital_discipline": "Capital Discipline",
    "earnings_revisions": "Est. Revisions",
}

# Short labels for the 8-spoke radar chart.
RADAR_FACTOR_LABELS: dict[str, str] = {
    "value":              "Value",
    "garp":               "GARP",
    "quality":            "Quality",
    "balance_sheet":      "Bal. Sheet",
    "momentum":           "Momentum",
    "low_volatility":     "Low Vol",
    "capital_discipline": "Cap. Disc.",
    "earnings_revisions": "Est. Rev.",
}

# Factor Scorecard display: 2 groups per column.
# Each entry: (group_label, accent_color, [factor_keys])
FACTOR_SCORECARD_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Valuation", "#14b8a6", ["value", "garp"]),
    ("Quality & Health", "#8b5cf6", ["quality", "balance_sheet"]),
    ("Returns & Capital", "#3b82f6", ["capital_discipline", "momentum"]),
    ("Market & Sentiment", "#f59e0b", ["low_volatility", "earnings_revisions"]),
]

FACTOR_COLORS = {
    "value":              "#14b8a6",
    "garp":               "#10b981",
    "quality":            "#8b5cf6",
    "balance_sheet":      "#60a5fa",
    "momentum":           "#3b82f6",
    "low_volatility":     "#f59e0b",
    "capital_discipline": "#34d399",
    "earnings_revisions": "#ec4899",
}

METRIC_HELP = {
    "composite_score": (
        "Single number from 0–100 that blends how this stock ranks on 8 factor groups "
        "(value, GARP, quality, balance sheet, momentum, low volatility, capital discipline, "
        "earnings revisions) vs S&P 500 peers. Each group rank-averages its own sub-signals "
        "before weighting, so no single ratio dominates. 50+ is the configured composite_min "
        "good-buy bar. Only groups with available data are included; check Factor Coverage."
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
}

# Hover copy for Factor Scorecard: what the metric means and how it's built.
FACTOR_HELP: dict[str, str] = {
    "value": (
        "Composite value rank: each of four sub-signals (earnings yield EBIT/EV, "
        "FCF yield, book-to-market, Graham ratio) is ranked cross-sectionally then "
        "averaged. Higher = cheaper vs peers on multiple measures."
    ),
    "garp": (
        "Growth at a reasonable price (Peter Lynch): (earnings growth % + dividend yield %) "
        "divided by P/E. Higher = more growth and income per dollar of valuation. "
        "Falls back to 1/PEG when analyst growth estimates are unavailable."
    ),
    "quality": (
        "Composite quality rank: seven sub-signals (gross profitability, ROE, ROA, "
        "profit margin, ROIC, earnings quality/accruals, Piotroski F-Score) are each "
        "ranked then averaged. Higher = more profitable and cleaner business."
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
    "earnings_revisions": (
        "Analyst recommendation momentum: recent upgrades minus downgrades from "
        "published recommendation history. Higher = improving analyst sentiment. "
        "Live-only signal (excluded from historical backtesting)."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][min(n % 10, 3)]}"


def fmt_large_number(n: float | None) -> str:
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"${n / 1e12:.2f}T"
    if n >= 1e9:
        return f"${n / 1e9:.1f}B"
    if n >= 1e6:
        return f"${n / 1e6:.1f}M"
    return f"${n:,.0f}"


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
            align-items: flex-end;
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

    score_txt = f"{score:.0f}" if score is not None else "N/A"
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


def render_composite_card(analysis: dict, *, bordered: bool = True) -> None:
    composite = analysis.get("composite")
    comp_color = gauge_score_color(composite)
    comp_label = gauge_score_label(composite)
    bargain = analysis.get("bargain") or {}
    bargain_score = bargain.get("score")
    bargain_color = gauge_score_color(bargain_score)
    bargain_label = gauge_score_label(bargain_score)

    composite_gauge = _arc_gauge_html(
        composite,
        comp_label,
        comp_color,
        subtitle="vs. Global Universe",
        aria_label="Composite score gauge",
        fill_color=comp_color,
        max_width=GAUGE_MAX_WIDTH,
    )
    bargain_gauge = _arc_gauge_html(
        bargain_score,
        bargain_label,
        bargain_color,
        subtitle="Graham · 52W discount · RSI",
        aria_label="Bargain score gauge",
        fill_color=bargain_color,
        max_width=GAUGE_MAX_WIDTH,
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
            + "</div></div></div>",
            unsafe_allow_html=True,
        )


def _factor_label_html(factor_key: str, short_label: str) -> str:
    """Metric name with hover tooltip (meaning + calculation)."""
    label_e = html.escape(short_label)
    help_text = FACTOR_HELP.get(factor_key)
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
) -> str:
    header = (
        f'<div style="font-size:0.58rem;font-weight:700;color:{accent};text-transform:uppercase;'
        f'letter-spacing:0.07em;margin:0 0 4px;padding-bottom:2px;'
        f'border-bottom:1px solid {accent}22;'
        f'overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{group_label}</div>'
    )
    rows = []
    for key in factor_keys:
        short_label = SHORT_FACTOR_LABELS.get(key, key)
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
            f"{_factor_label_html(key, short_label)}"
            f'<div class="factor-bar-track">'
            f'<div class="factor-bar-fill" style="width:{bar_w:.0f}%;background:{color};"></div>'
            f"</div>"
            f'<div class="factor-pct" style="color:{color};">{pct_text}</div>'
            f"</div>"
        )
    return f'<div class="factor-scorecard-group">{header}{"".join(rows)}</div>'


def render_factor_scorecard_card(analysis: dict, *, bordered: bool = True) -> None:
    breakdown = analysis.get("factor_breakdown", {})

    with _card_shell(bordered):
        # 2-column CSS grid: left = Valuation + Quality, right = Financial Health + Market
        left_html = "".join(
            _factor_group_html(lbl, acc, keys, breakdown)
            for lbl, acc, keys in FACTOR_SCORECARD_GROUPS[:2]
        )
        right_html = "".join(
            _factor_group_html(lbl, acc, keys, breakdown)
            for lbl, acc, keys in FACTOR_SCORECARD_GROUPS[2:]
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


def render_price_history_card(analysis: dict, *, bordered: bool = True) -> None:
    ticker = analysis.get("ticker", "")
    with _card_shell(bordered):
        st.markdown('<div class="dashboard-card-body price-history-card">', unsafe_allow_html=True)
        header_col, range_col = st.columns([3, 2])
        with header_col:
            st.markdown(
                '<div style="font-size:0.92rem;font-weight:700;color:#1e3a5f;padding-top:5px;">'
                "Price History</div>",
                unsafe_allow_html=True,
            )
        with range_col:
            selected = st.selectbox(
                "Timeframe",
                options=list(PRICE_HISTORY_RANGES.keys()),
                index=list(PRICE_HISTORY_RANGES.keys()).index(DEFAULT_PRICE_RANGE),
                label_visibility="collapsed",
                key=f"ph_{ticker}",
            )

        period = PRICE_HISTORY_RANGES[selected]
        hist = fetch_price_history(ticker, period=period)
        if hist.empty:
            st.info("No price history available for this timeframe.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        fig = go.Figure()

        if all(c in hist.columns for c in ["Open", "High", "Low", "Close"]):
            fig.add_trace(
                go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"],
                    name="Price",
                    increasing=dict(line=dict(color="#10b981", width=1), fillcolor="#10b981"),
                    decreasing=dict(line=dict(color="#ef4444", width=1), fillcolor="#ef4444"),
                )
            )

        ma20 = hist["Close"].rolling(20).mean()
        fig.add_trace(
            go.Scatter(
                x=hist.index,
                y=ma20,
                mode="lines",
                line=dict(color="#3b82f6", width=1.5),
                name="Close Price",
                opacity=0.85,
            )
        )

        fig.update_layout(
            height=CHART_HEIGHT_PRICE,
            margin=dict(l=10, r=10, t=16, b=14),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(
                orientation="h",
                y=1.06,
                x=0,
                font=dict(size=10, color="#6b7280"),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis=dict(
                showgrid=False,
                tickfont=dict(size=10, color="#9ca3af"),
                rangeslider=dict(visible=False),
                spikecolor="#94a3b8",
                spikethickness=1,
                spikesnap="cursor",
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor="#f3f4f6",
                gridwidth=1,
                tickfont=dict(size=10, color="#9ca3af"),
                tickprefix="$",
            ),
            hovermode="x unified",
        )
        st.markdown('<div class="dashboard-chart-slot">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
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
            st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})

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


def render_factor_radar_card(analysis: dict, ticker: str, *, bordered: bool = True) -> None:
    breakdown = analysis.get("factor_breakdown", {})

    with _card_shell(bordered):
        st.markdown(
            '<div class="dashboard-card-body factor-radar-card">'
            '<div style="font-size:0.88rem;font-weight:700;color:#1e3a5f;margin-bottom:0;">'
            "Factor Radar</div>"
            '<div class="dashboard-chart-slot">',
            unsafe_allow_html=True,
        )

        families = list(RADAR_FACTOR_LABELS.keys())
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
        theta_labels = [RADAR_FACTOR_LABELS[f] for f in families]

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
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div></div>", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# Main views
# ──────────────────────────────────────────────────────────────────────────────

def render_etf_view(ticker: str) -> None:
    info = fetch_etf_info(ticker)
    with st.container(border=True):
        st.subheader(f"{info.get('name') or ticker} (ETF)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Price",
            f"${info.get('current_price', 0):,.2f}" if info.get("current_price") else "N/A",
            help=METRIC_HELP["etf_price"],
        )
        c2.metric(
            "Expense Ratio",
            f"{(info.get('expense_ratio') or 0)*100:.2f}%" if info.get("expense_ratio") else "N/A",
            help=METRIC_HELP["etf_expense_ratio"],
        )
        c3.metric("Category", info.get("category") or "N/A", help=METRIC_HELP["etf_category"])
        c4.metric(
            "Yield",
            f"{(info.get('yield') or 0)*100:.2f}%" if info.get("yield") else "N/A",
            help=METRIC_HELP["etf_yield"],
        )

    st.info("ETFs are excluded from empirical factor scoring. Showing basic fund info only.")

    holdings = fetch_etf_holdings(ticker)
    if not holdings.empty:
        st.subheader("Top Holdings")
        st.dataframe(holdings, use_container_width=True)

    hist = fetch_price_history(ticker, period="1y")
    if not hist.empty:
        fig = go.Figure(
            go.Scatter(
                x=hist.index,
                y=hist["Close"],
                mode="lines",
                line=dict(color="#14b8a6", width=2),
                name="Price",
            )
        )
        fig.update_layout(
            title=f"{ticker} — 1Y Price",
            height=280,
            margin=dict(l=10, r=10, t=40, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(showgrid=True, gridcolor="#f3f4f6", tickprefix="$"),
            xaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    if info.get("description"):
        with st.expander("Description"):
            st.write(info["description"])


def render_stock_view(ticker: str, config: dict) -> None:
    with st.spinner(f"Analyzing {ticker}…"):
        analysis = score_ticker(ticker, config)

    if analysis.get("warning"):
        st.warning(analysis["warning"])

    data_warnings = analysis.get("data_warnings") or []
    if data_warnings:
        with st.expander("Data warnings", expanded=False):
            for w in data_warnings:
                st.warning(w)

    thresholds = get_thresholds(config)
    if analysis.get("is_good_buy"):
        st.success(
            f"Meets good-buy criteria (composite ≥ {thresholds['composite_min']}, "
            f"upside ≥ {thresholds['implied_upside_min_pct']}%, "
            f"bargain ≥ {thresholds.get('bargain_min', 50)})"
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
        render_composite_card(analysis, bordered=False)
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


def render_universe_rankings(config: dict) -> None:
    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        st.warning("No universe snapshot found. Run the monthly universe job to build one.")
        return

    scored = score_universe(config)
    if scored.empty or "composite" not in scored.columns:
        st.warning("Unable to score universe.")
        return

    scored["composite"] = pd.to_numeric(scored["composite"], errors="coerce")
    top = scored.nlargest(20, "composite")[["ticker", "name", "sector", "composite"]]
    st.markdown("### Top 20 by Composite Score (Universe)")
    st.dataframe(top, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    inject_css()
    config = load_config()

    with st.sidebar:
        st.header("Settings")
        default_ticker = st.query_params.get("ticker", "AAPL")
        ticker = st.text_input("Ticker", value=default_ticker).upper().strip()
        st.markdown("---")
        st.markdown("**Good-buy criteria**")
        thresholds = get_thresholds(config)
        st.write(f"Composite ≥ {thresholds['composite_min']}")
        st.write(f"Implied upside ≥ {thresholds['implied_upside_min_pct']}%")
        st.write(f"Bargain ≥ {thresholds.get('bargain_min', 50)}")
        if thresholds.get("exclude_sell_consensus"):
            st.write("Excludes sell-consensus names")
        st.markdown("---")
        st.markdown("**Composite factor weights**")
        st.caption(
            "Tuned on historical data (DCA k-fold cross-validation). "
            "Shown as a share of total; renormalized at runtime over factors with data."
        )
        factor_weights = get_factor_weights(config)
        factor_total = sum(factor_weights.values()) or 1.0
        for k, v in sorted(factor_weights.items(), key=lambda kv: kv[1], reverse=True):
            st.write(f"{FACTOR_LABELS.get(k, k)}: {v / factor_total:.1%}")
        st.markdown("---")
        st.markdown("**Bargain score weights**")
        bargain_weights = get_bargain_weights(config)
        bargain_total = sum(bargain_weights.values()) or 1.0
        for k, v in sorted(bargain_weights.items(), key=lambda kv: kv[1], reverse=True):
            st.write(f"{BARGAIN_LABELS.get(k, k)}: {v / bargain_total:.1%}")

        snapshot = load_universe_snapshot()
        if snapshot is not None and not snapshot.empty:
            date = snapshot["snapshot_date"].iloc[0] if "snapshot_date" in snapshot.columns else "unknown"
            st.caption(f"Universe: {len(snapshot)} tickers (snapshot: {date})")

    if not ticker:
        st.markdown("## Stock Metrics Tool")
        st.caption("Enter a ticker in the sidebar to get started.")
        render_universe_rankings(config)
        return

    if is_etf(ticker):
        render_etf_view(ticker)
    else:
        render_stock_view(ticker, config)

    st.markdown("---")
    render_universe_rankings(config)


if __name__ == "__main__":
    main()
