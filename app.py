"""Streamlit dashboard for stock metrics and analyst aggregation."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.config import get_factor_weights, get_thresholds, load_config
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

# Chart heights tuned so the stock dashboard fits one viewport (no scroll).
CHART_HEIGHT_PRICE = 225
CHART_HEIGHT_RADAR = 235
CHART_HEIGHT_ANALYST_PIE = 110

FACTOR_LABELS = {
    "value": "Value",
    "momentum": "Momentum (12-1)",
    "quality": "Quality / Profitability",
    "low_volatility": "Low Volatility",
    "investment": "Investment (asset growth)",
    "earnings_revisions": "Earnings Revisions",
    "financial_strength": "Financial Strength (Piotroski)",
    "garp": "GARP (Lynch PEG)",
    "balance_sheet_strength": "Balance Sheet Strength",
    "graham_value": "Graham Number Value",
    "downside_protection": "Downside Protection (Marks)",
    "earnings_quality": "Earnings Quality (Accruals)",
    "shareholder_yield": "Shareholder Yield",
    "capital_efficiency": "Capital Efficiency (ROIC)",
    "distress_risk": "Distress Risk (Altman Z)",
}

SHORT_FACTOR_LABELS = {
    "value": "Value",
    "momentum": "Momentum",
    "quality": "Quality",
    "low_volatility": "Low Volatility",
    "investment": "Investment",
    "earnings_revisions": "Earnings Revisions",
    "financial_strength": "Financial Strength",
    "garp": "GARP",
    "balance_sheet_strength": "Balance Sheet",
    "graham_value": "Graham Value",
    "downside_protection": "Downside Protection",
    "earnings_quality": "Earnings Quality",
    "shareholder_yield": "Shareholder Yield",
    "capital_efficiency": "Capital Efficiency",
    "distress_risk": "Distress Risk",
}

# Very short labels for the 15-spoke radar chart where space is tight.
RADAR_FACTOR_LABELS: dict[str, str] = {
    "value":                  "Value",
    "momentum":               "Momentum",
    "quality":                "Quality",
    "low_volatility":         "Low Vol",
    "investment":             "Investment",
    "earnings_revisions":     "Est. Rev.",
    "financial_strength":     "Piotroski",
    "garp":                   "GARP",
    "balance_sheet_strength": "Bal. Sheet",
    "graham_value":           "Graham",
    "downside_protection":    "Downside",
    "earnings_quality":       "Accruals",
    "shareholder_yield":      "Shr. Yield",
    "capital_efficiency":     "ROIC",
    "distress_risk":          "Altman Z",
}

# Conceptual groupings for the Factor Scorecard display.
# Each entry: (group_label, accent_color, [factor_keys])
FACTOR_SCORECARD_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Valuation", "#14b8a6", ["value", "garp", "graham_value", "shareholder_yield"]),
    ("Quality & Profitability", "#8b5cf6", ["quality", "capital_efficiency", "earnings_quality"]),
    ("Financial Health", "#3b82f6", ["financial_strength", "balance_sheet_strength", "distress_risk", "investment"]),
    ("Market & Sentiment", "#f59e0b", ["momentum", "earnings_revisions", "low_volatility", "downside_protection"]),
]

FACTOR_COLORS = {
    "value":                  "#14b8a6",
    "momentum":               "#3b82f6",
    "quality":                "#8b5cf6",
    "low_volatility":         "#f59e0b",
    "financial_strength":     "#6b7280",
    "investment":             "#06b6d4",
    "earnings_revisions":     "#ec4899",
    "garp":                   "#10b981",
    "balance_sheet_strength": "#60a5fa",
    "graham_value":           "#a78bfa",
    "downside_protection":    "#fbbf24",
    "earnings_quality":       "#f97316",
    "shareholder_yield":      "#34d399",
    "capital_efficiency":     "#ef4444",
    "distress_risk":          "#94a3b8",
}

METRIC_HELP = {
    "composite_score": (
        "Single number from 0–100 that blends how this stock ranks on value, quality, momentum, "
        "and other factors vs similar companies. Higher = the model likes it more overall; "
        "70+ is the default “good buy” bar in this app. Only factors with available data are "
        "included; check Factor Coverage to see how complete the score is."
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
        "when earnings are negative, and ignores growth and balance-sheet quality. See Lynch PEG, "
        "Graham Ratio, and the Value factor for richer valuation context."
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

FACTOR_HELP = {
    "value": (
        "How cheap the stock looks vs peers using earnings, book value, and cash flow. "
        "Higher percentile = relatively more “bang for your buck.”"
    ),
    "momentum": (
        "How much the share price rose over the past year (excluding the last month). "
        "Higher percentile = stronger recent price trend (“winners keep winning” idea)."
    ),
    "quality": (
        "How profitable and efficient the business is (margins, returns on assets/equity). "
        "Higher percentile = stronger underlying business quality."
    ),
    "low_volatility": (
        "How steady the price has been—less day‑to‑day and year‑to‑year swinging. "
        "Higher percentile = historically calmer, lower‑volatility stock."
    ),
    "investment": (
        "Whether the company is rapidly growing its asset base (plants, acquisitions, etc.). "
        "Research often favors companies that are not over‑investing; higher percentile = "
        "less aggressive asset growth vs peers."
    ),
    "earnings_revisions": (
        "Whether analyst estimates and sentiment have been moving up or down. "
        "Higher percentile = more positive revision trend."
    ),
    "financial_strength": (
        "Piotroski F‑Score style checklist (profitability, leverage, liquidity). "
        "Higher percentile = healthier financial signals vs peers."
    ),
    "garp": (
        "Growth at a reasonable price—same Lynch PEG family as the headline Lynch PEG metric, "
        "ranked vs other stocks. Higher percentile = better growth‑for‑price vs peers."
    ),
    "balance_sheet_strength": (
        "Cash vs debt and how leveraged the company is. Higher percentile = more cash cushion "
        "and less debt stress relative to peers."
    ),
    "graham_value": (
        "Classic Benjamin Graham checks (earnings, book value, liquidity). "
        "Higher percentile = more attractive on these old‑school value measures vs peers."
    ),
    "downside_protection": (
        "How severe past price drops and “bad day” volatility have been (Howard Marks–style). "
        "Higher percentile = historically smaller drawdowns and gentler downside moves."
    ),
    "earnings_quality": (
        "Whether reported profits are backed by actual cash (Sloan accruals anomaly). "
        "Lower accruals = earnings closer to cash flow = higher quality signal. "
        "Higher percentile = more cash-backed, less accounting-driven earnings vs peers."
    ),
    "shareholder_yield": (
        "Total cash returned to investors as dividends plus net stock buybacks, divided by market cap "
        "(Meb Faber's shareholder yield). A broader measure than dividend yield alone, and harder "
        "to game. Higher percentile = more capital returned per dollar of market value vs peers."
    ),
    "capital_efficiency": (
        "Return on invested capital (ROIC) — Greenblatt Magic Formula's second leg. "
        "Measures how much pre-tax operating profit the business generates per dollar of capital employed. "
        "Higher percentile = the business compounds capital more efficiently vs peers."
    ),
    "distress_risk": (
        "Altman Z-Score: a classic 5-ratio model (working capital, retained earnings, EBIT, "
        "market value vs liabilities, and asset turnover) predicting financial distress. "
        "Higher percentile = lower distress risk vs peers (Z above ~3 is generally considered safe)."
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


def score_label_and_color(score: float | None) -> tuple[str, str]:
    if score is None:
        return "N/A", "#6b7280"
    if score >= 85:
        return "Excellent", "#059669"
    if score >= 70:
        return "Strong", "#10b981"
    if score >= 55:
        return "Good", "#14b8a6"
    if score >= 45:
        return "Average", "#6b7280"
    if score >= 30:
        return "Below Average", "#f59e0b"
    return "Poor", "#ef4444"


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

        /* Columns: prevent overflow in narrow slots */
        [data-testid="stColumn"] { min-width: 0; }

        /* Sidebar */
        [data-testid="stSidebar"] { background-color: white; }

        /* Hide footer */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Card components
# ──────────────────────────────────────────────────────────────────────────────

def render_company_header(analysis: dict) -> None:
    ticker = analysis.get("ticker", "")
    name = analysis.get("name") or ticker
    exchange = analysis.get("exchange") or ""
    sector = analysis.get("sector") or ""
    industry = analysis.get("industry") or ""
    market_cap = analysis.get("market_cap")
    price = analysis.get("price")
    price_html = (
        f'<span style="font-size:1.25rem;font-weight:700;color:#1e3a5f;white-space:nowrap;">'
        f"${price:,.2f}</span>"
        if price
        else ""
    )

    left, right = st.columns([3, 2])
    with left:
        exchange_html = (
            f"<span style='color:#d1d5db;'>&nbsp;|&nbsp;</span>"
            f"<span style='font-size:0.88rem;color:#9ca3af;'>{exchange}</span>"
            if exchange else ""
        )
        st.markdown(
            f"""
            <div style="padding:0.05rem 0 0.1rem;">
                <div style="display:flex;align-items:baseline;gap:0.6rem;flex-wrap:wrap;line-height:1.05;">
                    <span style="font-size:1.55rem;font-weight:800;color:#1e3a5f;">{ticker}</span>
                    {price_html}
                </div>
                <span style="font-size:0.82rem;color:#6b7280;">{name}</span>{exchange_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        parts = []
        if sector:
            parts.append(
                f'<div><div style="font-size:0.68rem;color:#9ca3af;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;">Sector</div>'
                f'<div style="font-size:0.88rem;color:#374151;font-weight:500;">{sector}</div></div>'
            )
        if industry:
            parts.append(
                f'<div><div style="font-size:0.68rem;color:#9ca3af;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;">Industry</div>'
                f'<div style="font-size:0.88rem;color:#374151;font-weight:500;">{industry}</div></div>'
            )
        if market_cap:
            parts.append(
                f'<div><div style="font-size:0.68rem;color:#9ca3af;font-weight:600;'
                f'text-transform:uppercase;letter-spacing:0.06em;">Market Cap</div>'
                f'<div style="font-size:0.88rem;color:#374151;font-weight:700;">'
                f'{fmt_large_number(market_cap)}</div></div>'
            )
        if parts:
            st.markdown(
                f'<div style="display:flex;gap:1.75rem;justify-content:flex-end;'
                f'align-items:flex-start;padding:0.15rem 0 0.25rem;">{"".join(parts)}</div>',
                unsafe_allow_html=True,
            )


def _arc_gauge_html(
    score: float | None,
    label: str,
    label_color: str,
    percentile_rank: int | None,
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

    # Teal fill arc proportional to score
    fill_svg = ""
    if score is not None and score > 0:
        fd = score * span_deg / 100.0
        fe = pt(start_deg + fd)
        large = 1 if fd > 180 else 0
        fill_svg = (
            f'<path d="M {s[0]:.1f} {s[1]:.1f} A {r} {r} 0 {large} 1 {fe[0]:.1f} {fe[1]:.1f}" '
            f'fill="none" stroke="#14b8a6" stroke-width="{sw}" stroke-linecap="round"/>'
        )

    score_txt = f"{score:.0f}" if score is not None else "N/A"
    font_sz = 36 if score is not None else 22

    pct_html = ""
    if percentile_rank is not None:
        pct_html = (
            '<div style="background:#f0fdfa;border-radius:8px;padding:0.3rem 0.6rem;'
            'display:flex;align-items:center;gap:0.4rem;margin-top:0.25rem;text-align:left;">'
            '<span style="font-size:0.85rem;">📈</span>'
            '<div>'
            '<div style="font-size:0.62rem;color:#6b7280;font-weight:500;">Percentile Rank</div>'
            f'<div style="font-size:0.9rem;font-weight:700;color:#0d9488;">{ordinal(percentile_rank)}</div>'
            '<div style="font-size:0.62rem;color:#9ca3af;">vs. Global Universe</div>'
            '</div>'
            '</div>'
        )

    # viewBox clips the empty gap at the bottom (arc endpoints sit at y≈122, cut at y=135)
    return (
        '<div style="text-align:center;padding:0.3rem 0.25rem 0.15rem;">'
        '<svg width="100%" viewBox="5 5 150 130" '
        'style="max-width:130px;display:block;margin:0 auto;" '
        'aria-label="Composite score gauge">'
        f'<path d="{bg_path}" fill="none" stroke="#e8ecef" '
        f'stroke-width="{sw}" stroke-linecap="round"/>'
        f'{fill_svg}'
        f'<text x="{cx}" y="78" text-anchor="middle" dominant-baseline="middle" '
        f'font-size="{font_sz}" font-weight="800" fill="#1e3a5f" '
        f'font-family="Inter, Arial, sans-serif">{score_txt}</text>'
        f'<text x="{cx}" y="100" text-anchor="middle" font-size="13" fill="#9ca3af" '
        f'font-family="Inter, Arial, sans-serif">/ 100</text>'
        '</svg>'
        f'<div style="margin-top:0.15rem;">'
        f'<div style="font-size:0.95rem;font-weight:700;color:{label_color};">{label}</div>'
        '<div style="font-size:0.67rem;color:#9ca3af;margin-top:1px;">vs. Global Universe</div>'
        '</div>'
        f'{pct_html}'
        '</div>'
    )


def render_composite_card(analysis: dict) -> None:
    composite = analysis.get("composite")
    label, label_color = score_label_and_color(composite)
    percentile_rank = int(round(composite)) if composite is not None else None

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:0.83rem;font-weight:600;color:#6b7280;'
            'text-align:center;margin-bottom:0.1rem;">Composite Score</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            _arc_gauge_html(composite, label, label_color, percentile_rank),
            unsafe_allow_html=True,
        )


def _factor_group_html(
    group_label: str,
    accent: str,
    factor_keys: list[str],
    breakdown: dict,
    first: bool = False,
) -> str:
    top_margin = "0" if first else "8px"
    header = (
        f'<div style="font-size:0.58rem;font-weight:700;color:{accent};text-transform:uppercase;'
        f'letter-spacing:0.07em;margin:{top_margin} 0 3px;padding-bottom:2px;'
        f'border-bottom:1px solid {accent}22;">{group_label}</div>'
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
            f'<div style="display:flex;align-items:center;gap:4px;margin:2px 0;min-width:0;">'
            f'<div style="width:5px;height:5px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
            f'<div style="flex:1 1 0;min-width:0;font-size:0.64rem;color:#374151;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;line-height:1.15;">'
            f"{short_label}</div>"
            f'<div style="width:34px;flex-shrink:0;background:#f3f4f6;border-radius:3px;height:4px;overflow:hidden;">'
            f'<div style="width:{bar_w:.0f}%;height:4px;border-radius:3px;background:{color};"></div>'
            f"</div>"
            f'<div style="width:26px;font-size:0.62rem;font-weight:700;color:{color};'
            f'text-align:right;flex-shrink:0;">{pct_text}</div>'
            f"</div>"
        )
    return header + "\n".join(rows)


def render_factor_scorecard_card(analysis: dict) -> None:
    breakdown = analysis.get("factor_breakdown", {})

    with st.container(border=True):
        st.markdown(
            '<div style="display:flex;justify-content:space-between;align-items:center;'
            'margin-bottom:0.35rem;">'
            '<span style="font-size:0.88rem;font-weight:700;color:#1e3a5f;">Factor Scorecard</span>'
            '<span style="font-size:0.58rem;font-weight:600;color:#9ca3af;'
            'text-transform:uppercase;letter-spacing:0.05em;">Percentile Rank</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # 4-column CSS grid — one column per conceptual group, each only 3-4 rows tall
        cols_html = ""
        for i, (lbl, acc, keys) in enumerate(FACTOR_SCORECARD_GROUPS):
            cols_html += f"<div>{_factor_group_html(lbl, acc, keys, breakdown, first=True)}</div>"

        st.markdown(
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);column-gap:10px;">'
            f"{cols_html}"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_price_history_card(ticker: str) -> None:
    with st.container(border=True):
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
            margin=dict(l=10, r=10, t=16, b=8),
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
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _analyst_target_range_html(analyst: dict) -> str:
    parts: list[str] = []
    if analyst.get("target_low") is not None:
        parts.append(f"Low <b>${analyst['target_low']:,.0f}</b>")
    if analyst.get("target_mean") is not None:
        parts.append(f"Mean <b>${analyst['target_mean']:,.0f}</b>")
    if analyst.get("target_high") is not None:
        parts.append(f"High <b>${analyst['target_high']:,.0f}</b>")
    if not parts:
        return ""
    return (
        f'<div style="font-size:0.68rem;color:#6b7280;line-height:1.3;margin-top:0.2rem;">'
        f"{' · '.join(parts)}</div>"
    )


def render_analyst_card(analysis: dict) -> None:
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

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:0.88rem;font-weight:700;color:#1e3a5f;margin-bottom:0.2rem;">'
            "Analyst Consensus</div>",
            unsafe_allow_html=True,
        )

        hero_col, pie_col = st.columns([1.1, 0.9], gap="small")

        with hero_col:
            st.markdown(
                f"""
                <div style="padding:0 0 0.1rem;">
                    <div style="margin-bottom:0.25rem;">
                        <span style="display:inline-block;background:{bg_color};border-radius:999px;
                            padding:0.18rem 0.85rem;">
                            <span style="font-size:1.2rem;font-weight:800;color:{txt_color};">
                                {consensus}</span>
                        </span>
                    </div>
                    <div style="font-size:1.45rem;font-weight:800;color:{upside_color};line-height:1.05;">
                        {upside_display} <span style="font-size:0.95rem;">{upside_arrow}</span>
                    </div>
                    <div style="font-size:0.64rem;font-weight:600;color:{upside_color};
                        text-transform:uppercase;letter-spacing:0.04em;margin-top:0.05rem;">
                        Implied upside</div>
                    {target_range_html}
                    <div style="font-size:0.64rem;color:#374151;margin-top:0.2rem;">
                        Upgrades <b>{upgrades}</b> · Downgrades <b>{downgrades}</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with pie_col:
            pie_fig = _analyst_recommendations_pie(analyst)
            if pie_fig is not None:
                st.plotly_chart(pie_fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.markdown(
                    '<div style="font-size:0.72rem;color:#9ca3af;padding:1.5rem 0;text-align:center;">'
                    "No rating mix</div>",
                    unsafe_allow_html=True,
                )
            if num_analysts:
                st.markdown(
                    f'<div style="font-size:0.68rem;color:#9ca3af;text-align:center;margin-top:-0.1rem;">'
                    f"{int(num_analysts)} analysts</div>",
                    unsafe_allow_html=True,
                )


def render_factor_radar_card(analysis: dict, ticker: str) -> None:
    breakdown = analysis.get("factor_breakdown", {})

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:0.88rem;font-weight:700;color:#1e3a5f;margin-bottom:0.15rem;">'
            "Factor Radar</div>",
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
            margin=dict(l=45, r=45, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


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
            f"upside ≥ {thresholds['implied_upside_min_pct']}%)"
        )

    # Company header card
    with st.container(border=True):
        render_company_header(analysis)

    st.markdown("<div style='margin-top:0.35rem;'></div>", unsafe_allow_html=True)

    # Row 1: composite, factor scorecard, analyst consensus (with ratings pie)
    row1_left, row1_mid, row1_right = st.columns([1.2, 3.6, 2.6], gap="small")
    with row1_left:
        render_composite_card(analysis)
    with row1_mid:
        render_factor_scorecard_card(analysis)
    with row1_right:
        render_analyst_card(analysis)

    st.markdown("<div style='margin-top:0.35rem;'></div>", unsafe_allow_html=True)

    # Row 2: price history and factor radar (wider now that recommendations are merged)
    row2_left, row2_right = st.columns([3.35, 2.15], gap="small")
    with row2_left:
        render_price_history_card(ticker)
    with row2_right:
        render_factor_radar_card(analysis, ticker)

    with st.expander("Raw factor values"):
        st.json(analysis.get("factors_raw", {}))


def render_universe_rankings(config: dict) -> None:
    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        st.warning("No universe snapshot found. Run the daily job to build one.")
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
        st.markdown("**Good-buy thresholds**")
        thresholds = get_thresholds(config)
        st.write(f"Composite ≥ {thresholds['composite_min']}")
        st.write(f"Upside ≥ {thresholds['implied_upside_min_pct']}%")
        st.markdown("---")
        st.markdown("**Factor weights**")
        for k, v in get_factor_weights(config).items():
            st.write(f"{FACTOR_LABELS.get(k, k)}: {v:.0%}")

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
