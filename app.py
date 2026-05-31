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
    """Smooth red → orange → yellow gradient based on percentile rank (0=red, 100=yellow)."""
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "#d1d5db"
    p = max(0.0, min(100.0, float(pct))) / 100.0
    # Anchor colors: red (0), orange (0.5), yellow (1.0)
    if p <= 0.5:
        t = p * 2
        r = int(0xef + t * (0xf9 - 0xef))
        g = int(0x44 + t * (0x73 - 0x44))
        b = int(0x44 + t * (0x16 - 0x44))
    else:
        t = (p - 0.5) * 2
        r = int(0xf9 + t * (0xea - 0xf9))
        g = int(0x73 + t * (0xb3 - 0x73))
        b = int(0x16 + t * (0x08 - 0x16))
    return f"#{r:02x}{g:02x}{b:02x}"


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
        .main .block-container { padding-top: 0.75rem; padding-bottom: 2rem; }

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

    left, right = st.columns([3, 2])
    with left:
        exchange_html = (
            f"<span style='color:#d1d5db;'>&nbsp;|&nbsp;</span>"
            f"<span style='font-size:0.88rem;color:#9ca3af;'>{exchange}</span>"
            if exchange else ""
        )
        st.markdown(
            f"""
            <div style="padding:0.15rem 0 0.25rem;">
                <span style="font-size:2rem;font-weight:800;color:#1e3a5f;line-height:1.1;">{ticker}</span><br>
                <span style="font-size:0.92rem;color:#6b7280;">{name}</span>{exchange_html}
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


def _make_composite_gauge(score: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            gauge={
                "axis": {"range": [0, 100], "visible": False},
                "bar": {"color": "#14b8a6", "thickness": 0.72},
                "bgcolor": "#e8ecef",
                "borderwidth": 0,
                "steps": [],
            },
            number={
                "font": {"size": 62, "color": "#1e3a5f"},
                "valueformat": ".0f",
            },
            domain={"x": [0, 1], "y": [0.12, 1]},
        )
    )
    fig.add_annotation(
        text="/ 100",
        x=0.5,
        y=0.17,
        font=dict(size=16, color="#9ca3af"),
        showarrow=False,
        xanchor="center",
    )
    fig.update_layout(
        height=215,
        margin=dict(l=20, r=20, t=30, b=5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def render_composite_card(analysis: dict) -> None:
    composite = analysis.get("composite")
    label, label_color = score_label_and_color(composite)
    percentile_rank = int(round(composite)) if composite is not None else None

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:0.83rem;font-weight:600;color:#6b7280;text-align:center;'
            'margin-bottom:-0.25rem;">Composite Score</div>',
            unsafe_allow_html=True,
        )

        if composite is not None:
            st.plotly_chart(
                _make_composite_gauge(composite),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        else:
            st.markdown(
                '<div style="text-align:center;font-size:3rem;font-weight:700;'
                'color:#9ca3af;padding:1.5rem 0;">N/A</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style="text-align:center;margin-top:-0.75rem;margin-bottom:0.75rem;">
                <div style="font-size:1.1rem;font-weight:700;color:{label_color};">{label}</div>
                <div style="font-size:0.73rem;color:#9ca3af;margin-top:2px;">vs. Global Universe</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if percentile_rank is not None:
            st.markdown(
                f"""
                <div style="background:#f0fdfa;border-radius:8px;padding:0.55rem 0.8rem;
                    display:flex;align-items:center;gap:0.65rem;">
                    <span style="font-size:1rem;">📈</span>
                    <div>
                        <div style="font-size:0.68rem;color:#6b7280;font-weight:500;">Percentile Rank</div>
                        <div style="font-size:1.05rem;font-weight:700;color:#0d9488;">{ordinal(percentile_rank)}</div>
                        <div style="font-size:0.68rem;color:#9ca3af;">vs. Global Universe</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_factor_scorecard_card(analysis: dict) -> None:
    breakdown = analysis.get("factor_breakdown", {})

    with st.container(border=True):
        st.markdown(
            """
            <div style="display:flex;justify-content:space-between;align-items:center;
                margin-bottom:0.5rem;">
                <span style="font-size:0.92rem;font-weight:700;color:#1e3a5f;">Factor Scorecard</span>
                <span style="font-size:0.68rem;font-weight:600;color:#9ca3af;
                    text-transform:uppercase;letter-spacing:0.05em;">Percentile Rank</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        rows = []
        for family, short_label in SHORT_FACTOR_LABELS.items():
            fb = breakdown.get(family, {})
            pct = fb.get("percentile")
            color = percentile_color(pct)

            if pct is None or (isinstance(pct, float) and math.isnan(pct)):
                bar_w = 0
                pct_text = "N/A"
            else:
                bar_w = min(max(float(pct), 0), 100)
                pct_text = ordinal(int(round(float(pct))))

            rows.append(
                f'<div style="display:flex;align-items:center;gap:9px;margin:4px 0;">'
                f'<div style="width:9px;height:9px;border-radius:50%;background:{color};flex-shrink:0;"></div>'
                f'<div style="width:130px;font-size:0.78rem;color:#374151;flex-shrink:0;">{short_label}</div>'
                f'<div style="flex:1;background:#f3f4f6;border-radius:4px;height:8px;overflow:hidden;">'
                f'<div style="width:{bar_w:.0f}%;height:8px;border-radius:4px;background:{color};"></div>'
                f'</div>'
                f'<div style="width:36px;font-size:0.78rem;font-weight:600;color:{color};'
                f'text-align:right;flex-shrink:0;">{pct_text}</div>'
                f'</div>'
            )

        st.markdown("\n".join(rows), unsafe_allow_html=True)


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
            height=290,
            margin=dict(l=10, r=10, t=20, b=10),
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


def render_analyst_card(analysis: dict) -> None:
    analyst = analysis.get("analyst", {})
    factors_raw = analysis.get("factors_raw", {})

    consensus = analyst.get("consensus_label", "N/A")
    target_mean = analyst.get("target_mean")
    implied_upside = analyst.get("implied_upside_pct")
    num_analysts = analyst.get("num_analysts")
    price = analysis.get("price")

    market_cap = analysis.get("market_cap")
    dividend_yield = analysis.get("dividend_yield")
    trailing_pe = factors_raw.get("trailing_pe")
    wk52_high = analysis.get("fifty_two_week_high")
    wk52_low = analysis.get("fifty_two_week_low")

    txt_color, bg_color = consensus_style(consensus)

    with st.container(border=True):
        st.markdown(
            '<div style="font-size:0.92rem;font-weight:700;color:#1e3a5f;margin-bottom:0.65rem;">'
            "Analyst Consensus</div>",
            unsafe_allow_html=True,
        )

        # Consensus pill
        st.markdown(
            f"""
            <div style="text-align:center;margin-bottom:0.9rem;">
                <div style="display:inline-block;background:{bg_color};border-radius:999px;
                    padding:0.35rem 1.8rem;">
                    <span style="font-size:2rem;font-weight:700;color:{txt_color};">{consensus}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Average price target
        if target_mean is not None:
            price_str = f"vs. Current Price ${price:,.2f}" if price else ""
            st.markdown(
                f"""
                <div style="text-align:center;margin-bottom:0.4rem;">
                    <div style="font-size:0.72rem;color:#9ca3af;font-weight:500;">Average Price Target</div>
                    <div style="font-size:1.85rem;font-weight:700;color:#1e3a5f;">${target_mean:,.2f}</div>
                    <div style="font-size:0.76rem;color:#6b7280;">{price_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Implied upside
        if implied_upside is not None:
            upside_color = "#10b981" if implied_upside >= 0 else "#ef4444"
            arrow = "↗" if implied_upside >= 0 else "↘"
            analysts_note = f"Based on {int(num_analysts)} analysts" if num_analysts else ""
            st.markdown(
                f"""
                <div style="text-align:center;margin-bottom:0.9rem;">
                    <span style="font-size:1.55rem;font-weight:700;color:{upside_color};">
                        {implied_upside:+.0f}%&nbsp;{arrow}
                    </span><br>
                    <span style="font-size:0.8rem;font-weight:600;color:{upside_color};">Implied Upside</span><br>
                    <span style="font-size:0.7rem;color:#9ca3af;">{analysts_note}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Quick stats table
        st.markdown(
            '<hr style="border:none;border-top:1px solid #f3f4f6;margin:0.4rem 0 0.5rem;">',
            unsafe_allow_html=True,
        )

        stats: list[tuple[str, str, str]] = []
        if wk52_high is not None and wk52_low is not None:
            stats.append(("📊", "52-Week Range", f"${wk52_low:,.2f} – ${wk52_high:,.2f}"))
        if market_cap:
            stats.append(("📈", "Market Cap", fmt_large_number(market_cap)))
        if trailing_pe and trailing_pe > 0:
            stats.append(("💲", "P/E (TTM)", f"{trailing_pe:.2f}"))
        if dividend_yield and dividend_yield > 0:
            stats.append(("💰", "Dividend Yield", f"{dividend_yield * 100:.2f}%"))

        if stats:
            stat_rows = []
            for icon, stat_label, stat_val in stats:
                stat_rows.append(
                    f'<div style="display:flex;align-items:center;gap:0.55rem;padding:0.3rem 0;'
                    f'border-bottom:1px solid #f9fafb;">'
                    f'<span style="font-size:0.88rem;">{icon}</span>'
                    f'<span style="font-size:0.78rem;color:#6b7280;flex:1;">{stat_label}</span>'
                    f'<span style="font-size:0.8rem;font-weight:600;color:#374151;">{stat_val}</span>'
                    f'</div>'
                )
            st.markdown("\n".join(stat_rows), unsafe_allow_html=True)


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

    st.markdown("<div style='margin-top:0.6rem;'></div>", unsafe_allow_html=True)

    # Three-column main layout
    col_left, col_mid, col_right = st.columns([2.2, 4.5, 2.8], gap="medium")

    with col_left:
        render_composite_card(analysis)

    with col_mid:
        render_factor_scorecard_card(analysis)
        st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
        render_price_history_card(ticker)

    with col_right:
        render_analyst_card(analysis)

    # ── Collapsible detail sections ──────────────────────────────────────────
    st.markdown("<div style='margin-top:0.75rem;'></div>", unsafe_allow_html=True)
    analyst = analysis.get("analyst", {})
    col_a, col_b = st.columns(2)

    with col_a:
        with st.expander("Analyst Recommendations", expanded=True):
            total = (
                analyst.get("buy_count", 0)
                + analyst.get("hold_count", 0)
                + analyst.get("sell_count", 0)
            )
            if total > 0:
                fig = px.pie(
                    values=[
                        analyst.get("buy_count", 0),
                        analyst.get("hold_count", 0),
                        analyst.get("sell_count", 0),
                    ],
                    names=["Buy", "Hold", "Sell"],
                    color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"],
                )
                fig.update_layout(height=260, margin=dict(l=0, r=0, t=20, b=0))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            else:
                st.write(f"Consensus: **{analyst.get('consensus_label', 'Unknown')}**")

            t1, t2, t3 = st.columns(3)
            t1.metric(
                "Target Low",
                f"${analyst.get('target_low', 0):,.2f}" if analyst.get("target_low") else "N/A",
                help=METRIC_HELP["target_low"],
            )
            t2.metric(
                "Target Mean",
                f"${analyst.get('target_mean', 0):,.2f}" if analyst.get("target_mean") else "N/A",
                help=METRIC_HELP["target_mean"],
            )
            t3.metric(
                "Target High",
                f"${analyst.get('target_high', 0):,.2f}" if analyst.get("target_high") else "N/A",
                help=METRIC_HELP["target_high"],
            )

            st.metric(
                "Recent Upgrades / Downgrades",
                f"{analyst.get('recent_upgrades', 0)} / {analyst.get('recent_downgrades', 0)}",
                help=METRIC_HELP["upgrades_downgrades"],
            )

            actions = analyst.get("recent_actions", [])
            if actions:
                st.markdown("**Recent Actions**")
                st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)

    with col_b:
        with st.expander("Factor Radar", expanded=True):
            breakdown = analysis.get("factor_breakdown", {})
            families = list(FACTOR_LABELS.keys())
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

            fig = go.Figure(
                go.Scatterpolar(
                    r=values + [values[0]],
                    theta=[FACTOR_LABELS[f] for f in families] + [FACTOR_LABELS[families[0]]],
                    fill="toself",
                    fillcolor="rgba(20, 184, 166, 0.2)",
                    line=dict(color="#14b8a6", width=2),
                    name=ticker,
                )
            )
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                height=360,
                margin=dict(l=50, r=50, t=50, b=50),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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
