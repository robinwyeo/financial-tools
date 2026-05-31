"""Streamlit dashboard for stock metrics and analyst aggregation."""

from __future__ import annotations

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
}


def render_factor_gauge(label: str, percentile: float | None) -> None:
    import math

    unavailable = percentile is None or (isinstance(percentile, float) and math.isnan(percentile))
    pct = 50.0 if unavailable else float(percentile)
    color = "#2ecc71" if pct >= 70 else "#f39c12" if pct >= 40 else "#e74c3c"
    suffix = " _(insufficient data)_" if unavailable else f" — {pct:.0f}th percentile"
    st.markdown(
        f"**{label}**{suffix} "
        f'<span style="color:{color}">{"●" if pct >= 70 else "○"}</span>',
        unsafe_allow_html=True,
    )
    st.progress(min(max(pct / 100, 0.0), 1.0))


def render_etf_view(ticker: str) -> None:
    info = fetch_etf_info(ticker)
    st.subheader(f"{info.get('name') or ticker} (ETF)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${info.get('current_price', 0):,.2f}" if info.get("current_price") else "N/A")
    c2.metric("Expense Ratio", f"{(info.get('expense_ratio') or 0)*100:.2f}%" if info.get("expense_ratio") else "N/A")
    c3.metric("Category", info.get("category") or "N/A")
    c4.metric("Yield", f"{(info.get('yield') or 0)*100:.2f}%" if info.get("yield") else "N/A")

    st.info("ETFs are excluded from empirical factor scoring. Showing basic fund info only.")

    holdings = fetch_etf_holdings(ticker)
    if not holdings.empty:
        st.subheader("Top Holdings")
        st.dataframe(holdings, use_container_width=True)

    hist = fetch_price_history(ticker, period="1y")
    if not hist.empty:
        fig = px.line(hist.reset_index(), x="Date", y="Close", title=f"{ticker} — 1Y Price")
        st.plotly_chart(fig, use_container_width=True)

    if info.get("description"):
        with st.expander("Description"):
            st.write(info["description"])


def render_stock_view(ticker: str, config: dict) -> None:
    with st.spinner(f"Analyzing {ticker}..."):
        analysis = score_ticker(ticker, config)

    if analysis.get("warning"):
        st.warning(analysis["warning"])

    name = analysis.get("name") or ticker
    st.subheader(f"{name} ({ticker})")
    st.caption(f"{analysis.get('sector')} · {analysis.get('industry')}")

    analyst = analysis.get("analyst", {})
    composite = analysis.get("composite")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Composite Score", f"{composite:.1f}" if composite is not None else "N/A")
    c2.metric("Price", f"${analysis.get('price', 0):,.2f}" if analysis.get("price") else "N/A")
    c3.metric("Analyst Upside", f"{analyst.get('implied_upside_pct', 0):.1f}%" if analyst.get("implied_upside_pct") is not None else "N/A")
    c4.metric("Consensus", analyst.get("consensus_label", "N/A"))

    thresholds = get_thresholds(config)
    if analysis.get("is_good_buy"):
        st.success(
            f"Meets good-buy criteria (composite ≥ {thresholds['composite_min']}, "
            f"upside ≥ {thresholds['implied_upside_min_pct']}%)"
        )

    factors_raw = analysis.get("factors_raw", {})
    book_cols = st.columns(2)
    peg = factors_raw.get("dividend_adjusted_peg") or factors_raw.get("peg_ratio")
    book_cols[0].metric(
        "Lynch PEG",
        f"{peg:.2f}" if peg is not None else "N/A",
        help="Dividend-adjusted PEG (growth% + yield%) / P/E; Lynch: >2 is attractive",
    )
    graham_ratio = factors_raw.get("graham_ratio")
    book_cols[1].metric(
        "Graham Ratio",
        f"{graham_ratio:.2f}" if graham_ratio is not None else "N/A",
        help="Graham fair value / price; >1 means below Graham Number",
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Factor Scorecard")
        breakdown = analysis.get("factor_breakdown", {})
        for family, label in FACTOR_LABELS.items():
            fb = breakdown.get(family, {})
            render_factor_gauge(label, fb.get("percentile"))

        # Radar chart
        import math

        families = list(FACTOR_LABELS.keys())
        values = [
            50.0 if (
                (v := breakdown.get(f, {}).get("percentile")) is None
                or (isinstance(v, float) and math.isnan(v))
            ) else float(v)
            for f in families
        ]
        fig = go.Figure(
            data=go.Scatterpolar(
                r=values + [values[0]],
                theta=[FACTOR_LABELS[f] for f in families] + [FACTOR_LABELS[families[0]]],
                fill="toself",
                name=ticker,
            )
        )
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Analyst Recommendations")
        total = (analyst.get("buy_count") or 0) + (analyst.get("hold_count") or 0) + (analyst.get("sell_count") or 0)
        if total > 0:
            fig = px.pie(
                values=[analyst.get("buy_count", 0), analyst.get("hold_count", 0), analyst.get("sell_count", 0)],
                names=["Buy", "Hold", "Sell"],
                title="Recommendation Distribution",
                color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write(f"Consensus: **{analyst.get('consensus_label', 'Unknown')}**")

        t1, t2, t3 = st.columns(3)
        t1.metric("Target Low", f"${analyst.get('target_low', 0):,.2f}" if analyst.get("target_low") else "N/A")
        t2.metric("Target Mean", f"${analyst.get('target_mean', 0):,.2f}" if analyst.get("target_mean") else "N/A")
        t3.metric("Target High", f"${analyst.get('target_high', 0):,.2f}" if analyst.get("target_high") else "N/A")

        st.metric("Recent Upgrades / Downgrades", f"{analyst.get('recent_upgrades', 0)} / {analyst.get('recent_downgrades', 0)}")

        actions = analyst.get("recent_actions", [])
        if actions:
            st.markdown("**Recent Actions**")
            st.dataframe(pd.DataFrame(actions), use_container_width=True, hide_index=True)

    hist = fetch_price_history(ticker, period="2y")
    if not hist.empty:
        st.markdown("### Price History")
        fig = px.line(hist.reset_index(), x="Date", y="Close", title=f"{ticker} — 2Y Price")
        st.plotly_chart(fig, use_container_width=True)

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


def main() -> None:
    config = load_config()

    st.title("Stock Metrics & Analyst Aggregation")
    st.caption("Empirical factor scores + aggregated analyst recommendations (free data sources)")

    with st.sidebar:
        st.header("Settings")
        ticker = st.text_input("Ticker", value="AAPL").upper().strip()
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
        st.info("Enter a ticker symbol in the sidebar.")
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
