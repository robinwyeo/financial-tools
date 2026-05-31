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

PRICE_HISTORY_RANGES: dict[str, str] = {
    "Last month": "1mo",
    "Last 3 months": "3mo",
    "Last 6 months": "6mo",
    "YTD": "ytd",
    "1 year": "1y",
    "2 year": "2y",
    "5 year": "5y",
    "All-time": "max",
}
DEFAULT_PRICE_HISTORY_RANGE = "2 year"

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
        "How far the average Wall Street 12‑month price target sits above or below today’s price, "
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
        "Compares Benjamin Graham’s estimated fair price to today’s price. Above 1.0 means the "
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
    "etf_category": "Broad type of fund (e.g. large-cap equity, bond) from the provider’s classification.",
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
        "(Meb Faber’s shareholder yield). A broader measure than dividend yield alone, and harder "
        "to game. Higher percentile = more capital returned per dollar of market value vs peers."
    ),
    "capital_efficiency": (
        "Return on invested capital (ROIC) — Greenblatt Magic Formula’s second leg. "
        "Measures how much pre-tax operating profit the business generates per dollar of capital employed. "
        "Higher percentile = the business compounds capital more efficiently vs peers."
    ),
    "distress_risk": (
        "Altman Z-Score: a classic 5-ratio model (working capital, retained earnings, EBIT, "
        "market value vs liabilities, and asset turnover) predicting financial distress. "
        "Higher percentile = lower distress risk vs peers (Z above ~3 is generally considered safe)."
    ),
}


def render_price_history(ticker: str) -> None:
    """Price chart with selectable lookback; default is 2 year."""
    range_labels = list(PRICE_HISTORY_RANGES.keys())
    default_idx = range_labels.index(DEFAULT_PRICE_HISTORY_RANGE)

    header_col, range_col = st.columns([4, 1])
    with header_col:
        st.markdown("### Price History")
    with range_col:
        selected_range = st.selectbox(
            "Timeframe",
            options=range_labels,
            index=default_idx,
            label_visibility="collapsed",
        )

    period = PRICE_HISTORY_RANGES[selected_range]
    hist = fetch_price_history(ticker, period=period)
    if hist.empty:
        st.info("No price history available for this timeframe.")
        return

    fig = px.line(
        hist.reset_index(),
        x="Date",
        y="Close",
        title=f"{ticker} — {selected_range}",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_factor_gauge(label: str, percentile: float | None, help: str | None = None) -> None:
    import math

    unavailable = percentile is None or (isinstance(percentile, float) and math.isnan(percentile))
    if unavailable:
        st.metric(label, "N/A", help=help)
        return
    pct = float(percentile)
    st.metric(label, f"{pct:.0f}th percentile", help=help)
    st.progress(min(max(pct / 100, 0.0), 1.0))


def render_etf_view(ticker: str) -> None:
    info = fetch_etf_info(ticker)
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
    factor_coverage = analysis.get("factor_coverage_pct")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Composite Score",
        f"{composite:.1f}" if composite is not None else "N/A",
        help=METRIC_HELP["composite_score"],
    )
    c2.metric(
        "Factor Coverage",
        f"{factor_coverage:.0f}%" if factor_coverage is not None else "N/A",
        help="Share of composite weight backed by computed factor percentiles (not imputed).",
    )
    c3.metric(
        "Price",
        f"${analysis.get('price', 0):,.2f}" if analysis.get("price") else "N/A",
        help=METRIC_HELP["price"],
    )
    c4.metric(
        "Analyst Upside",
        f"{analyst.get('implied_upside_pct', 0):.1f}%" if analyst.get("implied_upside_pct") is not None else "N/A",
        help=METRIC_HELP["analyst_upside"],
    )
    c5.metric(
        "Consensus",
        analyst.get("consensus_label", "N/A"),
        help=METRIC_HELP["consensus"],
    )

    factors_raw = analysis.get("factors_raw", {})
    trailing_pe = factors_raw.get("trailing_pe")
    peg = factors_raw.get("dividend_adjusted_peg") or factors_raw.get("peg_ratio")
    graham_ratio = factors_raw.get("graham_ratio")

    c1.metric(
        "Trailing P/E",
        f"{trailing_pe:.1f}x" if trailing_pe is not None and trailing_pe > 0 else "N/A",
        help=METRIC_HELP["trailing_pe"],
    )
    c2.metric(
        "Lynch PEG",
        f"{peg:.2f}" if peg is not None else "N/A",
        help=METRIC_HELP["lynch_peg"],
    )
    c3.metric(
        "Graham Ratio",
        f"{graham_ratio:.2f}" if graham_ratio is not None else "N/A",
        help=METRIC_HELP["graham_ratio"],
    )

    data_warnings = analysis.get("data_warnings") or []
    if data_warnings:
        with st.expander("Data warnings", expanded=False):
            for warning in data_warnings:
                st.warning(warning)

    thresholds = get_thresholds(config)
    if analysis.get("is_good_buy"):
        st.success(
            f"Meets good-buy criteria (composite ≥ {thresholds['composite_min']}, "
            f"upside ≥ {thresholds['implied_upside_min_pct']}%)"
        )

    render_price_history(ticker)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### Factor Scorecard")
        breakdown = analysis.get("factor_breakdown", {})
        for family, label in FACTOR_LABELS.items():
            fb = breakdown.get(family, {})
            render_factor_gauge(label, fb.get("percentile"), help=FACTOR_HELP.get(family))

        # Radar chart
        import math

        families = list(FACTOR_LABELS.keys())
        raw_values = [breakdown.get(f, {}).get("percentile") for f in families]
        available = [
            float(v)
            for v in raw_values
            if v is not None and not (isinstance(v, float) and math.isnan(v))
        ]
        fill = sum(available) / len(available) if available else 50.0
        values = [
            float(v)
            if v is not None and not (isinstance(v, float) and math.isnan(v))
            else fill
            for v in raw_values
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
