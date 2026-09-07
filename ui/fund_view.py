"""ETF / mutual-fund dashboard view."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from core.data import currency_symbol, fetch_etf_holdings
from ui.charts import render_factor_radar_card, render_price_history_card
from ui.constants import (
    FUND_FACTOR_HELP,
    FUND_SCORECARD_GROUPS,
    GAUGE_MAX_WIDTH,
    RADAR_FUND_FACTOR_LABELS,
    SECURITY_TYPE_BADGES,
    SHORT_FUND_FACTOR_LABELS,
)
from ui.formatting import (
    _arc_gauge_html,
    _format_snapshot_date,
    fmt_large_number,
    gauge_score_color,
    gauge_score_label,
)
from ui.layout import _card_shell, _dashboard_row_anchor, inject_equal_height_js
from ui.state import apply_fund_snapshot, score_fund_cached
from ui.stock_view import render_factor_scorecard_card

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
        analysis = score_fund_cached(ticker, config)
        if scored_fund_universe is not None and not scored_fund_universe.empty:
            analysis = apply_fund_snapshot(analysis, scored_fund_universe, ticker)

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
