"""Single-stock dashboard view."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from core.config import get_thresholds
from core.data import currency_symbol
from ui.charts import (
    _analyst_recommendations_pie,
    _plotly_chart,
    render_factor_radar_card,
    render_price_history_card,
)
from ui.constants import CHART_HEIGHT_ANALYST_PIE
from ui.constants import (
    FACTOR_HELP,
    FACTOR_SCORECARD_GROUPS,
    GAUGE_MAX_WIDTH,
    SHORT_FACTOR_LABELS,
)
from ui.formatting import (
    _arc_gauge_html,
    _factor_group_html,
    _format_snapshot_date,
    _overlay_badge_spans,
    _sparkline_svg,
    decision_card_html,
    fmt_large_number,
    gauge_score_color,
    gauge_score_label,
    valuation_card_html,
    consensus_style,
    _analyst_target_range_html,
)
from ui.layout import _card_shell, _dashboard_row_anchor, inject_equal_height_js
from ui.state import apply_stock_snapshot, score_ticker_cached

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
    sym = currency_symbol(analysis.get("currency"))

    ticker_e = html.escape(str(ticker))
    name_e = html.escape(str(name))
    exchange_e = html.escape(str(exchange)) if exchange else ""
    sector_e = html.escape(str(sector)) if sector else ""
    industry_e = html.escape(str(industry)) if industry else ""

    price_html = (
        f' <span style="font-size:1.25rem;font-weight:700;color:#1e3a5f;white-space:nowrap;">'
        f"{sym}{price:,.2f}</span>"
        if price
        else ""
    )
    exchange_html = (
        f' <span style="color:#d1d5db;">|</span> '
        f'<span style="font-size:0.88rem;color:#9ca3af;">{exchange_e}</span>'
        if exchange_e
        else ""
    )

    badge_spans = "".join(_overlay_badge_spans(analysis))
    left, right = st.columns([3, 2])
    with left:
        # Single-level markup: Streamlit strips nested <div>s and can leak closing tags as text.
        st.markdown(
            f'<div style="padding:0.05rem 0 0.1rem;line-height:1.35;">'
            f'<span style="font-size:1.55rem;font-weight:800;color:#1e3a5f;">{ticker_e}</span>'
            f"{price_html}<br>"
            f'<span style="font-size:0.82rem;color:#6b7280;">{name_e}</span>'
            f"{exchange_html}<br>"
            f"{badge_spans}"
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

    spark = analysis.get("eps_trend_sparkline") or []
    spark_html = ""
    if len(spark) >= 2:
        # Own markdown block — nested <div>s inside the gauge markup are stripped.
        spark_html = (
            '<div style="text-align:center;margin-top:0.2rem;">'
            '<span style="display:block;font-size:0.68rem;color:#6b7280;letter-spacing:0.04em;'
            'text-transform:uppercase;">FY1 consensus EPS (90d → now)</span>'
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
            + "</div></div></div>",
            unsafe_allow_html=True,
        )
        if spark_html:
            st.markdown(spark_html, unsafe_allow_html=True)


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


def render_analyst_card(analysis: dict, *, bordered: bool = True) -> None:
    analyst = analysis.get("analyst", {})

    consensus = analyst.get("consensus_label", "N/A")
    implied_upside = analyst.get("implied_upside_pct")
    num_analysts = analyst.get("num_analysts")

    txt_color, bg_color = consensus_style(consensus)
    upside_color = "#10b981" if (implied_upside or 0) >= 0 else "#ef4444"
    upside_arrow = "↗" if (implied_upside or 0) >= 0 else "↘"
    upside_display = f"{implied_upside:+.0f}%" if implied_upside is not None else "—"
    target_range_html = _analyst_target_range_html(analyst, analysis.get("currency"))
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


def render_decision_card(analysis: dict, *, bordered: bool = True) -> None:
    decision = analysis.get("decision") or {}
    if not decision:
        return
    html = decision_card_html(decision, currency=analysis.get("currency"))
    with _card_shell(bordered):
        st.markdown("**Decision**")
        st.markdown(html, unsafe_allow_html=True)
        flags = [f for f in (decision.get("flags") or analysis.get("value_trap_flags") or []) if f.get("triggered")]
        if flags:
            st.caption("Flags: " + ", ".join(f.get("code", "") for f in flags))
        dq = analysis.get("data_quality") or {}
        if dq.get("grade"):
            st.caption(f"Data quality grade {dq.get('grade')}")


def render_valuation_card(analysis: dict, *, bordered: bool = True) -> None:
    valuation = analysis.get("valuation") or {}
    if not valuation:
        return
    html = valuation_card_html(valuation, currency=analysis.get("currency"))
    with _card_shell(bordered):
        st.markdown("**Intrinsic value**")
        st.markdown(html, unsafe_allow_html=True)
        rel = valuation.get("relative") or {}
        if rel:
            with st.expander("Relative valuation vs own history"):
                st.json(rel)
        assumptions = valuation.get("assumptions") or {}
        if assumptions:
            with st.expander("Valuation assumptions"):
                st.json(assumptions)


def _gate_name_passed(gate) -> tuple[str, bool]:
    if isinstance(gate, dict):
        return str(gate.get("name") or ""), bool(gate.get("passed"))
    return str(getattr(gate, "name", "") or ""), bool(getattr(gate, "passed", False))


def good_buy_success_message(analysis: dict) -> str | None:
    """Success banner when is_good_buy. Quotes decision.label and gates, not composite_min."""
    if not analysis.get("is_good_buy"):
        return None
    decision = analysis.get("decision") or {}
    label = str(decision.get("label") or "Accumulate")
    passed: list[str] = []
    failed: list[str] = []
    for gate in decision.get("gates") or []:
        name, ok = _gate_name_passed(gate)
        if not name:
            continue
        (passed if ok else failed).append(name)
    parts = [label]
    if passed:
        parts.append("passed: " + ", ".join(passed))
    if failed:
        parts.append("failed: " + ", ".join(failed))
    return " — ".join(parts)


def render_stock_view(
    ticker: str,
    config: dict,
    scored_universe: pd.DataFrame | None = None,
    snapshot_date: str | None = None,
) -> None:
    with st.spinner(f"Analyzing {ticker}…"):
        analysis = score_ticker_cached(ticker, config)
        if scored_universe is not None and not scored_universe.empty:
            analysis = apply_stock_snapshot(analysis, scored_universe, ticker, config)

    if analysis.get("warning"):
        st.warning(analysis["warning"])

    data_warnings = analysis.get("data_warnings") or []
    if data_warnings:
        with st.expander("Data warnings", expanded=False):
            for w in data_warnings:
                from core.data_quality import warning_message

                st.warning(warning_message(w))

    thresholds = get_thresholds(config)
    if analysis.get("distress_flag"):
        z_pp = analysis.get("altman_z_pp")
        z_txt = f"{z_pp:.2f}" if z_pp is not None else "n/a"
        st.error(
            f"Distress zone: Altman Z'' = {z_txt} "
            f"(below {thresholds.get('altman_zpp_min', 1.1)}). "
            "Blocked from Buy regardless of composite/bargain scores."
        )
    banner = good_buy_success_message(analysis)
    if banner:
        st.success(banner)

    decision = analysis.get("decision") or {}
    if decision.get("label") == "Watch" and decision.get("buy_below_price") is not None:
        bb = float(decision["buy_below_price"])
        pct = decision.get("pct_to_buy")
        extra = f" ({pct:+.1f}%)" if pct is not None else ""
        st.info(
            f"Watch — Accumulate below {currency_symbol(analysis.get('currency'))}{bb:,.2f}{extra}"
        )

    # Company header card
    with st.container(border=True):
        render_company_header(analysis)

    render_decision_card(analysis)
    render_valuation_card(analysis)

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
