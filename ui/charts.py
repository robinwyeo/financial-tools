"""Plotly charts. Views import this; this module does not import views."""

from __future__ import annotations

import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from ui.constants import (
    CHART_HEIGHT_ANALYST_PIE,
    CHART_HEIGHT_PRICE,
    CHART_HEIGHT_RADAR,
    DEFAULT_PRICE_RANGE,
    PRICE_HISTORY_RANGES,
    RADAR_FACTOR_LABELS,
)
from ui.formatting import _price_position_strip_html
from ui.layout import _card_shell

_PRICE_UP = "#188038"


_PRICE_DOWN = "#d93025"


_PRICE_CHART_HOVER_CSS = """
.js-plotly-plot .hoverlayer .hovertext {
    filter: drop-shadow(0 1px 4px rgba(60, 64, 67, 0.18));
}
"""

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
        from ui.state import fetch_price_history
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
        values = []
        for family in families:
            v = breakdown.get(family, {}).get("percentile")
            if v is None or (isinstance(v, float) and math.isnan(v)):
                values.append(None)
            else:
                values.append(float(v))
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
