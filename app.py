"""Streamlit dashboard for stock metrics and analyst aggregation."""

from __future__ import annotations

import streamlit as st

from core.data import FUND_QUOTE_TYPES, get_security_type
from ui.fund_view import render_fund_view
from ui.layout import inject_css
from ui.rankings import render_fund_universe_rankings, render_universe_rankings
from ui.sidebar import render_sidebar
from ui.state import (
    load_config,
    load_fund_universe_snapshot,
    load_universe_snapshot,
    score_fund_universe_cached,
    score_universe_cached,
)
from ui.stock_view import render_stock_view

st.set_page_config(
    page_title="Stock & Fund Metrics Tool",
    page_icon="📊",
    layout="wide",
)


def main() -> None:
    inject_css()
    config = load_config()
    with st.sidebar:
        ticker = render_sidebar(config)
        viewing_fund = bool(ticker) and get_security_type(ticker) in FUND_QUOTE_TYPES

    if viewing_fund:
        fund_snapshot = load_fund_universe_snapshot()
        fund_snapshot_date = None
        if (
            fund_snapshot is not None
            and not fund_snapshot.empty
            and "snapshot_date" in fund_snapshot.columns
        ):
            fund_snapshot_date = str(fund_snapshot["snapshot_date"].iloc[0])
        scored_fund_universe = score_fund_universe_cached(config)
        render_fund_view(ticker, config, scored_fund_universe, fund_snapshot_date)
        st.markdown("---")
        render_fund_universe_rankings(scored_fund_universe)
        return

    snapshot = load_universe_snapshot()
    snapshot_date = None
    if snapshot is not None and not snapshot.empty and "snapshot_date" in snapshot.columns:
        snapshot_date = str(snapshot["snapshot_date"].iloc[0])
    scored_universe = score_universe_cached(config)

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
