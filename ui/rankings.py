"""Universe ranking tables."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.constants import SECURITY_TYPE_BADGES
from ui.state import load_universe_snapshot, score_universe_cached

def render_universe_rankings(
    config: dict,
    scored_universe: pd.DataFrame | None = None,
) -> None:
    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        st.warning("No universe snapshot found. Run the monthly universe job to build one.")
        return

    scored = scored_universe if scored_universe is not None else score_universe_cached(config)
    if scored is not None and not scored.empty and "universe" in scored.columns:
        members = sorted(str(u) for u in scored["universe"].dropna().unique())
        if len(members) > 1:
            choice = st.multiselect("Universes", members, default=members)
            if choice:
                scored = scored[scored["universe"].astype(str).isin(choice)]
    if scored.empty or "composite" not in scored.columns:
        st.warning("Unable to score universe.")
        return

    scored["composite"] = pd.to_numeric(scored["composite"], errors="coerce")
    cols = [c for c in ["ticker", "name", "sector", "universe", "composite"] if c in scored.columns]
    top = scored.nlargest(20, "composite")[cols]
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
