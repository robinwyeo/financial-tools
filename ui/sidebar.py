"""Sidebar settings."""

from __future__ import annotations

import streamlit as st

from core.config import (
    get_bargain_weights,
    get_decision_config,
    get_factor_weights,
    get_fund_factor_weights,
    get_thresholds,
)
from core.factors import FACTOR_SCORE_COLUMNS
from core.fund_factors import FUND_FACTOR_SCORE_COLUMNS
from core.data import FUND_QUOTE_TYPES, get_security_type
from ui.constants import BARGAIN_LABELS, FACTOR_LABELS, FUND_FACTOR_LABELS
from ui.state import load_fund_universe_snapshot, load_universe_snapshot

_MOS_ORDER = ("Low", "Medium", "High")


def _mos_sidebar_text(mos: dict) -> str:
    keys = [k for k in _MOS_ORDER if k in mos]
    keys.extend(k for k in mos if k not in _MOS_ORDER)
    return " / ".join(f"{k} {float(mos[k]) * 100:.0f}%" for k in keys)


def legacy_good_buy_lines(config: dict) -> list[str]:
    """Composite/bargain/coverage bullets — primary copy only when mode is legacy."""
    thresholds = get_thresholds(config)
    lines = [
        f"Composite ≥ {thresholds['composite_min']}",
        f"Bargain ≥ {thresholds.get('bargain_min', 50)}",
        f"Coverage ≥ {thresholds.get('coverage_min_pct', 70):.0f}% of factor-group weight",
        (
            f"Altman Z'' ≥ {thresholds.get('altman_zpp_min', 1.1)} "
            "(all non-financials; Financials exempt)"
        ),
        (
            f"High uncertainty adds +{thresholds.get('uncertainty_high_bump', 6):.0f} "
            "to both hurdles"
        ),
    ]
    if thresholds.get("exclude_sell_consensus"):
        extra = " and Underperform" if thresholds.get("exclude_underperform", True) else ""
        lines.append(f"Excludes Sell{extra} consensus names")
    return lines


def intrinsic_accumulate_lines(config: dict) -> list[str]:
    """Gates that decide() applies in intrinsic mode (coverage is not a gate)."""
    decision_cfg = get_decision_config(config)
    thresholds = get_thresholds(config)
    min_q = float(decision_cfg.get("min_quality_percentile", 40))
    max_flags = int(decision_cfg.get("max_value_trap_flags", 1))
    mos = decision_cfg.get("required_margin_of_safety") or {}
    min_gap = float(decision_cfg.get("min_expected_return_over_hurdle", 0.0))
    block_c = bool(decision_cfg.get("block_on_data_quality_c", True))
    hurdle_line = "Expected return ≥ hurdle"
    if min_gap:
        hurdle_line += f" (min gap {min_gap:.0%})"
    dq_line = (
        "Data-quality grade C blocks Accumulate"
        if block_c
        else "Data-quality grade C allowed"
    )
    mos_line = (
        f"Price ≤ buy-below (MoS {_mos_sidebar_text(mos)})" if mos else "Price ≤ buy-below"
    )
    return [
        f"Quality percentile ≥ {min_q:.0f}",
        f"Value-trap flags ≤ {max_flags}",
        mos_line,
        hurdle_line,
        dq_line,
        (
            f"Altman Z'' ≥ {thresholds.get('altman_zpp_min', 1.1)} "
            "(all non-financials; Financials exempt)"
        ),
    ]


def stock_sidebar_criteria(config: dict) -> dict:
    """Pure copy for the stock sidebar. Tests assert this without Streamlit widgets."""
    mode = get_decision_config(config).get("mode", "intrinsic")
    thresholds = get_thresholds(config)
    caption = (
        f"Analyst upside (info only; context ≥ {thresholds['implied_upside_min_pct']}%)"
    )
    legacy_lines = legacy_good_buy_lines(config)
    if mode == "legacy":
        return {
            "mode": "legacy",
            "heading": "Good-buy criteria",
            "lines": legacy_lines,
            "caption": caption,
            "legacy_lines": None,
        }
    return {
        "mode": "intrinsic",
        "heading": "Accumulate criteria",
        "lines": intrinsic_accumulate_lines(config),
        "caption": None,
        "legacy_lines": legacy_lines,
        "legacy_caption": caption,
    }


def _render_stock_sidebar_sections(config: dict) -> None:
    copy = stock_sidebar_criteria(config)
    st.markdown(f"**{copy['heading']}**")
    for line in copy["lines"]:
        st.write(line)
    if copy.get("caption"):
        st.caption(copy["caption"])
    if copy.get("legacy_lines"):
        with st.expander("Legacy gate"):
            for line in copy["legacy_lines"]:
                st.write(line)
            if copy.get("legacy_caption"):
                st.caption(copy["legacy_caption"])
    st.markdown("---")
    st.markdown("**Composite factor weights**")
    st.caption(
        "Nine factor groups (revisions and insider buying are live-only). "
        "Shown as a share of total; renormalized at runtime over groups with data."
    )
    factor_weights = get_factor_weights(config)
    factor_total = sum(factor_weights.values()) or 1.0
    for family in sorted(FACTOR_SCORE_COLUMNS, key=lambda f: factor_weights.get(f, 0.0), reverse=True):
        weight = factor_weights.get(family, 0.0)
        st.write(f"{FACTOR_LABELS.get(family, family)}: {weight / factor_total:.1%}")
    st.markdown("---")
    st.markdown("**Bargain score weights**")
    bargain_weights = get_bargain_weights(config)
    bargain_total = sum(bargain_weights.values()) or 1.0
    for key in sorted(bargain_weights, key=lambda k: bargain_weights.get(k, 0.0), reverse=True):
        weight = bargain_weights[key]
        st.write(f"{BARGAIN_LABELS.get(key, key)}: {weight / bargain_total:.1%}")

    snapshot = load_universe_snapshot()
    if snapshot is not None and not snapshot.empty:
        date = snapshot["snapshot_date"].iloc[0] if "snapshot_date" in snapshot.columns else "unknown"
        st.caption(f"Universe: {len(snapshot)} tickers (snapshot: {date})")


def _render_fund_sidebar_sections(config: dict) -> None:
    st.markdown("**Fund composite weights**")
    st.caption(
        "Six fund factor groups (fees first — the strongest documented predictor "
        "of relative fund performance). Renormalized at runtime over groups with data."
    )
    fund_weights = get_fund_factor_weights(config)
    fund_total = sum(fund_weights.values()) or 1.0
    for family in sorted(FUND_FACTOR_SCORE_COLUMNS, key=lambda f: fund_weights.get(f, 0.0), reverse=True):
        weight = fund_weights.get(family, 0.0)
        st.write(f"{FUND_FACTOR_LABELS.get(family, family)}: {weight / fund_total:.1%}")

    snapshot = load_fund_universe_snapshot()
    if snapshot is not None and not snapshot.empty:
        date = snapshot["snapshot_date"].iloc[0] if "snapshot_date" in snapshot.columns else "unknown"
        st.caption(f"Fund universe: {len(snapshot)} funds (snapshot: {date})")


def render_sidebar(config: dict) -> str:
    st.header("Settings")
    default_ticker = st.query_params.get("ticker", "AAPL")
    ticker = st.text_input(
        "Ticker",
        value=default_ticker,
        help="Stocks, ETFs, and mutual funds (US and Canadian; use .TO for TSX listings).",
    ).upper().strip()
    viewing_fund = bool(ticker) and get_security_type(ticker) in FUND_QUOTE_TYPES
    st.markdown("---")
    if viewing_fund:
        _render_fund_sidebar_sections(config)
    else:
        _render_stock_sidebar_sections(config)
    return ticker
