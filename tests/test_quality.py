"""Tests for quality score buckets and value-trap flags."""

from __future__ import annotations

import pandas as pd
import pytest

from core.factors import compute_dilution, compute_stability
from core.quality import compute_quality_score, value_trap_flags


def test_declining_roic_and_dilution_trigger_two_flags():
    raw = {
        "sector": "Technology",
        "roic": 0.04,
        "roic_history": [0.12, 0.08, 0.04],
        "revenue_5y_cagr": 0.05,
        "gross_margin_5y_delta": 0.01,
        "net_debt_to_ebitda": 1.0,
        "interest_coverage": 8.0,
        "share_cagr_3y": 0.06,
        "accruals": 0.02,
        "fcf_conversion_3y": 0.8,
        "owner_earnings_norm": 10.0,
    }
    flags = {f.code: f.triggered for f in value_trap_flags(raw)}
    assert flags["ROIC_DECLINING"] is True
    assert flags["DILUTION"] is True
    assert sum(flags.values()) >= 2


def test_negative_equity_does_not_trigger_leverage_when_nd_ebitda_healthy():
    raw = {
        "sector": "Consumer Cyclical",
        "book_equity": -5e9,
        "net_debt_to_ebitda": 1.5,
        "interest_coverage": 12.0,
        "roic": 0.20,
        "revenue_5y_cagr": 0.08,
        "share_cagr_3y": -0.02,
        "accruals": 0.01,
        "fcf_conversion_3y": 1.1,
        "owner_earnings_norm": 4e9,
    }
    flags = value_trap_flags(raw)
    lev = next(f for f in flags if f.code == "LEVERAGE")
    assert lev.triggered is False


def test_compute_quality_score_adds_columns():
    rows = []
    for i, t in enumerate(["A", "B", "C", "D", "E"]):
        rows.append(
            {
                "ticker": t,
                "sector": "Technology",
                "gross_profitability": 0.2 + i * 0.05,
                "roic": 0.1 + i * 0.02,
                "fcf_margin": 0.1,
                "earnings_quality": 0.05,
                "fcf_conversion_3y": 0.8,
                "financial_strength": 6 + i,
                "leverage_quality": -1.0,
                "interest_coverage": 8.0,
                "roic_5y_mean": 0.12,
                "stability_roic": -0.02,
                "gross_margin_5y_delta": 0.0,
                "revenue_5y_cagr": 0.05,
                "shareholder_yield": 0.03,
                "investment": 0.0,
                "anti_dilution": 0.01,
            }
        )
    df = pd.DataFrame(rows)
    out = compute_quality_score(df, group_col="sector")
    assert "quality_score" in out.columns
    assert out["quality_score"].notna().mean() >= 0.95
    assert (out["quality_score"] >= 0).all()


def test_dilution_and_stability_helpers():
    dil = compute_dilution({"share_cagr_3y": 0.04})
    assert dil["anti_dilution"] == pytest.approx(-0.04)
    stab = compute_stability({"roic_5y_std": 0.05, "roic_5y_mean": 0.12})
    assert stab["stability_roic"] == pytest.approx(-0.05)
