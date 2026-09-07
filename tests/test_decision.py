"""Tests for Accumulate / Watch / Avoid gates."""

from __future__ import annotations

import pytest

from core.decision import decide


def _base(price: float, dcf_base: float = 120.0, dcf_bear: float = 95.0, **extra) -> dict:
    analysis = {
        "ticker": "ABC",
        "price": price,
        "sector": "Technology",
        "quality_score": 60,
        "quality_percentile": 60,
        "data_quality": {"grade": "A"},
        "distress_flag": False,
        "altman_z_pp": 3.0,
        "value_trap_flags": extra.get("flags", []),
        "uncertainty": {"label": "Medium", "threshold_bump": 3},
        "valuation": {
            "relative_only": False,
            "dcf": {
                "base": {"per_share": dcf_base},
                "bear": {"per_share": dcf_bear},
            },
            "expected_return": extra.get("expected_return", 0.12),
            "hurdle": {"rate": 0.087},
        },
        "analyst": {"consensus_label": "Buy"},
        "bargain": {"score": 60},
        "composite": 70,
        "factor_coverage_pct": 90,
    }
    analysis.update({k: v for k, v in extra.items() if k != "flags" and k != "expected_return"})
    return analysis


def test_accumulate_when_price_at_or_below_buy_below():
    # Medium MoS 30%: min(120*0.7, 95) = min(84, 95) = 84
    d = decide(_base(80), {"decision": {"mode": "intrinsic"}})
    assert d.buy_below_price == pytest.approx(84.0)
    assert d.label == "Accumulate"
    assert d.pct_to_buy == pytest.approx((84 / 80 - 1) * 100, rel=1e-3)


def test_watch_when_price_above_buy_below():
    d = decide(_base(90), {"decision": {"mode": "intrinsic"}})
    assert d.label == "Watch"
    assert d.pct_to_buy == pytest.approx((84 / 90 - 1) * 100, rel=1e-3)


def test_grade_c_blocks_accumulate():
    a = _base(80)
    a["data_quality"] = {"grade": "C"}
    d = decide(a, {"decision": {"mode": "intrinsic"}})
    assert d.label == "Avoid"
    dq = next(g for g in d.gates if g.name == "data_quality")
    assert dq.passed is False


def test_legacy_mode_uses_composite_bargain():
    a = _base(90)
    d = decide(
        a,
        {
            "decision": {"mode": "legacy"},
            "thresholds": {
                "composite_min": 50,
                "bargain_min": 50,
                "exclude_sell_consensus": True,
            },
        },
    )
    assert d.mode == "legacy"
    assert d.label == "Accumulate"


def test_value_trap_max_flags_controls_flag_count_gate():
    flags = [
        {"code": "ROIC_DECLINING", "triggered": True},
        {"code": "REVENUE_CAGR_NEG", "triggered": True},
    ]
    analysis = _base(80, flags=flags)
    blocked = decide(
        analysis,
        {"decision": {"mode": "intrinsic"}, "value_trap": {"max_flags": 1}},
    )
    trap = next(g for g in blocked.gates if g.name == "value_trap_count")
    assert trap.passed is False
    assert trap.threshold == 1
    assert blocked.label == "Avoid"

    allowed = decide(
        analysis,
        {"decision": {"mode": "intrinsic"}, "value_trap": {"max_flags": 2}},
    )
    trap_ok = next(g for g in allowed.gates if g.name == "value_trap_count")
    assert trap_ok.passed is True
    assert trap_ok.threshold == 2
    assert allowed.label == "Accumulate"


def test_financials_relative_only_path():
    a = _base(80)
    a["sector"] = "Financial Services"
    a["valuation"] = {
        "relative_only": True,
        "price_to_book": 1.1,
        "relative": {"ev_ebit": {"percentile": 70}},
        "hurdle": {"rate": 0.087},
        "expected_return": None,
        "dcf": None,
    }
    d = decide(a, {"decision": {"mode": "intrinsic"}})
    assert d.label == "Accumulate"
    assert any(g.name == "relative_cheap" for g in d.gates)
