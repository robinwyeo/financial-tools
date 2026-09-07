"""Tests for pure UI HTML builders."""

from core.config import get_decision_config, load_config
from ui.formatting import decision_card_html, ordinal, valuation_card_html
from ui.sidebar import stock_sidebar_criteria
from ui.stock_view import good_buy_success_message


def test_ordinal():
    assert ordinal(1) == "1st"
    assert ordinal(2) == "2nd"
    assert ordinal(3) == "3rd"
    assert ordinal(11) == "11th"


def test_decision_card_watch_shows_buy_below():
    html = decision_card_html(
        {
            "label": "Watch",
            "buy_below_price": 84.0,
            "pct_to_buy": -6.7,
            "gates": [
                {"name": "price_vs_buy_below", "passed": False, "actual": 90, "threshold": 84, "note": ""},
            ],
            "timing_context": {"hint": "negative 12-1 momentum: consider tranches"},
        },
        currency="USD",
    )
    assert "Watch" in html
    assert "Accumulate below" in html
    assert "$84.00" in html
    assert "FAIL" in html


def test_valuation_card_shows_range():
    html = valuation_card_html(
        {
            "dcf": {"base": {"per_share": 120.0}, "bear": {"per_share": 95.0}},
            "epv": {"per_share": 90.0},
            "reverse_dcf": {"implied_g1": 0.08},
            "growth": {"g_hist": 0.06},
            "expected_return": 0.12,
            "hurdle": {"rate": 0.087, "rf": 0.042, "erp": 0.045},
        }
    )
    assert "120.00" in html
    assert "95.00" in html
    assert "hurdle" in html


def test_valuation_card_shows_rf_as_of():
    html = valuation_card_html(
        {
            "dcf": {"base": {"per_share": 120.0}, "bear": {"per_share": 95.0}},
            "epv": {"per_share": 90.0},
            "reverse_dcf": {"implied_g1": 0.08},
            "growth": {"g_hist": 0.06},
            "expected_return": 0.12,
            "hurdle": {
                "rate": 0.087,
                "rf": 0.042,
                "erp": 0.045,
                "rf_source": "FRED:DGS10",
                "rf_as_of": "2024-01-01",
            },
        }
    )
    assert "2024-01-01" in html
    assert "FRED:DGS10 as of 2024-01-01" in html


def _primary_text(copy: dict) -> str:
    return "\n".join(copy["lines"]).lower()


def test_default_config_sidebar_is_intrinsic_not_legacy_hurdles():
    config = load_config()
    assert get_decision_config(config)["mode"] == "intrinsic"
    copy = stock_sidebar_criteria(config)
    assert copy["mode"] == "intrinsic"
    assert copy["heading"] == "Accumulate criteria"
    primary = _primary_text(copy)
    assert "quality percentile" in primary
    assert "value-trap flags" in primary
    assert "buy-below" in primary
    assert "mos" in primary
    assert "hurdle" in primary
    assert "data-quality" in primary and "grade c" in primary
    assert "z''" in primary
    assert "composite ≥" not in primary
    assert "bargain ≥" not in primary
    assert "coverage ≥" not in primary
    legacy = "\n".join(copy["legacy_lines"] or []).lower()
    assert "composite ≥" in legacy
    assert "coverage ≥" in legacy


def test_legacy_sidebar_shows_composite_bargain_coverage():
    copy = stock_sidebar_criteria(
        {
            "decision": {"mode": "legacy"},
            "thresholds": {
                "composite_min": 58.9,
                "bargain_min": 49.5,
                "coverage_min_pct": 70,
                "altman_zpp_min": 1.1,
                "exclude_sell_consensus": True,
            },
        }
    )
    assert copy["mode"] == "legacy"
    assert copy["heading"] == "Good-buy criteria"
    assert copy["legacy_lines"] is None
    primary = _primary_text(copy)
    assert "composite ≥ 58.9" in primary
    assert "bargain ≥ 49.5" in primary
    assert "coverage ≥ 70%" in primary


def test_success_banner_quotes_label_and_gates_not_composite_min():
    msg = good_buy_success_message(
        {
            "is_good_buy": True,
            "decision": {
                "label": "Accumulate",
                "gates": [
                    {"name": "data_quality", "passed": True},
                    {"name": "distress", "passed": True},
                    {"name": "value_trap_count", "passed": True},
                    {"name": "quality_percentile", "passed": True},
                    {"name": "price_vs_buy_below", "passed": True},
                    {"name": "expected_return_vs_hurdle", "passed": True},
                ],
            },
        }
    )
    assert msg is not None
    assert msg.startswith("Accumulate")
    assert "passed: data_quality" in msg
    assert "price_vs_buy_below" in msg
    assert "composite" not in msg.lower()
    assert "58.9" not in msg
    assert good_buy_success_message({"is_good_buy": False, "decision": {"label": "Watch"}}) is None
