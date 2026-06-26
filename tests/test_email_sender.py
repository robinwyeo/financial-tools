"""Tests for scorecard email formatting."""

from jobs.email_sender import _get_smtp_config, format_scorecard_email


def _sample_result(ticker: str, *, is_buy: bool, composite: float) -> dict:
    return {
        "ticker": ticker,
        "name": f"{ticker} Inc",
        "composite": composite,
        "bargain": {"score": 55.0},
        "analyst": {"implied_upside_pct": 20.0},
        "is_good_buy": is_buy,
    }


def test_smtp_config_strips_whitespace(monkeypatch):
    monkeypatch.setenv("SMTP_FROM", " alerts@example.com ")
    monkeypatch.setenv("SMTP_TO", " to@example.com\n")
    monkeypatch.setenv("SMTP_PASSWORD", " abcd efgh ijkl mnop ")
    smtp = _get_smtp_config({"email": {}})
    assert smtp["from_address"] == "alerts@example.com"
    assert smtp["to_address"] == "to@example.com"
    assert smtp["password"] == "abcdefghijklmnop"


def test_smtp_login_uses_from_address_over_stale_username(monkeypatch):
    monkeypatch.setenv("SMTP_FROM", "alerts@example.com")
    monkeypatch.setenv("SMTP_USERNAME", "old@example.com")
    smtp = _get_smtp_config(
        {"email": {"from_address": "fallback@example.com", "to_address": "to@example.com"}}
    )
    assert smtp["from_address"] == "alerts@example.com"
    assert smtp["username"] == "alerts@example.com"


def test_format_scorecard_email_lists_buy_first():
    results = [
        _sample_result("ZZZ", is_buy=False, composite=80),
        _sample_result("AAA", is_buy=True, composite=60),
    ]
    subject, html = format_scorecard_email(
        results,
        {"thresholds": {"composite_min": 50, "bargain_min": 50, "implied_upside_min_pct": 15}},
        title="Test Scorecard",
    )
    assert "1 Buy / 2 total" in subject
    assert html.index("AAA") < html.index("ZZZ")
    assert "Buy" in html
    assert "Not Buy" in html
    assert "Composite" in html
    assert "Bargain" in html
