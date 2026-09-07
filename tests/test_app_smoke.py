"""Smoke tests for the thin app and ui package."""

from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import pytest


def test_app_py_is_thin_entrypoint():
    text = Path("app.py").read_text(encoding="utf-8")
    assert "st.set_page_config" in text
    assert "def main" in text
    assert "render_stock_view" in text
    assert text.count("\n") < 120


def test_import_ui_stock_view_without_running_app():
    import ui.stock_view as stock_view

    assert hasattr(stock_view, "render_stock_view")
    assert hasattr(stock_view, "render_decision_card")


def test_import_ui_fund_view():
    import ui.fund_view as fund_view

    assert hasattr(fund_view, "render_fund_view")


def _price_range_app():
    """Score the universe once, then render the live price-range radio."""
    from ui.charts import render_price_history_card
    from ui.state import score_universe_cached

    score_universe_cached({"k": 1})
    render_price_history_card({"ticker": "TEST", "currency": "USD"})


def test_price_range_radio_does_not_rescore(monkeypatch):
    """Changing the price-range radio must not call score_universe again."""
    testing = pytest.importorskip("streamlit.testing.v1")
    AppTest = testing.AppTest

    import ui.state as state

    score_calls = {"n": 0}
    cached_calls = {"n": 0}

    def fake_score(_config):
        score_calls["n"] += 1
        return pd.DataFrame({"ticker": ["TEST"], "composite": [50.0]})

    real_cached = state.score_universe_cached

    def counting_cached(config):
        cached_calls["n"] += 1
        return real_cached(config)

    monkeypatch.setattr(state, "score_universe", fake_score)
    monkeypatch.setattr(state, "score_universe_cached", counting_cached)
    monkeypatch.setattr(state, "fetch_price_history", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr("ui.charts._card_shell", lambda _bordered=True: nullcontext())
    state._CACHED.clear()

    at = AppTest.from_function(_price_range_app, default_timeout=15)
    at.run()
    assert not at.exception
    assert len(at.radio) == 1
    assert at.radio[0].value == "2Y"
    assert cached_calls["n"] == 1
    assert score_calls["n"] == 1

    at.radio[0].set_value("1Y").run()
    assert not at.exception
    assert at.radio[0].value == "1Y"
    assert cached_calls["n"] == 2
    assert score_calls["n"] == 1
