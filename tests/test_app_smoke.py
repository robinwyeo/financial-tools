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


def test_card_shell_is_usable_as_context_manager(monkeypatch):
    """Regression: a bare generator is not a context manager (TypeError on Streamlit Cloud)."""
    import ui.layout as layout

    class FakeContainer:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(layout.st, "container", lambda **_kwargs: FakeContainer())

    with layout._card_shell(True):
        pass
    with layout._card_shell(False):
        pass


def test_normalize_ticker_accepts_query_param_shapes():
    from ui.sidebar import normalize_ticker

    assert normalize_ticker("nvda") == "NVDA"
    assert normalize_ticker(["msft"]) == "MSFT"
    assert normalize_ticker("  aapl  ") == "AAPL"
    assert normalize_ticker(None) == ""
    assert normalize_ticker([]) == ""


def test_is_fund_ticker_uses_snapshots_not_yahoo(monkeypatch):
    from ui import sidebar

    calls = {"n": 0}

    def boom(_ticker):
        calls["n"] += 1
        return "EQUITY"

    monkeypatch.setattr(
        sidebar,
        "load_universe_snapshot",
        lambda: pd.DataFrame({"ticker": ["AAPL", "NVDA"]}),
    )
    monkeypatch.setattr(
        sidebar,
        "load_fund_universe_snapshot",
        lambda: pd.DataFrame({"ticker": ["VTI", "VFV.TO"]}),
    )
    monkeypatch.setattr(sidebar, "get_security_type", boom)

    assert sidebar.is_fund_ticker("nvda") is False
    assert sidebar.is_fund_ticker("VTI") is True
    assert calls["n"] == 0
    sidebar.is_fund_ticker("XYZ")
    assert calls["n"] == 1


def _ticker_sidebar_app():
    from ui.sidebar import render_sidebar
    import streamlit as st

    ticker = render_sidebar({"decision": {"mode": "intrinsic"}, "thresholds": {}})
    st.write(f"ACTIVE:{ticker}")


def test_sidebar_ticker_input_can_change_from_default(monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    AppTest = testing.AppTest
    from ui import sidebar

    monkeypatch.setattr(sidebar, "is_fund_ticker", lambda _t: False)
    monkeypatch.setattr(sidebar, "load_universe_snapshot", lambda: None)
    monkeypatch.setattr(sidebar, "load_fund_universe_snapshot", lambda: None)

    at = AppTest.from_function(_ticker_sidebar_app, default_timeout=15)
    at.run()
    assert not at.exception
    assert at.text_input[0].value == "AAPL"
    assert any("Viewing AAPL" in str(getattr(el, "value", el)) for el in at.caption)

    at.text_input[0].set_value("NVDA")
    assert at.button
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["active_ticker"] == "NVDA"
    texts = [str(getattr(el, "value", el)) for el in list(at.markdown) + list(at.caption)]
    assert any("ACTIVE:NVDA" in t or "Viewing NVDA" in t for t in texts)


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
