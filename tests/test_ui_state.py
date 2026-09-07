"""Tests for Streamlit cache wrappers in ui.state."""

from __future__ import annotations

import inspect
import types

import pandas as pd
import pytest

import ui.state as state


def _install_identity_cache(monkeypatch):
    """Replace st.cache_data with an identity wrapper that records cache keys."""
    recorded_keys: list[tuple] = []
    inner_calls = {"n": 0}

    def fake_cache_data(*_args, **_kwargs):
        def decorator(fn):
            def wrapped(*args):
                recorded_keys.append(args)
                inner_calls["n"] += 1
                return fn(*args)

            return wrapped

        return decorator

    monkeypatch.setattr("streamlit.cache_data", fake_cache_data)
    state._CACHED.clear()
    return recorded_keys, inner_calls


def _install_streamlit_like_cache(monkeypatch):
    """Cache by non-underscore arguments, matching st.cache_data semantics."""
    stores: list[dict[tuple, object]] = []

    def fake_cache_data(*_args, **_kwargs):
        def decorator(fn):
            signature = inspect.signature(fn)
            store: dict[tuple, object] = {}
            stores.append(store)

            def wrapped(*args):
                bound = signature.bind(*args)
                key = tuple(
                    (name, repr(value))
                    for name, value in bound.arguments.items()
                    if not name.startswith("_")
                )
                if key not in store:
                    store[key] = fn(*args)
                return store[key]

            return wrapped

        return decorator

    monkeypatch.setattr("streamlit.cache_data", fake_cache_data)
    state._CACHED.clear()
    return stores


def test_score_ticker_cache_key_includes_ticker(monkeypatch):
    """META/NVDA must not receive the first cached AAPL dashboard result."""
    monkeypatch.setattr(state, "load_universe_snapshot", lambda: pd.DataFrame())
    monkeypatch.setattr(state, "_load_uni_snap", lambda: pd.DataFrame())
    calls: list[str] = []

    def fake_score(ticker, config, snap):
        calls.append(ticker)
        return {"ticker": ticker}

    monkeypatch.setattr(state, "score_ticker", fake_score)
    _install_streamlit_like_cache(monkeypatch)

    assert state.score_ticker_cached("AAPL", {})["ticker"] == "AAPL"
    assert state.score_ticker_cached("META", {})["ticker"] == "META"
    assert state.score_ticker_cached("NVDA", {})["ticker"] == "NVDA"
    assert state.score_ticker_cached("META", {})["ticker"] == "META"
    assert calls == ["AAPL", "META", "NVDA"]


def test_price_history_cache_key_includes_ticker_and_period(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_history(ticker, period):
        calls.append((ticker, period))
        return pd.DataFrame({"ticker": [ticker], "period": [period]})

    monkeypatch.setattr(state, "_fetch_price_history", fake_history)
    _install_streamlit_like_cache(monkeypatch)

    assert state.fetch_price_history("AAPL", "2y").iloc[0]["ticker"] == "AAPL"
    assert state.fetch_price_history("META", "2y").iloc[0]["ticker"] == "META"
    assert state.fetch_price_history("META", "1y").iloc[0]["period"] == "1y"
    state.fetch_price_history("META", "2y")
    assert calls == [("AAPL", "2y"), ("META", "2y"), ("META", "1y")]


def test_score_fund_cached_different_mtimes_are_different_keys(monkeypatch):
    mtime = {"value": 100.0}
    fake_path = types.SimpleNamespace(
        exists=lambda: True,
        stat=lambda: types.SimpleNamespace(st_mtime=mtime["value"]),
    )
    monkeypatch.setattr(state, "FUND_SNAPSHOT_PATH", fake_path)
    monkeypatch.setattr(
        state,
        "load_fund_universe_snapshot",
        lambda: pd.DataFrame({"ticker": ["VTI"], "snapshot_date": ["2026-01-01"]}),
    )
    monkeypatch.setattr(state, "_load_fund_snap", lambda: pd.DataFrame({"ticker": ["VTI"]}))
    monkeypatch.setattr(
        state,
        "score_fund",
        lambda ticker, config, snap: {"ticker": ticker, "mtime": mtime["value"]},
    )

    recorded_keys, inner_calls = _install_identity_cache(monkeypatch)

    first = state.score_fund_cached("VTI", {})
    mtime["value"] = 200.0
    second = state.score_fund_cached("VTI", {})

    assert inner_calls["n"] == 2
    assert recorded_keys[0] != recorded_keys[1]
    assert recorded_keys[0][0] == recorded_keys[1][0] == "VTI"
    # args: ticker, snapshot_date, mtime, config
    assert recorded_keys[0][2] == pytest.approx(100.0)
    assert recorded_keys[1][2] == pytest.approx(200.0)
    assert first["mtime"] == 100.0
    assert second["mtime"] == 200.0


def test_score_fund_cached_loads_live_snapshot_inside_inner(monkeypatch):
    """_inner must not close over the wrapper's first snapshot."""
    snaps = [
        pd.DataFrame({"ticker": ["AAA"]}),
        pd.DataFrame({"ticker": ["BBB"]}),
    ]
    seen: list[str] = []

    def fake_load_snap():
        return snaps.pop(0)

    def fake_score(ticker, config, snap):
        seen.append(str(snap["ticker"].iloc[0]))
        return {"ticker": ticker, "peer": seen[-1]}

    mtime = {"value": 1.0}
    monkeypatch.setattr(
        state,
        "FUND_SNAPSHOT_PATH",
        types.SimpleNamespace(
            exists=lambda: True,
            stat=lambda: types.SimpleNamespace(st_mtime=mtime["value"]),
        ),
    )
    monkeypatch.setattr(
        state,
        "load_fund_universe_snapshot",
        lambda: pd.DataFrame({"ticker": ["VTI"], "snapshot_date": ["2026-01-01"]}),
    )
    monkeypatch.setattr(state, "_load_fund_snap", fake_load_snap)
    monkeypatch.setattr(state, "score_fund", fake_score)
    _install_identity_cache(monkeypatch)

    state.score_fund_cached("VTI", {})
    mtime["value"] = 2.0
    state.score_fund_cached("VTI", {})

    assert seen == ["AAA", "BBB"]
