"""Tests for hurdle-rate construction."""

from datetime import date

import pandas as pd
import pytest

from core.rates import hurdle_rate, pit_risk_free


def test_hurdle_fallback_without_fred(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr("core.rates.fetch_fred_series", lambda *a, **k: None)
    out = hurdle_rate()
    assert out["rate"] == pytest.approx(0.087)
    assert out["rf"] == pytest.approx(0.042)
    assert out["rf_source"] == "config_fallback"


def test_hurdle_uses_dgs10():
    out = hurdle_rate(rf=0.046)
    assert out["rate"] == pytest.approx(0.091)
    assert out["rf"] == pytest.approx(0.046)


def test_hurdle_high_uncertainty_bump():
    out = hurdle_rate("High", rf=0.046)
    assert out["rate"] == pytest.approx(0.101)
    assert out["uncertainty_bump"] == pytest.approx(0.01)


def test_hurdle_floor_binds():
    out = hurdle_rate(rf=0.02)
    assert out["rate"] == pytest.approx(0.08)


def test_fetch_fred_series_parses_percent(monkeypatch):
    from core.rates import fetch_fred_series

    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {"observations": [{"date": "2026-09-04", "value": "4.20"}]}

    monkeypatch.setenv("FRED_API_KEY", "test")
    monkeypatch.setattr("core.rates._read_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.rates._write_cache", lambda *a, **k: None)
    monkeypatch.setattr("core.rates.requests.get", lambda *a, **k: _Resp())
    val = fetch_fred_series("BAA10Y", api_key="test", force=True)
    assert val == pytest.approx(0.042)


def test_fetch_fred_history_empty_without_key(monkeypatch):
    from core.rates import fetch_fred_history

    monkeypatch.delenv("FRED_API_KEY", raising=False)
    monkeypatch.setattr("core.rates._read_cache", lambda *a, **k: None)
    series = fetch_fred_history("T10YIE")
    assert series.empty


def test_pit_risk_free_steps_in_2015_vs_2020():
    hist = pd.Series(
        [0.022, 0.006],
        index=pd.to_datetime(["2015-06-15", "2020-06-15"]),
        dtype="float64",
    )
    rf_2015 = pit_risk_free(date(2015, 12, 31), hist)
    rf_2020 = pit_risk_free(date(2020, 12, 31), hist)
    assert rf_2015 == pytest.approx(0.022)
    assert rf_2020 == pytest.approx(0.006)
    assert rf_2015 != rf_2020
