"""Provider protocol and stubs."""

import pandas as pd
import pytest

from core.providers.base import FundamentalsProvider, get_fundamentals_providers
from core.providers.edgar import EdgarProvider
from core.providers.finnhub import FinnhubProvider
from core.providers.sharadar import SharadarProvider
from core.providers.yahoo import YahooProvider


class _FakeProvider:
    name = "fake"

    def ttm(self, ticker: str):
        return {"revenue": 1, "ticker": ticker}

    def annual(self, ticker: str):
        return pd.DataFrame({"revenue": [1]})

    def estimates(self, ticker: str):
        return {}


def test_fake_conforms_to_protocol():
    provider: FundamentalsProvider = _FakeProvider()
    assert provider.ttm("X")["revenue"] == 1
    assert not provider.annual("X").empty
    assert provider.estimates("X") == {}


def test_concrete_providers_have_protocol_methods():
    for cls in (YahooProvider, EdgarProvider, FinnhubProvider, SharadarProvider):
        inst = cls()
        assert hasattr(inst, "ttm")
        assert hasattr(inst, "annual")
        assert hasattr(inst, "estimates")
        assert inst.name


def test_sharadar_stub_raises():
    with pytest.raises(NotImplementedError):
        SharadarProvider().ttm("AAPL")


def test_finnhub_skipped_without_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    providers = get_fundamentals_providers({"providers": {"fundamentals": ["finnhub", "yahoo"]}})
    names = [p.name for p in providers]
    assert "finnhub" not in names
    assert "yahoo" in names


def test_factory_default_order():
    providers = get_fundamentals_providers({"providers": {"fundamentals": ["edgar", "yahoo"]}})
    assert [p.name for p in providers] == ["edgar", "yahoo"]
