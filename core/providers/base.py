"""FundamentalsProvider protocol and factory."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FundamentalsProvider(Protocol):
    """Swap-in source for TTM/annual statements and estimates."""

    name: str

    def ttm(self, ticker: str) -> dict[str, Any] | pd.Series:
        ...

    def annual(self, ticker: str) -> pd.DataFrame:
        ...

    def estimates(self, ticker: str) -> dict[str, Any]:
        ...


def get_fundamentals_providers(config: dict[str, Any] | None = None) -> list[FundamentalsProvider]:
    """Instantiate providers named in ``config.providers.fundamentals``."""
    from core.config import get_provider_names

    names = get_provider_names(config)
    out: list[FundamentalsProvider] = []
    for name in names:
        provider = _build_provider(name)
        if provider is not None:
            out.append(provider)
    return out


def _build_provider(name: str) -> FundamentalsProvider | None:
    key = str(name).lower().strip()
    if key == "edgar":
        from core.providers.edgar import EdgarProvider

        return EdgarProvider()
    if key == "yahoo":
        from core.providers.yahoo import YahooProvider

        return YahooProvider()
    if key == "finnhub":
        from core.providers.finnhub import FinnhubProvider

        provider = FinnhubProvider()
        if not provider.available():
            return None
        return provider
    if key == "sharadar":
        from core.providers.sharadar import SharadarProvider

        return SharadarProvider()
    return None
