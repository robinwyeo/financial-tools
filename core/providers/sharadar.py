"""Sharadar (Nasdaq Data Link) adapter stub.

Paid Core US Equities Bundle would supply PIT fundamentals (SF1), prices (SEP),
and delisted tickers. Not implemented until a paid-data decision is made —
see ``docs/paid_data.md``.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class SharadarProvider:
    name = "sharadar"

    # Documented mapping for a future implementation:
    #   SF1  → annual/TTM fundamentals (ART/MRQ dimension, ARQ as-reported)
    #   SEP  → end-of-day prices including delisted tickers
    #   TICKERS → ticker / FIGI / delist date catalogue
    TABLE_MAP = {
        "fundamentals": "SF1",
        "prices": "SEP",
        "tickers": "TICKERS",
    }

    def ttm(self, ticker: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Sharadar is a paid adapter stub. See docs/paid_data.md "
            f"(would fetch SF1 TTM for {ticker})."
        )

    def annual(self, ticker: str) -> pd.DataFrame:
        raise NotImplementedError(
            "Sharadar is a paid adapter stub. See docs/paid_data.md "
            f"(would fetch SF1 annual for {ticker})."
        )

    def estimates(self, ticker: str) -> dict[str, Any]:
        raise NotImplementedError(
            "Sharadar SF1 does not include analyst estimates; use Yahoo/Finnhub."
        )
