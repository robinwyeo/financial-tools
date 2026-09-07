"""SEC EDGAR companyfacts adapter wrapping ``core.fundamentals`` / ``core.edgar_facts``."""

from __future__ import annotations

from typing import Any

import pandas as pd


class EdgarProvider:
    name = "edgar"

    def ttm(self, ticker: str) -> pd.Series:
        from core.fundamentals import get_fundamentals

        fund = get_fundamentals(ticker)
        if fund.ttm is None:
            return pd.Series(dtype="float64")
        return fund.ttm

    def annual(self, ticker: str) -> pd.DataFrame:
        from core.fundamentals import get_fundamentals

        fund = get_fundamentals(ticker)
        if fund.annual is None:
            return pd.DataFrame()
        return fund.annual

    def estimates(self, ticker: str) -> dict[str, Any]:
        # EDGAR has no forward estimates.
        del ticker
        return {}
