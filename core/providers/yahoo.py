"""Yahoo Finance adapter wrapping existing ``core.data`` / ``core.estimates`` helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd


class YahooProvider:
    name = "yahoo"

    def ttm(self, ticker: str) -> dict[str, Any]:
        from core.data import fetch_ttm_financials

        return fetch_ttm_financials(ticker) or {}

    def annual(self, ticker: str) -> pd.DataFrame:
        from core.data import fetch_financials

        stmts = fetch_financials(ticker) or {}
        income = stmts.get("income")
        if isinstance(income, pd.DataFrame):
            return income
        return pd.DataFrame()

    def estimates(self, ticker: str) -> dict[str, Any]:
        from core.estimates import fetch_estimate_tables

        return fetch_estimate_tables(ticker)
