"""Optional Finnhub adapter. Absent ``FINNHUB_API_KEY`` → skipped by the factory."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests


class FinnhubProvider:
    name = "finnhub"
    BASE = "https://finnhub.io/api/v1"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("FINNHUB_API_KEY")

    def available(self) -> bool:
        return bool(self.api_key)

    def ttm(self, ticker: str) -> dict[str, Any]:
        if not self.available():
            return {}
        try:
            resp = requests.get(
                f"{self.BASE}/stock/metric",
                params={"symbol": ticker.upper(), "metric": "all", "token": self.api_key},
                timeout=20,
            )
            resp.raise_for_status()
            return (resp.json() or {}).get("metric") or {}
        except Exception:
            return {}

    def annual(self, ticker: str) -> pd.DataFrame:
        if not self.available():
            return pd.DataFrame()
        try:
            resp = requests.get(
                f"{self.BASE}/stock/financials-reported",
                params={"symbol": ticker.upper(), "token": self.api_key},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json() or {}
            rows = data.get("data") or []
            return pd.DataFrame(rows)
        except Exception:
            return pd.DataFrame()

    def estimates(self, ticker: str) -> dict[str, Any]:
        if not self.available():
            return {}
        try:
            resp = requests.get(
                f"{self.BASE}/stock/recommendation",
                params={"symbol": ticker.upper(), "token": self.api_key},
                timeout=20,
            )
            resp.raise_for_status()
            return {"recommendation": resp.json() or []}
        except Exception:
            return {}
