"""Market-data adapters. Scoring talks to this layer, not to yfinance/SEC directly."""

from __future__ import annotations

from core.providers.base import FundamentalsProvider, get_fundamentals_providers

__all__ = ["FundamentalsProvider", "get_fundamentals_providers"]
