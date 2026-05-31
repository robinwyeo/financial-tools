"""Core library for stock metrics and analyst aggregation."""

from core.config import load_config
from core.scoring import score_ticker, score_universe

__all__ = ["load_config", "score_ticker", "score_universe"]
