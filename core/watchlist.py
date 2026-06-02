"""Load watchlist tickers from the repo watchlist file."""

from __future__ import annotations

from pathlib import Path

from core.config import ROOT

WATCHLIST_PATH = ROOT / "watchlist"


def load_watchlist() -> list[str]:
    """Read tickers from `watchlist` at repo root (one per line, # for comments)."""
    if not WATCHLIST_PATH.exists():
        return []

    tickers: list[str] = []
    for line in WATCHLIST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tickers.append(stripped.upper())
    return tickers
