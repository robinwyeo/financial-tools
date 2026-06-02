"""Load watchlist tickers from the repo watchlist file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from core.config import ROOT, load_config

WATCHLIST_PATH = ROOT / "watchlist"


def load_watchlist(config: dict[str, Any] | None = None) -> list[str]:
    """
    Read tickers from `watchlist` at repo root (one per line).
    Falls back to config.yaml `watchlist` if the file is missing or empty.
    """
    tickers: list[str] = []
    if WATCHLIST_PATH.exists():
        for line in WATCHLIST_PATH.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            tickers.append(stripped.upper())

    if tickers:
        return tickers

    cfg = config or load_config()
    return [str(t).upper().strip() for t in cfg.get("watchlist", []) if str(t).strip()]
