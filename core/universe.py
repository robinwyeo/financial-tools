"""Universe snapshot builder for cross-sectional scoring."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.config import ROOT
from core.data import build_raw_metrics, throttle
from core.factors import compute_all_factors

logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
SNAPSHOT_PATH = DATA_DIR / "universe_snapshot.parquet"


def snapshot_path() -> Path:
    return SNAPSHOT_PATH


def load_universe_snapshot() -> pd.DataFrame | None:
    if not SNAPSHOT_PATH.exists():
        return None
    try:
        return pd.read_parquet(SNAPSHOT_PATH)
    except Exception as exc:
        logger.warning("Failed to load universe snapshot: %s", exc)
        return None


def fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    import io
    import urllib.request

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )
    }
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(io.StringIO(html))
        df = tables[0]
        symbol_col = "Symbol" if "Symbol" in df.columns else df.columns[0]
        tickers = df[symbol_col].astype(str).str.replace(".", "-", regex=False).tolist()
        logger.info("Fetched %d S&P 500 tickers from Wikipedia", len(tickers))
        return tickers
    except Exception as exc:
        logger.warning("Wikipedia S&P 500 fetch failed: %s; using fallback list", exc)
        return _fallback_sp500()


def _fallback_sp500() -> list[str]:
    """Subset fallback when Wikipedia is unavailable."""
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "BRK-B", "JPM", "V", "JNJ",
        "UNH", "XOM", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "KO", "PEP",
        "COST", "AVGO", "WMT", "MCD", "CSCO", "TMO", "ACN", "ABT", "DHR", "NEE",
        "LIN", "TXN", "PM", "UNP", "HON", "QCOM", "LOW", "INTC", "AMD", "IBM",
        "GE", "CAT", "BA", "GS", "MS", "BLK", "AXP", "SPGI", "DE", "RTX",
    ]


def build_universe_snapshot(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    throttle_seconds: float = 0.25,
) -> pd.DataFrame:
    """
    Build cross-sectional factor snapshot for the universe.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    universe = tickers or fetch_sp500_tickers()
    if max_tickers:
        universe = universe[:max_tickers]

    rows = []
    for i, ticker in enumerate(universe):
        try:
            raw = build_raw_metrics(ticker)
            factors = compute_all_factors(raw)
            row = {
                "ticker": ticker.upper(),
                "name": raw.get("name"),
                "sector": raw.get("sector"),
                "industry": raw.get("industry"),
                **factors,
            }
            rows.append(row)
            if (i + 1) % 25 == 0:
                logger.info("Processed %d / %d tickers", i + 1, len(universe))
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
        throttle(throttle_seconds)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["snapshot_date"] = datetime.now(timezone.utc).isoformat()
        df.to_parquet(SNAPSHOT_PATH, index=False)
        logger.info("Saved universe snapshot with %d tickers to %s", len(df), SNAPSHOT_PATH)
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build universe factor snapshot")
    parser.add_argument("--max", type=int, default=None, help="Max tickers to process")
    parser.add_argument("--fast", action="store_true", help="Use fallback smaller universe")
    args = parser.parse_args()

    if args.fast:
        tickers = _fallback_sp500()
    else:
        tickers = fetch_sp500_tickers()

    build_universe_snapshot(tickers=tickers, max_tickers=args.max)
