"""Universe snapshot builder for cross-sectional scoring."""

from __future__ import annotations

import json
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
SNAPSHOT_META_PATH = DATA_DIR / "universe_snapshot.meta.json"


class SnapshotIncompleteError(RuntimeError):
    """Raised when too few tickers succeeded; the parquet is left untouched."""


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


def _wiki_symbol_list(url: str, fallback: list[str], *, suffix: str = "") -> list[str]:
    import io
    import urllib.request

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
        symbol_col = None
        for cand in ("Symbol", "Ticker", "Ticker symbol"):
            if cand in df.columns:
                symbol_col = cand
                break
        if symbol_col is None:
            symbol_col = df.columns[0]
        tickers = df[symbol_col].astype(str).str.replace(".", "-", regex=False).str.upper().tolist()
        if suffix:
            tickers = [t if t.endswith(suffix) else f"{t}{suffix}" for t in tickers]
        logger.info("Fetched %d tickers from %s", len(tickers), url)
        return tickers
    except Exception as exc:
        logger.warning("Wikipedia fetch failed (%s): %s; using fallback", url, exc)
        tickers = list(fallback)
        if suffix:
            tickers = [t if str(t).endswith(suffix) else f"{t}{suffix}" for t in tickers]
        return tickers


def fetch_sp500_tickers() -> list[str]:
    """Fetch S&P 500 constituents from Wikipedia."""
    return _wiki_symbol_list(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        _fallback_sp500(),
    )


def fetch_sp400_tickers() -> list[str]:
    return _wiki_symbol_list(
        "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
        _fallback_sp400(),
    )


def fetch_tsx60_tickers() -> list[str]:
    return _wiki_symbol_list(
        "https://en.wikipedia.org/wiki/S%26P/TSX_60",
        _fallback_tsx60(),
        suffix=".TO",
    )


def _fallback_sp400() -> list[str]:
    return ["RSG", "GWW", "URI", "FICO", "WAB", "CSL", "NDSN", "IEX", "PKG", "TRMB"]


def _fallback_tsx60() -> list[str]:
    return [
        "RY.TO", "TD.TO", "SHOP.TO", "ENB.TO", "CNQ.TO", "CP.TO", "CNR.TO",
        "BMO.TO", "BNS.TO", "SU.TO",
    ]


def prefer_us_listings(tickers: list[str]) -> list[str]:
    """Drop `.TO` dual listings when a US ticker of the same root is present."""
    upper = [str(t).upper() for t in tickers]
    us_roots = {t.split(".")[0] for t in upper if not t.endswith(".TO")}
    out: list[str] = []
    seen: set[str] = set()
    for t in upper:
        if t in seen:
            continue
        if t.endswith(".TO") and t[:-3] in us_roots:
            continue
        seen.add(t)
        out.append(t)
    return out


def fetch_universe_tickers(members: list[str]) -> list[tuple[str, str]]:
    """Return (ticker, universe_name) pairs for the configured members."""
    fetchers = {
        "sp500": fetch_sp500_tickers,
        "sp400": fetch_sp400_tickers,
        "tsx60": fetch_tsx60_tickers,
    }
    tagged: list[tuple[str, str]] = []
    for member in members:
        key = str(member).lower()
        fn = fetchers.get(key)
        if fn is None:
            logger.warning("Unknown universe member %s", member)
            continue
        for ticker in fn():
            tagged.append((ticker.upper(), key))
    preferred = set(prefer_us_listings([t for t, _ in tagged]))
    return [(t, u) for t, u in tagged if t in preferred]


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
    min_success_ratio: float = 0.9,
    universes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Build cross-sectional factor snapshot for the universe.

    Writes nothing if fewer than ``min_success_ratio`` of tickers succeed.
    """
    from core.config import get_universe_members, load_config

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    universe_tag: dict[str, str] = {}
    if tickers is None:
        members = universes or get_universe_members(load_config())
        tagged = fetch_universe_tickers(members)
        tickers = [t for t, _ in tagged]
        universe_tag = {t: u for t, u in tagged}
    else:
        tickers = [str(t).upper() for t in tickers]
        for t in tickers:
            universe_tag[t] = "custom"
    if max_tickers:
        tickers = tickers[:max_tickers]

    rows = []
    failures: list[dict[str, str]] = []
    for i, ticker in enumerate(tickers):
        try:
            raw = build_raw_metrics(ticker)
            factors = compute_all_factors(raw)
            currency = raw.get("currency") or "USD"
            country = "CA" if str(ticker).endswith(".TO") else "US"
            row = {
                "ticker": ticker.upper(),
                "name": raw.get("name"),
                "sector": raw.get("sector"),
                "industry": raw.get("industry"),
                "universe": universe_tag.get(ticker.upper(), "sp500"),
                "currency": currency,
                "country": country,
                **factors,
            }
            rows.append(row)
            if (i + 1) % 25 == 0:
                logger.info("Processed %d / %d tickers", i + 1, len(tickers))
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            failures.append({"ticker": ticker, "error": str(exc)})
        throttle(throttle_seconds)

    n = len(tickers)
    success_ratio = (len(rows) / n) if n else 0.0
    if n and success_ratio < min_success_ratio:
        raise SnapshotIncompleteError(
            f"Universe snapshot incomplete: {len(rows)}/{n} succeeded "
            f"({success_ratio:.0%} < {min_success_ratio:.0%}); parquet not written"
        )

    df = pd.DataFrame(rows)
    snapshot_date = datetime.now(timezone.utc).isoformat()
    if not df.empty:
        df["snapshot_date"] = snapshot_date
        to_write = df
        try:
            from core.config import load_config
            from core.scoring import score_universe_df

            to_write = score_universe_df(df, load_config())
        except Exception as exc:
            logger.warning(
                "Universe snapshot scoring failed; writing raw factors: %s", exc
            )
        to_write.to_parquet(SNAPSHOT_PATH, index=False)
        logger.info(
            "Saved universe snapshot with %d tickers to %s", len(to_write), SNAPSHOT_PATH
        )
        df = to_write
    meta = {
        "snapshot_date": snapshot_date,
        "attempted": n,
        "succeeded": len(rows),
        "failed": len(failures),
        "failures": failures[:50],
        "path": str(SNAPSHOT_PATH),
    }
    SNAPSHOT_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build universe factor snapshot")
    parser.add_argument("--max", type=int, default=None, help="Max tickers to process")
    parser.add_argument("--fast", action="store_true", help="Use fallback smaller universe")
    parser.add_argument(
        "--universes",
        default=None,
        help="Comma-separated universe members (default: config universe.members)",
    )
    args = parser.parse_args()

    if args.fast:
        tickers = _fallback_sp500()
        build_universe_snapshot(tickers=tickers, max_tickers=args.max)
    elif args.universes:
        members = [m.strip() for m in args.universes.split(",") if m.strip()]
        build_universe_snapshot(universes=members, max_tickers=args.max)
    else:
        build_universe_snapshot(max_tickers=args.max)
