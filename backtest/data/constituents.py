"""Historical S&P 500 constituent membership and PIT sector via SIC."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from backtest.constants import DATA_STORE, QUARTER_ENDS
from core.config import ROOT
from core.data import _cache_key, _read_cache, _write_cache
from core.sec import cik_padded, sec_get, ticker_to_cik

logger = logging.getLogger(__name__)

CONSTITUENTS_STORE = DATA_STORE / "constituents"
MEMBERSHIP_PATH = CONSTITUENTS_STORE / "sp500_membership.parquet"
HISTORY_PATH = CONSTITUENTS_STORE / "sp500_history_raw.parquet"
SIC_MAP_PATH = ROOT / "data" / "reference" / "sic_to_sector.csv"

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)

_SIC_RANGES: list[tuple[int, int, str]] | None = None
_TICKER_SECTOR_CACHE: dict[str, str | None] = {}


def _normalize_ticker(ticker: str) -> str:
    """Map historical tickers to yfinance-style symbols."""
    t = ticker.strip().upper()
    if "." in t:
        # BRK.B -> BRK-B
        parts = t.split(".")
        if len(parts) == 2 and len(parts[1]) <= 2:
            return f"{parts[0]}-{parts[1]}"
    return t


def download_sp500_history(force: bool = False) -> pd.DataFrame:
    """Download daily S&P 500 membership history (1996+)."""
    if HISTORY_PATH.exists() and not force:
        return pd.read_parquet(HISTORY_PATH)

    CONSTITUENTS_STORE.mkdir(parents=True, exist_ok=True)
    resp = requests.get(SP500_HISTORY_URL, timeout=120)
    resp.raise_for_status()
    raw = pd.read_csv(
        pd.io.common.StringIO(resp.text),
        dtype={"date": str, "tickers": str},
    )
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date")
    raw.to_parquet(HISTORY_PATH, index=False)
    logger.info("Downloaded S&P 500 history: %d daily rows", len(raw))
    return raw


def build_membership_panel(
    as_of_dates: list[date] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """
    Build quarter-end membership from daily historical constituent lists.
    Returns long panel: quarter_end, ticker.
    """
    if MEMBERSHIP_PATH.exists() and not force and as_of_dates is None:
        return pd.read_parquet(MEMBERSHIP_PATH)

    history = download_sp500_history(force=force)
    if history.empty:
        raise RuntimeError("No S&P 500 history data downloaded")

    if as_of_dates is None:
        as_of_dates = QUARTER_ENDS

    hist_dates = history["date"].sort_values().reset_index(drop=True)
    rows: list[dict] = []

    for qend in as_of_dates:
        qts = pd.Timestamp(qend)
        eligible = hist_dates[hist_dates <= qts]
        if eligible.empty:
            continue
        snap_date = eligible.iloc[-1]
        tickers_raw = history.loc[history["date"] == snap_date, "tickers"].iloc[0]
        tickers = [_normalize_ticker(t) for t in str(tickers_raw).split(",") if t.strip()]
        for ticker in sorted(set(tickers)):
            rows.append({"quarter_end": qend, "ticker": ticker, "snapshot_date": snap_date.date()})

    panel = pd.DataFrame(rows)
    CONSTITUENTS_STORE.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(MEMBERSHIP_PATH, index=False)
    logger.info(
        "Built membership panel: %d quarter-ticker rows across %d dates",
        len(panel),
        panel["quarter_end"].nunique(),
    )
    return panel


def load_membership() -> pd.DataFrame:
    if not MEMBERSHIP_PATH.exists():
        return build_membership_panel()
    return pd.read_parquet(MEMBERSHIP_PATH)


def constituents_as_of(membership: pd.DataFrame, as_of: date) -> list[str]:
    sub = membership[membership["quarter_end"] == as_of]
    return sorted(sub["ticker"].astype(str).unique().tolist())


def load_sic_ranges(path: Path | None = None) -> list[tuple[int, int, str]]:
    global _SIC_RANGES
    if _SIC_RANGES is not None and path is None:
        return _SIC_RANGES
    csv_path = path or SIC_MAP_PATH
    df = pd.read_csv(csv_path)
    ranges = [
        (int(row.sic_from), int(row.sic_to), str(row.sector))
        for row in df.itertuples(index=False)
    ]
    if path is None:
        _SIC_RANGES = ranges
    return ranges


def sector_from_sic(sic: int | str | None) -> str | None:
    if sic is None or (isinstance(sic, float) and pd.isna(sic)):
        return None
    try:
        code = int(str(sic).strip())
    except (TypeError, ValueError):
        return None
    for lo, hi, sector in load_sic_ranges():
        if lo <= code <= hi:
            return sector
    return None


def fetch_sic_for_cik(cik: int, *, force: bool = False) -> int | None:
    """SIC from SEC submissions JSON (cached 30 days)."""
    cache_path = _cache_key("subsic", str(int(cik)))
    if not force:
        cached = _read_cache(cache_path, max_age_hours=24 * 30)
        if cached is not None:
            sic = cached.get("sic")
            try:
                return int(sic) if sic is not None else None
            except (TypeError, ValueError):
                return None
    url = f"https://data.sec.gov/submissions/CIK{cik_padded(cik)}.json"
    try:
        payload = sec_get(url, timeout=30).json()
    except Exception as exc:
        logger.debug("Submissions fetch failed for CIK %s: %s", cik, exc)
        return None
    sic = payload.get("sic") if isinstance(payload, dict) else None
    _write_cache(cache_path, {"sic": sic})
    try:
        return int(sic) if sic is not None else None
    except (TypeError, ValueError):
        return None


def sector_for(cik: int) -> str | None:
    """Yahoo-style sector name from a CIK's SIC code."""
    return sector_from_sic(fetch_sic_for_cik(cik))


def sector_for_ticker(ticker: str) -> str | None:
    key = ticker.upper().strip()
    if key in _TICKER_SECTOR_CACHE:
        return _TICKER_SECTOR_CACHE[key]
    cik = ticker_to_cik(key)
    sector = sector_for(cik) if cik is not None else None
    _TICKER_SECTOR_CACHE[key] = sector
    return sector


def sector_map_for_tickers(tickers: list[str] | tuple[str, ...]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in tickers:
        sector = sector_for_ticker(str(t))
        if sector:
            out[str(t).upper()] = sector
    return out


def price_coverage_pct(
    membership: pd.DataFrame,
    prices: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Share of PIT constituents with a price on each quarter-end."""
    if membership is None or membership.empty or prices is None or prices.empty:
        return []
    px = prices.copy()
    px["Date"] = pd.to_datetime(px["Date"])
    px["ticker"] = px["ticker"].astype(str).str.upper()
    rows: list[dict[str, Any]] = []
    for qend, grp in membership.groupby("quarter_end"):
        names = set(grp["ticker"].astype(str).str.upper())
        if not names:
            continue
        qts = pd.Timestamp(qend)
        window = px[(px["Date"] <= qts) & (px["Date"] > qts - pd.Timedelta(days=10))]
        have = set(window["ticker"].unique()) & names
        rows.append(
            {
                "quarter_end": str(pd.Timestamp(qend).date()),
                "n_constituents": len(names),
                "n_with_price": len(have),
                "price_coverage_pct": len(have) / len(names),
            }
        )
    return rows
