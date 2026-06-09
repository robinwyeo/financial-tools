"""Historical S&P 500 constituent membership."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from backtest.constants import DATA_STORE, QUARTER_ENDS

logger = logging.getLogger(__name__)

CONSTITUENTS_STORE = DATA_STORE / "constituents"
MEMBERSHIP_PATH = CONSTITUENTS_STORE / "sp500_membership.parquet"
HISTORY_PATH = CONSTITUENTS_STORE / "sp500_history_raw.parquet"

SP500_HISTORY_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


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
