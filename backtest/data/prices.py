"""Historical price ingestion and return utilities."""

from __future__ import annotations

import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.constants import DATA_STORE, DEFAULT_DELIST_RETURN
from backtest.data.constituents import load_membership

logger = logging.getLogger(__name__)

PRICES_STORE = DATA_STORE / "prices"
PRICES_PANEL_PATH = PRICES_STORE / "daily_prices.parquet"
DELISTED_PATH = PRICES_STORE / "delisted_tickers.parquet"
BENCHMARK_TICKER = "SPY"


def _normalize_price_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], utc=True).dt.tz_localize(None)
        out = out.set_index("Date")
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out.sort_index()
    out["ticker"] = ticker.upper()
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume", "ticker"] if c in out.columns]
    return out[keep]


def fetch_ticker_prices(ticker: str, period: str = "max") -> pd.DataFrame:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d", auto_adjust=True)
        return _normalize_price_frame(hist, ticker)
    except Exception as exc:
        logger.debug("Price fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _all_tickers_from_membership() -> list[str]:
    membership = load_membership()
    tickers = sorted(membership["ticker"].astype(str).str.upper().unique().tolist())
    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)
    return tickers


def ingest_prices(
    tickers: list[str] | None = None,
    throttle_sec: float = 0.15,
    force: bool = False,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """Bulk-download max-history prices and cache as parquet."""
    if PRICES_PANEL_PATH.exists() and not force:
        return pd.read_parquet(PRICES_PANEL_PATH)

    PRICES_STORE.mkdir(parents=True, exist_ok=True)
    tickers = tickers or _all_tickers_from_membership()
    if max_tickers is not None:
        tickers = tickers[: max(0, max_tickers - 1)]
    if BENCHMARK_TICKER not in tickers:
        tickers.append(BENCHMARK_TICKER)
    frames: list[pd.DataFrame] = []
    delisted: list[str] = []

    for i, ticker in enumerate(tickers, start=1):
        if i % 25 == 0:
            logger.info("Downloading prices %d/%d", i, len(tickers))
        frame = fetch_ticker_prices(ticker)
        if frame.empty:
            delisted.append(ticker)
        else:
            frames.append(frame.reset_index().rename(columns={"index": "Date"}))
        time.sleep(throttle_sec)

    if not frames:
        raise RuntimeError("No price data downloaded")

    panel = pd.concat(frames, ignore_index=True)
    panel["Date"] = pd.to_datetime(panel["Date"])
    panel.to_parquet(PRICES_PANEL_PATH, index=False)

    pd.DataFrame({"ticker": delisted}).to_parquet(DELISTED_PATH, index=False)
    logger.info(
        "Saved prices for %d tickers; %d delisted/missing",
        panel["ticker"].nunique(),
        len(delisted),
    )
    return panel


def load_prices() -> pd.DataFrame:
    if not PRICES_PANEL_PATH.exists():
        return ingest_prices()
    return pd.read_parquet(PRICES_PANEL_PATH)


def ensure_benchmark_prices(prices: pd.DataFrame, ticker: str = BENCHMARK_TICKER) -> pd.DataFrame:
    """Return prices with benchmark ticker present, fetching on demand if missing."""
    if not prices.empty and ticker in prices["ticker"].astype(str).str.upper().unique():
        return prices
    logger.info("Benchmark %s missing from cache; fetching from yfinance", ticker)
    frame = fetch_ticker_prices(ticker)
    if frame.empty:
        return prices
    extra = frame.reset_index().rename(columns={"index": "Date"})
    if "Date" not in extra.columns and frame.index.name:
        extra = frame.reset_index()
    extra["Date"] = pd.to_datetime(extra["Date"])
    return pd.concat([prices, extra], ignore_index=True)


def load_delisted_catalog() -> set[str]:
    if not DELISTED_PATH.exists():
        return set()
    return set(pd.read_parquet(DELISTED_PATH)["ticker"].astype(str).str.upper())


def price_on_or_before(prices: pd.DataFrame, ticker: str, as_of: date) -> float | None:
    sub = prices[(prices["ticker"] == ticker.upper()) & (prices["Date"] <= pd.Timestamp(as_of))]
    if sub.empty:
        return None
    return float(sub.sort_values("Date").iloc[-1]["Close"])


def price_history_as_of(
    prices: pd.DataFrame,
    ticker: str,
    as_of: date,
    lookback_days: int = 400,
) -> pd.DataFrame:
    sub = prices[
        (prices["ticker"] == ticker.upper())
        & (prices["Date"] <= pd.Timestamp(as_of))
    ].sort_values("Date")
    if sub.empty:
        return pd.DataFrame()
    return sub.tail(lookback_days).set_index("Date")


def monthly_returns(
    prices: pd.DataFrame,
    tickers: list[str],
    start: date,
    end: date,
    delist_return: float = DEFAULT_DELIST_RETURN,
) -> pd.DataFrame:
    """
    Compute monthly total returns for tickers between start and end.
    Missing price paths (delisted) get delist_return applied once at first missing month.
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sub = prices[
        (prices["ticker"].isin([t.upper() for t in tickers]))
        & (prices["Date"] >= start_ts)
        & (prices["Date"] <= end_ts)
    ].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.sort_values(["ticker", "Date"])
    rows: list[dict] = []
    for ticker, grp in sub.groupby("ticker"):
        series = grp.set_index("Date")["Close"].resample("ME").last().pct_change()
        for dt, ret in series.items():
            if pd.notna(ret):
                rows.append({"month": pd.Timestamp(dt), "ticker": ticker, "return": float(ret)})
    if not rows:
        return pd.DataFrame(columns=["month", "ticker", "return"])
    return pd.DataFrame(rows)


def portfolio_monthly_returns(
    holdings_by_month: dict[pd.Timestamp, list[str]],
    prices: pd.DataFrame,
    delist_return: float = DEFAULT_DELIST_RETURN,
) -> pd.Series:
    """Equal-weight portfolio monthly returns from month -> tickers map."""
    all_tickers = sorted({t for ts in holdings_by_month for t in holdings_by_month[ts]})
    if not all_tickers:
        return pd.Series(dtype=float)

    months = sorted(holdings_by_month.keys())
    start = months[0]
    end = months[-1]
    ticker_returns = monthly_returns(prices, all_tickers, start.date(), end.date(), delist_return)
    if ticker_returns.empty:
        return pd.Series(dtype=float)

    rets: list[tuple[pd.Timestamp, float]] = []
    delisted = load_delisted_catalog()
    for month in months:
        tickers = holdings_by_month[month]
        if not tickers:
            continue
        sub = ticker_returns[
            (ticker_returns["month"] == month) & (ticker_returns["ticker"].isin(tickers))
        ]
        if sub.empty:
            continue
        values = sub.set_index("ticker")["return"]
        # Penalize delisted names with no return observation.
        for t in tickers:
            if t not in values.index and t in delisted:
                values.loc[t] = delist_return
        if values.empty:
            continue
        rets.append((month, float(values.mean())))
    if not rets:
        return pd.Series(dtype=float)
    out = pd.Series({m: r for m, r in rets}).sort_index()
    return out


def benchmark_monthly_returns(
    prices: pd.DataFrame,
    start: date,
    end: date,
    ticker: str = BENCHMARK_TICKER,
) -> pd.Series:
    monthly = monthly_returns(prices, [ticker], start, end)
    if monthly.empty:
        return pd.Series(dtype=float)
    return monthly.set_index("month")["return"].sort_index()
