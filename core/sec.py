"""Shared SEC EDGAR HTTP helpers for the live scoring path."""

from __future__ import annotations

import logging
import time
import pandas as pd
import requests

from core.data import _cache_key, _read_cache, _write_cache

logger = logging.getLogger(__name__)

SEC_USER_AGENT = "financial-tools contact@example.com"
SEC_HEADERS = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
}
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_MIN_INTERVAL_SEC = 0.12
_last_request_at = 0.0


def sec_get(url: str, timeout: int = 30) -> requests.Response:
    """GET with the SEC-required User-Agent and a conservative rate limit."""
    global _last_request_at
    wait = _MIN_INTERVAL_SEC - (time.time() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    resp = requests.get(url, headers=SEC_HEADERS, timeout=timeout)
    _last_request_at = time.time()
    resp.raise_for_status()
    return resp


def fetch_cik_ticker_map(*, force: bool = False) -> pd.DataFrame:
    """CIK/ticker map from SEC company_tickers.json (cached 7 days)."""
    cache_path = _cache_key("cikmap", "sec")
    if not force:
        cached = _read_cache(cache_path, max_age_hours=168)
        if cached and isinstance(cached.get("rows"), list):
            return pd.DataFrame(cached["rows"])

    try:
        from backtest.data.edgar import CIK_TICKER_PATH, fetch_cik_ticker_map as _bt_fetch

        if CIK_TICKER_PATH.exists() and not force:
            df = pd.read_parquet(CIK_TICKER_PATH)
            _write_cache(cache_path, {"rows": df.to_dict(orient="records")})
            return df
        df = _bt_fetch(force=force)
        _write_cache(cache_path, {"rows": df.to_dict(orient="records")})
        return df
    except Exception as exc:
        logger.debug("Backtest CIK map unavailable (%s); fetching from SEC", exc)

    data = sec_get(SEC_TICKERS_URL).json()
    rows = [
        {
            "cik": int(entry["cik_str"]),
            "ticker": str(entry["ticker"]).upper().strip(),
            "name": entry.get("title", ""),
        }
        for entry in data.values()
    ]
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="first")
    _write_cache(cache_path, {"rows": df.to_dict(orient="records")})
    return df


def ticker_to_cik(ticker: str) -> int | None:
    """Resolve a Yahoo-style ticker (BRK-B) to a numeric CIK."""
    key = ticker.upper().strip().replace(".", "-")
    try:
        mapping = fetch_cik_ticker_map()
    except Exception as exc:
        logger.warning("CIK map fetch failed: %s", exc)
        return None
    if mapping.empty:
        return None
    hits = mapping[mapping["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False) == key]
    if hits.empty:
        return None
    cik = hits.iloc[0]["cik"]
    try:
        return int(cik)
    except (TypeError, ValueError):
        return None


def cik_padded(cik: int) -> str:
    return f"{int(cik):010d}"
