"""SEC EDGAR companyfacts ingestion for the backtest harness.

Public names are preserved: ``load_fundamentals``, ``fundamentals_as_of``,
``fetch_cik_ticker_map``, ``EDGAR_FUNDAMENTALS_PATH``, ``CIK_TICKER_PATH``.
The store is now consolidated companyfacts rows (not FSDS num.txt).
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable

import pandas as pd

from backtest.constants import DATA_STORE, SEC_USER_AGENT
from core.edgar_facts import (
    fetch_companyfacts,
    facts_to_rows,
    flatten_fundamentals,
    fundamentals_as_of_structured,
    rows_from_bulk_zip,
)
from core.sec import ticker_to_cik

logger = logging.getLogger(__name__)

EDGAR_STORE = DATA_STORE / "edgar"
EDGAR_FUNDAMENTALS_PATH = EDGAR_STORE / "fundamentals.parquet"
CIK_TICKER_PATH = EDGAR_STORE / "cik_ticker_map.parquet"
BULK_ZIP_PATH = EDGAR_STORE / "companyfacts.zip"

# Companyfacts rows (not legacy FSDS num.txt: amount/period, no end/filed).
COMPANYFACTS_REQUIRED_COLUMNS = frozenset({"end", "filed", "qtrs", "field", "val", "tag"})

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SESSION_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}

# Kept for callers/tests that inspect the old tag map. Prefer FIELD_TAGS in
# core.edgar_facts for new work.
from core.edgar_facts import TAG_TO_FIELD as TAG_MAP  # noqa: E402


def _sec_get(url: str, timeout: int = 120):
    import requests

    resp = requests.get(url, headers=SESSION_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


def fetch_cik_ticker_map(force: bool = False) -> pd.DataFrame:
    """Download SEC company tickers JSON and return CIK/ticker mapping."""
    if CIK_TICKER_PATH.exists() and not force:
        return pd.read_parquet(CIK_TICKER_PATH)

    EDGAR_STORE.mkdir(parents=True, exist_ok=True)
    data = _sec_get(SEC_TICKERS_URL).json()
    rows = []
    for entry in data.values():
        cik = int(entry["cik_str"])
        ticker = str(entry["ticker"]).upper().strip()
        title = entry.get("title", "")
        rows.append({"cik": cik, "ticker": ticker, "name": title})
    df = pd.DataFrame(rows).drop_duplicates(subset=["cik"], keep="first")
    df.to_parquet(CIK_TICKER_PATH, index=False)
    return df


def ingest_edgar(
    quarters: Iterable[str] | None = None,
    force: bool = False,
    max_quarters: int | None = None,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """
    Ingest consolidated companyfacts for historical S&P 500 members.

    ``quarters`` / ``max_quarters`` are accepted for CLI compatibility but the
    companyfacts payload is a full history per CIK, so they only limit the
    membership window used to choose tickers. ``max_tickers`` caps CIKs.
    """
    del quarters  # history is per-CIK, not per FSDS quarter zip
    if EDGAR_FUNDAMENTALS_PATH.exists() and not force:
        existing = pd.read_parquet(EDGAR_FUNDAMENTALS_PATH)
        missing = COMPANYFACTS_REQUIRED_COLUMNS.difference(existing.columns)
        if not missing:
            return existing
        logger.warning(
            "Existing EDGAR store at %s is missing companyfacts columns %s; "
            "refusing stale or FSDS-shaped parquet. Re-run ingest with --force.",
            EDGAR_FUNDAMENTALS_PATH,
            sorted(missing),
        )
        raise RuntimeError(
            f"EDGAR fundamentals at {EDGAR_FUNDAMENTALS_PATH} are not companyfacts "
            f"schema (missing {sorted(missing)}). Re-run `ingest --force` to rebuild."
        )

    EDGAR_STORE.mkdir(parents=True, exist_ok=True)
    cik_map = fetch_cik_ticker_map(force=force)

    tickers: list[str] = []
    try:
        from backtest.data.constituents import load_membership

        membership = load_membership()
        if max_quarters is not None and not membership.empty:
            qends = sorted(membership["quarter_end"].unique())[:max_quarters]
            membership = membership[membership["quarter_end"].isin(qends)]
        tickers = sorted(membership["ticker"].astype(str).str.upper().unique().tolist())
    except Exception as exc:
        logger.warning("Membership unavailable (%s); falling back to CIK map", exc)
        tickers = cik_map["ticker"].astype(str).str.upper().tolist()

    if max_tickers is not None:
        tickers = tickers[:max_tickers]

    ticker_to_cik_map: dict[str, int] = {}
    for _, row in cik_map.iterrows():
        ticker_to_cik_map[str(row["ticker"]).upper()] = int(row["cik"])

    ciks: list[tuple[str, int]] = []
    for ticker in tickers:
        cik = ticker_to_cik_map.get(ticker)
        if cik is None:
            resolved = ticker_to_cik(ticker)
            if resolved is None:
                logger.debug("No CIK for %s", ticker)
                continue
            cik = resolved
        ciks.append((ticker, int(cik)))

    cik_set = {c for _, c in ciks}
    frames: list[pd.DataFrame] = []
    if BULK_ZIP_PATH.exists():
        logger.info("Parsing bulk companyfacts zip for %d CIKs", len(cik_set))
        bulk = rows_from_bulk_zip(BULK_ZIP_PATH, cik_set)
        if not bulk.empty:
            frames.append(bulk)

    missing = cik_set - set() if not frames else cik_set - set(frames[0]["cik"].unique().tolist()) if frames else cik_set
    if frames and not frames[0].empty:
        have = set(int(c) for c in frames[0]["cik"].unique())
        missing = cik_set - have
    else:
        missing = cik_set

    for i, (ticker, cik) in enumerate(ciks, start=1):
        if cik not in missing:
            continue
        if i % 25 == 0:
            logger.info("Companyfacts %d/%d (%s)", i, len(ciks), ticker)
        payload = fetch_companyfacts(cik)
        if not payload:
            continue
        frame = facts_to_rows(payload, cik)
        if frame.empty:
            continue
        frames.append(frame)

    if not frames:
        raise RuntimeError("No SEC companyfacts ingested")

    df = pd.concat(frames, ignore_index=True)
    reverse = {cik: ticker for ticker, cik in ciks}
    if "ticker" not in df.columns:
        df["ticker"] = df["cik"].map(reverse)
    else:
        df["ticker"] = df["ticker"].fillna(df["cik"].map(reverse))
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df.dropna(subset=["ticker"])
    df = df.sort_values(["ticker", "field", "end", "filed"])
    df = df.drop_duplicates(
        subset=["ticker", "field", "tag", "start", "end", "filed"],
        keep="last",
    )
    df.to_parquet(EDGAR_FUNDAMENTALS_PATH, index=False)
    logger.info("Saved %d companyfacts rows to %s", len(df), EDGAR_FUNDAMENTALS_PATH)
    return df


def load_fundamentals() -> pd.DataFrame:
    if not EDGAR_FUNDAMENTALS_PATH.exists():
        raise FileNotFoundError(
            f"EDGAR fundamentals missing at {EDGAR_FUNDAMENTALS_PATH}. Run ingest first."
        )
    return pd.read_parquet(EDGAR_FUNDAMENTALS_PATH)


def fundamentals_as_of(as_of: date, ticker: str, fundamentals: pd.DataFrame) -> dict[str, float]:
    """Latest TTM/MRQ field values filed on or before ``as_of`` (flat dict)."""
    structured = fundamentals_as_of_structured(as_of, fundamentals, ticker=ticker)
    return flatten_fundamentals(structured)
