"""SEC EDGAR Financial Statement Data Sets ingestion."""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from backtest.constants import DATA_STORE, SEC_USER_AGENT

logger = logging.getLogger(__name__)

EDGAR_STORE = DATA_STORE / "edgar"
EDGAR_FUNDAMENTALS_PATH = EDGAR_STORE / "fundamentals.parquet"
CIK_TICKER_PATH = EDGAR_STORE / "cik_ticker_map.parquet"

SEC_DATASETS_URL = "https://www.sec.gov/files/dera/data/financial-statement-data-sets"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# Primary us-gaap tags mapped to internal field names.
TAG_MAP: dict[str, str] = {
    "Assets": "total_assets",
    "Liabilities": "total_liabilities",
    "AssetsCurrent": "current_assets",
    "LiabilitiesCurrent": "current_liabilities",
    "LongTermDebt": "long_term_debt",
    "LongTermDebtNoncurrent": "long_term_debt",
    "CommonStockSharesOutstanding": "shares_outstanding",
    "WeightedAverageNumberOfSharesOutstandingBasic": "shares_outstanding",
    "GrossProfit": "gross_profit",
    "NetIncomeLoss": "net_income",
    "OperatingIncomeLoss": "ebit",
    "Revenues": "revenue",
    "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
    "SalesRevenueNet": "revenue",
    "NetCashProvidedByUsedInOperatingActivities": "operating_cashflow",
    "PaymentsOfDividends": "dividends_paid",
    "PaymentsOfDividendsCommonStock": "dividends_paid",
    "PaymentsForRepurchaseOfCommonStock": "repurchase_of_stock",
    "PaymentsForRepurchaseOfEquity": "repurchase_of_stock",
    "RetainedEarningsAccumulatedDeficit": "retained_earnings",
    "CashAndCashEquivalentsAtCarryingValue": "total_cash",
    "CashCashEquivalentsAndShortTermInvestments": "total_cash",
    "StockholdersEquity": "book_equity",
    "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest": "book_equity",
    "LongTermDebtAndCapitalLeaseObligations": "total_debt",
    "LongTermDebtAndCapitalLeaseObligationsCurrent": "total_debt_current",
    "LongTermDebtAndCapitalLeaseObligationsNoncurrent": "total_debt_noncurrent",
}

SESSION_HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


def _sec_get(url: str, timeout: int = 120) -> requests.Response:
    resp = requests.get(url, headers=SESSION_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return resp


def _quarter_labels(start_year: int = 2009, start_quarter: int = 2, end_year: int = 2025) -> list[str]:
    """SEC dataset labels like 2009q2 through 2025q4."""
    labels: list[str] = []
    for year in range(start_year, end_year + 1):
        q_start = start_quarter if year == start_year else 1
        for quarter in range(q_start, 5):
            labels.append(f"{year}q{quarter}")
    return labels


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


def _parse_submissions(sub_bytes: bytes) -> pd.DataFrame:
    cols = [
        "adsh",
        "cik",
        "name",
        "sic",
        "countryba",
        "stprba",
        "cityba",
        "zipba",
        "bas1",
        "bas2",
        "baph",
        "countryma",
        "stprma",
        "cityma",
        "zipma",
        "mas1",
        "mas2",
        "countryinc",
        "stprinc",
        "ein",
        "former",
        "changed",
        "afs",
        "wksi",
        "form",
        "period",
        "fy",
        "fp",
        "filed",
        "accepted",
        "prevrpt",
        "detail",
        "instance",
        "nciks",
        "aciks",
    ]
    df = pd.read_csv(io.BytesIO(sub_bytes), sep="\t", dtype=str, low_memory=False)
    df = df[[c for c in cols if c in df.columns]]
    df["cik"] = pd.to_numeric(df["cik"], errors="coerce").astype("Int64")
    df["period"] = pd.to_datetime(df["period"], format="%Y%m%d", errors="coerce")
    df["filed"] = pd.to_datetime(df["filed"], format="%Y%m%d", errors="coerce")
    return df


def _parse_numbers(num_bytes: bytes) -> pd.DataFrame:
    usecols = ["adsh", "tag", "version", "ddate", "qtrs", "value", "uval"]
    df = pd.read_csv(
        io.BytesIO(num_bytes),
        sep="\t",
        usecols=lambda c: c in usecols,
        dtype={"adsh": str, "tag": str, "version": str, "qtrs": str},
        low_memory=False,
    )
    df["ddate"] = pd.to_datetime(df["ddate"], format="%Y%m%d", errors="coerce")
    df["qtrs"] = pd.to_numeric(df["qtrs"], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["amount"] = pd.to_numeric(df["value"], errors="coerce")
    else:
        df["amount"] = pd.to_numeric(df.get("uval"), errors="coerce")
    return df


def _download_quarter_zip(label: str) -> bytes | None:
    url = f"{SEC_DATASETS_URL}/{label}.zip"
    try:
        return _sec_get(url).content
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            logger.warning("SEC dataset not found: %s", label)
            return None
        raise


def ingest_quarter(label: str, cik_map: pd.DataFrame) -> pd.DataFrame:
    """Parse one quarterly SEC dataset into normalized fundamentals rows."""
    content = _download_quarter_zip(label)
    if content is None:
        return pd.DataFrame()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        sub_name = next((n for n in zf.namelist() if n.endswith("sub.txt")), None)
        num_name = next((n for n in zf.namelist() if n.endswith("num.txt")), None)
        if not sub_name or not num_name:
            logger.warning("Missing sub/num in %s", label)
            return pd.DataFrame()
        sub = _parse_submissions(zf.read(sub_name))
        num = _parse_numbers(zf.read(num_name))

    # Keep 10-K (annual) and 10-Q (quarterly) filings.
    sub = sub[sub["form"].isin(["10-K", "10-K/A", "10-Q", "10-Q/A"])].copy()
    if sub.empty:
        return pd.DataFrame()

    num = num[num["tag"].isin(TAG_MAP.keys())].copy()
    if num.empty:
        return pd.DataFrame()

    merged = num.merge(sub[["adsh", "cik", "period", "filed", "form"]], on="adsh", how="inner")
    merged["field"] = merged["tag"].map(TAG_MAP)
    merged = merged.dropna(subset=["field", "amount", "filed", "period"])

    # Prefer quarterly rows (qtrs in 1,4) for point-in-time; annual fills gaps.
    merged["is_quarterly"] = merged["qtrs"].isin([1, 4])
    merged = merged.sort_values(
        ["cik", "field", "period", "is_quarterly", "filed"],
        ascending=[True, True, True, False, True],
    )
    merged = merged.drop_duplicates(subset=["cik", "field", "period"], keep="last")

    ticker_map = cik_map.set_index("cik")["ticker"].to_dict()
    merged["ticker"] = merged["cik"].map(ticker_map)
    merged = merged.dropna(subset=["ticker"])
    merged["ticker"] = merged["ticker"].astype(str).str.upper()

    out = merged[
        ["ticker", "cik", "field", "amount", "period", "filed", "form", "tag", "qtrs"]
    ].copy()
    out["dataset_quarter"] = label
    return out


def ingest_edgar(
    quarters: Iterable[str] | None = None,
    force: bool = False,
    max_quarters: int | None = None,
) -> pd.DataFrame:
    """
    Download and normalize SEC fundamentals into a point-in-time parquet store.
    Values are keyed by filing date (when the market could observe them).
    """
    if EDGAR_FUNDAMENTALS_PATH.exists() and not force:
        return pd.read_parquet(EDGAR_FUNDAMENTALS_PATH)

    EDGAR_STORE.mkdir(parents=True, exist_ok=True)
    cik_map = fetch_cik_ticker_map(force=force)
    labels = list(quarters) if quarters else _quarter_labels()
    if max_quarters is not None:
        labels = labels[:max_quarters]

    frames: list[pd.DataFrame] = []
    for i, label in enumerate(labels, start=1):
        logger.info("Ingesting SEC quarter %s (%d/%d)", label, i, len(labels))
        try:
            frame = ingest_quarter(label, cik_map)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            logger.warning("Failed quarter %s: %s", label, exc)

    if not frames:
        raise RuntimeError("No SEC fundamentals ingested")

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ticker", "field", "period", "filed"])
    df = df.drop_duplicates(subset=["ticker", "field", "period", "filed"], keep="last")
    df.to_parquet(EDGAR_FUNDAMENTALS_PATH, index=False)
    logger.info("Saved %d fundamentals rows to %s", len(df), EDGAR_FUNDAMENTALS_PATH)
    return df


def load_fundamentals() -> pd.DataFrame:
    if not EDGAR_FUNDAMENTALS_PATH.exists():
        raise FileNotFoundError(
            f"EDGAR fundamentals missing at {EDGAR_FUNDAMENTALS_PATH}. Run ingest first."
        )
    return pd.read_parquet(EDGAR_FUNDAMENTALS_PATH)


def fundamentals_as_of(as_of: date, ticker: str, fundamentals: pd.DataFrame) -> dict[str, float]:
    """
    Return latest known fundamental field values filed on or before as_of.
    Also returns prior-year values where available for YoY factors.
    """
    as_of_ts = pd.Timestamp(as_of)
    sub = fundamentals[
        (fundamentals["ticker"] == ticker.upper())
        & (fundamentals["filed"] <= as_of_ts)
    ].copy()
    if sub.empty:
        return {}

    latest_by_field: dict[str, tuple[pd.Timestamp, float]] = {}
    for field, grp in sub.groupby("field"):
        row = grp.sort_values(["period", "filed"]).iloc[-1]
        latest_by_field[field] = (row["period"], float(row["amount"]))

    out: dict[str, float] = {}
    for field, (period, amount) in latest_by_field.items():
        out[field] = amount
        prior = sub[
            (sub["field"] == field)
            & (sub["period"] < period)
        ].sort_values(["period", "filed"])
        if not prior.empty:
            out[f"{field}_prior"] = float(prior.iloc[-1]["amount"])

    # Derived fields
    if "book_equity" in out and "shares_outstanding" in out and out["shares_outstanding"]:
        out["book_value"] = out["book_equity"] / out["shares_outstanding"]
    if "total_debt_noncurrent" in out or "total_debt_current" in out:
        out["total_debt"] = (out.get("total_debt_noncurrent") or 0.0) + (
            out.get("total_debt_current") or 0.0
        )
    elif "long_term_debt" in out:
        out["total_debt"] = out["long_term_debt"]

    return out
