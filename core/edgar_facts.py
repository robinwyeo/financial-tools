"""SEC companyfacts: consolidated, duration-aware, filing-date point-in-time facts.

This is the shared ingest path for live 10y history and the backtest harness.
Unlike the old FSDS zip parser, companyfacts rows are consolidated-only (no
segment/coreg overwrite) and carry start/end/filed so TTM can be constructed.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from core.data import _cache_key, _read_cache, _write_cache
from core.sec import cik_padded, sec_get, ticker_to_cik

logger = logging.getLogger(__name__)

ALLOWED_UNITS = frozenset({"USD", "shares", "USD/shares", "pure"})

# Internal field -> us-gaap/dei tags in priority order. First tag with data
# for a given period wins; later tags never silently overwrite.
FIELD_TAGS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
    ],
    "ebit": ["OperatingIncomeLoss"],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_cashflow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "dividends_paid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
    ],
    "repurchase_of_stock": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseNonoperating",
        "InterestPaidNet",
    ],
    "tax_expense": ["IncomeTaxExpenseBenefit"],
    "tax_paid": ["IncomeTaxesPaidNet"],
    "amortization": ["AmortizationOfIntangibleAssets"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAndAmortization",
    ],
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "total_cash": [
        "CashCashEquivalentsAndShortTermInvestments",
        "CashAndCashEquivalentsAtCarryingValue",
    ],
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebtAndCapitalLeaseObligations",
        "LongTermDebt",
    ],
    "debt_st": [
        "LongTermDebtCurrent",
        "DebtCurrent",
        "LongTermDebtAndCapitalLeaseObligationsCurrent",
    ],
    "book_equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "goodwill": ["Goodwill"],
    "ppe_net": ["PropertyPlantAndEquipmentNet"],
    "shares_outstanding": [
        "EntityCommonStockSharesOutstanding",
        "CommonStockSharesOutstanding",
    ],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "eps_diluted": ["EarningsPerShareDiluted"],
}

TAG_TO_FIELD: dict[str, str] = {}
for _field, _tags in FIELD_TAGS.items():
    for _tag in _tags:
        TAG_TO_FIELD.setdefault(_tag, _field)

FLOW_FIELDS = frozenset(
    {
        "revenue",
        "ebit",
        "net_income",
        "gross_profit",
        "operating_cashflow",
        "capex",
        "sbc",
        "dividends_paid",
        "repurchase_of_stock",
        "interest_expense",
        "tax_expense",
        "tax_paid",
        "amortization",
        "pretax_income",
        "depreciation",
        "shares_diluted",
        "eps_diluted",
    }
)

INSTANT_FIELDS = frozenset(
    {
        "total_assets",
        "total_liabilities",
        "current_assets",
        "current_liabilities",
        "total_cash",
        "long_term_debt",
        "debt_st",
        "book_equity",
        "retained_earnings",
        "goodwill",
        "ppe_net",
        "shares_outstanding",
    }
)

BULK_COMPANYFACTS_URL = (
    "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
)


def _parse_date(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def _qtrs_from_span(start: Any, end: Any) -> int:
    """Duration in fiscal quarters; 0 for instants (no start)."""
    end_ts = _parse_date(end)
    start_ts = _parse_date(start)
    if end_ts is None:
        return 0
    if start_ts is None:
        return 0
    days = (end_ts - start_ts).days
    if days <= 0:
        return 0
    return int(round(days / 91.0))


def _iter_fact_entries(facts: dict[str, Any]) -> Iterable[tuple[str, str, str, dict[str, Any]]]:
    """Yield (taxonomy, tag, unit, entry) from a companyfacts payload."""
    taxonomies = facts.get("facts") or {}
    for taxonomy, tags in taxonomies.items():
        if not isinstance(tags, dict):
            continue
        for tag, node in tags.items():
            units = (node or {}).get("units") or {}
            for unit, entries in units.items():
                if unit not in ALLOWED_UNITS:
                    continue
                if not isinstance(entries, list):
                    continue
                for entry in entries:
                    if isinstance(entry, dict):
                        yield taxonomy, tag, unit, entry


def facts_to_rows(facts: dict[str, Any], cik: int) -> pd.DataFrame:
    """Flatten a companyfacts JSON into a normalized row table."""
    rows: list[dict[str, Any]] = []
    for taxonomy, tag, unit, entry in _iter_fact_entries(facts):
        field = TAG_TO_FIELD.get(tag)
        if field is None:
            continue
        end = _parse_date(entry.get("end"))
        if end is None:
            continue
        val = entry.get("val")
        if val is None:
            continue
        try:
            amount = float(val)
        except (TypeError, ValueError):
            continue
        start = _parse_date(entry.get("start"))
        filed = _parse_date(entry.get("filed"))
        qtrs = _qtrs_from_span(start, end)
        rows.append(
            {
                "cik": int(cik),
                "taxonomy": taxonomy,
                "tag": tag,
                "field": field,
                "unit": unit,
                "start": start,
                "end": end,
                "qtrs": qtrs,
                "val": amount,
                "form": str(entry.get("form") or ""),
                "fy": entry.get("fy"),
                "fp": str(entry.get("fp") or ""),
                "filed": filed,
                "accn": str(entry.get("accn") or ""),
                "frame": entry.get("frame"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "cik",
                "taxonomy",
                "tag",
                "field",
                "unit",
                "start",
                "end",
                "qtrs",
                "val",
                "form",
                "fy",
                "fp",
                "filed",
                "accn",
                "frame",
            ]
        )
    df = pd.DataFrame(rows)
    # Prefer framed (consolidated CY) rows when both framed and unframed exist.
    df["_has_frame"] = df["frame"].notna() & (df["frame"].astype(str).str.len() > 0)
    df = df.sort_values(["tag", "start", "end", "filed", "_has_frame"])
    df = df.drop_duplicates(subset=["tag", "start", "end", "filed"], keep="last")
    return df.drop(columns=["_has_frame"]).reset_index(drop=True)


def fetch_companyfacts(cik: int, *, force: bool = False) -> dict[str, Any] | None:
    """Fetch / cache a single CIK companyfacts payload (7-day TTL)."""
    cache_path = _cache_key("cfacts", str(int(cik)))
    if not force:
        cached = _read_cache(cache_path, max_age_hours=168)
        if isinstance(cached, dict) and cached.get("facts"):
            return cached
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_padded(cik)}.json"
    try:
        payload = sec_get(url, timeout=60).json()
    except Exception as exc:
        logger.warning("Companyfacts fetch failed for CIK %s: %s", cik, exc)
        return None
    if isinstance(payload, dict):
        _write_cache(cache_path, payload)
    return payload if isinstance(payload, dict) else None


def download_bulk_companyfacts(dest: Path) -> Path | None:
    """Download the SEC bulk companyfacts zip. Returns dest or None on failure."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = sec_get(BULK_COMPANYFACTS_URL, timeout=300)
        dest.write_bytes(resp.content)
        return dest
    except Exception as exc:
        logger.warning("Bulk companyfacts download failed: %s", exc)
        return None


def rows_from_bulk_zip(zip_path: Path, ciks: set[int] | None = None) -> pd.DataFrame:
    """Parse companyfacts JSON files out of the bulk zip for selected CIKs."""
    frames: list[pd.DataFrame] = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".json"):
                continue
            stem = Path(name).stem.upper()
            digits = "".join(ch for ch in stem if ch.isdigit())
            if not digits:
                continue
            cik = int(digits)
            if ciks is not None and cik not in ciks:
                continue
            import json

            try:
                payload = json.loads(zf.read(name))
            except Exception as exc:
                logger.debug("Skip %s: %s", name, exc)
                continue
            if not isinstance(payload, dict):
                continue
            frame = facts_to_rows(payload, cik)
            if not frame.empty:
                frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def fetch_companyfacts_for_ticker(ticker: str, *, force: bool = False) -> pd.DataFrame:
    cik = ticker_to_cik(ticker)
    if cik is None:
        return pd.DataFrame()
    payload = fetch_companyfacts(cik, force=force)
    if not payload:
        return pd.DataFrame()
    df = facts_to_rows(payload, cik)
    df["ticker"] = ticker.upper().strip()
    return df


def _filed_on_or_before(df: pd.DataFrame, as_of: date | pd.Timestamp) -> pd.DataFrame:
    as_of_ts = pd.Timestamp(as_of).normalize()
    if df.empty or "filed" not in df.columns:
        return df.iloc[0:0]
    filed = pd.to_datetime(df["filed"], errors="coerce")
    return df.loc[filed.notna() & (filed <= as_of_ts)].copy()


def _latest_per_period(sub: pd.DataFrame) -> pd.DataFrame:
    """Keep the last-filed row per (field, end, qtrs, start)."""
    if sub.empty:
        return sub
    out = sub.copy()
    out["_prio"] = [
        _tag_priority_rank(str(f), str(t)) for f, t in zip(out["field"], out["tag"])
    ]
    out = out.sort_values(
        ["field", "end", "qtrs", "start", "_prio", "filed"],
        ascending=[True, True, True, True, True, False],
    )
    out = out.drop_duplicates(subset=["field", "end", "qtrs", "start"], keep="first")
    return out.drop(columns=["_prio"])


def _tag_priority_rank(field: str, tag: str) -> int:
    tags = FIELD_TAGS.get(field, [])
    try:
        return tags.index(tag)
    except ValueError:
        return len(tags) + 1


def _resolve_field_at_period(
    sub: pd.DataFrame,
    field: str,
    *,
    end: pd.Timestamp,
    qtrs: int | None = None,
) -> dict[str, Any] | None:
    rows = sub[(sub["field"] == field) & (pd.to_datetime(sub["end"]) == pd.Timestamp(end))]
    if qtrs is not None:
        rows = rows[rows["qtrs"] == qtrs]
    if rows.empty:
        return None
    rows = rows.copy()
    rows["_prio"] = rows["tag"].map(lambda t: _tag_priority_rank(field, str(t)))
    rows = rows.sort_values(["_prio", "filed"], ascending=[True, False])
    row = rows.iloc[0]
    return {
        "field": field,
        "tag": row["tag"],
        "val": float(row["val"]),
        "end": pd.Timestamp(row["end"]),
        "start": row.get("start"),
        "qtrs": int(row["qtrs"]) if pd.notna(row.get("qtrs")) else 0,
        "filed": row.get("filed"),
        "form": row.get("form"),
        "fp": row.get("fp"),
        "accn": row.get("accn"),
    }


def _quarterly_durations(sub: pd.DataFrame, field: str) -> pd.DataFrame:
    """Build a unique quarterly (qtrs=1) series, deriving Q4 from FY - 9M YTD."""
    rows = sub[sub["field"] == field].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["end"] = pd.to_datetime(rows["end"])
    rows = rows.sort_values(["end", "filed"])

    derived: list[dict[str, Any]] = []
    by_end: dict[pd.Timestamp, dict[int, dict[str, Any]]] = {}
    for _, row in rows.iterrows():
        end = pd.Timestamp(row["end"]).normalize()
        qtrs = int(row["qtrs"]) if pd.notna(row["qtrs"]) else 0
        prio = _tag_priority_rank(field, str(row["tag"]))
        bucket = by_end.setdefault(end, {})
        existing = bucket.get(qtrs)
        if existing is None or prio < existing["_prio"] or (
            prio == existing["_prio"] and pd.Timestamp(row["filed"]) >= pd.Timestamp(existing["filed"])
        ):
            bucket[qtrs] = {
                "end": end,
                "qtrs": qtrs,
                "val": float(row["val"]),
                "tag": row["tag"],
                "filed": row["filed"],
                "form": row.get("form"),
                "fp": row.get("fp"),
                "start": row.get("start"),
                "accn": row.get("accn"),
                "_prio": prio,
            }

    for end, bucket in sorted(by_end.items()):
        if 1 in bucket:
            derived.append(bucket[1])
            continue
        # Derive a quarter from YTD differences when a 1-quarter row is missing.
        if 2 in bucket:
            # Q2 = YTD6 - Q1 (same year, ~90 days earlier)
            q1_end = end - pd.DateOffset(months=3)
            q1 = None
            for cand_end, cand_bucket in by_end.items():
                if abs((pd.Timestamp(cand_end) - pd.Timestamp(q1_end)).days) <= 15 and 1 in cand_bucket:
                    q1 = cand_bucket[1]
                    break
            if q1 is not None:
                derived.append(
                    {
                        **bucket[2],
                        "qtrs": 1,
                        "val": bucket[2]["val"] - q1["val"],
                        "fp": "Q2",
                    }
                )
        if 3 in bucket:
            ytd6_end = end - pd.DateOffset(months=3)
            ytd6 = None
            for cand_end, cand_bucket in by_end.items():
                if abs((pd.Timestamp(cand_end) - pd.Timestamp(ytd6_end)).days) <= 15 and 2 in cand_bucket:
                    ytd6 = cand_bucket[2]
                    break
            if ytd6 is not None:
                derived.append(
                    {
                        **bucket[3],
                        "qtrs": 1,
                        "val": bucket[3]["val"] - ytd6["val"],
                        "fp": "Q3",
                    }
                )
        if 4 in bucket:
            ytd9_end = end - pd.DateOffset(months=3)
            ytd9 = None
            for cand_end, cand_bucket in by_end.items():
                if abs((pd.Timestamp(cand_end) - pd.Timestamp(ytd9_end)).days) <= 15 and 3 in cand_bucket:
                    ytd9 = cand_bucket[3]
                    break
            if ytd9 is not None:
                derived.append(
                    {
                        **bucket[4],
                        "qtrs": 1,
                        "val": bucket[4]["val"] - ytd9["val"],
                        "fp": "Q4",
                    }
                )

    if not derived:
        return pd.DataFrame()
    out = pd.DataFrame(derived)
    out["end"] = pd.to_datetime(out["end"])
    out = out.sort_values(["end", "filed"])
    out = out.drop_duplicates(subset=["end"], keep="last")
    return out.reset_index(drop=True)


def _ttm_for_field(sub: pd.DataFrame, field: str) -> dict[str, Any] | None:
    quarters = _quarterly_durations(sub, field)
    if not quarters.empty and len(quarters) >= 4:
        last4 = quarters.tail(4)
        span_days = (last4["end"].iloc[-1] - last4["end"].iloc[0]).days
        if 240 <= span_days <= 400:
            return {
                "field": field,
                "val": float(last4["val"].sum()),
                "tag": last4["tag"].iloc[-1],
                "end": last4["end"].iloc[-1],
                "filed": last4["filed"].iloc[-1],
                "form": last4["form"].iloc[-1] if "form" in last4.columns else None,
                "qtrs": 4,
                "basis": "ttm_sum",
                "n_quarters": 4,
            }
    # Fall back to latest annual (qtrs=4).
    annuals = sub[(sub["field"] == field) & (sub["qtrs"] == 4)].copy()
    if annuals.empty:
        return None
    annuals["_prio"] = annuals["tag"].map(lambda t: _tag_priority_rank(field, str(t)))
    annuals["end"] = pd.to_datetime(annuals["end"])
    annuals = annuals.sort_values(["end", "_prio", "filed"], ascending=[True, True, False])
    latest_end = annuals["end"].max()
    row = annuals[annuals["end"] == latest_end].iloc[0]
    return {
        "field": field,
        "val": float(row["val"]),
        "tag": row["tag"],
        "end": pd.Timestamp(row["end"]),
        "filed": row.get("filed"),
        "form": row.get("form"),
        "qtrs": 4,
        "basis": "annual",
        "n_quarters": 1,
    }


def _fy_series(sub: pd.DataFrame, field: str) -> pd.DataFrame:
    annuals = sub[(sub["field"] == field) & (sub["qtrs"] == 4)].copy()
    if annuals.empty:
        # FY-flagged instants or fp==FY durations with qtrs != 4 still count as annual.
        annuals = sub[
            (sub["field"] == field)
            & (
                sub["fp"].astype(str).str.upper().eq("FY")
                | sub["form"].astype(str).str.contains("10-K", na=False)
            )
        ].copy()
    if annuals.empty:
        return pd.DataFrame()
    annuals["_prio"] = annuals["tag"].map(lambda t: _tag_priority_rank(field, str(t)))
    annuals["end"] = pd.to_datetime(annuals["end"])
    annuals = annuals.sort_values(
        ["end", "_prio", "filed"], ascending=[True, True, False]
    )
    annuals = annuals.drop_duplicates(subset=["end"], keep="first")
    return annuals.reset_index(drop=True)


def _instant_at(sub: pd.DataFrame, field: str, as_of_end: pd.Timestamp | None = None) -> dict[str, Any] | None:
    rows = sub[sub["field"] == field].copy()
    if rows.empty:
        return None
    rows["end"] = pd.to_datetime(rows["end"])
    if as_of_end is not None:
        rows = rows[rows["end"] <= pd.Timestamp(as_of_end)]
    if rows.empty:
        return None
    rows["_prio"] = rows["tag"].map(lambda t: _tag_priority_rank(field, str(t)))
    rows = rows.sort_values(["end", "_prio", "filed"], ascending=[True, True, False])
    latest_end = rows["end"].max()
    row = rows[rows["end"] == latest_end].iloc[0]
    return {
        "field": field,
        "val": float(row["val"]),
        "tag": row["tag"],
        "end": pd.Timestamp(row["end"]),
        "filed": row.get("filed"),
        "form": row.get("form"),
        "qtrs": int(row["qtrs"]) if pd.notna(row.get("qtrs")) else 0,
    }


def fundamentals_as_of_structured(
    as_of: date | pd.Timestamp,
    facts_df: pd.DataFrame,
    *,
    ticker: str | None = None,
) -> dict[str, Any]:
    """TTM flows + FY flows + MRQ instants filed on or before ``as_of``."""
    empty = {
        "flow_ttm": {},
        "flow_fy": {},
        "flow_fy_prior": {},
        "instant_mrq": {},
        "instant_prior_fy": {},
        "basis": {},
        "as_of": pd.Timestamp(as_of).date() if hasattr(pd.Timestamp(as_of), "date") else as_of,
        "ticker": ticker,
    }
    if facts_df is None or facts_df.empty:
        return empty
    sub = facts_df
    if ticker and "ticker" in sub.columns:
        sub = sub[sub["ticker"].astype(str).str.upper() == ticker.upper()]
    sub = _filed_on_or_before(sub, as_of)
    if sub.empty:
        return empty
    sub = _latest_per_period(sub)

    flow_ttm: dict[str, float] = {}
    flow_fy: dict[str, float] = {}
    flow_fy_prior: dict[str, float] = {}
    instant_mrq: dict[str, float] = {}
    instant_prior_fy: dict[str, float] = {}
    basis: dict[str, Any] = {}

    for field in FLOW_FIELDS:
        ttm = _ttm_for_field(sub, field)
        if ttm is not None:
            flow_ttm[field] = ttm["val"]
            basis[f"ttm_{field}"] = {k: ttm[k] for k in ("tag", "end", "filed", "basis") if k in ttm}
        fy = _fy_series(sub, field)
        if not fy.empty:
            flow_fy[field] = float(fy.iloc[-1]["val"])
            basis[f"fy_{field}"] = {
                "tag": fy.iloc[-1]["tag"],
                "end": fy.iloc[-1]["end"],
                "filed": fy.iloc[-1]["filed"],
            }
            if len(fy) >= 2:
                flow_fy_prior[field] = float(fy.iloc[-2]["val"])

    fy_end = None
    if flow_fy:
        # Use total_assets FY-end if present, else max flow FY end from basis.
        asset_fy = _fy_series(sub, "total_assets")
        if not asset_fy.empty:
            fy_end = pd.Timestamp(asset_fy.iloc[-1]["end"])
        else:
            ends = [v.get("end") for k, v in basis.items() if k.startswith("fy_") and v.get("end") is not None]
            if ends:
                fy_end = max(pd.Timestamp(e) for e in ends)

    for field in INSTANT_FIELDS:
        mrq = _instant_at(sub, field)
        if mrq is not None:
            instant_mrq[field] = mrq["val"]
            basis[f"mrq_{field}"] = {k: mrq[k] for k in ("tag", "end", "filed") if k in mrq}
        if fy_end is not None:
            fy_instant = _instant_at(sub, field, as_of_end=fy_end)
            prior_cut = fy_end - pd.DateOffset(days=200)
            prior = _instant_at(sub, field, as_of_end=prior_cut)
            if prior is not None:
                instant_prior_fy[field] = prior["val"]
            elif fy_instant is not None:
                # At least expose the FY-end instant as current FY.
                pass

    # total_debt = LT + ST when either is present.
    for bucket in (instant_mrq, instant_prior_fy):
        lt = bucket.get("long_term_debt")
        st = bucket.get("debt_st")
        if lt is not None or st is not None:
            bucket["total_debt"] = (lt or 0.0) + (st or 0.0)

    return {
        "flow_ttm": flow_ttm,
        "flow_fy": flow_fy,
        "flow_fy_prior": flow_fy_prior,
        "instant_mrq": instant_mrq,
        "instant_prior_fy": instant_prior_fy,
        "basis": basis,
        "as_of": pd.Timestamp(as_of).date(),
        "ticker": ticker,
    }


def flatten_fundamentals(structured: dict[str, Any]) -> dict[str, float]:
    """Map structured TTM/MRQ views onto the historic flat field names.

    Flows use TTM (fallback FY). Instants use MRQ. ``*_prior`` uses FY-prior
    flows and prior-FY instants so Piotroski-style YoY still has a value.
    """
    out: dict[str, float] = {}
    ttm = structured.get("flow_ttm") or {}
    fy = structured.get("flow_fy") or {}
    fy_prior = structured.get("flow_fy_prior") or {}
    mrq = structured.get("instant_mrq") or {}
    prior = structured.get("instant_prior_fy") or {}

    for field in FLOW_FIELDS:
        val = ttm.get(field)
        if val is None:
            val = fy.get(field)
        if val is not None:
            out[field] = float(val)
        if field in fy_prior:
            out[f"{field}_prior"] = float(fy_prior[field])

    for field, val in mrq.items():
        out[field] = float(val)
    for field, val in prior.items():
        out[f"{field}_prior"] = float(val)

    if "book_equity" in out and "shares_outstanding" in out and out["shares_outstanding"]:
        out["book_value"] = out["book_equity"] / out["shares_outstanding"]
    if "shares_outstanding" in prior:
        out["shares_outstanding_prior"] = float(prior["shares_outstanding"])
    if "operating_cashflow" in out and "capex" in out:
        # CapEx is typically reported as a cash outflow (negative) or a positive spend.
        capex = out["capex"]
        ocf = out["operating_cashflow"]
        spend = abs(capex)
        out["free_cashflow"] = ocf - spend
    return out


def annual_series(
    facts_df: pd.DataFrame,
    field: str,
    *,
    years: int = 10,
    as_of: date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """FY-end series (period, value, filed, tag) for one field."""
    if facts_df is None or facts_df.empty:
        return pd.DataFrame(columns=["period", "value", "filed", "tag"])
    sub = facts_df
    if as_of is not None:
        sub = _filed_on_or_before(sub, as_of)
    fy = _fy_series(sub, field)
    if fy.empty:
        return pd.DataFrame(columns=["period", "value", "filed", "tag"])
    cutoff = pd.Timestamp.now().normalize() - pd.DateOffset(years=years)
    fy = fy[pd.to_datetime(fy["end"]) >= cutoff]
    renamed = fy.rename(columns={"end": "period", "val": "value"})
    cols = [c for c in ("period", "value", "filed", "tag") if c in renamed.columns]
    return renamed[cols].reset_index(drop=True)


def period_table_from_facts(facts_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Annual period table used by valuation-vs-history (EBIT, OCF, equity, shares, debt, cash)."""
    if facts_df is None or facts_df.empty:
        return pd.DataFrame()
    sub = facts_df
    if "ticker" in sub.columns:
        sub = sub[sub["ticker"].astype(str).str.upper() == ticker.upper()]
    if sub.empty:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    mapping = {
        "ebit": "ebit",
        "operating_cashflow": "operating_cashflow",
        "book_equity": "book_equity",
        "shares_outstanding": "shares_outstanding",
        "long_term_debt": "total_debt",
        "total_cash": "total_cash",
    }
    for src, dest in mapping.items():
        fy = _fy_series(sub, src)
        if fy.empty:
            continue
        piece = fy[["end", "val"]].rename(columns={"end": "period", "val": dest})
        frames.append(piece)
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for extra in frames[1:]:
        out = out.merge(extra, on="period", how="outer")
    if "total_debt" not in out.columns:
        st = _fy_series(sub, "debt_st")
        if not st.empty:
            piece = st[["end", "val"]].rename(columns={"end": "period", "val": "debt_st"})
            out = out.merge(piece, on="period", how="outer")
            if "total_debt" not in out.columns:
                out["total_debt"] = out.get("debt_st")
    return out.sort_values("period").reset_index(drop=True)
