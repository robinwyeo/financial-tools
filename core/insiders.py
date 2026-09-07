"""SEC Form 4 open-market insider cluster-buy signal."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Any

from core.data import _cache_key, _read_cache, _safe_float, _write_cache
from core.sec import cik_padded, sec_get, ticker_to_cik

logger = logging.getLogger(__name__)

INSIDER_WINDOW_DAYS = 90
CLUSTER_MIN_BUYERS = 2
FORM4_CACHE_HOURS = 24
FORM4_ERROR_CACHE_HOURS = 1
_OPEN_MARKET = frozenset({"P", "S"})
_XSL_PREFIX = re.compile(r"^xsl[^/]+/", re.IGNORECASE)


def _raw_document_path(document: str) -> str:
    """SEC lists the XSL-rendered HTML as primaryDocument; the raw XML is the same
    path without the ``xsl…/`` prefix."""
    return _XSL_PREFIX.sub("", document.strip())


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(node: ET.Element | None, *names: str) -> str | None:
    """First matching tag's text, including the common Form 4 `<value>` wrapper."""
    if node is None:
        return None
    wanted = {n.lower() for n in names}
    for child in node.iter():
        if _local(child.tag).lower() not in wanted:
            continue
        if child.text and child.text.strip():
            return child.text.strip()
        for sub in child:
            if _local(sub.tag).lower() == "value" and sub.text and sub.text.strip():
                return sub.text.strip()
    return None


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


def _footnote_map(root: ET.Element) -> dict[str, str]:
    notes: dict[str, str] = {}
    for node in root.iter():
        if _local(node.tag).lower() not in {"footnote", "footnotes"}:
            continue
        fid = node.attrib.get("id") or node.attrib.get("footnoteId") or ""
        text = " ".join(node.itertext()).strip()
        if fid:
            notes[fid] = text
        elif text:
            notes.setdefault("_all", "")
            notes["_all"] = (notes["_all"] + " " + text).strip()
    return notes


def _transaction_is_10b5_1(tx_node: ET.Element, root: ET.Element) -> bool:
    """True when the transaction is tagged as a Rule 10b5-1 planned trade."""
    notes = _footnote_map(root)
    blob = " ".join(notes.values()).lower()
    ids: list[str] = []
    for node in tx_node.iter():
        fid = node.attrib.get("footnoteId") or node.attrib.get("id")
        if fid:
            ids.append(fid)
        if _local(node.tag).lower() == "footnoteid" and (node.text or "").strip():
            ids.append(node.text.strip())
    pointed = " ".join(notes.get(i, "") for i in ids).lower()
    haystack = pointed or blob
    if "10b5-1" in haystack or "10b5–1" in haystack:
        if ids and not pointed:
            return "10b5-1" in blob or "10b5–1" in blob
        return True
    return False


def parse_form4_xml(xml_text: str) -> list[dict[str, Any]]:
    """Parse open-market P/S transactions out of a Form 4 ownership XML."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    insider = _child_text(root, "rptOwnerName") or "unknown"
    title = _child_text(root, "officerTitle") or ""
    is_officer = _truthy(_child_text(root, "isOfficer")) or bool(title)
    is_director = _truthy(_child_text(root, "isDirector"))
    rows: list[dict[str, Any]] = []
    for node in root.iter():
        if _local(node.tag) != "nonDerivativeTransaction":
            continue
        code = (_child_text(node, "transactionCode") or "").upper()
        if code not in _OPEN_MARKET:
            continue
        shares = _safe_float(_child_text(node, "transactionShares"))
        price = _safe_float(_child_text(node, "transactionPricePerShare"))
        tx_date = _child_text(node, "transactionDate") or _child_text(node, "value")
        acquired = (_child_text(node, "transactionAcquiredDisposedCode") or "").upper()
        if shares is None or shares <= 0:
            continue
        sign = 1.0 if code == "P" or acquired == "A" else -1.0
        if code == "S":
            sign = -1.0
        value = shares * (price or 0.0) * sign
        rows.append(
            {
                "insider": insider,
                "title": title,
                "code": code,
                "shares": shares,
                "price": price,
                "value": value,
                "date": tx_date,
                "is_officer": is_officer,
                "is_director": is_director,
                "planned_10b5_1": _transaction_is_10b5_1(node, root),
            }
        )
    return rows


def _recent_form4_accessions(cik: int, since: date) -> list[dict[str, str]]:
    url = f"https://data.sec.gov/submissions/CIK{cik_padded(cik)}.json"
    data = sec_get(url, timeout=45).json()
    recent = ((data.get("filings") or {}).get("recent") or {})
    forms = recent.get("form") or []
    accessions = recent.get("accessionNumber") or []
    documents = recent.get("primaryDocument") or []
    dates = recent.get("filingDate") or []
    out: list[dict[str, str]] = []
    for form, acc, doc, filed in zip(forms, accessions, documents, dates):
        if str(form).split("/")[0] != "4":
            continue
        try:
            filed_d = date.fromisoformat(str(filed)[:10])
        except ValueError:
            continue
        if filed_d < since:
            continue
        out.append(
            {
                "accession": str(acc).replace("-", ""),
                "document": _raw_document_path(str(doc)),
                "filed": str(filed)[:10],
            }
        )
        if len(out) >= 20:
            break
    return out


def fetch_form4_transactions(ticker: str, *, days: int = INSIDER_WINDOW_DAYS) -> list[dict[str, Any]]:
    """Open-market Form 4 trades for ``ticker`` in the last ``days`` (cached 24h)."""
    cache_path = _cache_key("form4v2", ticker.upper(), str(days))
    cached = _read_cache(cache_path, max_age_hours=FORM4_CACHE_HOURS)
    if cached is not None and isinstance(cached.get("transactions"), list):
        if not cached.get("error"):
            return cached["transactions"]
        # Negative cache: honour only for the short TTL, then retry.
        if _read_cache(cache_path, max_age_hours=FORM4_ERROR_CACHE_HOURS) is not None:
            return cached["transactions"]

    cik = ticker_to_cik(ticker)
    if cik is None:
        _write_cache(cache_path, {"transactions": []})
        return []

    since = date.today() - timedelta(days=days)
    transactions: list[dict[str, Any]] = []
    try:
        filings = _recent_form4_accessions(cik, since)
    except Exception as exc:
        logger.warning("Form 4 index failed for %s: %s", ticker, exc)
        _write_cache(cache_path, {"transactions": [], "error": True})
        return []

    for filing in filings:
        url = (
            f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
            f"{filing['accession']}/{filing['document']}"
        )
        try:
            text = sec_get(url, timeout=30).text
        except Exception as exc:
            logger.debug("Form 4 download failed %s %s: %s", ticker, filing["accession"], exc)
            continue
        if "<ownershipDocument" not in text and "<nonDerivativeTransaction" not in text:
            continue
        transactions.extend(parse_form4_xml(text))

    _write_cache(cache_path, {"transactions": transactions})
    return transactions


def compute_insider_factor(raw: dict[str, Any]) -> dict[str, Any]:
    """
    ``insider_buying`` = net open-market value / market cap (can be negative).

    ``insider_cluster_buy`` is true when at least two distinct insiders filed
    open-market purchases in the window — the badge signal.
    """
    transactions = raw.get("form4_transactions")
    if not isinstance(transactions, list):
        transactions = []

    net_value = 0.0
    buyers: set[str] = set()
    sellers: set[str] = set()
    for tx in transactions:
        value = _safe_float(tx.get("value")) or 0.0
        net_value += value
        name = str(tx.get("insider") or "unknown")
        officerish = tx.get("is_officer")
        directorish = tx.get("is_director")
        is_od = True if officerish is None and directorish is None else bool(officerish or directorish)
        if tx.get("code") == "P":
            if is_od and not tx.get("planned_10b5_1"):
                buyers.add(name)
        elif tx.get("code") == "S":
            sellers.add(name)

    market_cap = _safe_float(raw.get("market_cap"))
    insider_buying = None
    if market_cap and market_cap > 0 and transactions:
        insider_buying = net_value / market_cap

    return {
        "insider_buying": insider_buying,
        "insider_cluster_buy": len(buyers) >= CLUSTER_MIN_BUYERS,
        "insider_buyers_90d": len(buyers),
        "insider_sellers_90d": len(sellers),
        "insider_net_value_90d": net_value if transactions else None,
    }
