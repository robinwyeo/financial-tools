"""Tests for Form 4 parsing and the insider factor."""

from datetime import date, timedelta
from types import SimpleNamespace

import pytest

from core.insiders import (
    _raw_document_path,
    _recent_form4_accessions,
    compute_insider_factor,
    fetch_form4_transactions,
    parse_form4_xml,
)

_FORM4 = """<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId><rptOwnerName>Jane CEO</rptOwnerName></reportingOwnerId>
    <reportingOwnerRelationship><officerTitle>Chief Executive Officer</officerTitle></reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-07-01</value></transactionDate>
      <transactionCoding><transactionCode>P</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>1000</value></transactionShares>
        <transactionPricePerShare><value>50</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>A</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>
    <nonDerivativeTransaction>
      <transactionDate><value>2026-07-02</value></transactionDate>
      <transactionCoding><transactionCode>A</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>5000</value></transactionShares>
        <transactionPricePerShare><value>0</value></transactionPricePerShare>
      </transactionAmounts>
    </nonDerivativeTransaction>
  </nonDerivativeTable>
</ownershipDocument>
"""


def test_parse_form4_keeps_open_market_purchases_only():
    rows = parse_form4_xml(_FORM4)
    assert len(rows) == 1
    assert rows[0]["code"] == "P"
    assert rows[0]["insider"] == "Jane CEO"
    assert rows[0]["value"] == 50_000


def test_insider_cluster_buy_and_yield():
    txs = [
        {"insider": "A", "code": "P", "value": 100_000},
        {"insider": "B", "code": "P", "value": 50_000},
        {"insider": "C", "code": "S", "value": -20_000},
    ]
    out = compute_insider_factor({"market_cap": 10_000_000, "form4_transactions": txs})
    assert out["insider_cluster_buy"] is True
    assert out["insider_buyers_90d"] == 2
    assert out["insider_sellers_90d"] == 1
    assert out["insider_buying"] == pytest.approx(0.013)


def test_insider_no_cluster_for_single_buyer():
    txs = [{"insider": "A", "code": "P", "value": 10_000}]
    out = compute_insider_factor({"market_cap": 1_000_000, "form4_transactions": txs})
    assert out["insider_cluster_buy"] is False
    assert out["insider_buyers_90d"] == 1


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("xslF345X06/wk-form4_1.xml", "wk-form4_1.xml"),
        ("wk-form4_1.xml", "wk-form4_1.xml"),
        ("xslF345X05/doc4.xml", "doc4.xml"),
    ],
)
def test_raw_document_path_strips_xsl_prefix(raw, expected):
    assert _raw_document_path(raw) == expected


def test_recent_form4_accessions_normalizes_document(monkeypatch):
    filed = date.today().isoformat()

    def fake_sec_get(url, timeout=45):
        return SimpleNamespace(
            json=lambda: {
                "filings": {
                    "recent": {
                        "form": ["4", "10-Q"],
                        "accessionNumber": ["0001628280-26-058684", "0001703057-26-000046"],
                        "primaryDocument": ["xslF345X06/wk-form4_1.xml", "abcl-20260630.htm"],
                        "filingDate": [filed, filed],
                    }
                }
            }
        )

    monkeypatch.setattr("core.insiders.sec_get", fake_sec_get)
    rows = _recent_form4_accessions(1703057, date.today() - timedelta(days=90))
    assert len(rows) == 1
    assert rows[0]["document"] == "wk-form4_1.xml"
    assert rows[0]["accession"] == "000162828026058684"


def test_fetch_form4_index_failure_uses_short_negative_cache(monkeypatch):
    written = []
    monkeypatch.setattr("core.insiders._read_cache", lambda *a, **kw: None)
    monkeypatch.setattr("core.insiders._write_cache", lambda path, data: written.append(data))
    monkeypatch.setattr("core.insiders.ticker_to_cik", lambda ticker: 1)

    def boom(cik, since):
        raise RuntimeError("SEC 403")

    monkeypatch.setattr("core.insiders._recent_form4_accessions", boom)
    out = fetch_form4_transactions("ABCL")
    assert out == []
    assert written == [{"transactions": [], "error": True}]


def test_fetch_form4_negative_cache_expires_before_full_ttl(monkeypatch):
    calls = {"index": 0}

    def fake_read_cache(path, max_age_hours=6):
        if max_age_hours == 24:
            return {"transactions": [], "error": True}
        if max_age_hours == 1:
            return None
        return None

    def fake_index(cik, since):
        calls["index"] += 1
        return []

    monkeypatch.setattr("core.insiders._read_cache", fake_read_cache)
    monkeypatch.setattr("core.insiders._write_cache", lambda *a, **kw: None)
    monkeypatch.setattr("core.insiders.ticker_to_cik", lambda ticker: 1)
    monkeypatch.setattr("core.insiders._recent_form4_accessions", fake_index)
    out = fetch_form4_transactions("ABCL")
    assert out == []
    assert calls["index"] == 1


def test_fetch_form4_success_path(monkeypatch):
    monkeypatch.setattr("core.insiders._read_cache", lambda *a, **kw: None)
    monkeypatch.setattr("core.insiders._write_cache", lambda *a, **kw: None)
    monkeypatch.setattr("core.insiders.ticker_to_cik", lambda ticker: 1)
    monkeypatch.setattr(
        "core.insiders._recent_form4_accessions",
        lambda cik, since: [{"accession": "0001", "document": "form4.xml", "filed": "2026-07-01"}],
    )
    monkeypatch.setattr(
        "core.insiders.sec_get",
        lambda url, timeout=30: SimpleNamespace(text=_FORM4),
    )
    rows = fetch_form4_transactions("AAPL")
    assert len(rows) == 1
    assert rows[0]["code"] == "P"
    assert rows[0]["is_officer"] is True


def test_10b5_1_purchase_excluded_from_cluster():
    xml = _FORM4.replace(
        "<transactionCoding><transactionCode>P</transactionCode></transactionCoding>",
        '<transactionCoding footnoteId="F1"><transactionCode>P</transactionCode></transactionCoding>',
    ).replace(
        "</ownershipDocument>",
        "<footnotes><footnote id='F1'>Sale under Rule 10b5-1 plan</footnote></footnotes></ownershipDocument>",
    )
    rows = parse_form4_xml(xml)
    assert rows[0]["planned_10b5_1"] is True
    out = compute_insider_factor({"market_cap": 1_000_000, "form4_transactions": rows})
    assert out["insider_cluster_buy"] is False
    assert out["insider_buyers_90d"] == 0
