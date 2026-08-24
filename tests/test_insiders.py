"""Tests for Form 4 parsing and the insider factor."""

import pytest

from core.insiders import compute_insider_factor, parse_form4_xml

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
