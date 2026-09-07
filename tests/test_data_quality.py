"""Yahoo vs EDGAR reconciliation grades."""

from core.data_quality import DataQuality, apply_substitutions, attach_data_quality, reconcile


def _complete_raw(**overrides):
    base = {
        "ticker": "AAPL",
        "price": 180.0,
        "market_cap": 2.8e12,
        "revenue": 400e9,
        "ebit": 120e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_assets": 350e9,
        "total_debt": 100e9,
        "total_cash": 50e9,
        "book_equity": 70e9,
        "shares_outstanding": 15e9,
        "data_warnings": [],
    }
    base.update(overrides)
    return base


def test_grade_a_when_yahoo_matches_edgar():
    raw = _complete_raw()
    edgar = {
        "revenue": 400e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_debt": 100e9,
        "total_cash": 50e9,
        "book_equity": 70e9,
        "shares_outstanding": 15e9,
    }
    q = reconcile(raw, edgar)
    assert q.grade == "A"
    assert q.substitutions == []
    assert q.presence_pct >= 0.90


def test_dual_class_shares_warn_without_substituting():
    raw = _complete_raw(ticker="BRK-B", shares_outstanding=2e9)
    edgar = {
        "revenue": 400e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_debt": 100e9,
        "total_cash": 50e9,
        "book_equity": 70e9,
        "shares_outstanding": 941481,
    }
    q = reconcile(raw, edgar)
    fields = {s["field"] for s in q.substitutions}
    assert "shares_outstanding" not in fields
    assert "book_equity" not in fields
    assert any(
        w.get("code") == "RECONCILE" and "keeping listing shares" in w.get("message", "")
        for w in q.warnings
    )
    apply_substitutions(raw, q)
    assert raw["shares_outstanding"] == 2e9


def test_grade_b_or_c_with_equity_substitution():
    raw = _complete_raw(ticker="BRK-B", book_equity=800e12)  # BVPS × B-shares blow-up
    edgar = {
        "revenue": 400e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_debt": 100e9,
        "total_cash": 50e9,
        "book_equity": 70e9,
        "shares_outstanding": 15e9,
    }
    q = reconcile(raw, edgar)
    assert q.grade in {"B", "C"}
    fields = {s["field"] for s in q.substitutions}
    assert "book_equity" in fields
    assert any(s["chosen"] == "edgar" for s in q.substitutions if s["field"] == "book_equity")


def test_grade_b_when_no_edgar_filer():
    raw = _complete_raw(ticker="SHOP.TO")
    q = reconcile(raw, None)
    assert q.grade == "B"
    assert "no EDGAR filer" in q.reasons
    assert q.edgar_available is False


def test_financials_cash_mismatch_does_not_force_grade_c():
    raw = _complete_raw(
        ticker="BRK-B",
        sector="Financial Services",
        total_cash=365e9,
        revenue=400e9 * 1.12,
        shares_outstanding=15e9 * 1.34,
    )
    edgar = {
        "revenue": 400e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_debt": 100e9,
        "total_cash": 31e9,
        "book_equity": 70e9,
        "shares_outstanding": 15e9,
    }
    q = reconcile(raw, edgar)
    assert q.grade in {"A", "B"}
    fields = {s["field"] for s in q.substitutions}
    assert "total_cash" not in fields


def test_grade_c_more_than_two_substitutions():
    raw = _complete_raw(
        revenue=1.0,
        net_income=1.0,
        operating_cashflow=1.0,
        book_equity=1.0,
    )
    edgar = {
        "revenue": 400e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_debt": 100e9,
        "total_cash": 50e9,
        "book_equity": 70e9,
        "shares_outstanding": 15e9,
    }
    q = reconcile(raw, edgar)
    assert q.grade == "C"
    assert len(q.substitutions) > 2


def test_grade_c_low_presence():
    raw = {"ticker": "THIN", "price": 10.0, "data_warnings": []}
    q = reconcile(raw, None)
    assert q.grade == "C"
    assert q.presence_pct < 0.75


def test_grade_c_currency_inconsistency():
    raw = _complete_raw(currency="CAD", financial_currency="USD")
    edgar = {
        "revenue": 400e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_debt": 100e9,
        "total_cash": 50e9,
        "book_equity": 70e9,
        "shares_outstanding": 15e9,
    }
    q = reconcile(raw, edgar)
    assert q.grade == "C"
    assert any("currency" in r for r in q.reasons)


def test_attach_applies_edgar_substitution(monkeypatch):
    raw = _complete_raw(book_equity=1.0)
    edgar = {
        "revenue": 400e9,
        "net_income": 100e9,
        "operating_cashflow": 110e9,
        "total_debt": 100e9,
        "total_cash": 50e9,
        "book_equity": 70e9,
        "shares_outstanding": 15e9,
    }
    monkeypatch.setattr("core.data_quality.load_edgar_snapshot", lambda t: edgar)
    q = attach_data_quality(raw)
    assert raw["book_equity"] == 70e9
    assert raw["data_quality"]["grade"] in {"B", "C"}
    assert isinstance(raw["data_warnings"][0], dict)
    assert q.grade == raw["data_quality"]["grade"]


def test_dataquality_to_dict_roundtrip():
    q = DataQuality(grade="A", presence_pct=1.0, reasons=["ok"])
    d = q.to_dict()
    assert d["grade"] == "A"
    assert d["substitutions"] == []
