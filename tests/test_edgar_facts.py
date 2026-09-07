"""Tests for consolidated companyfacts ingest and TTM construction."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from core.edgar_facts import (
    facts_to_rows,
    flatten_fundamentals,
    fundamentals_as_of_structured,
    period_table_from_facts,
)


def _entry(end, val, *, start=None, filed=None, form="10-Q", fy=None, fp=None, frame=None, accn="a"):
    item = {
        "end": end,
        "val": val,
        "form": form,
        "filed": filed or end,
        "accn": accn,
    }
    if start is not None:
        item["start"] = start
    if fy is not None:
        item["fy"] = fy
    if fp is not None:
        item["fp"] = fp
    if frame is not None:
        item["frame"] = frame
    return item


def _companyfacts_fixture() -> dict:
    """Two fiscal years of revenue with YTD-only Q3 and a restatement.

    FY 2023: Q1 10, Q2 12 (YTD 22), Q3 YTD-only 33 (implies Q3=11), FY 48 (implies Q4=15)
    FY 2024: Q1 11, Q2 13 (YTD 24), Q3 YTD 36, FY 50
    Restatement of FY 2023 revenue: original 48 filed 2024-02-01, restated 49 filed 2024-08-01.
    Instant assets at each quarter/FY end.
    """
    revenue = [
        _entry("2023-03-31", 10, start="2023-01-01", filed="2023-05-01", fy=2023, fp="Q1", frame="CY2023Q1"),
        _entry("2023-06-30", 22, start="2023-01-01", filed="2023-08-01", fy=2023, fp="Q2", frame="CY2023Q2YTD"),
        _entry("2023-09-30", 33, start="2023-01-01", filed="2023-11-01", fy=2023, fp="Q3", frame="CY2023Q3YTD"),
        _entry("2023-12-31", 48, start="2023-01-01", filed="2024-02-01", form="10-K", fy=2023, fp="FY", frame="CY2023"),
        _entry("2023-12-31", 49, start="2023-01-01", filed="2024-08-01", form="10-K/A", fy=2023, fp="FY", frame="CY2023"),
        _entry("2024-03-31", 11, start="2024-01-01", filed="2024-05-01", fy=2024, fp="Q1", frame="CY2024Q1"),
        _entry("2024-06-30", 24, start="2024-01-01", filed="2024-08-01", fy=2024, fp="Q2", frame="CY2024Q2YTD"),
        _entry("2024-09-30", 36, start="2024-01-01", filed="2024-11-01", fy=2024, fp="Q3", frame="CY2024Q3YTD"),
        _entry("2024-12-31", 50, start="2024-01-01", filed="2025-02-01", form="10-K", fy=2024, fp="FY", frame="CY2024"),
    ]
    # Segment row that must NOT overwrite consolidated FY 2024.
    revenue.append(
        _entry("2024-12-31", 5, start="2024-01-01", filed="2025-02-01", form="10-K", fy=2024, fp="FY", frame=None)
    )
    assets = [
        _entry("2023-12-31", 200, filed="2024-02-01", form="10-K", fy=2023, fp="FY", frame="CY2023"),
        _entry("2024-12-31", 220, filed="2025-02-01", form="10-K", fy=2024, fp="FY", frame="CY2024"),
        _entry("2024-09-30", 210, filed="2024-11-01", fy=2024, fp="Q3", frame="CY2024Q3"),
    ]
    ocf = [
        _entry("2023-12-31", 20, start="2023-01-01", filed="2024-02-01", form="10-K", fy=2023, fp="FY", frame="CY2023"),
        _entry("2024-12-31", 22, start="2024-01-01", filed="2025-02-01", form="10-K", fy=2024, fp="FY", frame="CY2024"),
    ]
    equity = [
        _entry("2023-12-31", 80, filed="2024-02-01", form="10-K", fy=2023, fp="FY"),
        _entry("2024-12-31", 90, filed="2025-02-01", form="10-K", fy=2024, fp="FY"),
    ]
    shares = [
        _entry("2023-12-31", 10, filed="2024-02-01", form="10-K", fy=2023, fp="FY"),
        _entry("2024-12-31", 10, filed="2025-02-01", form="10-K", fy=2024, fp="FY"),
    ]
    ebit = [
        _entry("2023-12-31", 11.3e9, start="2023-01-01", filed="2024-02-20", form="10-K", fy=2023, fp="FY", frame="CY2023"),
        _entry("2024-12-31", 12.0e9, start="2024-01-01", filed="2025-02-20", form="10-K", fy=2024, fp="FY", frame="CY2024"),
    ]
    return {
        "cik": 123,
        "entityName": "FAKE INC",
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": revenue}},
                "Assets": {"units": {"USD": assets}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ocf}},
                "StockholdersEquity": {"units": {"USD": equity}},
                "CommonStockSharesOutstanding": {"units": {"shares": shares}},
                "OperatingIncomeLoss": {"units": {"USD": ebit}},
            }
        },
    }


def test_facts_to_rows_dedupes_unframed_segment():
    rows = facts_to_rows(_companyfacts_fixture(), cik=123)
    fy24 = rows[(rows["tag"] == "Revenues") & (rows["end"] == pd.Timestamp("2024-12-31")) & (rows["qtrs"] == 4)]
    # Framed consolidated 50 should win over unframed segment 5 when filed is equal...
    # drop_duplicates keep last after sorting _has_frame, so framed wins.
    assert not fy24.empty
    framed = fy24[fy24["frame"].notna()]
    assert float(framed.iloc[0]["val"]) == 50.0


def test_tag_priority_prefers_first_tag():
    facts = {
        "facts": {
            "us-gaap": {
                "Revenues": {
                    "units": {
                        "USD": [
                            _entry("2024-12-31", 100, start="2024-01-01", form="10-K", fp="FY", filed="2025-02-01"),
                        ]
                    }
                },
                "SalesRevenueNet": {
                    "units": {
                        "USD": [
                            _entry("2024-12-31", 1, start="2024-01-01", form="10-K", fp="FY", filed="2025-02-01"),
                        ]
                    }
                },
            }
        }
    }
    rows = facts_to_rows(facts, cik=1)
    structured = fundamentals_as_of_structured(date(2025, 12, 31), rows)
    assert structured["flow_fy"]["revenue"] == pytest.approx(100.0)


def test_ttm_derives_q4_from_fy_minus_ytd():
    rows = facts_to_rows(_companyfacts_fixture(), cik=123)
    as_of = date(2025, 3, 15)  # after FY2024 10-K
    structured = fundamentals_as_of_structured(as_of, rows)
    # Last 4 quarters of 2024: Q1 11 + Q2 13 + Q3 12 + Q4 14 = 50
    # Q2 derived = 24-11=13; Q3 derived = 36-24=12; Q4 derived = 50-36=14
    assert structured["flow_ttm"]["revenue"] == pytest.approx(50.0)
    assert structured["flow_fy"]["revenue"] == pytest.approx(50.0)


def test_restatement_is_point_in_time():
    rows = facts_to_rows(_companyfacts_fixture(), cik=123)
    before = fundamentals_as_of_structured(date(2024, 3, 1), rows)
    after = fundamentals_as_of_structured(date(2024, 9, 1), rows)
    assert before["flow_fy"]["revenue"] == pytest.approx(48.0)
    assert after["flow_fy"]["revenue"] == pytest.approx(49.0)


def test_flatten_maps_ttm_and_mrq():
    rows = facts_to_rows(_companyfacts_fixture(), cik=123)
    structured = fundamentals_as_of_structured(date(2025, 12, 31), rows)
    flat = flatten_fundamentals(structured)
    assert flat["revenue"] == pytest.approx(50.0)
    assert flat["total_assets"] == pytest.approx(220.0)
    assert flat["book_equity"] == pytest.approx(90.0)
    assert flat["ebit"] == pytest.approx(12.0e9)


def test_constant_revenue_ttm_does_not_oscillate():
    """Equal quarterly revenue of 25 should produce a flat 100 TTM every quarter-end."""
    entries = []
    # Four years of explicit qtrs=1 rows (start/end ~90 days).
    starts = [
        ("2021-01-01", "2021-03-31"),
        ("2021-04-01", "2021-06-30"),
        ("2021-07-01", "2021-09-30"),
        ("2021-10-01", "2021-12-31"),
        ("2022-01-01", "2022-03-31"),
        ("2022-04-01", "2022-06-30"),
        ("2022-07-01", "2022-09-30"),
        ("2022-10-01", "2022-12-31"),
        ("2023-01-01", "2023-03-31"),
        ("2023-04-01", "2023-06-30"),
        ("2023-07-01", "2023-09-30"),
        ("2023-10-01", "2023-12-31"),
    ]
    for start, end in starts:
        filed = str(pd.Timestamp(end) + pd.Timedelta(days=30).to_pytimedelta())[:10]
        entries.append(_entry(end, 25, start=start, filed=filed, form="10-Q" if not end.endswith("12-31") else "10-K"))
    facts = {"facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}}}
    rows = facts_to_rows(facts, cik=9)
    values = []
    for as_of in [
        date(2022, 4, 15),
        date(2022, 7, 15),
        date(2022, 10, 15),
        date(2023, 1, 15),
        date(2023, 4, 15),
    ]:
        ttm = fundamentals_as_of_structured(as_of, rows)["flow_ttm"].get("revenue")
        assert ttm is not None
        values.append(ttm)
    for a, b in zip(values, values[1:]):
        assert abs(a - b) / a < 0.25
        assert a == pytest.approx(100.0)


def test_period_table_from_facts_annual():
    rows = facts_to_rows(_companyfacts_fixture(), cik=123)
    rows["ticker"] = "FAKE"
    table = period_table_from_facts(rows, "FAKE")
    assert not table.empty
    assert "ebit" in table.columns
    assert table["ebit"].max() == pytest.approx(12.0e9)


def test_ingest_edgar_rejects_stale_fsds_parquet(tmp_path, monkeypatch):
    """Leftover FSDS dumps (amount/period) must not be returned as companyfacts."""
    from backtest.data import edgar as edgar_mod

    path = tmp_path / "fundamentals.parquet"
    fsds = pd.DataFrame(
        {
            "adsh": ["0001"],
            "tag": ["Revenues"],
            "period": ["2020-12-31"],
            "amount": [100.0],
            "qtrs": [4],
        }
    )
    fsds.to_parquet(path, index=False)
    monkeypatch.setattr(edgar_mod, "EDGAR_FUNDAMENTALS_PATH", path)

    with pytest.raises(RuntimeError, match="ingest --force"):
        edgar_mod.ingest_edgar()
