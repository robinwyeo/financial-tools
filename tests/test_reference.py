"""Parse Damodaran reference CSVs."""

from pathlib import Path

from core.reference import parse_damodaran_erp, parse_damodaran_industry, latest_erp


def test_parse_committed_erp_csv():
    path = Path("data/reference/damodaran_erp.csv")
    df = parse_damodaran_erp(path)
    assert not df.empty
    assert {"year", "implied_erp"} <= set(df.columns)
    assert df["implied_erp"].iloc[-1] == 0.045
    assert latest_erp() == 0.045


def test_parse_committed_industry_csv():
    path = Path("data/reference/damodaran_industry.csv")
    df = parse_damodaran_industry(path)
    assert "Software (Internet)" in set(df["industry"])
    assert "wacc" in df.columns
