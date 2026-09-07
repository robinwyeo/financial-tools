"""Universe snapshot integrity."""

from pathlib import Path

import pandas as pd
import pytest

from core.universe import SNAPSHOT_PATH, SnapshotIncompleteError, build_universe_snapshot


def test_incomplete_snapshot_does_not_write(tmp_path, monkeypatch):
    existing = tmp_path / "universe_snapshot.parquet"
    pd.DataFrame({"ticker": ["KEEP"]}).to_parquet(existing, index=False)
    monkeypatch.setattr("core.universe.SNAPSHOT_PATH", existing)
    monkeypatch.setattr("core.universe.SNAPSHOT_META_PATH", tmp_path / "universe_snapshot.meta.json")
    monkeypatch.setattr("core.universe.DATA_DIR", tmp_path)
    monkeypatch.setattr("core.universe.throttle", lambda *_a, **_k: None)

    def boom(ticker: str):
        if ticker.startswith("F"):
            raise RuntimeError("fail")
        return {"name": ticker, "sector": "Tech", "industry": None}

    monkeypatch.setattr("core.universe.build_raw_metrics", boom)
    monkeypatch.setattr("core.universe.compute_all_factors", lambda raw: {"earnings_yield": 0.1})

    with pytest.raises(SnapshotIncompleteError):
        build_universe_snapshot(tickers=["OK1", "FAIL1", "FAIL2", "FAIL3"], min_success_ratio=0.9)

    kept = pd.read_parquet(existing)
    assert list(kept["ticker"]) == ["KEEP"]


def test_successful_snapshot_writes(tmp_path, monkeypatch):
    dest = tmp_path / "universe_snapshot.parquet"
    monkeypatch.setattr("core.universe.SNAPSHOT_PATH", dest)
    monkeypatch.setattr("core.universe.SNAPSHOT_META_PATH", tmp_path / "universe_snapshot.meta.json")
    monkeypatch.setattr("core.universe.DATA_DIR", tmp_path)
    monkeypatch.setattr("core.universe.throttle", lambda *_a, **_k: None)
    monkeypatch.setattr(
        "core.universe.build_raw_metrics",
        lambda t: {"name": t, "sector": "Tech", "industry": None},
    )
    monkeypatch.setattr("core.universe.compute_all_factors", lambda raw: {"earnings_yield": 0.1})

    df = build_universe_snapshot(tickers=["AAA", "BBB"], min_success_ratio=0.9)
    assert len(df) == 2
    assert dest.exists()
    meta = Path(tmp_path / "universe_snapshot.meta.json")
    assert meta.exists()
    assert "universe" in df.columns
    assert "currency" in df.columns
    assert "country" in df.columns


def test_prefer_us_listings_drops_tsx_dual():
    from core.universe import prefer_us_listings

    out = prefer_us_listings(["SHOP", "SHOP.TO", "RY.TO", "AAPL"])
    assert "SHOP" in out
    assert "SHOP.TO" not in out
    assert "RY.TO" in out
    assert "AAPL" in out


def test_multi_universe_assembly(monkeypatch):
    from core.universe import fetch_universe_tickers

    monkeypatch.setattr("core.universe.fetch_sp500_tickers", lambda: ["AAPL", "SHOP"])
    monkeypatch.setattr("core.universe.fetch_sp400_tickers", lambda: ["RSG"])
    monkeypatch.setattr("core.universe.fetch_tsx60_tickers", lambda: ["SHOP.TO", "RY.TO"])
    tagged = fetch_universe_tickers(["sp500", "sp400", "tsx60"])
    tickers = [t for t, _ in tagged]
    assert "AAPL" in tickers
    assert "RSG" in tickers
    assert "RY.TO" in tickers
    assert "SHOP.TO" not in tickers
    assert "SHOP" in tickers


def test_snapshot_persists_quality_score(tmp_path, monkeypatch):
    """Rebuild writes scored columns (quality_score) without hitting the network."""
    dest = tmp_path / "universe_snapshot.parquet"
    monkeypatch.setattr("core.universe.SNAPSHOT_PATH", dest)
    monkeypatch.setattr("core.universe.SNAPSHOT_META_PATH", tmp_path / "universe_snapshot.meta.json")
    monkeypatch.setattr("core.universe.DATA_DIR", tmp_path)
    monkeypatch.setattr("core.universe.throttle", lambda *_a, **_k: None)

    sectors = {"AAA": "Technology", "BBB": "Healthcare", "CCC": "Technology"}

    def fake_raw(ticker: str):
        return {
            "name": ticker,
            "sector": sectors[ticker],
            "industry": "Widgets",
            "currency": "USD",
        }

    def fake_factors(raw: dict):
        seed = {"AAA": 0.0, "BBB": 1.0, "CCC": 2.0}[raw["name"]]
        return {
            "earnings_yield": 0.06 + 0.02 * seed,
            "fcf_yield": 0.04 + 0.01 * seed,
            "book_to_market": 0.3 + 0.05 * seed,
            "garp": 1.2 + 0.1 * seed,
            "gross_profitability": 0.25 + 0.05 * seed,
            "roe": 0.12 + 0.03 * seed,
            "roa": 0.07 + 0.01 * seed,
            "profit_margin": 0.10 + 0.02 * seed,
            "roic": 0.14 + 0.02 * seed,
            "fcf_margin": 0.08 + 0.01 * seed,
            "earnings_quality": 0.04,
            "fcf_conversion_3y": 0.7 + 0.05 * seed,
            "financial_strength": 6.0 + seed,
            "leverage_quality": -0.8,
            "interest_coverage": 7.0 + seed,
            "net_cash_to_mcap": -0.04,
            "low_leverage": 0.4,
            "altman_z": 3.0 + 0.2 * seed,
            "momentum_12_1": 0.10 - 0.02 * seed,
            "low_volatility": 8.0,
            "shareholder_yield": 0.02 + 0.01 * seed,
            "investment": -0.03,
            "anti_dilution": 0.01,
            "roic_5y_mean": 0.12,
            "stability_roic": -0.02,
            "gross_margin_5y_delta": 0.01,
            "revenue_5y_cagr": 0.05,
            "revision_agreement": 0.5,
            "revision_magnitude": 0.02,
            "earnings_surprise": 0.03,
            "insider_buying": 0.001,
        }

    monkeypatch.setattr("core.universe.build_raw_metrics", fake_raw)
    monkeypatch.setattr("core.universe.compute_all_factors", fake_factors)

    df = build_universe_snapshot(tickers=["AAA", "BBB", "CCC"], min_success_ratio=0.0)
    assert "quality_score" in df.columns
    persisted = pd.read_parquet(dest)
    required = {"quality_score", "currency", "universe"}
    missing = required - set(persisted.columns)
    assert not missing, f"persisted snapshot missing {sorted(missing)}"
    assert set(persisted["ticker"]) == {"AAA", "BBB", "CCC"}
    assert "quality_coverage_pct" in persisted.columns


def test_committed_snapshot_has_post_overhaul_columns():
    """Honesty gate: the committed live snapshot must be post-overhaul, not pre-fix."""
    if not SNAPSHOT_PATH.exists():
        return
    df = pd.read_parquet(SNAPSHOT_PATH)
    required = {"quality_score", "currency", "universe"}
    missing = required - set(df.columns)
    assert not missing, (
        f"committed universe snapshot is pre-overhaul; missing columns {sorted(missing)}. "
        "Rebuild with python -m core.universe (or the monthly job); do not xfail this test."
    )
    brk = df[df["ticker"].astype(str).str.upper() == "BRK-B"]
    if brk.empty:
        return
    row = brk.iloc[0]
    btm = row["book_to_market"]
    graham = row["graham_ratio"]
    roic = row["roic"]
    assert btm < 5, (
        f"BRK-B book_to_market={btm} is a pre-overhaul dual-class blow-up; expected < 5. "
        "Rebuild data/universe_snapshot.parquet after FIND-001."
    )
    assert graham < 2, (
        f"BRK-B graham_ratio={graham} exceeds WP0.2 bound < 2. "
        "Rebuild data/universe_snapshot.parquet after FIND-001."
    )
    assert roic <= 2, (
        f"BRK-B roic={roic} exceeds the post-fix clip of 2. "
        "Rebuild data/universe_snapshot.parquet."
    )
