"""Jobs end-to-end with mocked scoring and email."""

import pandas as pd

from jobs.runlog import exceeds_failure_threshold, write_run_summary
from jobs.universe_monthly import run_monthly
from jobs.watchlist_weekly import run_weekly, score_tickers


def _analysis(ticker: str, accumulate: bool = True) -> dict:
    return {
        "ticker": ticker,
        "is_etf": False,
        "is_good_buy": accumulate,
        "composite": 70.0,
        "bargain": {"score": 60.0},
        "decision": {
            "label": "Accumulate" if accumulate else "Avoid",
            "buy_below_price": 90.0,
            "pct_to_buy": -5.0,
            "gates": [{"name": "price_vs_buy_below", "passed": accumulate}],
        },
        "quality_score": 65.0,
        "data_quality": {"grade": "A"},
        "price": 80.0,
        "currency": "USD",
    }


def test_failure_threshold():
    assert exceeds_failure_threshold(2, 10) is True
    assert exceeds_failure_threshold(1, 10) is False


def test_write_run_summary(tmp_path, monkeypatch):
    monkeypatch.setattr("jobs.runlog.RUNS_DIR", tmp_path)
    path = write_run_summary("test", {"ok": True, "scored": 3}, run_id="abc")
    assert path.exists()
    assert (tmp_path / "latest.json").exists()
    assert path.read_text(encoding="utf-8").find("abc") >= 0


def test_watchlist_weekly_writes_summary(tmp_path, monkeypatch):
    snapshot = pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "snapshot_date": ["2026-09-01", "2026-09-01"]}
    )
    monkeypatch.setattr("jobs.watchlist_weekly.load_config", lambda: {"email": {}})
    monkeypatch.setattr("jobs.watchlist_weekly.load_universe_snapshot", lambda: snapshot)
    monkeypatch.setattr("jobs.watchlist_weekly.load_watchlist", lambda: ["AAA", "BBB"])
    monkeypatch.setattr("jobs.watchlist_weekly.score_universe_df", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        "jobs.watchlist_weekly.score_ticker",
        lambda ticker, *a, **k: _analysis(ticker),
    )
    monkeypatch.setattr(
        "jobs.watchlist_weekly.apply_universe_snapshot_scoring",
        lambda result, *a, **k: result,
    )
    monkeypatch.setattr("jobs.watchlist_weekly.hurdle_rate", lambda *a, **k: {"rate": 0.087})
    monkeypatch.setattr("jobs.runlog.RUNS_DIR", tmp_path)
    monkeypatch.setattr("jobs.watchlist_weekly.write_run_summary", lambda *a, **k: tmp_path / "x.json")

    written = {}

    def fake_write(kind, payload, run_id=None):
        written["kind"] = kind
        written["payload"] = payload
        written["run_id"] = run_id
        path = tmp_path / f"{run_id}.json"
        path.write_text("{}", encoding="utf-8")
        return path

    monkeypatch.setattr("jobs.watchlist_weekly.write_run_summary", fake_write)
    rc = run_weekly(refresh_universe=False, send_report=False)
    assert rc == 0
    assert written["kind"] == "watchlist_weekly"
    assert written["payload"]["scored"] == 2
    assert written["payload"]["accumulate"] == 2


def test_watchlist_weekly_fails_on_high_error_ratio(tmp_path, monkeypatch):
    snapshot = pd.DataFrame({"ticker": ["AAA"], "snapshot_date": ["2026-09-01"]})
    monkeypatch.setattr("jobs.watchlist_weekly.load_config", lambda: {"email": {}})
    monkeypatch.setattr("jobs.watchlist_weekly.load_universe_snapshot", lambda: snapshot)
    monkeypatch.setattr("jobs.watchlist_weekly.load_watchlist", lambda: ["AAA", "BBB", "CCC"])
    monkeypatch.setattr("jobs.watchlist_weekly.score_universe_df", lambda *a, **k: snapshot)

    def boom(ticker, *a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr("jobs.watchlist_weekly.score_ticker", boom)
    monkeypatch.setattr("jobs.watchlist_weekly.hurdle_rate", lambda *a, **k: {"rate": 0.087})
    monkeypatch.setattr("jobs.watchlist_weekly.write_run_summary", lambda *a, **k: tmp_path / "x.json")
    rc = run_weekly(refresh_universe=False, send_report=False)
    assert rc == 1


def test_monthly_job_no_refresh(tmp_path, monkeypatch):
    snapshot = pd.DataFrame(
        {"ticker": ["AAA"], "snapshot_date": ["2026-09-01"]}
    )
    monkeypatch.setattr("jobs.universe_monthly.load_config", lambda: {"email": {}})
    monkeypatch.setattr("jobs.universe_monthly.load_universe_snapshot", lambda: snapshot)
    monkeypatch.setattr("jobs.universe_monthly.score_universe_df", lambda *a, **k: snapshot)
    monkeypatch.setattr(
        "jobs.universe_monthly.score_ticker",
        lambda ticker, *a, **k: _analysis(ticker),
    )
    monkeypatch.setattr(
        "jobs.universe_monthly.apply_universe_snapshot_scoring",
        lambda result, *a, **k: result,
    )
    monkeypatch.setattr("jobs.universe_monthly.hurdle_rate", lambda *a, **k: {"rate": 0.087})
    monkeypatch.setattr("jobs.universe_monthly.write_run_summary", lambda *a, **k: tmp_path / "x.json")
    rc = run_monthly(refresh_universe=False, send_report=False)
    assert rc == 0


def test_score_tickers_skips_etfs(monkeypatch):
    monkeypatch.setattr("jobs.watchlist_weekly.score_universe_df", lambda *a, **k: pd.DataFrame({"ticker": ["SPY"]}))
    monkeypatch.setattr(
        "jobs.watchlist_weekly.score_ticker",
        lambda ticker, *a, **k: {"ticker": ticker, "is_etf": True},
    )
    results, failures = score_tickers(["SPY"], {}, pd.DataFrame({"ticker": ["SPY"]}))
    assert results == []
    assert failures == []
