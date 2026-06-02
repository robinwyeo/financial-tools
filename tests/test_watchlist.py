"""Tests for watchlist file loading."""

from pathlib import Path

from core.watchlist import WATCHLIST_PATH, load_watchlist


def test_load_watchlist_from_file(tmp_path, monkeypatch):
    watchlist_file = tmp_path / "watchlist"
    watchlist_file.write_text("# comment\nAAPL\n\nMSFT\n", encoding="utf-8")
    monkeypatch.setattr("core.watchlist.WATCHLIST_PATH", watchlist_file)
    assert load_watchlist({"watchlist": ["NVDA"]}) == ["AAPL", "MSFT"]


def test_load_watchlist_falls_back_to_config(monkeypatch):
    monkeypatch.setattr("core.watchlist.WATCHLIST_PATH", Path("/nonexistent/watchlist"))
    assert load_watchlist({"watchlist": ["NVDA", "cost"]}) == ["NVDA", "COST"]


def test_load_watchlist_repo_file_exists():
    if WATCHLIST_PATH.exists():
        tickers = load_watchlist()
        assert isinstance(tickers, list)
        assert all(t.isupper() for t in tickers)
