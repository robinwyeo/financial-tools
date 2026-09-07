"""Cached loaders. Streamlit cache is applied lazily on first call."""

from __future__ import annotations

from typing import Any

import pandas as pd

from core.config import load_config as _load_config
from core.data import fetch_price_history as _fetch_price_history
from core.fund_universe import FUND_SNAPSHOT_PATH as FUND_SNAPSHOT_PATH
from core.fund_universe import load_fund_universe_snapshot as _load_fund_snap
from core.scoring import (
    apply_fund_snapshot_scoring,
    apply_universe_snapshot_scoring,
    score_fund,
    score_fund_universe,
    score_ticker,
    score_universe,
)
from core.universe import SNAPSHOT_PATH as STOCK_SNAPSHOT_PATH
from core.universe import load_universe_snapshot as _load_uni_snap

_CACHED: dict[str, Any] = {}


def _cached(name: str, fn, ttl: int = 3600):
    import streamlit as st

    wrapped = _CACHED.get(name)
    if wrapped is None:
        wrapped = st.cache_data(ttl=ttl)(fn)
        _CACHED[name] = wrapped
    return wrapped


def load_config() -> dict:
    return _cached("load_config", _load_config)()


def load_universe_snapshot() -> pd.DataFrame | None:
    mtime = STOCK_SNAPSHOT_PATH.stat().st_mtime if STOCK_SNAPSHOT_PATH.exists() else 0.0

    def _inner(snapshot_mtime: float):
        del snapshot_mtime  # Included in Streamlit's cache key.
        return _load_uni_snap()

    return _cached("uni_snap", _inner)(mtime)


def load_fund_universe_snapshot() -> pd.DataFrame | None:
    mtime = FUND_SNAPSHOT_PATH.stat().st_mtime if FUND_SNAPSHOT_PATH.exists() else 0.0

    def _inner(snapshot_mtime: float):
        del snapshot_mtime  # Included in Streamlit's cache key.
        return _load_fund_snap()

    return _cached("fund_snap", _inner)(mtime)


def score_universe_cached(config: dict) -> pd.DataFrame:
    mtime = STOCK_SNAPSHOT_PATH.stat().st_mtime if STOCK_SNAPSHOT_PATH.exists() else 0.0

    def _inner(snapshot_mtime: float, scoring_config: dict):
        del snapshot_mtime  # Included in Streamlit's cache key.
        return score_universe(scoring_config)

    return _cached("score_uni", _inner)(mtime, config)


def score_fund_universe_cached(config: dict) -> pd.DataFrame:
    mtime = FUND_SNAPSHOT_PATH.stat().st_mtime if FUND_SNAPSHOT_PATH.exists() else 0.0

    def _inner(snapshot_mtime: float, scoring_config: dict):
        del snapshot_mtime  # Included in Streamlit's cache key.
        return score_fund_universe(scoring_config)

    return _cached("score_fund_uni", _inner)(mtime, config)


def score_ticker_cached(ticker: str, config: dict) -> dict:
    snap = load_universe_snapshot()
    date = ""
    if snap is not None and not snap.empty and "snapshot_date" in snap.columns:
        date = str(snap["snapshot_date"].iloc[0])

    def _inner(cache_ticker: str, snapshot_date: str, scoring_config: dict):
        del snapshot_date  # Included in Streamlit's cache key.
        live_snap = _load_uni_snap()
        return score_ticker(cache_ticker, scoring_config, live_snap)

    return _cached("score_ticker", _inner, ttl=1800)(ticker, date, config)


def score_fund_cached(ticker: str, config: dict) -> dict:
    snap = load_fund_universe_snapshot()
    date = ""
    if snap is not None and not snap.empty and "snapshot_date" in snap.columns:
        date = str(snap["snapshot_date"].iloc[0])
    mtime = FUND_SNAPSHOT_PATH.stat().st_mtime if FUND_SNAPSHOT_PATH.exists() else 0.0

    def _inner(
        cache_ticker: str,
        snapshot_date: str,
        snapshot_mtime: float,
        scoring_config: dict,
    ):
        del snapshot_date, snapshot_mtime  # Included in Streamlit's cache key.
        live_snap = _load_fund_snap()
        return score_fund(cache_ticker, scoring_config, live_snap)

    return _cached("score_fund", _inner, ttl=1800)(ticker, date, mtime, config)


def fetch_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    def _inner(cache_ticker: str, cache_period: str):
        return _fetch_price_history(cache_ticker, period=cache_period)

    return _cached("px", _inner, ttl=3600)(ticker, period)


def apply_stock_snapshot(analysis: dict, scored: pd.DataFrame, ticker: str, config: dict) -> dict:
    return apply_universe_snapshot_scoring(analysis, scored, ticker, config)


def apply_fund_snapshot(analysis: dict, scored: pd.DataFrame, ticker: str) -> dict:
    return apply_fund_snapshot_scoring(analysis, scored, ticker)
