"""Zacks-style earnings-estimate revision signals from yfinance."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from core.data import _cache_key, _read_cache, _safe_float, _write_cache

logger = logging.getLogger(__name__)

_TREND_PERIODS = ("0y", "1y", "+1y", "0q", "+1q")
_PRIMARY_PERIODS = ("0y", "+1y", "1y")


def _df_from_records(payload: Any) -> pd.DataFrame:
    if not payload:
        return pd.DataFrame()
    if isinstance(payload, list):
        return pd.DataFrame(payload)
    return pd.DataFrame()


def _df_to_cache(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    out = df.reset_index()
    out.columns = [str(c) for c in out.columns]
    return out.to_dict(orient="records")


def fetch_estimate_tables(ticker: str) -> dict[str, pd.DataFrame]:
    """Fetch eps_trend / eps_revisions / earnings_history (cached 24h)."""
    cache_path = _cache_key("ests", ticker.upper())
    cached = _read_cache(cache_path, max_age_hours=24)
    if cached is not None:
        return {
            "eps_trend": _df_from_records(cached.get("eps_trend")),
            "eps_revisions": _df_from_records(cached.get("eps_revisions")),
            "earnings_history": _df_from_records(cached.get("earnings_history")),
        }

    empty = {
        "eps_trend": pd.DataFrame(),
        "eps_revisions": pd.DataFrame(),
        "earnings_history": pd.DataFrame(),
    }
    try:
        t = yf.Ticker(ticker)
        trend = getattr(t, "eps_trend", None)
        revisions = getattr(t, "eps_revisions", None)
        history = getattr(t, "earnings_history", None)
        if callable(trend):
            trend = trend()
        if callable(revisions):
            revisions = revisions()
        if callable(history):
            history = history()
        tables = {
            "eps_trend": trend if isinstance(trend, pd.DataFrame) else pd.DataFrame(),
            "eps_revisions": revisions if isinstance(revisions, pd.DataFrame) else pd.DataFrame(),
            "earnings_history": history if isinstance(history, pd.DataFrame) else pd.DataFrame(),
        }
        _write_cache(
            cache_path,
            {key: _df_to_cache(df) for key, df in tables.items()},
        )
        return tables
    except Exception as exc:
        logger.warning("Estimate tables failed for %s: %s", ticker, exc)
        return empty


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "") for c in out.columns]
    if "index" in out.columns and out.index.name is None:
        # Cached frames often park the period label in an 'index' column.
        labeled = out.set_index("index")
        labeled.index = labeled.index.astype(str)
        return labeled
    out.index = out.index.astype(str)
    return out


def _period_row(df: pd.DataFrame, period: str) -> pd.Series | None:
    if df.empty:
        return None
    idx = df.index.astype(str).str.lower()
    matches = df.loc[idx == period.lower()]
    if matches.empty:
        return None
    return matches.iloc[0]


def _col(row: pd.Series, *names: str) -> float | None:
    lowered = {str(k).strip().lower().replace(" ", ""): k for k in row.index}
    for name in names:
        key = lowered.get(name.lower().replace(" ", ""))
        if key is not None:
            return _safe_float(row[key])
    return None


def _revision_agreement(revisions: pd.DataFrame) -> float | None:
    """Share of FY1/FY2 revisions that were upgrades over the last 30 days (0-1)."""
    revisions = _normalize_columns(revisions)
    scores: list[float] = []
    for period in _PRIMARY_PERIODS:
        row = _period_row(revisions, period)
        if row is None:
            continue
        up = _col(row, "uplast30days", "uplast7days") or 0.0
        down = _col(row, "downlast30days", "downlast7days") or 0.0
        total = up + down
        if total <= 0:
            continue
        scores.append(up / total)
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _revision_magnitude(trend: pd.DataFrame) -> float | None:
    """Percent change in consensus FY1/FY2 EPS from ~90 days ago to current."""
    trend = _normalize_columns(trend)
    scores: list[float] = []
    for period in _PRIMARY_PERIODS:
        row = _period_row(trend, period)
        if row is None:
            continue
        current = _col(row, "current")
        ago = _col(row, "90daysago", "60daysago", "30daysago", "7daysago")
        if current is None or ago is None or ago == 0:
            continue
        scores.append((current - ago) / abs(ago))
    if not scores:
        return None
    return float(sum(scores) / len(scores))


def _earnings_surprise(history: pd.DataFrame) -> float | None:
    """Most recent quarterly EPS surprise as a fraction (0.05 = +5%)."""
    history = _normalize_columns(history)
    if history.empty:
        return None
    surprise_col = None
    for name in ("surprisepercent", "surprise%", "surprisespercent"):
        if name in history.columns:
            surprise_col = name
            break
    if surprise_col is None:
        return None
    series = pd.to_numeric(history[surprise_col], errors="coerce").dropna()
    if series.empty:
        return None
    val = float(series.iloc[-1])
    # yfinance sometimes reports 5.0 for +5%, sometimes 0.05.
    if abs(val) > 1.5:
        val = val / 100.0
    return val


def _trend_sparkline(trend: pd.DataFrame) -> list[float]:
    """FY1 consensus path [90d, 60d, 30d, 7d, current] for the dashboard."""
    trend = _normalize_columns(trend)
    row = None
    for period in _PRIMARY_PERIODS:
        row = _period_row(trend, period)
        if row is not None:
            break
    if row is None:
        return []
    points = []
    for col in ("90daysago", "60daysago", "30daysago", "7daysago", "current"):
        val = _col(row, col)
        if val is not None:
            points.append(val)
    return points


def compute_revision_factors(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Three Zacks-style sub-signals (higher = more positive revisions):

    - revision_agreement: share of FY1/FY2 estimate revisions that were upgrades
    - revision_magnitude: % change in the FY1/FY2 consensus over ~90 days
    - earnings_surprise: last reported quarterly EPS surprise
    """
    tables = raw.get("estimate_tables")
    if not isinstance(tables, dict):
        tables = {}
    trend = tables.get("eps_trend") if tables else pd.DataFrame()
    revisions = tables.get("eps_revisions") if tables else pd.DataFrame()
    history = tables.get("earnings_history") if tables else pd.DataFrame()
    if not isinstance(trend, pd.DataFrame):
        trend = pd.DataFrame()
    if not isinstance(revisions, pd.DataFrame):
        revisions = pd.DataFrame()
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame()

    return {
        "revision_agreement": _revision_agreement(revisions),
        "revision_magnitude": _revision_magnitude(trend),
        "earnings_surprise": _earnings_surprise(history),
        "eps_trend_sparkline": _trend_sparkline(trend),
    }
