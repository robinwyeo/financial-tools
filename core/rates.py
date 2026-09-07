"""Hurdle rate: max(8%, 10y Treasury + ERP), +1% for High uncertainty."""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Any

import pandas as pd
import requests

from core.config import get_valuation_config
from core.data import _cache_key, _read_cache, _write_cache

logger = logging.getLogger(__name__)

FRED_OBS_URL = "https://api.stlouisfed.org/fred/series/observations"

_SERIES_LABELS = {
    "DGS10": "10-Year Treasury",
    "BAA10Y": "Moody’s Baa corporate yield",
    "T10YIE": "10y breakeven inflation",
}


def fetch_fred_series(
    series_id: str,
    api_key: str | None = None,
    *,
    force: bool = False,
) -> float | None:
    """Latest observation for a FRED series, as a decimal yield (4.2 → 0.042). Cached 24h."""
    api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")
    cache_path = _cache_key("fred", series_id)
    if not force:
        cached = _read_cache(cache_path, max_age_hours=24)
        if isinstance(cached, dict) and cached.get("value") is not None:
            return float(cached["value"])
    if not api_key:
        return None
    try:
        resp = requests.get(
            FRED_OBS_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": 8,
            },
            timeout=20,
        )
        resp.raise_for_status()
        obs = (resp.json() or {}).get("observations") or []
        value = None
        as_of = None
        for item in obs:
            raw = str(item.get("value") or "").strip()
            if raw in {"", "."}:
                continue
            value = float(raw) / 100.0 if float(raw) > 1.0 else float(raw)
            as_of = item.get("date")
            break
        if value is None:
            return None
        _write_cache(cache_path, {"value": value, "as_of": as_of, "series_id": series_id})
        return value
    except Exception as exc:
        logger.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None


def fetch_fred_history(
    series_id: str = "DGS10",
    api_key: str | None = None,
    *,
    force: bool = False,
) -> pd.Series:
    """Daily FRED history as a decimal yield series, cached 7 days."""
    api_key = api_key if api_key is not None else os.environ.get("FRED_API_KEY")
    cache_path = _cache_key("fredhist", series_id)
    if not force:
        cached = _read_cache(cache_path, max_age_hours=168)
        if isinstance(cached, dict) and cached.get("rows"):
            df = pd.DataFrame(cached["rows"])
            df["date"] = pd.to_datetime(df["date"])
            return df.set_index("date")["value"].sort_index()
    if not api_key:
        return pd.Series(dtype="float64")
    try:
        resp = requests.get(
            FRED_OBS_URL,
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "sort_order": "asc",
            },
            timeout=30,
        )
        resp.raise_for_status()
        rows = []
        for item in (resp.json() or {}).get("observations") or []:
            raw = str(item.get("value") or "").strip()
            if raw in {"", "."}:
                continue
            val = float(raw)
            if val > 1.0:
                val = val / 100.0
            rows.append({"date": item.get("date"), "value": val})
        _write_cache(cache_path, {"rows": rows, "series_id": series_id})
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.Series(dtype="float64")
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")["value"].sort_index()
    except Exception as exc:
        logger.warning("FRED history fetch failed for %s: %s", series_id, exc)
        return pd.Series(dtype="float64")


def pit_risk_free(as_of: date | pd.Timestamp, history: pd.Series | None = None) -> float | None:
    """Point-in-time DGS10 on or before ``as_of``."""
    if history is None or history.empty:
        history = fetch_fred_history("DGS10")
    if history is None or history.empty:
        return None
    ts = pd.Timestamp(as_of)
    eligible = history[history.index <= ts]
    if eligible.empty:
        return None
    return float(eligible.iloc[-1])


def fred_observation(series_id: str) -> dict[str, Any]:
    """Latest FRED point plus ``as_of`` date from cache (after ``fetch_fred_series``)."""
    value = fetch_fred_series(series_id)
    cached = _read_cache(_cache_key("fred", series_id), max_age_hours=24) or {}
    return {
        "series_id": series_id,
        "value": value,
        "as_of": cached.get("as_of"),
        "label": _SERIES_LABELS.get(series_id, series_id),
    }


def market_rate_context() -> dict[str, Any]:
    """DGS10 plus credit (BAA10Y) and breakeven inflation (T10YIE) for display."""
    dgs = fred_observation("DGS10")
    baa = fred_observation("BAA10Y")
    ie = fred_observation("T10YIE")
    spread = None
    if dgs.get("value") is not None and baa.get("value") is not None:
        spread = float(baa["value"]) - float(dgs["value"])
    return {
        "dgs10": dgs,
        "baa10y": baa,
        "t10yie": ie,
        "baa_spread": spread,
    }


def hurdle_rate(
    uncertainty_label: str | None = None,
    *,
    rf: float | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    r = max(floor, rf + ERP) + 1% if High uncertainty.

    With no FRED key, rf falls back to config (default 4.2%) so hurdle = 8.7%.
    """
    cfg = get_valuation_config(config)
    hurdle_cfg = cfg.get("hurdle") or {}
    floor = float(hurdle_cfg.get("floor", 0.08))
    erp = float(hurdle_cfg.get("erp", 0.045))
    fallback_rf = float(hurdle_cfg.get("fallback_rf", 0.042))
    high_bump = float(hurdle_cfg.get("high_uncertainty_bump", 0.01))

    rf_source = "config_fallback"
    rf_as_of = None
    if rf is None:
        obs = fred_observation("DGS10")
        fetched = obs.get("value")
        if fetched is not None:
            rf = fetched
            rf_source = "FRED:DGS10"
            rf_as_of = obs.get("as_of")
        else:
            rf = fallback_rf
            rf_source = "config_fallback"

    base = max(floor, float(rf) + erp)
    bump = high_bump if str(uncertainty_label or "").lower() == "high" else 0.0
    ctx = market_rate_context()
    return {
        "rate": base + bump,
        "rf": float(rf),
        "erp": erp,
        "floor": floor,
        "uncertainty_bump": bump,
        "rf_source": rf_source,
        "rf_as_of": rf_as_of,
        "label": _SERIES_LABELS["DGS10"],
        "baa10y": (ctx.get("baa10y") or {}).get("value"),
        "t10yie": (ctx.get("t10yie") or {}).get("value"),
        "baa_spread": ctx.get("baa_spread"),
    }
