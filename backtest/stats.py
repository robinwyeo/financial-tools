"""Honest statistics for the backtest harness: Newey-West, block bootstrap, evidence cards."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_RUN_ID: str | None = None


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_run_id() -> str:
    global _RUN_ID
    if not _RUN_ID:
        _RUN_ID = new_run_id()
    return _RUN_ID


def set_run_id(run_id: str) -> None:
    global _RUN_ID
    _RUN_ID = run_id


def write_results_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a results artifact stamped with the current run_id."""
    out = dict(payload)
    out["run_id"] = get_run_id()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")


def n_independent_windows(n_quarters: int, horizon_quarters: int) -> float:
    """How many non-overlapping horizon-length windows fit in ``n_quarters``."""
    if horizon_quarters <= 0:
        return float(n_quarters)
    return float(n_quarters) / float(horizon_quarters)


def newey_west_tstat(series: list[float] | np.ndarray | pd.Series, lag: int) -> float:
    """
    Newey-West t-stat of the mean of ``series``.

    ``lag`` should be horizon_quarters - 1 for overlapping forward-return ICs.
    """
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n < 3:
        return float("nan")
    lag = max(0, int(lag))
    lag = min(lag, n - 1)
    mean = float(arr.mean())
    u = arr - mean
    gamma0 = float(np.dot(u, u) / n)
    nw_var = gamma0
    for k in range(1, lag + 1):
        weight = 1.0 - k / (lag + 1)
        gamma_k = float(np.dot(u[k:], u[:-k]) / n)
        nw_var += 2.0 * weight * gamma_k
    se = float(np.sqrt(max(nw_var, 0.0) / n))
    if se <= 0:
        return float("nan")
    return mean / se


def block_bootstrap_ci(
    values_by_quarter: list[float] | np.ndarray | pd.Series,
    *,
    block: int = 12,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """Circular block-bootstrap CI for the mean of a quarterly series."""
    arr = np.asarray(values_by_quarter, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = int(arr.size)
    if n == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    mean = float(arr.mean())
    if n == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean}
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        pieces: list[np.ndarray] = []
        for _ in range(n_blocks):
            start = int(rng.integers(0, n))
            if start + block <= n:
                pieces.append(arr[start : start + block])
            else:
                pieces.append(np.concatenate([arr[start:], arr[: block - (n - start)]]))
        sample = np.concatenate(pieces)[:n]
        means[i] = sample.mean()
    alpha = (1.0 - ci) / 2.0
    return {
        "mean": mean,
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
    }


def decile_spread(
    scored: pd.DataFrame,
    fwd: pd.DataFrame,
    score_col: str,
    horizon: str,
    *,
    n_deciles: int = 10,
    return_prefix: str = "fwd_",
) -> dict[str, float]:
    """
    Mean (D10 − D1) forward return. Higher scores are assumed better (long D10).
    """
    col = f"{return_prefix}{horizon}"
    if col not in fwd.columns:
        col = f"excess_{horizon}"
    if col not in fwd.columns:
        return {"spread": float("nan"), "n_quarters": 0.0}
    scored = scored.copy()
    scored["as_of_quarter"] = pd.to_datetime(scored["quarter_end"])
    spreads: list[float] = []
    for qend, grp in scored.groupby("as_of_quarter"):
        valid = grp.dropna(subset=[score_col])
        if len(valid) < n_deciles * 3:
            continue
        try:
            valid = valid.copy()
            valid["decile"] = pd.qcut(valid[score_col], n_deciles, labels=False, duplicates="drop")
        except ValueError:
            continue
        f_df = fwd[fwd["as_of_quarter"] == qend][["ticker", col]].dropna()
        merged = valid.merge(f_df, on="ticker", how="inner")
        if merged["decile"].nunique() < 2:
            continue
        d1 = merged.loc[merged["decile"] == merged["decile"].min(), col]
        d10 = merged.loc[merged["decile"] == merged["decile"].max(), col]
        if d1.empty or d10.empty:
            continue
        spreads.append(float(d10.mean() - d1.mean()))
    if not spreads:
        return {"spread": float("nan"), "n_quarters": 0.0, "series": []}
    ci = block_bootstrap_ci(spreads, block=12, seed=42)
    return {
        "spread": float(np.mean(spreads)),
        "ci_low": ci["ci_low"],
        "ci_high": ci["ci_high"],
        "n_quarters": float(len(spreads)),
        "series": spreads,
    }


def gated_pick_horizon_stats(
    picks: dict[pd.Timestamp, list[str]],
    multi: pd.DataFrame,
    horizon: str = "3y",
) -> dict[str, Any]:
    """Hit rate and mean excess of gated picks vs the horizon excess column."""
    col = f"excess_{horizon}"
    if col not in multi.columns:
        col = f"fwd_{horizon}"
    if col not in multi.columns or multi.empty:
        return {"hit_rate": float("nan"), "mean_excess": float("nan"), "n_quarters": 0}
    multi = multi.copy()
    multi["as_of_quarter"] = pd.to_datetime(multi["as_of_quarter"])
    hit_rates: list[float] = []
    means: list[float] = []
    for q, tickers in picks.items():
        if not tickers:
            continue
        qts = pd.Timestamp(q)
        sub = multi[(multi["as_of_quarter"] == qts) & (multi["ticker"].isin(tickers))]
        vals = sub[col].dropna() if col in sub.columns else pd.Series(dtype=float)
        if vals.empty:
            continue
        hit_rates.append(float((vals > 0).mean()))
        means.append(float(vals.mean()))
    if not means:
        return {"hit_rate": float("nan"), "mean_excess": float("nan"), "n_quarters": 0}
    ci = block_bootstrap_ci(means, block=12, seed=42)
    return {
        "hit_rate": float(np.mean(hit_rates)),
        "mean_excess": float(np.mean(means)),
        "mean_excess_ci_low": ci["ci_low"],
        "mean_excess_ci_high": ci["ci_high"],
        "n_quarters": len(means),
        "mean_picks_per_quarter": float(
            np.mean([len(v) for v in picks.values() if v]) if picks else 0.0
        ),
    }


def evidence_card(
    *,
    name: str,
    ic_series: list[float],
    horizon_quarters: int,
    coverage: float | None = None,
    spread: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Standard evidence card for a registered metric or score."""
    arr = [float(v) for v in ic_series if v is not None and np.isfinite(v)]
    n_q = len(arr)
    lag = max(horizon_quarters - 1, 0)
    nw = newey_west_tstat(arr, lag)
    ci = block_bootstrap_ci(arr, block=12, seed=42)
    monotonic = None
    if spread and spread.get("series"):
        s = spread["series"]
        monotonic = bool(np.mean(s) > 0)
    return {
        "name": name,
        "mean_ic": float(np.mean(arr)) if arr else float("nan"),
        "nw_tstat": nw,
        "ic_ci_low": ci["ci_low"],
        "ic_ci_high": ci["ci_high"],
        "n_quarters": n_q,
        "n_independent_windows": n_independent_windows(n_q, horizon_quarters),
        "coverage": coverage,
        "decile_spread": None if spread is None else spread.get("spread"),
        "decile_spread_ci_low": None if spread is None else spread.get("ci_low"),
        "decile_spread_ci_high": None if spread is None else spread.get("ci_high"),
        "monotonic": monotonic,
    }
