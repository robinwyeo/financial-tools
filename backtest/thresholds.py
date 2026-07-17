"""Threshold calibration from historical score-return buckets."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from backtest.constants import PRIMARY_EVAL_HORIZON, RESULTS_DIR
from backtest.data.prices import load_prices
from backtest.engine import precompute_multi_horizon_returns, score_factor_panel
from backtest.factors import load_factor_panel

logger = logging.getLogger(__name__)


def _bucket_excess(
    scored: pd.DataFrame,
    fwd: pd.DataFrame,
    score_col: str,
    return_col: str,
    n_buckets: int = 10,
) -> pd.DataFrame:
    rows: list[dict] = []
    scored = scored.copy()
    scored["as_of_quarter"] = pd.to_datetime(scored["quarter_end"])
    for qend, grp in scored.groupby("as_of_quarter"):
        valid = grp.dropna(subset=[score_col])
        if len(valid) < n_buckets * 3:
            continue
        valid = valid.copy()
        valid["bucket"] = pd.qcut(valid[score_col], n_buckets, labels=False, duplicates="drop")
        f_df = fwd[fwd["as_of_quarter"] == qend][["ticker", return_col]].dropna()
        merged = valid.merge(f_df, on="ticker", how="inner")
        if merged.empty:
            continue
        for bucket, bgrp in merged.groupby("bucket"):
            rows.append(
                {
                    "quarter_end": qend,
                    "bucket": int(bucket),
                    "score_min": float(bgrp[score_col].min()),
                    "score_max": float(bgrp[score_col].max()),
                    "mean_fwd_return": float(bgrp[return_col].mean()),
                    "n": len(bgrp),
                }
            )
    return pd.DataFrame(rows)


def calibrate_thresholds(
    factor_weights: dict[str, float],
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    target_positive_excess_bucket: int = 7,
    horizon: str = PRIMARY_EVAL_HORIZON,
) -> dict[str, Any]:
    """
    Set composite_min / bargain_min where long-horizon forward excess turns positive.

    Defaults to the 3y excess-return horizon (buy-and-hold relevant).
    """
    panel = panel if panel is not None else load_factor_panel()
    prices = prices if prices is not None else load_prices()
    scored = score_factor_panel(panel, factor_weights)
    multi = precompute_multi_horizon_returns(panel, prices)

    excess_col = f"excess_{horizon}"
    if excess_col not in multi.columns:
        excess_col = f"fwd_{horizon}"
    if excess_col not in multi.columns:
        raise ValueError(f"Horizon column missing for {horizon}")

    comp_buckets = _bucket_excess(scored, multi, "composite", excess_col)
    bargain_buckets = _bucket_excess(scored, multi, "bargain_score", excess_col)

    composite_min = 50.0
    bargain_min = 50.0

    if not comp_buckets.empty:
        agg = comp_buckets.groupby("bucket").agg(
            score_min=("score_min", "mean"),
            mean_fwd_return=("mean_fwd_return", "mean"),
        )
        positive = agg[agg["mean_fwd_return"] > 0]
        if not positive.empty:
            target = max(target_positive_excess_bucket, int(positive.index.min()))
            target = min(target, int(agg.index.max()))
            composite_min = float(agg.loc[target, "score_min"])

    if not bargain_buckets.empty:
        agg = bargain_buckets.groupby("bucket").agg(
            score_min=("score_min", "mean"),
            mean_fwd_return=("mean_fwd_return", "mean"),
        )
        positive = agg[agg["mean_fwd_return"] > 0]
        if not positive.empty:
            target = max(target_positive_excess_bucket, int(positive.index.min()))
            target = min(target, int(agg.index.max()))
            bargain_min = float(agg.loc[target, "score_min"])

    composite_min = float(np.clip(composite_min, 30.0, 80.0))
    bargain_min = float(np.clip(bargain_min, 30.0, 80.0))

    out = {
        "horizon": horizon,
        "return_column": excess_col,
        "composite_min": composite_min,
        "bargain_min": bargain_min,
        "composite_bucket_stats": comp_buckets.groupby("bucket").mean(numeric_only=True).to_dict()
        if not comp_buckets.empty
        else {},
        "bargain_bucket_stats": bargain_buckets.groupby("bucket").mean(numeric_only=True).to_dict()
        if not bargain_buckets.empty
        else {},
    }
    path = RESULTS_DIR / "threshold_calibration.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    logger.info(
        "Calibrated thresholds on %s: composite_min=%.1f bargain_min=%.1f",
        horizon,
        composite_min,
        bargain_min,
    )
    return out
