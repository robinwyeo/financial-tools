"""Threshold calibration from historical score-return buckets."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from backtest.constants import RESULTS_DIR
from backtest.data.prices import load_prices
from backtest.engine import _forward_quarter_returns, score_factor_panel
from backtest.factors import load_factor_panel

logger = logging.getLogger(__name__)


def _bucket_excess(
    scored: pd.DataFrame,
    fwd: pd.DataFrame,
    score_col: str,
    n_buckets: int = 10,
) -> pd.DataFrame:
    rows: list[dict] = []
    for qend, grp in scored.groupby("quarter_end"):
        valid = grp.dropna(subset=[score_col])
        if len(valid) < n_buckets * 3:
            continue
        valid = valid.copy()
        valid["bucket"] = pd.qcut(valid[score_col], n_buckets, labels=False, duplicates="drop")
        f_df = fwd[fwd["quarter_end"] == qend]
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
                    "mean_fwd_return": float(bgrp["fwd_return"].mean()),
                    "n": len(bgrp),
                }
            )
    return pd.DataFrame(rows)


def calibrate_thresholds(
    factor_weights: dict[str, float],
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    target_positive_excess_bucket: int = 7,
) -> dict[str, Any]:
    """
    Set composite_min / bargain_min where forward returns turn reliably positive.
    Uses top buckets from decile analysis.
    """
    panel = panel if panel is not None else load_factor_panel()
    prices = prices if prices is not None else load_prices()
    scored = score_factor_panel(panel, factor_weights)
    fwd = _forward_quarter_returns(panel, prices)

    comp_buckets = _bucket_excess(scored, fwd, "composite")
    bargain_buckets = _bucket_excess(scored, fwd, "bargain_score")

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
            composite_min = float(agg.loc[target, "score_min"])

    if not bargain_buckets.empty:
        agg = bargain_buckets.groupby("bucket").agg(
            score_min=("score_min", "mean"),
            mean_fwd_return=("mean_fwd_return", "mean"),
        )
        positive = agg[agg["mean_fwd_return"] > 0]
        if not positive.empty:
            target = max(target_positive_excess_bucket, int(positive.index.min()))
            bargain_min = float(agg.loc[target, "score_min"])

    composite_min = float(np.clip(composite_min, 30.0, 80.0))
    bargain_min = float(np.clip(bargain_min, 30.0, 80.0))

    out = {
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
    return out
