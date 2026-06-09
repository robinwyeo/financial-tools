"""Walk-forward hyperparameter tuning for factor and bargain weights."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from backtest.constants import RESULTS_DIR
from backtest.engine import BacktestResult, objective_tuple, run_backtest, split_period
from backtest.factors import load_factor_panel
from backtest.weights import (
    current_baseline_bargain_weights,
    current_baseline_factor_weights,
    random_bargain_weights,
    random_theme_weights,
    theme_weights_to_factor_weights,
)

logger = logging.getLogger(__name__)

TUNING_RESULTS_PATH = RESULTS_DIR / "tuning_results.json"


@dataclass
class TuningCandidate:
    name: str
    theme_weights: dict[str, float]
    factor_weights: dict[str, float]
    train: dict[str, float]
    valid: dict[str, float]
    test: dict[str, float]


def _result_metrics(result: BacktestResult) -> dict[str, float]:
    return {
        "rolling_win_rate": result.rolling_win_rate,
        "median_excess": result.median_excess,
        "mean_ic": result.mean_ic,
        "cagr": result.cagr,
        "benchmark_cagr": result.benchmark_cagr,
        "max_drawdown": result.max_drawdown,
    }


def evaluate_factor_weights(
    factor_weights: dict[str, float],
    panel: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    panel = panel if panel is not None else load_factor_panel()
    out: dict[str, dict[str, float]] = {}
    for period in ("train", "valid", "test"):
        start, end = split_period(period)
        result = run_backtest(factor_weights, panel=panel, start=start, end=end)
        out[period] = _result_metrics(result)
    return out


def tune_factor_weights(
    n_samples: int = 500,
    seed: int = 42,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Dirichlet search over theme weights with walk-forward evaluation."""
    panel = panel if panel is not None else load_factor_panel()
    rng = np.random.default_rng(seed)
    baseline_fw = current_baseline_factor_weights()
    baseline_metrics = evaluate_factor_weights(baseline_fw, panel)

    candidates: list[TuningCandidate] = []
    baseline_obj = objective_tuple_from_metrics(baseline_metrics["valid"])

    for i in range(n_samples):
        theme_w = random_theme_weights(rng)
        factor_w = theme_weights_to_factor_weights(theme_w)
        metrics = evaluate_factor_weights(factor_w, panel)
        candidates.append(
            TuningCandidate(
                name=f"sample_{i}",
                theme_weights=theme_w,
                factor_weights=factor_w,
                train=metrics["train"],
                valid=metrics["valid"],
                test=metrics["test"],
            )
        )

    # Include baseline explicitly.
    candidates.append(
        TuningCandidate(
            name="baseline",
            theme_weights={},
            factor_weights=baseline_fw,
            train=baseline_metrics["train"],
            valid=baseline_metrics["valid"],
            test=baseline_metrics["test"],
        )
    )

    def sort_key(c: TuningCandidate) -> tuple[float, float, float]:
        return objective_tuple_from_metrics(c.valid)

    candidates.sort(key=sort_key, reverse=True)
    winner = candidates[0]

    beats_baseline = sort_key(winner) > baseline_obj
    if winner.name != "baseline" and not beats_baseline:
        winner = next(c for c in candidates if c.name == "baseline")

    # Winner must also beat baseline on test when not baseline itself.
    if winner.name != "baseline":
        if objective_tuple_from_metrics(winner.test) <= objective_tuple_from_metrics(
            baseline_metrics["test"]
        ):
            winner = next(c for c in candidates if c.name == "baseline")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "winner": {
            "name": winner.name,
            "theme_weights": winner.theme_weights,
            "factor_weights": winner.factor_weights,
            "train": winner.train,
            "valid": winner.valid,
            "test": winner.test,
        },
        "baseline": {
            "factor_weights": baseline_fw,
            **baseline_metrics,
        },
        "top_10": [
            {
                "name": c.name,
                "theme_weights": c.theme_weights,
                "factor_weights": c.factor_weights,
                "valid": c.valid,
                "test": c.test,
            }
            for c in candidates[:10]
        ],
        "n_samples": n_samples,
    }
    TUNING_RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Tuning complete; winner=%s", winner.name)
    return payload


def objective_tuple_from_metrics(metrics: dict[str, float]) -> tuple[float, float, float]:
    return (
        metrics.get("rolling_win_rate", 0.0),
        metrics.get("median_excess", 0.0),
        metrics.get("mean_ic", 0.0),
    )


def tune_bargain_weights(
    n_samples: int = 200,
    seed: int = 7,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Tune bargain component weights by ranking on historical bargain_score vs forward return.
    """
    panel = panel if panel is not None else load_factor_panel()
    from backtest.data.prices import load_prices
    from backtest.engine import _forward_quarter_returns

    prices = load_prices()
    fwd = _forward_quarter_returns(panel, prices)
    baseline = current_baseline_bargain_weights()
    rng = np.random.default_rng(seed)

    def score_with_weights(weights: dict[str, float]) -> float:
        ics: list[float] = []
        comp_cols = {
            "margin_of_safety": "bargain_margin_of_safety",
            "discount_ath": "bargain_discount_ath",
            "discount_52w": "bargain_discount_52w",
            "rsi_oversold": "bargain_rsi_oversold",
        }
        for qend, grp in panel.groupby("quarter_end"):
            scores = []
            for _, row in grp.iterrows():
                weighted = 0.0
                avail = 0.0
                for key, col in comp_cols.items():
                    val = row.get(col)
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        continue
                    w = weights.get(key, 0.0)
                    weighted += float(val) * w
                    avail += w
                score = weighted / avail if avail > 0 else np.nan
                scores.append({"ticker": row["ticker"], "bargain_score": score})
            if not scores:
                continue
            s_df = pd.DataFrame(scores).dropna()
            f_df = fwd[fwd["quarter_end"] == qend]
            merged = s_df.merge(f_df, on="ticker", how="inner")
            if len(merged) < 10:
                continue
            ic = merged["bargain_score"].corr(merged["fwd_return"], method="spearman")
            if ic is not None and not np.isnan(ic):
                ics.append(float(ic))
        return float(np.mean(ics)) if ics else 0.0

    baseline_ic = score_with_weights(baseline)
    best_w = baseline
    best_ic = baseline_ic
    samples: list[dict[str, Any]] = []

    for i in range(n_samples):
        w = random_bargain_weights(rng)
        ic = score_with_weights(w)
        samples.append({"weights": w, "mean_ic": ic})
        if ic > best_ic:
            best_ic = ic
            best_w = w

    out = {
        "winner_weights": best_w,
        "winner_mean_ic": best_ic,
        "baseline_weights": baseline,
        "baseline_mean_ic": baseline_ic,
        "top_5": sorted(samples, key=lambda x: x["mean_ic"], reverse=True)[:5],
    }
    path = RESULTS_DIR / "bargain_tuning_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out
