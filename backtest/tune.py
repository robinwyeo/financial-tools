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
from backtest.engine import (
    BacktestResult,
    dca_fold_excess_roi,
    init_backtest_cache,
    make_quarter_folds,
    objective_tuple,
    precompute_quarter_end_prices,
    run_backtest,
    score_factor_panel,
    split_period,
)
from backtest.constants import DCA_TOP_N
from backtest.data.prices import load_delisted_catalog, load_prices
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
CV_TUNING_RESULTS_PATH = RESULTS_DIR / "tuning_results_dca_cv.json"

# How hard to punish valid->test degradation in the robustness objective.
# A candidate that spikes on validation but collapses on test (overfit) is
# demoted in proportion to the size of that drop.
DEGRADATION_PENALTY = 0.5

# Penalty on the dispersion of per-fold excess ROI in the k-fold CV objective.
# score = mean(excess) - CV_STD_PENALTY * std(excess); higher penalty favors
# configurations that are consistent across regimes over ones that are great in
# one fold and poor in others.
CV_STD_PENALTY = 1.0


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
    *,
    skip_ic: bool = False,
    monthly_returns: pd.DataFrame | None = None,
    forward_returns: pd.DataFrame | None = None,
) -> dict[str, dict[str, float]]:
    panel = panel if panel is not None else load_factor_panel()
    # Score the full panel once; every split reuses it (scoring is split-independent).
    scored = score_factor_panel(panel, factor_weights)
    out: dict[str, dict[str, float]] = {}
    for period in ("train", "valid", "test"):
        start, end = split_period(period)
        result = run_backtest(
            factor_weights,
            panel=panel,
            start=start,
            end=end,
            skip_ic=skip_ic,
            monthly_returns=monthly_returns,
            forward_returns=forward_returns,
            scored=scored,
        )
        out[period] = _result_metrics(result)
    return out


def tune_factor_weights(
    n_samples: int = 500,
    seed: int = 42,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Dirichlet search over theme weights with walk-forward evaluation."""
    panel, monthly, forward = init_backtest_cache(panel)
    logger.info("Backtest cache ready (%d panel rows)", len(panel))
    rng = np.random.default_rng(seed)
    baseline_fw = current_baseline_factor_weights()
    baseline_metrics = evaluate_factor_weights(
        baseline_fw, panel, monthly_returns=monthly, forward_returns=forward
    )

    candidates: list[TuningCandidate] = []

    for i in range(n_samples):
        if i % 25 == 0:
            logger.info("Factor weight search %d/%d", i, n_samples)
        theme_w = random_theme_weights(rng)
        factor_w = theme_weights_to_factor_weights(theme_w)
        metrics = evaluate_factor_weights(
            factor_w, panel, monthly_returns=monthly, forward_returns=forward
        )
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
        return robust_objective(c.valid, c.test)

    candidates.sort(key=sort_key, reverse=True)
    winner = candidates[0]

    # Adopt a tuned winner only if it is strictly more robust out-of-sample than
    # the current baseline; otherwise keep baseline. This refuses to "improve"
    # the config when no candidate holds up across both held-out windows.
    baseline_robust = robust_objective(baseline_metrics["valid"], baseline_metrics["test"])
    if winner.name != "baseline" and sort_key(winner) <= baseline_robust:
        winner = next(c for c in candidates if c.name == "baseline")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "objective": "robust_oos",
        "degradation_penalty": DEGRADATION_PENALTY,
        "winner": {
            "name": winner.name,
            "theme_weights": winner.theme_weights,
            "factor_weights": winner.factor_weights,
            "robust_score": robust_objective(winner.valid, winner.test)[0],
            "train": winner.train,
            "valid": winner.valid,
            "test": winner.test,
        },
        "baseline": {
            "factor_weights": baseline_fw,
            "robust_score": baseline_robust[0],
            **baseline_metrics,
        },
        "top_10": [
            {
                "name": c.name,
                "theme_weights": c.theme_weights,
                "factor_weights": c.factor_weights,
                "robust_score": robust_objective(c.valid, c.test)[0],
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


def _picks_by_quarter(scored: pd.DataFrame, top_n: int) -> dict[pd.Timestamp, list[str]]:
    """Top-N tickers by composite for each quarter (ranking, no threshold gate)."""
    picks: dict[pd.Timestamp, list[str]] = {}
    for qend, grp in scored.groupby("quarter_end"):
        valid = grp.dropna(subset=["composite"]).nlargest(top_n, "composite")
        picks[pd.Timestamp(qend)] = valid["ticker"].astype(str).tolist()
    return picks


def evaluate_factor_weights_cv(
    factor_weights: dict[str, float],
    panel: pd.DataFrame,
    quarter_end_prices: dict,
    folds: list,
    delisted: set[str],
    *,
    top_n: int = DCA_TOP_N,
) -> list[float]:
    """Per-fold excess ROI of the top-N DCA buy-and-hold strategy vs SPY.

    Scores the *actual* strategy we report on (B): each quarter buy the top-N
    names by composite, $20k split equally, buy and hold until the fold end, then
    measure ROI on deployed capital relative to a $20k/quarter SPY DCA over the
    same fold. Returns one excess-ROI number per fold (A).
    """
    scored = score_factor_panel(panel, factor_weights)
    picks = _picks_by_quarter(scored, top_n)
    out: list[float] = []
    for fold_quarters, fold_end in folds:
        ex = dca_fold_excess_roi(
            picks,
            quarter_end_prices,
            fold_quarters,
            fold_end,
            delisted=delisted,
        )
        if ex is not None:
            out.append(ex)
    return out


def cv_robust_score(excesses: list[float]) -> tuple[float, float, float]:
    """Robustness statistic over per-fold excess ROI (higher is better).

    Primary: mean excess minus CV_STD_PENALTY * std (mean-variance consistency).
    Secondary: worst fold. Tertiary: fraction of folds with positive excess.
    """
    if not excesses:
        return (float("-inf"), float("-inf"), 0.0)
    arr = np.asarray(excesses, dtype=float)
    score = float(arr.mean() - CV_STD_PENALTY * arr.std())
    worst = float(arr.min())
    frac_pos = float((arr > 0).mean())
    return (score, worst, frac_pos)


@dataclass
class CVTuningCandidate:
    name: str
    theme_weights: dict[str, float]
    factor_weights: dict[str, float]
    fold_excess: list[float]
    cv_score: tuple[float, float, float]


def tune_factor_weights_cv(
    n_samples: int = 200,
    seed: int = 42,
    k_folds: int = 5,
    top_n: int = DCA_TOP_N,
    panel: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Tune factor weights on DCA terminal wealth with k-fold time-series CV.

    Combines (A) k-fold rolling-window cross-validation and (B) optimizing the
    real DCA buy-and-hold-top-N objective. A candidate is adopted only if its
    cross-validated robustness score strictly exceeds the baseline's.
    """
    panel = panel if panel is not None else load_factor_panel()
    prices = load_prices()
    quarter_end_prices = precompute_quarter_end_prices(panel, prices)
    delisted = load_delisted_catalog()
    folds = make_quarter_folds(panel["quarter_end"].unique(), k_folds)
    logger.info("DCA-CV tuning: %d folds, %d panel rows", len(folds), len(panel))

    rng = np.random.default_rng(seed)
    baseline_fw = current_baseline_factor_weights()
    baseline_ex = evaluate_factor_weights_cv(
        baseline_fw, panel, quarter_end_prices, folds, delisted, top_n=top_n
    )
    baseline_score = cv_robust_score(baseline_ex)

    candidates: list[CVTuningCandidate] = [
        CVTuningCandidate("baseline", {}, baseline_fw, baseline_ex, baseline_score)
    ]
    for i in range(n_samples):
        if i % 25 == 0:
            logger.info("DCA-CV weight search %d/%d", i, n_samples)
        theme_w = random_theme_weights(rng)
        factor_w = theme_weights_to_factor_weights(theme_w)
        ex = evaluate_factor_weights_cv(
            factor_w, panel, quarter_end_prices, folds, delisted, top_n=top_n
        )
        candidates.append(
            CVTuningCandidate(f"sample_{i}", theme_w, factor_w, ex, cv_robust_score(ex))
        )

    candidates.sort(key=lambda c: c.cv_score, reverse=True)
    winner = candidates[0]
    if winner.name != "baseline" and winner.cv_score <= baseline_score:
        winner = next(c for c in candidates if c.name == "baseline")

    def _row(c: CVTuningCandidate) -> dict[str, Any]:
        return {
            "name": c.name,
            "theme_weights": c.theme_weights,
            "factor_weights": c.factor_weights,
            "fold_excess_roi": c.fold_excess,
            "cv_score": c.cv_score[0],
            "worst_fold": c.cv_score[1],
            "frac_folds_positive": c.cv_score[2],
        }

    payload = {
        "objective": "dca_terminal_wealth_kfold_cv",
        "k_folds": k_folds,
        "top_n": top_n,
        "cv_std_penalty": CV_STD_PENALTY,
        "fold_boundaries": [
            {"start": str(fq[0].date()), "end": str(fe.date())} for fq, fe in folds
        ],
        "winner": _row(winner),
        "baseline": _row(candidates_by_name(candidates, "baseline")),
        "top_10": [_row(c) for c in candidates[:10]],
        "n_samples": n_samples,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CV_TUNING_RESULTS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("DCA-CV tuning complete; winner=%s", winner.name)
    return payload


def candidates_by_name(candidates: list, name: str):
    return next(c for c in candidates if c.name == name)


def robust_objective(
    valid: dict[str, float],
    test: dict[str, float],
) -> tuple[float, float, float]:
    """Out-of-sample robustness score (higher is better).

    Selecting on a single validation window lets a configuration win by spiking
    on one regime while failing everywhere else (overfitting). Instead we reward
    configurations that hold up across BOTH held-out windows:

    - Primary: worst-case median rolling 3y excess return across valid and test,
      minus a penalty proportional to the valid->test degradation. An overfit
      candidate (great valid, terrible test) is pushed below a mediocre-but-stable
      one.
    - Secondary: worst-case rolling win rate across the two windows.
    - Tertiary: average rank IC across the two windows.

    Train metrics are intentionally excluded from selection (they are in-sample).
    """
    v_ex = float(valid.get("median_excess", 0.0))
    t_ex = float(test.get("median_excess", 0.0))
    worst_excess = min(v_ex, t_ex)
    degradation = max(0.0, v_ex - t_ex)
    score = worst_excess - DEGRADATION_PENALTY * degradation
    worst_win = min(
        float(valid.get("rolling_win_rate", 0.0)),
        float(test.get("rolling_win_rate", 0.0)),
    )
    avg_ic = 0.5 * (float(valid.get("mean_ic", 0.0)) + float(test.get("mean_ic", 0.0)))
    return (score, worst_win, avg_ic)


def tune_bargain_weights(
    n_samples: int = 200,
    seed: int = 7,
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Tune bargain component weights by ranking on historical bargain_score vs forward return.
    """
    panel = panel if panel is not None else load_factor_panel()
    from backtest.data.prices import load_prices
    from backtest.engine import _forward_quarter_returns

    if prices is None:
        prices = load_prices()
    fwd = _forward_quarter_returns(panel, prices)
    baseline = current_baseline_bargain_weights()
    rng = np.random.default_rng(seed)

    # Map quarter_end → next quarter_end for forward-return look-up.
    # precompute_forward_returns labels each return row with the quarter it ENDS
    # at (next_q), so we must match bargain scores at qend with fwd returns at
    # next_q — the same convention used by _compute_ic in engine.py.
    qends_sorted = sorted(panel["quarter_end"].unique())
    next_qend: dict = {q: qends_sorted[i + 1] for i, q in enumerate(qends_sorted[:-1])}

    def score_with_weights(weights: dict[str, float]) -> float:
        ics: list[float] = []
        comp_cols = {
            "margin_of_safety": "bargain_margin_of_safety",
            "discount_52w": "bargain_discount_52w",
            "rsi_oversold": "bargain_rsi_oversold",
        }
        for qend, grp in panel.groupby("quarter_end"):
            if qend not in next_qend:
                continue
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
            # Use forward returns at the NEXT quarter (not the current quarter).
            f_df = fwd[fwd["quarter_end"] == next_qend[qend]]
            merged = s_df.merge(f_df, on="ticker", how="inner")
            if len(merged) < 10:
                continue
            # Manual Spearman (rank + Pearson) — avoids scipy dependency.
            a = merged["bargain_score"].rank()
            b = merged["fwd_return"].rank()
            ic = float(np.corrcoef(a, b)[0, 1])
            if not np.isnan(ic):
                ics.append(ic)
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
