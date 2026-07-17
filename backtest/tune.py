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

from backtest.constants import (
    BARGAIN_BACKTEST_COMPONENTS,
    DCA_TOP_N,
    PRIMARY_EVAL_HORIZON,
    RESULTS_DIR,
)
from backtest.engine import (
    BacktestResult,
    bootstrap_mean_ci,
    compute_horizon_ics,
    dca_fold_excess_roi,
    init_backtest_cache,
    make_expanding_window_folds,
    make_quarter_folds,
    objective_tuple,
    precompute_multi_horizon_returns,
    precompute_quarter_end_prices,
    run_backtest,
    score_factor_panel,
    split_period,
)
from backtest.data.prices import load_delisted_catalog, load_prices
from backtest.factors import load_factor_panel
from backtest.weights import (
    current_baseline_bargain_weights,
    current_baseline_factor_weights,
    named_weight_candidates,
    random_bargain_weights,
    random_theme_weights,
    theme_weights_to_factor_weights,
)

logger = logging.getLogger(__name__)

TUNING_RESULTS_PATH = RESULTS_DIR / "tuning_results.json"
CV_TUNING_RESULTS_PATH = RESULTS_DIR / "tuning_results_dca_cv.json"
CANDIDATE_COMPARISON_PATH = RESULTS_DIR / "weight_candidate_comparison.json"

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


def _bargain_score_with_weights(panel: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    """Compute per-row bargain scores from component columns and weights."""
    comp_cols = {key: f"bargain_{key}" for key in BARGAIN_BACKTEST_COMPONENTS}
    rows: list[dict[str, Any]] = []
    for _, row in panel.iterrows():
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
        rows.append(
            {
                "quarter_end": row["quarter_end"],
                "ticker": row["ticker"],
                "bargain_score": score,
            }
        )
    return pd.DataFrame(rows)


def validate_bargain_weights(
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    horizon: str = PRIMARY_EVAL_HORIZON,
) -> dict[str, Any]:
    """
    Validate (not search) default long-horizon bargain weights via horizon IC.

    Optionally compares a few fixed alternative weightings for reporting.
    """
    panel = panel if panel is not None else load_factor_panel()
    prices = prices if prices is not None else load_prices()
    multi = precompute_multi_horizon_returns(panel, prices)
    baseline = current_baseline_bargain_weights()

    candidates = {
        "default_long_horizon": baseline,
        "graham_heavy": {"margin_of_safety": 0.55, "valuation_vs_history": 0.30, "discount_52w": 0.15},
        "equal": {"margin_of_safety": 1 / 3, "valuation_vs_history": 1 / 3, "discount_52w": 1 / 3},
    }

    results: dict[str, Any] = {}
    for name, weights in candidates.items():
        scored = _bargain_score_with_weights(panel, weights)
        scored = scored.rename(columns={"bargain_score": "score"})
        # Adapt to compute_horizon_ics expecting score_col on scored with quarter_end.
        scored_for_ic = scored.rename(columns={"score": "bargain_score"})
        ics = compute_horizon_ics(scored_for_ic, multi, score_col="bargain_score")
        results[name] = {"weights": weights, "horizon_ics": ics, "primary_ic": ics.get(horizon, 0.0)}

    # Prefer default unless another candidate is clearly better on the primary horizon.
    winner_name = max(results, key=lambda n: results[n]["primary_ic"])
    winner = results[winner_name]
    # Stick with default when indistinguishable (within 0.01 IC).
    default_ic = results["default_long_horizon"]["primary_ic"]
    if winner_name != "default_long_horizon" and winner["primary_ic"] - default_ic < 0.01:
        winner_name = "default_long_horizon"
        winner = results[winner_name]

    out = {
        "objective": f"validate_bargain_ic_{horizon}",
        "horizon": horizon,
        "winner_name": winner_name,
        "winner_weights": winner["weights"],
        "winner_mean_ic": winner["primary_ic"],
        "baseline_weights": baseline,
        "baseline_mean_ic": default_ic,
        "candidates": results,
    }
    path = RESULTS_DIR / "bargain_tuning_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def tune_bargain_weights(
    n_samples: int = 200,
    seed: int = 7,
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Legacy Dirichlet bargain search kept for CLI compatibility.

    Prefer ``validate_bargain_weights`` — the primary path validates fixed
    long-horizon weights instead of searching.
    """
    del n_samples, seed  # unused; validation path does not search
    return validate_bargain_weights(panel=panel, prices=prices)


def compare_named_candidates(
    panel: pd.DataFrame | None = None,
    prices: pd.DataFrame | None = None,
    composite_min: float = 50.0,
    bargain_min: float = 50.0,
    top_n: int = DCA_TOP_N,
) -> dict[str, Any]:
    """
    Compare evidence-based / legacy-tuned / equal weights on gated DCA + horizon ICs.

    This is the primary evaluation path (validation, not weight search).
    """
    panel = panel if panel is not None else load_factor_panel()
    prices = prices if prices is not None else load_prices()
    quarter_end_prices = precompute_quarter_end_prices(panel, prices)
    delisted = load_delisted_catalog()
    multi = precompute_multi_horizon_returns(panel, prices)
    folds = make_expanding_window_folds(panel["quarter_end"].unique())
    if not folds:
        # Fall back to contiguous k-folds if the series is short.
        kfolds = make_quarter_folds(panel["quarter_end"].unique(), k_folds=5)
        folds = [(fq, [fe]) for fq, fe in kfolds]

    candidates = named_weight_candidates()
    rows: list[dict[str, Any]] = []

    for name, weights in candidates.items():
        logger.info("Evaluating weight candidate: %s", name)
        scored = score_factor_panel(panel, weights)
        horizon_ics = compute_horizon_ics(scored, multi, score_col="composite")

        # Gated DCA picks: composite + bargain thresholds, top-N.
        picks: dict[pd.Timestamp, list[str]] = {}
        for qend, grp in scored.groupby("quarter_end"):
            valid = grp.dropna(subset=["composite", "bargain_score"])
            valid = valid[
                (valid["composite"] >= composite_min) & (valid["bargain_score"] >= bargain_min)
            ].nlargest(top_n, "composite")
            picks[pd.Timestamp(qend)] = valid["ticker"].astype(str).tolist()

        fold_excess: list[float] = []
        for train_q, test_q in folds:
            # Evaluate test window as a DCA campaign marked at the last test quarter.
            if not test_q:
                continue
            fold_end = test_q[-1] if isinstance(test_q, list) else test_q
            test_list = test_q if isinstance(test_q, list) else [test_q]
            ex = dca_fold_excess_roi(
                picks,
                quarter_end_prices,
                test_list,
                fold_end,
                delisted=delisted,
            )
            if ex is not None:
                fold_excess.append(ex)

        ci = bootstrap_mean_ci(fold_excess)
        rows.append(
            {
                "name": name,
                "factor_weights": weights,
                "horizon_ics": horizon_ics,
                "primary_ic": horizon_ics.get(PRIMARY_EVAL_HORIZON, 0.0),
                "fold_excess_roi": fold_excess,
                "excess_mean": ci["mean"],
                "excess_ci_low": ci["ci_low"],
                "excess_ci_high": ci["ci_high"],
                "frac_folds_positive": float(np.mean([e > 0 for e in fold_excess])) if fold_excess else 0.0,
            }
        )

    # Rank by mean excess; flag pairs whose CIs overlap as indistinguishable.
    rows.sort(key=lambda r: (r["excess_mean"] if not np.isnan(r["excess_mean"]) else -np.inf), reverse=True)
    winner = rows[0]
    comparisons: list[dict[str, Any]] = []
    for other in rows[1:]:
        overlap = (
            winner["excess_ci_low"] <= other["excess_ci_high"]
            and other["excess_ci_low"] <= winner["excess_ci_high"]
        )
        comparisons.append(
            {
                "winner": winner["name"],
                "challenger": other["name"],
                "indistinguishable": bool(overlap),
            }
        )

    # Prefer evidence_based when indistinguishable from the statistical winner.
    recommended = winner["name"]
    evidence = next((r for r in rows if r["name"] == "evidence_based"), None)
    if evidence is not None and recommended != "evidence_based":
        overlap = (
            winner["excess_ci_low"] <= evidence["excess_ci_high"]
            and evidence["excess_ci_low"] <= winner["excess_ci_high"]
        )
        if overlap:
            recommended = "evidence_based"

    payload = {
        "objective": "gated_dca_expanding_window_with_bootstrap",
        "primary_horizon": PRIMARY_EVAL_HORIZON,
        "composite_min": composite_min,
        "bargain_min": bargain_min,
        "top_n": top_n,
        "candidates": rows,
        "comparisons": comparisons,
        "recommended": recommended,
        "recommended_weights": next(r["factor_weights"] for r in rows if r["name"] == recommended),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_COMPARISON_PATH.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Candidate comparison complete; recommended=%s", recommended)
    return payload
