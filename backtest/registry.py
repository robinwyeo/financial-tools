"""Single registry of backtest metrics and decision signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd

from backtest.stats import block_bootstrap_ci, evidence_card, n_independent_windows

MetricFn = Callable[[pd.DataFrame], pd.Series]
SignalFn = Callable[[pd.Series], str]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    family: str
    inputs: tuple[str, ...]
    fn: MetricFn
    higher_is_better: bool = True


@dataclass(frozen=True)
class SignalSpec:
    name: str
    fn: Callable[[pd.Series], str]


_METRICS: list[MetricSpec] = []
_SIGNALS: list[SignalSpec] = []


def register(spec: MetricSpec) -> MetricSpec:
    _METRICS.append(spec)
    return spec


def register_signal(spec: SignalSpec) -> SignalSpec:
    _SIGNALS.append(spec)
    return spec


def all_metrics() -> list[MetricSpec]:
    return list(_METRICS)


def all_signals() -> list[SignalSpec]:
    return list(_SIGNALS)


def get_metric(name: str) -> MetricSpec | None:
    for spec in _METRICS:
        if spec.name == name:
            return spec
    return None


def get_signal(name: str) -> SignalSpec | None:
    for spec in _SIGNALS:
        if spec.name == name:
            return spec
    return None


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _register_defaults() -> None:
    if _METRICS:
        return
    register(MetricSpec("composite", "score", ("composite",), lambda d: _col(d, "composite")))
    register(MetricSpec("quality_score", "quality", ("quality_score",), lambda d: _col(d, "quality_score")))
    register(MetricSpec("bargain_score", "bargain", ("bargain_score",), lambda d: _col(d, "bargain_score")))
    register(MetricSpec("earnings_yield", "value", ("earnings_yield",), lambda d: _col(d, "earnings_yield")))
    register(MetricSpec("roic", "quality", ("roic",), lambda d: _col(d, "roic")))
    register(MetricSpec("ev_to_ebit", "value", ("ev_to_ebit",), lambda d: -_col(d, "ev_to_ebit"), higher_is_better=True))

    def _decision_label(row: pd.Series) -> str:
        val = row.get("decision_label")
        if isinstance(val, str) and val:
            return val
        if bool(row.get("is_good_buy")):
            return "Accumulate"
        return "Avoid"

    register_signal(SignalSpec("decision", _decision_label))


_register_defaults()


def evaluate_metric(
    panel: pd.DataFrame,
    spec: MetricSpec,
    *,
    fwd_col: str = "excess_3y",
    horizon_quarters: int = 12,
) -> dict[str, Any]:
    from backtest.stats import decile_spread

    work = panel.copy()
    work["_metric"] = spec.fn(work)
    if not spec.higher_is_better:
        work["_metric"] = -work["_metric"]
    ics: list[float] = []
    if "quarter_end" in work.columns:
        for _, grp in work.groupby("quarter_end"):
            sub = grp.dropna(subset=["_metric", fwd_col]) if fwd_col in grp.columns else grp.dropna(subset=["_metric"])
            if len(sub) < 10 or fwd_col not in sub.columns:
                continue
            ic = sub["_metric"].rank().corr(sub[fwd_col].rank())
            if ic is not None and np.isfinite(ic):
                ics.append(float(ic))
    spread = None
    if fwd_col in work.columns:
        try:
            spread = decile_spread(work, fwd_col, "_metric", horizon_quarters)
        except Exception:
            spread = None
    coverage = float(work["_metric"].notna().mean()) if len(work) else 0.0
    return evidence_card(
        name=spec.name,
        ic_series=ics,
        horizon_quarters=horizon_quarters,
        coverage=coverage,
        spread=spread,
    )


def evaluate_signal(
    panel: pd.DataFrame,
    spec: SignalSpec,
    *,
    fwd_3y: str = "excess_3y",
    fwd_5y: str = "excess_5y",
) -> dict[str, Any]:
    work = panel.copy()
    labels = work.apply(spec.fn, axis=1)
    work["_label"] = labels
    acc = work[work["_label"] == "Accumulate"]
    by_q = acc.groupby("quarter_end").size() if "quarter_end" in acc.columns else pd.Series(dtype=int)

    def _empty_hit() -> dict[str, Any]:
        ci = block_bootstrap_ci([], block=12, seed=42)
        return {
            "hit_rate": None,
            "mean_excess": None,
            "n": 0,
            "mean_excess_ci_low": ci["ci_low"],
            "mean_excess_ci_high": ci["ci_high"],
        }

    def _hit(col: str) -> dict[str, Any]:
        if col not in acc.columns or acc.empty:
            return _empty_hit()
        s = pd.to_numeric(acc[col], errors="coerce").dropna()
        if s.empty:
            return _empty_hit()
        if "quarter_end" in acc.columns:
            q_means = acc.loc[s.index].assign(_ex=s).groupby("quarter_end")["_ex"].mean()
            ci = block_bootstrap_ci(q_means, block=12, seed=42)
        else:
            ci = block_bootstrap_ci(s, block=12, seed=42)
        return {
            "hit_rate": float((s > 0).mean()),
            "mean_excess": float(s.mean()),
            "n": int(len(s)),
            "mean_excess_ci_low": ci["ci_low"],
            "mean_excess_ci_high": ci["ci_high"],
        }
    return {
        "signal": spec.name,
        "accumulate_per_quarter": by_q.tolist(),
        "mean_accumulate_per_quarter": float(by_q.mean()) if len(by_q) else 0.0,
        "n_quarters": int(by_q.shape[0]),
        "n_independent_windows_3y": n_independent_windows(int(by_q.shape[0]), 12),
        "stats_3y": _hit(fwd_3y),
        "stats_5y": _hit(fwd_5y),
    }


def format_card(card: dict[str, Any]) -> str:
    lines = [
        f"metric: {card.get('name')}",
        f"mean IC: {card.get('mean_ic')}",
        f"NW t-stat: {card.get('nw_tstat')}",
        f"decile spread: {card.get('decile_spread')} "
        f"[{card.get('decile_spread_ci_low')}, {card.get('decile_spread_ci_high')}]",
        f"n quarters: {card.get('n_quarters')}",
        f"n independent windows: {card.get('n_independent_windows')}",
        f"coverage: {card.get('coverage')}",
    ]
    return "\n".join(lines)
