"""Weight configuration helpers for backtesting."""

from __future__ import annotations

import numpy as np

from backtest.constants import (
    BACKTEST_FACTOR_FAMILIES,
    DEFAULT_BARGAIN_WEIGHTS,
    EQUAL_FACTOR_WEIGHTS,
    EVIDENCE_BASED_FACTOR_WEIGHTS,
    EXCLUDED_COMPOSITE_FACTORS,
    FACTOR_THEMES,
    LEGACY_TUNED_FACTOR_WEIGHTS,
    WITHIN_THEME_PROPORTIONS,
)
from core.config import get_factor_weights, load_config
from core.scoring import BARGAIN_COMPONENT_WEIGHTS


def theme_weights_to_factor_weights(theme_weights: dict[str, float]) -> dict[str, float]:
    """Expand theme-level weights into per-factor weights."""
    out: dict[str, float] = {}
    for theme, factors in FACTOR_THEMES.items():
        theme_w = float(theme_weights.get(theme, 0.0))
        props = WITHIN_THEME_PROPORTIONS[theme]
        prop_sum = sum(props.values())
        if prop_sum <= 0:
            continue
        for factor, prop in props.items():
            out[factor] = theme_w * (prop / prop_sum)
    return out


def normalize_backtest_weights(weights: dict[str, float]) -> dict[str, float]:
    """Keep only reconstructable factor groups; renormalize to the backtestable mass."""
    cfg = load_config()
    full = get_factor_weights(cfg)
    excluded_mass = sum(full.get(f, 0.0) for f in EXCLUDED_COMPOSITE_FACTORS)
    target_sum = sum(full.values()) - excluded_mass

    kept = {f: float(weights.get(f, 0.0)) for f in BACKTEST_FACTOR_FAMILIES}
    total = sum(kept.values())
    if total <= 0:
        return kept
    scale = target_sum / total
    return {f: w * scale for f, w in kept.items()}


def current_baseline_factor_weights() -> dict[str, float]:
    full = get_factor_weights()
    return normalize_backtest_weights(full)


def named_weight_candidates() -> dict[str, dict[str, float]]:
    """Named factor-weight candidates for validation (not search)."""
    return {
        "evidence_based": normalize_backtest_weights(EVIDENCE_BASED_FACTOR_WEIGHTS),
        "legacy_tuned": normalize_backtest_weights(LEGACY_TUNED_FACTOR_WEIGHTS),
        "equal": normalize_backtest_weights(EQUAL_FACTOR_WEIGHTS),
    }


def random_theme_weights(rng: np.random.Generator) -> dict[str, float]:
    themes = list(FACTOR_THEMES.keys())
    sample = rng.dirichlet(np.ones(len(themes)))
    return {theme: float(w) for theme, w in zip(themes, sample)}


def normalize_bargain_weights(weights: dict[str, float]) -> dict[str, float]:
    """Renormalize bargain weights over the active components."""
    kept = {k: float(weights.get(k, 0.0)) for k in DEFAULT_BARGAIN_WEIGHTS}
    total = sum(kept.values())
    if total <= 0:
        return dict(DEFAULT_BARGAIN_WEIGHTS)
    return {k: v / total for k, v in kept.items()}


def current_baseline_bargain_weights() -> dict[str, float]:
    return normalize_bargain_weights(BARGAIN_COMPONENT_WEIGHTS)


def random_bargain_weights(rng: np.random.Generator) -> dict[str, float]:
    keys = list(DEFAULT_BARGAIN_WEIGHTS.keys())
    sample = rng.dirichlet(np.ones(len(keys)))
    raw = {k: float(w) for k, w in zip(keys, sample)}
    return normalize_bargain_weights(raw)
