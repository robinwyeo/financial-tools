"""Configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.factors import FACTOR_SCORE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# CV-tuned defaults from DCA k-fold cross-validation on 2010-2026 S&P 500 panel.
# 8 orthogonal groups replace the old 15 collinear families.
# earnings_revisions is live-only (excluded from backtest tuning; kept at 0.05).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "value": 0.0286,
    "garp": 0.4131,
    "quality": 0.1042,
    "balance_sheet": 0.1365,
    "momentum": 0.0688,
    "low_volatility": 0.1321,
    "capital_discipline": 0.0667,
    "earnings_revisions": 0.0500,
}


def get_factor_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return factor-group weights from config for the 8 composite groups only."""
    cfg = config or load_config()
    weights = cfg.get("factor_weights", {})
    return {
        family: float(weights.get(family, _DEFAULT_WEIGHTS.get(family, 0.0)))
        for family in FACTOR_SCORE_COLUMNS
    }


# Bargain component defaults from historical IC tuning (correct forward-return alignment).
# RSI oversold leads (mean-reversion signal); margin of safety is secondary.
# discount_52w has near-zero weight empirically (correlated with RSI oversold).
_DEFAULT_BARGAIN_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.2489,
    "discount_52w": 0.0024,
    "rsi_oversold": 0.7488,
}

_BARGAIN_COMPONENT_KEYS: tuple[str, ...] = tuple(_DEFAULT_BARGAIN_WEIGHTS.keys())


def get_bargain_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return bargain component weights from config for the three active components only."""
    cfg = config or load_config()
    weights = cfg.get("bargain_weights", {})
    return {
        key: float(weights.get(key, _DEFAULT_BARGAIN_WEIGHTS.get(key, 0.0)))
        for key in _BARGAIN_COMPONENT_KEYS
    }


def get_thresholds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    t = cfg.get("thresholds", {})
    return {
        "composite_min": float(t.get("composite_min", 51.3)),
        "bargain_min": float(t.get("bargain_min", 49.3)),
        "implied_upside_min_pct": float(t.get("implied_upside_min_pct", 15)),
        "exclude_sell_consensus": bool(t.get("exclude_sell_consensus", True)),
    }
