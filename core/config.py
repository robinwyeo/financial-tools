"""Configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

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
    "value": 0.0301,
    "garp": 0.4348,
    "quality": 0.1097,
    "balance_sheet": 0.1437,
    "momentum": 0.0724,
    "low_volatility": 0.1390,
    "capital_discipline": 0.0703,
    "earnings_revisions": 0.0500,
}


def get_factor_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return factor weights from config, falling back to defaults for any missing key."""
    cfg = config or load_config()
    weights = cfg.get("factor_weights", {})
    return {family: float(weights.get(family, default)) for family, default in _DEFAULT_WEIGHTS.items()}


# Bargain component defaults from historical IC tuning (correct forward-return alignment).
# RSI oversold leads (mean-reversion signal); margin of safety is secondary.
# discount_52w has near-zero weight empirically (correlated with RSI oversold).
_DEFAULT_BARGAIN_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.2489,
    "discount_52w": 0.0024,
    "rsi_oversold": 0.7488,
}


def get_bargain_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return bargain component weights from config, falling back to defaults."""
    cfg = config or load_config()
    weights = cfg.get("bargain_weights", {})
    return {key: float(weights.get(key, default)) for key, default in _DEFAULT_BARGAIN_WEIGHTS.items()}


def get_thresholds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    t = cfg.get("thresholds", {})
    return {
        "composite_min": float(t.get("composite_min", 51.3)),
        "bargain_min": float(t.get("bargain_min", 49.3)),
        "implied_upside_min_pct": float(t.get("implied_upside_min_pct", 15)),
        "exclude_sell_consensus": bool(t.get("exclude_sell_consensus", True)),
    }
