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


_DEFAULT_WEIGHTS: dict[str, float] = {
    "value": 0.12,
    "momentum": 0.10,
    "quality": 0.12,
    "low_volatility": 0.05,
    "investment": 0.04,
    "earnings_revisions": 0.07,
    "financial_strength": 0.06,
    "garp": 0.08,
    "balance_sheet_strength": 0.04,
    "graham_value": 0.06,
    "downside_protection": 0.04,
    "earnings_quality": 0.06,
    "shareholder_yield": 0.06,
    "capital_efficiency": 0.06,
    "distress_risk": 0.04,
}


def get_factor_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return factor weights from config, falling back to defaults for any missing key."""
    cfg = config or load_config()
    weights = cfg.get("factor_weights", {})
    return {family: float(weights.get(family, default)) for family, default in _DEFAULT_WEIGHTS.items()}


def get_thresholds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    t = cfg.get("thresholds", {})
    return {
        "composite_min": float(t.get("composite_min", 70)),
        "implied_upside_min_pct": float(t.get("implied_upside_min_pct", 15)),
        "exclude_sell_consensus": bool(t.get("exclude_sell_consensus", True)),
    }
