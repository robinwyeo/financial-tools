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


# Theme-grouped defaults (kept in sync with config.yaml). Weights are assigned
# per theme then split across correlated sub-factors; theme totals in comments.
_DEFAULT_WEIGHTS: dict[str, float] = {
    # Value theme (0.21)
    "value": 0.07,
    "garp": 0.07,
    "graham_value": 0.07,
    # Quality theme (0.24)
    "quality": 0.06,
    "financial_strength": 0.06,
    "earnings_quality": 0.06,
    "capital_efficiency": 0.06,
    # Trend theme (0.15)
    "momentum": 0.08,
    "earnings_revisions": 0.07,
    # Risk theme (0.10)
    "low_volatility": 0.05,
    "downside_protection": 0.05,
    # Solvency theme (0.10)
    "balance_sheet_strength": 0.05,
    "distress_risk": 0.05,
    # Capital-allocation theme (0.20)
    "shareholder_yield": 0.10,
    "investment": 0.10,
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
        "composite_min": float(t.get("composite_min", 50)),
        "bargain_min": float(t.get("bargain_min", 50)),
        "implied_upside_min_pct": float(t.get("implied_upside_min_pct", 15)),
        "exclude_sell_consensus": bool(t.get("exclude_sell_consensus", True)),
    }
