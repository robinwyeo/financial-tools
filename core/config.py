"""Configuration loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from core.factors import FACTOR_SCORE_COLUMNS
from core.fund_factors import FUND_FACTOR_SCORE_COLUMNS

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Evidence-based priors for long-horizon buy-and-hold (validated, not searched).
_DEFAULT_WEIGHTS: dict[str, float] = {
    "quality": 0.25,
    "value": 0.25,
    "capital_discipline": 0.125,
    "balance_sheet": 0.10,
    "garp": 0.10,
    "momentum": 0.075,
    "low_volatility": 0.05,
    "earnings_revisions": 0.05,
}


def get_factor_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return factor-group weights from config for the 8 composite groups only."""
    cfg = config or load_config()
    weights = cfg.get("factor_weights", {})
    return {
        family: float(weights.get(family, _DEFAULT_WEIGHTS.get(family, 0.0)))
        for family in FACTOR_SCORE_COLUMNS
    }


# Fund (ETF / mutual fund) composite priors: fees are the strongest documented
# predictor of long-run relative fund performance, then realized risk-adjusted
# returns; momentum and income are supporting signals.
_DEFAULT_FUND_WEIGHTS: dict[str, float] = {
    "cost": 0.25,
    "performance": 0.20,
    "risk_adjusted": 0.20,
    "low_volatility": 0.15,
    "momentum": 0.10,
    "income": 0.10,
}


def get_fund_factor_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return fund factor-group weights from config for the fund composite groups."""
    cfg = config or load_config()
    weights = cfg.get("fund_factor_weights", {})
    return {
        family: float(weights.get(family, _DEFAULT_FUND_WEIGHTS.get(family, 0.0)))
        for family in FUND_FACTOR_SCORE_COLUMNS
    }


# Long-horizon valuation bargain defaults (RSI removed).
_DEFAULT_BARGAIN_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.40,
    "valuation_vs_history": 0.35,
    "discount_52w": 0.25,
}

_BARGAIN_COMPONENT_KEYS: tuple[str, ...] = tuple(_DEFAULT_BARGAIN_WEIGHTS.keys())


def get_bargain_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return bargain component weights from config for the active components only."""
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
        "composite_min": float(t.get("composite_min", 50.0)),
        "bargain_min": float(t.get("bargain_min", 50.0)),
        # Informational only — not used as a hard good-buy gate.
        "implied_upside_min_pct": float(t.get("implied_upside_min_pct", 15)),
        "exclude_sell_consensus": bool(t.get("exclude_sell_consensus", True)),
        "require_implied_upside": bool(t.get("require_implied_upside", False)),
    }
