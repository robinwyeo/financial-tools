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


def get_factor_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = config or load_config()
    weights = cfg.get("factor_weights", {})
    return {
        "value": float(weights.get("value", 0.14)),
        "momentum": float(weights.get("momentum", 0.12)),
        "quality": float(weights.get("quality", 0.14)),
        "low_volatility": float(weights.get("low_volatility", 0.08)),
        "investment": float(weights.get("investment", 0.06)),
        "earnings_revisions": float(weights.get("earnings_revisions", 0.08)),
        "financial_strength": float(weights.get("financial_strength", 0.08)),
        "garp": float(weights.get("garp", 0.10)),
        "balance_sheet_strength": float(weights.get("balance_sheet_strength", 0.06)),
        "graham_value": float(weights.get("graham_value", 0.08)),
        "downside_protection": float(weights.get("downside_protection", 0.06)),
    }


def get_thresholds(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    t = cfg.get("thresholds", {})
    return {
        "composite_min": float(t.get("composite_min", 70)),
        "implied_upside_min_pct": float(t.get("implied_upside_min_pct", 15)),
        "exclude_sell_consensus": bool(t.get("exclude_sell_consensus", True)),
    }
