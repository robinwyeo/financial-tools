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
# earnings_revisions was dropped (it was recommendation-keyword scraping, never
# historically validated); its 0.05 went to momentum/low_volatility.
_DEFAULT_WEIGHTS: dict[str, float] = {
    "quality": 0.225,
    "value": 0.25,
    "capital_discipline": 0.10,
    "balance_sheet": 0.10,
    "garp": 0.05,
    "momentum": 0.10,
    "low_volatility": 0.05,
    "estimate_revisions": 0.10,
    "insider": 0.025,
}


def get_factor_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    """Return factor-group weights from config for the composite groups only."""
    cfg = config or load_config()
    weights = cfg.get("factor_weights", {})
    return {
        family: float(weights.get(family, _DEFAULT_WEIGHTS.get(family, 0.0)))
        for family in FACTOR_SCORE_COLUMNS
    }


# Fund (ETF / mutual fund) composite priors. Fees are the single robust
# predictor of long-run relative fund performance, so cost dominates. Past
# returns are a weak-to-negative predictor after fees (performance chasing),
# so raw 3y/5y returns and the return/vol ratio are deliberately down-weighted.
# Distribution yield is a payout preference, not a return signal.
_DEFAULT_FUND_WEIGHTS: dict[str, float] = {
    "cost": 0.35,
    "performance": 0.15,
    "risk_adjusted": 0.15,
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


# Long-horizon valuation bargain defaults (RSI removed). graham_heavy weights
# (0.55/0.30/0.15) are the live default. The old 0.40/0.35/0.25 mix is registered
# as bargain candidate `legacy_040_035_025`. Do not cite pre-companyfacts IC.
_DEFAULT_BARGAIN_WEIGHTS: dict[str, float] = {
    "margin_of_safety": 0.55,
    "valuation_vs_history": 0.30,
    "discount_52w": 0.15,
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
        # Minimum share of factor-group weight that must have data before a
        # Buy is allowed (guards against renormalization over sparse data).
        "coverage_min_pct": float(t.get("coverage_min_pct", 70.0)),
        # Hard distress disqualifier: Altman Z below this blocks a Buy outright
        # (1.8 = classic distress-zone boundary). Missing Z never blocks.
        "altman_z_min": float(t.get("altman_z_min", 1.8)),
        "altman_zpp_min": float(t.get("altman_zpp_min", 1.1)),
        "uncertainty_medium_bump": float(t.get("uncertainty_medium_bump", 3.0)),
        "uncertainty_high_bump": float(t.get("uncertainty_high_bump", 6.0)),
        "uncertainty_coverage_max": float(t.get("uncertainty_coverage_max", 80.0)),
        "uncertainty_vol_min": float(t.get("uncertainty_vol_min", 0.35)),
        "uncertainty_dispersion_min": float(t.get("uncertainty_dispersion_min", 0.40)),
        "short_interest_days_min": float(t.get("short_interest_days_min", 5.0)),
        "short_interest_float_pct_min": float(t.get("short_interest_float_pct_min", 0.10)),
        # Informational only — not used as a hard good-buy gate.
        "implied_upside_min_pct": float(t.get("implied_upside_min_pct", 15)),
        "exclude_sell_consensus": bool(t.get("exclude_sell_consensus", True)),
        "exclude_underperform": bool(t.get("exclude_underperform", True)),
        "require_implied_upside": bool(t.get("require_implied_upside", False)),
    }


_DEFAULT_HURDLE: dict[str, float] = {
    "floor": 0.08,
    "erp": 0.045,
    "fallback_rf": 0.042,
    "high_uncertainty_bump": 0.01,
}

_DEFAULT_VALUATION: dict[str, Any] = {
    "terminal_growth": 0.025,
    "explicit_years": 10,
    "fade_start_year": 5,
    "base_growth_cap": 0.15,
    "base_growth_floor": 0.0,
    "bear_growth_multiplier": 0.5,
    "bear_growth_cap": 0.04,
    "bear_rate_bump": 0.01,
    "subtract_sbc": True,
    "normalization_years": 3,
    "cyclical_normalization_years": 7,
    "cyclical_sectors": [
        "Energy",
        "Basic Materials",
        "Industrials",
        "Consumer Cyclical",
    ],
    "cyclical_industries": [
        "Semiconductors",
        "Semiconductor Equipment & Materials",
    ],
    "min_tax_rate": 0.15,
    "max_tax_rate": 0.35,
    "expected_return_growth_cap": 0.06,
}

_DEFAULT_QUALITY_WEIGHTS: dict[str, float] = {
    "profitability": 0.30,
    "earnings_quality": 0.20,
    "financial_strength": 0.20,
    "stability": 0.15,
    "capital_discipline": 0.15,
}

_DEFAULT_VALUE_TRAP: dict[str, Any] = {
    "max_flags": 1,
    "roic_hurdle": 0.08,
    "revenue_cagr_min": 0.0,
    "gross_margin_drop_max": 0.05,
    "interest_coverage_min": 3.0,
    "share_cagr_max": 0.03,
    "accruals_max": 0.10,
    "fcf_conversion_min": 0.50,
    "net_debt_ebitda_max": 3.5,
    "net_debt_ebitda_sector_caps": {
        "Utilities": 6.0,
        "Real Estate": 7.0,
        "Communication Services": 4.5,
        "Energy": 3.0,
    },
}

_DEFAULT_DECISION: dict[str, Any] = {
    "mode": "intrinsic",
    "required_margin_of_safety": {"Low": 0.20, "Medium": 0.30, "High": 0.40},
    "min_quality_percentile": 40.0,
    "max_value_trap_flags": 1,
    "min_expected_return_over_hurdle": 0.0,
    "block_on_data_quality_c": True,
}


def get_valuation_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    raw = dict(_DEFAULT_VALUATION)
    user = cfg.get("valuation") or {}
    for key, default in _DEFAULT_VALUATION.items():
        if key == "hurdle":
            continue
        raw[key] = user.get(key, default)
    hurdle_user = user.get("hurdle") or cfg.get("hurdle") or {}
    hurdle = dict(_DEFAULT_HURDLE)
    hurdle.update({k: float(hurdle_user[k]) for k in _DEFAULT_HURDLE if k in hurdle_user})
    raw["hurdle"] = hurdle
    return raw


def get_quality_weights(config: dict[str, Any] | None = None) -> dict[str, float]:
    cfg = config or load_config()
    weights = cfg.get("quality_weights") or {}
    return {
        k: float(weights.get(k, _DEFAULT_QUALITY_WEIGHTS[k]))
        for k in _DEFAULT_QUALITY_WEIGHTS
    }


def get_value_trap_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    user = cfg.get("value_trap") or {}
    out = dict(_DEFAULT_VALUE_TRAP)
    out.update(user)
    caps = dict(_DEFAULT_VALUE_TRAP["net_debt_ebitda_sector_caps"])
    caps.update(user.get("net_debt_ebitda_sector_caps") or {})
    out["net_debt_ebitda_sector_caps"] = caps
    return out


def get_decision_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = config or load_config()
    user = cfg.get("decision") or {}
    out = dict(_DEFAULT_DECISION)
    out.update(user)
    mos = dict(_DEFAULT_DECISION["required_margin_of_safety"])
    mos.update(user.get("required_margin_of_safety") or {})
    out["required_margin_of_safety"] = {k: float(v) for k, v in mos.items()}
    out["mode"] = str(out.get("mode") or "intrinsic").lower()
    return out


def get_universe_members(config: dict[str, Any] | None = None) -> list[str]:
    cfg = config or load_config()
    members = (cfg.get("universe") or {}).get("members") or ["sp500"]
    return [str(m).lower() for m in members]


def get_provider_names(config: dict[str, Any] | None = None) -> list[str]:
    cfg = config or load_config()
    names = (cfg.get("providers") or {}).get("fundamentals") or ["edgar", "yahoo"]
    return [str(n).lower() for n in names]
