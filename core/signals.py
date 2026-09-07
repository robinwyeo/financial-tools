"""Overlay signals that are not blended into a factor group: short interest + uncertainty."""

from __future__ import annotations

from typing import Any

from core.data import _safe_float


def compute_short_interest(raw: dict[str, Any], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    FINRA/Yahoo short-interest overlay. Used as a flag, not a weighted factor.

    High short interest: days-to-cover >= 5 or short % of float >= 10%.
    """
    thresholds = thresholds or {}
    days_min = float(thresholds.get("short_interest_days_min", 5.0))
    float_min = float(thresholds.get("short_interest_float_pct_min", 0.10))

    days = _safe_float(raw.get("short_ratio"))
    pct_float = _safe_float(raw.get("short_percent_of_float"))
    shares_short = _safe_float(raw.get("shares_short"))
    if pct_float is not None and pct_float > 1.5:
        pct_float = pct_float / 100.0

    high = False
    if days is not None and days >= days_min:
        high = True
    if pct_float is not None and pct_float >= float_min:
        high = True

    return {
        "short_ratio": days,
        "short_percent_of_float": pct_float,
        "shares_short": shares_short,
        "high_short_interest": high,
    }


def compute_uncertainty(
    *,
    factor_coverage_pct: float | None,
    volatility_12m: float | None,
    target_high: float | None = None,
    target_low: float | None = None,
    target_mean: float | None = None,
    thresholds: dict[str, Any] | None = None,
    data_quality_grade: str | None = None,
    dcf_base: float | None = None,
    dcf_bear: float | None = None,
) -> dict[str, Any]:
    """
    Morningstar-style uncertainty badge: Low / Medium / High.

    Points (0-3): coverage < 80, 12m vol > 35%, analyst target range > 40% of mean.
    High uncertainty widens the Buy composite/bargain thresholds.
    """
    thresholds = thresholds or {}
    coverage_cut = float(thresholds.get("uncertainty_coverage_max", 80.0))
    vol_cut = float(thresholds.get("uncertainty_vol_min", 0.35))
    disp_cut = float(thresholds.get("uncertainty_dispersion_min", 0.40))
    medium_bump = float(thresholds.get("uncertainty_medium_bump", 3.0))
    high_bump = float(thresholds.get("uncertainty_high_bump", 6.0))

    dispersion = None
    if (
        target_high is not None
        and target_low is not None
        and target_mean is not None
        and target_mean > 0
        and target_high >= target_low
    ):
        dispersion = (target_high - target_low) / target_mean

    points = 0
    if factor_coverage_pct is not None and factor_coverage_pct < coverage_cut:
        points += 1
    if volatility_12m is not None and volatility_12m > vol_cut:
        points += 1
    if dispersion is not None and dispersion > disp_cut:
        points += 1
    if str(data_quality_grade or "").upper() == "B":
        points += 1
    if (
        dcf_base is not None
        and dcf_bear is not None
        and dcf_base > 0
        and abs(dcf_base - dcf_bear) / dcf_base > 0.50
    ):
        points += 1

    if points >= 2:
        label = "High"
        bump = high_bump
    elif points == 1:
        label = "Medium"
        bump = medium_bump
    else:
        label = "Low"
        bump = 0.0

    return {
        "label": label,
        "points": points,
        "threshold_bump": bump,
        "estimate_dispersion": dispersion,
        "coverage_pct": factor_coverage_pct,
        "volatility_12m": volatility_12m,
    }
