"""Layer A quality score and Layer B value-trap flags."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from core.config import get_quality_weights, get_value_trap_config
from core.factors import (
    FINANCIALS_QUALITY_COLUMNS,
    QUALITY_SCORE_COLUMNS,
    QUALITY_SUB_BUCKETS,
)
from core.fundamentals import Fundamentals


@dataclass
class Flag:
    code: str
    triggered: bool
    value: float | None
    threshold: float | None
    note: str = ""


def _num(val: Any) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out


def compute_quality_score(
    df: pd.DataFrame,
    config: dict[str, Any] | None = None,
    *,
    group_col: str | None = "sector",
) -> pd.DataFrame:
    """Add quality_score (0-100) and quality_coverage_pct to a factor snapshot."""
    from core.scoring import compute_family_percentile

    result = df.copy()
    weights = get_quality_weights(config)
    use_sector = group_col is not None and group_col in result.columns

    for family, cols in QUALITY_SCORE_COLUMNS.items():
        sector_rel = family in {"profitability", "financial_strength"} and use_sector
        result[f"q_{family}"] = compute_family_percentile(
            result,
            cols,
            group_col=group_col if sector_rel else None,
            buckets=QUALITY_SUB_BUCKETS.get(family),
        )

    # Financials reduced set overwrites those rows.
    if "sector" in result.columns:
        fin_mask = result["sector"].astype(str) == "Financial Services"
        if fin_mask.any():
            fin_df = result.loc[fin_mask].copy()
            for family, cols in FINANCIALS_QUALITY_COLUMNS.items():
                fin_df[f"q_{family}"] = compute_family_percentile(fin_df, cols, group_col=None)
            for family in QUALITY_SCORE_COLUMNS:
                result.loc[fin_mask, f"q_{family}"] = fin_df[f"q_{family}"]

    weight_total = sum(weights.values()) or 1.0
    scores: list[float | None] = []
    coverages: list[float] = []
    for _, row in result.iterrows():
        weighted = 0.0
        avail = 0.0
        cov = 0.0
        families = QUALITY_SCORE_COLUMNS
        if str(row.get("sector") or "") == "Financial Services":
            families = FINANCIALS_QUALITY_COLUMNS
        for family, cols in families.items():
            w = float(weights.get(family, 0.0))
            pct = row.get(f"q_{family}")
            if pct is not None and not (isinstance(pct, float) and np.isnan(pct)):
                weighted += float(pct) * w
                avail += w
            present = 0
            for c in cols:
                val = row[c] if c in row.index else None
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    present += 1
            cov += w * (present / len(cols) if cols else 0.0)
        scores.append(None if avail == 0 else weighted / avail)
        coverages.append(cov / weight_total * 100.0)
    result["quality_score"] = scores
    result["quality_coverage_pct"] = coverages
    return result


def value_trap_flags(
    raw: dict[str, Any],
    fund: Fundamentals | None = None,
    config: dict[str, Any] | None = None,
) -> list[Flag]:
    """Named Layer B flags. Financials skip the leverage flag."""
    cfg = get_value_trap_config(config)
    sector = str(raw.get("sector") or "")
    flags: list[Flag] = []

    roic = _num(raw.get("roic"))
    hurdle = float(cfg.get("roic_hurdle", 0.08))
    declining = False
    roic_hist = raw.get("roic_history")
    if isinstance(roic_hist, (list, tuple)) and len(roic_hist) >= 3:
        tail = [_num(v) for v in list(roic_hist)[-3:]]
        if all(v is not None for v in tail):
            declining = tail[0] > tail[1] > tail[2]  # type: ignore[operator]
    elif fund is not None and fund.annual is not None and not fund.annual.empty and "ebit" in fund.annual.columns:
        # Approximate 3y ROIC path via EBIT / assets when a dedicated history is absent.
        ebit = pd.to_numeric(fund.annual.get("ebit"), errors="coerce").dropna().tail(3)
        assets = pd.to_numeric(fund.annual.get("total_assets"), errors="coerce").reindex(ebit.index)
        if len(ebit) == 3 and assets.notna().all():
            series = (ebit / assets.replace(0, np.nan)).dropna()
            if len(series) == 3:
                declining = bool(series.iloc[0] > series.iloc[1] > series.iloc[2])
    roic_below = roic is not None and roic < hurdle
    flags.append(
        Flag(
            "ROIC_DECLINING",
            bool(roic_below and declining),
            roic,
            hurdle,
            "ROIC below hurdle and declining 3 consecutive years",
        )
    )

    rev_cagr = _num(raw.get("revenue_5y_cagr"))
    rev_min = float(cfg.get("revenue_cagr_min", 0.0))
    flags.append(
        Flag("REVENUE_CAGR_NEG", rev_cagr is not None and rev_cagr < rev_min, rev_cagr, rev_min, "5y revenue CAGR < 0")
    )

    gm_delta = _num(raw.get("gross_margin_5y_delta"))
    gm_drop = float(cfg.get("gross_margin_drop_max", 0.05))
    flags.append(
        Flag(
            "GROSS_MARGIN_DOWN",
            gm_delta is not None and gm_delta < -gm_drop,
            gm_delta,
            -gm_drop,
            "Gross margin down more than 500 bps over 5y",
        )
    )

    nd = _num(raw.get("net_debt_to_ebitda"))
    icov = _num(raw.get("interest_coverage"))
    default_cap = float(cfg.get("net_debt_ebitda_max", 3.5))
    caps = cfg.get("net_debt_ebitda_sector_caps") or {}
    cap = float(caps.get(sector, default_cap))
    icov_min = float(cfg.get("interest_coverage_min", 3.0))
    lev_trig = False
    if sector != "Financial Services":
        if nd is not None and nd > cap:
            lev_trig = True
        if icov is not None and icov < icov_min:
            lev_trig = True
    flags.append(
        Flag(
            "LEVERAGE",
            lev_trig,
            nd if nd is not None else icov,
            cap,
            f"Net debt/EBITDA above {cap} or interest coverage < {icov_min}",
        )
    )

    share_cagr = _num(raw.get("share_cagr_3y"))
    share_max = float(cfg.get("share_cagr_max", 0.03))
    flags.append(
        Flag("DILUTION", share_cagr is not None and share_cagr > share_max, share_cagr, share_max, "Share count CAGR 3y > 3%")
    )

    accruals = _num(raw.get("accruals"))
    acc_max = float(cfg.get("accruals_max", 0.10))
    flags.append(
        Flag("ACCRUALS", accruals is not None and accruals > acc_max, accruals, acc_max, "Accruals ratio > 10% of assets")
    )

    conv = _num(raw.get("fcf_conversion_3y"))
    conv_min = float(cfg.get("fcf_conversion_min", 0.50))
    flags.append(
        Flag("FCF_CONVERSION", conv is not None and conv < conv_min, conv, conv_min, "FCF conversion < 50% over 3y")
    )

    oe = _num(raw.get("owner_earnings_norm"))
    flags.append(Flag("NEGATIVE_OE", oe is not None and oe < 0, oe, 0.0, "Negative normalized owner earnings"))
    return flags


def flags_to_dicts(flags: list[Flag]) -> list[dict[str, Any]]:
    return [asdict(f) for f in flags]
