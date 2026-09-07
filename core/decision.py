"""Accumulate / Watch / Avoid decision with visible gates and a buy-below price."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from core.config import get_decision_config, get_thresholds, get_value_trap_config


@dataclass
class GateResult:
    name: str
    passed: bool
    actual: Any
    threshold: Any
    note: str = ""


@dataclass
class Decision:
    label: str
    buy_below_price: float | None
    pct_to_buy: float | None
    gates: list[GateResult] = field(default_factory=list)
    flags: list[dict[str, Any]] = field(default_factory=list)
    timing_context: dict[str, Any] = field(default_factory=dict)
    mode: str = "intrinsic"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


def _mos_for(label: str | None, cfg: dict[str, Any]) -> float:
    table = cfg.get("required_margin_of_safety") or {}
    key = str(label or "Medium")
    if key not in table:
        key = "Medium"
    return float(table.get(key, 0.30))


def _timing_context(analysis: dict[str, Any]) -> dict[str, Any]:
    mom = analysis.get("momentum_12_1")
    if mom is None:
        mom = (analysis.get("factors_raw") or {}).get("momentum_12_1")
    hint = None
    if mom is not None and float(mom) < 0:
        hint = "negative 12-1 momentum: consider tranches"
    return {
        "momentum_12_1": mom,
        "discount_52w": (analysis.get("bargain") or {}).get("components", {}).get("discount_52w")
        if isinstance(analysis.get("bargain"), dict)
        else None,
        "rsi_14": analysis.get("rsi_14"),
        "revisions": (analysis.get("factor_breakdown") or {}).get("estimate_revisions"),
        "insider": analysis.get("insider_cluster_buy") or (analysis.get("factors_raw") or {}).get("insider_buying"),
        "hint": hint,
    }


def _legacy_decision(analysis: dict[str, Any], config: dict[str, Any] | None) -> Decision:
    from core.scoring import _evaluate_good_buy

    thresholds = get_thresholds(config)
    analyst = analysis.get("analyst") or {}
    bump = (analysis.get("uncertainty") or {}).get("threshold_bump") or 0.0
    ok = _evaluate_good_buy(
        analysis.get("composite"),
        analyst.get("implied_upside_pct"),
        analyst,
        thresholds,
        bargain_score=(analysis.get("bargain") or {}).get("score"),
        factor_coverage_pct=analysis.get("factor_coverage_pct"),
        altman_z=analysis.get("altman_z"),
        sector=analysis.get("sector"),
        uncertainty_bump=bump,
        altman_z_pp=analysis.get("altman_z_pp"),
    )
    composite_min = float(thresholds.get("composite_min", 50)) + float(bump)
    bargain_min = float(thresholds.get("bargain_min", 50)) + float(bump)
    gates = [
        GateResult("composite", analysis.get("composite") is not None and float(analysis.get("composite")) >= composite_min, analysis.get("composite"), composite_min),
        GateResult("bargain", (analysis.get("bargain") or {}).get("score") is not None and float((analysis.get("bargain") or {}).get("score")) >= bargain_min, (analysis.get("bargain") or {}).get("score"), bargain_min),
        GateResult("coverage", analysis.get("factor_coverage_pct") is None or float(analysis.get("factor_coverage_pct")) >= float(thresholds.get("coverage_min_pct", 70)), analysis.get("factor_coverage_pct"), thresholds.get("coverage_min_pct", 70)),
        GateResult("distress", not bool(analysis.get("distress_flag")), analysis.get("altman_z_pp") or analysis.get("altman_z"), thresholds.get("altman_zpp_min", 1.1)),
        GateResult("sell", analyst.get("consensus_label") not in {"Sell", "Underperform"} if thresholds.get("exclude_sell_consensus", True) else True, analyst.get("consensus_label"), "not Sell/Underperform"),
    ]
    return Decision(
        label="Accumulate" if ok else "Avoid",
        buy_below_price=None,
        pct_to_buy=None,
        gates=gates,
        flags=analysis.get("value_trap_flags") or [],
        timing_context=_timing_context(analysis),
        mode="legacy",
    )


def decide(analysis: dict[str, Any], config: dict[str, Any] | None = None) -> Decision:
    """Layer B + Layer C decision. Legacy mode wraps the old composite/bargain gate."""
    cfg = get_decision_config(config)
    if cfg.get("mode") == "legacy":
        return _legacy_decision(analysis, config)

    flags = analysis.get("value_trap_flags") or []
    triggered = [f for f in flags if f.get("triggered")]
    quality_pct = analysis.get("quality_percentile")
    if quality_pct is None:
        quality_pct = analysis.get("quality_score")
    dq = analysis.get("data_quality") or {}
    grade = str(dq.get("grade") or "")
    valuation = analysis.get("valuation") or {}
    relative_only = bool(valuation.get("relative_only"))
    dcf = valuation.get("dcf") or {}
    base = (dcf.get("base") or {}) if isinstance(dcf, dict) else {}
    bear = (dcf.get("bear") or {}) if isinstance(dcf, dict) else {}
    dcf_base = base.get("per_share")
    dcf_bear = bear.get("per_share")
    uncertainty = (analysis.get("uncertainty") or {}).get("label") or "Medium"
    mos = _mos_for(uncertainty, cfg)
    price = analysis.get("price")

    buy_below = None
    if dcf_base is not None and dcf_bear is not None:
        buy_below = min(float(dcf_base) * (1.0 - mos), float(dcf_bear))
    elif dcf_base is not None:
        buy_below = float(dcf_base) * (1.0 - mos)

    pct_to_buy = None
    if buy_below is not None and price and float(price) > 0:
        pct_to_buy = (float(buy_below) / float(price) - 1.0) * 100.0

    exp_ret = valuation.get("expected_return")
    hurdle = ((valuation.get("hurdle") or {}).get("rate"))
    exp_gap = None
    if exp_ret is not None and hurdle is not None:
        exp_gap = float(exp_ret) - float(hurdle)

    max_flags = int(get_value_trap_config(config).get("max_flags", 1))
    min_q = float(cfg.get("min_quality_percentile", 40))
    min_gap = float(cfg.get("min_expected_return_over_hurdle", 0.0))
    block_c = bool(cfg.get("block_on_data_quality_c", True))

    from core.scoring import is_distressed

    distressed = bool(analysis.get("distress_flag")) or is_distressed(
        analysis.get("altman_z"),
        get_thresholds(config),
        analysis.get("sector"),
        altman_z_pp=analysis.get("altman_z_pp"),
    )

    gates = [
        GateResult(
            "data_quality",
            not (block_c and grade.upper() == "C"),
            grade or None,
            "A or B" if block_c else "any",
            "Grade C blocks Accumulate",
        ),
        GateResult(
            "distress",
            not distressed,
            analysis.get("altman_z_pp") or analysis.get("altman_z"),
            get_thresholds(config).get("altman_zpp_min", 1.1),
        ),
        GateResult(
            "value_trap_count",
            len(triggered) <= max_flags,
            len(triggered),
            max_flags,
            ", ".join(f.get("code", "") for f in triggered) or "ok",
        ),
        GateResult(
            "quality_percentile",
            quality_pct is None or float(quality_pct) >= min_q,
            quality_pct,
            min_q,
        ),
    ]

    if relative_only:
        pb = valuation.get("price_to_book")
        rel = valuation.get("relative") or {}
        pb_pct = (rel.get("p_oe") or {}).get("percentile") or (rel.get("ev_ebit") or {}).get("percentile")
        cheap = pb_pct is not None and float(pb_pct) >= 50
        gates.append(GateResult("relative_cheap", cheap, pb_pct or pb, 50, "Financials: relative-only"))
        layer_b = all(g.passed for g in gates if g.name != "relative_cheap")
        if not layer_b:
            label = "Avoid"
        elif cheap:
            label = "Accumulate"
        else:
            label = "Watch"
        return Decision(label, None, None, gates, flags, _timing_context(analysis), "intrinsic")

    price_ok = buy_below is not None and price is not None and float(price) <= float(buy_below)
    gates.append(GateResult("price_vs_buy_below", price_ok, price, buy_below))
    exp_ok = exp_gap is None or exp_gap >= min_gap
    if exp_ret is None or hurdle is None:
        exp_ok = False
        gates.append(GateResult("expected_return_vs_hurdle", False, exp_ret, hurdle, "missing expected return or hurdle"))
    else:
        gates.append(GateResult("expected_return_vs_hurdle", exp_ok, exp_ret, hurdle, f"gap={exp_gap:.3f}" if exp_gap is not None else ""))

    layer_b = all(
        g.passed
        for g in gates
        if g.name in {"data_quality", "distress", "value_trap_count", "quality_percentile"}
    )
    if not layer_b:
        label = "Avoid"
    elif price_ok and exp_ok:
        label = "Accumulate"
    else:
        label = "Watch"

    return Decision(
        label=label,
        buy_below_price=buy_below,
        pct_to_buy=pct_to_buy,
        gates=gates,
        flags=flags,
        timing_context=_timing_context(analysis),
        mode="intrinsic",
    )
