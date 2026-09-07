"""Pure formatting helpers for the Streamlit UI (no st.* at import)."""

from __future__ import annotations

import html
import math
from typing import Any

from core.data import currency_symbol
from ui.constants import FACTOR_HELP, GAUGE_MAX_WIDTH, SHORT_FACTOR_LABELS

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return f"{n}th"
    return f"{n}{['th', 'st', 'nd', 'rd'][min(n % 10, 3)]}"


def fmt_large_number(n: float | None, currency: str | None = None) -> str:
    sym = currency_symbol(currency)
    if n is None:
        return "N/A"
    if n >= 1e12:
        return f"{sym}{n / 1e12:.2f}T"
    if n >= 1e9:
        return f"{sym}{n / 1e9:.1f}B"
    if n >= 1e6:
        return f"{sym}{n / 1e6:.1f}M"
    return f"{sym}{n:,.0f}"


def percentile_color(pct: float | None) -> str:
    """Red (0–30), yellow (31–70), green (71–100) by percentile rank."""
    if pct is None or (isinstance(pct, float) and math.isnan(pct)):
        return "#d1d5db"
    p = max(0.0, min(100.0, float(pct)))
    if p <= 30:
        return "#ef4444"  # red
    if p <= 70:
        return "#eab308"  # yellow
    return "#22c55e"


def gauge_score_color(score: float | None) -> str:
    """Red ≤25, yellow 25–50, green ≥50 (aligned with scorecard traffic lights)."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "#d1d5db"
    s = max(0.0, min(100.0, float(score)))
    if s >= 50:
        return "#22c55e"
    if s <= 25:
        return "#ef4444"
    return "#eab308"


def gauge_score_label(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "N/A"
    s = float(score)
    if s >= 50:
        return "Good"
    if s <= 25:
        return "Weak"
    return "Fair"


def consensus_style(label: str) -> tuple[str, str]:
    """Return (text_color, bg_color) for a consensus label."""
    lower = label.lower()
    if "buy" in lower:
        return "#0d9488", "#ccfbf1"
    if "hold" in lower or "neutral" in lower:
        return "#92400e", "#fef3c7"
    if "sell" in lower or "underperform" in lower:
        return "#dc2626", "#fee2e2"
    return "#6b7280", "#f3f4f6"


def _proximity_color(pct_below: float | None) -> str:
    """Green = discounted, yellow = moderate, red = near highs."""
    if pct_below is None:
        return "#d1d5db"
    if pct_below <= 0.10:
        return "#ef4444"
    if pct_below <= 0.25:
        return "#eab308"
    return "#22c55e"


def _pct_below_high(price: float | None, high: float | None) -> float | None:
    if price is None or high is None or high <= 0 or price <= 0:
        return None
    return max(0.0, 1.0 - (price / high))


def _arc_gauge_html(
    score: float | None,
    label: str,
    label_color: str,
    *,
    subtitle: str = "vs. Global Universe",
    aria_label: str = "Score gauge",
    fill_color: str | None = None,
    max_width: str = "130px",
) -> str:
    """
    Pure SVG 3/4-circle arc gauge.
    Starts at the 7:30 clock position, sweeps 270° clockwise to 4:30.
    Uses width="100%" so it scales from a narrow sidebar column to full mobile width.
    """
    cx, cy, r, sw = 80, 80, 60, 12
    start_deg = 135.0   # 7:30 clock position in SVG angle space
    span_deg = 270.0

    def pt(deg: float) -> tuple[float, float]:
        rad = math.radians(deg)
        return cx + r * math.cos(rad), cy + r * math.sin(rad)

    # Background arc (full 270°)
    s = pt(start_deg)
    e = pt(start_deg + span_deg)
    bg_path = f"M {s[0]:.1f} {s[1]:.1f} A {r} {r} 0 1 1 {e[0]:.1f} {e[1]:.1f}"

    # Fill arc proportional to score
    fill_svg = ""
    stroke = fill_color or "#14b8a6"
    if score is not None and score > 0:
        fd = score * span_deg / 100.0
        fe = pt(start_deg + fd)
        large = 1 if fd > 180 else 0
        fill_svg = (
            f'<path d="M {s[0]:.1f} {s[1]:.1f} A {r} {r} 0 {large} 1 {fe[0]:.1f} {fe[1]:.1f}" '
            f'fill="none" stroke="{stroke}" stroke-width="{sw}" stroke-linecap="round"/>'
        )

    score_txt = f"{score:.1f}" if score is not None else "N/A"
    compact = max_width != "130px"
    font_sz = 34 if score is not None and compact else (36 if score is not None else 22)

    # viewBox clips the empty gap at the bottom (arc endpoints sit at y≈122, cut at y=135)
    return (
        '<div style="text-align:center;padding:0.2rem 0.15rem 0.1rem;">'
        '<svg width="100%" viewBox="5 5 150 130" '
        f'style="max-width:{max_width};display:block;margin:0 auto;" '
        f'aria-label="{aria_label}">'
        f'<path d="{bg_path}" fill="none" stroke="#e8ecef" '
        f'stroke-width="{sw}" stroke-linecap="round"/>'
        f'{fill_svg}'
        f'<text x="{cx}" y="78" text-anchor="middle" dominant-baseline="middle" '
        f'font-size="{font_sz}" font-weight="800" fill="#1e3a5f" '
        f'font-family="Inter, Arial, sans-serif">{score_txt}</text>'
        f'<text x="{cx}" y="100" text-anchor="middle" font-size="13" fill="#9ca3af" '
        f'font-family="Inter, Arial, sans-serif">/ 100</text>'
        '</svg>'
        f'<div style="margin-top:0.1rem;">'
        f'<div style="font-size:0.88rem;font-weight:700;color:{label_color};">{label}</div>'
        f'<div style="font-size:0.6rem;color:#9ca3af;margin-top:1px;line-height:1.25;">{subtitle}</div>'
        '</div>'
        '</div>'
    )


def _format_snapshot_date(raw_date: str | None) -> str | None:
    if not raw_date:
        return None
    return raw_date[:10] if len(raw_date) >= 10 else raw_date


def _sparkline_svg(values: list[float], *, width: int = 128, height: int = 28) -> str:
    if len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    span = hi - lo if hi != lo else 1.0
    pts: list[str] = []
    for i, val in enumerate(values):
        x = 1 + i / (len(values) - 1) * (width - 2)
        y = height - 2 - ((val - lo) / span) * (height - 4)
        pts.append(f"{x:.1f},{y:.1f}")
    color = "#16a34a" if values[-1] >= values[0] else "#dc2626"
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'aria-label="FY1 EPS estimate trend">'
        f'<polyline fill="none" stroke="{color}" stroke-width="1.8" '
        f'points="{" ".join(pts)}"/></svg>'
    )


def _signal_badge(label: str, tone: str) -> str:
    colors = {
        "success": ("#166534", "#dcfce7"),
        "warning": ("#92400e", "#fef3c7"),
        "danger": ("#991b1b", "#fee2e2"),
        "info": ("#1e3a5f", "#e0e7ff"),
        "neutral": ("#374151", "#f3f4f6"),
    }
    fg, bg = colors.get(tone, colors["neutral"])
    return (
        f'<span style="display:inline-block;margin:0 6px 4px 0;padding:2px 8px;'
        f'border-radius:999px;font-size:0.72rem;font-weight:700;'
        f'color:{fg};background:{bg};">{html.escape(label)}</span>'
    )


def _overlay_badge_spans(analysis: dict) -> list[str]:
    """Pills for uncertainty / insider / short interest / valuation history.

    Always include Uncertainty (default Low). Streamlit strips nested <div>s, so
    callers must place these spans in a single top-level markup block.
    """
    uncertainty = analysis.get("uncertainty") or {}
    unc_label = uncertainty.get("label") or "Low"
    unc_tone = {"Low": "success", "Medium": "warning", "High": "danger"}.get(unc_label, "neutral")
    bump = uncertainty.get("threshold_bump") or 0
    bump_txt = f" · +{bump:.0f} buy hurdle" if bump else ""
    badges = [_signal_badge(f"Uncertainty: {unc_label}{bump_txt}", unc_tone)]

    insider = analysis.get("insider") or {}
    if insider.get("cluster_buy"):
        n_buyers = insider.get("buyers_90d") or 0
        badges.append(_signal_badge(f"Insider cluster buy ({n_buyers} officers, 90d)", "success"))
    short = analysis.get("short_interest") or {}
    if short.get("high_short_interest"):
        days = short.get("short_ratio")
        days_txt = f"{days:.1f}d to cover" if days is not None else "elevated"
        badges.append(_signal_badge(f"High short interest · {days_txt}", "warning"))
    hist = analysis.get("valuation_history") or {}
    if hist.get("source") and hist.get("n_points"):
        metric = hist.get("metric") or "yield"
        badges.append(
            _signal_badge(
                f"History: {hist.get('n_points')} pts · {metric} · {hist.get('source')}",
                "info",
            )
        )
    return badges


def _factor_label_html(
    factor_key: str,
    short_label: str,
    help_texts: dict[str, str] | None = None,
) -> str:
    """Metric name with hover tooltip (meaning + calculation)."""
    label_e = html.escape(short_label)
    help_text = (help_texts if help_texts is not None else FACTOR_HELP).get(factor_key)
    if not help_text:
        return f'<div class="factor-label">{label_e}</div>'

    return (
        f'<div class="factor-label factor-has-tip" tabindex="0">'
        f'<span class="factor-label-text">{label_e}</span>'
        f'<span class="factor-tooltip" role="tooltip">{html.escape(help_text)}</span>'
        f"</div>"
    )


def _factor_group_html(
    group_label: str,
    accent: str,
    factor_keys: list[str],
    breakdown: dict,
    labels: dict[str, str] | None = None,
    help_texts: dict[str, str] | None = None,
) -> str:
    header = (
        f'<div style="font-size:0.58rem;font-weight:700;color:{accent};text-transform:uppercase;'
        f'letter-spacing:0.07em;margin:0 0 4px;padding-bottom:2px;'
        f'border-bottom:1px solid {accent}22;'
        f'overflow:hidden;white-space:nowrap;text-overflow:ellipsis;">{group_label}</div>'
    )
    label_map = labels if labels is not None else SHORT_FACTOR_LABELS
    rows = []
    for key in factor_keys:
        short_label = label_map.get(key, key)
        fb = breakdown.get(key, {})
        pct = fb.get("percentile")
        color = percentile_color(pct)

        if pct is None or (isinstance(pct, float) and math.isnan(pct)):
            bar_w, pct_text = 0, "N/A"
        else:
            bar_w = min(max(float(pct), 0), 100)
            pct_text = ordinal(int(round(float(pct))))

        rows.append(
            f'<div class="factor-row">'
            f'<div class="factor-dot" style="background:{color};"></div>'
            f"{_factor_label_html(key, short_label, help_texts)}"
            f'<div class="factor-bar-track">'
            f'<div class="factor-bar-fill" style="width:{bar_w:.0f}%;background:{color};"></div>'
            f"</div>"
            f'<div class="factor-pct" style="color:{color};">{pct_text}</div>'
            f"</div>"
        )
    return f'<div class="factor-scorecard-group">{header}{"".join(rows)}</div>'


def _price_position_strip_html(analysis: dict) -> str:
    """Colored pills: % below 52W high, % below ATH, optional 52wk range bar."""
    price = analysis.get("price")
    high_52 = analysis.get("fifty_two_week_high")
    low_52 = analysis.get("fifty_two_week_low")
    ath = analysis.get("all_time_high")

    def pill(label: str, pct_below: float | None) -> str:
        color = _proximity_color(pct_below)
        if pct_below is None:
            txt = "N/A"
        else:
            txt = f"{pct_below * 100:.1f}% below"
        return (
            f'<div style="flex:1;min-width:0;background:{color}18;border:1px solid {color}55;'
            f'border-radius:8px;padding:0.35rem 0.5rem;text-align:center;">'
            f'<div style="font-size:0.55rem;font-weight:600;color:#6b7280;text-transform:uppercase;'
            f'letter-spacing:0.04em;">{label}</div>'
            f'<div style="font-size:0.82rem;font-weight:700;color:{color};">{txt}</div>'
            "</div>"
        )

    range_bar = ""
    if (
        price is not None
        and low_52 is not None
        and high_52 is not None
        and high_52 > low_52
    ):
        pos = max(0.0, min(1.0, (price - low_52) / (high_52 - low_52)))
        range_bar = (
            '<div style="margin-top:0.45rem;padding-bottom:0.15rem;">'
            '<div style="font-size:0.55rem;color:#9ca3af;margin-bottom:4px;">52W range</div>'
            '<div style="position:relative;height:7px;background:linear-gradient(90deg,#22c55e,#eab308,#ef4444);'
            'border-radius:3px;">'
            f'<div style="position:absolute;left:{pos * 100:.1f}%;top:50%;transform:translate(-50%,-50%);'
            'width:10px;height:10px;background:#1e3a5f;border:2px solid #fff;border-radius:50%;'
            'box-shadow:0 0 0 1px #94a3b8;"></div></div></div>'
        )

    return (
        '<div class="price-position-strip">'
        '<div style="display:flex;gap:0.4rem;">'
        + pill("vs 52W High", _pct_below_high(price, high_52))
        + pill("vs ATH", _pct_below_high(price, ath))
        + "</div>"
        + range_bar
        + "</div>"
    )


def _analyst_target_range_html(analyst: dict, currency: str | None = None) -> str:
    pills: list[str] = []
    sym = currency_symbol(currency)
    for lbl, key in [("Low", "target_low"), ("Mean", "target_mean"), ("High", "target_high")]:
        val = analyst.get(key)
        if val is not None:
            pills.append(
                f'<div class="analyst-target-pill">'
                f'<div class="lbl">{lbl}</div>'
                f'<div class="val">{sym}{val:,.0f}</div>'
                f"</div>"
            )
    if not pills:
        return ""
    return f'<div class="analyst-targets">{"".join(pills)}</div>'

def decision_card_html(decision: dict[str, Any], *, currency: str | None = None) -> str:
    """HTML for the Accumulate/Watch/Avoid card (testable without Streamlit)."""
    label = str(decision.get("label") or "Avoid")
    colors = {
        "Accumulate": ("#166534", "#dcfce7"),
        "Watch": ("#92400e", "#fef3c7"),
        "Avoid": ("#991b1b", "#fee2e2"),
    }
    fg, bg = colors.get(label, ("#6b7280", "#f3f4f6"))
    sym = currency_symbol(currency)
    buy_below = decision.get("buy_below_price")
    pct = decision.get("pct_to_buy")
    buy_txt = f"{sym}{buy_below:,.2f}" if isinstance(buy_below, (int, float)) else "n/a"
    pct_txt = f"{pct:+.1f}%" if isinstance(pct, (int, float)) else ""
    gates = decision.get("gates") or []
    gate_rows = []
    for g in gates:
        ok = bool(g.get("passed"))
        mark = "PASS" if ok else "FAIL"
        color = "#166534" if ok else "#991b1b"
        gate_rows.append(
            f"<tr><td>{g.get('name')}</td>"
            f"<td style='color:{color};font-weight:700;'>{mark}</td>"
            f"<td>{g.get('actual')}</td>"
            f"<td>{g.get('threshold')}</td>"
            f"<td>{g.get('note') or ''}</td></tr>"
        )
    hint = (decision.get("timing_context") or {}).get("hint") or ""
    watch_line = ""
    if label == "Watch" and isinstance(buy_below, (int, float)):
        watch_line = f"<p>Accumulate below <b>{buy_txt}</b> ({pct_txt})</p>"
    return (
        f"<div style='background:{bg};padding:12px;border-radius:10px;'>"
        f"<div style='font-size:1.4rem;font-weight:800;color:{fg};'>{label}</div>"
        f"<p>Buy below <b>{buy_txt}</b> {pct_txt}</p>"
        f"{watch_line}"
        f"<table style='width:100%;font-size:0.85rem;'>"
        f"<tr><th align='left'>Gate</th><th>Result</th><th>Actual</th><th>Threshold</th><th>Note</th></tr>"
        f"{''.join(gate_rows)}</table>"
        f"<p style='color:#6b7280;font-size:0.85rem;'>{hint}</p>"
        f"</div>"
    )


def valuation_card_html(valuation: dict[str, Any], *, currency: str | None = None) -> str:
    """HTML for the IV range / reverse DCF / expected return card."""
    sym = currency_symbol(currency)
    if valuation.get("relative_only"):
        return (
            "<div><b>Financials: relative-only valuation</b>"
            f"<p>P/B {valuation.get('price_to_book')} · P/E {valuation.get('price_to_earnings')}</p></div>"
        )
    dcf = valuation.get("dcf") or {}
    base = (dcf.get("base") or {}).get("per_share")
    bear = (dcf.get("bear") or {}).get("per_share")
    epv = (valuation.get("epv") or {}).get("per_share")
    rev = (valuation.get("reverse_dcf") or {}).get("implied_g1")
    g_hist = (valuation.get("growth") or {}).get("g_hist")
    exp = valuation.get("expected_return")
    hurdle_info = valuation.get("hurdle") or {}
    hurdle = hurdle_info.get("rate")
    rf = hurdle_info.get("rf")
    erp = hurdle_info.get("erp")
    rf_source = hurdle_info.get("rf_source")
    rf_as_of = hurdle_info.get("rf_as_of")
    if rf_source and rf_as_of:
        rf_origin = f" {html.escape(str(rf_source))} as of {html.escape(str(rf_as_of))}"
    elif rf_source:
        rf_origin = f" {html.escape(str(rf_source))}"
    elif rf_as_of:
        rf_origin = f" as of {html.escape(str(rf_as_of))}"
    else:
        rf_origin = ""

    def _p(v: Any) -> str:
        return f"{sym}{v:,.2f}" if isinstance(v, (int, float)) else "n/a"

    def _pct(v: Any) -> str:
        return f"{100 * v:.1f}%" if isinstance(v, (int, float)) else "n/a"

    return (
        "<div>"
        f"<p>IV range · base {_p(base)} · bear {_p(bear)} · EPV {_p(epv)}</p>"
        f"<p>Reverse DCF implies {_pct(rev)} 5y growth vs history {_pct(g_hist)}</p>"
        f"<p>Expected return {_pct(exp)} vs hurdle {_pct(hurdle)}"
        f" (10y Treasury {_pct(rf)}, ERP {_pct(erp)}{rf_origin})</p>"
        "</div>"
    )
