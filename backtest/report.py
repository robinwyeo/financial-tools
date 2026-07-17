"""Generate backtest results report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from backtest.constants import RESULTS_DIR


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_money(val: float | None) -> str:
    if val is None:
        return "—"
    sign = "+" if val >= 0 else "−"
    return f"{sign}${abs(val):,.0f}"


def _fmt_pct(val: float | None, *, signed: bool = False) -> str:
    if val is None:
        return "—"
    if signed and val > 0:
        return f"+{val:.1%}"
    if signed and val < 0:
        return f"−{abs(val):.1%}"
    return f"{val:.1%}"


def _beat_label(beat: bool | None) -> str:
    if beat is None:
        return "—"
    return "Yes" if beat else "No"


def _dca_vs_spy_section(dca: dict) -> list[str]:
    """Render investment comparison vs S&P 500 benchmark."""
    sim = dca.get("simulation", {})
    bh = dca.get("buy_and_hold", {})
    old = bh.get("old", {})
    new = bh.get("new", {})
    spy = bh.get("spy", {})
    cmp_ = dca.get("comparison_vs_spy", {})

    invested = spy.get("total_invested") or old.get("total_invested") or 0
    period_start = sim.get("period_start") or spy.get("period_start") or "—"
    period_end = sim.get("period_end") or spy.get("period_end") or "—"
    per_q = sim.get("investment_per_quarter_usd", 20_000)
    benchmark = sim.get("benchmark_ticker", "SPY")
    quarters = sim.get("quarters_with_investment") or spy.get("quarters_invested") or 0

    lines = [
        "## Investment Comparison vs S&P 500",
        "",
        f"Each quarter, **${per_q:,.0f}** was invested into the top {sim.get('top_n_stocks', 5)} "
        f"good-buy stocks (old or new parameters). The benchmark invests the same "
        f"**${per_q:,.0f}/quarter** into **{benchmark}** (S&P 500 total return) on the "
        f"identical schedule.",
        "",
        f"- Simulation period: **{period_start}** → **{period_end}**",
        f"- Quarters with investment: **{quarters}**",
        f"- Total capital deployed (each strategy): **${invested:,.0f}**",
        "",
        "### Side-by-side results",
        "",
        "| Metric | Old params | New params | S&P 500 (SPY) |",
        "| --- | ---: | ---: | ---: |",
        f"| Terminal wealth | ${old.get('terminal_wealth', 0):,.0f} | "
        f"${new.get('terminal_wealth', 0):,.0f} | ${spy.get('terminal_wealth', 0):,.0f} |",
        f"| Total invested | ${old.get('total_invested', 0):,.0f} | "
        f"${new.get('total_invested', 0):,.0f} | ${spy.get('total_invested', 0):,.0f} |",
        f"| Total return | {_fmt_pct(old.get('total_return'))} | "
        f"{_fmt_pct(new.get('total_return'))} | {_fmt_pct(spy.get('total_return'))} |",
        f"| CAGR | {_fmt_pct(old.get('cagr'))} | "
        f"{_fmt_pct(new.get('cagr'))} | {_fmt_pct(spy.get('cagr'))} |",
        f"| Max drawdown | {_fmt_pct(old.get('max_drawdown'))} | "
        f"{_fmt_pct(new.get('max_drawdown'))} | {_fmt_pct(spy.get('max_drawdown'))} |",
        "",
        "### Outperformance vs S&P 500",
        "",
        "| Comparison | Terminal wealth Δ | Return Δ | CAGR Δ | Higher ROI? |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]

    for key, label in [
        ("old_vs_spy", "Old params vs SPY"),
        ("new_vs_spy", "New params vs SPY"),
        ("new_vs_old", "New params vs Old params"),
    ]:
        row = cmp_.get(key, {})
        beat_col = _beat_label(row.get("beat_benchmark"))
        wealth_pct = row.get("terminal_wealth_delta_pct")
        wealth_str = _fmt_money(row.get("terminal_wealth_delta"))
        if wealth_pct is not None:
            wealth_str += f" ({_fmt_pct(wealth_pct, signed=True)})"
        lines.append(
            f"| {label} | {wealth_str} | "
            f"{_fmt_pct(row.get('total_return_delta'), signed=True)} | "
            f"{_fmt_pct(row.get('cagr_delta'), signed=True)} | {beat_col} |"
        )

    unequal_capital = (
        old.get("total_invested") != spy.get("total_invested")
        or new.get("total_invested") != spy.get("total_invested")
    )
    if unequal_capital:
        lines.extend(
            [
                "",
                "> **Note:** Stock strategies only deploy capital in quarters where good-buy "
                "screens pass, while SPY receives $20k every quarter. Terminal wealth "
                "is therefore not directly comparable — use **total return** and **CAGR** "
                "(return on capital actually deployed) as the primary efficiency metrics.",
            ]
        )

    lines.extend(["", "### Interpretation", ""])

    old_vs = cmp_.get("old_vs_spy", {})
    new_vs = cmp_.get("new_vs_spy", {})
    if spy.get("terminal_wealth", 0) <= 0:
        lines.append(
            "- SPY benchmark data was unavailable for this run; re-run "
            "`python -m backtest.run dca` after price ingest includes SPY."
        )
    else:
        for name, strat, vs in [
            ("Old parameters", old, old_vs),
            ("New parameters", new, new_vs),
        ]:
            wealth_delta = vs.get("terminal_wealth_delta") or 0
            if vs.get("beat_on_wealth"):
                wealth_clause = (
                    f"finished {_fmt_money(wealth_delta)} ahead of SPY in terminal wealth"
                )
            else:
                wealth_clause = (
                    f"finished {_fmt_money(wealth_delta)} behind SPY in terminal wealth"
                )
            roi_clause = (
                f"return on deployed capital was {_fmt_pct(strat.get('total_return'))} "
                f"({_fmt_pct(strat.get('cagr'))} CAGR) vs SPY's "
                f"{_fmt_pct(spy.get('total_return'))} ({_fmt_pct(spy.get('cagr'))} CAGR)"
            )
            lines.append(f"- **{name}** {wealth_clause}; {roi_clause}.")

    new_vs_old = cmp_.get("new_vs_old", {})
    delta = new_vs_old.get("terminal_wealth_delta") or 0
    if delta > 0:
        lines.append(
            f"- **New parameters** beat old parameters by {_fmt_money(delta)} terminal "
            f"wealth and {_fmt_pct(new_vs_old.get('total_return_delta'), signed=True)} "
            f"higher return on deployed capital."
        )
    elif delta < 0:
        lines.append(
            f"- **Old parameters** beat new parameters by {_fmt_money(-delta)} terminal "
            f"wealth and {_fmt_pct(-(new_vs_old.get('total_return_delta') or 0), signed=True)} "
            f"higher return on deployed capital."
        )
    else:
        lines.append("- **Old and new parameters** finished with equal terminal wealth.")

    lines.append("")
    return lines


def generate_report(output_path: Path | None = None) -> str:
    """Build markdown report from saved result artifacts."""
    comparison = _load_json(RESULTS_DIR / "weight_candidate_comparison.json")
    tuning = _load_json(RESULTS_DIR / "tuning_results.json")
    bargain = _load_json(RESULTS_DIR / "bargain_tuning_results.json")
    thresholds = _load_json(RESULTS_DIR / "threshold_calibration.json")
    dca = _load_json(RESULTS_DIR / "dca_validation.json")

    lines = [
        "# Score Weight & Threshold Backtest Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Summary",
        "",
        "This report validates composite/bargain score weights and good-buy thresholds",
        "using long-horizon (1y/3y/5y) forward returns and gated DCA buy-and-hold",
        "simulations on historical S&P 500 constituents.",
        "",
        "## Caveats",
        "",
        "- Residual survivorship bias: delisted tickers may lack price history in free data.",
        "- `earnings_revisions` is live-only and excluded from historical validation.",
        "- Analyst upside is informational (not a hard gate) and not historically backtested.",
        "- Expanding-window folds + bootstrap CIs reduce but do not eliminate path dependence.",
        "- SEC EDGAR fundamentals are point-in-time by filing date; reporting lags apply.",
        "",
    ]

    if comparison:
        lines.extend(
            [
                "## Named Weight Candidate Comparison",
                "",
                f"Recommended: **{comparison.get('recommended', 'n/a')}**",
                f"Primary horizon: **{comparison.get('primary_horizon', '3y')}**",
                "",
                "| Candidate | 3y IC | Excess mean | 95% CI | % folds > 0 |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison.get("candidates", []):
            ics = row.get("horizon_ics") or {}
            ci = f"[{row.get('excess_ci_low', float('nan')):.2%}, {row.get('excess_ci_high', float('nan')):.2%}]"
            lines.append(
                f"| {row.get('name')} | {ics.get('3y', 0):.3f} | "
                f"{row.get('excess_mean', 0):.2%} | {ci} | "
                f"{row.get('frac_folds_positive', 0):.0%} |"
            )
        lines.append("")
        for cmp_row in comparison.get("comparisons", []):
            if cmp_row.get("indistinguishable"):
                lines.append(
                    f"- **{cmp_row.get('winner')}** vs **{cmp_row.get('challenger')}**: "
                    "indistinguishable (bootstrap CIs overlap)."
                )
        lines.extend(
            [
                "",
                "### Recommended factor weights",
                "",
                "```yaml",
            ]
        )
        for k, v in sorted((comparison.get("recommended_weights") or {}).items()):
            lines.append(f"  {k}: {v:.4f}")
        lines.extend(["```", ""])
    elif tuning:
        winner = tuning.get("winner", {})
        lines.extend(
            [
                "## Factor Weight Tuning (legacy search)",
                "",
                f"Winner: **{winner.get('name', 'n/a')}**",
                "",
                "```yaml",
            ]
        )
        for k, v in sorted((winner.get("factor_weights") or {}).items()):
            lines.append(f"  {k}: {v:.4f}")
        lines.extend(["```", ""])

    if bargain:
        lines.extend(
            [
                "## Bargain Weight Validation",
                "",
                f"- Winner: **{bargain.get('winner_name', 'n/a')}**",
                f"- Winner mean IC ({bargain.get('horizon', '3y')}): {bargain.get('winner_mean_ic', 0):.3f}",
                f"- Baseline mean IC: {bargain.get('baseline_mean_ic', 0):.3f}",
                "",
                "### Recommended bargain weights",
                "",
                "```yaml",
            ]
        )
        for k, v in sorted((bargain.get("winner_weights") or {}).items()):
            lines.append(f"  {k}: {v:.4f}")
        lines.extend(["```", ""])

    if thresholds:
        lines.extend(
            [
                "## Threshold Calibration",
                "",
                f"- Horizon: **{thresholds.get('horizon', '3y')}**",
                f"- composite_min: **{thresholds.get('composite_min', 50):.1f}**",
                f"- bargain_min: **{thresholds.get('bargain_min', 50):.1f}**",
                "",
            ]
        )

    if dca:
        lines.extend(_dca_vs_spy_section(dca))
        lines.extend(
            [
                "### Survivorship sensitivity (terminal wealth vs SPY)",
                "",
                "| Delist assumption | Old | New | SPY | Old − SPY | New − SPY |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in dca.get("survivorship_sensitivity", []):
            old_delta = row.get("old_vs_spy_wealth_delta")
            new_delta = row.get("new_vs_spy_wealth_delta")
            lines.append(
                f"| {row.get('delist_return', 0):.0%} | "
                f"${row.get('old_terminal_wealth', 0):,.0f} | "
                f"${row.get('new_terminal_wealth', 0):,.0f} | "
                f"${row.get('spy_terminal_wealth', 0):,.0f} | "
                f"{_fmt_money(old_delta)} | {_fmt_money(new_delta)} |"
            )
        lines.append("")

    text = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = output_path or (RESULTS_DIR / "backtest_report.md")
    out.write_text(text, encoding="utf-8")
    return text
