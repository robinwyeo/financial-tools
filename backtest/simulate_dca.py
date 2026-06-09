"""$20k/quarter DCA simulation for final validation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pandas as pd

from backtest.constants import (
    BACKTEST_END,
    BACKTEST_START,
    DCA_INVESTMENT_USD,
    DCA_TOP_N,
    DEFAULT_DELIST_RETURN,
    RESULTS_DIR,
)
from backtest.data.prices import (
    BENCHMARK_TICKER,
    ensure_benchmark_prices,
    load_delisted_catalog,
    load_prices,
    price_on_or_before,
)
from backtest.engine import score_factor_panel
from backtest.factors import load_factor_panel

logger = logging.getLogger(__name__)


@dataclass
class DCAResult:
    strategy: str
    terminal_wealth: float
    total_invested: float
    total_return: float
    cagr: float
    max_drawdown: float
    period_start: str | None = None
    period_end: str | None = None
    quarters_invested: int = 0
    holdings_log: list[dict[str, Any]] = field(default_factory=list)


def _select_top_good_buys(
    scored_q: pd.DataFrame,
    composite_min: float,
    bargain_min: float,
) -> list[str]:
    valid = scored_q.dropna(subset=["composite", "bargain_score"])
    valid = valid[
        (valid["composite"] >= composite_min) & (valid["bargain_score"] >= bargain_min)
    ].sort_values(["composite", "bargain_score"], ascending=False)
    return valid.head(DCA_TOP_N)["ticker"].astype(str).tolist()


def _years_between(start: date, end: date) -> float:
    return max((end - start).days / 365.25, 1e-6)


def _max_drawdown_from_wealth(wealth_history: list[tuple[date, float]]) -> float:
    if len(wealth_history) < 2:
        return 0.0
    series = pd.Series({pd.Timestamp(d): v for d, v in wealth_history}).sort_index()
    rets = series.pct_change().dropna()
    if rets.empty:
        return 0.0
    wealth = (1 + rets).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def _finalize_dca_result(
    strategy: str,
    wealth_history: list[tuple[date, float]],
    invested: float,
    holdings_log: list[dict[str, Any]],
) -> DCAResult:
    terminal = wealth_history[-1][1] if wealth_history else 0.0
    start = wealth_history[0][0] if wealth_history else BACKTEST_START
    end = wealth_history[-1][0] if wealth_history else BACKTEST_END
    years = _years_between(start, end)
    total_return = (terminal / invested - 1.0) if invested > 0 else 0.0
    cagr = (terminal / invested) ** (1 / years) - 1 if invested > 0 else 0.0
    return DCAResult(
        strategy=strategy,
        terminal_wealth=float(terminal),
        total_invested=float(invested),
        total_return=float(total_return),
        cagr=float(cagr),
        max_drawdown=_max_drawdown_from_wealth(wealth_history),
        period_start=str(start),
        period_end=str(end),
        quarters_invested=len(holdings_log),
        holdings_log=holdings_log,
    )


def _simulate_dca(
    scored: pd.DataFrame,
    prices: pd.DataFrame,
    composite_min: float,
    bargain_min: float,
    strategy: str = "buy_and_hold",
    delist_return: float = DEFAULT_DELIST_RETURN,
) -> DCAResult:
    delisted = load_delisted_catalog()
    cash = 0.0
    positions: dict[str, dict[str, float]] = {}
    invested = 0.0
    wealth_history: list[tuple[date, float]] = []
    log: list[dict[str, Any]] = []

    qends = sorted(scored["quarter_end"].unique())
    for qend in qends:
        qdate = pd.Timestamp(qend).date() if not isinstance(qend, date) else qend
        grp = scored[scored["quarter_end"] == qend]
        picks = _select_top_good_buys(grp, composite_min, bargain_min)

        if strategy == "rebalance" and positions:
            for ticker, pos in list(positions.items()):
                if ticker not in picks:
                    px = price_on_or_before(prices, ticker, qdate)
                    if px is None and ticker in delisted:
                        px = pos["cost_basis"] * (1 + delist_return)
                    if px:
                        cash += pos["shares"] * px
                    del positions[ticker]

        if picks:
            per_stock = DCA_INVESTMENT_USD / len(picks)
            invested += DCA_INVESTMENT_USD
            bought: list[str] = []
            for ticker in picks:
                px = price_on_or_before(prices, ticker, qdate)
                if px is None or px <= 0:
                    continue
                shares = per_stock / px
                if ticker in positions:
                    old = positions[ticker]
                    total_shares = old["shares"] + shares
                    avg_cost = (old["cost_basis"] * old["shares"] + px * shares) / total_shares
                    positions[ticker] = {"shares": total_shares, "cost_basis": avg_cost}
                else:
                    positions[ticker] = {"shares": shares, "cost_basis": px}
                bought.append(ticker)
            log.append({"quarter_end": str(qdate), "bought": bought, "invested": DCA_INVESTMENT_USD})

        port_value = cash
        for ticker, pos in positions.items():
            px = price_on_or_before(prices, ticker, qdate)
            if px is None:
                if ticker in delisted:
                    px = pos["cost_basis"] * (1 + delist_return)
                else:
                    continue
            port_value += pos["shares"] * px
        wealth_history.append((qdate, port_value))

    return _finalize_dca_result(strategy, wealth_history, invested, log)


def _spy_dca_benchmark(prices: pd.DataFrame, quarter_ends: list) -> DCAResult:
    """Invest $20k/quarter into SPY on the same schedule as the stock strategies."""
    prices = ensure_benchmark_prices(prices, BENCHMARK_TICKER)
    cash_shares = 0.0
    invested = 0.0
    wealth_history: list[tuple[date, float]] = []
    log: list[dict[str, Any]] = []

    for qend in sorted(quarter_ends):
        qdate = pd.Timestamp(qend).date() if not isinstance(qend, date) else qend
        px = price_on_or_before(prices, BENCHMARK_TICKER, qdate)
        if px and px > 0:
            cash_shares += DCA_INVESTMENT_USD / px
            invested += DCA_INVESTMENT_USD
            log.append(
                {
                    "quarter_end": str(qdate),
                    "bought": [BENCHMARK_TICKER],
                    "invested": DCA_INVESTMENT_USD,
                    "price": px,
                    "shares_added": DCA_INVESTMENT_USD / px,
                }
            )
        mark_px = price_on_or_before(prices, BENCHMARK_TICKER, qdate) or px or 0.0
        wealth_history.append((qdate, cash_shares * mark_px))

    return _finalize_dca_result("spy_benchmark", wealth_history, invested, log)


def _result_to_dict(result: DCAResult) -> dict[str, Any]:
    return {
        "strategy": result.strategy,
        "terminal_wealth": result.terminal_wealth,
        "total_invested": result.total_invested,
        "total_return": result.total_return,
        "cagr": result.cagr,
        "max_drawdown": result.max_drawdown,
        "period_start": result.period_start,
        "period_end": result.period_end,
        "quarters_invested": result.quarters_invested,
        "holdings_log": result.holdings_log,
    }


def _comparison_row(
    label: str,
    strategy: DCAResult,
    benchmark: DCAResult,
    *,
    vs_spy: bool = False,
) -> dict[str, Any]:
    wealth_delta = strategy.terminal_wealth - benchmark.terminal_wealth
    wealth_delta_pct = (
        wealth_delta / benchmark.terminal_wealth if benchmark.terminal_wealth > 0 else None
    )
    return_delta = strategy.total_return - benchmark.total_return
    # vs SPY: compare return on deployed capital (strategies invest fewer quarters).
    # vs old: compare terminal wealth (same deployment rules).
    beat = return_delta > 0 if vs_spy else strategy.terminal_wealth > benchmark.terminal_wealth
    return {
        "label": label,
        "terminal_wealth_delta": wealth_delta,
        "terminal_wealth_delta_pct": wealth_delta_pct,
        "total_return_delta": return_delta,
        "cagr_delta": strategy.cagr - benchmark.cagr,
        "beat_benchmark": beat,
        "beat_on_roi": return_delta > 0,
        "beat_on_wealth": strategy.terminal_wealth > benchmark.terminal_wealth,
    }


def run_dca_validation(
    old_weights: dict[str, float],
    new_weights: dict[str, float],
    old_thresholds: dict[str, float],
    new_thresholds: dict[str, float],
    panel: pd.DataFrame | None = None,
    delist_sensitivity: list[float] | None = None,
) -> dict[str, Any]:
    panel = panel if panel is not None else load_factor_panel()
    prices = ensure_benchmark_prices(load_prices())
    delist_sensitivity = delist_sensitivity or [0.0, DEFAULT_DELIST_RETURN, -1.0]
    quarter_ends = sorted(panel["quarter_end"].unique())

    def run_all(delist_return: float) -> dict[str, Any]:
        scored_old = score_factor_panel(panel, old_weights)
        scored_new = score_factor_panel(panel, new_weights)
        old = _simulate_dca(
            scored_old,
            prices,
            old_thresholds["composite_min"],
            old_thresholds["bargain_min"],
            "buy_and_hold",
            delist_return,
        )
        new = _simulate_dca(
            scored_new,
            prices,
            new_thresholds["composite_min"],
            new_thresholds["bargain_min"],
            "buy_and_hold",
            delist_return,
        )
        spy = _spy_dca_benchmark(prices, quarter_ends)
        return {
            "delist_return": delist_return,
            "old_buy_hold": old,
            "new_buy_hold": new,
            "spy": spy,
        }

    sensitivity = []
    for dr in delist_sensitivity:
        res = run_all(dr)
        spy = res["spy"]
        old = res["old_buy_hold"]
        new = res["new_buy_hold"]
        sensitivity.append(
            {
                "delist_return": dr,
                "old_terminal_wealth": old.terminal_wealth,
                "new_terminal_wealth": new.terminal_wealth,
                "spy_terminal_wealth": spy.terminal_wealth,
                "old_total_return": old.total_return,
                "new_total_return": new.total_return,
                "spy_total_return": spy.total_return,
                "old_cagr": old.cagr,
                "new_cagr": new.cagr,
                "spy_cagr": spy.cagr,
                "old_vs_spy_wealth_delta": old.terminal_wealth - spy.terminal_wealth,
                "new_vs_spy_wealth_delta": new.terminal_wealth - spy.terminal_wealth,
            }
        )

    base = run_all(DEFAULT_DELIST_RETURN)
    old = base["old_buy_hold"]
    new = base["new_buy_hold"]
    spy = base["spy"]

    payload = {
        "simulation": {
            "investment_per_quarter_usd": DCA_INVESTMENT_USD,
            "top_n_stocks": DCA_TOP_N,
            "benchmark_ticker": BENCHMARK_TICKER,
            "period_start": spy.period_start or old.period_start,
            "period_end": spy.period_end or old.period_end,
            "quarters_with_investment": spy.quarters_invested,
        },
        "default_delist_return": DEFAULT_DELIST_RETURN,
        "old_weights": old_weights,
        "new_weights": new_weights,
        "old_thresholds": old_thresholds,
        "new_thresholds": new_thresholds,
        "buy_and_hold": {
            "old": _result_to_dict(old),
            "new": _result_to_dict(new),
            "spy": _result_to_dict(spy),
        },
        "comparison_vs_spy": {
            "old_vs_spy": _comparison_row("old_vs_spy", old, spy, vs_spy=True),
            "new_vs_spy": _comparison_row("new_vs_spy", new, spy, vs_spy=True),
            "new_vs_old": _comparison_row("new_vs_old", new, old, vs_spy=False),
        },
        "survivorship_sensitivity": sensitivity,
    }
    path = RESULTS_DIR / "dca_validation.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
