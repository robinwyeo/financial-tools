"""Unit tests for backtest harness."""

from __future__ import annotations

from datetime import date
import json

import numpy as np
import pandas as pd
import pytest

from backtest.constants import BACKTEST_FACTOR_FAMILIES, BARGAIN_BACKTEST_COMPONENTS
from backtest.engine import (
    bootstrap_mean_ci,
    precompute_multi_horizon_returns,
    run_backtest,
    score_factor_panel,
)
from backtest.factors import enrich_factor_panel
from backtest.thresholds import calibrate_thresholds
from backtest.weights import (
    current_baseline_factor_weights,
    named_weight_candidates,
    normalize_backtest_weights,
    theme_weights_to_factor_weights,
)
from core.data import percentile_rank_in_history
from core.factors import FACTOR_SCORE_COLUMNS


def _synthetic_panel(n_tickers: int = 20, n_quarters: int = 8) -> pd.DataFrame:
    """Synthetic panel with all sub-signal columns for the backtestable groups."""
    rng = np.random.default_rng(0)
    qends = [date(2012, 3, 31), date(2012, 6, 30), date(2012, 9, 30), date(2012, 12, 31)]
    qends += [date(2013, 3, 31), date(2013, 6, 30), date(2013, 9, 30), date(2013, 12, 31)]
    qends = qends[:n_quarters]
    rows = []
    backtest_sub_cols = [
        col
        for family in BACKTEST_FACTOR_FAMILIES
        for col in FACTOR_SCORE_COLUMNS.get(family, [])
    ]
    for q in qends:
        for i in range(n_tickers):
            row = {
                "quarter_end": q,
                "ticker": f"T{i:02d}",
                "price": 100 + i,
                "sector": "Tech" if i % 2 == 0 else "Health",
                "bargain_score": rng.random() * 100,
                "graham_ratio": 0.5 + rng.random(),
                "bargain_discount_52w": rng.random() * 100,
            }
            for col in backtest_sub_cols:
                row[col] = rng.normal()
            rows.append(row)
    return pd.DataFrame(rows)


def _synthetic_prices(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    tickers = panel["ticker"].unique()
    for ticker in tickers:
        for d in pd.date_range("2011-01-01", "2014-12-31", freq="B"):
            rows.append({"Date": d, "ticker": ticker, "Close": 100.0})
    # Mild upward SPY path for excess-return math.
    for d in pd.date_range("2011-01-01", "2014-12-31", freq="B"):
        rows.append({"Date": d, "ticker": "SPY", "Close": 100.0 + (d - pd.Timestamp("2011-01-01")).days * 0.01})
    return pd.DataFrame(rows)


def test_normalize_backtest_weights_excludes_analyst_factor():
    full = current_baseline_factor_weights()
    assert "earnings_revisions" not in full
    assert 0.80 < sum(full.values()) < 1.05


def test_theme_weights_expand_to_factors():
    theme = {k: 1 / len(BACKTEST_FACTOR_FAMILIES) for k in BACKTEST_FACTOR_FAMILIES}
    fw = theme_weights_to_factor_weights(theme)
    for family in BACKTEST_FACTOR_FAMILIES:
        assert family in fw
    assert sum(fw.values()) == pytest.approx(1.0, rel=1e-6)


def test_named_weight_candidates_include_evidence_based():
    cands = named_weight_candidates()
    assert set(cands.keys()) == {"evidence_based", "legacy_tuned", "equal"}
    assert cands["evidence_based"]["quality"] > cands["legacy_tuned"]["quality"]


def test_score_factor_panel_adds_composite():
    panel = _synthetic_panel()
    weights = normalize_backtest_weights(current_baseline_factor_weights())
    scored = score_factor_panel(panel, weights)
    assert "composite" in scored.columns
    assert scored["composite"].notna().any()


def test_score_factor_panel_pct_columns_in_range():
    """All group percentile columns should be in [0, 100]."""
    panel = _synthetic_panel()
    weights = normalize_backtest_weights(current_baseline_factor_weights())
    scored = score_factor_panel(panel, weights)
    for family in BACKTEST_FACTOR_FAMILIES:
        col = f"pct_{family}"
        assert col in scored.columns, f"Missing {col}"
        vals = scored[col].dropna()
        assert (vals >= 0).all() and (vals <= 100).all(), f"{col} out of [0,100]"


def test_run_backtest_returns_metrics():
    panel = _synthetic_panel()
    prices = _synthetic_prices(panel)
    weights = current_baseline_factor_weights()
    result = run_backtest(
        weights,
        panel=panel,
        prices=prices,
        start=date(2012, 3, 31),
        end=date(2013, 12, 31),
        skip_ic=True,
    )
    assert 0.0 <= result.rolling_win_rate <= 1.0
    assert isinstance(result.cagr, float)


def test_multi_horizon_returns_have_expected_columns():
    panel = _synthetic_panel(n_quarters=8)
    prices = _synthetic_prices(panel)
    multi = precompute_multi_horizon_returns(panel, prices)
    assert not multi.empty
    for col in ("fwd_1q", "fwd_1y", "fwd_3y", "excess_1q"):
        assert col in multi.columns


def test_bootstrap_mean_ci_bounds():
    ci = bootstrap_mean_ci([0.1, 0.2, 0.15, 0.05, 0.12], n_boot=200, seed=1)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_percentile_rank_in_history():
    assert percentile_rank_in_history(0.10, [0.05, 0.08, 0.10, 0.12, 0.15]) == pytest.approx(60.0)
    assert percentile_rank_in_history(None, [0.1, 0.2, 0.3, 0.4]) is None
    assert percentile_rank_in_history(0.1, [0.1, 0.2]) is None  # need >= 4


def test_enrich_factor_panel_adds_valuation_bargain():
    panel = _synthetic_panel(n_tickers=5, n_quarters=8)
    # Ensure earnings_yield varies so history percentiles are defined.
    panel["earnings_yield"] = np.linspace(0.02, 0.12, len(panel))
    enriched = enrich_factor_panel(panel)
    assert "valuation_vs_history" in enriched.columns
    assert "bargain_valuation_vs_history" in enriched.columns
    assert "bargain_rsi_oversold" not in enriched.columns
    for comp in BARGAIN_BACKTEST_COMPONENTS:
        assert f"bargain_{comp}" in enriched.columns


def test_bargain_validation_uses_forward_horizons(tmp_path):
    """validate_bargain_weights should run and return a primary IC float."""
    from backtest.tune import validate_bargain_weights

    rng = np.random.default_rng(42)
    qends = [date(2012, 3, 31), date(2012, 6, 30), date(2012, 9, 30), date(2012, 12, 31)]
    n_tickers = 30
    rows = []
    for q in qends:
        for i in range(n_tickers):
            rows.append({
                "quarter_end": q,
                "ticker": f"T{i:02d}",
                "price": 100 + i,
                "bargain_score": rng.random() * 100,
                "bargain_margin_of_safety": rng.random() * 100,
                "bargain_discount_52w": rng.random() * 100,
                "bargain_valuation_vs_history": rng.random() * 100,
                "earnings_yield": 0.05 + rng.random() * 0.05,
                "graham_ratio": 0.5 + rng.random(),
            })
    panel = pd.DataFrame(rows)

    price_rows = []
    for ticker in [f"T{i:02d}" for i in range(n_tickers)] + ["SPY"]:
        for d in pd.date_range("2011-01-01", "2014-06-30", freq="B"):
            price_rows.append({"Date": d, "ticker": ticker, "Close": 100.0})
    prices = pd.DataFrame(price_rows)

    # Clear engine caches so synthetic prices are used.
    import backtest.engine as eng
    eng._FORWARD_RETURNS = None
    eng._MULTI_HORIZON_RETURNS = None
    eng._MONTHLY_RETURNS = None
    eng._QUARTER_END_PRICES = None

    # results_dir=tmp_path keeps the test from clobbering real artifacts
    # in backtest/results/ (which previously silently zeroed them out).
    result = validate_bargain_weights(panel=panel, prices=prices, results_dir=tmp_path)
    assert isinstance(result["winner_mean_ic"], float)
    assert "valuation_vs_history" in result["winner_weights"]
    assert "rsi_oversold" not in result["winner_weights"]


def test_calibrate_thresholds_returns_bounds(tmp_path):
    panel = _synthetic_panel()
    prices = _synthetic_prices(panel)
    weights = current_baseline_factor_weights()

    import backtest.engine as eng
    eng._FORWARD_RETURNS = None
    eng._MULTI_HORIZON_RETURNS = None
    eng._MONTHLY_RETURNS = None
    eng._QUARTER_END_PRICES = None

    out = calibrate_thresholds(
        weights, panel=panel, prices=prices, horizon="1q", results_dir=tmp_path
    )
    assert 30.0 <= out["composite_min"] <= 80.0
    assert 30.0 <= out["bargain_min"] <= 80.0
    assert "train_only" in out
    assert "full_sample" in out


def test_sector_from_sic():
    from backtest.data.constituents import sector_from_sic

    assert sector_from_sic(7372) == "Technology"
    assert sector_from_sic(4911) == "Utilities"
    assert sector_from_sic(6021) == "Financial Services"
    assert sector_from_sic(2834) == "Healthcare"


def test_attach_sector_uses_sic_map(monkeypatch):
    from backtest.engine import _attach_sector_if_missing

    monkeypatch.setattr(
        "backtest.data.constituents.sector_map_for_tickers",
        lambda tickers: {"AAA": "Energy"},
    )
    df = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": [None, None]})
    out = _attach_sector_if_missing(df)
    assert out.loc[out["ticker"] == "AAA", "sector"].iloc[0] == "Energy"
    assert pd.isna(out.loc[out["ticker"] == "BBB", "sector"].iloc[0]) or out.loc[
        out["ticker"] == "BBB", "sector"
    ].iloc[0] in {None, np.nan}


def test_apply_refuses_thresholds_without_allow_in_sample(tmp_path, monkeypatch):
    import argparse
    import yaml

    from backtest import run as run_mod

    (tmp_path / "config.yaml").write_text(
        "thresholds:\n  composite_min: 50.0\n  bargain_min: 50.0\nfactor_weights: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "threshold_calibration.json").write_text(
        '{"composite_min": 99.0, "bargain_min": 88.0}',
        encoding="utf-8",
    )
    monkeypatch.setattr(run_mod, "ROOT", tmp_path)
    monkeypatch.setattr(run_mod, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(
        run_mod,
        "load_config",
        lambda: {"thresholds": {"composite_min": 50.0, "bargain_min": 50.0}, "factor_weights": {}},
    )
    monkeypatch.setattr(run_mod, "get_factor_weights", lambda cfg=None: {})
    monkeypatch.setattr(run_mod, "get_bargain_weights", lambda cfg=None: {})
    args = argparse.Namespace(allow_in_sample=False, use_dca_cv=False)
    run_mod.cmd_apply(args)
    written = yaml.safe_load((tmp_path / "config.yaml").read_text(encoding="utf-8"))
    assert written["thresholds"]["composite_min"] == 50.0
    assert written["thresholds"]["bargain_min"] == 50.0
    assert written["provenance"]["thresholds_updated"] is False


def test_registry_round_trip_and_decision_signal():
    from backtest.registry import (
        MetricSpec,
        all_metrics,
        evaluate_metric,
        evaluate_signal,
        get_metric,
        get_signal,
        register,
    )

    panel = _synthetic_panel()
    panel["excess_3y"] = np.linspace(-0.2, 0.3, len(panel))
    panel["quality_score"] = np.linspace(10, 90, len(panel))
    spec = get_metric("quality_score")
    assert spec is not None
    card = evaluate_metric(panel, spec, fwd_col="excess_3y")
    assert "mean_ic" in card
    assert "n_independent_windows" in card
    dummy = MetricSpec("dummy_metric", "test", ("quality_score",), spec.fn)
    register(dummy)
    assert get_metric("dummy_metric") is not None
    assert dummy in all_metrics() or get_metric("dummy_metric").name == "dummy_metric"

    panel["is_good_buy"] = panel["quality_score"] > 50
    panel["decision_label"] = np.where(panel["is_good_buy"], "Accumulate", "Avoid")
    sig = get_signal("decision")
    out = evaluate_signal(panel, sig)
    assert "stats_3y" in out
    assert "mean_excess_ci_low" in out["stats_3y"]
    assert "mean_excess_ci_high" in out["stats_3y"]
    assert "mean_excess_ci_low" in out["stats_5y"]
    assert "mean_excess_ci_high" in out["stats_5y"]
    assert out["mean_accumulate_per_quarter"] >= 0


def _synthetic_companyfacts(
    ticker: str, *, revenue0: float = 100.0, years: list[int] | None = None
) -> pd.DataFrame:
    """FY companyfacts rows — default seven years, enough for 5y CAGRs and owner earnings."""
    years = list(years) if years is not None else list(range(2014, 2021))
    n = len(years)
    flow_fields = {
        ("revenue", "Revenues"): [revenue0 + 10 * i for i in range(n)],
        ("gross_profit", "GrossProfit"): [40 + 4 * i for i in range(n)],
        ("ebit", "OperatingIncomeLoss"): [20 + 2 * i for i in range(n)],
        ("net_income", "NetIncomeLoss"): [15 + i for i in range(n)],
        ("operating_cashflow", "NetCashProvidedByUsedInOperatingActivities"): [25 + i for i in range(n)],
        ("capex", "PaymentsToAcquirePropertyPlantAndEquipment"): [-5.0] * n,
        ("sbc", "ShareBasedCompensation"): [1.0] * n,
        ("interest_expense", "InterestExpense"): [2.0] * n,
        ("shares_diluted", "WeightedAverageNumberOfDilutedSharesOutstanding"): [10.0] * n,
    }
    instant_fields = {
        ("total_assets", "Assets"): [200.0] * n,
        ("total_liabilities", "Liabilities"): [120.0] * n,
        ("current_assets", "AssetsCurrent"): [50.0] * n,
        ("current_liabilities", "LiabilitiesCurrent"): [20.0] * n,
        ("total_cash", "CashAndCashEquivalentsAtCarryingValue"): [20.0] * n,
        ("long_term_debt", "LongTermDebt"): [30.0] * n,
        ("book_equity", "StockholdersEquity"): [80.0] * n,
        ("retained_earnings", "RetainedEarningsAccumulatedDeficit"): [40.0] * n,
        ("ppe_net", "PropertyPlantAndEquipmentNet"): [80.0] * n,
        ("shares_outstanding", "CommonStockSharesOutstanding"): [10.0] * n,
        ("goodwill", "Goodwill"): [8.0] * n,
    }
    rows: list[dict] = []
    for (field, tag), values in flow_fields.items():
        for year, val in zip(years, values):
            rows.append(
                {
                    "ticker": ticker,
                    "cik": 1,
                    "field": field,
                    "tag": tag,
                    "val": float(val),
                    "end": pd.Timestamp(f"{year}-12-31"),
                    "start": pd.Timestamp(f"{year}-01-01"),
                    "filed": pd.Timestamp(f"{year + 1}-02-15"),
                    "qtrs": 4,
                    "fp": "FY",
                    "form": "10-K",
                }
            )
    for (field, tag), values in instant_fields.items():
        for year, val in zip(years, values):
            rows.append(
                {
                    "ticker": ticker,
                    "cik": 1,
                    "field": field,
                    "tag": tag,
                    "val": float(val),
                    "end": pd.Timestamp(f"{year}-12-31"),
                    "start": pd.NaT,
                    "filed": pd.Timestamp(f"{year + 1}-02-15"),
                    "qtrs": 0,
                    "fp": "FY",
                    "form": "10-K",
                }
            )
    return pd.DataFrame(rows)


def test_complete_fundamentals_row_gets_decision_label(monkeypatch):
    """PIT quality/valuation/decide on a row with complete companyfacts."""
    from core.factors import QUALITY_SCORE_COLUMNS

    from backtest.factors import (
        _QUALITY_INPUT_COLUMNS,
        _attach_quality_and_decision,
        _build_raw_row,
        compute_historical_factors,
    )

    monkeypatch.setattr("backtest.factors.pit_risk_free", lambda as_of, history=None: 0.042)

    as_of = date(2021, 3, 31)
    tickers = ["AAA", "BBB", "CCC"]
    facts = pd.concat(
        [_synthetic_companyfacts(t, revenue0=100.0 + 10 * i) for i, t in enumerate(tickers)],
        ignore_index=True,
    )
    price_rows = []
    for ticker in tickers:
        for d in pd.date_range("2019-01-01", "2021-03-31", freq="B"):
            price_rows.append({"Date": d, "ticker": ticker, "Close": 12.0})
    prices = pd.DataFrame(price_rows)

    panel_rows = []
    for ticker in tickers:
        raw = _build_raw_row(ticker, as_of, facts, prices, rf=0.042)
        assert raw["decision_label"] in {"Accumulate", "Watch", "Avoid"}
        assert raw["revenue_5y_cagr"] is not None
        assert raw["owner_earnings_norm"] is not None
        assert raw.get("ev_to_ebit") is not None or raw.get("p_to_oe") is not None
        factors = compute_historical_factors(raw)
        row = {
            "quarter_end": as_of,
            "ticker": ticker,
            "price": raw["price"],
            "sector": "Technology",
            "decision_label": raw["decision_label"],
            "_valuation": raw.get("_valuation") or {},
            "_value_trap_flags": raw.get("_value_trap_flags") or [],
            "ev_to_ebit": raw.get("ev_to_ebit"),
            "p_to_oe": raw.get("p_to_oe"),
            "altman_z": factors.get("altman_z"),
            "altman_z_pp": factors.get("altman_z_pp"),
        }
        for col in _QUALITY_INPUT_COLUMNS:
            val = factors.get(col)
            row[col] = val if val is not None else raw.get(col)
        for family_cols in QUALITY_SCORE_COLUMNS.values():
            for col in family_cols:
                if col not in row:
                    row[col] = factors.get(col)
        panel_rows.append(row)

    panel = pd.DataFrame(panel_rows)
    out = _attach_quality_and_decision(panel)
    assert out["decision_label"].notna().all()
    assert out["decision_label"].isin(["Accumulate", "Watch", "Avoid"]).all()
    assert out["quality_score"].notna().any()


def test_build_raw_row_uses_pit_risk_free_by_as_of(monkeypatch):
    """Fake DGS10 that steps in 2015 vs 2020 must reach hurdle_rate via PIT rf."""
    from backtest.factors import _build_raw_row
    from core.rates import pit_risk_free

    hist = pd.Series(
        [0.022, 0.006],
        index=pd.to_datetime(["2015-06-15", "2020-06-15"]),
        dtype="float64",
    )
    monkeypatch.setattr("core.rates.fetch_fred_history", lambda *a, **k: hist)
    monkeypatch.setattr("core.rates.fetch_fred_series", lambda *a, **k: None)
    monkeypatch.delenv("FRED_API_KEY", raising=False)

    ticker = "AAA"
    facts = _synthetic_companyfacts(ticker, years=list(range(2009, 2021)))
    prices = pd.DataFrame(
        [
            {"Date": d, "ticker": ticker, "Close": 12.0}
            for d in pd.date_range("2014-01-01", "2020-12-31", freq="B")
        ]
    )

    as_of_2015 = date(2015, 12, 31)
    as_of_2020 = date(2020, 12, 31)
    raw_2015 = _build_raw_row(ticker, as_of_2015, facts, prices)
    raw_2020 = _build_raw_row(ticker, as_of_2020, facts, prices)

    rf_2015 = ((raw_2015.get("_valuation") or {}).get("hurdle") or {}).get("rf")
    rf_2020 = ((raw_2020.get("_valuation") or {}).get("hurdle") or {}).get("rf")
    assert rf_2015 == pytest.approx(pit_risk_free(as_of_2015, hist))
    assert rf_2020 == pytest.approx(pit_risk_free(as_of_2020, hist))
    assert rf_2015 == pytest.approx(0.022)
    assert rf_2020 == pytest.approx(0.006)
    assert rf_2015 != rf_2020


def test_committed_report_has_no_survivorship_sensitivity_table():
    from backtest.constants import RESULTS_DIR

    report = (RESULTS_DIR / "backtest_report.md").read_text(encoding="utf-8")
    assert "Survivorship sensitivity" not in report


def test_artifacts_share_run_id_when_present():
    from backtest.constants import RESULTS_DIR

    json_paths = list(RESULTS_DIR.glob("*.json"))
    ids = []
    for path in json_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict) and data.get("run_id"):
            ids.append(data["run_id"])
    if json_paths and not ids:
        pytest.fail(
            "committed backtest JSON artifacts have no run_id; "
            "run `python -m backtest.run pipeline` or delete stale files"
        )
    if not ids:
        pytest.skip("no JSON artifacts in results/")
    assert len(set(ids)) == 1
