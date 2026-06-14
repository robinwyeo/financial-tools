#!/usr/bin/env python3
"""CLI entry point for the backtest harness."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backtest.constants import RESULTS_DIR
from backtest.data.constituents import build_membership_panel
from backtest.data.edgar import ingest_edgar
from backtest.data.prices import ingest_prices
from backtest.factors import build_factor_panel
from backtest.report import generate_report
from backtest.simulate_dca import run_dca_validation
from backtest.thresholds import calibrate_thresholds
from backtest.tune import tune_bargain_weights, tune_factor_weights, tune_factor_weights_cv
from backtest.weights import current_baseline_factor_weights
from core.config import get_bargain_weights, get_factor_weights, get_thresholds, load_config
from core.scoring import BARGAIN_COMPONENT_WEIGHTS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def cmd_ingest(args: argparse.Namespace) -> None:
    logger.info("Building S&P 500 membership panel")
    build_membership_panel(force=args.force)
    logger.info("Ingesting SEC EDGAR fundamentals")
    ingest_edgar(force=args.force, max_quarters=args.max_edgar_quarters)
    logger.info("Downloading price history")
    ingest_prices(force=args.force, max_tickers=args.max_tickers)


def cmd_build_factors(args: argparse.Namespace) -> None:
    build_factor_panel(force=args.force, max_quarters=args.max_quarters)


def cmd_tune(args: argparse.Namespace) -> None:
    tune_factor_weights(n_samples=args.n_samples, seed=args.seed)
    tune_bargain_weights(n_samples=args.bargain_samples, seed=args.seed + 1)


def cmd_tune_cv(args: argparse.Namespace) -> None:
    tune_factor_weights_cv(
        n_samples=args.n_samples,
        seed=args.seed,
        k_folds=args.k_folds,
    )


def cmd_calibrate(args: argparse.Namespace) -> None:
    tuning = json.loads((RESULTS_DIR / "tuning_results.json").read_text(encoding="utf-8"))
    weights = tuning["winner"]["factor_weights"]
    calibrate_thresholds(weights)


def cmd_dca(args: argparse.Namespace) -> None:
    tuning = json.loads((RESULTS_DIR / "tuning_results.json").read_text(encoding="utf-8"))
    thresholds = json.loads((RESULTS_DIR / "threshold_calibration.json").read_text(encoding="utf-8"))
    bargain = json.loads((RESULTS_DIR / "bargain_tuning_results.json").read_text(encoding="utf-8"))

    old_weights = get_factor_weights()
    new_weights = {**old_weights, **tuning["winner"]["factor_weights"]}
    old_thresholds = get_thresholds()
    new_thresholds = {
        **old_thresholds,
        "composite_min": thresholds["composite_min"],
        "bargain_min": thresholds["bargain_min"],
    }
    run_dca_validation(old_weights, new_weights, old_thresholds, new_thresholds)


def cmd_report(args: argparse.Namespace) -> None:
    text = generate_report()
    print(text)


def cmd_apply(args: argparse.Namespace) -> None:
    """Apply tuned weights/thresholds to config.yaml."""
    import yaml

    use_dca_cv = getattr(args, "use_dca_cv", False)
    tuning_path = (
        RESULTS_DIR / "tuning_results_dca_cv.json"
        if use_dca_cv
        else RESULTS_DIR / "tuning_results.json"
    )
    bargain_path = RESULTS_DIR / "bargain_tuning_results.json"
    threshold_path = RESULTS_DIR / "threshold_calibration.json"

    if not tuning_path.exists():
        which = "tune-cv" if use_dca_cv else "tune"
        raise FileNotFoundError(f"Run {which} first to produce {tuning_path.name}")

    logger.info("Applying factor weights from %s", tuning_path.name)
    tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
    bargain = json.loads(bargain_path.read_text(encoding="utf-8")) if bargain_path.exists() else {}
    thresholds = json.loads(threshold_path.read_text(encoding="utf-8")) if threshold_path.exists() else {}

    cfg = load_config()
    winner_fw = tuning["winner"]["factor_weights"]
    # Merge tuned reconstructable factors; keep excluded factors at current values.
    merged = get_factor_weights(cfg)
    merged.update(winner_fw)

    cfg["factor_weights"] = {k: round(float(v), 4) for k, v in merged.items()}
    # The DCA k-fold CV artifact only tunes factor weights; its thresholds and
    # bargain weights were not recalibrated for the new weights, so we leave the
    # existing thresholds/bargain weights untouched when applying it.
    if not use_dca_cv:
        if thresholds:
            cfg.setdefault("thresholds", {})
            cfg["thresholds"]["composite_min"] = round(float(thresholds["composite_min"]), 1)
            cfg["thresholds"]["bargain_min"] = round(float(thresholds["bargain_min"]), 1)
        if bargain.get("winner_weights"):
            full_bw = get_bargain_weights(cfg)
            full_bw.update({k: round(float(v), 4) for k, v in bargain["winner_weights"].items()})
            cfg["bargain_weights"] = full_bw

    config_path = ROOT / "config.yaml"
    with config_path.open("w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("Updated %s with tuned factor weights and thresholds", config_path)


def cmd_pipeline(args: argparse.Namespace) -> None:
    cmd_ingest(args)
    cmd_build_factors(args)
    cmd_tune(args)
    cmd_calibrate(args)
    cmd_dca(args)
    cmd_report(args)
    if args.apply:
        cmd_apply(args)


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--force", action="store_true", help="Rebuild cached artifacts")
    p.add_argument("--max-edgar-quarters", type=int, default=None, help="Limit SEC quarters ingested")
    p.add_argument("--max-quarters", type=int, default=None, help="Limit factor panel quarters")
    p.add_argument("--max-tickers", type=int, default=None, help="Limit price downloads")
    p.add_argument("--n-samples", type=int, default=500, help="Factor weight search samples")
    p.add_argument("--bargain-samples", type=int, default=200, help="Bargain weight search samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k-folds", type=int, default=5, help="Folds for DCA-CV tuning")
    p.add_argument("--apply", action="store_true", help="Write tuned values to config.yaml")
    p.add_argument(
        "--use-dca-cv",
        action="store_true",
        help="On apply, take factor weights from the DCA k-fold CV winner "
        "(tuning_results_dca_cv.json) instead of the default tuning_results.json",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Historical score weight backtest harness")
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_text in [
        ("ingest", "Download constituents, EDGAR, prices"),
        ("build-factors", "Build quarterly factor panel"),
        ("tune", "Tune factor and bargain weights"),
        ("tune-cv", "Tune factor weights on DCA terminal wealth with k-fold CV"),
        ("calibrate", "Calibrate good-buy thresholds"),
        ("dca", "Run DCA validation simulation"),
        ("report", "Generate markdown report"),
        ("apply", "Apply tuned config to config.yaml"),
        ("pipeline", "Run full pipeline"),
    ]:
        sp = sub.add_parser(name, help=help_text)
        _add_shared_args(sp)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "ingest": cmd_ingest,
        "build-factors": cmd_build_factors,
        "tune": cmd_tune,
        "tune-cv": cmd_tune_cv,
        "calibrate": cmd_calibrate,
        "dca": cmd_dca,
        "report": cmd_report,
        "apply": cmd_apply,
        "pipeline": cmd_pipeline,
    }
    handlers[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
