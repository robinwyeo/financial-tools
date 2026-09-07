#!/usr/bin/env python3
"""Weekly job: score watchlist and email scorecard (uses existing universe snapshot)."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_universe_members, load_config
from core.rates import hurdle_rate
from core.scoring import apply_universe_snapshot_scoring, score_ticker, score_universe_df
from core.universe import build_universe_snapshot, load_universe_snapshot
from core.watchlist import load_watchlist
from jobs.email_sender import email_is_enabled, format_scorecard_email, send_email, smtp_config_status
from jobs.runlog import exceeds_failure_threshold, run_id_now, write_run_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def score_tickers(tickers: list[str], config: dict, uni) -> tuple[list[dict], list[dict[str, str]]]:
    scored_universe = None
    if uni is not None and not uni.empty:
        try:
            scored_universe = score_universe_df(uni, config)
        except Exception as exc:
            logger.warning("Failed to score universe snapshot for alignment: %s", exc)

    results = []
    failures: list[dict[str, str]] = []
    for ticker in tickers:
        try:
            result = score_ticker(ticker, config, uni)
            if result.get("is_etf"):
                logger.info("%s: skipped (ETF)", ticker)
                continue
            if scored_universe is not None:
                result = apply_universe_snapshot_scoring(
                    result, scored_universe, ticker, config
                )
            results.append(result)
            bargain = (result.get("bargain") or {}).get("score")
            label = (result.get("decision") or {}).get("label")
            if not label:
                label = "Accumulate" if result.get("is_good_buy") else "Avoid"
            logger.info(
                "%s: composite=%s bargain=%s -> %s",
                ticker,
                f"{result.get('composite'):.1f}" if result.get("composite") is not None else "N/A",
                f"{bargain:.1f}" if bargain is not None else "N/A",
                label,
            )
        except Exception as exc:
            logger.warning("Failed to score %s: %s", ticker, exc)
            failures.append({"ticker": ticker, "error": str(exc)})
    return results, failures


def run_weekly(
    refresh_universe: bool = False,
    max_universe: int | None = None,
    send_report: bool = True,
    fast_universe: bool = False,
) -> int:
    started = time.monotonic()
    run_id = run_id_now()
    config = load_config()
    extra_failures: list[dict[str, str]] = []

    if refresh_universe:
        members = get_universe_members(config)
        logger.info("Refreshing universe snapshot (members=%s, max=%s, fast=%s)", members, max_universe, fast_universe)
        if fast_universe:
            from core.universe import _fallback_sp500

            build_universe_snapshot(tickers=_fallback_sp500(), max_tickers=max_universe)
        else:
            build_universe_snapshot(universes=members, max_tickers=max_universe)
    else:
        logger.info("Skipping universe refresh (watchlist-only run; use --refresh or universe_monthly.py to rebuild)")

    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        logger.error(
            "Universe snapshot is empty. Run `python jobs/universe_monthly.py` or "
            "`python -m core.universe` first, or pass --refresh."
        )
        write_run_summary("watchlist_weekly", {"ok": False, "error": "empty snapshot"}, run_id=run_id)
        return 1

    watchlist = load_watchlist()
    if not watchlist:
        logger.warning("Watchlist is empty (edit the `watchlist` file at repo root)")
        write_run_summary("watchlist_weekly", {"ok": False, "error": "empty watchlist"}, run_id=run_id)
        return 1

    logger.info("Scoring watchlist (%d tickers): %s", len(watchlist), watchlist)
    results, failures = score_tickers(watchlist, config, uni)
    failures.extend(extra_failures)
    buy_count = sum(1 for r in results if r.get("is_good_buy"))
    logger.info("Watchlist complete: %d Accumulate / %d scored", buy_count, len(results))

    snapshot_date = None
    if "snapshot_date" in uni.columns and not uni.empty:
        snapshot_date = str(uni["snapshot_date"].iloc[0])
    hurdle = hurdle_rate()
    email_ok = True
    if send_report and email_is_enabled(config):
        ready, status = smtp_config_status(config)
        logger.info("Email config: %s", status)
        subject, body = format_scorecard_email(
            results,
            config,
            title="Weekly Watchlist Scorecard",
            subtitle=f"{len(results)} ticker(s) from your watchlist.",
            snapshot_date=snapshot_date,
            run_id=run_id,
            hurdle_rate=hurdle.get("rate"),
        )
        sent, message = send_email(subject, body, config)
        if sent:
            logger.info("Watchlist scorecard email sent (%s)", message)
        else:
            logger.error("Email not sent: %s", message)
            print(f"::error title=Email delivery failed::{message}")
            email_ok = False
    elif send_report:
        logger.info("Email disabled; set email.enabled or SMTP_PASSWORD in environment")

    attempted = len(watchlist)
    failed_n = len(failures)
    too_many = exceeds_failure_threshold(failed_n, attempted)
    write_run_summary(
        "watchlist_weekly",
        {
            "ok": email_ok and not too_many,
            "attempted": attempted,
            "scored": len(results),
            "accumulate": buy_count,
            "failed": failed_n,
            "failures": failures[:50],
            "snapshot_date": snapshot_date,
            "hurdle": hurdle.get("rate"),
            "duration_s": round(time.monotonic() - started, 2),
        },
        run_id=run_id,
    )
    if too_many:
        logger.error("Failure ratio %d/%d exceeds 10%%", failed_n, attempted)
        return 1
    if not email_ok:
        return 1
    return 0


run_daily = run_weekly  # backward-compatible alias


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly watchlist scorecard email")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild universe snapshot first (slow; normally done by universe_monthly.py)",
    )
    parser.add_argument(
        "--max-universe",
        type=int,
        default=None,
        help="Max tickers in universe refresh (default: full list)",
    )
    parser.add_argument("--no-email", action="store_true", help="Skip email sending")
    parser.add_argument("--fast", action="store_true", help="Use smaller fallback universe")
    args = parser.parse_args()

    sys.exit(
        run_weekly(
            refresh_universe=args.refresh,
            max_universe=args.max_universe,
            send_report=not args.no_email,
            fast_universe=args.fast,
        )
    )
