#!/usr/bin/env python3
"""Monthly job: refresh universe snapshot and email scorecard."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import get_universe_members, load_config
from core.fund_universe import build_fund_universe_snapshot
from core.rates import hurdle_rate
from core.scoring import apply_universe_snapshot_scoring, score_ticker, score_universe_df
from core.universe import build_universe_snapshot, load_universe_snapshot
from jobs.email_sender import email_is_enabled, format_scorecard_email, send_email, smtp_config_status
from jobs.runlog import exceeds_failure_threshold, run_id_now, write_run_summary

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_monthly(
    refresh_universe: bool = True,
    max_tickers: int | None = None,
    send_report: bool = True,
    fast_universe: bool = False,
) -> int:
    started = time.monotonic()
    run_id = run_id_now()
    config = load_config()
    failures: list[dict[str, str]] = []

    if refresh_universe:
        members = get_universe_members(config)
        logger.info(
            "Refreshing universe snapshot (members=%s, max=%s, fast=%s)",
            members,
            max_tickers,
            fast_universe,
        )
        if fast_universe:
            from core.universe import _fallback_sp500

            build_universe_snapshot(tickers=_fallback_sp500(), max_tickers=max_tickers)
        else:
            build_universe_snapshot(universes=members, max_tickers=max_tickers)

        logger.info("Refreshing fund universe snapshot (US + Canadian ETFs and mutual funds)")
        try:
            build_fund_universe_snapshot()
        except Exception as exc:
            logger.warning("Fund universe snapshot refresh failed: %s", exc)
            failures.append({"ticker": "_fund_universe", "error": str(exc)})

    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        logger.error("Universe snapshot is empty; cannot score universe")
        write_run_summary(
            "universe_monthly",
            {"ok": False, "error": "empty snapshot", "failures": failures},
            run_id=run_id,
        )
        return 1

    tickers = uni["ticker"].astype(str).str.upper().tolist()
    logger.info("Scoring universe (%d tickers)", len(tickers))

    scored_universe = score_universe_df(uni, config)
    results = []
    for i, ticker in enumerate(tickers, start=1):
        try:
            result = score_ticker(ticker, config, uni)
            if result.get("is_etf"):
                continue
            result = apply_universe_snapshot_scoring(
                result, scored_universe, ticker, config
            )
            results.append(result)
            if i % 25 == 0 or i == len(tickers):
                buy_count = sum(1 for r in results if r.get("is_good_buy"))
                logger.info("Progress: %d / %d scored (%d Accumulate so far)", i, len(tickers), buy_count)
        except Exception as exc:
            logger.warning("Failed to score %s: %s", ticker, exc)
            failures.append({"ticker": ticker, "error": str(exc)})

    buy_count = sum(1 for r in results if r.get("is_good_buy"))
    logger.info("Monthly scan complete: %d Accumulate / %d scored", buy_count, len(results))

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
            title="Monthly Universe Scorecard",
            subtitle=f"Full universe scan — {len(results)} ticker(s). Accumulate listed first.",
            snapshot_date=snapshot_date,
            run_id=run_id,
            hurdle_rate=hurdle.get("rate"),
        )
        sent, message = send_email(subject, body, config)
        if sent:
            logger.info("Monthly scorecard email sent (%s)", message)
        else:
            logger.error("Email not sent: %s", message)
            email_ok = False
    elif send_report:
        logger.info("Email disabled; set email.enabled or SMTP_PASSWORD in environment")

    attempted = len(tickers)
    failed_n = len([f for f in failures if f.get("ticker") != "_fund_universe"])
    too_many = exceeds_failure_threshold(failed_n, attempted)
    write_run_summary(
        "universe_monthly",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monthly universe scorecard email")
    parser.add_argument("--no-refresh", action="store_true", help="Skip universe refresh")
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max tickers to score (default: full snapshot)",
    )
    parser.add_argument("--no-email", action="store_true", help="Skip email sending")
    parser.add_argument("--fast", action="store_true", help="Use smaller fallback universe")
    args = parser.parse_args()

    sys.exit(
        run_monthly(
            refresh_universe=not args.no_refresh,
            max_tickers=args.max,
            send_report=not args.no_email,
            fast_universe=args.fast,
        )
    )
