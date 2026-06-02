#!/usr/bin/env python3
"""Weekly job: refresh full S&P 500 snapshot and email Monday scorecard."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.scoring import score_ticker
from core.universe import build_universe_snapshot, fetch_sp500_tickers, load_universe_snapshot
from jobs.email_sender import email_is_enabled, format_scorecard_email, send_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_weekly(
    refresh_universe: bool = True,
    max_tickers: int | None = None,
    send_report: bool = True,
    fast_universe: bool = False,
) -> int:
    config = load_config()

    if refresh_universe:
        logger.info("Refreshing full universe snapshot (max=%s, fast=%s)", max_tickers, fast_universe)
        if fast_universe:
            from core.universe import _fallback_sp500

            tickers = _fallback_sp500()
        else:
            tickers = fetch_sp500_tickers()
        if max_tickers:
            tickers = tickers[:max_tickers]
        build_universe_snapshot(tickers=tickers)

    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        logger.error("Universe snapshot is empty; cannot score S&P 500")
        return 1

    tickers = uni["ticker"].astype(str).str.upper().tolist()
    logger.info("Scoring S&P 500 universe (%d tickers)", len(tickers))

    results = []
    for i, ticker in enumerate(tickers, start=1):
        try:
            result = score_ticker(ticker, config, uni)
            if result.get("is_etf"):
                continue
            results.append(result)
            if i % 25 == 0 or i == len(tickers):
                buy_count = sum(1 for r in results if r.get("is_good_buy"))
                logger.info("Progress: %d / %d scored (%d Buy so far)", i, len(tickers), buy_count)
        except Exception as exc:
            logger.warning("Failed to score %s: %s", ticker, exc)

    buy_count = sum(1 for r in results if r.get("is_good_buy"))
    logger.info("Weekly scan complete: %d Buy / %d scored", buy_count, len(results))

    if send_report and email_is_enabled(config):
        subject, body = format_scorecard_email(
            results,
            config,
            title="Weekly S&P 500 Scorecard",
            subtitle=f"Full universe scan — {len(results)} ticker(s). Buys listed first.",
        )
        if send_email(subject, body, config):
            logger.info("Weekly scorecard email sent")
        else:
            logger.warning("Email not sent (check SMTP credentials)")
    elif send_report:
        logger.info("Email disabled; set email.enabled or SMTP_PASSWORD in environment")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly S&P 500 scorecard email")
    parser.add_argument("--no-refresh", action="store_true", help="Skip universe refresh")
    parser.add_argument(
        "--max",
        type=int,
        default=None,
        help="Max tickers to score (default: full snapshot / S&P 500)",
    )
    parser.add_argument("--no-email", action="store_true", help="Skip email sending")
    parser.add_argument("--fast", action="store_true", help="Use smaller fallback universe")
    args = parser.parse_args()

    sys.exit(
        run_weekly(
            refresh_universe=not args.no_refresh,
            max_tickers=args.max,
            send_report=not args.no_email,
            fast_universe=args.fast,
        )
    )
