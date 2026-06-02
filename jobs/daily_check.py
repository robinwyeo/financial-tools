#!/usr/bin/env python3
"""Daily job: refresh universe snapshot, score watchlist, email morning scorecard."""

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
from core.watchlist import load_watchlist
from jobs.email_sender import email_is_enabled, format_scorecard_email, send_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def score_tickers(tickers: list[str], config: dict, uni) -> list[dict]:
    results = []
    for ticker in tickers:
        try:
            result = score_ticker(ticker, config, uni)
            if result.get("is_etf"):
                logger.info("%s: skipped (ETF)", ticker)
                continue
            results.append(result)
            bargain = (result.get("bargain") or {}).get("score")
            upside = (result.get("analyst") or {}).get("implied_upside_pct")
            verdict = "Buy" if result.get("is_good_buy") else "Not Buy"
            logger.info(
                "%s: composite=%s bargain=%s upside=%s%% -> %s",
                ticker,
                f"{result.get('composite'):.1f}" if result.get("composite") is not None else "N/A",
                f"{bargain:.1f}" if bargain is not None else "N/A",
                f"{upside:.1f}" if upside is not None else "N/A",
                verdict,
            )
        except Exception as exc:
            logger.warning("Failed to score %s: %s", ticker, exc)
    return results


def run_daily(
    refresh_universe: bool = True,
    max_universe: int | None = None,
    send_report: bool = True,
    fast_universe: bool = False,
) -> int:
    config = load_config()

    if refresh_universe:
        logger.info("Refreshing universe snapshot (max=%s, fast=%s)", max_universe, fast_universe)
        if fast_universe:
            from core.universe import _fallback_sp500

            tickers = _fallback_sp500()
        else:
            tickers = fetch_sp500_tickers()
        build_universe_snapshot(tickers=tickers, max_tickers=max_universe)

    uni = load_universe_snapshot()
    if uni is None or uni.empty:
        logger.error("Universe snapshot is empty; cannot score watchlist")
        return 1

    watchlist = load_watchlist(config)
    if not watchlist:
        logger.warning("Watchlist is empty (edit the `watchlist` file at repo root)")
        return 1

    logger.info("Scoring watchlist (%d tickers): %s", len(watchlist), watchlist)
    results = score_tickers(watchlist, config, uni)
    buy_count = sum(1 for r in results if r.get("is_good_buy"))
    logger.info("Watchlist complete: %d Buy / %d scored", buy_count, len(results))

    if send_report and email_is_enabled(config):
        subject, body = format_scorecard_email(
            results,
            config,
            title="Daily Watchlist Scorecard",
            subtitle=f"{len(results)} ticker(s) from your watchlist.",
        )
        if send_email(subject, body, config):
            logger.info("Watchlist scorecard email sent")
        else:
            logger.warning("Email not sent (check SMTP credentials)")
    elif send_report:
        logger.info("Email disabled; set email.enabled or SMTP_PASSWORD in environment")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily watchlist scorecard email")
    parser.add_argument("--no-refresh", action="store_true", help="Skip universe refresh")
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
        run_daily(
            refresh_universe=not args.no_refresh,
            max_universe=args.max_universe,
            send_report=not args.no_email,
            fast_universe=args.fast,
        )
    )
