#!/usr/bin/env python3
"""Daily job: refresh universe snapshot, score watchlist, email alerts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.config import load_config
from core.scoring import evaluate_watchlist, score_ticker
from core.universe import build_universe_snapshot, fetch_sp500_tickers, load_universe_snapshot
from jobs.email_sender import format_alert_email, send_email

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_daily(
    refresh_universe: bool = True,
    max_universe: int | None = 100,
    send_alerts: bool = True,
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

    watchlist = config.get("watchlist", [])
    logger.info("Scoring watchlist: %s", watchlist)

    all_results = []
    for ticker in watchlist:
        try:
            result = score_ticker(ticker, config, uni)
            all_results.append(result)
            status = "GOOD BUY" if result.get("is_good_buy") else "no signal"
            logger.info("%s: composite=%.1f, upside=%s%% -> %s",
                ticker,
                result.get("composite") or 0,
                (result.get("analyst") or {}).get("implied_upside_pct"),
                status,
            )
        except Exception as exc:
            logger.warning("Failed to score %s: %s", ticker, exc)

    alerts = [r for r in all_results if r.get("is_good_buy")]
    logger.info("%d good-buy alert(s) found", len(alerts))

    email_cfg = config.get("email", {})
    email_enabled = email_cfg.get("enabled", False) or bool(__import__("os").environ.get("SMTP_PASSWORD"))

    if send_alerts and alerts and email_enabled:
        subject, body = format_alert_email(alerts, config)
        if send_email(subject, body, config):
            logger.info("Alert email sent")
        else:
            logger.warning("Email not sent (check SMTP credentials)")
    elif alerts:
        logger.info("Alerts found but email disabled; set email.enabled or SMTP_PASSWORD")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Daily stock metrics check")
    parser.add_argument("--no-refresh", action="store_true", help="Skip universe refresh")
    parser.add_argument("--max-universe", type=int, default=100, help="Max tickers in universe refresh")
    parser.add_argument("--no-email", action="store_true", help="Skip email sending")
    parser.add_argument("--fast", action="store_true", help="Use smaller fallback universe")
    args = parser.parse_args()

    sys.exit(
        run_daily(
            refresh_universe=not args.no_refresh,
            max_universe=args.max_universe,
            send_alerts=not args.no_email,
            fast_universe=args.fast,
        )
    )
