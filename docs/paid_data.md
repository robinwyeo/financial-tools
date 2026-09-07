# Paid data: what it would change, and why it is deferred

This memo records the **not now** decision on buying a survivorship-free US
equities feed. The live tool and the backtest both use free sources
(Yahoo Finance + SEC companyfacts + FRED). That is enough for a satellite
sleeve screener. It is **not** enough for an honest historical simulation of
every name that was in the S&P 500.

## The gap

The backtest universe is today's (or historically reconstituted Wikipedia)
S&P 500 list priced with yfinance. Names that delisted, were acquired, or
went to zero typically have **no** usable Yahoo history, so they can never
be bought in the simulation. Strategy returns are therefore biased upward.
Shumway (1997) documents that ignoring delisting returns, especially
performance-related delists around −30% to −55%, inflates mean returns of
distress/value sorts. The committed report already labels full-period wealth
tables as in-sample and survivorship-biased; paid data is how you would
close that gap, not how you would invent alpha.

Rough size of the hole: S&P 500 membership turns over. Over 2010–2026 that
is on the order of **~175** historical members missing from a Yahoo-only
price panel. Those are exactly the names most likely to have been cheap,
distressed, or acquired — the left tail the quality screen is trying to
avoid.

## Vendors considered (2026)

| Vendor | What you get | Ballpark | Fit |
|--------|----------------|----------|-----|
| **Sharadar Core US Equities** (Nasdaq Data Link) | ~14k tickers incl. delisted; PIT fundamentals (SF1), prices (SEP) | ~$50/month | Best match. Adapter stub: `core/providers/sharadar.py` |
| **EODHD** fundamentals + delisted | Prices + fundamentals, delisted flag | ~$30–80/month | Cheaper; PIT quality weaker than Sharadar SF1 |
| **Norgate** | Point-in-time index constituents and prices | ~$30/month | Prices/constituents only; still need fundamentals |

Sources **not** purchased: 13F extracts (45-day lag, low value at 3–20y),
Alpha Vantage free (25 calls/day), Polygon free (prices, 5/min), FMP free
(250/day — cannot cover 500 names plus history).

## What would change in this repo

1. `core/providers/sharadar.py` would implement `FundamentalsProvider`
   against SF1 (ART/MRQ) and SEP.
2. `backtest/data/prices.py` would prefer SEP, including delist rows, and
   apply Shumway-style delist returns when the last price is a performance
   delist.
3. `backtest/data/constituents.py` would still be Wikipedia/S&P 500 for the
   membership list unless Norgate PIT constituents were added. Sharadar does
   not magically give historical S&P 400 or TSX 60 membership.
4. Live scoring would keep EDGAR + Yahoo as primary; Sharadar would be a
   backtest/survivorship store, not a second live fundamental source.

## Expected effect

Closing delist coverage removes the mechanical upward bias. It will **not**
turn a small 3y rank IC into a tradable edge. With ~5 independent 3-year
windows the sample still cannot support weight optimization. Revisit after
a clean companyfacts pipeline run shows per-quarter `price_coverage_pct`
in the evidence block — if coverage is already high, the paid feed's
marginal value is mostly the delisted left tail, not extra live tickers.

## Decision

**Do not subscribe yet.** Keep the Sharadar stub importing cleanly. Revisit
when the clean backtest publishes price coverage and the owner wants to
treat historical Accumulate hit-rates as decision evidence rather than as
a methodology check.
