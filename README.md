# Stock & Fund Metrics Tool

Empirical factor scoring and aggregated analyst recommendations for stocks — plus fund-appropriate factor scoring for US and Canadian ETFs and mutual funds — with weekly watchlist and monthly S&P 500 scorecard emails.

## What this tool is (and isn't)

The composite score has **not demonstrated a statistically significant edge over
the S&P 500** in out-of-sample validation (see "What the backtest actually
shows" below). Treat it as a **disqualifier and diversification aid** — a
structured way to refuse stocks with poor quality, leverage, or accrual
profiles, and to pick low-cost funds — not as an alpha-generating ranking to
chase. Keep a broad low-cost index core.

## Features

- **Streamlit dashboard** — enter a ticker for factor scorecard, analyst consensus, price targets, and implied upside
- **Nine factor groups** — Value, GARP, Quality, Balance Sheet, Momentum (12-1), Low Volatility, Capital Discipline, Estimate Revisions (Zacks-style), Insider Buying
- **Cross-sectional scoring** — empirical percentile ranks vs S&P 500 universe (sector-adjusted when enabled). The quality group is scored in three de-correlated sub-buckets (profitability, earnings quality, financial strength) so five correlated profitability ratios cannot dominate the group
- **Bargain score** — long-horizon cheapness (Graham margin of safety 55%, valuation vs own 10y EDGAR history 30%, 52-week discount 15%). History uses EBIT/EV, OCF yield, and book-to-market, scoring against the multiple that best tracked the stock's own price
- **Distress gate** — Altman Z below 1.8 blocks a Buy outright, regardless of composite/bargain scores (financials, real estate, and utilities are exempt — the classic Z model is invalid for those balance sheets)
- **Uncertainty badge** — Low/Medium/High from coverage, 12m volatility, and analyst target dispersion; High widens both Buy hurdles
- **Short-interest flag** — Yahoo/FINRA days-to-cover and short % of float (overlay only, not a weighted factor)
- **Fund view (ETFs & mutual funds, US + Canada)** — full dashboard with a fund composite score, factor scorecard, radar, and price history, scored against a peer universe of well-known US and Canadian funds
- **Weekly email** — Monday watchlist scorecard (composite, bargain, upside, Buy/Not Buy)
- **Monthly email** — full S&P 500 scorecard (1st of each month)

## Quick start (local)

```bash
cd financial-tools
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Build initial universe snapshot (use --fast for ~50 tickers, or full S&P 500)
python -m core.universe --fast --max 50

# Build the fund universe snapshot (US + Canadian ETFs and mutual funds)
python -m core.fund_universe

# Launch dashboard
streamlit run app.py
```

## Fund scoring (ETFs & mutual funds)

Stock factors depend on company financial statements and analyst coverage that
do not exist for pooled funds, so funds are **not** squeezed into the stock
model. Instead they get their own composite built from six fund factor groups,
each ranked cross-sectionally against a curated peer universe of ~170 US and
Canadian ETFs and mutual funds (within fund category when the category is
large enough):

| Group | Signal | Default weight |
|-------|--------|----------------|
| `cost` | Expense ratio (inverted — fees are the one robust predictor of relative fund performance, so cost dominates) | 35% |
| `performance` | 3y and 5y annualized total returns (down-weighted: past fund returns are a weak predictor after fees) | 15% |
| `risk_adjusted` | Trailing 1y return / annualized volatility (no risk-free adjustment; a coarse ratio, not a true Sharpe) | 15% |
| `low_volatility` | Inverse 12m volatility + max-drawdown protection | 15% |
| `momentum` | 12-1 month return | 10% |
| `income` | Distribution yield (a payout preference, not a return signal) | 10% |

Weights live under `fund_factor_weights` in `config.yaml`. Metrics that can't
be converted from the stock model (Graham margin of safety, balance-sheet
strength, analyst consensus) are intentionally excluded rather than
approximated, and the coverage figure shows which groups had data.

Canadian listings use Yahoo's `.TO` suffix (e.g. `XEQT.TO`, `VFV.TO`) and are
displayed in C$. Canadian mutual funds are carried by Yahoo under
Morningstar-style IDs (e.g. `0P0000A0F2.TO` for RBC Balanced D) — see
`core/fund_universe.py` for the curated list of verified symbols.

## Configuration

Edit [`config.yaml`](config.yaml):

| Section | Purpose |
|---------|---------|
| `thresholds` | Buy rules (composite, bargain, exclude sell; upside is informational) |
| `factor_weights` | Weight each factor family in composite score |
| `fund_factor_weights` | Weight each fund factor group in the fund composite |
| `bargain_weights` | Long-horizon valuation bargain components |
| `email` | SMTP settings (prefer env vars / GitHub Secrets for addresses) |

**Watchlist:** edit the [`watchlist`](watchlist) file at the repo root — one ticker per line (`#` for comments). This file is used by the weekly watchlist job.

Default buy rule: `composite >= threshold` AND `bargain >= threshold` AND factor
coverage ≥ 70% AND Altman Z ≥ 1.8 (no distress) AND consensus is not Sell (see
`config.yaml` for current values, which are written by
`python -m backtest.run apply` from committed calibration artifacts in
`backtest/results/`). Analyst implied upside is shown for context
but is not a hard gate. The coverage gate prevents stocks with sparse financial
data from passing on a composite renormalized over only a few factor groups.
The Altman gate treats distress non-linearly: a linear composite would only
nudge a distress-zone stock down a few points, so it is excluded outright.
Missing Z never blocks, and Financial Services / Real Estate / Utilities are
exempt (the classic 1968 Z model was built for industrials and misclassifies
those sectors' structural leverage as distress).

## Email alerts setup

### 1. Edit your watchlist

```text
# watchlist
AAPL
MSFT
NVDA
```

### 2. Configure SMTP

**Option A — local / cron:** set environment variables:

```bash
export SMTP_HOST=smtp.gmail.com
export SMTP_PORT=587
export SMTP_FROM=you@gmail.com
export SMTP_TO=you@gmail.com
export SMTP_USERNAME=you@gmail.com
export SMTP_PASSWORD=your-gmail-app-password
```

**Option B — GitHub Actions:** add the same values as repository secrets (Settings → Secrets and variables → Actions):

| Secret | Example |
|--------|---------|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_FROM` | your Gmail address |
| `SMTP_TO` | alert recipient |
| `SMTP_USERNAME` | same as FROM |
| `SMTP_PASSWORD` | Gmail [app password](https://support.google.com/accounts/answer/185833) |

Also set `email.enabled: true` in `config.yaml` (or rely on `SMTP_PASSWORD` being set).

### 3. Run jobs manually (test first)

```bash
# Weekly watchlist scorecard (use --no-email to dry-run)
python jobs/watchlist_weekly.py --no-email

# Monthly S&P 500 scorecard (slow — full universe; use --fast --max 50 for dev)
python jobs/universe_monthly.py --no-email --fast --max 50
```

When ready, omit `--no-email` to send the HTML scorecard to your inbox.

### 4. Schedule automatically

**GitHub Actions** (included in repo):

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `.github/workflows/watchlist-weekly.yml` | Mondays 14:00 UTC | Watchlist scorecard email |
| `.github/workflows/universe-monthly.yml` | 1st of month 14:00 UTC | Full S&P 500 scorecard + stock and fund snapshot refresh |

Adjust cron times in the workflow files for your timezone (14:00 UTC ≈ 7:00 AM Pacific).

**Local cron (macOS/Linux)** example:

```cron
0 7 * * 1 cd /path/to/financial-tools && .venv/bin/python jobs/watchlist_weekly.py
0 7 1 * * cd /path/to/financial-tools && .venv/bin/python jobs/universe_monthly.py
```

Each email includes a table with **Composite**, **Bargain**, **Upside**, and **Buy / Not Buy** for every ticker. Buys are sorted to the top.

## Deploy

### Streamlit Community Cloud

1. Push repo to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → select repo, main file `app.py`
3. Ensure `data/universe_snapshot.parquet` and `data/fund_universe_snapshot.parquet` are committed (monthly GHA job refreshes them)

## Scheduled jobs

```bash
python jobs/watchlist_weekly.py    # weekly watchlist scorecard
python jobs/universe_monthly.py   # monthly full S&P 500 (slow)
```

Options for both:
- `--refresh` (watchlist job only) — rebuild universe snapshot first (slow; monthly job does this by default)
- `--no-email` — skip email
- `--fast` — smaller fallback universe (faster, good for dev)
- `--max-universe N` / `--max N` — limit tickers processed

- **yfinance** — prices, fundamentals, analyst recommendations, price targets
- **Wikipedia** — S&P 500 constituent list
- **OpenBB** (optional) — unified wrapper when installed

## Project structure

```
core/           # data fetch, factors (stock + fund), scoring, analysts, universes
app.py          # Streamlit dashboard
jobs/           # watchlist_weekly.py, universe_monthly.py, email_sender.py
watchlist       # your watchlist (one ticker per line)
config.yaml     # thresholds, weights (stock + fund), email
data/           # universe_snapshot.parquet + fund_universe_snapshot.parquet
```

## Historical backtest (validation harness)

The `backtest/` package validates composite factor weights, bargain weights, and
good-buy thresholds using SEC EDGAR point-in-time fundamentals and yfinance
prices, covering historical S&P 500 constituents from 2010 to 2026.

**How parameters are set (buy-and-hold aligned):**

- Factor weights use **evidence-based priors** (quality/value-led). The harness
  compares a small set of named candidates (`evidence_based`, `legacy_tuned`,
  `equal`) on gated DCA buy-and-hold performance with ~10 bps transaction costs
  and bootstrap confidence intervals — it does **not** search for overfit weights.
- Evaluation uses **1y / 3y / 5y** forward excess returns (plus next-quarter IC).
- Scoring matches live: sector-adjusted empirical percentile ranks (the backtest
  engine calls the same `compute_family_percentile` used by the dashboard).
- Bargain weights (Graham margin of safety, valuation vs own 10y history, 52w discount)
  are validated via long-horizon rank IC; the applied `graham_heavy` weights
  (0.55/0.30/0.15) roughly doubled the 3y/5y bargain IC vs the old 0.40/0.35/0.25.
  Live bargain history now uses the same EDGAR store as the backtest (EBIT/EV,
  OCF yield, book-to-market; pick the series that tracked price, else average).
- Estimate revisions and Form 4 insider buying are **live-only**. They are
  excluded from the historical composite and the remaining groups are
  renormalized — those signals have no reconstructed point-in-time panel here.
- Good-buy thresholds are calibrated on **3-year** forward excess-return buckets.

### What the backtest actually shows

Judge the tool by the **out-of-sample fold statistics**, not by any full-period
terminal-wealth simulation (those evaluate parameters on the same window used
to choose them, and free price data excludes delisted tickers entirely, so
strategy results are survivorship-biased upward).

The honest summary from the expanding-window folds (see
`backtest/results/weight_candidate_comparison.json`):

- 3-year rank IC of the composite is small (~0.07 across candidates; ~0.08 at
  5y under the rank-percentile scoring).
- Mean 3-year excess return of the gated DCA strategy vs SPY has a bootstrap
  CI that **includes zero** — no candidate is statistically distinguishable
  from the index or from the other candidates.
- Bargain-score rank ICs improved with the `graham_heavy` weights (~0.016 at
  3y, ~0.022 at 5y, up from ~0.008/0.012) but remain small, so the bargain
  gate is best understood as an entry-discipline heuristic, not a validated
  return predictor.

Known limitations: ~175 delisted historical S&P 500 members (bankruptcies and
acquisitions) have no free price history and can never be selected by the
simulation; 2010–2026 is a single, mostly-bull regime with few independent
3-year windows; live fundamentals come from Yahoo while backtest fundamentals
come from EDGAR. Fixing the survivorship gap requires paid data (e.g. Norgate,
Sharadar, or EODHD delisted coverage).

```bash
# Full validation pipeline (slow: hours for complete SEC + price ingest)
python -m backtest.run pipeline

# Individual steps
python -m backtest.run ingest
python -m backtest.run build-factors
python -m backtest.run compare       # primary: named candidate comparison (recommended)
python -m backtest.run calibrate     # thresholds on 3y forward excess
python -m backtest.run dca
python -m backtest.run report
python -m backtest.run apply         # write validated weights/thresholds to config.yaml

# Legacy search (kept available, not the source of truth)
python -m backtest.run tune
python -m backtest.run tune-cv
```

Quick dev run (limited quarters/tickers):

```bash
python -m backtest.run pipeline --max-edgar-quarters 8 --max-quarters 8 --max-tickers 80
```

Results are written to `backtest/results/` (report, comparison JSON, DCA validation).
Cached data lives in `backtest/data/store/` (gitignored).

## Spot-checking a snapshot

After building a snapshot, sanity-check a few names with known profiles:
mature dividend payers (e.g. KO) should rank high on value, recent runners
(e.g. NVDA) high on momentum but low on value, and a balanced mega-cap
(e.g. AAPL) should land mid-pack on the composite. A stock can show high
analyst upside while still scoring poorly on value/quality — upside alone
never forces a Buy. Validate `thresholds` and `factor_weights` via
`python -m backtest.run compare` or edit `config.yaml` directly.

To rebuild with more tickers for better cross-sections:

```bash
python -m core.universe --max 100   # or full S&P 500 without --max
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

## Disclaimer

This tool is for informational purposes only. Not investment advice.
