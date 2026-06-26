# Stock Metrics & Analyst Aggregation Tool

Empirical factor scoring and aggregated analyst recommendations for stocks, with weekly watchlist and monthly S&P 500 scorecard emails.

## Features

- **Streamlit dashboard** — enter a ticker for factor scorecard, analyst consensus, price targets, and implied upside
- **Extended factor set** — Value, Momentum (12-1), Quality, Low Volatility, Investment, Earnings Revisions, Piotroski F-Score
- **Cross-sectional scoring** — percentile ranks vs S&P 500 universe (sector-adjusted when enabled)
- **Bargain score** — absolute cheapness signal (margin of safety, drawdown, RSI, upside)
- **ETF view** — basic fund info (no factor scoring)
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

# Launch dashboard
streamlit run app.py
```

## Configuration

Edit [`config.yaml`](config.yaml):

| Section | Purpose |
|---------|---------|
| `thresholds` | Buy rules (composite, bargain, upside %, exclude sell) |
| `factor_weights` | Weight each factor family in composite score |
| `email` | SMTP settings (or use env vars / GitHub Secrets) |

**Watchlist:** edit the [`watchlist`](watchlist) file at the repo root — one ticker per line (`#` for comments). This file is used by the weekly watchlist job.

Default buy rule: `composite >= 50.9` AND `bargain >= 43.2` AND `implied_upside >= 15%` AND consensus is not Sell. These thresholds were calibrated by the backtest harness.

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
python jobs/daily_check.py --no-email

# Monthly S&P 500 scorecard (slow — full universe; use --fast --max 50 for dev)
python jobs/weekly_check.py --no-email --fast --max 50
```

When ready, omit `--no-email` to send the HTML scorecard to your inbox.

### 4. Schedule automatically

**GitHub Actions** (included in repo):

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `.github/workflows/daily.yml` | Mondays 14:00 UTC | Watchlist scorecard email |
| `.github/workflows/weekly.yml` | 1st of month 14:00 UTC | Full S&P 500 scorecard + snapshot refresh |

Adjust cron times in the workflow files for your timezone (14:00 UTC ≈ 7:00 AM Pacific).

**Local cron (macOS/Linux)** example:

```cron
0 7 * * 1 cd /path/to/financial-tools && .venv/bin/python jobs/daily_check.py
0 7 1 * * cd /path/to/financial-tools && .venv/bin/python jobs/weekly_check.py
```

Each email includes a table with **Composite**, **Bargain**, **Upside**, and **Buy / Not Buy** for every ticker. Buys are sorted to the top.

## Deploy

### Streamlit Community Cloud

1. Push repo to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → select repo, main file `app.py`
3. Ensure `data/universe_snapshot.parquet` is committed (monthly GHA job refreshes it)

## Scheduled jobs

```bash
python jobs/daily_check.py    # weekly watchlist scorecard
python jobs/weekly_check.py   # monthly full S&P 500 (slow)
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
core/           # data fetch, factors, scoring, analysts, universe
app.py          # Streamlit dashboard
jobs/           # daily_check.py, weekly_check.py, email_sender.py
watchlist       # your watchlist (one ticker per line)
config.yaml     # thresholds, weights, email
data/           # universe_snapshot.parquet (refreshed by jobs)
```

## Historical backtest (weight & threshold tuning)

The `backtest/` package tunes composite factor weights, bargain weights, and
good-buy thresholds using SEC EDGAR point-in-time fundamentals and yfinance
prices, covering historical S&P 500 constituents from 2010 to 2026.

**How parameters are set:** factor weights are tuned with a k-fold
time-series cross-validation that runs the actual DCA strategy — each quarter
invest $20k equally across the top-5 composite-ranked stocks, hold until the
end of the fold, and compare ROI on deployed capital against a same-schedule
SPY DCA. Candidates are ranked by `mean(excess ROI across folds) − std(excess
ROI)`, so the search explicitly rewards consistency across regimes rather than
spiking in one period. The winner is only adopted if it is strictly more robust
than the existing baseline. Good-buy thresholds are calibrated separately by
bucketing composite/bargain scores against historical forward returns. Bargain
weights are tuned by maximising the rank IC of the bargain score against
next-quarter returns.

Current factor weights reflect `sample_134`, the k-fold CV winner from a
200-candidate search over 5 folds (2010–2026), which beat SPY on 4 of 5 folds
including the 2023–26 period where the prior baseline lost −14%.

```bash
# Full pipeline (slow: hours for complete SEC + price ingest)
python -m backtest.run pipeline

# Individual steps
python -m backtest.run ingest
python -m backtest.run build-factors
python -m backtest.run tune          # rolling win-rate objective (with walk-forward splits)
python -m backtest.run tune-cv       # DCA k-fold CV objective (recommended)
python -m backtest.run calibrate
python -m backtest.run dca
python -m backtest.run report
python -m backtest.run apply                  # apply default tune results
python -m backtest.run apply --use-dca-cv     # apply k-fold CV winner (factor weights only)
```

Quick dev run (limited quarters/tickers):

```bash
python -m backtest.run pipeline --max-edgar-quarters 8 --max-quarters 8 --max-tickers 80 --n-samples 30
```

Results are written to `backtest/results/` (report, tuning JSON, DCA validation).
Cached data lives in `backtest/data/store/` (gitignored).

## Calibration (sample run, 30-ticker universe)

Spot-check results after building the snapshot:

| Ticker | Expected signal | Sample result |
|--------|-----------------|---------------|
| KO | High value | ~88th percentile value |
| NVDA | High momentum / revisions | ~73rd momentum, ~91st earnings revisions |
| JPM | Moderate value, strong momentum | ~31st value, ~93rd momentum |
| AAPL | Balanced mega-cap | ~48 composite (below default 50 threshold) |

NVDA shows high analyst upside (~40%) but composite below 50 due to weak value/low-vol scores — the default rule correctly avoids flagging it as a "good buy" on factors alone. Tune `thresholds` and `factor_weights` via the backtest harness or edit `config.yaml` directly.

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
