# Stock & Fund Metrics Tool

Empirical factor scoring and aggregated analyst recommendations for stocks — plus fund-appropriate factor scoring for US and Canadian ETFs and mutual funds — with weekly watchlist and monthly S&P 500 scorecard emails.

## Features

- **Streamlit dashboard** — enter a ticker for factor scorecard, analyst consensus, price targets, and implied upside
- **Extended factor set** — Value, Momentum (12-1), Quality, Low Volatility, Investment, Earnings Revisions, Piotroski F-Score
- **Cross-sectional scoring** — percentile ranks vs S&P 500 universe (sector-adjusted when enabled)
- **Bargain score** — long-horizon cheapness (Graham margin of safety, valuation vs own 5y history, 52-week discount)
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
| `cost` | Expense ratio (inverted — fees are the strongest predictor of relative fund performance) | 25% |
| `performance` | 3y and 5y annualized total returns | 20% |
| `risk_adjusted` | Trailing 1y return / annualized volatility (Sharpe-style) | 20% |
| `low_volatility` | Inverse 12m volatility + max-drawdown protection | 15% |
| `momentum` | 12-1 month return | 10% |
| `income` | Distribution yield | 10% |

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

Default buy rule: `composite >= 57.3` AND `bargain >= 49.4` AND consensus is not Sell. Analyst implied upside is shown for context but is not a hard gate. Thresholds were recalibrated on 3-year forward excess returns.

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
| `.github/workflows/weekly.yml` | 1st of month 14:00 UTC | Full S&P 500 scorecard + stock and fund snapshot refresh |

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
3. Ensure `data/universe_snapshot.parquet` and `data/fund_universe_snapshot.parquet` are committed (monthly GHA job refreshes them)

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
core/           # data fetch, factors (stock + fund), scoring, analysts, universes
app.py          # Streamlit dashboard
jobs/           # daily_check.py, weekly_check.py, email_sender.py
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
- Scoring matches live: sector-adjusted z-scores when sector data is available.
- Bargain weights (Graham margin of safety, valuation vs own history, 52w discount)
  are validated via long-horizon rank IC.
- Good-buy thresholds are calibrated on **3-year** forward excess-return buckets.

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

## Calibration (sample run, 30-ticker universe)

Spot-check results after building the snapshot:

| Ticker | Expected signal | Sample result |
|--------|-----------------|---------------|
| KO | High value | ~88th percentile value |
| NVDA | High momentum / revisions | ~73rd momentum, ~91st earnings revisions |
| JPM | Moderate value, strong momentum | ~31st value, ~93rd momentum |
| AAPL | Balanced mega-cap | ~48 composite (below default 50 threshold) |

NVDA can show high analyst upside while still scoring poorly on value/quality — upside alone no longer forces a Buy. Validate `thresholds` and `factor_weights` via `python -m backtest.run compare` or edit `config.yaml` directly.

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
