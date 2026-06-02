# Stock Metrics & Analyst Aggregation Tool

Empirical factor scoring and aggregated analyst recommendations for stocks, with daily watchlist and weekly S&P 500 scorecard emails.

## Features

- **Streamlit dashboard** — enter a ticker for factor scorecard, analyst consensus, price targets, and implied upside
- **Extended factor set** — Value, Momentum (12-1), Quality, Low Volatility, Investment, Earnings Revisions, Piotroski F-Score
- **Cross-sectional scoring** — percentile ranks vs S&P 500 universe (sector-adjusted when enabled)
- **Bargain score** — absolute cheapness signal (margin of safety, drawdown, RSI, upside)
- **ETF view** — basic fund info (no factor scoring)
- **Daily email** — morning scorecard for your watchlist (composite, bargain, upside, Buy/Not Buy)
- **Weekly email** — Monday full S&P 500 scorecard

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

**Watchlist:** edit the [`watchlist`](watchlist) file at the repo root — one ticker per line (`#` for comments). This file is used by the daily job.

Default buy rule: `composite >= 50` AND `bargain >= 50` AND `implied_upside >= 15%` AND consensus is not Sell.

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
# Daily watchlist scorecard (use --no-email to dry-run)
python jobs/daily_check.py --no-email

# Weekly S&P 500 scorecard (slow — full universe; use --fast --max 50 for dev)
python jobs/weekly_check.py --no-email --fast --max 50
```

When ready, omit `--no-email` to send the HTML scorecard to your inbox.

### 4. Schedule automatically

**GitHub Actions** (included in repo):

| Workflow | Schedule | What it does |
|----------|----------|--------------|
| `.github/workflows/daily.yml` | Every day 14:00 UTC | Watchlist scorecard email |
| `.github/workflows/weekly.yml` | Mondays 14:00 UTC | Full S&P 500 scorecard email |

Adjust cron times in the workflow files for your timezone (14:00 UTC ≈ 7:00 AM Pacific).

**Local cron (macOS/Linux)** example:

```cron
0 7 * * * cd /path/to/financial-tools && .venv/bin/python jobs/daily_check.py
0 7 * * 1 cd /path/to/financial-tools && .venv/bin/python jobs/weekly_check.py
```

Each email includes a table with **Composite**, **Bargain**, **Upside**, and **Buy / Not Buy** for every ticker. Buys are sorted to the top.

## Deploy

### Streamlit Community Cloud

1. Push repo to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → select repo, main file `app.py`
3. Ensure `data/universe_snapshot.parquet` is committed (daily GHA refreshes it)

## Daily / weekly jobs

```bash
python jobs/daily_check.py    # watchlist scorecard
python jobs/weekly_check.py   # full S&P 500 (slow)
```

Options for both:
- `--refresh` (daily only) — rebuild universe snapshot first (slow; weekly job does this by default)
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
watchlist       # your daily watchlist (one ticker per line)
config.yaml     # thresholds, weights, email
data/           # universe_snapshot.parquet (refreshed by jobs)
```

## Calibration (sample run, 30-ticker universe)

Spot-check results after building the snapshot:

| Ticker | Expected signal | Sample result |
|--------|-----------------|---------------|
| KO | High value | ~88th percentile value |
| NVDA | High momentum / revisions | ~73rd momentum, ~91st earnings revisions |
| JPM | Moderate value, strong momentum | ~31st value, ~93rd momentum |
| AAPL | Balanced mega-cap | ~48 composite (below default 70 threshold) |

NVDA shows high analyst upside (~40%) but composite below 70 due to weak value/low-vol scores — the default rule correctly avoids flagging it as a "good buy" on factors alone. Tune `thresholds` and `factor_weights` to match your strategy.

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
