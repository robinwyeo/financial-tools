# Stock Metrics & Analyst Aggregation Tool

Empirical factor scoring and aggregated analyst recommendations for stocks, with a daily email alert for watchlist names that meet your "good buy" threshold.

## Features

- **Streamlit dashboard** — enter a ticker for factor scorecard, analyst consensus, price targets, and implied upside
- **Extended factor set** — Value, Momentum (12-1), Quality, Low Volatility, Investment, Earnings Revisions, Piotroski F-Score
- **Cross-sectional scoring** — percentile ranks vs S&P 500 universe (sector-adjusted when enabled)
- **ETF view** — basic fund info (no factor scoring)
- **Daily GitHub Actions job** — refreshes universe snapshot and emails good-buy alerts

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
| `watchlist` | Tickers checked daily |
| `thresholds` | Good-buy rules (composite, upside %, exclude sell) |
| `factor_weights` | Weight each factor family in composite score |
| `email` | SMTP settings (or use env vars / GitHub Secrets) |

Default good-buy rule: `composite >= 70` AND `implied_upside >= 15%` AND consensus is not Sell.

## Daily job

```bash
python jobs/daily_check.py --fast --max-universe 50
```

Options:
- `--no-refresh` — skip universe rebuild
- `--no-email` — skip email
- `--fast` — smaller fallback universe (faster, good for dev)

## Deploy

### Streamlit Community Cloud

1. Push repo to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → select repo, main file `app.py`
3. Ensure `data/universe_snapshot.parquet` is committed (daily GHA refreshes it)

### GitHub Actions (daily alerts)

Add repository secrets:

| Secret | Example |
|--------|---------|
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_FROM` | your Gmail address |
| `SMTP_TO` | alert recipient |
| `SMTP_USERNAME` | same as FROM |
| `SMTP_PASSWORD` | Gmail [app password](https://support.google.com/accounts/answer/185833) |

Enable email in `config.yaml` (`email.enabled: true`) or rely on `SMTP_PASSWORD` being set.

Workflow runs weekdays at 06:30 UTC (`.github/workflows/daily.yml`).

## Data sources (free)

- **yfinance** — prices, fundamentals, analyst recommendations, price targets
- **Wikipedia** — S&P 500 constituent list
- **OpenBB** (optional) — unified wrapper when installed

## Project structure

```
core/           # data fetch, factors, scoring, analysts, universe
app.py          # Streamlit dashboard
jobs/           # daily_check.py, email_sender.py
config.yaml     # watchlist, thresholds, weights
data/           # universe_snapshot.parquet (refreshed daily)
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

## Disclaimer

This tool is for informational purposes only. Not investment advice.
