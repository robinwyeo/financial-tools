# Stock & Fund Metrics Tool

[![CI](https://github.com/robinwyeo/financial-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/robinwyeo/financial-tools/actions/workflows/ci.yml)

Satellite-sleeve screener for a **buy-and-hold** investor whose core already
lives in broad ETFs. The stock path is a **quality screen plus intrinsic
value**, not a composite-rank trading system. Funds (US and Canadian ETFs and
mutual funds) keep a separate fee-first composite.

This is **not** a portfolio manager, not tax software, and not a claim of
alpha. Keep a low-cost index core.

## What the tool is

For a ticker you get:

1. **Layer A — quality** (relative, sector-aware): profitability, earnings
   quality, financial strength, stability, capital discipline.
2. **Layer B — hard gates** (absolute): Altman Z'' distress, value-trap flags,
   data-quality grade C, quality percentile floor.
3. **Layer C — valuation** (absolute, per share): owner-earnings DCF (base and
   bear), earnings-power value, reverse DCF, expected return vs hurdle,
   **buy-below price**.
4. **Decision:** Accumulate / Watch / Avoid. `is_good_buy` is true only for
   Accumulate (jobs and the fund path keep that boolean).
5. **Timing context only:** momentum, estimate revisions, insider cluster
   buys, RSI, 52-week discount — shown, not used as a gate.

Legacy composite + bargain thresholds remain available with
`decision.mode: legacy` in `config.yaml`.

## Valuation assumptions (and how to change them)

All knobs live under `valuation:` and `decision:` in [`config.yaml`](config.yaml).

| Input | Default | Where |
|-------|---------|--------|
| Hurdle | `max(8%, 10y Treasury + 4.5% ERP)`, +1% if uncertainty is High | `valuation.hurdle` |
| ERP | 4.5% (Damodaran implied ERP; update yearly from `data/reference/damodaran_erp.csv`) | `valuation.hurdle.erp` |
| Fallback 10y yield if no FRED key | 4.2% → hurdle 8.7% | `valuation.hurdle.fallback_rf` |
| Terminal growth | 2.5% | `valuation.terminal_growth` |
| Explicit years / fade start | 10 / year 5 | `explicit_years`, `fade_start_year` |
| Historical growth clip | 0–15% | `base_growth_floor`, `base_growth_cap` |
| Bear case | g1 × 0.5 capped at 4%, r + 1% | `bear_*` |
| Subtract SBC from owner earnings | true | `subtract_sbc` |
| Margin of safety | Low 20% / Medium 30% / High 40% | `decision.required_margin_of_safety` |
| Buy-below | `min(DCF_base × (1 − MoS), DCF_bear)` | `core/decision.py` |

Growth used in the **decision** is 5-year historical owner-earnings-per-share
CAGR. Analyst 5-year growth is display-only so live and backtest stay aligned.
Financials skip DCF/EPV (`relative_only`: P/B vs ROE, P/E vs history).

Set `FRED_API_KEY` (free) for live 10y yields. Without it the fallback RF is
used. Optional `FINNHUB_API_KEY` enables a secondary estimates provider;
everything runs without it.

## Data sources

| Source | Used for | Reliability notes |
|--------|----------|-------------------|
| SEC companyfacts | 10y annual + TTM fundamentals, PIT by filed date, consolidated only | Shared by live and backtest. Canadian non-filers skip EDGAR (grade capped at B). |
| Yahoo Finance (yfinance) | Price, TTM stmts, MRQ balance sheet, analysts, short interest, funds | Fallback when EDGAR is empty. Statement mix is reconciled and graded. |
| SEC Form 4 | Insider cluster-buy overlay (officer/director open-market P; 10b5-1 excluded) | Live-only; no historical panel. |
| FRED | DGS10 (hurdle), BAA10Y, T10YIE | Cached 24h (series) / 7d (history). Optional API key. |
| Damodaran files | ERP and industry WACC reference | Yearly; `python jobs/refresh_reference.py --from-file …` |
| Wikipedia | S&P 500 / 400 and TSX 60 constituents | Live universe only. Backtest stays S&P 500 (no free PIT mid-cap/TSX history). |

Paid survivorship-free prices (Sharadar etc.) are **not** in use. See
[`docs/paid_data.md`](docs/paid_data.md).

## Decision layers in one page

**Accumulate** if Layer B passes, price ≤ buy-below, and expected return ≥ hurdle
(financials: relative percentile ≥ 50 instead of DCF).

**Watch** if Layer B passes but the price is still above buy-below — the UI and
email show the Accumulate price and % decline needed.

**Avoid** on distress, too many value-trap flags, quality below the configured
percentile, or data-quality grade C.

## Quick start (local)

```bash
cd financial-tools
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.lock   # or requirements.txt

# Stock universe (config universe.members, default S&P 500)
python -m core.universe --fast --max 50

# Fund universe (US + Canadian ETFs and mutual funds)
python -m core.fund_universe

# Dashboard — run from the repo root
streamlit run app.py
```

Python 3.11–3.12 is what CI pins (`requirements.lock`). A 3.14 venv works
locally with the numpy extra marker in the lockfile.

## Fund scoring (ETFs & mutual funds)

Funds are **not** forced through the stock model. Six groups, ranked against a
peer universe of US and Canadian ETFs and mutual funds:

| Group | Signal | Default weight |
|-------|--------|----------------|
| `cost` | Expense ratio (inverted) | 35% |
| `performance` | 3y and 5y annualized total returns | 15% |
| `risk_adjusted` | Trailing 1y return / volatility | 15% |
| `low_volatility` | Inverse 12m vol + max-drawdown protection | 15% |
| `momentum` | 12-1 month return | 10% |
| `income` | Distribution yield | 10% |

Canadian listings use Yahoo `.TO` suffixes and display in C$.

## Configuration

Edit [`config.yaml`](config.yaml):

| Section | Purpose |
|---------|---------|
| `universe.members` | Live cross-section: `sp500` (default); add `sp400`, `tsx60` |
| `thresholds` | Legacy composite/bargain gates; Z'' cutoff; sell/underperform filter |
| `factor_weights` | Legacy composite group weights |
| `quality_weights` / `value_trap` | Layer A/B |
| `valuation` / `decision` | Layer C and Accumulate/Watch/Avoid |
| `providers.fundamentals` | Adapter order (`edgar`, `yahoo`; optional `finnhub`) |
| `fund_factor_weights` | Fund composite |
| `email` | SMTP (prefer env vars / GitHub Secrets) |

**Watchlist:** [`watchlist`](watchlist) at repo root, one ticker per line.

## Email jobs

```bash
python jobs/watchlist_weekly.py --no-email
python jobs/universe_monthly.py --no-email --fast --max 50
python jobs/refresh_reference.py --from-file path/to/histimpl.csv --kind erp
```

Weekly and monthly GitHub Actions install from `requirements.lock`, write
`data/runs/<timestamp>.json`, and fail the job if more than 10% of tickers
error. Emails sort Accumulate first, then Watch by % to buy, and include Why,
buy-below, grade, snapshot date, run id, and hurdle.

SMTP secrets: `SMTP_FROM`, `SMTP_TO`, `SMTP_PASSWORD` (and optional
`SMTP_HOST` / `SMTP_PORT`). Optional `FRED_API_KEY`.

## What the backtest actually shows

<!-- evidence:start -->

### What the backtest actually shows

Judge the tool by the **out-of-sample fold statistics**, not by any full-period
terminal-wealth simulation (those evaluate parameters on the same window used
to choose them, and free price data excludes delisted tickers entirely, so
strategy results are survivorship-biased upward).

Evidence `run_id`: `2026-09-07T06:30:02Z`.

| Candidate | 3y IC | NW t | n independent windows | Block ROI |
| --- | ---: | ---: | ---: | ---: |
| evidence_based | 0.000 | — | 0.0 | -0.00% |
| legacy_tuned | 0.000 | — | 0.0 | -0.00% |
| equal | 0.000 | — | 0.0 | -0.00% |

Block ROI is excess return inside each ≤2y walk-forward block, not 3-year excess.

Mean PIT constituent price coverage: **83.7%**.

Full-period wealth tables are **in-sample and survivorship-biased**.
Do not treat them as alpha.

<!-- evidence:end -->

```bash
python -m backtest.run pipeline          # hours; local
python -m backtest.run evaluate --metric quality_score
python -m backtest.run evaluate-signal --signal decision
python -m backtest.run apply             # refuses in-sample writes without --allow-in-sample
```

Results: `backtest/results/`. Cached store: `backtest/data/store/` (gitignored).

## Project structure

```
app.py                 # set_page_config + main()
ui/                    # layout, charts, stock/fund views, rankings, sidebar, cache
core/                  # fundamentals, rates, valuation, quality, decision, scoring, providers
jobs/                  # weekly watchlist, monthly universe, reference refresh, email
backtest/              # companyfacts ingest, factors, engine, registry, report
tests/
docs/ARCHITECTURE.md
docs/methodology.md
docs/paid_data.md
config.yaml
data/reference/        # SIC map, Damodaran ERP / industry tables
```

## Tests

```bash
pip install -r requirements.lock
pytest tests/ -q
pytest tests/ -q --cov=core --cov=jobs
```

## More

- Architecture diagram and module map: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Formulas and citations: [`docs/methodology.md`](docs/methodology.md)
- Streamlit Cloud notes: [`DEPLOY.md`](DEPLOY.md)

## Disclaimer

Informational only. Not investment advice.
