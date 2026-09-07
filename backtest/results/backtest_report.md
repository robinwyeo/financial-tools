# Score Weight & Threshold Backtest Report

Generated: 2026-09-07 11:18 UTC

## Summary

This report validates composite/bargain score weights and good-buy thresholds
using long-horizon (1y/3y/5y) forward returns and gated DCA buy-and-hold
simulations on historical S&P 500 constituents.

## Caveats

- Survivorship bias: delisted tickers absent from free price data cannot be selected at all, so strategy results are biased upward regardless of the delist return assumption.
- Analyst upside is informational (not a hard gate) and not historically backtested.
- Expanding-window folds + bootstrap CIs reduce but do not eliminate path dependence.
- Scoring uses SIC→sector (SEC submissions) rather than today's live snapshot.
- SEC EDGAR fundamentals are point-in-time by filing date; reporting lags apply.

## Named Weight Candidate Comparison

Recommended: **evidence_based**
Primary horizon: **3y**

| Candidate | 3y IC | NW t | n windows | Block ROI | 95% CI | % blocks > 0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| evidence_based | 0.089 | 4.52 | 4.4 | 7.88% | [-0.19%, 18.78%] | 67% |
| equal | 0.097 | 6.65 | 4.4 | 6.73% | [0.27%, 13.70%] | 83% |
| legacy_tuned | 0.087 | 7.87 | 4.4 | 2.36% | [-2.67%, 9.04%] | 50% |

Block ROI is excess return of the gated DCA campaign inside each walk-forward block (≤ 2 years), not 3-year excess.
- **evidence_based** vs **equal**: indistinguishable (bootstrap CIs overlap).
- **evidence_based** vs **legacy_tuned**: indistinguishable (bootstrap CIs overlap).

### Recommended factor weights

```yaml
  balance_sheet: 0.1000
  capital_discipline: 0.1000
  garp: 0.0500
  low_volatility: 0.0500
  momentum: 0.1000
  quality: 0.2250
  value: 0.2500
```

## Bargain Weight Validation

- Winner: **default_long_horizon**
- Winner mean IC (3y): -0.051
- Baseline mean IC: -0.051

### Recommended bargain weights

```yaml
  discount_52w: 0.1500
  margin_of_safety: 0.5500
  valuation_vs_history: 0.3000
```

## Threshold Calibration

- Horizon: **3y**
- composite_min: **59.7**
- bargain_min: **50.0**

## Investment Comparison vs S&P 500

Each quarter, **$20,000** was invested into the top 5 good-buy stocks (old or new parameters). The benchmark invests the same **$20,000/quarter** into **SPY** (S&P 500 total return) on the identical schedule.

- Simulation period: **2010-03-31 00:00:00** → **2026-03-31 00:00:00**
- Quarters with investment: **65**
- Total capital deployed (each strategy): **$1,300,000**

### Side-by-side results (in-sample, survivorship-biased)

| Metric | Old params | New params | S&P 500 (SPY) |
| --- | ---: | ---: | ---: |
| Terminal wealth | $11,167,421 | $16,186,724 | $4,326,313 |
| Total invested | $1,300,000 | $1,300,000 | $1,300,000 |
| Total return | 759.0% | 1145.1% | 232.8% |
| CAGR | 14.4% | 17.1% | 7.8% |
| Max drawdown | -28.4% | -33.8% | -21.9% |

### Outperformance vs S&P 500

| Comparison | Terminal wealth Δ | Return Δ | CAGR Δ | Higher ROI? |
| --- | ---: | ---: | ---: | :---: |
| Old params vs SPY | +$6,841,108 (+158.1%) | +526.2% | +6.6% | Yes |
| New params vs SPY | +$11,860,411 (+274.1%) | +912.3% | +9.3% | Yes |
| New params vs Old params | +$5,019,303 (+44.9%) | +386.1% | +2.7% | Yes |

### Interpretation

- **Old parameters** finished +$6,841,108 ahead of SPY in terminal wealth; return on deployed capital was 759.0% (14.4% CAGR) vs SPY's 232.8% (7.8% CAGR).
- **New parameters** finished +$11,860,411 ahead of SPY in terminal wealth; return on deployed capital was 1145.1% (17.1% CAGR) vs SPY's 232.8% (7.8% CAGR).
- **New parameters** beat old parameters by +$5,019,303 terminal wealth and +386.1% higher return on deployed capital.

### Survivorship (price coverage of PIT constituents)

Mean quarter-end price coverage: **83.7%**.

DEFAULT_DELIST_RETURN applies only when a held name loses prices mid-life; names with no price at pick time are never bought.
