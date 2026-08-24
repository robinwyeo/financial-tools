# Score Weight & Threshold Backtest Report

Generated: 2026-08-24 15:38 UTC

## Summary

This report validates composite/bargain score weights and good-buy thresholds
using long-horizon (1y/3y/5y) forward returns and gated DCA buy-and-hold
simulations on historical S&P 500 constituents.

## Caveats

- Survivorship bias: delisted tickers absent from free price data cannot be selected at all, so strategy results are biased upward regardless of the delist return assumption.
- Analyst upside is informational (not a hard gate) and not historically backtested.
- Expanding-window folds + bootstrap CIs reduce but do not eliminate path dependence.
- SEC EDGAR fundamentals are point-in-time by filing date; reporting lags apply.

## Named Weight Candidate Comparison

Recommended: **evidence_based**
Primary horizon: **3y**

| Candidate | 3y IC | Excess mean | 95% CI | % folds > 0 |
| --- | ---: | ---: | ---: | ---: |
| evidence_based | 0.064 | 1.86% | [-2.58%, 6.99%] | 33% |
| equal | 0.080 | 0.01% | [-3.91%, 4.69%] | 50% |
| legacy_tuned | 0.070 | -0.17% | [-5.55%, 5.35%] | 50% |

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
- Winner mean IC (3y): 0.015
- Baseline mean IC: 0.015

### Recommended bargain weights

```yaml
  discount_52w: 0.1500
  margin_of_safety: 0.5500
  valuation_vs_history: 0.3000
```

## Threshold Calibration

- Horizon: **3y**
- composite_min: **58.9**
- bargain_min: **49.5**

## Investment Comparison vs S&P 500

Each quarter, **$20,000** was invested into the top 5 good-buy stocks (old or new parameters). The benchmark invests the same **$20,000/quarter** into **SPY** (S&P 500 total return) on the identical schedule.

- Simulation period: **2010-03-31 00:00:00** → **2026-03-31 00:00:00**
- Quarters with investment: **65**
- Total capital deployed (each strategy): **$1,300,000**

### Side-by-side results

| Metric | Old params | New params | S&P 500 (SPY) |
| --- | ---: | ---: | ---: |
| Terminal wealth | $6,317,571 | $9,297,724 | $4,326,313 |
| Total invested | $1,300,000 | $1,300,000 | $1,300,000 |
| Total return | 386.0% | 615.2% | 232.8% |
| CAGR | 10.4% | 13.1% | 7.8% |
| Max drawdown | -23.4% | -26.1% | -21.9% |

### Outperformance vs S&P 500

| Comparison | Terminal wealth Δ | Return Δ | CAGR Δ | Higher ROI? |
| --- | ---: | ---: | ---: | :---: |
| Old params vs SPY | +$1,991,258 (+46.0%) | +153.2% | +2.6% | Yes |
| New params vs SPY | +$4,971,411 (+114.9%) | +382.4% | +5.3% | Yes |
| New params vs Old params | +$2,980,153 (+47.2%) | +229.2% | +2.7% | Yes |

### Interpretation

- **Old parameters** finished +$1,991,258 ahead of SPY in terminal wealth; return on deployed capital was 386.0% (10.4% CAGR) vs SPY's 232.8% (7.8% CAGR).
- **New parameters** finished +$4,971,411 ahead of SPY in terminal wealth; return on deployed capital was 615.2% (13.1% CAGR) vs SPY's 232.8% (7.8% CAGR).
- **New parameters** beat old parameters by +$2,980,153 terminal wealth and +229.2% higher return on deployed capital.

### Survivorship sensitivity (terminal wealth vs SPY)

| Delist assumption | Old | New | SPY | Old − SPY | New − SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | $6,317,571 | $9,297,724 | $4,326,313 | +$1,991,258 | +$4,971,411 |
| -50% | $6,317,571 | $9,297,724 | $4,326,313 | +$1,991,258 | +$4,971,411 |
| -100% | $6,317,571 | $9,297,724 | $4,326,313 | +$1,991,258 | +$4,971,411 |
