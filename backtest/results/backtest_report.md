# Score Weight & Threshold Backtest Report

Generated: 2026-07-26 20:45 UTC

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
| evidence_based | 0.060 | 1.87% | [-2.62%, 6.48%] | 50% |
| equal | 0.070 | 0.81% | [-4.70%, 5.52%] | 67% |
| legacy_tuned | 0.065 | -0.93% | [-8.35%, 5.25%] | 50% |

- **evidence_based** vs **equal**: indistinguishable (bootstrap CIs overlap).
- **evidence_based** vs **legacy_tuned**: indistinguishable (bootstrap CIs overlap).

### Recommended factor weights

```yaml
  balance_sheet: 0.1000
  capital_discipline: 0.1250
  garp: 0.1000
  low_volatility: 0.0750
  momentum: 0.1000
  quality: 0.2500
  value: 0.2500
```

## Bargain Weight Validation

- Winner: **default_long_horizon**
- Winner mean IC (3y): 0.008
- Baseline mean IC: 0.008

### Recommended bargain weights

```yaml
  discount_52w: 0.2500
  margin_of_safety: 0.4000
  valuation_vs_history: 0.3500
```

## Threshold Calibration

- Horizon: **3y**
- composite_min: **57.5**
- bargain_min: **49.4**

## Investment Comparison vs S&P 500

Each quarter, **$20,000** was invested into the top 5 good-buy stocks (old or new parameters). The benchmark invests the same **$20,000/quarter** into **SPY** (S&P 500 total return) on the identical schedule.

- Simulation period: **2010-03-31 00:00:00** → **2026-03-31 00:00:00**
- Quarters with investment: **65**
- Total capital deployed (each strategy): **$1,300,000**

### Side-by-side results

| Metric | Old params | New params | S&P 500 (SPY) |
| --- | ---: | ---: | ---: |
| Terminal wealth | $7,559,854 | $6,727,997 | $4,326,313 |
| Total invested | $1,300,000 | $1,300,000 | $1,300,000 |
| Total return | 481.5% | 417.5% | 232.8% |
| CAGR | 11.6% | 10.8% | 7.8% |
| Max drawdown | -25.7% | -22.5% | -21.9% |

### Outperformance vs S&P 500

| Comparison | Terminal wealth Δ | Return Δ | CAGR Δ | Higher ROI? |
| --- | ---: | ---: | ---: | :---: |
| Old params vs SPY | +$3,233,541 (+74.7%) | +248.7% | +3.8% | Yes |
| New params vs SPY | +$2,401,684 (+55.5%) | +184.7% | +3.0% | Yes |
| New params vs Old params | −$831,856 (−11.0%) | −64.0% | −0.8% | No |

### Interpretation

- **Old parameters** finished +$3,233,541 ahead of SPY in terminal wealth; return on deployed capital was 481.5% (11.6% CAGR) vs SPY's 232.8% (7.8% CAGR).
- **New parameters** finished +$2,401,684 ahead of SPY in terminal wealth; return on deployed capital was 417.5% (10.8% CAGR) vs SPY's 232.8% (7.8% CAGR).
- **Old parameters** beat new parameters by +$831,856 terminal wealth and +64.0% higher return on deployed capital.

### Survivorship sensitivity (terminal wealth vs SPY)

| Delist assumption | Old | New | SPY | Old − SPY | New − SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0% | $7,559,854 | $6,727,997 | $4,326,313 | +$3,233,541 | +$2,401,684 |
| -50% | $7,559,854 | $6,727,997 | $4,326,313 | +$3,233,541 | +$2,401,684 |
| -100% | $7,559,854 | $6,727,997 | $4,326,313 | +$3,233,541 | +$2,401,684 |
