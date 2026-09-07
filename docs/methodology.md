# Methodology

Formulas used by the live decision and, where noted, the backtest. This is a
3–20 year satellite sleeve, not a 1-month momentum book.

## Owner earnings

For a period (annual row or TTM):

\[
OE = OCF - CapEx - \mathbf{1}_{\text{subtract SBC}} \cdot SBC
\]

`subtract_sbc` defaults to true (conservative; material for SBC-heavy tech).
CapEx is the cash outflow (typically negative in Yahoo; the code uses the
signed cash-flow value).

**Normalized OE (non-cyclical):** mean of TTM, FY, FY-1, FY-2.

**Normalized OE (cyclical):** Energy, Basic Materials, Industrials, Consumer
Cyclical, and semiconductor industries use 7-year mean OE/revenue × TTM
revenue (Greenwald-style mid-cycle margin).

## Hurdle rate

\[
r = \max(0.08,\; r_f + ERP) + \mathbf{1}_{\text{High uncertainty}} \cdot 0.01
\]

\(r_f\) is FRED DGS10 (decimal). ERP defaults to Damodaran's implied equity
risk premium (config `valuation.hurdle.erp`, 4.5%). No per-company CAPM WACC.

Damodaran, A. Implied Equity Risk Premiums. Updated annually;
`data/reference/damodaran_erp.csv`.

## DCF (base and bear)

10-year explicit forecast of owner earnings:

- Years 1–5 grow at \(g_1 = \mathrm{clip}(g_{\text{hist 5y OEPS}}, 0, 0.15)\).
- Years 6–10 fade linearly to terminal \(g = 2.5\%\).
- Terminal value \(= CF_{11} / (r - g)\).
- Equity value \(= PV(\text{CFs} + TV) + \text{cash} - \text{debt}\), per diluted share.

**Bear:** \(g_1' = \min(0.5 \cdot g_1, 0.04)\), discount \(r + 1\%\).

Analyst 5-year growth is a **display-only** third scenario. The decision never
uses it (live ≡ backtest).

## Earnings-power value (Greenwald)

\[
EPV = \frac{EBIT_{\text{norm}} \cdot (1 - t)}{r} + \text{cash} - \text{debt}
\]

Tax rate \(t\) is clipped to \([0.15, 0.35]\). This is the no-growth value.

Greenwald, B. et al. *Value Investing* (EPV / reproduction value).

## Reverse DCF

Bisection on \(g_1 \in [-0.20, 0.40]\) so that DCF(base structure) equals
price. Display: market-implied 5-year growth vs historical \(g_{\text{hist}}\).

## Expected return

\[
\mathbb{E}[r] = \frac{OE_{\text{norm}}}{\text{market cap}} + \min(g_{\text{hist}}, 0.06)
\]

Accumulate requires \(\mathbb{E}[r] \ge r\) (configurable slack).

## Buy-below price

\[
P_{\text{buy}} = \min\bigl(DCF_{\text{base}} \cdot (1 - \text{MoS}),\; DCF_{\text{bear}}\bigr)
\]

MoS by uncertainty: Low 20%, Medium 30%, High 40% (Morningstar-style).

## Altman Z'' (1995)

Four-variable non-manufacturer model for all non-financials (no Utilities /
Real Estate exemption):

\[
Z'' = 6.56 X_1 + 3.26 X_2 + 6.72 X_3 + 1.05 X_4
\]

Distress gate: \(Z'' < 1.1\). Requires all four components. Financials exempt.

Altman, E. I. (1995). Emerging-market / non-manufacturer Z''.

## Quality score

Weighted average of five bucket percentiles vs the live universe
(profitability and stability sector-relative when the sector has ≥ 5 names;
accruals, dilution, FCF conversion universe-wide):

| Bucket | Weight | Ingredients |
|--------|--------|-------------|
| Profitability | 0.30 | Gross profitability, ROIC (Greenblatt IC), FCF margin |
| Earnings quality | 0.20 | Accruals, 3y FCF/NI conversion |
| Financial strength | 0.20 | Piotroski F, −ND/EBITDA, interest coverage |
| Stability | 0.15 | 5y ROIC mean/std, 5y gross-margin change, revenue CAGR |
| Capital discipline | 0.15 | Asset growth, shareholder yield, 3y share-count CAGR |

Financials use a reduced set (ROE, ROA, equity/assets, accruals, dilution,
stability).

Literature (screen, not a blended alpha recipe): Piotroski (2000) F-score;
Novy-Marx (2013) gross profitability; Asness–Frazzini–Pedersen QMJ (2019);
Sloan (1996) accruals; Cooper–Gulen–Schill (2008) asset growth.

## Value-trap flags

Each flag is `{code, triggered, value, threshold, note}`. Avoid if ≥ 2 fire
(config `value_trap.max_flags` = 1 allowed):

- ROIC below hurdle and declining 3 years
- Revenue 5y CAGR < 0
- Gross margin down > 500 bps over 5y
- ND/EBITDA above sector cap or interest coverage < 3
- Share-count CAGR 3y > 3%
- Accruals > 10% of assets
- 3y FCF conversion < 50%
- Negative normalized owner earnings

ND/EBITDA caps: default 3.5; Utilities 6.0; Real Estate 7.0; Communication
Services 4.5; Energy 3.0; Financials exempt.

## Relative valuation

EV/EBIT and P/owner-earnings vs the stock's own 10y annual history
(percentile and distance from median) and vs sector peer median. The old
correlation-based metric picker is retired.

## Uncertainty

Points from: coverage < 80%, 12m vol > 35%, analyst target range > 40% of
mean, data-quality grade B, DCF base/bear spread > 50%. 0 → Low, 1 → Medium,
≥2 → High. High adds 1% to the hurdle and 6 points to legacy composite/bargain
gates.

## Backtest honesty

- Sectors from SIC (not today's live snapshot).
- Newey–West t-stats and block bootstrap CIs; n independent windows reported.
- `apply` will not write in-sample thresholds without `--allow-in-sample`.
- Survivorship: Yahoo misses delists; see `docs/paid_data.md` and Shumway
  (1997). Do not read full-period wealth tables as out-of-sample alpha.
