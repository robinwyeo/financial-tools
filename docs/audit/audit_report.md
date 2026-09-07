# Finance Tools Overhaul — Audit Report

Date: 2026-09-06  
Scope: offline code review + full pytest/coverage + live-network spot checks (SEC companyfacts, Yahoo, FX). The hours-long `python -m backtest.run pipeline` was **not** run.  
Audience: a fleet of cheap Cursor Grok 4.6 models. Each `FIND-NNN` is a self-contained fix task. Do **not** edit this report as part of a fix; do **not** run the full backtest pipeline unless the finding explicitly says so.

## How to use this report

Pick one `FIND-NNN` with `Depends-on: none`. Implement only that finding. Run its Verify command. Stop.

Do not batch unrelated findings in one PR if you are a cheap model; prefer one finding per change set.

---

## 1. Baseline measurements

| Check | Result |
| --- | --- |
| `pytest tests/ -q` | **257 passed, 1 skipped**, 24 numpy divide warnings from `tests/test_backtest.py` |
| Skipped test | `tests/test_backtest.py::test_artifacts_share_run_id_when_present` (`no run_id-stamped artifacts yet`) |
| `core/` coverage | **90%** (4544 stmts, 457 miss) — meets WP3.4 ≥ 85% |
| `jobs/` coverage | mixed: `runlog.py` 96%, `refresh_reference.py` 78%, `email_sender.py` 68%, `watchlist_weekly.py` 69%, `universe_monthly.py` **59%** |
| Combined `--cov=core --cov=jobs` | 88% (jobs drag the total; WP3.4 acceptance is `core/` only) |
| Lowest `core/` modules | `universe.py` 81%, `fund_universe.py` 84% |

## 2. Live-network spot checks (2026-09-06)

| Check | Expected (plan) | Observed | Verdict |
| --- | --- | --- | --- |
| AAPL TTM revenue via companyfacts | within 1% of 416e9 | **416.161e9** | **pass** |
| KO FY2023 EBIT | within 2% of 11.3e9 | **11.311e9** (`OperatingIncomeLoss`, filed 2024-02-20) | **pass** |
| AAPL `get_fundamentals` | ≥ 9 annual EDGAR rows | **10** rows, `source=edgar` | **pass** |
| `hurdle_rate()` with no `FRED_API_KEY` | 0.087 | **0.087**, `rf_source=config_fallback`, `baa10y=None` | **pass** |
| SHOP.TO FX | listing CAD, ratios in USD | `currency=CAD`, `financial_currency=USD`, fx **0.7229**, price 200.76→**145.12**, `C$`, grade **B** `no EDGAR filer`, `FX_CONVERTED` | **pass** |
| RY.TO | `C$`, grade B | CAD, `C$`, Watch, grade **B** | **pass** |
| BRK-B live `book_to_market` (no snapshot overlay) | no dual-class blow-up | **0.69** (was 682 on committed snapshot) | **code pass** |
| BRK-B live `roic` | clip ≤ 2.0 | **0.13** | **code pass** |
| BRK-B live `graham_ratio` | **< 2** (WP0.2) | **52.69** | **fail** — FIND-001 |
| BRK-B live EV / EY | sane EV | Yahoo EV **-233.9e9**, EY **-0.48** | **fail** — FIND-003 |
| BRK-B data-quality | A or B with equity sub | **grade C**, 3 substitutions (revenue 12%, cash 91%, shares 34%) | **fail** — FIND-001, FIND-002 |
| Committed `data/universe_snapshot.parquet` | rebuilt post-fix | dated **2026-09-01**, 503 rows, max BTM **682**, max ROIC **164**, BRK-B graham **42.9**, `insider_buying` **100% NaN**, **no `quality_score`** | **stale** — FIND-008 |
| 50-ticker `python -m core.universe --fast --max 50` | inspect rebuilt parquet | Started; **aborted** so the audit would not overwrite the 503-row committed snapshot. Per-ticker live checks used instead. | n/a |

`FRED_API_KEY` was unset in this environment, so DGS10 / BAA10Y / T10YIE were not live-fetched.

---

## 3. Work-package status

Statuses: `pass` (acceptance met in code + tests, or live-verified) / `partial` / `fail` / `not-demonstrated`.

| WP | Status | One-line reason |
| --- | --- | --- |
| WP0.1 companyfacts ingest | **partial** | Live TTM/PIT/Q4 pass AAPL/KO. Backtest store is gitignored and not rebuilt; `ingest_edgar` early-returns any existing parquet with no schema check. |
| WP0.2 live fundamentals | **partial** | TTM/MRQ, statement equity, ROIC floor, Z'' implemented and unit-tested. Live BRK-B graham still 52.7; committed snapshot still has the Section 1.3 outliers. |
| WP0.3 DataQuality | **pass** | A/B/C grades + tests. Live BRK-B C is arguably too aggressive (see FIND-001/002) but the mechanism works. |
| WP0.4 `.TO` currency | **pass** | Live SHOP.TO converts CAD→USD at 0.72; header uses `C$`. |
| WP0.5 backtest stats | **partial** | NW t-stat, block bootstrap, SIC sectors, `apply --allow-in-sample` guard all in code+tests. Committed `backtest_report.md` still has the inert delist table. |
| WP0.6 clean pipeline + README evidence | **not-demonstrated** | `run_id: pending-clean-pipeline`; result JSONs have **no** `run_id`. Owner command below. |
| WP0.7 CI + snapshot guard | **partial** | CI on 3.11/3.12 from lock; `SnapshotIncompleteError` real. No CI badge. `git add … \|\| true` remains. |
| WP0.8 small fixes | **pass** | Sub-signal coverage fraction; Underperform gate; `run_daily = run_weekly` alias kept on purpose. |
| WP1.1 fundamentals API | **pass** | Live AAPL 10 EDGAR rows; Yahoo `.TO` fallback tested. |
| WP1.2 hurdle | **pass** | Fallback 0.087 live; DGS10 bump unit-tested. |
| WP1.3 valuation | **pass** | DCF/reverse-DCF/cyclical fixtures in `tests/test_valuation.py`. |
| WP1.4 retire picker | **pass** | `pick_best_metric` gone; no `correlation` key. |
| WP1.5 quality + flags | **partial** | Implemented + unit-tested; committed snapshot has no `quality_score` column. |
| WP1.6 decision engine | **pass** | Accumulate/Watch/Avoid fixtures in `tests/test_decision.py`. Default `decision.mode: intrinsic`. |
| WP1.7 backtest parity + registry | **partial** | Registry/CLI exist. `_build_raw_row` leaves quality/history fields `None`; no `valuation_summary` / `decide` / `pit_risk_free` under `backtest/`. Signal cards omit CI. |
| WP1.8 min UI/email | **pass** | Watch banner + email Why/Accumulate-first sort. |
| WP2.1 rates + Damodaran | **partial** | BAA10Y/T10YIE + refresh job exist. Backtest does not call `pit_risk_free`. UI omits `rf_as_of`. |
| WP2.2 S&P 400 + TSX 60 | **partial** | Fetchers + dual-list dedupe tested. Config default is `sp500` only; ~960-row snapshot not built. |
| WP2.3 providers + Form 4 extras | **pass** | Protocol + optional Finnhub; officer/director + 10b5-1. |
| WP2.4 paid-data memo | **pass** | `docs/paid_data.md` + Sharadar stub. |
| WP3.1 UI split | **pass** | `app.py` 68 lines; `python -c "import ui.stock_view"` works. |
| WP3.2 cache + vectorize | **partial** | `@st.cache_data` + vectorized composite. No AppTest that the price-range radio does not rescore. `score_fund_cached` not mtime-keyed. |
| WP3.3 jobs reliability | **pass** | Why column, run summaries, 10% failure exit. |
| WP3.4 tests ≥ 85% core | **pass** | core **90%**. Some backfill tests are tautological (FIND-016). |
| WP3.5 docs | **partial** | ARCHITECTURE/methodology exist. Evidence markers honest-but-pending; `tests/test_readme_evidence.py` allows that forever. Stale ~2× IC comment in `backtest/constants.py`. |

Fund path (`score_fund`, `build_fund_raw_metrics`, `score_universe_df` with `factor_columns=FUND_FACTOR_SCORE_COLUMNS`): **compatible**. No stock FX conversion and no DCF/decision coupling.

---

## 4. Section 1.3 live-bug scorecard

| Bug | Code | Guarding test | Residual |
| --- | --- | --- | --- |
| BRK-B `bookValue × shares` BTM | statement `book_equity` in `build_raw_metrics` | `tests/test_data.py:198-263` | Live BTM 0.69. Graham still broken (FIND-001). Fallback `book_value * shares` remains in `core/factors.py` (FIND-004). |
| ROIC 45x–164x | floor 10% assets, clip [-1, 2] | `tests/test_factors.py:168-181` | Live 0.13. Committed snapshot still 164. |
| `insider_buying` 100% NaN | cache key `form4v2` | `tests/test_insiders.py` | Code OK. Committed snapshot still 100% NaN until rebuild. |
| Negative equity `low_leverage` | `None` + `negative_equity=True` | `tests/test_data.py:266-299` | OK |
| `normalize_debt_to_equity` on live path | live uses `total_debt / book_equity` | helper tests only | FIND-015 |
| TTM flows / MRQ BS | `fetch_ttm_financials` | `tests/test_data.py:198-256` | OK |
| Silent `(debt or 0)` in EV | EV requires both or Yahoo EV | `tests/test_data.py:291-294` | IC and `_ey_point` still zero-fill (FIND-005, FIND-006). Negative Yahoo EV accepted (FIND-003). |
| Altman classic Z gate | Z'' all-or-nothing; financials only exempt | `tests/test_factors.py`, `tests/test_scoring.py:481-513` | OK |
| `.TO` currency | `fetch_fx_rate` | tests + live SHOP.TO | OK |
| Coverage any-one = 100% | weight × sub-signal fraction | `tests/test_scoring.py:571-588` | OK |
| Sell gate misses Underperform | `{"Sell","Underperform"}` | `tests/test_scoring.py:547-568` | OK |

---

## 5. Owner-only (not a cheap-model task)

These are **not-demonstrated**, not `fail`. Do not invent numbers.

WP0.6 / WP1.7 evidence regeneration (hours):

```bash
python -m backtest.run pipeline
python -m backtest.run report
python -m backtest.run build-factors --force --max-quarters 12 --max-tickers 80
python -m backtest.run evaluate --metric all
python -m backtest.run evaluate-signal --signal decision
```

(Plan text said `evaluate --all`; actual CLI is `--metric all`.)

After that, all `backtest/results/*.json` must share one `run_id`, README evidence must be rewritten by `report.py`, and `bargain_tuning_results.json` must include candidate `legacy_040_035_025` (already registered in `backtest/tune.py:443`).

Rebuild the live snapshot (overwrites `data/universe_snapshot.parquet`):

```bash
python -m core.universe --fast --max 50   # smoke
# then the monthly job / full universe
```

Acceptance on the rebuilt file: no `book_to_market > 5`, no `roic > 2`, BRK-B `graham_ratio < 2` (blocked today by FIND-001), `quality_score` present on ≥ 95% of rows, `insider_buying` not 100% NaN.

---

## 6. Findings

### FIND-001 Dual-class EDGAR share substitution blows Graham (and per-share metrics)

- Severity: high
- Area: WP0.2 / WP0.3
- Files: `core/data_quality.py:24-31`, `core/data_quality.py:222-228`, `core/factors.py:263-292`, `core/edgar_facts.py:97-100`
- Evidence: Live `build_raw_metrics("BRK-B")` on 2026-09-06: Yahoo `sharesOutstanding=1_431_693`, EDGAR `EntityCommonStockSharesOutstanding=941_481` (Class A count), `book_equity=747.91e9`, listing price **$506**. After `apply_substitutions`, `shares_outstanding=941481`, BVPS = equity/A-shares ≈ $794k (A-share book), `graham_ratio=52.69`. WP0.2 requires BRK-B graham **< 2**. The unit test in `tests/test_data.py:198-263` mocks `load_edgar_snapshot` to `None`, so EDGAR share substitution never runs; it uses matching 2e9 statement shares and therefore cannot catch this.
- Expected: Per-share metrics for the **listed** share class (BRK-B) must not mix Class-A share counts with Class-B price. Statement **equity / market cap** (BTM 0.69) is already correct.
- Fix: In `apply_substitutions` / `COMPARE_SPEC`, do **not** overwrite `shares_outstanding` when Yahoo vs EDGAR differ by more than the 5% tolerance in a way that looks like share-class mismatch (e.g. skip the shares field entirely for substitution, or only substitute when both counts are within 5% after a 1500:1 BRK heuristic is unnecessary — simplest: remove `shares_outstanding` from `COMPARE_SPEC` and keep Yahoo/listing shares for per-share work while still recording a RECONCILE warning). Keep substituting `book_equity`. Extend `tests/test_data.py::test_build_raw_metrics_uses_ttm_flows_mrq_equity` with a non-None EDGAR snapshot that returns A-share count `941481` vs Yahoo B-equivalent `2e9` and assert `graham_ratio < 2` and `raw["shares_outstanding"]` stays on the listing class.
- Verify: `pytest tests/test_data.py tests/test_data_quality.py -q`
- Depends-on: none

### FIND-002 EDGAR cash tag for financials is too narrow (BRK-B grade C)

- Severity: high
- Area: WP0.3 / WP0.1
- Files: `core/edgar_facts.py` FIELD_TAGS `total_cash` (CashCashEquivalentsAndShortTermInvestments then CashAndCashEquivalentsAtCarryingValue), `core/data_quality.py:28`, `core/data_quality.py:215-216`
- Evidence: Live BRK-B: Yahoo `totalCash=365.5e9` vs EDGAR `31.58e9` (91% off). That single substitution plus revenue (12%) plus shares (34%) yields **3 substitutions → grade C**, which blocks Accumulate (`core/decision.py` data_quality gate). For an insurer/holding company, 31e9 is cash-only; 365e9 is cash + investments. Grade C here is a false disqualifier, not “Yahoo is wrong.”
- Expected: Plan grade C is for >2 substitutions that indicate bad data. Cash-tag mismatch on financials should not count as three independent Yahoo errors. Financials are already relative-only.
- Fix: Either (a) skip `total_cash` reconciliation when `sector == "Financial Services"`, or (b) try additional us-gaap tags (e.g. `AvailableForSaleSecurities`, `TradingSecurities`, `Investments`) before substituting, or (c) require grade C only when substitutions are on revenue/equity/shares **and** non-financial. Prefer (a) as the smallest change. Add a test in `tests/test_data_quality.py` with Yahoo cash 365e9 vs EDGAR cash 31e9, sector Financial Services, and assert grade is **B** (or A) not C solely from cash.
- Verify: `pytest tests/test_data_quality.py tests/test_decision.py -q`
- Depends-on: none

### FIND-003 Negative Yahoo EV is used as-is (BRK-B EY = -0.48)

- Severity: high
- Area: WP0.2
- Files: `core/data.py:864-871`
- Evidence: Live BRK-B `enterprise_value = -233_925_820_416`, `earnings_yield = -0.479`. The EV path prefers `yahoo_ev` without checking `> 0`. Plan: “EV requires debt and cash present or falls back to Yahoo `enterpriseValue`; otherwise None.” Negative EV is not a usable enterprise value.
- Expected: Non-positive EV → `None` + a `data_warnings` entry; do not compute EY from it.
- Fix: After assigning `enterprise_value = yahoo_ev`, if it is not `None` and `<= 0`, treat as missing and fall through to the debt+cash construction or None. Add a unit test in `tests/test_data.py` with `enterpriseValue: -1e9` and valid debt/cash, asserting EV is the constructed `market_cap + debt - cash` (or None if those missing) and never negative.
- Verify: `pytest tests/test_data.py tests/test_factors.py -q`
- Depends-on: none

### FIND-004 Residual `book_value * shares` fallbacks

- Severity: medium
- Area: WP0.2 / Section 1.3
- Files: `core/factors.py:28-30`, `core/factors.py:58-59`, `core/factors.py:392-395`, `core/factors.py:561-565`
- Evidence: `compute_value_factors`, `compute_quality_factors`, `compute_capital_efficiency`, and `compute_altman_z_double_prime` still do `book_equity = book_value * shares` when `book_equity` is None. Live path now sets `book_equity`, but any caller without it (backtest row, partial snapshot) reintroduces the BRK-B blow-up.
- Expected: Plan: never use `bookValue * sharesOutstanding` as book equity on the live path; statement equity only.
- Fix: Delete the four fallbacks. If `book_equity` is None, leave BTM/ROE/IC/Z'' as None. Add a test that `compute_value_factors({"book_value": 400_000, "shares_outstanding": 2e9, "market_cap": 900e9})` returns `book_to_market is None`.
- Verify: `pytest tests/test_factors.py tests/test_data.py -q`
- Depends-on: none

### FIND-005 Invested-capital path still zero-fills missing cash

- Severity: medium
- Area: WP0.2
- Files: `core/factors.py:406-413`
- Evidence: `cash = total_cash or 0.0` and `invested_capital = total_debt + book_equity - (total_cash or 0.0)`. Plan forbade silent zeroing of debt/cash in EV/IC.
- Expected: If preferred NWC+PPE inputs are incomplete **and** `total_cash is None`, return `roic=None` rather than treating cash as 0.
- Fix: Only take the NWC path when `total_cash is not None` (or pass cash through as optional with an explicit `roic_basis` that records the assumption). Do not use `or 0.0`. Test a raw dict with CA/CL/PPE present and `total_cash=None` → `roic is None`.
- Verify: `pytest tests/test_factors.py -q -k roic`
- Depends-on: none

### FIND-006 `_ey_point` still zero-fills debt and cash

- Severity: medium
- Area: WP0.2 / WP1.4
- Files: `core/data.py:1201-1203`
- Evidence: `enterprise_value = price * shares + (total_debt or 0.0) - (total_cash or 0.0)` in Yahoo own-history EY. Same silent-zeroing class as the old live EV bug.
- Expected: Missing debt **or** cash → skip that history point (`return None`), matching live EV.
- Fix: If `total_debt is None` or `total_cash is None`, return None. Add a test around `build_earnings_yield_history` / `_ey_point` with a balance sheet missing Cash → that date omitted.
- Verify: `pytest tests/test_data.py tests/test_edgar_history.py -q`
- Depends-on: none

### FIND-007 `ingest_edgar` trusts any existing parquet (no schema/version guard)

- Severity: high
- Area: WP0.1
- Files: `backtest/data/edgar.py:81-82`
- Evidence: `if EDGAR_FUNDAMENTALS_PATH.exists() and not force: return pd.read_parquet(...)`. A leftover FSDS file (columns `amount`/`period`, no `end`/`filed`) is loaded and then `fundamentals_as_of` → `KeyError: 'end'`. The path is gitignored (`backtest/data/store/`) so CI is clean, but any developer machine with the old store is silently wrong.
- Expected: Companyfacts schema only; refuse or auto-rebuild if required columns (`end`, `filed`, `qtrs`, `field`, `val`, `tag`) are missing.
- Fix: After read, if required columns missing, log a warning and fall through to a real ingest (or raise a clear `RuntimeError` telling the user to `ingest --force`). Add a unit test with a temp FSDS-shaped parquet that asserts ingest does not return it as-is.
- Verify: `pytest tests/test_backtest.py tests/test_edgar_facts.py -q`
- Depends-on: none

### FIND-008 Committed universe snapshot is pre-overhaul (serves live UI/email)

- Severity: high
- Area: WP0.2 / WP1.5 / WP0.8
- Files: `data/universe_snapshot.parquet` (snapshot_date `2026-09-01T18:13:32Z`), `core/scoring.py:578-588` (`_merge_ticker_row_with_universe` keeps structural factors from the snapshot)
- Evidence: 503 rows; max `book_to_market=682`; max `roic=163.8`; BRK-B graham `42.9`; `insider_buying` NaN 100%; **no `quality_score`**. `score_ticker("BRK-B")` with default `universe_df=None` **overlays** these stale factors (observed BTM 682 on the analysis even after live `build_raw_metrics` produces 0.69).
- Expected: Live rankings use post-fix fundamentals; snapshot has `quality_score`, `universe`, `currency`.
- Fix: Do **not** run a full universe rebuild in a cheap-model session unless asked. Add `tests/test_universe.py::test_committed_snapshot_has_post_overhaul_columns` that, if `data/universe_snapshot.parquet` exists, asserts columns include `quality_score`, `currency`, `universe` and that BRK-B (if present) has `book_to_market < 5` and `graham_ratio < 2` and `roic <= 2`. That test **will fail today** and is the honesty gate until the owner rebuilds. If you cannot rebuild, xfail with a reason is **not** acceptable; fail loudly.
- Verify: `pytest tests/test_universe.py -q`
- Depends-on: FIND-001 (graham assertion will fail until shares substitution is fixed)

### FIND-009 Committed backtest artifacts have no `run_id`; tests skip / soft-pass

- Severity: high
- Area: WP0.6 / WP3.5
- Files: `backtest/results/*.json`, `tests/test_backtest.py:311-324`, `tests/test_readme_evidence.py:20-33`
- Evidence: Four JSON result files exist and **none** contain `run_id`. `test_artifacts_share_run_id_when_present` **skips**. `test_evidence_run_id_matches_artifacts_when_present` accepts `"pending"` / `"pre-companyfacts"` and returns. README still cites composite IC ~0.07 inside the evidence block.
- Expected: Plan: tests assert artifacts share a `run_id`. Pending mode must not hide existing unstamped files.
- Fix: If `RESULTS_DIR.glob("*.json")` is non-empty and none have `run_id`, **fail** (not skip) with a message to run `python -m backtest.run pipeline` or delete stale files. In `test_readme_evidence.py`, if the evidence block contains a numeric IC / CAGR claim **and** `run_id` is pending, fail (the current README bullet at line 165–167 is the example). Do not invent a `run_id`.
- Verify: `pytest tests/test_backtest.py tests/test_readme_evidence.py -q` (expect fail until README/artifacts are honest)
- Depends-on: none

### FIND-010 Stale delist-sensitivity table in committed report

- Severity: medium
- Area: WP0.5
- Files: `backtest/results/backtest_report.md:96-102`
- Evidence: Generator `backtest/report.py` no longer emits this table (survivorship is `price_coverage_pct` only). The committed markdown still has three identical delist rows (Generated 2026-08-24). Plan: “report contains no delist-sensitivity table.”
- Fix: Delete the `### Survivorship sensitivity` section (and the identical three-row table) from `backtest/results/backtest_report.md`. Add a test `tests/test_backtest.py` that ` "Survivorship sensitivity" not in (RESULTS_DIR / "backtest_report.md").read_text()`.
- Verify: `pytest tests/test_backtest.py -q -k survivorship or pytest tests/test_backtest.py -q`
- Depends-on: none

### FIND-011 Stale “graham_heavy ~2× IC” comment

- Severity: high
- Area: WP0.6 / WP3.5
- Files: `backtest/constants.py:94-95`
- Evidence: Comment says graham_heavy is the “validated winner … (~2x 3y/5y IC vs the previous 0.40/0.35/0.25 default).” README retracts that claim. Committed `bargain_tuning_results.json` has identical 0.55/0.30/0.15 weights for `default_long_horizon` and `graham_heavy` and **no** 0.40 candidate.
- Expected: No README/code comment about backtest results that is not in a `run_id`-stamped artifact.
- Fix: Replace the comment with: weights are design defaults (graham 55% of bargain); do not cite pre-companyfacts IC. Point at `legacy_040_035_025` in `backtest/tune.py` as the historical mix to evaluate on the next pipeline.
- Verify: `rg -n "2x|doubled IC|roughly doubled" backtest/ core/ README.md docs/` returns nothing except the README sentence that the old claim is **not** reproducible.
- Depends-on: none

### FIND-012 Backtest panel does not compute quality, valuation, or decision

- Severity: high
- Area: WP1.7
- Files: `backtest/factors.py:46-180` (`_build_raw_row`), `backtest/registry.py:74-92`
- Evidence: Registry registers `quality_score` and signal `decision`, but `_build_raw_row` sets `owner_earnings_norm`, `revenue_5y_cagr`, `share_cagr_3y`, `fcf_conversion_3y`, `roic_5y_mean`, etc. to **None**. Grep of `backtest/` finds **no** calls to `compute_quality_score`, `valuation_summary`, `decide`, or `pit_risk_free`. `evaluate --metric quality_score` on a current panel would be all-NaN.
- Expected: Plan: PIT `Fundamentals` from the companyfacts store, quality inputs, valuation with PIT DGS10, `decision_label`.
- Fix: In `_build_raw_row` (or a follow-on enricher called from panel build): construct `Fundamentals` from `fundamentals_as_of_structured`; call `history_quality_inputs`; call `hurdle_rate`/`pit_risk_free(as_of)`; call `valuation_summary` with `analyst_growth=None`; call `decide`; store `quality_score` (or the raw inputs the cross-section needs), `decision_label`, `ev_to_ebit`, `p_to_oe`. Keep estimate_revisions/insider excluded. Extend `tests/test_backtest.py` synthetic panel so a row with complete fundamentals gets a non-null `decision_label`.
- Verify: `pytest tests/test_backtest.py tests/test_quality.py tests/test_decision.py -q`
- Depends-on: none (can land before the hours-long pipeline)

### FIND-013 `evaluate_signal` omits block-bootstrap CI

- Severity: medium
- Area: WP1.7
- Files: `backtest/registry.py:136-167`
- Evidence: Return payload has `hit_rate` / `mean_excess` / `n` only. Plan: “3y/5y hit rate and mean excess **with CI**” using the same block bootstrap.
- Expected: `stats_3y` / `stats_5y` include CI bounds from `backtest.stats.block_bootstrap_ci`.
- Fix: Feed per-quarter mean excess (or pick-level excess grouped by quarter) into `block_bootstrap_ci`. Add keys `mean_excess_ci_low` / `mean_excess_ci_high`. Assert they exist in `tests/test_backtest.py` registry round-trip.
- Verify: `pytest tests/test_backtest.py tests/test_backtest_stats.py -q`
- Depends-on: none

### FIND-014 `value_trap.max_flags` is loaded and unused

- Severity: medium
- Area: WP1.6 / WP1.5
- Files: `config.yaml` `value_trap.max_flags`, `core/config.py:167`, `core/decision.py` (uses `decision.max_value_trap_flags` only), `docs/methodology.md` (if it documents `value_trap.max_flags`)
- Evidence: Changing yaml `value_trap.max_flags` does not change Avoid. The live gate reads `get_decision_config()["max_value_trap_flags"]`.
- Expected: One source of truth.
- Fix: Either delete `value_trap.max_flags` from config + `_DEFAULT_VALUE_TRAP`, or make `decide()` read `get_value_trap_config()["max_flags"]`. Update `docs/methodology.md` to match. Add a test that toggling the **documented** key changes the flag-count gate.
- Verify: `pytest tests/test_decision.py tests/test_config.py -q`
- Depends-on: none

### FIND-015 No test that live D/E skips `normalize_debt_to_equity`

- Severity: medium
- Area: WP0.2
- Files: `core/data.py:820-823`, `backtest/factors.py:63-65` (still calls the helper), `tests/test_data.py:18-28`
- Evidence: Live path computes `total_debt / book_equity`. Only tests of the helper exist. A regression that re-imports the heuristic onto live D/E would not fail CI.
- Expected: Plan: remove `normalize_debt_to_equity` from the live path.
- Fix: In `test_build_raw_metrics_uses_ttm_flows_mrq_equity`, assert `raw["debt_to_equity"] == pytest.approx(50e9/500e9)` (ratio, not percent). Optionally spy that `normalize_debt_to_equity` is not called from `build_raw_metrics`.
- Verify: `pytest tests/test_data.py -q`
- Depends-on: none

### FIND-016 Coverage-backfill tautological asserts

- Severity: medium
- Area: WP3.4
- Files: `tests/test_coverage_backfill2.py:442`, `tests/test_coverage_backfill2.py:872`, `tests/test_coverage_backfill.py:366-369`
- Evidence: `assert list(zero.fillna(False))` is true for any non-empty list. `assert abs(sum(qw.values()) - 1.0) < 1e-6 or qw` passes for any non-empty dict. Kitchen-sink `compute_all_factors` only checks `is not None` / key presence, not the kitchen-sink numbers.
- Expected: Tests pin behavior, not coverage.
- Fix: Replace with `assert list(zero.fillna(False)) == [False, True, False]` (adjust if `_meaningful_mask` treats 0 as meaningful — `ZERO_NEUTRAL_COLUMNS` is empty, so 0 is meaningful and the expected mask is `[True, True, False]`). Change weights to `assert sum(qw.values()) == pytest.approx(1.0)`. For `_rich_raw()`, assert `out["earnings_yield"] == pytest.approx(80/1000)`.
- Verify: `pytest tests/test_coverage_backfill.py tests/test_coverage_backfill2.py -q`
- Depends-on: none

### FIND-017 Intrinsic-mode UI still advertises composite/bargain/coverage Buy rules

- Severity: medium
- Area: WP1.6 / WP1.8 / WP0.8
- Files: `ui/sidebar.py:15-19`, `ui/stock_view.py:367-380`, `config.yaml:98` (`decision.mode: intrinsic`)
- Evidence: Default mode is intrinsic (`decide()` gates: data_quality, distress, value_trap_count, quality_percentile, price_vs_buy_below / relative_cheap). Sidebar still lists Composite ≥ 58.9, Bargain ≥ 49.5, Coverage ≥ 70% as “Good-buy criteria.” Success banner on `is_good_buy` repeats those legacy hurdles. Coverage is **not** an intrinsic gate.
- Expected: UI copy matches `decision.mode`. Legacy criteria only when `mode == "legacy"`.
- Fix: Branch on `get_decision_config(config)["mode"]`. Intrinsic sidebar: quality percentile, max flags, MoS / buy-below, hurdle, data-quality C, Z''. Keep the current bullets under a “Legacy gate” expander or only when mode is legacy. Success banner should quote `decision.label` and failed/passed `gates`, not composite_min.
- Verify: `pytest tests/test_ui_decision_card.py tests/test_decision.py -q`
- Depends-on: none

### FIND-018 Valuation card omits FRED `as_of`

- Severity: medium
- Area: WP2.1
- Files: `ui/formatting.py:423-438`, `core/rates.py:198-209` (already returns `rf_as_of`)
- Evidence: Card shows “10y Treasury {rf}, ERP {erp}” with no date. Plan: UI shows hurdle “with source dates.”
- Expected: Include `hurdle["rf_as_of"]` and `hurdle["rf_source"]` (e.g. `FRED:DGS10 as of 2026-09-05` or `config_fallback`).
- Fix: Extend `valuation_card_html` to append source/date. Update `tests/test_ui_decision_card.py` with a fixture that has `rf_as_of="2024-01-01"` and assert the string appears.
- Verify: `pytest tests/test_ui_decision_card.py -q`
- Depends-on: none

### FIND-019 `score_fund_cached` is not snapshot-mtime keyed

- Severity: medium
- Area: WP3.2
- Files: `ui/state.py:90-94` vs `score_ticker_cached` at `77-87` and `score_fund_universe_cached` at `69-74`
- Evidence: `score_fund_cached` keys only `(ticker, config)` with TTL 1800s. After a fund snapshot rebuild, gauges can stay stale for 30 minutes. Stock ticker cache at least keys `snapshot_date`.
- Expected: Plan: scored fund universe keyed by snapshot mtime; ticker scores keyed by ticker + snapshot date.
- Fix: Mirror `score_ticker_cached`: read fund snapshot date/mtime inside the wrapper and pass it into `_inner`. Load the live snapshot inside `_inner` (do not close over the first snap).
- Verify: `pytest tests/test_app_smoke.py tests/test_fund_factors.py -q` plus a small unit test that two calls with different mtimes are different cache keys (can test the wrapper by mocking `st.cache_data` as identity and counting inner calls).
- Depends-on: none

### FIND-020 WP3.2 AppTest gap (price radio must not rescore)

- Severity: medium
- Area: WP3.2
- Files: `tests/test_app_smoke.py:6-24`
- Evidence: Smoke tests only check `app.py` contains `set_page_config` / `main` / line count < 120 and that `render_stock_view` exists. Plan: `streamlit.testing.v1.AppTest` with a counter proving the price-range radio does not call `score_universe`.
- Fix: Add `tests/test_app_smoke.py::test_price_range_radio_does_not_rescore` using AppTest and monkeypatched `score_universe` / `score_universe_cached` with a call counter, or skip with `pytest.importorskip` if AppTest is unavailable. If AppTest is too brittle, a cheaper equivalent: assert `ui/charts.py` price-range control does not import/call scoring (static), **and** assert `score_universe_cached` is wrapped in `st.cache_data`.
- Verify: `pytest tests/test_app_smoke.py -q`
- Depends-on: none

### FIND-021 CI workflow has no README badge; `git add || true` remains

- Severity: low
- Area: WP0.7
- Files: `README.md` (no Actions badge), `.github/workflows/universe-monthly.yml:81-82`
- Evidence: Plan acceptance “CI badge green.” `git add data/universe_snapshot.parquet … || true` still masks missing paths. `git push || true` **was** removed (good).
- Fix: Add a standard GitHub Actions badge for `.github/workflows/ci.yml` near the top of README. Change `git add … || true` to `git add …` without `|| true` (the following `git diff --staged --quiet || git commit` already no-ops when nothing is staged).
- Verify: `rg -n "git add.*\\|\\| true" .github/workflows/` returns nothing; README contains a `github.com/.*/actions` badge URL.
- Depends-on: none

### FIND-022 `pit_risk_free` is never used by the backtest

- Severity: medium
- Area: WP2.1 / WP1.7
- Files: `core/rates.py:123-133`, `backtest/` (no callers)
- Evidence: `pit_risk_free` exists and is unit-tested. Plan: “backtest uses PIT DGS10.” Nothing in `backtest/` imports it.
- Fix: Same change-set as FIND-012 is fine, but this finding can land alone: in `_build_raw_row` or valuation attach, set `rf = pit_risk_free(as_of)` and pass `rf=` into `hurdle_rate`. Test with a fake FRED series that steps in 2015 vs 2020 and assert two `as_of` dates get different `rf`.
- Verify: `pytest tests/test_rates.py tests/test_backtest.py -q`
- Depends-on: none

### FIND-023 Config still holds in-sample composite/bargain thresholds as if they were live gates

- Severity: low
- Area: WP0.5 / WP1.6
- Files: `config.yaml:8-11` (`composite_min: 58.9`, `bargain_min: 49.5`)
- Evidence: Those numbers are the old in-sample calibration. Intrinsic mode does not use them as the Accumulate gate, but the sidebar still displays them (FIND-017). Plan: `apply` must not write return-fitted thresholds without `--allow-in-sample` going forward; existing yaml still advertises them.
- Fix: If FIND-017 is done, optionally rename a comment in `config.yaml` above those keys: “legacy gate only (`decision.mode: legacy`); not calibrated in the current release.” Do not change the numbers in this task (that would look like a new calibration).
- Verify: `pytest tests/test_config.py -q`
- Depends-on: none (comment-only)

### FIND-024 Dead parquet helpers in `core/edgar_history.py`

- Severity: low
- Area: WP0.1 / WP1.4
- Files: `core/edgar_history.py:44-89` (`_load_edgar_parquet`, `_period_table_from_parquet`)
- Evidence: `load_period_table` always uses companyfacts (lines 280–298) and never calls these. They still exist and are easy to accidentally re-wire to the old FSDS store.
- Fix: Delete `_load_edgar_parquet` and `_period_table_from_parquet` and any tests that only exist to cover them. Keep `_period_table_from_companyfacts` as the HTTP fallback.
- Verify: `pytest tests/test_edgar_history.py tests/test_coverage_backfill2.py -q`
- Depends-on: none

---

## 7. Suggested shard order for cheap models

Independent first wave (no shared files if split carefully):

1. FIND-004 (`core/factors.py` fallbacks)
2. FIND-005 (`core/factors.py` IC zeros) — same file as 004; **do not parallelize 004+005**
3. FIND-003 (`core/data.py` EV)
4. FIND-006 (`core/data.py` `_ey_point`) — same file as 003; **do not parallelize 003+006**
5. FIND-011 (`backtest/constants.py` comment)
6. FIND-010 (delete stale report section)
7. FIND-014 (`value_trap.max_flags`)
8. FIND-015 (one assertion in `tests/test_data.py`)
9. FIND-016 (backfill asserts)
10. FIND-018 (valuation card date)
11. FIND-019 (`ui/state.py` fund cache)
12. FIND-021 (badge + `|| true`)
13. FIND-023 (yaml comment)
14. FIND-024 (delete dead helpers)

Second wave (need FIND-001 before snapshot graham assert):

15. FIND-001 (shares substitution)
16. FIND-002 (financials cash grade)
17. FIND-007 (ingest schema guard)
18. FIND-009 (fail unstamped artifacts)
19. FIND-017 (UI copy vs intrinsic)
20. FIND-008 (snapshot column/outlier test — will fail until owner rebuilds **and** FIND-001)
21. FIND-012 + FIND-022 (backtest PIT quality/valuation/rf)
22. FIND-013 (signal CI)
23. FIND-020 (AppTest)

Owner-only after code fixes: WP0.6 pipeline, full snapshot rebuild.

---

## 8. What is in good shape (do not “fix”)

- Live companyfacts TTM for AAPL/KO matches the plan’s dollar checks.
- Fund path signatures and FX isolation.
- `apply --allow-in-sample` guard.
- SIC sector map (not live snapshot) in `backtest/engine.py`.
- `load_period_table` does not read `fundamentals.parquet`.
- Altman Z'', Underperform gate, sub-signal coverage, `.TO` FX, ROIC clip.
- Thin `app.py`, no module-level `st.*` in `ui/`, radar gaps are `None` (not mean-filled), `FACTOR_COLORS`/`METRIC_HELP` gone.
- `data/runs/` gitignored; CI matrix 3.11/3.12 installs `requirements.lock`; monthly timeout 420; weekly timeout 60; `SnapshotIncompleteError` prevents parquet write; `run_daily = run_weekly` alias is intentional.
- README evidence block **already** labels pre-companyfacts numbers as not current. Do not restore “graham_heavy doubled IC” as a fact.
