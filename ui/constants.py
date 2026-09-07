"""Display labels and chart constants. No Streamlit imports."""

from __future__ import annotations

PRICE_HISTORY_RANGES: dict[str, str] = {
    "1M": "1mo",
    "3M": "3mo",
    "6M": "6mo",
    "YTD": "ytd",
    "1Y": "1y",
    "2Y": "2y",
    "5Y": "5y",
    "All": "max",
}


DEFAULT_PRICE_RANGE = "2Y"


CHART_HEIGHT_ROW2 = 198


CHART_HEIGHT_PRICE = CHART_HEIGHT_ROW2


CHART_HEIGHT_RADAR = 208


CHART_HEIGHT_ANALYST_PIE = 158


GAUGE_MAX_WIDTH = "132px"


FACTOR_LABELS = {
    "value": "Value (earnings yield · B/M · FCF)",
    "garp": "GARP (Lynch dividend-adjusted PEG)",
    "quality": "Quality / Profitability",
    "balance_sheet": "Balance Sheet Strength",
    "momentum": "Momentum (12-1)",
    "low_volatility": "Low Volatility",
    "capital_discipline": "Capital Discipline (yield + asset growth)",
    "estimate_revisions": "Estimate Revisions (agreement · magnitude · surprise)",
    "insider": "Insider Buying (net Form 4 open-market)",
}


BARGAIN_LABELS = {
    "margin_of_safety": "Margin of Safety (Graham)",
    "valuation_vs_history": "Valuation vs Own History (10y EDGAR)",
    "discount_52w": "Discount to 52-Week High",
}


SHORT_FACTOR_LABELS = {
    "value": "Value",
    "garp": "GARP",
    "quality": "Quality",
    "balance_sheet": "Balance Sheet",
    "momentum": "Momentum",
    "low_volatility": "Low Volatility",
    "capital_discipline": "Capital Discipline",
    "estimate_revisions": "Est. Revisions",
    "insider": "Insider Buying",
}


RADAR_FACTOR_LABELS: dict[str, str] = {
    "value":              "Value",
    "garp":               "GARP",
    "quality":            "Quality",
    "balance_sheet":      "Bal. Sheet",
    "momentum":           "Momentum",
    "low_volatility":     "Low Vol",
    "capital_discipline": "Cap. Disc.",
    "estimate_revisions": "Revisions",
    "insider":            "Insiders",
}


FACTOR_SCORECARD_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Valuation", "#14b8a6", ["value", "garp"]),
    ("Quality & Health", "#8b5cf6", ["quality", "balance_sheet"]),
    ("Capital & Insiders", "#3b82f6", ["capital_discipline", "insider"]),
    ("Market & Estimates", "#f59e0b", ["momentum", "low_volatility", "estimate_revisions"]),
]


FUND_FACTOR_LABELS = {
    "cost": "Low Cost (expense ratio)",
    "performance": "Performance (3y · 5y annualized return)",
    "risk_adjusted": "Risk-Adjusted Return (return / volatility)",
    "low_volatility": "Low Volatility & Drawdown",
    "momentum": "Momentum (12-1)",
    "income": "Income (distribution yield)",
}


SHORT_FUND_FACTOR_LABELS = {
    "cost": "Low Fees",
    "performance": "Returns 3-5Y",
    "risk_adjusted": "Risk-Adj. Return",
    "low_volatility": "Low Volatility",
    "momentum": "Momentum",
    "income": "Yield",
}


RADAR_FUND_FACTOR_LABELS: dict[str, str] = {
    "cost":           "Low Fees",
    "performance":    "Returns",
    "risk_adjusted":  "Risk-Adj.",
    "low_volatility": "Low Vol",
    "momentum":       "Momentum",
    "income":         "Yield",
}


FUND_SCORECARD_GROUPS: list[tuple[str, str, list[str]]] = [
    ("Fees & Income", "#14b8a6", ["cost", "income"]),
    ("Performance", "#3b82f6", ["performance", "risk_adjusted"]),
    ("Risk & Trend", "#f59e0b", ["low_volatility", "momentum"]),
]


FUND_FACTOR_HELP: dict[str, str] = {
    "cost": (
        "Expense ratio ranked against the fund universe, inverted so cheaper "
        "funds score higher. Fees are the strongest documented predictor of "
        "long-run relative fund performance."
    ),
    "performance": (
        "Annualized 3-year and 5-year total returns (price/NAV based), each "
        "ranked cross-sectionally vs peer funds then averaged."
    ),
    "risk_adjusted": (
        "Sharpe-style ratio: trailing 12-month return divided by annualized "
        "volatility. Higher = more return per unit of risk taken, vs peers."
    ),
    "low_volatility": (
        "Inverse of annualized 12-month volatility plus max-drawdown "
        "protection, each ranked then averaged. Calmer funds rank higher."
    ),
    "momentum": (
        "Trailing 12-month return, skipping the most recent month to avoid "
        "short-term reversals. Higher rank = stronger persistent uptrend."
    ),
    "income": (
        "Trailing distribution/dividend yield ranked vs peer funds. Higher = "
        "more income paid out per dollar invested."
    ),
}


SECURITY_TYPE_BADGES = {
    "ETF": "ETF",
    "MUTUALFUND": "Mutual Fund",
}


FACTOR_HELP: dict[str, str] = {
    "value": (
        "Composite value rank: each of three sub-signals (earnings yield EBIT/EV, "
        "FCF yield, book-to-market) is ranked cross-sectionally then averaged. "
        "Higher = cheaper vs peers on multiple measures. The Graham ratio is "
        "tracked separately in the bargain score to avoid double-counting."
    ),
    "garp": (
        "Growth at a reasonable price (Peter Lynch): (earnings growth % + dividend yield %) "
        "divided by P/E. Higher = more growth and income per dollar of valuation. "
        "Falls back to 1/PEG when analyst growth estimates are unavailable."
    ),
    "quality": (
        "Composite quality rank in three de-correlated buckets: profitability "
        "(gross profitability, ROE, ROA, margin, ROIC), earnings quality (accruals), "
        "and financial strength (Piotroski). Higher = more profitable and cleaner business."
    ),
    "estimate_revisions": (
        "Zacks-style estimate-revision rank: revision agreement (% of FY1/FY2 "
        "upgrades), revision magnitude (90-day consensus EPS change), and the last "
        "quarterly earnings surprise. Higher = analysts are revising earnings up."
    ),
    "insider": (
        "Net open-market Form 4 buying (purchases minus sales) over the last 90 days "
        "as a fraction of market cap, ranked vs peers. Cluster buys by multiple "
        "officers also show as a badge. Grants and option exercises are excluded."
    ),
    "balance_sheet": (
        "Composite balance-sheet rank: net cash / market cap, low debt-to-equity "
        "(1/(1+D/E)), and Altman Z-Score are each ranked then averaged. "
        "Higher = more financial cushion and lower distress risk."
    ),
    "momentum": (
        "Trailing 12-month price return, skipping the most recent month to avoid "
        "short-term reversals. A higher rank means a stronger, more persistent uptrend."
    ),
    "low_volatility": (
        "Inverse of annualized 12-month return volatility (1/σ). "
        "Calmer, steadier stocks rank higher; high-swing names rank lower."
    ),
    "capital_discipline": (
        "Composite capital-discipline rank: shareholder yield (dividends + buybacks "
        "/ market cap) and investment factor (inverted asset growth) are each ranked "
        "then averaged. Higher = more cash returned, less balance-sheet expansion."
    ),
}
