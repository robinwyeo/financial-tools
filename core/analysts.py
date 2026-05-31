"""Analyst recommendation aggregation."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

RATING_MAP = {
    "strong_buy": 5,
    "buy": 4,
    "outperform": 4,
    "overweight": 4,
    "hold": 3,
    "neutral": 3,
    "market perform": 3,
    "equal-weight": 3,
    "underperform": 2,
    "sell": 1,
    "strong_sell": 1,
}


def _rating_to_score(label: str | None) -> float | None:
    if not label:
        return None
    key = str(label).lower().strip().replace(" ", "_")
    if key in RATING_MAP:
        return float(RATING_MAP[key])
    for k, v in RATING_MAP.items():
        if k in key or key in k:
            return float(v)
    return None


def _score_to_label(score: float) -> str:
    if score >= 4.5:
        return "Strong Buy"
    if score >= 3.5:
        return "Buy"
    if score >= 2.5:
        return "Hold"
    if score >= 1.5:
        return "Underperform"
    return "Sell"


def aggregate_analyst_data(raw: dict[str, Any]) -> dict[str, Any]:
    """
    Aggregate analyst recommendations, price targets, and implied upside.
    """
    info_key = raw.get("recommendation_key")
    price = raw.get("price")
    target_mean = raw.get("target_mean")
    target_low = raw.get("target_low")
    target_high = raw.get("target_high")
    num_analysts = raw.get("num_analysts")
    recs: pd.DataFrame = raw.get("recommendations", pd.DataFrame())

    # Distribution from recommendation history if available
    buy_count = hold_count = sell_count = 0
    recent_actions: list[dict] = []

    if recs is not None and not recs.empty:
        df = recs.copy()
        col_map = {c.lower(): c for c in df.columns}

        # New yfinance format: period-based aggregate counts
        # Columns: period, strongBuy, buy, hold, sell, strongSell
        if "strongbuy" in col_map and "buy" in col_map and "hold" in col_map:
            period_col = col_map.get("period")
            if period_col is not None:
                current_rows = df[df[period_col] == "0m"]
                row = current_rows.iloc[0] if not current_rows.empty else df.iloc[0]
            else:
                row = df.iloc[0]
            buy_count = int(row.get(col_map["strongbuy"], 0) or 0) + int(row.get(col_map["buy"], 0) or 0)
            hold_count = int(row.get(col_map["hold"], 0) or 0)
            sell_count = (
                int(row.get(col_map["sell"], 0) or 0) if "sell" in col_map else 0
            ) + (
                int(row.get(col_map["strongsell"], 0) or 0) if "strongsell" in col_map else 0
            )
        else:
            # Legacy format: individual recommendation rows with toGrade / rating column
            grade_col = col_map.get("tograde") or col_map.get("to_grade") or col_map.get("rating")
            firm_col = col_map.get("firm")
            action_col = col_map.get("action")
            date_col = col_map.get("date") or col_map.get("index")

            for _, row in df.tail(50).iterrows():
                grade = str(row.get(grade_col, "")) if grade_col else ""
                score = _rating_to_score(grade)
                if score is not None:
                    if score >= 4:
                        buy_count += 1
                    elif score >= 3:
                        hold_count += 1
                    else:
                        sell_count += 1

                recent_actions.append(
                    {
                        "date": str(row.get(date_col, "")) if date_col else "",
                        "firm": str(row.get(firm_col, "")) if firm_col else "",
                        "rating": grade,
                        "action": str(row.get(action_col, "")) if action_col else "",
                    }
                )

    total = buy_count + hold_count + sell_count
    if total == 0 and info_key:
        # Fallback from consensus key
        score = _rating_to_score(info_key)
        if score is not None:
            if score >= 4:
                buy_count = 1
            elif score >= 3:
                hold_count = 1
            else:
                sell_count = 1
            total = 1

    mean_rating = None
    if total > 0:
        mean_rating = (buy_count * 4.5 + hold_count * 3.0 + sell_count * 1.5) / total
    elif info_key:
        mean_rating = _rating_to_score(info_key)

    consensus_label = _score_to_label(mean_rating) if mean_rating else "Unknown"

    implied_upside_pct = None
    if price and target_mean and price > 0:
        implied_upside_pct = ((target_mean / price) - 1.0) * 100

    upgrades = sum(1 for a in recent_actions if "up" in a.get("action", "").lower() or "raise" in a.get("action", "").lower())
    downgrades = sum(
        1 for a in recent_actions if "down" in a.get("action", "").lower() or "lower" in a.get("action", "").lower()
    )

    return {
        "consensus_label": consensus_label,
        "mean_rating_score": mean_rating,
        "recommendation_key": info_key,
        "num_analysts": num_analysts,
        "buy_count": buy_count,
        "hold_count": hold_count,
        "sell_count": sell_count,
        "target_mean": target_mean,
        "target_low": target_low,
        "target_high": target_high,
        "implied_upside_pct": implied_upside_pct,
        "recent_upgrades": upgrades,
        "recent_downgrades": downgrades,
        "recent_actions": recent_actions[-10:],
    }


def norm_cdf(z: float) -> float:
    """Standard normal CDF without scipy."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
