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


def _is_period_aggregate_format(col_map: dict[str, str]) -> bool:
    return "strongbuy" in col_map and "buy" in col_map and "hold" in col_map


def _period_row(df: pd.DataFrame, col_map: dict[str, str], period: str) -> pd.Series | None:
    period_col = col_map.get("period")
    if period_col is None:
        return None
    rows = df[df[period_col] == period]
    if rows.empty:
        return None
    return rows.iloc[0]


def _bucket_counts(row: pd.Series, col_map: dict[str, str]) -> dict[str, int]:
    return {
        "strong_buy": int(row.get(col_map["strongbuy"], 0) or 0),
        "buy": int(row.get(col_map["buy"], 0) or 0),
        "hold": int(row.get(col_map["hold"], 0) or 0),
        "sell": int(row.get(col_map["sell"], 0) or 0) if "sell" in col_map else 0,
        "strong_sell": int(row.get(col_map["strongsell"], 0) or 0) if "strongsell" in col_map else 0,
    }


def _weighted_mean_from_buckets(counts: dict[str, int]) -> float | None:
    total = sum(counts.values())
    if total == 0:
        return None
    weighted = (
        counts["strong_buy"] * 5
        + counts["buy"] * 4
        + counts["hold"] * 3
        + counts["sell"] * 2
        + counts["strong_sell"] * 1
    )
    return weighted / total


def _buy_hold_sell_from_buckets(counts: dict[str, int]) -> tuple[int, int, int]:
    buy = counts["strong_buy"] + counts["buy"]
    hold = counts["hold"]
    sell = counts["sell"] + counts["strong_sell"]
    return buy, hold, sell


def recommendation_period_shift(
    recs: pd.DataFrame,
) -> tuple[float | None, int, int]:
    """
    Sentiment shift from yfinance period aggregates (0m vs -1m).
    Returns (revision_score, buy-side upgrades, buy-side downgrades).
    """
    if recs is None or recs.empty:
        return None, 0, 0

    col_map = {c.lower(): c for c in recs.columns}
    if not _is_period_aggregate_format(col_map):
        return None, 0, 0

    current = _period_row(recs, col_map, "0m")
    prior = _period_row(recs, col_map, "-1m")
    if current is None:
        current = recs.iloc[0]
    if prior is None:
        return None, 0, 0

    current_counts = _bucket_counts(current, col_map)
    prior_counts = _bucket_counts(prior, col_map)

    current_mean = _weighted_mean_from_buckets(current_counts)
    prior_mean = _weighted_mean_from_buckets(prior_counts)
    revision_score = None
    if current_mean is not None and prior_mean is not None:
        revision_score = float(current_mean - prior_mean)

    buy_current, _, sell_current = _buy_hold_sell_from_buckets(current_counts)
    buy_prior, _, sell_prior = _buy_hold_sell_from_buckets(prior_counts)
    net_bullish_current = buy_current - sell_current
    net_bullish_prior = buy_prior - sell_prior
    net_change = net_bullish_current - net_bullish_prior
    upgrades = max(0, net_change)
    downgrades = max(0, -net_change)
    return revision_score, upgrades, downgrades


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

    buy_count = hold_count = sell_count = 0
    recent_actions: list[dict] = []
    upgrades = downgrades = 0

    if recs is not None and not recs.empty:
        df = recs.copy()
        col_map = {c.lower(): c for c in df.columns}

        if _is_period_aggregate_format(col_map):
            period_col = col_map.get("period")
            if period_col is not None:
                current_rows = df[df[period_col] == "0m"]
                row = current_rows.iloc[0] if not current_rows.empty else df.iloc[0]
            else:
                row = df.iloc[0]
            counts = _bucket_counts(row, col_map)
            buy_count, hold_count, sell_count = _buy_hold_sell_from_buckets(counts)
            _, upgrades, downgrades = recommendation_period_shift(df)
        else:
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

            upgrades = sum(
                1
                for a in recent_actions
                if "up" in a.get("action", "").lower() or "raise" in a.get("action", "").lower()
            )
            downgrades = sum(
                1
                for a in recent_actions
                if "down" in a.get("action", "").lower() or "lower" in a.get("action", "").lower()
            )

    total = buy_count + hold_count + sell_count
    if total == 0 and info_key:
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
    if recs is not None and not recs.empty:
        col_map = {c.lower(): c for c in recs.columns}
        if _is_period_aggregate_format(col_map):
            period_col = col_map.get("period")
            row = recs.iloc[0]
            if period_col is not None:
                current_rows = recs[recs[period_col] == "0m"]
                if not current_rows.empty:
                    row = current_rows.iloc[0]
            mean_rating = _weighted_mean_from_buckets(_bucket_counts(row, col_map))

    if mean_rating is None and total > 0:
        mean_rating = (buy_count * 4.5 + hold_count * 3.0 + sell_count * 1.5) / total
    elif mean_rating is None and info_key:
        mean_rating = _rating_to_score(info_key)

    consensus_label = _score_to_label(mean_rating) if mean_rating else "Unknown"

    implied_upside_pct = None
    if price and target_mean and price > 0:
        implied_upside_pct = ((target_mean / price) - 1.0) * 100

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
