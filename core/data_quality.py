"""Yahoo vs EDGAR reconciliation and a letter-grade for live fundamentals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any

REQUIRED_FIELDS: tuple[str, ...] = (
    "price",
    "market_cap",
    "revenue",
    "ebit",
    "net_income",
    "operating_cashflow",
    "total_assets",
    "total_debt",
    "total_cash",
    "book_equity",
    "shares_outstanding",
)

# Yahoo raw field -> (edgar flat field, relative tolerance)
# shares_outstanding is compared for a RECONCILE warning only: EDGAR often
# reports a single share class (e.g. BRK Class A) that must not overwrite
# listing-class shares used for per-share metrics / Graham.
COMPARE_SPEC: dict[str, tuple[str, float]] = {
    "revenue": ("revenue", 0.10),
    "net_income": ("net_income", 0.10),
    "operating_cashflow": ("operating_cashflow", 0.10),
    "total_debt": ("total_debt", 0.10),
    "total_cash": ("total_cash", 0.10),
    "book_equity": ("book_equity", 0.10),
}

SHARES_RECONCILE_TOL = 0.05


@dataclass
class DataQuality:
    grade: str
    substitutions: list[dict[str, Any]] = field(default_factory=list)
    presence_pct: float = 0.0
    reasons: list[str] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    source: str = "yahoo"
    edgar_available: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def warning_message(item: Any) -> str:
    """Display text for a structured or legacy string warning."""
    if isinstance(item, dict):
        return str(item.get("message") or item.get("code") or "")
    return str(item)


def _as_warning(item: Any, *, code: str = "INFO", severity: str = "warning") -> dict[str, str]:
    if isinstance(item, dict) and item.get("message"):
        return {
            "code": str(item.get("code") or code),
            "message": str(item["message"]),
            "severity": str(item.get("severity") or severity),
        }
    return {"code": code, "message": str(item), "severity": severity}


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    if out != out:  # NaN
        return None
    return out


def _pct_diff(yahoo: float, edgar: float) -> float:
    denom = max(abs(edgar), abs(yahoo), 1e-9)
    return abs(yahoo - edgar) / denom


def presence_pct(raw: dict[str, Any], required: tuple[str, ...] = REQUIRED_FIELDS) -> float:
    if not required:
        return 1.0
    present = sum(1 for key in required if _safe_float(raw.get(key)) is not None)
    return present / len(required)


def load_edgar_snapshot(ticker: str) -> dict[str, float] | None:
    """TTM/MRQ EDGAR fundamentals for a ticker, or None if not a US filer / fetch failed."""
    from core.edgar_facts import (
        fetch_companyfacts_for_ticker,
        flatten_fundamentals,
        fundamentals_as_of_structured,
    )
    from core.sec import ticker_to_cik

    if ticker_to_cik(ticker) is None:
        return None
    facts = fetch_companyfacts_for_ticker(ticker)
    if facts is None or facts.empty:
        return None
    structured = fundamentals_as_of_structured(date.today(), facts, ticker=ticker)
    flat = flatten_fundamentals(structured)
    return flat or None


def reconcile(
    raw: dict[str, Any],
    edgar_ttm: dict[str, float] | None,
    *,
    ticker: str | None = None,
) -> DataQuality:
    """Compare Yahoo live fields to EDGAR TTM/MRQ and grade the snapshot."""
    warnings = [_as_warning(w) for w in (raw.get("data_warnings") or [])]
    substitutions: list[dict[str, Any]] = []
    reasons: list[str] = []
    edgar_available = bool(edgar_ttm)

    if not edgar_available:
        reasons.append("no EDGAR filer")
        warnings.append(
            {
                "code": "NO_EDGAR",
                "message": "no EDGAR filer",
                "severity": "info",
            }
        )

    # Insurers/holding companies: Yahoo cash includes investments; EDGAR's
    # cash tag is cash-only. Do not treat that mismatch as a substitution.
    skip_cash = str(raw.get("sector") or "") == "Financial Services"

    if edgar_ttm:
        for yahoo_field, (edgar_field, tol) in COMPARE_SPEC.items():
            if yahoo_field == "total_cash" and skip_cash:
                continue
            yv = _safe_float(raw.get(yahoo_field))
            ev = _safe_float(edgar_ttm.get(edgar_field))
            if yv is None or ev is None:
                continue
            diff = _pct_diff(yv, ev)
            if diff > tol:
                substitutions.append(
                    {
                        "field": yahoo_field,
                        "yahoo": yv,
                        "edgar": ev,
                        "pct_diff": diff,
                        "chosen": "edgar",
                    }
                )
                warnings.append(
                    {
                        "code": "RECONCILE",
                        "message": (
                            f"{yahoo_field}: Yahoo {yv:.4g} vs EDGAR {ev:.4g} "
                            f"({diff:.0%} off); using EDGAR"
                        ),
                        "severity": "warning",
                    }
                )

        yv = _safe_float(raw.get("shares_outstanding"))
        ev = _safe_float(edgar_ttm.get("shares_outstanding"))
        if yv is not None and ev is not None:
            diff = _pct_diff(yv, ev)
            if diff > SHARES_RECONCILE_TOL:
                warnings.append(
                    {
                        "code": "RECONCILE",
                        "message": (
                            f"shares_outstanding: Yahoo {yv:.4g} vs EDGAR {ev:.4g} "
                            f"({diff:.0%} off); keeping listing shares"
                        ),
                        "severity": "warning",
                    }
                )

    present = presence_pct(raw)
    currency_inconsistent = _currency_inconsistent(raw)
    if currency_inconsistent:
        reasons.append("price/currency inconsistency")
        warnings.append(
            {
                "code": "FX_MISSING",
                "message": "listing currency differs from financials with no FX conversion",
                "severity": "error",
            }
        )

    grade = _grade(
        n_sub=len(substitutions),
        presence=present,
        edgar_available=edgar_available,
        currency_inconsistent=currency_inconsistent,
    )
    if grade == "A":
        reasons.append("Yahoo and EDGAR agree")
    elif substitutions:
        reasons.append(f"{len(substitutions)} field substitution(s)")
    if present < 0.90:
        reasons.append(f"field presence {present:.0%}")

    source = "yahoo"
    if edgar_available and substitutions:
        source = "mixed"

    return DataQuality(
        grade=grade,
        substitutions=substitutions,
        presence_pct=present,
        reasons=reasons,
        warnings=warnings,
        source=source,
        edgar_available=edgar_available,
    )


def _currency_inconsistent(raw: dict[str, Any]) -> bool:
    listing = (raw.get("currency") or "").upper() or None
    financial = (raw.get("financial_currency") or "").upper() or None
    if not listing or not financial or listing == financial:
        return False
    fx = raw.get("fx_to_financial")
    return fx is None


def _grade(
    *,
    n_sub: int,
    presence: float,
    edgar_available: bool,
    currency_inconsistent: bool,
) -> str:
    if currency_inconsistent or n_sub > 2 or presence < 0.75:
        return "C"
    if (not edgar_available) or n_sub > 0 or presence < 0.90:
        return "B"
    return "A"


def apply_substitutions(raw: dict[str, Any], quality: DataQuality) -> dict[str, Any]:
    """Overwrite Yahoo fields with the chosen EDGAR values."""
    for sub in quality.substitutions:
        field = sub.get("field")
        if field and field != "shares_outstanding":
            raw[field] = sub.get("edgar")
    return raw


def attach_data_quality(raw: dict[str, Any]) -> DataQuality:
    """Fetch EDGAR (if possible), grade, apply substitutions, rewrite warnings."""
    ticker = str(raw.get("ticker") or "")
    edgar: dict[str, float] | None = None
    if ticker:
        try:
            edgar = load_edgar_snapshot(ticker)
        except Exception:
            edgar = None
    quality = reconcile(raw, edgar, ticker=ticker or None)
    apply_substitutions(raw, quality)
    # Presence after substitution still uses the same keys; recompute for the record.
    quality.presence_pct = presence_pct(raw)
    raw["data_quality"] = quality.to_dict()
    raw["data_warnings"] = quality.warnings
    return quality
