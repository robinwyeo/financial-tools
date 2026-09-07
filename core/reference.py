"""Damodaran ERP / industry-cost-of-capital reference files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT / "data" / "reference"
ERP_PATH = REFERENCE_DIR / "damodaran_erp.csv"
INDUSTRY_PATH = REFERENCE_DIR / "damodaran_industry.csv"


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    return out


def parse_damodaran_erp(path: Path | str) -> pd.DataFrame:
    """Parse histimpl-style ERP history into year / implied_erp / riskfree_rate."""
    df = _norm_cols(_read_table(Path(path)))
    year_col = next((c for c in df.columns if c in {"year", "date", "period"}), None)
    erp_col = next(
        (c for c in df.columns if c in {"implied_erp", "erp", "implied_premium", "equity_risk_premium"}),
        None,
    )
    rf_col = next(
        (c for c in df.columns if c in {"riskfree_rate", "risk_free", "t_bond_rate", "rf"}),
        None,
    )
    if year_col is None or erp_col is None:
        raise ValueError(f"ERP file missing year/implied_erp columns: {list(df.columns)}")
    out = pd.DataFrame(
        {
            "year": pd.to_numeric(df[year_col], errors="coerce"),
            "implied_erp": pd.to_numeric(df[erp_col], errors="coerce"),
        }
    )
    if rf_col:
        out["riskfree_rate"] = pd.to_numeric(df[rf_col], errors="coerce")
        out.loc[out["riskfree_rate"] > 1, "riskfree_rate"] = out["riskfree_rate"] / 100.0
    out.loc[out["implied_erp"] > 1, "implied_erp"] = out["implied_erp"] / 100.0
    out = out.dropna(subset=["year", "implied_erp"])
    out["year"] = out["year"].astype(int)
    out["source"] = "damodaran"
    return out.sort_values("year").reset_index(drop=True)


def parse_damodaran_industry(path: Path | str) -> pd.DataFrame:
    """Parse industry WACC table into industry / cost_of_equity / wacc / beta."""
    df = _norm_cols(_read_table(Path(path)))
    ind_col = next((c for c in df.columns if c in {"industry", "industry_name", "name"}), None)
    if ind_col is None:
        raise ValueError(f"Industry file missing industry column: {list(df.columns)}")
    rename = {
        "cost_of_equity": "cost_of_equity",
        "costofequity": "cost_of_equity",
        "cost_of_debt": "cost_of_debt",
        "costofdebt": "cost_of_debt",
        "wacc": "wacc",
        "beta": "beta",
    }
    out = pd.DataFrame({"industry": df[ind_col].astype(str)})
    for src, dest in rename.items():
        if src in df.columns:
            out[dest] = pd.to_numeric(df[src], errors="coerce")
    return out.dropna(subset=["industry"]).reset_index(drop=True)


def write_reference_csv(df: pd.DataFrame, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def load_erp_table() -> pd.DataFrame:
    if not ERP_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ERP_PATH)


def latest_erp() -> float | None:
    df = load_erp_table()
    if df.empty or "implied_erp" not in df.columns:
        return None
    return float(pd.to_numeric(df["implied_erp"], errors="coerce").dropna().iloc[-1])
