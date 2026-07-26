"""Fund universe snapshot builder (US + Canadian ETFs and mutual funds)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.config import ROOT
from core.data import build_fund_raw_metrics, throttle
from core.fund_factors import compute_fund_factors

logger = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
FUND_SNAPSHOT_PATH = DATA_DIR / "fund_universe_snapshot.parquet"

# Metadata carried into the snapshot alongside factor columns.
_FUND_META_COLUMNS = [
    "ticker",
    "name",
    "quote_type",
    "category",
    "fund_family",
    "currency",
]

# Curated peer universe. Broad, liquid, well-known funds so cross-sectional
# percentiles are meaningful. Canadian listings use Yahoo's .TO suffix.
US_ETFS: list[str] = [
    # Broad US equity
    "SPY", "VOO", "IVV", "VTI", "ITOT", "SCHB", "RSP", "QQQ", "ONEQ", "DIA",
    "IWM", "VB", "VO", "IJH", "IJR", "SCHG", "SCHV",
    # Style
    "VUG", "VTV", "IWF", "IWD", "MTUM", "QUAL", "VLUE", "USMV", "SPLV", "MOAT",
    "COWZ",
    # Dividend / income
    "SCHD", "VYM", "VIG", "DGRO", "HDV", "NOBL", "SPHD", "JEPI", "JEPQ",
    # Sector
    "XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
    "VGT", "VHT", "VNQ", "SMH", "SOXX", "IBB", "XBI",
    # International
    "VXUS", "VEA", "VWO", "EFA", "EEM", "IEFA", "IEMG", "IXUS", "VT",
    # Fixed income
    "AGG", "BND", "TLT", "IEF", "SHY", "LQD", "HYG", "MUB", "TIP", "BNDX",
    # Commodities / alternatives
    "GLD", "IAU", "SLV",
]

US_MUTUAL_FUNDS: list[str] = [
    "VTSAX", "VFIAX", "VTIAX", "VBTLX", "VIGAX", "VVIAX", "VWELX", "VWINX",
    "VWENX", "VPMAX", "FXAIX", "FSKAX", "FTIHX", "FXNAX", "FCNTX", "FMAGX",
    "FBGRX", "FDGRX", "SWPPX", "SWTSX", "SWISX", "SWAGX", "DODGX", "DODFX",
    "DODIX", "PRWCX", "TRBCX", "PRGFX", "AGTHX", "ANWPX", "AWSHX", "ABALX",
]

CANADIAN_ETFS: list[str] = [
    # Broad Canadian equity
    "XIU.TO", "XIC.TO", "VCN.TO", "ZCN.TO", "HXT.TO",
    # US equity (CAD-listed)
    "VFV.TO", "VSP.TO", "XUS.TO", "ZSP.TO", "XSP.TO", "XQQ.TO", "ZQQ.TO",
    "VUN.TO", "XUU.TO",
    # Asset allocation
    "XEQT.TO", "VEQT.TO", "XGRO.TO", "VGRO.TO", "XBAL.TO", "VBAL.TO",
    "ZBAL.TO", "ZGRO.TO", "VCNS.TO", "XCNS.TO",
    # International
    "XEF.TO", "XEC.TO", "VIU.TO", "VEE.TO", "ZEA.TO", "XAW.TO", "VXC.TO",
    # Dividend / income
    "VDY.TO", "XDV.TO", "XEI.TO", "ZDV.TO", "CDZ.TO", "ZWB.TO", "ZEB.TO",
    # Sector / factor
    "XIT.TO", "TEC.TO", "ZLB.TO", "XST.TO", "XFN.TO", "XEG.TO", "XRE.TO",
    # Fixed income
    "ZAG.TO", "XBB.TO", "VAB.TO", "XSB.TO", "ZFL.TO", "XHY.TO", "VSB.TO",
]

# Canadian mutual funds. Yahoo lists these under opaque Morningstar-style IDs
# (fund-code symbols like MAW104/TDB902 are not carried); all verified live.
CANADIAN_MUTUAL_FUNDS: list[str] = [
    "0P0000MOFR.TO",  # Mawer Global Equity F
    "0P0000820B.TO",  # Mawer Canadian Equity Series O
    "0P0000820E.TO",  # Mawer Canadian Bond Series O
    "0P0000A0F2.TO",  # RBC Balanced Fund D
    "0P0000A0F6.TO",  # RBC Canadian Dividend Fund D
    "0P0000IL31.TO",  # PH&N Balanced Fund A
    "0P000074R6.TO",  # Fidelity Canadian Growth Company Series B
    "0P0000Q5PH.TO",  # Beutel Goodman Canadian Equity Class B
    "0P0001HFBC.TO",  # TD Canadian Index Fund O-Series
    "0P0000720M.TO",  # TD Dividend Growth Series A
    "0P0000NDSP.TO",  # Mackenzie Ivy Foreign Equity Series G
    "0P0000737Y.TO",  # Dynamic Power Global Growth Class
]


def default_fund_universe() -> list[str]:
    return US_ETFS + US_MUTUAL_FUNDS + CANADIAN_ETFS + CANADIAN_MUTUAL_FUNDS


def fund_snapshot_path() -> Path:
    return FUND_SNAPSHOT_PATH


def load_fund_universe_snapshot() -> pd.DataFrame | None:
    if not FUND_SNAPSHOT_PATH.exists():
        return None
    try:
        return pd.read_parquet(FUND_SNAPSHOT_PATH)
    except Exception as exc:
        logger.warning("Failed to load fund universe snapshot: %s", exc)
        return None


def build_fund_universe_snapshot(
    tickers: list[str] | None = None,
    max_tickers: int | None = None,
    throttle_seconds: float = 0.25,
) -> pd.DataFrame:
    """Build cross-sectional factor snapshot for the fund universe."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    universe = tickers or default_fund_universe()
    if max_tickers:
        universe = universe[:max_tickers]

    rows = []
    for i, ticker in enumerate(universe):
        try:
            raw = build_fund_raw_metrics(ticker)
            if raw.get("price") is None:
                logger.warning("Skipping %s: no price data", ticker)
                continue
            factors = compute_fund_factors(raw)
            row = {col: raw.get(col) for col in _FUND_META_COLUMNS}
            row["ticker"] = ticker.upper()
            row.update(factors)
            rows.append(row)
            if (i + 1) % 25 == 0:
                logger.info("Processed %d / %d funds", i + 1, len(universe))
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
        throttle(throttle_seconds)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["snapshot_date"] = datetime.now(timezone.utc).isoformat()
        df.to_parquet(FUND_SNAPSHOT_PATH, index=False)
        logger.info("Saved fund universe snapshot with %d funds to %s", len(df), FUND_SNAPSHOT_PATH)
    return df


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build fund universe factor snapshot")
    parser.add_argument("--max", type=int, default=None, help="Max funds to process")
    args = parser.parse_args()

    build_fund_universe_snapshot(max_tickers=args.max)
