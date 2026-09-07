#!/usr/bin/env python3
"""Refresh Damodaran ERP / industry WACC reference files.

Damodaran URLs change; pass a downloaded spreadsheet or CSV with ``--from-file``.
Committed outputs live under ``data/reference/``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.reference import (  # noqa: E402
    INDUSTRY_PATH,
    ERP_PATH,
    parse_damodaran_erp,
    parse_damodaran_industry,
    write_reference_csv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh Damodaran reference CSVs")
    parser.add_argument(
        "--from-file",
        dest="from_file",
        required=True,
        help="Local histimpl.xls / wacc.xls / CSV (URLs change yearly)",
    )
    parser.add_argument(
        "--kind",
        choices=["erp", "industry", "auto"],
        default="auto",
        help="Which table to parse (auto = detect from columns / filename)",
    )
    args = parser.parse_args()
    path = Path(args.from_file)
    if not path.exists():
        logger.error("File not found: %s", path)
        return 1

    kind = args.kind
    name = path.name.lower()
    if kind == "auto":
        if "wacc" in name or "industry" in name:
            kind = "industry"
        else:
            kind = "erp"

    if kind == "erp":
        df = parse_damodaran_erp(path)
        dest = ERP_PATH
    else:
        df = parse_damodaran_industry(path)
        dest = INDUSTRY_PATH

    write_reference_csv(df, dest)
    logger.info("Wrote %d rows to %s", len(df), dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
