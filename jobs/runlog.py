"""Run summaries for monthly/weekly jobs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "data" / "runs"

FAILURE_RATIO_LIMIT = 0.10


def run_id_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_run_summary(kind: str, payload: dict[str, Any], *, run_id: str | None = None) -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rid = run_id or run_id_now()
    path = RUNS_DIR / f"{rid}.json"
    body = {"kind": kind, "run_id": rid, **payload}
    path.write_text(json.dumps(body, indent=2, default=str), encoding="utf-8")
    latest = RUNS_DIR / "latest.json"
    latest.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def failure_ratio(failed: int, attempted: int) -> float:
    if attempted <= 0:
        return 0.0
    return failed / attempted


def exceeds_failure_threshold(failed: int, attempted: int, limit: float = FAILURE_RATIO_LIMIT) -> bool:
    return failure_ratio(failed, attempted) > limit
