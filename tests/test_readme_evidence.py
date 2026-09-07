"""README evidence block must stay honest vs committed artifacts."""

from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
RESULTS = ROOT / "backtest" / "results"


def test_evidence_markers_present():
    text = README.read_text(encoding="utf-8")
    assert "<!-- evidence:start -->" in text
    assert "<!-- evidence:end -->" in text
    block = text.split("<!-- evidence:start -->", 1)[1].split("<!-- evidence:end -->", 1)[0]
    assert "run_id" in block


_NUMERIC_IC_CLAIM = re.compile(
    r"\bIC\b(?:\s+was|\s+of|\s*[=~≈]|[^.\n]{0,50}[~≈]\s*\d)",
    re.IGNORECASE,
)
_NUMERIC_CAGR_CLAIM = re.compile(
    r"(?:\bCAGR\b[^.\n]{0,40}\d|\d[\d.]*%?\s+CAGR)",
    re.IGNORECASE,
)


def test_evidence_run_id_matches_artifacts_when_present():
    text = README.read_text(encoding="utf-8")
    block = text.split("<!-- evidence:start -->", 1)[1].split("<!-- evidence:end -->", 1)[0]
    json_paths = list(RESULTS.glob("*.json"))
    artifact_ids = []
    for path in json_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("run_id"):
            artifact_ids.append(str(payload["run_id"]))

    run_id_match = re.search(r"run_id:\s*(\S+)", block)
    run_id_value = run_id_match.group(1) if run_id_match else ""
    is_pending = "pending" in run_id_value.lower()
    has_numeric_claim = bool(
        _NUMERIC_IC_CLAIM.search(block) or _NUMERIC_CAGR_CLAIM.search(block)
    )
    if has_numeric_claim and is_pending:
        raise AssertionError(
            "README evidence block cites a numeric IC/CAGR while run_id is pending; "
            "regenerate artifacts with `python -m backtest.run pipeline` or remove the numeric claim"
        )
    if json_paths and not artifact_ids:
        raise AssertionError(
            "committed backtest JSON artifacts have no run_id; "
            "run `python -m backtest.run pipeline` or delete stale files"
        )
    if not artifact_ids:
        assert "pending" in block.lower() or "pre-companyfacts" in block.lower()
        return
    unique = set(artifact_ids)
    assert len(unique) == 1
    run_id = unique.pop()
    assert run_id in block
