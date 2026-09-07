"""SMTP email sender for watchlist and universe scorecard reports."""

from __future__ import annotations

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

from core.config import get_thresholds
from core.data import currency_symbol


def _clean(value: str | None) -> str:
    return (value or "").strip()


def _get_smtp_config(config: dict[str, Any]) -> dict[str, str]:
    email_cfg = config.get("email", {})
    from_address = _clean(os.environ.get("SMTP_FROM") or email_cfg.get("from_address", ""))
    # Gmail requires logging in as the sending account. Prefer FROM over a stale
    # SMTP_USERNAME secret that may still point at an old mailbox.
    username = from_address or _clean(os.environ.get("SMTP_USERNAME"))
    return {
        "host": _clean(os.environ.get("SMTP_HOST")) or email_cfg.get("smtp_host", "smtp.gmail.com"),
        "port": str(_clean(os.environ.get("SMTP_PORT")) or email_cfg.get("smtp_port", 587)),
        "from_address": from_address,
        "to_address": _clean(os.environ.get("SMTP_TO") or email_cfg.get("to_address", "")),
        "username": username,
        "password": _clean(os.environ.get("SMTP_PASSWORD")).replace(" ", ""),
    }


def smtp_config_status(config: dict[str, Any]) -> tuple[bool, str]:
    """Return (ready, reason). Does not expose secrets."""
    smtp = _get_smtp_config(config)
    if not smtp["from_address"]:
        return False, "missing from address (set SMTP_FROM or email.from_address in config.yaml)"
    if not smtp["to_address"]:
        return False, "missing to address (set SMTP_TO or email.to_address in config.yaml)"
    if not smtp["password"]:
        return False, "missing SMTP password (set SMTP_PASSWORD secret or env var)"
    return True, (
        f"from={smtp['from_address']} to={smtp['to_address']} "
        f"host={smtp['host']} user={smtp['username']}"
    )


def email_is_enabled(config: dict[str, Any]) -> bool:
    email_cfg = config.get("email", {})
    return bool(email_cfg.get("enabled", False) or os.environ.get("SMTP_PASSWORD"))


def _fmt_price(analysis: dict[str, Any]) -> str:
    price = analysis.get("price")
    if price is None:
        return "N/A"
    try:
        return f"{currency_symbol(analysis.get('currency'))}{float(price):,.2f}"
    except (TypeError, ValueError):
        return "N/A"


def _fmt_score(value: float | None, *, suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{value:.1f}{suffix}"


def _decision_sort_key(result: dict[str, Any]) -> tuple:
    decision = result.get("decision") or {}
    label = decision.get("label")
    if not label:
        label = "Accumulate" if result.get("is_good_buy") else "Avoid"
    rank = {"Accumulate": 0, "Watch": 1, "Avoid": 2}.get(label, 3)
    pct = decision.get("pct_to_buy")
    # Watch: closer to buy-below (higher pct_to_buy, e.g. -5 before -20).
    proximity = float(pct) if pct is not None else -999.0
    return (rank, -proximity, result.get("ticker", ""))


def _sort_scorecard_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(results, key=_decision_sort_key)


def _failed_gate_name(analysis: dict[str, Any]) -> str:
    decision = analysis.get("decision") or {}
    for gate in decision.get("gates") or []:
        if not gate.get("passed", True):
            return str(gate.get("name") or "")
    return ""


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def format_scorecard_email(
    results: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    title: str,
    subtitle: str = "",
    snapshot_date: str | None = None,
    run_id: str | None = None,
    hurdle_rate: float | None = None,
) -> tuple[str, str]:
    """Return (subject, html_body) for a full scorecard email."""
    thresholds = get_thresholds(config)
    buy_count = sum(
        1
        for r in results
        if r.get("is_good_buy") or (r.get("decision") or {}).get("label") == "Accumulate"
    )
    subject = f"Stock Metrics: {title} ({buy_count} Accumulate / {len(results)} total)"

    rows = []
    for a in _sort_scorecard_results(results):
        decision = a.get("decision") or {}
        label = decision.get("label") or ("Accumulate" if a.get("is_good_buy") else "Avoid")
        color = {"Accumulate": "#166534", "Watch": "#92400e", "Avoid": "#6b7280"}.get(label, "#6b7280")
        bg = {"Accumulate": "#dcfce7", "Watch": "#fef3c7", "Avoid": "#f3f4f6"}.get(label, "#f3f4f6")
        flags = a.get("value_trap_flags") or decision.get("flags") or []
        flag_n = sum(1 for f in flags if f.get("triggered"))
        quality = a.get("quality_percentile")
        if quality is None:
            quality = a.get("quality_score")
        buy_below = decision.get("buy_below_price")
        why = a.get("why") or _failed_gate_name(a) or (decision.get("timing_context") or {}).get("hint") or ""
        grade = (a.get("data_quality") or {}).get("grade") or ""
        buy_below_txt = _fmt_price({**a, "price": buy_below}) if buy_below is not None else "N/A"

        rows.append(
            f"<tr>"
            f"<td><b>{a.get('ticker', '')}</b></td>"
            f"<td align='center' style='background:{bg};color:{color};font-weight:700;'>{label}</td>"
            f"<td align='right'>{_fmt_price(a)}</td>"
            f"<td align='right'>{buy_below_txt}</td>"
            f"<td align='right'>{_fmt_pct(decision.get('pct_to_buy'))}</td>"
            f"<td align='right'>{_fmt_score(quality)}</td>"
            f"<td align='center'>{flag_n}</td>"
            f"<td>{why}</td>"
            f"<td align='center'>{grade}</td>"
            f"</tr>"
        )

    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    table_body = (
        "".join(rows)
        if rows
        else "<tr><td colspan='9'><i>No tickers scored.</i></td></tr>"
    )

    footer_bits = ["Generated by financial-tools."]
    if snapshot_date:
        footer_bits.append(f"snapshot {snapshot_date}")
    if run_id:
        footer_bits.append(f"run_id {run_id}")
    if hurdle_rate is not None:
        footer_bits.append(f"hurdle {100 * float(hurdle_rate):.1f}%")
    footer = " · ".join(footer_bits)

    html = f"""
    <html><body style="font-family:Arial,sans-serif;color:#1f2937;">
    <h2>{title}</h2>
    {subtitle_html}
    <p style="color:#6b7280;font-size:14px;">
    Intrinsic decision: Accumulate / Watch / Avoid.
    Buy-below = min(DCF base × (1 − MoS), DCF bear). Grade C and distress block Accumulate.
    Legacy composite ≥ {thresholds['composite_min']} / bargain ≥ {thresholds['bargain_min']} still shown in dashboard when decision.mode=legacy.
    </p>
    <table border="1" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;max-width:1100px;">
    <tr style="background:#f9fafb;">
      <th align="left">Ticker</th>
      <th align="center">Decision</th>
      <th align="right">Price</th>
      <th align="right">Buy-below</th>
      <th align="right">% to buy</th>
      <th align="right">Quality</th>
      <th align="center">Flags</th>
      <th align="left">Why</th>
      <th align="center">Grade</th>
    </tr>
    {table_body}
    </table>
    <p style="color:#9ca3af;font-size:12px;margin-top:16px;">
    <i>{footer}</i>
    </p>
    </body></html>
    """
    return subject, html


def format_alert_email(alerts: list[dict[str, Any]], config: dict[str, Any]) -> tuple[str, str]:
    """Legacy wrapper — formats good-buy alerts as a scorecard."""
    return format_scorecard_email(
        alerts,
        config,
        title=f"Good Buy Alerts ({len(alerts)})",
    )


def send_email(subject: str, html_body: str, config: dict[str, Any]) -> tuple[bool, str]:
    """Send HTML email. Returns (success, message)."""
    ready, status = smtp_config_status(config)
    if not ready:
        return False, status

    smtp = _get_smtp_config(config)
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp["from_address"]
        msg["To"] = smtp["to_address"]
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(smtp["host"], int(smtp["port"])) as server:
            server.starttls()
            server.login(smtp["username"], smtp["password"])
            server.sendmail(smtp["from_address"], [smtp["to_address"]], msg.as_string())
        return True, f"sent to {smtp['to_address']}"
    except smtplib.SMTPAuthenticationError:
        return False, (
            f"SMTP authentication failed for {smtp['username']!r} "
            "(use a Gmail app password for the FROM account; paste without spaces)"
        )
    except Exception as exc:
        return False, f"SMTP error: {exc}"
