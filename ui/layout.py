"""Page CSS, equal-height JS, and card shells."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import streamlit as st
import streamlit.components.v1 as components

def _dashboard_row_anchor(row: int) -> None:
    """Marker for JS equal-height pass on the following st.columns() row."""
    st.markdown(
        f'<div id="stock-row-{row}-anchor" class="stock-dashboard-row-anchor" '
        f'aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )


@contextmanager
def _card_shell(bordered: bool) -> Iterator[None]:
    """Dashboard cards use column borders; standalone cards keep st.container(border=True)."""
    if bordered:
        with st.container(border=True):
            yield
    else:
        yield


def inject_css() -> None:
    st.markdown(
        """
        <style>
        /* Page background */
        [data-testid="stAppViewContainer"] { background-color: #f0f4f8; }
        [data-testid="stHeader"] { background-color: #f0f4f8; }
        .main .block-container {
            padding-top: 0.5rem;
            padding-bottom: 1rem;
            padding-left: 1.5rem;
            padding-right: 1.5rem;
            max-width: 100%;
        }

        /* Cards (st.container with border=True) */
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
            border: 1px solid #e5e7eb !important;
            box-shadow: 0 1px 6px rgba(0, 0, 0, 0.07) !important;
            background: white !important;
            overflow: hidden !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"] > div {
            background: white !important;
        }

        .dashboard-card-body {
            display: flex;
            flex-direction: column;
            flex: 1 1 auto;
            min-height: 100%;
        }
        .composite-score-card {
            justify-content: flex-start;
        }
        .composite-gauges-row {
            display: flex;
            gap: 0.45rem;
            align-items: flex-start;
            justify-content: center;
            width: 100%;
            padding: 0.15rem 0 0.1rem;
        }
        .composite-gauges-row .gauge-cell {
            flex: 1 1 0;
            min-width: 0;
        }
        .composite-gauges-row .gauge-title {
            font-size: 0.72rem;
            font-weight: 600;
            color: #6b7280;
            text-align: center;
            margin-bottom: 0.15rem;
            line-height: 1.2;
        }
        .factor-scorecard-card .factor-scorecard-grid {
            align-content: start;
        }
        .factor-scorecard-card .factor-scorecard-col {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
        }
        .analyst-consensus-card {
            justify-content: center;
            gap: 0.25rem;
        }
        .analyst-consensus-card .analyst-header-wrap {
            flex: 0 0 auto;
        }
        .analyst-consensus-card .analyst-targets {
            display: flex;
            gap: 0.35rem;
            margin-top: 0.3rem;
        }
        .analyst-consensus-card .analyst-target-pill {
            flex: 1;
            min-width: 0;
            text-align: center;
            background: #f8fafc;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.4rem 0.35rem;
        }
        .analyst-consensus-card .analyst-target-pill .lbl {
            font-size: 0.55rem;
            font-weight: 700;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .analyst-consensus-card .analyst-target-pill .val {
            font-size: 0.95rem;
            font-weight: 800;
            color: #1e3a5f;
            margin-top: 0.15rem;
        }
        .analyst-consensus-card .analyst-chart-slot {
            flex: 0 0 auto;
            min-height: unset;
            margin: 0.2rem 0 0.05rem;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .factor-radar-card {
            justify-content: center;
            gap: 0.2rem;
        }
        .price-history-card {
            padding-bottom: 0.45rem;
        }
        .price-history-card .price-position-strip {
            margin-top: 0.4rem;
            padding: 0.45rem 0 0.55rem;
            border-top: 1px solid #e5e7eb;
        }
        /* Google Finance-style timeframe tabs (widget key prefix ph-range-). */
        div[class*="st-key-ph-range"] {
            width: 100% !important;
            margin-top: -0.15rem;
            margin-bottom: 0.1rem;
        }
        div[class*="st-key-ph-range"] [data-testid="stWidgetLabel"] {
            display: none !important;
        }
        div[class*="st-key-ph-range"] [data-testid="stRadioGroup"],
        div[class*="st-key-ph-range"] div[role="radiogroup"] {
            gap: 0 !important;
            flex-wrap: nowrap !important;
            justify-content: space-between !important;
            width: 100%;
            border-bottom: 1px solid #e8eaed;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"] {
            flex: 1 1 0;
            justify-content: center !important;
            padding: 0.12rem 0.2rem 0.28rem !important;
            margin: 0 !important;
            border-right: 1px solid #e8eaed;
            border-bottom: 3px solid transparent;
            border-radius: 0 !important;
            background: transparent !important;
            min-height: 0 !important;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:last-child {
            border-right: none;
        }
        /* Hide the Streamlit radio circle (nested inside the option, after the visually-hidden input). */
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"] > div > div > div:first-child {
            display: none !important;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"] p {
            font-size: 0.74rem !important;
            font-weight: 500 !important;
            color: #5f6368 !important;
            letter-spacing: 0.01em;
            text-align: center;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:hover p {
            color: #202124 !important;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:has(input:checked),
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"][data-selected="true"] {
            border-bottom-color: #1a73e8;
        }
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"]:has(input:checked) p,
        div[class*="st-key-ph-range"] label[data-testid="stRadioOption"][data-selected="true"] p {
            color: #1a73e8 !important;
            font-weight: 600 !important;
        }
        .factor-radar-card .dashboard-chart-slot {
            flex: 0 0 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 0.15rem 0;
            margin: auto 0;
        }
        .dashboard-chart-slot {
            flex: 1 1 auto;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 0;
        }

        /* Factor scorecard rows: flexible percentile bars (wider on large viewports) */
        .factor-scorecard-card {
            overflow: visible !important;
        }
        .factor-scorecard-grid .factor-row {
            display: grid;
            grid-template-columns: 5px minmax(4.8em, 1.05fr) minmax(56px, 2.85fr) 26px;
            column-gap: 6px;
            align-items: center;
            margin: 2px 0;
        }
        .factor-scorecard-grid .factor-dot {
            grid-column: 1;
            width: 5px;
            height: 5px;
            border-radius: 50%;
        }
        .factor-scorecard-grid .factor-label {
            grid-column: 2;
            font-size: 0.64rem;
            color: #374151;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            line-height: 1.15;
            min-width: 0;
        }
        .factor-scorecard-grid .factor-label.factor-has-tip {
            position: relative;
            cursor: help;
            overflow: hidden;
            z-index: 1;
        }
        .factor-scorecard-grid .factor-label.factor-has-tip:hover,
        .factor-scorecard-grid .factor-label.factor-has-tip:focus-within {
            z-index: 200;
            overflow: visible;
        }
        .factor-scorecard-grid .factor-label-text {
            display: block;
            overflow: hidden;
            text-overflow: ellipsis;
            border-bottom: 1px dotted #9ca3af;
        }
        .factor-scorecard-grid .factor-tooltip {
            visibility: hidden;
            opacity: 0;
            pointer-events: none;
            position: absolute;
            left: 0;
            top: calc(100% + 5px);
            width: 11.5rem;
            max-width: min(11.5rem, 70vw);
            padding: 0.4rem 0.5rem;
            background: #1e293b;
            color: #f8fafc;
            font-size: 0.6rem;
            font-weight: 400;
            line-height: 1.3;
            border-radius: 6px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.28);
            z-index: 201;
            text-transform: none;
            letter-spacing: normal;
            white-space: normal;
            transition: opacity 0.12s ease, visibility 0.12s ease;
        }
        .factor-scorecard-grid .factor-tooltip::before {
            content: "";
            position: absolute;
            bottom: 100%;
            left: 10px;
            border: 5px solid transparent;
            border-bottom-color: #1e293b;
        }
        .factor-scorecard-grid .factor-has-tip:hover .factor-tooltip,
        .factor-scorecard-grid .factor-has-tip:focus-within .factor-tooltip {
            visibility: visible;
            opacity: 1;
        }
        .factor-scorecard-grid .factor-bar-track {
            grid-column: 3;
            background: #f3f4f6;
            border-radius: 3px;
            height: 5px;
            overflow: hidden;
            min-width: 0;
        }
        .factor-scorecard-grid .factor-bar-fill {
            height: 5px;
            border-radius: 3px;
        }
        .factor-scorecard-grid .factor-pct {
            grid-column: 4;
            font-size: 0.62rem;
            font-weight: 700;
            text-align: right;
        }
        @media (min-width: 860px) {
            .factor-scorecard-grid .factor-row {
                grid-template-columns: 5px minmax(5em, 0.95fr) minmax(68px, 3.4fr) 26px;
                column-gap: 6px;
            }
        }
        @media (min-width: 1100px) {
            .factor-scorecard-grid .factor-row {
                grid-template-columns: 5px minmax(5.2em, 0.78fr) minmax(110px, 5.5fr) 28px;
                column-gap: 7px;
            }
        }
        @media (min-width: 1400px) {
            .factor-scorecard-grid .factor-row {
                grid-template-columns: 5px minmax(5.5em, 0.62fr) minmax(160px, 7fr) 28px;
                column-gap: 8px;
            }
        }

        /* Columns: prevent overflow in narrow slots */
        [data-testid="stColumn"] { min-width: 0; }

        /* Sidebar */
        [data-testid="stSidebar"] { background-color: white; }

        /* Hide footer */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        .stock-dashboard-row-anchor {
            display: none;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_equal_height_js() -> None:
    """Equalize dashboard row column heights in the main document (markdown strips <script>)."""
    js = """
(function () {
  const doc = window.parent && window.parent.document ? window.parent.document : document;

  function findRowForAnchor(anchorId) {
    const anchor = doc.getElementById(anchorId);
    if (!anchor) return null;
    let box = anchor.closest(".element-container") || anchor.parentElement;
    while (box) {
      const sibling = box.nextElementSibling;
      if (!sibling) break;
      const row = sibling.querySelector('[data-testid="stHorizontalBlock"]');
      if (row) return row;
      box = sibling;
    }
    return null;
  }

  function columnShell(col) {
    return (
      col.querySelector('[data-testid="stVerticalBlockBorderWrapper"]') ||
      col.querySelector(':scope > div > [data-testid="stVerticalBlock"]') ||
      col
    );
  }

  function equalizeRow(anchorId) {
    const row = findRowForAnchor(anchorId);
    if (!row) return;
    const cols = row.querySelectorAll('[data-testid="stColumn"]');
    if (cols.length < 2) return;

    const shells = Array.from(cols).map(columnShell);
    shells.forEach((el) => {
      el.style.minHeight = "";
    });
    cols.forEach((c) => {
      c.style.minHeight = "";
    });

    let maxH = 0;
    shells.forEach((el) => {
      maxH = Math.max(maxH, el.getBoundingClientRect().height);
    });
    if (maxH < 1) return;

    if (anchorId === "stock-row-2-anchor") {
      maxH = Math.min(maxH, 340);
    }

    const px = Math.ceil(maxH) + "px";
    cols.forEach((c) => {
      c.style.minHeight = px;
    });
    shells.forEach((el) => {
      el.style.minHeight = px;
    });
  }

  function run() {
    equalizeRow("stock-row-1-anchor");
    equalizeRow("stock-row-2-anchor");
  }

  const schedule = () => requestAnimationFrame(() => requestAnimationFrame(run));
  if (!doc.defaultView.__stockRowEqualize) {
    const win = doc.defaultView;
    win.__stockRowEqualize = schedule;
    win.addEventListener("resize", schedule);
    const root = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
    new MutationObserver(schedule).observe(root, { childList: true, subtree: true });
  }
  doc.defaultView.__stockRowEqualize();
})();
"""
    html_fn = getattr(st, "html", None)
    if html_fn is not None:
        try:
            html_fn(f"<script>{js}</script>", unsafe_allow_javascript=True)
            return
        except TypeError:
            pass
    components.html(f"<script>{js}</script>", height=0)
