"""Reusable presentation components for the Streamlit dashboard."""

from html import escape
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STYLESHEET_PATH = PROJECT_ROOT / "assets" / "dashboard.css"

_KPI_TONES = {"cyan", "blue", "red", "green"}


def apply_dashboard_styles() -> None:
    """Load the project stylesheet using a path portable across platforms."""
    stylesheet = STYLESHEET_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{stylesheet}</style>", unsafe_allow_html=True)


def page_header(title: str, subtitle: str, eyebrow: str = "FRAUD // INTEL") -> None:
    """Render a consistent, accessible page title block."""
    st.markdown(
        (
            '<header class="page-heading">'
            f'<div class="eyebrow">{escape(eyebrow)}</div>'
            f"<h1>{escape(title)}</h1>"
            f"<p>{escape(subtitle)}</p>"
            "</header>"
        ),
        unsafe_allow_html=True,
    )


def kpi_card(
    label: str,
    value: str,
    icon: str,
    tone: str = "cyan",
    detail: str = "",
) -> None:
    """Render a reusable metric card with escaped dynamic text."""
    safe_tone = tone if tone in _KPI_TONES else "cyan"
    st.markdown(
        (
            f'<div class="metric-card metric-card--{safe_tone}">'
            '<div class="metric-card-top">'
            f'<span class="metric-label">{escape(label)}</span>'
            f'<span class="metric-icon" aria-hidden="true">{escape(icon)}</span>'
            "</div>"
            f'<div class="metric-value">{escape(value)}</div>'
            f'<div class="metric-detail">{escape(detail)}</div>'
            "</div>"
        ),
        unsafe_allow_html=True,
    )
