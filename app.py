"""
app.py — PhishDetect AI  |  Streamlit front-end
Dark / Light theme toggle, email scanning, URL checker, dashboard, about.
"""

import re
import base64
import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from collections import Counter

from phishdetect_util import (
    predict_email,
    get_lime_explanation,
    extract_text_from_file,
    generate_pdf_report,
    check_url,
    check_urls_in_email,
    extract_urls,
    COLOR_MAP,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishDetect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state defaults ─────────────────────────────────────────────────────
for key, default in {
    "history":   [],
    "page":      "home",
    "dark_mode": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

dark = st.session_state.dark_mode

# ── SVG icon set — consistent 16px stroke, no emojis ──────────────────────────
_I = {
    "shield":    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "mail":      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m2 7 10 7 10-7"/></svg>',
    "link":      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>',
    "chart":     '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="18" x2="18" y1="20" y2="10"/><line x1="12" x2="12" y1="20" y2="4"/><line x1="6" x2="6" y1="20" y2="14"/></svg>',
    "info":      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/></svg>',
    "home":      '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/></svg>',
    "alert":     '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4M12 17h.01"/></svg>',
    "check":     '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "bot":       '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="10" x="3" y="11" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" x2="8" y1="16" y2="16"/><line x1="16" x2="16" y1="16" y2="16"/></svg>',
    "scan":      '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.35-4.35"/></svg>',
    "download":  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" x2="12" y1="15" y2="3"/></svg>',
    "sun":       '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/></svg>',
    "moon":      '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    "signal":    '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M2 20h.01M7 20v-4"/><path d="M12 20v-8"/><path d="M17 20V8"/><path d="M22 4v16"/></svg>',
}

# ── Theme tokens ───────────────────────────────────────────────────────────────
DARK = {
    "bg":            "#07111C",
    "bg2":           "#0C1A2A",
    "bg3":           "#101E30",
    "border":        "#1A2D42",
    "border_header": "#162235",
    "text":          "#BDC8DC",
    "text_muted":    "#3D5470",
    "text_sub":      "#6A8DAA",
    "accent":        "#2B72CC",
    "accent_hover":  "#3A85E8",
    "scrollbar_bg":  "#07111C",
    "input_bg":      "#0C1A2A",
    "input_text":    "#BDC8DC",
    "input_border":  "#1A2D42",
    "card_title":    "#3D5470",
    "conf_track":    "#1A2D42",
    "url_domain":    "#BDC8DC",
    "url_flags":     "#3D5470",
    "chart_font":    "#BDC8DC",
    "metric_val":    "#2B72CC",
    "toggle_label":  "Light",
    "toggle_bg":     "rgba(43,114,204,0.1)",
    "toggle_color":  "#5A9AE0",
    "toggle_border": "rgba(43,114,204,0.2)",
}

LIGHT = {
    "bg":            "#FAF7F2",
    "bg2":           "#FFFFFF",
    "bg3":           "#F5EDE0",
    "border":        "#E8D8C4",
    "border_header": "#DEC9AE",
    "text":          "#2C1A0A",
    "text_muted":    "#9E8068",
    "text_sub":      "#6B4A32",
    "accent":        "#C05E1A",
    "accent_hover":  "#9A4910",
    "scrollbar_bg":  "#F5EDE0",
    "input_bg":      "#FFFFFF",
    "input_text":    "#2C1A0A",
    "input_border":  "#E8D8C4",
    "card_title":    "#9E8068",
    "conf_track":    "#E8D8C4",
    "url_domain":    "#2C1A0A",
    "url_flags":     "#9E8068",
    "chart_font":    "#2C1A0A",
    "metric_val":    "#C05E1A",
    "toggle_label":  "Dark",
    "toggle_bg":     "rgba(192,94,26,0.08)",
    "toggle_color":  "#C05E1A",
    "toggle_border": "rgba(192,94,26,0.2)",
}

T = DARK if dark else LIGHT


@st.cache_resource
def _logo_b64() -> str:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logo.png")
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css(t):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,400;0,14..32,500;0,14..32,600;0,14..32,700;1,14..32,400&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.stApp {{
    background-color: {t["bg"]} !important;
    color: {t["text"]} !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 14px;
    line-height: 1.5;
}}

[data-testid="stHeader"]  {{ display: none !important; }}
[data-testid="stSidebar"] {{ display: none !important; }}
.block-container {{ padding: 1.5rem 2.5rem 2rem !important; max-width: 960px !important; margin: 0 auto !important; }}
footer {{ display: none !important; }}
#MainMenu {{ display: none !important; }}

::-webkit-scrollbar {{ width: 3px; height: 3px; }}
::-webkit-scrollbar-track {{ background: {t["bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {t["border"]}; border-radius: 1px; }}

/* ── Header ── */
.pd-header {{
    background: {t["bg"]};
    border-bottom: 1px solid {t["border_header"]};
    padding: 0.75rem 0;
    display: flex; align-items: center;
    justify-content: space-between;
    gap: 12px;
}}
.pd-wordmark {{
    font-family: 'Inter', sans-serif;
    font-size: 13px; font-weight: 700;
    letter-spacing: 0.02em;
    color: {t["text"]};
    display: flex; align-items: center; gap: 8px;
}}
.pd-logo-img {{
    height: 38px; width: auto; display: block;
}}
.pd-wordmark-shield {{
    width: 26px; height: 26px;
    background: {t["accent"]};
    border-radius: 5px;
    display: flex; align-items: center; justify-content: center;
    color: #fff; flex-shrink: 0;
}}
.pd-wordmark-text {{ color: {t["text"]}; }}
.pd-wordmark-text b {{ color: {t["accent"]}; font-weight: 700; }}
.pd-ver {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: {t["text_muted"]};
    border: 1px solid {t["border"]};
    padding: 1px 5px; border-radius: 3px;
    letter-spacing: 0.04em; vertical-align: middle;
}}
.pd-status {{
    display: flex; align-items: center; gap: 5px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 9px; color: #1FAD72;
    letter-spacing: 0.05em;
}}
.pd-status-dot {{
    width: 5px; height: 5px; background: #1FAD72;
    border-radius: 50%;
    animation: blink 3s step-end infinite;
}}
@keyframes blink {{
    0%,100% {{ opacity:1; }} 50% {{ opacity:0.35; }}
}}

/* ── Nav — flat tab style ── */
.pd-nav-bar {{
    background: {t["bg"]};
    border-bottom: 1px solid {t["border"]};
    padding: 0;
    display: flex; gap: 0;
    margin-bottom: 1.25rem;
}}

/* Flatten ALL nav buttons to look like text tabs */
[data-testid="stHorizontalBlock"] button {{
    background: transparent !important;
    color: {t["text_muted"]} !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 0.4rem !important;
    box-shadow: none !important;
    transition: color 0.15s !important;
    width: 100% !important;
}}
[data-testid="stHorizontalBlock"] button:hover {{
    background: transparent !important;
    color: {t["text"]} !important;
    transform: none !important;
    box-shadow: none !important;
    border-bottom-color: {t["border"]} !important;
    opacity: 1 !important;
}}

/* ── Layout ── */
.pd-content {{ padding: 0.25rem 0 1.5rem; }}
@media (max-width: 768px) {{
    .block-container {{ padding: 1rem 1.25rem 1.5rem !important; }}
    .pd-header  {{ padding: 0.5rem 0; }}
    .pd-content {{ padding: 0.25rem 0 1rem; }}
}}

/* ── Cards ── */
.pd-card {{
    background: {t["bg2"]};
    border: 1px solid {t["border"]};
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin-bottom: 10px;
    transition: border-color 0.15s;
}}
.pd-card:hover {{ border-color: {t["accent"]}40; }}
.pd-card-title {{
    font-size: 10px; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: {t["card_title"]};
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid {t["border"]};
}}

/* ── Metrics ── */
.pd-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(110px, 1fr)); gap: 8px; margin-bottom: 10px; }}
.pd-metric {{ background: {t["bg2"]}; border: 1px solid {t["border"]}; border-radius: 5px; padding: 0.75rem 0.9rem; }}
.pd-metric-val {{ font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 500; color: {t["metric_val"]}; line-height: 1; }}
.pd-metric-lbl {{ font-size: 10px; color: {t["text_muted"]}; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.06em; }}

/* ── Verdict strip — left-border, data-first ── */
.verdict-row {{
    display: flex; align-items: center; gap: 12px;
    padding: 0.8rem 1rem;
    border-radius: 4px;
    margin-bottom: 10px;
}}
.verdict-row-legit {{ background: rgba(13,175,128,0.07); border-left: 3px solid #0DAF80; }}
.verdict-row-phish {{ background: rgba(217,119,6,0.07);  border-left: 3px solid #D97706; }}
.verdict-row-ai    {{ background: rgba(139,92,246,0.07); border-left: 3px solid #8B5CF6; }}
.verdict-icon {{
    width: 30px; height: 30px; border-radius: 4px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}}
.verdict-icon-legit {{ background: rgba(13,175,128,0.12);  color: #0DAF80; }}
.verdict-icon-phish {{ background: rgba(217,119,6,0.12);   color: #D97706; }}
.verdict-icon-ai    {{ background: rgba(139,92,246,0.12);  color: #8B5CF6; }}
.verdict-label {{ font-size: 15px; font-weight: 700; letter-spacing: -0.01em; }}
.verdict-meta  {{ font-size: 10px; color: {t["text_muted"]}; margin-top: 2px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.03em; }}

/* ── Confidence bar ── */
.conf-wrap {{ margin: 0.5rem 0 0.75rem; }}
.conf-hdr  {{
    display: flex; justify-content: space-between;
    font-size: 10px; color: {t["text_muted"]};
    font-family: 'JetBrains Mono', monospace;
    margin-bottom: 5px; letter-spacing: 0.04em;
}}
.conf-track {{ background: {t["conf_track"]}; border-radius: 1px; height: 3px; }}
.conf-fill  {{ height: 3px; border-radius: 1px; transition: width 0.3s ease; }}

/* ── Risk badges ── */
.risk-critical {{ background: rgba(139,92,246,0.12); color: #8B5CF6; border: 1px solid rgba(139,92,246,0.25); border-radius: 3px; padding: 1px 7px; font-size: 10px; font-weight: 600; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; }}
.risk-high     {{ background: rgba(217,119,6,0.12);  color: #D97706; border: 1px solid rgba(217,119,6,0.25);  border-radius: 3px; padding: 1px 7px; font-size: 10px; font-weight: 600; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; }}
.risk-medium   {{ background: rgba(43,114,204,0.12); color: #5A9AE0; border: 1px solid rgba(43,114,204,0.25); border-radius: 3px; padding: 1px 7px; font-size: 10px; font-weight: 600; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; }}
.risk-low      {{ background: rgba(13,175,128,0.12); color: #0DAF80; border: 1px solid rgba(13,175,128,0.25); border-radius: 3px; padding: 1px 7px; font-size: 10px; font-weight: 600; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.05em; }}

/* ── LIME tokens ── */
.lime-wrap {{ display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }}
.lime-pill {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; padding: 2px 7px; border-radius: 3px; }}
.lime-pos  {{ background: rgba(13,175,128,0.1); color: #0DAF80; border: 1px solid rgba(13,175,128,0.2); }}
.lime-neg  {{ background: rgba(217,119,6,0.1);  color: #D97706; border: 1px solid rgba(217,119,6,0.2); }}

/* ── URL rows ── */
.url-row-safe     {{ border-left: 2px solid #0DAF80; padding: 5px 10px; margin-bottom: 4px; background: rgba(13,175,128,0.04); border-radius: 0 4px 4px 0; }}
.url-row-medium   {{ border-left: 2px solid #D97706; padding: 5px 10px; margin-bottom: 4px; background: rgba(217,119,6,0.04);  border-radius: 0 4px 4px 0; }}
.url-row-high     {{ border-left: 2px solid #E03A3A; padding: 5px 10px; margin-bottom: 4px; background: rgba(224,58,58,0.04);  border-radius: 0 4px 4px 0; }}
.url-row-critical {{ border-left: 2px solid #8B5CF6; padding: 5px 10px; margin-bottom: 4px; background: rgba(139,92,246,0.04); border-radius: 0 4px 4px 0; }}
.url-domain {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: {t["url_domain"]}; word-break: break-all; }}
.url-flags  {{ font-size: 10px; color: {t["url_flags"]}; margin-top: 2px; font-family: 'JetBrains Mono', monospace; }}

/* ── Header analysis rows ── */
.hdr-row {{ display: flex; justify-content: space-between; align-items: flex-start; padding: 5px 0; border-bottom: 1px solid {t["border"]}; gap: 12px; }}
.hdr-label {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: {t["text_muted"]}; min-width: 96px; flex-shrink: 0; }}
.hdr-value {{ font-size: 11px; color: {t["text"]}; word-break: break-all; font-family: 'JetBrains Mono', monospace; }}
.hdr-flag  {{ font-size: 10px; color: #D97706; margin-top: 3px; font-family: 'JetBrains Mono', monospace; }}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
    background:    {t["input_bg"]}    !important;
    border:        1px solid {t["input_border"]} !important;
    border-radius: 5px               !important;
    color:         {t["input_text"]}  !important;
    font-family:   'JetBrains Mono', monospace !important;
    font-size:     12px               !important;
}}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {{
    border-color: {t["accent"]}      !important;
    box-shadow:   0 0 0 1px {t["accent"]}40 !important;
}}

/* ── Action buttons ── */
[data-testid="stButton"] > button {{
    background:    {t["accent"]}     !important;
    color:         #fff              !important;
    border:        none              !important;
    border-radius: 5px               !important;
    font-family:   'Inter', sans-serif !important;
    font-weight:   600               !important;
    font-size:     12px              !important;
    letter-spacing: 0.02em          !important;
    padding:       0.5rem 1.2rem    !important;
    width:         100%              !important;
    transition:    background 0.15s !important;
    box-shadow:    none              !important;
}}
[data-testid="stButton"] > button:hover {{
    background:  {t["accent_hover"]} !important;
    transform:   none                !important;
    box-shadow:  none                !important;
    opacity:     1                   !important;
}}
[data-testid="stDownloadButton"] > button {{
    background:    #0A7A5A           !important;
    color:         #fff              !important;
    border:        none              !important;
    border-radius: 5px               !important;
    font-family:   'Inter', sans-serif !important;
    font-weight:   600               !important;
    font-size:     12px              !important;
    padding:       0.5rem 1.2rem    !important;
    width:         100%              !important;
    transition:    background 0.15s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: #0C9468 !important;
    transform:  none    !important;
    box-shadow: none    !important;
}}

/* ── Misc Streamlit overrides ── */
[data-testid="stRadio"] label     {{ color: {t["text_sub"]} !important; font-size: 12px !important; font-family: 'Inter', sans-serif !important; }}
[data-testid="stFileUploader"]    {{ background: {t["bg2"]} !important; border: 1px dashed {t["border"]} !important; border-radius: 5px !important; }}
[data-testid="stSpinner"]         {{ color: {t["accent"]} !important; }}
[data-testid="stAlert"]           {{ background: {t["bg2"]} !important; border-left: 2px solid {t["accent"]} !important; color: {t["text"]} !important; border-radius: 0 4px 4px 0 !important; }}
.stDataFrame                      {{ background: {t["bg2"]} !important; border-radius: 5px !important; }}
[data-testid="stExpander"]        {{ background: {t["bg2"]} !important; border: 1px solid {t["border"]} !important; border-radius: 5px !important; }}
[data-testid="stMetricValue"]     {{ color: {t["metric_val"]} !important; font-family: 'JetBrains Mono', monospace !important; }}
[data-testid="stMetricLabel"]     {{ color: {t["text_muted"]} !important; font-size: 10px !important; }}
.js-plotly-plot .plotly .bg       {{ fill: transparent !important; }}

/* ── Threat class grid (home) ── */
.threat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(195px, 1fr)); gap: 8px; margin-top: 1.25rem; }}
.tc {{
    border-radius: 5px; padding: 0.85rem 1rem;
    border: 1px solid; font-size: 12px;
}}
.tc-safe  {{ background: rgba(13,175,128,0.06); border-color: rgba(13,175,128,0.2); }}
.tc-warn  {{ background: rgba(217,119,6,0.06);  border-color: rgba(217,119,6,0.2);  }}
.tc-ai    {{ background: rgba(139,92,246,0.06); border-color: rgba(139,92,246,0.2); }}
.tc-name  {{ font-weight: 700; font-size: 12px; margin-bottom: 4px; }}
.tc-safe .tc-name  {{ color: #0DAF80; }}
.tc-warn .tc-name  {{ color: #D97706; }}
.tc-ai   .tc-name  {{ color: #8B5CF6; }}
.tc-desc  {{ font-size: 11px; color: {t["text_muted"]}; line-height: 1.5; }}

/* ── Feature grid ── */
.feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 8px; margin-top: 1rem; }}
.feature-item {{ background: {t["bg2"]}; border: 1px solid {t["border"]}; border-radius: 5px; padding: 0.85rem 1rem; transition: border-color 0.15s; }}
.feature-item:hover {{ border-color: {t["accent"]}40; }}
.feature-icon {{ font-size: 13px; margin-bottom: 6px; color: {t["accent"]}; }}
.feature-name {{ font-size: 12px; font-weight: 600; color: {t["text"]}; }}
.feature-desc {{ font-size: 11px; color: {t["text_muted"]}; margin-top: 3px; line-height: 1.45; }}

/* ── Hero ── */
.hero-title  {{ font-size: clamp(1.2rem,2.4vw,1.75rem); font-weight: 700; color: {t["text"]}; line-height: 1.2; margin-bottom: 0.4rem; letter-spacing: -0.02em; }}
.hero-accent {{ color: {t["accent"]}; }}
.hero-sub    {{ font-size: 13px; color: {t["text_sub"]}; line-height: 1.65; max-width: 540px; }}

@media (max-width: 640px) {{
    .pd-metric-val {{ font-size: 1.1rem; }}
    .hero-title    {{ font-size: 1.1rem; }}
    .verdict-label {{ font-size: 13px; }}
}}

@media (prefers-reduced-motion: reduce) {{
    *, *::before, *::after {{ animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }}
}}
</style>
""", unsafe_allow_html=True)


inject_css(T)


# ── Email header parser ────────────────────────────────────────────────────────
def parse_email_headers(raw_text: str) -> dict:
    """
    Extract and analyse email header fields from raw email text.
    Returns a dict of parsed fields and a list of suspicious flags.
    """
    headers = {}
    flags   = []

    def _extract(field):
        m = re.search(rf"^{field}:\s*(.+)", raw_text, re.IGNORECASE | re.MULTILINE)
        return m.group(1).strip() if m else None

    # Core fields
    headers["From"]       = _extract("From")
    headers["To"]         = _extract("To")
    headers["Reply-To"]   = _extract("Reply-To")
    headers["Subject"]    = _extract("Subject")
    headers["Date"]       = _extract("Date")
    headers["Message-ID"] = _extract("Message-ID")
    headers["Return-Path"]= _extract("Return-Path")
    headers["X-Mailer"]   = _extract("X-Mailer")

    # Received chain (how many hops)
    received = re.findall(r"^Received:", raw_text, re.IGNORECASE | re.MULTILINE)
    headers["Received hops"] = str(len(received)) if received else "0"

    # SPF / DKIM / DMARC
    auth_results = _extract("Authentication-Results") or ""
    headers["SPF"]   = "pass" if "spf=pass"   in auth_results.lower() else ("fail" if "spf=fail"   in auth_results.lower() else "none")
    headers["DKIM"]  = "pass" if "dkim=pass"  in auth_results.lower() else ("fail" if "dkim=fail"  in auth_results.lower() else "none")
    headers["DMARC"] = "pass" if "dmarc=pass" in auth_results.lower() else ("fail" if "dmarc=fail" in auth_results.lower() else "none")

    # ── Suspicious checks ─────────────────────────────────────────────────────
    frm      = headers["From"]      or ""
    reply_to = headers["Reply-To"]  or ""
    ret_path = headers["Return-Path"] or ""
    subj     = headers["Subject"]   or ""

    # From/Reply-To mismatch
    def _domain(addr):
        m = re.search(r"@([\w.\-]+)", addr)
        return m.group(1).lower() if m else ""

    frm_domain = _domain(frm)
    rpt_domain = _domain(reply_to)
    ret_domain = _domain(ret_path)

    if reply_to and frm_domain and rpt_domain and frm_domain != rpt_domain:
        flags.append(f"From domain ({frm_domain}) ≠ Reply-To domain ({rpt_domain})")

    if ret_path and frm_domain and ret_domain and frm_domain != ret_domain:
        flags.append(f"From domain ({frm_domain}) ≠ Return-Path domain ({ret_domain})")

    # SPF/DKIM/DMARC failures
    if headers["SPF"]   == "fail": flags.append("SPF check FAILED — sender not authorised")
    if headers["DKIM"]  == "fail": flags.append("DKIM signature FAILED — message may be tampered")
    if headers["DMARC"] == "fail": flags.append("DMARC policy FAILED")

    # Suspicious subject keywords
    subj_triggers = ["urgent", "verify", "suspended", "confirm", "password", "prize",
                     "winner", "account", "click", "free", "limited", "act now", "alert"]
    for kw in subj_triggers:
        if kw in subj.lower():
            flags.append(f"Suspicious subject keyword: '{kw}'")
            break

    # No Message-ID
    if not headers["Message-ID"]:
        flags.append("Missing Message-ID header (common in spoofed emails)")

    # Excessive hops
    hops = int(headers["Received hops"])
    if hops > 6:
        flags.append(f"Unusually high hop count ({hops}) — may indicate routing obfuscation")

    return {"fields": headers, "flags": flags}


def _auth_badge(val: str) -> str:
    if val == "pass":
        return '<span style="color:#0DAF80;font-weight:600;font-family:\'JetBrains Mono\',monospace;font-size:.8rem">pass</span>'
    if val == "fail":
        return '<span style="color:#D97706;font-weight:600;font-family:\'JetBrains Mono\',monospace;font-size:.8rem">fail</span>'
    return '<span style="color:#3D5470;font-family:\'JetBrains Mono\',monospace;font-size:.8rem">—</span>'


# ── Helpers ────────────────────────────────────────────────────────────────────
def _map_label(pred) -> str:
    if isinstance(pred, str):
        return pred.lower()
    try:
        return {0: "legitimate", 1: "traditional phishing", 2: "ai generated phishing"}[int(pred)]
    except (TypeError, ValueError, KeyError):
        return "unknown"


def add_to_history(email_text: str, result: dict):
    st.session_state.history.append({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "email_preview": email_text[:80] + "…",
        "prediction":    _map_label(result["prediction"]),
        "confidence":    result["confidence"],
        "risk":          result.get("risk_level", "—"),
    })


def _risk_css(risk: str) -> str:
    r = risk.lower()
    if "critical" in r: return "risk-critical"
    if "high"     in r: return "risk-high"
    if "medium"   in r: return "risk-medium"
    return "risk-low"


def _url_row_cls(risk: str) -> str:
    r = risk.lower()
    if "critical" in r: return "url-row-critical"
    if "high"     in r: return "url-row-high"
    if "medium"   in r: return "url-row-medium"
    return "url-row-safe"


def _chart_layout(height: int = 220):
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["chart_font"], family="JetBrains Mono, monospace", size=11),
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
    )


def _verdict_html(pred_label: str, confidence: float, risk: str) -> str:
    if "ai" in pred_label or "generated" in pred_label:
        row_cls  = "verdict-row-ai"
        icon_cls = "verdict-icon-ai"
        icon_svg = _I["bot"]
        title    = "AI-Generated Phishing"
        meta     = "high-sophistication · AI-crafted language"
        conf_col = "#8B5CF6"
    elif "phishing" in pred_label:
        row_cls  = "verdict-row-phish"
        icon_cls = "verdict-icon-phish"
        icon_svg = _I["alert"]
        title    = "Traditional Phishing"
        meta     = "classic credential-harvesting pattern"
        conf_col = "#D97706"
    else:
        row_cls  = "verdict-row-legit"
        icon_cls = "verdict-icon-legit"
        icon_svg = _I["check"]
        title    = "Legitimate Email"
        meta     = "no phishing indicators found"
        conf_col = "#0DAF80"

    return f"""
    <div class="verdict-row {row_cls}">
      <div class="verdict-icon {icon_cls}">{icon_svg}</div>
      <div style="flex:1;min-width:0">
        <div class="verdict-label" style="color:{conf_col}">{title}</div>
        <div class="verdict-meta">{meta}</div>
      </div>
      <span class="{_risk_css(risk)}">{risk}</span>
    </div>
    <div class="conf-wrap">
      <div class="conf-hdr">
        <span>confidence</span>
        <span style="color:{conf_col}">{confidence:.1f}%</span>
      </div>
      <div class="conf-track">
        <div class="conf-fill" style="width:{confidence}%;background:{conf_col}"></div>
      </div>
    </div>
    """


# ── Header ─────────────────────────────────────────────────────────────────────
toggle_icon = _I["sun"] if dark else _I["moon"]
header_col, toggle_col = st.columns([9, 1])
with header_col:
    _lb = _logo_b64()
    _logo_html = (
        f'<img src="data:image/png;base64,{_lb}" class="pd-logo-img" alt="PhishDetect AI">'
        if _lb else
        f'<div class="pd-wordmark-shield">{_I["shield"]}</div><span class="pd-wordmark-text">PHISH<b>DETECT</b></span>'
    )
    st.markdown(f"""
    <div class="pd-header">
      <div class="pd-wordmark">
        {_logo_html}
        <span class="pd-ver">v2.0</span>
      </div>
      <div class="pd-status">
        <div class="pd-status-dot"></div>OPERATIONAL
      </div>
    </div>
    """, unsafe_allow_html=True)

with toggle_col:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    if st.button(f"{T['toggle_label']}", key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Nav — underline tab style ──────────────────────────────────────────────────
pages = {
    "home":      "Home",
    "scan":      "Scan Email",
    "url":       "URL Checker",
    "dashboard": "Dashboard",
    "about":     "About",
}
page = st.session_state.page

# Inject active-state underline for the current page tab
active_idx = list(pages.keys()).index(page) + 1
st.markdown(f"""
<style>
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child({active_idx}) button {{
    color: {T["text"]} !important;
    border-bottom: 2px solid {T["accent"]} !important;
}}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="pd-nav-bar">', unsafe_allow_html=True)
nav_cols = st.columns(len(pages))
for i, (key, label) in enumerate(pages.items()):
    with nav_cols[i]:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pd-card">
      <div class="hero-title">
        Email Threat Classification<br>
        <span class="hero-accent">Multi-Signal · Explainable · Actionable</span>
      </div>
      <p class="hero-sub">
        PhishDetect AI combines TF-IDF machine learning with real-time signal
        fusion — sender reputation, header authentication, URL risk, and template
        detection — to classify emails across three threat classes with word-level
        LIME explanations for every verdict.
      </p>
    </div>

    <div class="threat-grid">
      <div class="tc tc-safe">
        <div class="tc-name">{_I["check"]} Legitimate</div>
        <div class="tc-desc">Verified sender alignment, clean headers, no payload indicators</div>
      </div>
      <div class="tc tc-warn">
        <div class="tc-name">{_I["alert"]} Traditional Phishing</div>
        <div class="tc-desc">Credential harvesting patterns, spoofed domains, malicious URLs</div>
      </div>
      <div class="tc tc-ai">
        <div class="tc-name">{_I["bot"]} AI-Generated Phishing</div>
        <div class="tc-desc">High-fluency AI-crafted language with deceptive intent signatures</div>
      </div>
    </div>

    <div class="feature-grid" style="margin-top:10px">
      <div class="feature-item">
        <div class="feature-icon">{_I["scan"]}</div>
        <div class="feature-name">ML Classification</div>
        <div class="feature-desc">LR + SVM ensemble on 15K TF-IDF features, soft-vote blended</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">{_I["chart"]}</div>
        <div class="feature-name">LIME Explainability</div>
        <div class="feature-desc">Word-level explanations showing exactly what triggered the verdict</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">{_I["mail"]}</div>
        <div class="feature-name">Header Analysis</div>
        <div class="feature-desc">SPF, DKIM, DMARC, Reply-To mismatch, sender-brand alignment</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">{_I["link"]}</div>
        <div class="feature-name">URL Risk Scoring</div>
        <div class="feature-desc">Auto-extraction and risk-scoring of every URL in an email</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">{_I["signal"]}</div>
        <div class="feature-name">Signal Fusion</div>
        <div class="feature-desc">Weighted oracle interpolation blends ML output with domain signals</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">{_I["download"]}</div>
        <div class="feature-name">PDF Reports</div>
        <div class="feature-desc">ReportLab-generated threat reports for documentation and compliance</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SCAN EMAIL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "scan":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="medium")

    with left:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="pd-card-title">{_I["mail"]} Email Input</div>', unsafe_allow_html=True)

        input_method = st.radio(
            "Input method:", ["Paste Text", "Upload File"],
            horizontal=True, label_visibility="collapsed",
        )

        email_text = ""
        if input_method == "Paste Text":
            email_text = st.text_area(
                "Email content", height=240,
                placeholder="Paste the full email — headers, body, links...",
                label_visibility="collapsed",
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose a .txt / .pdf / .docx file",
                type=["txt", "pdf", "docx"],
                label_visibility="collapsed",
            )
            if uploaded_file:
                email_text = extract_text_from_file(uploaded_file)
                if email_text:
                    st.success(f"✓ Loaded: {uploaded_file.name} ({len(email_text):,} chars)")
                    with st.expander("Preview extracted text"):
                        st.text(email_text[:600] + ("…" if len(email_text) > 600 else ""))
                else:
                    st.error("Could not extract text from this file.")

        analyze_btn = st.button("Analyze Email", key="analyze", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Email Header Analysis card (left column, below input) ──────────
        if analyze_btn and email_text.strip():
            hdr = parse_email_headers(email_text)
            fields = hdr["fields"]
            hdr_flags = hdr["flags"]

            st.markdown('<div class="pd-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="pd-card-title">{_I["mail"]} Email Header Analysis</div>', unsafe_allow_html=True)

            # Core fields table
            display_fields = [
                ("From",        fields.get("From")        or "—"),
                ("To",          fields.get("To")          or "—"),
                ("Reply-To",    fields.get("Reply-To")    or "—"),
                ("Return-Path", fields.get("Return-Path") or "—"),
                ("Subject",     fields.get("Subject")     or "—"),
                ("Date",        fields.get("Date")        or "—"),
                ("Message-ID",  fields.get("Message-ID")  or "—"),
                ("X-Mailer",    fields.get("X-Mailer")    or "—"),
                ("Hops",        fields.get("Received hops") or "0"),
            ]
            rows_html = ""
            for label, value in display_fields:
                safe_val = str(value)[:90] + ("…" if len(str(value)) > 90 else "")
                rows_html += f"""
                <div class="hdr-row">
                  <span class="hdr-label">{label}</span>
                  <span class="hdr-value">{safe_val}</span>
                </div>"""

            # Auth row
            rows_html += f"""
            <div class="hdr-row" style="margin-top:8px">
              <span class="hdr-label">SPF</span>
              <span class="hdr-value">{_auth_badge(fields['SPF'])}</span>
            </div>
            <div class="hdr-row">
              <span class="hdr-label">DKIM</span>
              <span class="hdr-value">{_auth_badge(fields['DKIM'])}</span>
            </div>
            <div class="hdr-row">
              <span class="hdr-label">DMARC</span>
              <span class="hdr-value">{_auth_badge(fields['DMARC'])}</span>
            </div>"""

            st.markdown(rows_html, unsafe_allow_html=True)

            # Flags
            if hdr_flags:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="pd-card-title">🚩 Header Flags</div>', unsafe_allow_html=True)
                for f in hdr_flags:
                    st.markdown(
                        f'<div class="hdr-flag" style="margin-bottom:4px">{f}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<p style="font-size:.8rem;color:#34D399;margin-top:8px">✔ No suspicious header patterns detected</p>',
                    unsafe_allow_html=True,
                )

            st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="pd-card-title">{_I["scan"]} Analysis Results</div>', unsafe_allow_html=True)

        if analyze_btn:
            if not email_text.strip():
                st.warning("Please provide email content first.")
            else:
                with st.spinner("Analyzing…"):
                    result      = predict_email(email_text)
                    lime_result = get_lime_explanation(email_text)

                pred_label = _map_label(result["prediction"])
                confidence = float(result.get("confidence", 0))
                risk       = result.get("risk_level", "Unknown")

                add_to_history(email_text, result)

                # Verdict banner
                st.markdown(_verdict_html(pred_label, confidence, risk), unsafe_allow_html=True)

                # Probability chart
                probs = result.get("probabilities", {})
                if probs:
                    st.markdown("<br>", unsafe_allow_html=True)
                    labels = [k.title() for k in probs]
                    values = [round(v * 100, 1) if v <= 1 else round(v, 1) for v in probs.values()]
                    bar_colors = [COLOR_MAP.get(k, "#94A3B8") for k in probs]
                    fig = go.Figure(go.Bar(
                        x=labels, y=values,
                        marker_color=bar_colors,
                        text=[f"{v}%" for v in values],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        **_chart_layout(200),
                        yaxis=dict(range=[0, 115], showgrid=False, visible=False),
                        xaxis=dict(showgrid=False),
                        showlegend=False,
                        title=dict(text="Class Probabilities (%)", font=dict(size=11)),
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

                # LIME
                if lime_result:
                    pos = lime_result.get("positive", [])
                    neg = lime_result.get("negative", [])
                    st.markdown(f'<div class="pd-card-title" style="margin-top:12px">{_I["chart"]} LIME Word Explanations</div>', unsafe_allow_html=True)
                    pills_html = '<div class="lime-wrap">'
                    for w in pos[:8]:
                        pills_html += f'<span class="lime-pill lime-pos">+{w}</span>'
                    for w in neg[:8]:
                        pills_html += f'<span class="lime-pill lime-neg">−{w}</span>'
                    pills_html += "</div>"
                    st.markdown(pills_html, unsafe_allow_html=True)
                    st.markdown(
                        '<p style="font-size:.72rem;color:#64748B;margin-top:6px">'
                        'green = toward predicted class &nbsp;·&nbsp; red = away from it</p>',
                        unsafe_allow_html=True,
                    )

                # Multi-signal analysis panel
                signals = result.get("signals", [])
                used_ensemble = result.get("used_ensemble", False)
                if signals or used_ensemble:
                    st.markdown(f'<div class="pd-card-title" style="margin-top:12px">{_I["signal"]} Multi-Signal Analysis</div>', unsafe_allow_html=True)
                    net = sum(s["weight"] for s in signals)
                    net_label = "legitimate signals dominant" if net < 0 else ("phishing signals dominant" if net > 0 else "balanced")
                    net_color = "#0DAF80" if net < 0 else ("#D97706" if net > 0 else "#3D5470")
                    method_label = "LR + SVM ensemble" if used_ensemble else "LR only"
                    st.markdown(f"""
                    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:{T['text_muted']};margin-bottom:8px">
                      model: {method_label} &nbsp;·&nbsp;
                      <span style="color:{net_color}">net {net:+.2f} — {net_label}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    for sig in signals:
                        direction = sig.get("direction", "phishing" if sig["weight"] > 0 else "legitimate")
                        pill_cls  = "lime-neg" if direction == "phishing" else "lime-pos"
                        icon      = "↑ phish" if direction == "phishing" else "↓ legit"
                        st.markdown(
                            f'<div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:5px">'
                            f'<span class="lime-pill {pill_cls}" style="white-space:nowrap">{icon}</span>'
                            f'<span style="font-size:.78rem;color:{T["text_sub"]}">{sig["label"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    if not signals:
                        st.markdown(
                            '<p style="font-size:.78rem;color:#64748B">No additional signals detected in headers or URLs.</p>',
                            unsafe_allow_html=True,
                        )

                # URL analysis
                url_results = check_urls_in_email(email_text)
                if url_results:
                    st.markdown(f'<div class="pd-card-title" style="margin-top:12px">{_I["link"]} URLs Found</div>', unsafe_allow_html=True)
                    for ur in url_results[:8]:
                        cls        = _url_row_cls(ur["risk"])
                        flags_str  = " · ".join(ur.get("flags", [])) or "No issues detected"
                        badge_cls  = _risk_css(ur["risk"])
                        short_url  = ur["url"][:70] + ("…" if len(ur["url"]) > 70 else "")
                        st.markdown(f"""
                        <div class="{cls}">
                          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
                            <div class="url-domain">{short_url}</div>
                            <span class="{badge_cls}">{ur['risk']}</span>
                          </div>
                          <div class="url-flags">{flags_str}</div>
                        </div>
                        """, unsafe_allow_html=True)

                # PDF download
                pdf_bytes = generate_pdf_report(email_text, result, lime_result)
                if pdf_bytes:
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.download_button(
                        "Download PDF Report",
                        data=pdf_bytes,
                        file_name=f"phishdetect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:3rem 1rem;color:{T['text_muted']}">
              <div style="font-size:2.5rem;margin-bottom:.75rem">📬</div>
              <div style="font-size:.9rem">Paste or upload an email, then click Analyze Email</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# URL CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "url":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    left, right = st.columns([1, 1], gap="medium")

    with left:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="pd-card-title">{_I["link"]} URL Input</div>', unsafe_allow_html=True)
        url_input = st.text_input(
            "Enter URL", placeholder="https://example.com/path?query=value",
            label_visibility="collapsed",
        )
        check_btn = st.button("Check URL", key="check_url", use_container_width=True)

        st.markdown(f'<div class="pd-card-title" style="margin-top:1.5rem">{_I["link"]} Bulk Check (one URL per line)</div>', unsafe_allow_html=True)
        bulk_input = st.text_area(
            "Bulk URLs", height=140,
            placeholder="https://url1.com\nhttps://url2.com\n…",
            label_visibility="collapsed",
        )
        bulk_btn = st.button("Check All URLs", key="check_bulk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="pd-card-title">{_I["chart"]} URL Analysis</div>', unsafe_allow_html=True)

        if check_btn and url_input.strip():
            ur = check_url(url_input.strip())
            cls        = _url_row_cls(ur["risk"])
            badge_cls  = _risk_css(ur["risk"])
            flags_str  = " · ".join(ur.get("flags", [])) or "No issues detected"
            safe_label = "Safe" if ur["safe"] else "Suspicious"
            st.markdown(f"""
            <div class="{cls}" style="border-radius:8px;padding:12px 14px">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:8px">
                <div class="url-domain">{ur['url'][:80]}</div>
                <span class="{badge_cls}">{ur['risk']}</span>
              </div>
              <div style="font-size:1rem;font-weight:700;margin:8px 0 4px">{safe_label}</div>
              <div class="url-flags">{flags_str}</div>
            </div>
            """, unsafe_allow_html=True)

        elif bulk_btn and bulk_input.strip():
            urls = [u.strip() for u in bulk_input.strip().splitlines() if u.strip()]
            if urls:
                results = [check_url(u) for u in urls]
                for ur in results:
                    cls       = _url_row_cls(ur["risk"])
                    badge_cls = _risk_css(ur["risk"])
                    flags_str = " · ".join(ur.get("flags", [])) or "No issues"
                    short_url = ur["url"][:65] + ("…" if len(ur["url"]) > 65 else "")
                    st.markdown(f"""
                    <div class="{cls}">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px">
                        <div class="url-domain">{short_url}</div>
                        <span class="{badge_cls}">{ur['risk']}</span>
                      </div>
                      <div class="url-flags">{flags_str}</div>
                    </div>
                    """, unsafe_allow_html=True)

                risk_counts = Counter(ur["risk"] for ur in results)
                fig = px.pie(
                    values=list(risk_counts.values()),
                    names=list(risk_counts.keys()),
                    color=list(risk_counts.keys()),
                    color_discrete_map={
                        "Low": "#34D399", "Medium": "#F59E0B",
                        "High": "#EF4444", "Critical": "#DC2626",
                    },
                    title="Risk Distribution",
                    hole=0.55,
                )
                fig.update_layout(**_chart_layout(200))
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        else:
            st.markdown(f"""
            <div style="text-align:center;padding:3rem 1rem;color:{T['text_muted']}">
              <div style="opacity:.35;margin-bottom:.75rem">{_I["link"]}</div>
              <div style="font-size:.9rem">Enter a URL above and click Check URL</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "dashboard":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)

    history = st.session_state.history

    if not history:
        st.markdown(f"""
        <div class="pd-card" style="text-align:center;padding:3rem">
          <div style="opacity:.3;margin-bottom:1rem">{_I["chart"]}</div>
          <div style="font-size:1rem;font-weight:700;color:{T['text']}">No scans yet</div>
          <div style="font-size:.85rem;color:{T['text_muted']};margin-top:6px">
            Head to Scan Email to analyse your first email
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        df = pd.DataFrame(history)

        total     = len(df)
        phish     = int((df["prediction"] != "legitimate").sum())
        avg_conf  = df["confidence"].mean()
        high_risk = int(df["risk"].isin(["High", "Critical"]).sum())

        st.markdown(f"""
        <div class="pd-metrics">
          <div class="pd-metric"><div class="pd-metric-val">{total}</div><div class="pd-metric-lbl">Total scans</div></div>
          <div class="pd-metric"><div class="pd-metric-val">{phish}</div><div class="pd-metric-lbl">Threats detected</div></div>
          <div class="pd-metric"><div class="pd-metric-val">{avg_conf:.0f}%</div><div class="pd-metric-lbl">Avg confidence</div></div>
          <div class="pd-metric"><div class="pd-metric-val">{high_risk}</div><div class="pd-metric-lbl">High/Critical</div></div>
        </div>
        """, unsafe_allow_html=True)

        chart_l, chart_r = st.columns(2, gap="medium")

        with chart_l:
            st.markdown('<div class="pd-card">', unsafe_allow_html=True)
            pred_counts = df["prediction"].value_counts()
            fig = px.pie(
                values=pred_counts.values,
                names=[n.title() for n in pred_counts.index],
                color=pred_counts.index.tolist(),
                color_discrete_map={k: v for k, v in COLOR_MAP.items()},
                title="Prediction Breakdown",
                hole=0.55,
            )
            fig.update_layout(**_chart_layout(240))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_r:
            st.markdown('<div class="pd-card">', unsafe_allow_html=True)
            risk_counts = df["risk"].value_counts()
            fig2 = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                color=risk_counts.index,
                color_discrete_map={
                    "Low": "#34D399", "Medium": "#F59E0B",
                    "High": "#EF4444", "Critical": "#DC2626", "Unknown": "#94A3B8",
                },
                title="Risk Level Distribution",
                labels={"x": "Risk", "y": "Count"},
            )
            fig2.update_layout(**_chart_layout(240), showlegend=False)
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">📈 Confidence Over Time</div>', unsafe_allow_html=True)
        fig3 = px.line(
            df, x=df.index, y="confidence",
            color="prediction",
            color_discrete_map=COLOR_MAP,
            labels={"index": "Scan #", "confidence": "Confidence (%)"},
            markers=True,
        )
        fig3.update_layout(**_chart_layout(200))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">🗂️ Scan History</div>', unsafe_allow_html=True)
        display_df = df.copy()
        display_df["confidence"] = display_df["confidence"].apply(lambda x: f"{x:.1f}%")
        display_df.columns = ["Timestamp", "Email Preview", "Prediction", "Confidence", "Risk"]
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        if st.button("🗑️ Clear History", key="clear_hist"):
            st.session_state.history = []
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pd-card">
      <div class="pd-card-title">{_I["info"]} About PhishDetect AI</div>
      <p style="color:{T['text_sub']};line-height:1.8;font-size:.9rem">
        PhishDetect AI is a machine-learning-powered email threat detection tool built with
        Streamlit. It classifies emails into three categories: <strong>Legitimate</strong>,
        <strong>Traditional Phishing</strong>, and <strong>AI-Generated Phishing</strong>.
      </p>
    </div>

    <div class="pd-card">
      <div class="pd-card-title">Architecture</div>
      <div class="feature-grid">
        <div class="feature-item">
          <div class="feature-icon">{_I["scan"]}</div>
          <div class="feature-name">Text Preprocessing</div>
          <div class="feature-desc">HTML decoding, URL/email stripping, lowercasing, alpha-only filtering</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">{_I["info"]}</div>
          <div class="feature-name">TF-IDF Vectorizer</div>
          <div class="feature-desc">Fixed vocabulary trained at fit-time, always loaded from vectorizer.pkl</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">{_I["shield"]}</div>
          <div class="feature-name">LR + SVM Ensemble</div>
          <div class="feature-desc">Calibrated soft-vote blend of Logistic Regression and SVM classifiers</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">{_I["chart"]}</div>
          <div class="feature-name">LIME Explainer</div>
          <div class="feature-desc">Local interpretable model-agnostic explanations at word level</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">{_I["mail"]}</div>
          <div class="feature-name">Header Analysis</div>
          <div class="feature-desc">From/Reply-To mismatch, SPF/DKIM/DMARC, hop count, Message-ID checks</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">{_I["link"]}</div>
          <div class="feature-name">URL Heuristics</div>
          <div class="feature-desc">IP detection, typosquatting, shorteners, TLD abuse — no external API needed</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">{_I["download"]}</div>
          <div class="feature-name">PDF Reports</div>
          <div class="feature-desc">ReportLab-generated threat reports with verdict, probabilities, and LIME results</div>
        </div>
      </div>
    </div>

    <div class="pd-card">
      <div class="pd-card-title">{_I["alert"]} Disclaimer</div>
      <p style="color:{T['text_sub']};line-height:1.8;font-size:.85rem">
        PhishDetect AI is intended for security research and awareness purposes only.
        No automated tool provides 100% accuracy — always apply human judgement for
        high-stakes decisions. Do not submit confidential email content to any public deployment.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
