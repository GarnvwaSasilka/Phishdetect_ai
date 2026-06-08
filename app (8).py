"""
app.py — PhishDetect AI  |  Streamlit front-end
Dark / Light theme toggle, email scanning, URL checker, dashboard, about.
"""

import re
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
    "dark_mode": True,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

dark = st.session_state.dark_mode

# ── Theme tokens ───────────────────────────────────────────────────────────────
DARK = {
    "bg":            "#0F172A",
    "bg2":           "#1E293B",
    "bg3":           "#0F172A",
    "border":        "#334155",
    "border_header": "#1E3A5F",
    "text":          "#E2E8F0",
    "text_muted":    "#64748B",
    "text_sub":      "#94A3B8",
    "accent":        "#3B82F6",
    "accent_hover":  "#2563EB",
    "scrollbar_bg":  "#0F172A",
    "input_bg":      "#0F172A",
    "input_text":    "#E2E8F0",
    "input_border":  "#334155",
    "card_title":    "#64748B",
    "conf_track":    "#0F172A",
    "url_domain":    "#E2E8F0",
    "url_flags":     "#64748B",
    "chart_font":    "#E2E8F0",
    "metric_val":    "#3B82F6",
    "toggle_icon":   "☀️",
    "toggle_label":  "Light Mode",
    "toggle_bg":     "rgba(59,130,246,0.12)",
    "toggle_color":  "#60A5FA",
    "toggle_border": "rgba(59,130,246,0.25)",
}

LIGHT = {
    "bg":            "#F1F5F9",
    "bg2":           "#FFFFFF",
    "bg3":           "#E2E8F0",
    "border":        "#CBD5E1",
    "border_header": "#BFDBFE",
    "text":          "#0F172A",
    "text_muted":    "#475569",
    "text_sub":      "#334155",
    "accent":        "#2563EB",
    "accent_hover":  "#1D4ED8",
    "scrollbar_bg":  "#E2E8F0",
    "input_bg":      "#FFFFFF",
    "input_text":    "#0F172A",
    "input_border":  "#CBD5E1",
    "card_title":    "#475569",
    "conf_track":    "#E2E8F0",
    "url_domain":    "#0F172A",
    "url_flags":     "#475569",
    "chart_font":    "#0F172A",
    "metric_val":    "#2563EB",
    "toggle_icon":   "🌙",
    "toggle_label":  "Dark Mode",
    "toggle_bg":     "rgba(37,99,235,0.08)",
    "toggle_color":  "#2563EB",
    "toggle_border": "rgba(37,99,235,0.2)",
}

T = DARK if dark else LIGHT


# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_css(t):
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.stApp {{
    background-color: {t["bg"]} !important;
    color: {t["text"]} !important;
    font-family: 'Syne', sans-serif !important;
}}

[data-testid="stHeader"]  {{ display: none !important; }}
[data-testid="stSidebar"] {{ display: none !important; }}
.block-container {{ padding: 0 !important; max-width: 100% !important; }}
footer {{ display: none !important; }}
#MainMenu {{ display: none !important; }}

::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {t["scrollbar_bg"]}; }}
::-webkit-scrollbar-thumb {{ background: {t["accent"]}; border-radius: 2px; }}

/* ── Header ── */
.pd-header {{
    background: {t["bg"]};
    border-bottom: 1px solid {t["border_header"]};
    padding: 1rem 2rem;
    display: flex; align-items: center;
    justify-content: space-between;
    flex-wrap: wrap; gap: 12px;
}}
.pd-logo-row {{ display: flex; align-items: center; gap: 12px; }}
.pd-logo-icon {{
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #3B82F6, #1D4ED8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
}}
.pd-logo-text {{ font-size: 1.4rem; font-weight: 800; color: {t["text"]}; letter-spacing: -0.5px; }}
.pd-logo-text span {{ color: {t["accent"]}; }}
.pd-badge {{
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 600;
    background: {t["toggle_bg"]};
    color: {t["toggle_color"]};
    border: 1px solid {t["toggle_border"]};
    border-radius: 4px; padding: 2px 8px; letter-spacing: 1px;
}}
.pd-status {{ display: flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #34D399; }}
.pd-status-dot {{ width: 7px; height: 7px; background: #34D399; border-radius: 50%; animation: pulse-dot 2s infinite; }}
@keyframes pulse-dot {{
    0%,100% {{ opacity:1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }}
    50%      {{ opacity:.8; box-shadow: 0 0 0 5px rgba(52,211,153,0); }}
}}

/* ── Layout ── */
.pd-content {{ padding: 1.5rem 2rem; }}
@media (max-width: 768px) {{
    .pd-header  {{ padding: .75rem 1rem; }}
    .pd-content {{ padding: 1rem; }}
}}

/* ── Cards ── */
.pd-card {{
    background: {t["bg2"]};
    border: 1px solid {t["border"]};
    border-radius: 16px; padding: 1.5rem;
    margin-bottom: 16px; transition: border-color .2s;
}}
.pd-card:hover {{ border-color: {t["accent"]}; }}
.pd-card-title {{
    font-size: .68rem; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    color: {t["card_title"]}; margin-bottom: 1rem;
}}

/* ── Metrics ── */
.pd-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }}
.pd-metric {{ background: {t["bg2"]}; border: 1px solid {t["border"]}; border-radius: 12px; padding: 1rem; transition: border-color .2s; }}
.pd-metric:hover {{ border-color: {t["accent"]}; }}
.pd-metric-val {{ font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: {t["metric_val"]}; }}
.pd-metric-lbl {{ font-size: .72rem; color: {t["text_muted"]}; margin-top: 2px; }}

/* ── Verdicts ── */
.verdict-legit {{ background: rgba(16,185,129,.1);  border: 1px solid rgba(16,185,129,.3); color: #059669; border-radius: 12px; padding: 1.25rem; text-align: center; }}
.verdict-phish {{ background: rgba(239,68,68,.1);   border: 1px solid rgba(239,68,68,.3);  color: #DC2626; border-radius: 12px; padding: 1.25rem; text-align: center; }}
.verdict-ai    {{ background: rgba(245,158,11,.1);  border: 1px solid rgba(245,158,11,.3); color: #D97706; border-radius: 12px; padding: 1.25rem; text-align: center; }}
.verdict-title {{ font-size: 1.3rem; font-weight: 800; }}
.verdict-sub   {{ font-size: .8rem; color: inherit; opacity: .8; margin-top: 4px; }}

/* ── Confidence bar ── */
.conf-wrap {{ margin: .75rem 0; }}
.conf-hdr  {{ display: flex; justify-content: space-between; font-size: .75rem; color: {t["text_muted"]}; margin-bottom: 6px; }}
.conf-track {{ background: {t["conf_track"]}; border-radius: 99px; height: 8px; }}
.conf-fill  {{ height: 8px; border-radius: 99px; }}

/* ── LIME pills ── */
.lime-wrap {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
.lime-pill {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 3px 10px; border-radius: 99px; }}
.lime-pos  {{ background: rgba(16,185,129,.15); color: #059669; border: 1px solid rgba(16,185,129,.3); }}
.lime-neg  {{ background: rgba(239,68,68,.15);  color: #DC2626; border: 1px solid rgba(239,68,68,.3); }}

/* ── Risk badges ── */
.risk-critical {{ background: rgba(139,0,0,.15);   color: #DC2626; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; }}
.risk-high     {{ background: rgba(239,68,68,.15);  color: #DC2626; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; }}
.risk-medium   {{ background: rgba(245,158,11,.15); color: #D97706; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; }}
.risk-low      {{ background: rgba(16,185,129,.15); color: #059669; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; }}

/* ── URL rows ── */
.url-row-safe     {{ border-left: 3px solid #34D399; padding: 6px 10px; margin-bottom: 6px; background: rgba(16,185,129,.06);  border-radius: 0 8px 8px 0; }}
.url-row-medium   {{ border-left: 3px solid #F59E0B; padding: 6px 10px; margin-bottom: 6px; background: rgba(245,158,11,.06);  border-radius: 0 8px 8px 0; }}
.url-row-high     {{ border-left: 3px solid #EF4444; padding: 6px 10px; margin-bottom: 6px; background: rgba(239,68,68,.06);   border-radius: 0 8px 8px 0; }}
.url-row-critical {{ border-left: 3px solid #DC2626; padding: 6px 10px; margin-bottom: 6px; background: rgba(220,38,38,.08);   border-radius: 0 8px 8px 0; }}
.url-domain {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: {t["url_domain"]}; word-break: break-all; }}
.url-flags  {{ font-size: 11px; color: {t["url_flags"]}; margin-top: 2px; }}

/* ── Header analysis rows ── */
.hdr-row {{
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 6px 0; border-bottom: 1px solid {t["border"]}; gap: 12px;
}}
.hdr-label {{ font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: {t["text_muted"]}; min-width: 110px; flex-shrink: 0; }}
.hdr-value {{ font-size: 12px; color: {t["text"]}; word-break: break-all; }}
.hdr-flag  {{ font-size: 11px; color: #F59E0B; margin-top: 3px; }}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
    background:  {t["input_bg"]}     !important;
    border:      1px solid {t["input_border"]} !important;
    border-radius: 10px              !important;
    color:       {t["input_text"]}   !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size:   13px                !important;
}}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {{
    border-color: {t["accent"]}                       !important;
    box-shadow:   0 0 0 1px {t["accent"]}             !important;
}}

/* ── Buttons ── */
[data-testid="stButton"] > button {{
    background:    linear-gradient(135deg, {t["accent"]}, {t["accent_hover"]}) !important;
    color:         #fff   !important;
    border:        none   !important;
    border-radius: 10px   !important;
    font-family:   'Syne', sans-serif !important;
    font-weight:   700    !important;
    font-size:     .85rem !important;
    padding:       .65rem 1.5rem !important;
    width:         100%   !important;
    transition:    all .2s !important;
}}
[data-testid="stButton"] > button:hover {{
    opacity:    .92 !important;
    box-shadow: 0 0 20px rgba(59,130,246,.35) !important;
    transform:  translateY(-1px) !important;
}}
[data-testid="stDownloadButton"] > button {{
    background:    linear-gradient(135deg, #059669, #047857) !important;
    color:         #fff   !important;
    border:        none   !important;
    border-radius: 10px   !important;
    font-family:   'Syne', sans-serif !important;
    font-weight:   700    !important;
    font-size:     .85rem !important;
    padding:       .65rem 1.5rem !important;
    width:         100%   !important;
    transition:    all .2s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background:    linear-gradient(135deg, #10B981, #059669) !important;
    box-shadow:    0 0 20px rgba(16,185,129,.35) !important;
    transform:     translateY(-1px) !important;
}}

/* ── Misc Streamlit overrides ── */
[data-testid="stRadio"] label     {{ color: {t["text_sub"]} !important; font-size: .85rem !important; }}
[data-testid="stFileUploader"]    {{ background: {t["bg2"]} !important; border: 1px dashed {t["border"]} !important; border-radius: 10px !important; }}
[data-testid="stSpinner"]         {{ color: {t["accent"]} !important; }}
[data-testid="stAlert"]           {{ background: {t["bg2"]} !important; border-left: 3px solid {t["accent"]} !important; color: {t["text"]} !important; }}
.stDataFrame                      {{ background: {t["bg2"]} !important; border-radius: 10px !important; }}
[data-testid="stExpander"]        {{ background: {t["bg2"]} !important; border: 1px solid {t["border"]} !important; border-radius: 10px !important; }}
[data-testid="stMetricValue"]     {{ color: {t["metric_val"]} !important; }}
[data-testid="stMetricLabel"]     {{ color: {t["text_muted"]} !important; }}
.js-plotly-plot .plotly .bg       {{ fill: transparent !important; }}

/* ── Hero ── */
.hero-title  {{ font-size: clamp(1.6rem,3.5vw,2.4rem); font-weight: 800; color: {t["text"]}; line-height: 1.2; margin-bottom: .5rem; }}
.hero-accent {{ color: {t["accent"]}; }}
.hero-sub    {{ font-size: .9rem; color: {t["text_sub"]}; line-height: 1.7; max-width: 560px; }}

/* ── Feature grid ── */
.feature-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 1.5rem; }}
.feature-item {{
    background: {t["bg2"]}; border: 1px solid {t["border"]};
    border-radius: 12px; padding: 1.1rem;
    transition: border-color .2s, transform .2s;
}}
.feature-item:hover {{ border-color: {t["accent"]}; transform: translateY(-2px); }}
.feature-icon {{ font-size: 1.4rem; margin-bottom: 8px; }}
.feature-name {{ font-size: .85rem; font-weight: 700; color: {t["text"]}; }}
.feature-desc {{ font-size: .75rem; color: {t["text_muted"]}; margin-top: 3px; line-height: 1.5; }}

@media (max-width: 640px) {{
    .verdict-title {{ font-size: 1rem; }}
    .pd-metric-val {{ font-size: 1.2rem; }}
    .hero-title    {{ font-size: 1.4rem; }}
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
        flags.append(f"⚠️ From domain ({frm_domain}) ≠ Reply-To domain ({rpt_domain})")

    if ret_path and frm_domain and ret_domain and frm_domain != ret_domain:
        flags.append(f"⚠️ From domain ({frm_domain}) ≠ Return-Path domain ({ret_domain})")

    # SPF/DKIM/DMARC failures
    if headers["SPF"]   == "fail": flags.append("🚨 SPF check FAILED — sender not authorised")
    if headers["DKIM"]  == "fail": flags.append("🚨 DKIM signature FAILED — message may be tampered")
    if headers["DMARC"] == "fail": flags.append("🚨 DMARC policy FAILED")

    # Suspicious subject keywords
    subj_triggers = ["urgent", "verify", "suspended", "confirm", "password", "prize",
                     "winner", "account", "click", "free", "limited", "act now", "alert"]
    for kw in subj_triggers:
        if kw in subj.lower():
            flags.append(f"⚠️ Suspicious subject keyword: '{kw}'")
            break

    # No Message-ID
    if not headers["Message-ID"]:
        flags.append("⚠️ Missing Message-ID header (common in spoofed emails)")

    # Excessive hops
    hops = int(headers["Received hops"])
    if hops > 6:
        flags.append(f"⚠️ Unusually high hop count ({hops}) — may indicate routing obfuscation")

    return {"fields": headers, "flags": flags}


def _auth_badge(val: str) -> str:
    if val == "pass":
        return '<span style="color:#34D399;font-weight:700;font-family:\'IBM Plex Mono\',monospace">✔ pass</span>'
    if val == "fail":
        return '<span style="color:#EF4444;font-weight:700;font-family:\'IBM Plex Mono\',monospace">✘ fail</span>'
    return '<span style="color:#94A3B8;font-family:\'IBM Plex Mono\',monospace">— none</span>'


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
        font=dict(color=T["chart_font"], family="IBM Plex Mono, monospace"),
        height=height,
        margin=dict(l=0, r=0, t=30, b=0),
    )


def _verdict_html(pred_label: str, confidence: float, risk: str) -> str:
    if "ai" in pred_label or "generated" in pred_label:
        css   = "verdict-ai"
        icon  = "🤖"
        title = "AI-Generated Phishing"
        sub   = "High-sophistication AI-crafted threat detected"
    elif "phishing" in pred_label:
        css   = "verdict-phish"
        icon  = "🚨"
        title = "Traditional Phishing"
        sub   = "Classic phishing patterns detected"
    else:
        css   = "verdict-legit"
        icon  = "✅"
        title = "Legitimate Email"
        sub   = "No phishing indicators found"

    conf_color = "#34D399" if "legit" in pred_label else (
        "#F59E0B" if "ai" in pred_label else "#EF4444"
    )
    return f"""
    <div class="{css}">
      <div class="verdict-title">{icon} {title}</div>
      <div class="verdict-sub">{sub}</div>
    </div>
    <div class="conf-wrap">
      <div class="conf-hdr">
        <span>Confidence</span>
        <span style="color:{conf_color};font-family:'IBM Plex Mono',monospace;font-weight:600">
          {confidence:.1f}%
        </span>
      </div>
      <div class="conf-track">
        <div class="conf-fill" style="width:{confidence}%;background:{conf_color}"></div>
      </div>
    </div>
    <div style="display:flex;gap:8px;justify-content:center;margin-top:8px">
      <span class="{_risk_css(risk)}">{risk} Risk</span>
    </div>
    """


# ── Header ─────────────────────────────────────────────────────────────────────
header_col, toggle_col = st.columns([8, 1])
with header_col:
    st.markdown(f"""
    <div class="pd-header">
      <div class="pd-logo-row">
        <div class="pd-logo-icon">🛡️</div>
        <div>
          <div class="pd-logo-text">Phish<span>Detect</span> AI</div>
          <div class="pd-badge">ML-POWERED · CYBERSECURITY</div>
        </div>
      </div>
      <div class="pd-status"><div class="pd-status-dot"></div>SYSTEM OPERATIONAL</div>
    </div>
    """, unsafe_allow_html=True)

with toggle_col:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if st.button(f"{T['toggle_icon']} {T['toggle_label']}", key="theme_toggle", use_container_width=True):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# ── Nav ────────────────────────────────────────────────────────────────────────
pages = {
    "home":      "🏠 Home",
    "scan":      "📧 Scan Email",
    "url":       "🔗 URL Checker",
    "dashboard": "📊 Dashboard",
    "about":     "ℹ️ About",
}
nav_cols = st.columns(len(pages))
for i, (key, label) in enumerate(pages.items()):
    with nav_cols[i]:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pd-card">
      <div class="hero-title">
        Detect Phishing Emails with <span class="hero-accent">AI Precision</span>
      </div>
      <p class="hero-sub">
        PhishDetect AI uses machine learning to classify emails as Legitimate,
        Traditional Phishing, or AI-Generated Phishing — with LIME word-level
        explanations, email header analysis, and real-time URL safety checks.
      </p>
    </div>
    <div class="feature-grid">
      <div class="feature-item">
        <div class="feature-icon">🤖</div>
        <div class="feature-name">ML Classification</div>
        <div class="feature-desc">TF-IDF + classifier tuned for 3-class phishing detection</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">🔬</div>
        <div class="feature-name">LIME Explainability</div>
        <div class="feature-desc">Word-level explanations showing exactly what triggered the verdict</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">📨</div>
        <div class="feature-name">Header Analysis</div>
        <div class="feature-desc">Sender/receiver, SPF, DKIM, DMARC, Reply-To mismatch detection</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">🔗</div>
        <div class="feature-name">URL Analysis</div>
        <div class="feature-desc">Auto-extraction and risk-scoring of every URL in an email</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">📄</div>
        <div class="feature-name">PDF Reports</div>
        <div class="feature-desc">Downloadable threat reports for documentation and compliance</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">🎨</div>
        <div class="feature-name">Dark / Light Theme</div>
        <div class="feature-desc">Toggle between dark and light mode with the button top-right</div>
      </div>
    </div>
    <div class="pd-metrics" style="margin-top:16px">
      <div class="pd-metric"><div class="pd-metric-val">3</div><div class="pd-metric-lbl">Threat classes</div></div>
      <div class="pd-metric"><div class="pd-metric-val">LIME</div><div class="pd-metric-lbl">Explainability</div></div>
      <div class="pd-metric"><div class="pd-metric-val">HDR</div><div class="pd-metric-lbl">Header checks</div></div>
      <div class="pd-metric"><div class="pd-metric-val">PDF</div><div class="pd-metric-lbl">Reports</div></div>
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
        st.markdown('<div class="pd-card-title">✉️ Email Input</div>', unsafe_allow_html=True)

        input_method = st.radio(
            "Input method:", ["✏️ Paste Text", "📁 Upload File"],
            horizontal=True, label_visibility="collapsed",
        )

        email_text = ""
        if input_method == "✏️ Paste Text":
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

        analyze_btn = st.button("🔍 Analyze Email", key="analyze", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Email Header Analysis card (left column, below input) ──────────
        if analyze_btn and email_text.strip():
            hdr = parse_email_headers(email_text)
            fields = hdr["fields"]
            hdr_flags = hdr["flags"]

            st.markdown('<div class="pd-card">', unsafe_allow_html=True)
            st.markdown('<div class="pd-card-title">📨 Email Header Analysis</div>', unsafe_allow_html=True)

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
        st.markdown('<div class="pd-card-title">📊 Analysis Results</div>', unsafe_allow_html=True)

        if analyze_btn:
            if not email_text.strip():
                st.warning("⚠️ Please provide email content first.")
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
                    st.markdown('<div class="pd-card-title" style="margin-top:12px">🔬 LIME Word Explanations</div>', unsafe_allow_html=True)
                    pills_html = '<div class="lime-wrap">'
                    for w in pos[:8]:
                        pills_html += f'<span class="lime-pill lime-pos">+{w}</span>'
                    for w in neg[:8]:
                        pills_html += f'<span class="lime-pill lime-neg">−{w}</span>'
                    pills_html += "</div>"
                    st.markdown(pills_html, unsafe_allow_html=True)
                    st.markdown(
                        '<p style="font-size:.72rem;color:#64748B;margin-top:6px">'
                        '🟢 Green = pushes toward predicted class &nbsp;|&nbsp; 🔴 Red = pushes away</p>',
                        unsafe_allow_html=True,
                    )

                # URL analysis
                url_results = check_urls_in_email(email_text)
                if url_results:
                    st.markdown('<div class="pd-card-title" style="margin-top:12px">🔗 URLs Found</div>', unsafe_allow_html=True)
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
                        "📥 Download PDF Report",
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
        st.markdown('<div class="pd-card-title">🔗 URL Input</div>', unsafe_allow_html=True)
        url_input = st.text_input(
            "Enter URL", placeholder="https://example.com/path?query=value",
            label_visibility="collapsed",
        )
        check_btn = st.button("🔍 Check URL", key="check_url", use_container_width=True)

        st.markdown('<div class="pd-card-title" style="margin-top:1.5rem">📋 Bulk Check (one URL per line)</div>', unsafe_allow_html=True)
        bulk_input = st.text_area(
            "Bulk URLs", height=140,
            placeholder="https://url1.com\nhttps://url2.com\n…",
            label_visibility="collapsed",
        )
        bulk_btn = st.button("🔍 Check All URLs", key="check_bulk", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">📊 URL Analysis</div>', unsafe_allow_html=True)

        if check_btn and url_input.strip():
            ur = check_url(url_input.strip())
            cls        = _url_row_cls(ur["risk"])
            badge_cls  = _risk_css(ur["risk"])
            flags_str  = " · ".join(ur.get("flags", [])) or "No issues detected"
            safe_label = "✅ Safe" if ur["safe"] else "⚠️ Suspicious"
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
              <div style="font-size:2.5rem;margin-bottom:.75rem">🔗</div>
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
          <div style="font-size:3rem;margin-bottom:1rem">📊</div>
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
      <div class="pd-card-title">ℹ️ About PhishDetect AI</div>
      <p style="color:{T['text_sub']};line-height:1.8;font-size:.9rem">
        PhishDetect AI is a machine-learning-powered email threat detection tool built with
        Streamlit. It classifies emails into three categories: <strong>Legitimate</strong>,
        <strong>Traditional Phishing</strong>, and <strong>AI-Generated Phishing</strong>.
      </p>
    </div>

    <div class="pd-card">
      <div class="pd-card-title">🏗️ Architecture</div>
      <div class="feature-grid">
        <div class="feature-item">
          <div class="feature-icon">📝</div>
          <div class="feature-name">Text Preprocessing</div>
          <div class="feature-desc">HTML decoding, URL/email stripping, lowercasing, alpha-only filtering</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">📐</div>
          <div class="feature-name">TF-IDF Vectorizer</div>
          <div class="feature-desc">Fixed vocabulary trained at fit-time, always loaded from vectorizer.pkl</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">🌲</div>
          <div class="feature-name">ML Classifier</div>
          <div class="feature-desc">Scikit-learn model trained on labelled phishing datasets</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">🔬</div>
          <div class="feature-name">LIME Explainer</div>
          <div class="feature-desc">Local interpretable model-agnostic explanations at word level</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">📨</div>
          <div class="feature-name">Header Analysis</div>
          <div class="feature-desc">From/Reply-To mismatch, SPF/DKIM/DMARC, hop count, Message-ID checks</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">🔗</div>
          <div class="feature-name">URL Heuristics</div>
          <div class="feature-desc">IP detection, typosquatting, shorteners, TLD abuse — no external API needed</div>
        </div>
        <div class="feature-item">
          <div class="feature-icon">📄</div>
          <div class="feature-name">PDF Reports</div>
          <div class="feature-desc">ReportLab-generated threat reports with verdict, probabilities, and LIME results</div>
        </div>
      </div>
    </div>

    <div class="pd-card">
      <div class="pd-card-title">⚠️ Disclaimer</div>
      <p style="color:{T['text_sub']};line-height:1.8;font-size:.85rem">
        PhishDetect AI is intended for security research and awareness purposes only.
        No automated tool provides 100% accuracy — always apply human judgement for
        high-stakes decisions. Do not submit confidential email content to any public deployment.
      </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
