"""
app.py — PhishDetect AI  |  Streamlit front-end
Dark / Light theme toggle + URL checking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from collections import Counter

from phishdetect_util import (
    predict_email,
    get_lime_explanation,
    extract_text_from_file,
    COLOR_MAP,
    generate_pdf_report,
    check_url,
    check_urls_in_email,
    extract_urls,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishDetect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Session state ──────────────────────────────────────────────────────────────
if "history"    not in st.session_state: st.session_state.history   = []
if "page"       not in st.session_state: st.session_state.page      = "home"
if "dark_mode"  not in st.session_state: st.session_state.dark_mode = True

dark = st.session_state.dark_mode

# ── Theme token dictionaries ───────────────────────────────────────────────────
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


# ── CSS injection ──────────────────────────────────────────────────────────────
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
.verdict-legit  {{ background: rgba(16,185,129,.1);  border: 1px solid rgba(16,185,129,.3); color: #059669; border-radius: 12px; padding: 1.25rem; text-align: center; }}
.verdict-phish  {{ background: rgba(239,68,68,.1);   border: 1px solid rgba(239,68,68,.3);  color: #DC2626; border-radius: 12px; padding: 1.25rem; text-align: center; }}
.verdict-ai     {{ background: rgba(245,158,11,.1);  border: 1px solid rgba(245,158,11,.3); color: #D97706; border-radius: 12px; padding: 1.25rem; text-align: center; }}
.verdict-title  {{ font-size: 1.3rem; font-weight: 800; }}
.verdict-sub    {{ font-size: .8rem; color: inherit; opacity: .8; margin-top: 4px; }}

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
.url-domain {{ font-family: 'IBM Plex Mono', monospace; font-size: 12px; color: {t["url_domain"]}; }}
.url-flags  {{ font-size: 11px; color: {t["url_flags"]}; margin-top: 2px; }}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
    background: {t["input_bg"]} !important;
    border: 1px solid {t["input_border"]} !important;
    border-radius: 10px !important;
    color: {t["input_text"]} !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {{
    border-color: {t["accent"]} !important;
    box-shadow: 0 0 0 1px {t["accent"]} !important;
}}

/* ── Buttons ── */
[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, {t["accent"]}, {t["accent_hover"]}) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: .85rem !important;
    padding: .65rem 1.5rem !important; width: 100% !important;
    transition: all .2s !important;
}}
[data-testid="stButton"] > button:hover {{
    opacity: .92 !important;
    box-shadow: 0 0 20px rgba(59,130,246,.35) !important;
    transform: translateY(-1px) !important;
}}
[data-testid="stDownloadButton"] > button {{
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: .85rem !important;
    padding: .65rem 1.5rem !important; width: 100% !important;
    transition: all .2s !important;
}}
[data-testid="stDownloadButton"] > button:hover {{
    background: linear-gradient(135deg, #10B981, #059669) !important;
    box-shadow: 0 0 20px rgba(16,185,129,.35) !important;
    transform: translateY(-1px) !important;
}}

/* ── Misc Streamlit overrides ── */
[data-testid="stRadio"] label  {{ color: {t["text_sub"]} !important; font-size: .85rem !important; }}
[data-testid="stFileUploader"] {{ background: {t["bg2"]} !important; border: 1px dashed {t["border"]} !important; border-radius: 10px !important; }}
[data-testid="stSpinner"]      {{ color: {t["accent"]} !important; }}
[data-testid="stAlert"]        {{ background: {t["bg2"]} !important; border-left: 3px solid {t["accent"]} !important; color: {t["text"]} !important; }}
.stDataFrame                   {{ background: {t["bg2"]} !important; border-radius: 10px !important; }}
[data-testid="stExpander"]     {{ background: {t["bg2"]} !important; border: 1px solid {t["border"]} !important; border-radius: 10px !important; }}
[data-testid="stMetricValue"]  {{ color: {t["metric_val"]} !important; }}
[data-testid="stMetricLabel"]  {{ color: {t["text_muted"]} !important; }}

.js-plotly-plot .plotly .bg {{ fill: transparent !important; }}

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


# ── Helper functions ───────────────────────────────────────────────────────────
def _map_prediction_to_label(pred):
    if isinstance(pred, str):
        return pred.lower()
    try:
        pred_int = int(pred)
    except (TypeError, ValueError):
        return "unknown"
    return {0: "legitimate", 1: "traditional phishing", 2: "ai generated phishing"}.get(pred_int, "unknown")


def add_to_history(email_text, result):
    st.session_state.history.append({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "email_preview": email_text[:80] + "...",
        "prediction":    _map_prediction_to_label(result["prediction"]),
        "confidence":    result["confidence"],
        "risk":          result.get("risk_level", "—"),
    })


def _risk_css(risk):
    r = risk.lower()
    if "critical" in r: return "risk-critical"
    if "high"     in r: return "risk-high"
    if "medium"   in r: return "risk-medium"
    return "risk-low"


def _url_row_cls(risk):
    r = risk.lower()
    if "critical" in r: return "url-row-critical"
    if "high"     in r: return "url-row-high"
    if "medium"   in r: return "url-row-medium"
    return "url-row-safe"


def _chart_layout(height=220, t=None):
    if t is None: t = T
    return dict(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["chart_font"], family="IBM Plex Mono, monospace"),
        height=height, margin=dict(l=0, r=0, t=30, b=0),
    )


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
    # Small vertical spacer so the button lines up with the header
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if st.button(
        f"{T['toggle_icon']} {T['toggle_label']}",
        key="theme_toggle",
        use_container_width=True,
    ):
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
      <div class="hero-title">Detect Phishing Emails with <span class="hero-accent">AI Precision</span></div>
      <p class="hero-sub">PhishDetect AI uses machine learning to classify emails into Legitimate,
      Traditional Phishing, or AI-Generated Phishing — with LIME explanations and real-time URL safety analysis.</p>
    </div>
    <div class="feature-grid">
      <div class="feature-item"><div class="feature-icon">🤖</div>
        <div class="feature-name">ML Classification</div>
        <div class="feature-desc">TF-IDF + Logistic Regression tuned for 3-class phishing detection</div></div>
      <div class="feature-item"><div class="feature-icon">🔬</div>
        <div class="feature-name">LIME Explainability</div>
        <div class="feature-desc">Word-level explanations showing exactly what triggered the verdict</div></div>
      <div class="feature-item"><div class="feature-icon">🔗</div>
        <div class="feature-name">URL Analysis</div>
        <div class="feature-desc">Automatic extraction and risk-scoring of every URL in an email</div></div>
      <div class="feature-item"><div class="feature-icon">📄</div>
        <div class="feature-name">PDF Reports</div>
        <div class="feature-desc">Downloadable threat reports for documentation and compliance</div></div>
      <div class="feature-item"><div class="feature-icon">🎨</div>
        <div class="feature-name">Dark / Light Theme</div>
        <div class="feature-desc">Toggle between dark and light mode with the button in the top-right</div></div>
    </div>
    <div class="pd-metrics" style="margin-top:16px">
      <div class="pd-metric"><div class="pd-metric-val">3</div><div class="pd-metric-lbl">Threat classes</div></div>
      <div class="pd-metric"><div class="pd-metric-val">92.4%</div><div class="pd-metric-lbl">Model accuracy</div></div>
      <div class="pd-metric"><div class="pd-metric-val">LIME</div><div class="pd-metric-lbl">Explainability</div></div>
      <div class="pd-metric"><div class="pd-metric-val">URLs</div><div class="pd-metric-lbl">Auto-checked</div></div>
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
            horizontal=True, label_visibility="collapsed"
        )

        email_text = ""
        if input_method == "✏️ Paste Text":
            email_text = st.text_area(
                "Email content", height=240,
                placeholder="Paste the full email — headers, body, links...",
                label_visibility="collapsed"
            )
        else:
            uploaded_file = st.file_uploader(
                "Choose a .txt / .pdf / .docx file",
                type=["txt", "pdf", "docx"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                email_text = extract_text_from_file(uploaded_file)
                if email_text:
                    st.success(f"✓ Loaded: {uploaded_file.name}")
                    with st.expander("Preview"):
                        st.text(email_text[:400])
                else:
                    st.error("Could not extract text from this file.")

        analyze = st.button(
            "🔍  Analyze Email",
            use_container_width=True,
            disabled=not bool(email_text.strip())
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pd-card">
          <div class="pd-card-title">⚡ Threat classes</div>
          <div style="display:flex;flex-direction:column;gap:10px;font-size:.82rem">
            <div style="display:flex;gap:10px;align-items:flex-start">
              <span style="background:rgba(16,185,129,.15);color:#059669;border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;flex-shrink:0">LEGIT</span>
              <span style="color:{T['text_sub']}">Normal communication — no deceptive intent detected.</span>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
              <span style="background:rgba(59,130,246,.15);color:{T['accent']};border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;flex-shrink:0">PHISH</span>
              <span style="color:{T['text_sub']}">Traditional phishing — spoofed domains, social engineering patterns.</span>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
              <span style="background:rgba(245,158,11,.15);color:#D97706;border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;flex-shrink:0">AI-GEN</span>
              <span style="color:{T['text_sub']}">AI-generated phishing — sophisticated language, harder to detect manually.</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">🎯 Analysis Result</div>', unsafe_allow_html=True)

        if analyze and email_text.strip():
            with st.spinner("Scanning for threats..."):
                result      = predict_email(email_text)
                lime_result = get_lime_explanation(email_text)
                url_results = check_urls_in_email(email_text)

            add_to_history(email_text, result)

            pred_label = _map_prediction_to_label(result["prediction"])
            ll = pred_label.lower()

            if "legit" in ll:
                vclass, icon, bar_color = "verdict-legit", "✅", "#10B981"
            elif "ai" in ll or "generated" in ll:
                vclass, icon, bar_color = "verdict-ai",    "⚠️", "#F59E0B"
            else:
                vclass, icon, bar_color = "verdict-phish", "🚨", "#EF4444"

            risk     = result.get("risk_level", "—")
            conf     = result["confidence"]
            conf_pct = int(conf) if conf > 1 else int(conf * 100)

            st.markdown(f"""
            <div class="{vclass}">
              <div class="verdict-title">{icon}&nbsp;{pred_label.title()}</div>
              <div class="verdict-sub">Risk level: <span class="{_risk_css(str(risk))}">{risk}</span></div>
            </div>
            <div class="conf-wrap">
              <div class="conf-hdr">
                <span>Confidence</span>
                <span style="font-family:'IBM Plex Mono',monospace;color:{bar_color};font-weight:600">{conf_pct}%</span>
              </div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{conf_pct}%;background:{bar_color}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            probs = result.get("probabilities", {})
            if probs:
                prob_df = pd.DataFrame({
                    "Class":       [str(c).replace("_"," ").title() for c in probs],
                    "Probability": [round(float(v)*100,1) for v in probs.values()],
                })
                fig = px.bar(
                    prob_df, x="Probability", y="Class", orientation="h",
                    color="Class",
                    color_discrete_map={
                        "Legitimate":"#10B981","Traditional Phishing":"#3B82F6",
                        "Ai Generated Phishing":"#F59E0B","0":"#10B981","1":"#3B82F6","2":"#F59E0B",
                    },
                    text="Probability"
                )
                fig.update_layout(**_chart_layout(200))
                fig.update_traces(texttemplate='%{text}%', textposition='outside')
                st.plotly_chart(fig, use_container_width=True)

            # LIME pills
            if lime_result:
                pos, neg = lime_result.get("positive",[]), lime_result.get("negative",[])
                if pos or neg:
                    st.markdown('<div style="margin-top:16px;margin-bottom:6px"><strong>🔬 Key words influencing this verdict</strong></div>', unsafe_allow_html=True)
                    pills = (
                        "".join(f'<span class="lime-pill lime-pos">+ {w}</span>' for w in pos) +
                        "".join(f'<span class="lime-pill lime-neg">− {w}</span>' for w in neg)
                    )
                    st.markdown(f'<div class="lime-wrap">{pills}</div>', unsafe_allow_html=True)

            # URL results
            if url_results:
                high_risk = [u for u in url_results if u["risk"] in ("High","Critical")]
                badge = f' &nbsp;<span class="risk-high">{len(high_risk)} high-risk</span>' if high_risk else ""
                st.markdown(f'<div style="margin-top:20px;margin-bottom:8px"><strong>🔗 URLs found: {len(url_results)}</strong>{badge}</div>', unsafe_allow_html=True)
                for ur in url_results[:8]:
                    flags_txt = " · ".join(ur.get("flags",[])) or "No issues detected"
                    short = ur["url"][:70] + ("…" if len(ur["url"])>70 else "")
                    st.markdown(
                        f'<div class="{_url_row_cls(ur["risk"])}">'
                        f'<div class="url-domain">{short} <span class="{_risk_css(ur["risk"])}">{ur["risk"]}</span></div>'
                        f'<div class="url-flags">{flags_txt}</div></div>',
                        unsafe_allow_html=True
                    )

            # PDF download
            pdf_buf = generate_pdf_report(email_text, result, lime_result)
            if pdf_buf:
                st.download_button(
                    "📄 Download PDF Report", data=pdf_buf,
                    file_name=f"phish_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf", use_container_width=True
                )
        else:
            st.info("👈 Paste or upload an email and click 'Analyze Email' to see results here.")

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# URL CHECKER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "url":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    left_u, right_u = st.columns([1, 1], gap="medium")

    with left_u:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">🔗 URL Safety Checker</div>', unsafe_allow_html=True)

        mode = st.radio("Check mode:", ["Single URL","Paste email / bulk text"],
                        horizontal=True, label_visibility="collapsed")
        checked_urls = []

        if mode == "Single URL":
            single_url = st.text_input("Enter URL", placeholder="https://example.com/login",
                                       label_visibility="collapsed")
            if st.button("🔍 Check URL", use_container_width=True,
                         disabled=not bool(single_url.strip())):
                with st.spinner("Analysing URL..."):
                    checked_urls = [check_url(single_url.strip())]
        else:
            bulk = st.text_area("Paste email or text", height=200,
                                placeholder="Paste email body — all URLs will be extracted and checked.",
                                label_visibility="collapsed")
            if st.button("🔍 Extract & Check All URLs", use_container_width=True,
                         disabled=not bool(bulk.strip())):
                urls = extract_urls(bulk)
                if urls:
                    with st.spinner(f"Checking {len(urls)} URL(s)..."):
                        checked_urls = [check_url(u) for u in urls]
                else:
                    st.warning("No URLs found in the pasted text.")

        st.markdown(f"""
        <div style="margin-top:16px">
          <div class="pd-card-title">Risk levels</div>
          <div style="display:flex;flex-direction:column;gap:6px;font-size:.8rem;color:{T['text_sub']}">
            <div><span class="risk-low">LOW</span> &nbsp;No suspicious indicators detected.</div>
            <div><span class="risk-medium">MEDIUM</span> &nbsp;Minor concerns (HTTP, long URL).</div>
            <div><span class="risk-high">HIGH</span> &nbsp;Likely malicious (typosquatting, shortener).</div>
            <div><span class="risk-critical">CRITICAL</span> &nbsp;Strong phishing signal (raw IP, executable).</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right_u:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">🎯 URL Analysis Results</div>', unsafe_allow_html=True)

        if checked_urls:
            c1,c2,c3 = st.columns(3)
            c1.metric("Total", len(checked_urls))
            c2.metric("Safe",  sum(1 for u in checked_urls if u["safe"]))
            c3.metric("High+", sum(1 for u in checked_urls if u["risk"] in ("High","Critical")))

            st.markdown("<br>", unsafe_allow_html=True)
            for ur in checked_urls:
                flags_html = (
                    "".join(f'<li style="color:{T["url_flags"]};font-size:11px">{f}</li>' for f in ur.get("flags",[]))
                    or f'<li style="color:#059669;font-size:11px">No issues detected</li>'
                )
                st.markdown(
                    f'<div class="{_url_row_cls(ur["risk"])}" style="margin-bottom:10px">'
                    f'<div class="url-domain">{ur["url"][:90]}{"…" if len(ur["url"])>90 else ""} '
                    f'<span class="{_risk_css(ur["risk"])}">{ur["risk"]}</span></div>'
                    f'<ul style="margin:4px 0 0 16px;padding:0">{flags_html}</ul></div>',
                    unsafe_allow_html=True
                )

            if len(checked_urls) > 1:
                rc = Counter(u["risk"] for u in checked_urls)
                fig_u = px.pie(
                    values=list(rc.values()), names=list(rc.keys()),
                    color=list(rc.keys()),
                    color_discrete_map={"Low":"#10B981","Medium":"#F59E0B","High":"#EF4444","Critical":"#DC2626"},
                    title="Risk Distribution",
                )
                fig_u.update_layout(**_chart_layout(220))
                st.plotly_chart(fig_u, use_container_width=True)
        else:
            st.info("👈 Enter a URL or paste email text and click 'Check' to see results here.")

        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "dashboard":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    st.markdown('<div class="pd-card"><div class="pd-card-title">📈 Scan History & Analytics</div>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No scans yet. Go to 'Scan Email' to start.")
    else:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("Total Scans", len(df))
        c2.metric("Threats Detected", df[df["prediction"] != "legitimate"].shape[0] if "prediction" in df else 0)
        avg_conf = df["confidence"].mean() if "confidence" in df else 0
        c3.metric("Avg Confidence", f"{avg_conf:.1f}%")

        if "prediction" in df:
            fig = px.pie(df, names="prediction", title="Threat Distribution",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(**_chart_layout(300))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="pd-card">
      <div class="pd-card-title">🧠 About PhishDetect AI</div>
      <p style="color:{T['text']}"><strong>PhishDetect AI</strong> is an experimental cybersecurity tool that uses machine learning
      to classify emails into three categories: <strong>Legitimate</strong>, <strong>Traditional Phishing</strong>,
      and <strong>AI-Generated Phishing</strong>.</p>
      <p style="margin-top:12px;color:{T['text']}">The model uses TF-IDF vectorization and a logistic regression classifier,
      achieving ~92% accuracy on test data.</p>
      <p style="margin-top:12px;color:{T['text']}">For explainability, LIME highlights the most influential words behind each prediction.</p>
      <p style="margin-top:12px;color:{T['text']}">URL analysis runs locally — no external API — checking for raw IPs, URL shorteners,
      typosquatting, suspicious TLDs, and executable links.</p>
      <p style="margin-top:12px;color:{T['text']}">Use the <strong>{T['toggle_icon']} {T['toggle_label']}</strong> button (top right) to switch themes.</p>
      <p style="margin-top:12px;color:{T['text']}"><strong>Important:</strong> For educational and research purposes only.</p>
      <p style="margin-top:20px;font-size:0.75rem;color:{T['text_muted']}">
        PhishDetect AI — Built with Streamlit, scikit-learn, LIME, and ReportLab.</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
