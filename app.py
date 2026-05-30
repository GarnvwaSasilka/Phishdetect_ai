import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO

from phishdetect_util import (
    predict_email,
    get_lime_explanation,
    extract_text_from_file,
    COLOR_MAP,
    generate_pdf_report,
    check_url
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhishDetect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Syne:wght@400;600;800&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.stApp {
    background-color: #0F172A !important;
    color: #E2E8F0 !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stHeader"]  { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
footer { display: none !important; }
#MainMenu { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0F172A; }
::-webkit-scrollbar-thumb { background: #3B82F6; border-radius: 2px; }

/* ── Header ── */
.pd-header {
    background: #0F172A;
    border-bottom: 1px solid #1E3A5F;
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
}
.pd-logo-row { display: flex; align-items: center; gap: 12px; }
.pd-logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #3B82F6, #1D4ED8);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 22px; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(59,130,246,0.4);
}
.pd-logo-text { font-size: 1.4rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.5px; }
.pd-logo-text span { color: #3B82F6; }
.pd-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; font-weight: 600;
    background: rgba(59,130,246,0.12);
    color: #60A5FA;
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 4px;
    padding: 2px 8px;
    letter-spacing: 1px;
}
.pd-status { display: flex; align-items: center; gap: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #34D399; }
.pd-status-dot { width: 7px; height: 7px; background: #34D399; border-radius: 50%; animation: pulse-dot 2s infinite; }
@keyframes pulse-dot {
    0%,100% { opacity:1; box-shadow: 0 0 0 0 rgba(52,211,153,0.4); }
    50%      { opacity:.8; box-shadow: 0 0 0 5px rgba(52,211,153,0); }
}

/* ── Nav tabs ── */
.pd-nav {
    background: #0F172A;
    border-bottom: 1px solid #1E293B;
    padding: 0 2rem;
    display: flex;
    gap: 0;
    overflow-x: auto;
}
.pd-nav a {
    display: inline-block;
    padding: .75rem 1.25rem;
    font-size: .82rem;
    font-weight: 600;
    color: #64748B;
    text-decoration: none;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: color .15s, border-color .15s;
    cursor: pointer;
}
.pd-nav a.active { color: #3B82F6; border-bottom-color: #3B82F6; }
.pd-nav a:hover  { color: #E2E8F0; }

/* ── Content wrapper ── */
.pd-content { padding: 1.5rem 2rem; }
@media (max-width: 768px) {
    .pd-header  { padding: .75rem 1rem; }
    .pd-nav     { padding: 0 .75rem; }
    .pd-content { padding: 1rem; }
}

/* ── Cards ── */
.pd-card {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 16px;
    transition: border-color .2s;
}
.pd-card:hover { border-color: #3B82F6; }
.pd-card-title {
    font-size: .68rem;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #64748B;
    margin-bottom: 1rem;
}

/* ── Metric cards ── */
.pd-metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.pd-metric { background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 1rem; transition: border-color .2s; }
.pd-metric:hover { border-color: #3B82F6; }
.pd-metric-val { font-family: 'IBM Plex Mono', monospace; font-size: 1.6rem; font-weight: 600; color: #3B82F6; }
.pd-metric-lbl { font-size: .72rem; color: #64748B; margin-top: 2px; }

/* ── Result verdict ── */
.verdict-legit  { background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.3); color: #34D399; border-radius: 12px; padding: 1.25rem; text-align: center; }
.verdict-phish  { background: rgba(239,68,68,.1);  border: 1px solid rgba(239,68,68,.3);  color: #FCA5A5; border-radius: 12px; padding: 1.25rem; text-align: center; }
.verdict-ai     { background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3); color: #FCD34D; border-radius: 12px; padding: 1.25rem; text-align: center; }
.verdict-title  { font-size: 1.3rem; font-weight: 800; }
.verdict-sub    { font-size: .8rem; color: inherit; opacity: .75; margin-top: 4px; }

/* ── Confidence bar ── */
.conf-wrap { margin: .75rem 0; }
.conf-hdr  { display: flex; justify-content: space-between; font-size: .75rem; color: #64748B; margin-bottom: 6px; }
.conf-track { background: #0F172A; border-radius: 99px; height: 8px; }
.conf-fill  { height: 8px; border-radius: 99px; }

/* ── LIME pills ── */
.lime-wrap { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.lime-pill { font-family: 'IBM Plex Mono', monospace; font-size: 11px; padding: 3px 10px; border-radius: 99px; }
.lime-pos  { background: rgba(16,185,129,.15); color: #34D399; border: 1px solid rgba(16,185,129,.25); }
.lime-neg  { background: rgba(239,68,68,.15);  color: #FCA5A5; border: 1px solid rgba(239,68,68,.25); }

/* ── Risk badge ── */
.risk-high   { background: rgba(239,68,68,.15);  color: #FCA5A5; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; }
.risk-medium { background: rgba(245,158,11,.15); color: #FCD34D; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; }
.risk-low    { background: rgba(16,185,129,.15); color: #34D399; border-radius: 6px; padding: 2px 10px; font-size: 11px; font-weight: 700; }

/* ── Streamlit widget overrides ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {
    background: #0F172A !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
}
[data-testid="stTextArea"] textarea:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 1px #3B82F6 !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .85rem !important;
    padding: .65rem 1.5rem !important;
    width: 100% !important;
    transition: all .2s !important;
}
[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #3B82F6, #2563EB) !important;
    box-shadow: 0 0 20px rgba(59,130,246,.35) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .85rem !important;
    padding: .65rem 1.5rem !important;
    width: 100% !important;
    transition: all .2s !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #10B981, #059669) !important;
    box-shadow: 0 0 20px rgba(16,185,129,.35) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stRadio"] label  { color: #94A3B8 !important; font-size: .85rem !important; }
[data-testid="stFileUploader"] { background: #1E293B !important; border: 1px dashed #334155 !important; border-radius: 10px !important; }
[data-testid="stSpinner"]      { color: #3B82F6 !important; }
[data-testid="stAlert"]        { background: #1E293B !important; border-left: 3px solid #3B82F6 !important; color: #E2E8F0 !important; }
.stDataFrame                   { background: #1E293B !important; border-radius: 10px !important; }
[data-testid="stExpander"]     { background: #1E293B !important; border: 1px solid #334155 !important; border-radius: 10px !important; }

/* Plotly chart bg */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* ── Home hero ── */
.hero-title  { font-size: clamp(1.6rem, 3.5vw, 2.4rem); font-weight: 800; color: #F8FAFC; line-height: 1.2; margin-bottom: .5rem; }
.hero-accent { color: #3B82F6; }
.hero-sub    { font-size: .9rem; color: #94A3B8; line-height: 1.7; max-width: 560px; }

.feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-top: 1.5rem; }
.feature-item {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.1rem;
    transition: border-color .2s, transform .2s;
}
.feature-item:hover { border-color: #3B82F6; transform: translateY(-2px); }
.feature-icon { font-size: 1.4rem; margin-bottom: 8px; }
.feature-name { font-size: .85rem; font-weight: 700; color: #E2E8F0; }
.feature-desc { font-size: .75rem; color: #64748B; margin-top: 3px; line-height: 1.5; }

/* ── Dashboard ── */
.stat-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 12px; margin-bottom: 16px; }
.stat-card { background: #1E293B; border: 1px solid #334155; border-radius: 12px; padding: 1rem; text-align: center; }
.stat-val  { font-family: 'IBM Plex Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #3B82F6; }
.stat-lbl  { font-size: .72rem; color: #64748B; margin-top: 2px; }

/* ── Mobile ── */
@media (max-width: 640px) {
    .verdict-title { font-size: 1rem; }
    .pd-metric-val { font-size: 1.2rem; }
    .hero-title    { font-size: 1.4rem; }
    .pd-nav a      { padding: .65rem .9rem; font-size: .78rem; }
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "page" not in st.session_state:
    st.session_state.page = "home"

def add_to_history(email_text, result):
    # Convert numeric prediction to string for display in history
    pred_num = result["prediction"]
    pred_str = _map_prediction_to_label(pred_num)
    st.session_state.history.append({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "email_preview": email_text[:80] + "...",
        "prediction":    pred_str,
        "confidence":    result["confidence"],
        "risk":          result["risk_level"],
    })

def _map_prediction_to_label(pred_num):
    """Convert numeric model output (0,1,2) to string label."""
    if pred_num == 0:
        return "legitimate"
    elif pred_num == 1:
        return "traditional phishing"
    elif pred_num == 2:
        return "ai generated phishing"
    else:
        return "unknown"

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
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

# ── Nav ────────────────────────────────────────────────────────────────────────
pages = {"home": "🏠 Home", "scan": "📧 Scan Email", "dashboard": "📊 Dashboard", "about": "ℹ️ About"}
cols = st.columns(len(pages))
for i, (key, label) in enumerate(pages.items()):
    with cols[i]:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state.page = key
            st.rerun()

page = st.session_state.page

# ── HOME ───────────────────────────────────────────────────────────────────────
if page == "home":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)

    st.markdown("""
    <div class="pd-card">
      <div class="hero-title">Detect Phishing Emails with <span class="hero-accent">AI Precision</span></div>
      <p class="hero-sub">PhishDetect AI uses machine learning to classify emails into Legitimate, Traditional Phishing, or AI-Generated Phishing — with LIME-powered explanations so you know exactly why.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-item">
        <div class="feature-icon">🤖</div>
        <div class="feature-name">ML Classification</div>
        <div class="feature-desc">TF-IDF + Logistic Regression tuned for 3-class phishing detection</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">🔬</div>
        <div class="feature-name">LIME Explainability</div>
        <div class="feature-desc">Word-level explanations showing exactly what triggered the verdict</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">📄</div>
        <div class="feature-name">PDF Reports</div>
        <div class="feature-desc">Downloadable threat reports for documentation and compliance</div>
      </div>
      <div class="feature-item">
        <div class="feature-icon">📊</div>
        <div class="feature-name">Scan Analytics</div>
        <div class="feature-desc">Dashboard tracking your scan history, threat distribution, and trends</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="pd-metrics" style="margin-top:16px">
      <div class="pd-metric"><div class="pd-metric-val">3</div><div class="pd-metric-lbl">Threat classes</div></div>
      <div class="pd-metric"><div class="pd-metric-val">89.2%</div><div class="pd-metric-lbl">Model accuracy</div></div>
      <div class="pd-metric"><div class="pd-metric-val">LIME</div><div class="pd-metric-lbl">Explainability</div></div>
      <div class="pd-metric"><div class="pd-metric-val">PDF</div><div class="pd-metric-lbl">Export ready</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── SCAN EMAIL ─────────────────────────────────────────────────────────────────
elif page == "scan":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="medium")

    with left:
        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">✉️ Email Input</div>', unsafe_allow_html=True)

        input_method = st.radio("Input method:", ["✏️ Paste Text", "📁 Upload File"], horizontal=True, label_visibility="collapsed")

        email_text = ""
        if input_method == "✏️ Paste Text":
            email_text = st.text_area("Email content", height=240,
                                      placeholder="Paste the full email — headers, body, links...",
                                      label_visibility="collapsed")
        else:
            uploaded_file = st.file_uploader("Choose a .txt / .pdf / .docx file",
                                             type=["txt", "pdf", "docx"],
                                             label_visibility="collapsed")
            if uploaded_file:
                email_text = extract_text_from_file(uploaded_file)
                if email_text:
                    st.success(f"✓ Loaded: {uploaded_file.name}")
                    with st.expander("Preview"):
                        st.text(email_text[:400])
                else:
                    st.error("Could not extract text from this file.")

        analyze = st.button("🔍  Analyze Email", use_container_width=True, disabled=not bool(email_text.strip()))
        st.markdown('</div>', unsafe_allow_html=True)

        # Threat legend
        st.markdown("""
        <div class="pd-card">
          <div class="pd-card-title">⚡ Threat classes</div>
          <div style="display:flex;flex-direction:column;gap:10px;font-size:.82rem">
            <div style="display:flex;gap:10px;align-items:flex-start">
              <span style="background:rgba(16,185,129,.15);color:#34D399;border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;flex-shrink:0">LEGIT</span>
              <span style="color:#94A3B8">Normal communication — no deceptive intent detected.</span>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
              <span style="background:rgba(59,130,246,.15);color:#60A5FA;border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;flex-shrink:0">PHISH</span>
              <span style="color:#94A3B8">Traditional phishing — spoofed domains, known social engineering patterns.</span>
            </div>
            <div style="display:flex;gap:10px;align-items:flex-start">
              <span style="background:rgba(245,158,11,.15);color:#FCD34D;border-radius:6px;padding:2px 8px;font-size:10px;font-weight:700;flex-shrink:0">AI-GEN</span>
              <span style="color:#94A3B8">AI-generated phishing — sophisticated language, harder to detect manually.</span>
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

            add_to_history(email_text, result)

            # --- FIX: Convert numeric prediction to label string ---
            pred_num = result["prediction"]
            pred_label = _map_prediction_to_label(pred_num)
            ll = pred_label.lower()   # now a string, safe to call .lower()

            if "legit" in ll:
                vclass, icon, bar_color = "verdict-legit", "✅", "#34D399"
            elif "ai" in ll or "generated" in ll:
                vclass, icon, bar_color = "verdict-ai",    "⚠️", "#FCD34D"
            else:
                vclass, icon, bar_color = "verdict-phish", "🚨", "#FCA5A5"

            label_clean = pred_label.title()   # e.g., "Traditional Phishing"
            risk        = result.get("risk_level", "—")
            risk_cls    = "risk-high" if "high" in risk.lower() else ("risk-medium" if "med" in risk.lower() else "risk-low")
            conf        = result["confidence"]
            conf_pct    = int(conf) if conf > 1 else int(conf * 100)

            # Corrected f-string - properly terminated
            st.markdown(
                f"""
                <div class="{vclass}">
                  <div class="verdict-title">{icon}&nbsp;{label_clean}</div>
                  <div class="verdict-sub">Risk level: <span class="{risk_cls}">{risk}</span></div>
                </div>
                <div class="conf-wrap">
                  <div class="conf-hdr">
                    <span>Confidence</span>
                    <span style="font-family:'IBM Plex Mono',monospace;color:{bar_color};font-weight:600">{conf_pct}%</span>
                  </div>
                  <div class="conf-track">
                    <div class="conf-
