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
    st.session_state.history.append({
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M"),
        "email_preview": email_text[:80] + "...",
        "prediction":    result["prediction"],
        "confidence":    result["confidence"],
        "risk":          result["risk_level"],
    })

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

            pred = str(result.get("prediction") or "unknown")
            ll   = pred.lower()
            if "legit" in ll:
                vclass, icon, bar_color = "verdict-legit", "✅", "#34D399"
            elif "ai" in ll or "generated" in ll:
                vclass, icon, bar_color = "verdict-ai",    "⚠️", "#FCD34D"
            else:
                vclass, icon, bar_color = "verdict-phish", "🚨", "#FCA5A5"

            label_clean = pred.replace("_", " ").title()
            risk        = result.get("risk_level", "—")
            risk_cls    = "risk-high" if "high" in risk.lower() else ("risk-medium" if "med" in risk.lower() else "risk-low")
            conf        = result["confidence"]
            conf_pct    = int(conf) if conf > 1 else int(conf * 100)

            st.markdown(f"""
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
                <div class="conf-fill" style="width:{conf_pct}%;background:{bar_color}"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

            # Probability chart
            probs   = result.get("probabilities", {})
            prob_df = pd.DataFrame({
                "Class":       [c.replace("_", " ").title() for c in probs.keys()],
                "Probability": [round(v * 100, 1) for v in probs.values()],
            })
            fig_prob = px.bar(
                prob_df, x="Probability", y="Class", orientation="h",
                color="Class",
                color_discrete_map={
                    "Legitimate":              "#34D399",
                    "Traditional Phishing":    "#60A5FA",
                    "Ai Generated Phishing":   "#FCD34D",
                },
            )
            fig_prob.update_layout(
                showlegend=False,
                margin=dict(l=0, r=0, t=8, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", size=11),
                xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155"),
                yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                height=140,
            )
            st.plotly_chart(fig_prob, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # LIME section
            st.markdown('<div class="pd-card">', unsafe_allow_html=True)
            st.markdown('<div class="pd-card-title">🔬 Why this verdict? (LIME)</div>', unsafe_allow_html=True)

            word_weights = lime_result.get("word_weights", [])

            if word_weights:
                words   = [w[0] for w in word_weights][::-1]
                weights = [w[1] for w in word_weights][::-1]
                fig_lime = go.Figure(go.Bar(
                    x=weights, y=words, orientation="h",
                    marker_color=["#34D399" if w > 0 else "#FCA5A5" for w in weights],
                    marker_line_width=0,
                ))
                fig_lime.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8", size=11),
                    xaxis=dict(gridcolor="#334155", zerolinecolor="#475569"),
                    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
                    height=220,
                )
                st.plotly_chart(fig_lime, use_container_width=True)

                pills = "".join(
                    f'<span class="lime-pill {"lime-pos" if w>0 else "lime-neg"}">{word}</span>'
                    for word, w in word_weights[:14]
                )
                st.markdown(f'<div class="lime-wrap">{pills}</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # PDF download — fixed for Android & mobile
            st.markdown('<div style="margin-top:4px">', unsafe_allow_html=True)
            try:
                from reportlab.lib.pagesizes import A4
                from reportlab.lib.styles import ParagraphStyle
                from reportlab.lib.units import mm
                from reportlab.lib import colors
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
                from reportlab.lib.enums import TA_LEFT, TA_CENTER

                buf = BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4,
                                        leftMargin=20*mm, rightMargin=20*mm,
                                        topMargin=20*mm, bottomMargin=20*mm)

                C_BG   = colors.HexColor("#0F172A")
                C_CARD = colors.HexColor("#1E293B")
                C_TEXT = colors.HexColor("#E2E8F0")
                C_MUTE = colors.HexColor("#94A3B8")
                C_BLUE = colors.HexColor("#3B82F6")
                C_BORD = colors.HexColor("#334155")
                C_RES  = colors.HexColor(bar_color)

                t_title  = ParagraphStyle("tt", fontSize=20, textColor=C_TEXT, fontName="Helvetica-Bold", spaceAfter=4)
                t_sub    = ParagraphStyle("ts", fontSize=10, textColor=C_MUTE, fontName="Helvetica", spaceAfter=14)
                t_label  = ParagraphStyle("tl", fontSize=8,  textColor=C_MUTE, fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=2, leading=12)
                t_body   = ParagraphStyle("tb", fontSize=10, textColor=C_TEXT, fontName="Helvetica", leading=14)
                t_footer = ParagraphStyle("tf", fontSize=8,  textColor=C_MUTE, fontName="Helvetica", alignment=TA_CENTER)

                story = []
                story.append(Paragraph("🛡️ PhishDetect AI", t_title))
                story.append(Paragraph("Email Threat Analysis Report", t_sub))
                story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORD))
                story.append(Spacer(1, 10))

                ts_str = datetime.now().strftime("%B %d, %Y at %H:%M:%S")
                meta   = [["Generated", ts_str], ["Engine", "ML Classifier + LIME Explainer"], ["Version", "1.0"]]
                mt = Table(meta, colWidths=[40*mm, 125*mm])
                mt.setStyle(TableStyle([
                    ("FONTNAME",     (0,0),(-1,-1), "Helvetica"),
                    ("FONTSIZE",     (0,0),(-1,-1), 9),
                    ("TEXTCOLOR",    (0,0),(0,-1),  C_MUTE),
                    ("TEXTCOLOR",    (1,0),(1,-1),  C_TEXT),
                    ("ROWBACKGROUNDS",(0,0),(-1,-1), [C_CARD, C_BG]),
                    ("TOPPADDING",   (0,0),(-1,-1), 5),
                    ("BOTTOMPADDING",(0,0),(-1,-1), 5),
                    ("LEFTPADDING",  (0,0),(-1,-1), 8),
                ]))
                story.append(mt)
                story.append(Spacer(1, 14))

                rd = [["VERDICT", "CONFIDENCE", "RISK LEVEL"],
                      [label_clean.upper(), f"{conf_pct}%", risk]]
                rt = Table(rd, colWidths=[65*mm, 45*mm, 55*mm])
                rt.setStyle(TableStyle([
                    ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
                    ("FONTNAME",      (0,1),(-1,1),  "Helvetica-Bold"),
                    ("FONTSIZE",      (0,0),(-1,0),  8),
                    ("FONTSIZE",      (0,1),(-1,1),  15),
                    ("TEXTCOLOR",     (0,0),(-1,0),  C_MUTE),
                    ("TEXTCOLOR",     (0,1),(0,1),   C_RES),
                    ("TEXTCOLOR",     (1,1),(1,1),   C_BLUE),
                    ("TEXTCOLOR",     (2,1),(2,1),   C_TEXT),
                    ("BACKGROUND",    (0,0),(-1,-1), C_CARD),
                    ("GRID",          (0,0),(-1,-1), 0.5, C_BORD),
                    ("TOPPADDING",    (0,0),(-1,-1), 9),
                    ("BOTTOMPADDING", (0,0),(-1,-1), 9),
                    ("LEFTPADDING",   (0,0),(-1,-1), 12),
                ]))
                story.append(rt)
                story.append(Spacer(1, 14))

                if word_weights:
                    story.append(Paragraph("KEY INDICATORS (LIME)", t_label))
                    pos_w = [w for w, s in word_weights if s > 0][:8]
                    neg_w = [w for w, s in word_weights if s <= 0][:8]
                    if pos_w:
                        story.append(Paragraph(f'<font color="#34D399">Phishing signals:</font> {", ".join(pos_w)}', t_body))
                    if neg_w:
                        story.append(Paragraph(f'<font color="#FCA5A5">Counter signals:</font> {", ".join(neg_w)}', t_body))
                    story.append(Spacer(1, 10))

                story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORD))
                story.append(Paragraph("EMAIL CONTENT", t_label))
                safe_email = email_text[:2000].replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                if len(email_text) > 2000:
                    safe_email += "... [truncated]"
                story.append(Paragraph(safe_email, t_body))
                story.append(Spacer(1, 14))
                story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORD))
                story.append(Spacer(1, 6))
                story.append(Paragraph("Generated by PhishDetect AI — Final Year Cybersecurity Project. For academic use only.", t_footer))

                doc.build(story)
                buf.seek(0)
                pdf_bytes = buf.read()

                fname = f"PhishDetect_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
                st.download_button(
                    label="📥  Download PDF Report",
                    data=pdf_bytes,
                    file_name=fname,
                    mime="application/pdf",
                    use_container_width=True,
                )

            except ImportError:
                st.warning("reportlab not installed — add it to requirements.txt to enable PDF export.")

            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="text-align:center;padding:3.5rem 1rem;color:#334155">
              <div style="font-size:3rem;margin-bottom:.75rem">🛡️</div>
              <div style="font-size:.85rem;color:#475569">Paste an email and click <strong style="color:#3B82F6">Analyze</strong> to see the verdict.</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── DASHBOARD ──────────────────────────────────────────────────────────────────
elif page == "dashboard":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)

    if not st.session_state.history:
        st.markdown("""
        <div class="pd-card" style="text-align:center;padding:3rem">
          <div style="font-size:2.5rem;margin-bottom:.75rem">📊</div>
          <div style="color:#64748B;font-size:.9rem">No scans yet. Go to <strong style="color:#3B82F6">Scan Email</strong> to begin.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        hist   = st.session_state.history
        total  = len(hist)
        phish  = sum(1 for h in hist if "legit" not in h["prediction"].lower())
        legit  = total - phish
        avg_c  = np.mean([h["confidence"] for h in hist])
        avg_c  = avg_c if avg_c <= 100 else avg_c  # already pct

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="stat-val">{total}</div><div class="stat-lbl">Total scans</div></div>
          <div class="stat-card"><div class="stat-val" style="color:#FCA5A5">{phish}</div><div class="stat-lbl">Threats found</div></div>
          <div class="stat-card"><div class="stat-val" style="color:#34D399">{legit}</div><div class="stat-lbl">Legitimate</div></div>
          <div class="stat-card"><div class="stat-val">{avg_c:.1f}%</div><div class="stat-lbl">Avg confidence</div></div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1], gap="medium")

        with c1:
            st.markdown('<div class="pd-card">', unsafe_allow_html=True)
            st.markdown('<div class="pd-card-title">📊 Distribution</div>', unsafe_allow_html=True)
            pred_counts = {}
            for h in hist:
                k = h["prediction"].replace("_", " ").title()
                pred_counts[k] = pred_counts.get(k, 0) + 1
            pie = px.pie(
                values=list(pred_counts.values()),
                names=list(pred_counts.keys()),
                color=list(pred_counts.keys()),
                color_discrete_map={
                    "Legitimate":            "#34D399",
                    "Traditional Phishing":  "#60A5FA",
                    "Ai Generated Phishing": "#FCD34D",
                },
                hole=0.55,
            )
            pie.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", size=11),
                showlegend=True,
                legend=dict(font=dict(color="#94A3B8")),
                margin=dict(l=0, r=0, t=0, b=0),
                height=220,
            )
            st.plotly_chart(pie, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="pd-card">', unsafe_allow_html=True)
            st.markdown('<div class="pd-card-title">📈 Confidence over time</div>', unsafe_allow_html=True)
            df_hist = pd.DataFrame(hist)
            line = px.line(
                df_hist, y="confidence",
                markers=True,
                color_discrete_sequence=["#3B82F6"],
            )
            line.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94A3B8", size=11),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155", title=""),
                yaxis=dict(gridcolor="#1E293B", zerolinecolor="#334155", title=""),
                height=220,
            )
            st.plotly_chart(line, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="pd-card">', unsafe_allow_html=True)
        st.markdown('<div class="pd-card-title">📋 Scan history</div>', unsafe_allow_html=True)
        df_show = pd.DataFrame(hist)[["timestamp","email_preview","prediction","confidence","risk"]]
        st.dataframe(df_show, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── ABOUT ──────────────────────────────────────────────────────────────────────
elif page == "about":
    st.markdown('<div class="pd-content">', unsafe_allow_html=True)
    st.markdown("""
    <div class="pd-card">
      <div class="pd-card-title">ℹ️ About PhishDetect AI</div>
      <p style="color:#94A3B8;font-size:.9rem;line-height:1.8;margin-bottom:1rem">
        PhishDetect AI is a final year cybersecurity project that uses machine learning to detect
        AI-generated phishing emails — a new class of threats that often bypass traditional filters.
      </p>
      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-top:1rem">
        <div style="background:#0F172A;border:1px solid #334155;border-radius:10px;padding:1rem">
          <div style="font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">ML Stack</div>
          <div style="font-size:.82rem;color:#E2E8F0;line-height:1.8">scikit-learn<br>TF-IDF Vectorizer<br>Logistic Regression<br>LIME Explainer</div>
        </div>
        <div style="background:#0F172A;border:1px solid #334155;border-radius:10px;padding:1rem">
          <div style="font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">App Stack</div>
          <div style="font-size:.82rem;color:#E2E8F0;line-height:1.8">Streamlit<br>Plotly<br>ReportLab<br>Python 3.11+</div>
        </div>
        <div style="background:#0F172A;border:1px solid #334155;border-radius:10px;padding:1rem">
          <div style="font-size:.7rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px">Threat Classes</div>
          <div style="font-size:.82rem;line-height:1.8">
            <span style="color:#34D399">● Legitimate</span><br>
            <span style="color:#60A5FA">● Traditional Phishing</span><br>
            <span style="color:#FCD34D">● AI-Generated Phishing</span>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:1.5rem;color:#334155;font-size:.72rem;border-top:1px solid #1E293B;margin-top:1rem">
  PhishDetect AI &nbsp;·&nbsp; Final Year Cybersecurity Project &nbsp;·&nbsp; Built with Streamlit
</div>
""", unsafe_allow_html=True)
