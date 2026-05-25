import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import base64

from phishdetect_util import (
    predict_email,
    get_lime_explanation,
    extract_text_from_file,
    COLOR_MAP,
    generate_pdf_report
)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="PhishDetect AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM CSS ----------
st.markdown('''
<style>
    /* ── Background ── */
    .main { background-color: #fff0f5; }
    .stApp { background-color: #fff0f5; }
    .css-1d391kg { background-color: #ffe0ec; }
    [data-testid="stSidebar"] { background-color: #fce4ec; }

    /* ── Global text ── */
    html, body, [class*="css"] {
        color: #880e4f;
        font-weight: 600;
    }

    /* ── Headings ── */
    h1, h2, h3, h4 {
        color: #880e4f !important;
        font-weight: 800 !important;
        letter-spacing: 0.5px;
    }

    /* ── Sidebar text ── */
    .css-1d391kg p, .css-1d391kg label {
        color: #880e4f !important;
        font-weight: 700;
    }

    /* ── Metric cards ── */
    .metric-card {
        background: #fce4ec;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 2px solid #e91e8c;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 900;
        color: #880e4f;
        margin: 0;
    }
    .metric-label {
        color: #ad1457;
        font-size: 0.9rem;
        font-weight: 700;
    }

    /* ── Prediction verdict box ── */
    .pred-box {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .legitimate {
        background: #e8f5e9;
        border: 3px solid #2e7d32;
    }
    .traditional_phishing {
        background: #e3f2fd;
        border: 3px solid #1565c0;
    }
    .ai_generated_phishing {
        background: #fce4ec;
        border: 3px solid #c62828;
    }

    /* ── Buttons ── */
    .stButton>button {
        background-color: #e91e8c !important;
        color: white !important;
        border-radius: 10px !important;
        font-weight: 800 !important;
        font-size: 1rem !important;
        border: none !important;
        padding: 10px 24px !important;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #880e4f !important;
    }

    /* ── Text input / text area ── */
    .stTextArea textarea, .stTextInput input {
        background-color: #fff !important;
        border: 2px solid #e91e8c !important;
        color: #880e4f !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
    }

    /* ── Radio buttons ── */
    .stRadio label {
        color: #880e4f !important;
        font-weight: 700 !important;
    }

    /* ── Progress bar ── */
    .stProgress > div > div {
        background-color: #e91e8c !important;
    }

    /* ── Dataframe ── */
    .dataframe { border-color: #e91e8c !important; }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        color: #880e4f !important;
        font-weight: 800 !important;
    }

    /* ── Info / success / error boxes ── */
    .stAlert {
        border-left: 5px solid #e91e8c !important;
        font-weight: 700;
    }
</style>
''', unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if 'history' not in st.session_state:
    st.session_state.history = []

def add_to_history(email_text, result):
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'email_preview': email_text[:100] + '...',
        'prediction': result['prediction'],
        'confidence': result['confidence'],
        'risk': result['risk_level']
    })

# ---------- SIDEBAR ----------
with st.sidebar:
    st.image("https://img.icons8.com/external-flatart-icons-outline-flatarticons/64/ffffff/external-cyber-security-digital-marketing-flatart-icons-outline-flatarticons.png", width=80)
    st.title("PhishDetect AI")
    st.markdown("---")
    page = st.radio("Navigation", ["🏠 Home", "📧 Scan Email", "📊 Dashboard", "ℹ️ About"])
    st.markdown("---")
    st.markdown("### 🎨 Legend")
    st.success("🟢 Legitimate")
    st.info("🔵 Traditional Phishing")
    st.error("🔴 AI-Generated Phishing")

# ---------- HOME ----------
if page == "🏠 Home":
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("<h1 style='color:#00ff88;'>🛡️ PhishDetect AI</h1>", unsafe_allow_html=True)
        st.markdown("### AI-Powered Email Threat Detection")
        st.write("Protect yourself from sophisticated phishing attacks using machine learning.")
        st.markdown("---")
        st.markdown('''
        **🔍 Features:**
        - Classify emails into **Legitimate**, **Traditional Phishing**, or **AI-Generated Phishing**
        - **LIME explanations** show why an email is suspicious
        - **PDF reports** for documentation
        - **Dashboard analytics** to track scans
        ''')
    with col2:
        st.image("https://img.icons8.com/external-flatart-icons-outline-flatarticons/128/ffffff/external-shield-security-flatart-icons-outline-flatarticons.png", width=150)
        st.metric("Live Model Accuracy", "89.2%", "on test set")

# ---------- SCAN EMAIL ----------
elif page == "📧 Scan Email":
    st.markdown("<h2 style='color:#00ff88;'>📧 Email Scanner</h2>", unsafe_allow_html=True)

    input_method = st.radio("Input method:", ["✏️ Paste Text", "📁 Upload File"], horizontal=True)

    email_text = ""
    if input_method == "✏️ Paste Text":
        email_text = st.text_area("Paste email content:", height=250, placeholder="Paste the full email including headers...")
    else:
        uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'docx'])
        if uploaded_file:
            email_text = extract_text_from_file(uploaded_file)
            if email_text:
                st.success(f"File loaded: {uploaded_file.name}")
                with st.expander("Preview extracted text"):
                    st.text(email_text[:500])
            else:
                st.error("Could not extract text from file.")

    if st.button("🔍 Analyze Email", type="primary", disabled=not email_text.strip()):
        with st.spinner("Scanning for threats..."):
            result = predict_email(email_text)
            lime_result = get_lime_explanation(email_text)

        add_to_history(email_text, result)

        col1, col2 = st.columns([1, 1.5])
        with col1:
            st.markdown("### 🔮 Verdict")
            pred = result['prediction']
            box_class = pred
            emoji = COLOR_MAP[pred]['emoji']
            hex_color = COLOR_MAP[pred]['hex']
            st.markdown(f'''
            <div class="pred-box {box_class}">
                <h1 style="color:{hex_color};">{emoji} {pred.replace("_"," ").upper()}</h1>
                <h3>Confidence: {result["confidence"]:.2f}%</h3>
                <p>Risk Level: <strong>{result["risk_level"]}</strong></p>
            </div>
            ''', unsafe_allow_html=True)

            st.markdown("**Confidence Meter**")
            st.progress(int(result['confidence']))

            probs = result['probabilities']
            prob_df = pd.DataFrame({
                'Class': [c.replace('_',' ').title() for c in probs.keys()],
                'Probability': [v*100 for v in probs.values()]
            })
            fig = px.bar(prob_df, x='Probability', y='Class', orientation='h',
                         color='Class',
                         color_discrete_map={
                             'Legitimate': '#00ff88',
                             'Traditional Phishing': '#3399ff',
                             'Ai Generated Phishing': '#ff4444'
                         })
            fig.update_layout(showlegend=False, margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("### 🔎 Why this decision?")
            word_weights = lime_result['word_weights']
            words = [w[0] for w in word_weights][::-1]
            weights = [w[1] for w in word_weights][::-1]
            fig_lime = go.Figure(go.Bar(
                x=weights,
                y=words,
                orientation='h',
                marker_color=['#00ff88' if w>0 else '#ff4444' for w in weights]
            ))
            fig_lime.update_layout(margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_lime, use_container_width=True)

            st.markdown("**Top influential words:**")
            for word, weight in word_weights[:8]:
                icon = "✅" if weight > 0 else "❌"
                st.write(f"{icon} **{word}** ({weight:+.4f})")

        st.markdown("---")
        if st.button("📥 Download PDF Report"):
            with st.spinner("Generating report..."):
                pdf_path = generate_pdf_report(email_text, result, word_weights)
                with open(pdf_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="PhishDetect_Report.pdf">Click here to download</a>'
                    st.markdown(href, unsafe_allow_html=True)

# ---------- DASHBOARD ----------
elif page == "📊 Dashboard":
    st.markdown("<h2 style='color:#00ff88;'>📊 Threat Dashboard</h2>", unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("No scans yet. Go to **Scan Email** to start analyzing.")
    else:
        hist = st.session_state.history
        total = len(hist)
        phishing = sum(1 for h in hist if h['prediction'] != 'legitimate')
        legit = total - phishing
        avg_conf = np.mean([h['confidence'] for h in hist])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Scans", total)
        m2.metric("Phishing Detected", phishing, delta_color="inverse")
        m3.metric("Legitimate", legit)
        m4.metric("Avg Confidence", f"{avg_conf:.1f}%")

        pie_fig = px.pie(
            values=[legit, phishing],
            names=['Legitimate', 'Phishing'],
            color_discrete_sequence=['#00ff88', '#ff4444'],
            title="Scan Results Distribution"
        )
        st.plotly_chart(pie_fig, use_container_width=True)

        st.markdown("### 📋 Scan History")
        hist_df = pd.DataFrame(hist)
        st.dataframe(hist_df, use_container_width=True)

# ---------- ABOUT ----------
else:
    st.markdown("<h2 style='color:#00ff88;'>ℹ️ About</h2>", unsafe_allow_html=True)
    st.markdown('''
    **PhishDetect AI** is a final year project that uses machine learning to detect
    AI‑generated phishing emails – a new breed of threats that often bypass traditional filters.

    **Technology Stack:**
    - Python, scikit‑learn, TF‑IDF
    - Logistic Regression (tuned)
    - LIME for explainability
    - Streamlit for the dashboard
    - Plotly for interactive charts

    **Color Coding:**
    - 🟢 Legitimate
    - 🔵 Traditional phishing
    - 🔴 AI‑generated phishing
    ''')
