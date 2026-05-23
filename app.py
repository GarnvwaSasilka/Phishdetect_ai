import streamlit as st
import matplotlib.pyplot as plt
import datetime
import os
from utils import predict_email, get_lime_explanation, COLOR_MAP
from pdf_generator import generate_report

st.set_page_config(page_title="PhishDetect AI", page_icon="🛡️", layout="wide")
st.markdown("<style>.main-header{font-size:2.5rem;font-weight:bold;color:#003366;text-align:center;}.pred-box{padding:20px;border-radius:10px;margin:10px 0;text-align:center;}.legit{background-color:#e8f5e9;border:3px solid #4caf50;}.trad{background-color:#e3f2fd;border:3px solid #2196f3;}.ai{background-color:#ffebee;border:3px solid #f44336;}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.title("🛡️ PhishDetect AI")
    page = st.radio("Navigation", ["🏠 Home", "📧 Email Analyzer", "ℹ️ About"])
    st.success("🟢 Legitimate")
    st.info("🔵 Traditional Phishing")
    st.error("🔴 AI-Generated Phishing")

if page == "🏠 Home":
    st.markdown('<p class="main-header">🛡️ PhishDetect AI</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center;color:#666;">ML-Based Detection of AI-Generated Phishing Emails</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.success("### 🟢 Legitimate\nSafe emails")
    c2.info("### 🔵 Traditional Phishing\nClassic scams")
    c3.error("### 🔴 AI-Generated Phishing\nAdvanced threats")
    st.info("👈 Go to **Email Analyzer** to start!")

elif page == "📧 Email Analyzer":
    st.markdown("## 📧 Email Analyzer")
    email_text = st.text_area("Paste email:", height=200)
    if st.button("🔍 Analyze Email", type="primary", disabled=not email_text.strip()):
        with st.spinner("Analyzing..."):
            result = predict_email(email_text)
            lime_result = get_lime_explanation(email_text)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 🔮 Prediction")
            pred = result['prediction']
            bc = {'legitimate':'legit','traditional_phishing':'trad','ai_generated_phishing':'ai'}[pred]
            st.markdown(f'<div class="pred-box {bc}"><h2 style="color:{COLOR_MAP[pred]["hex"]};">{COLOR_MAP[pred]["emoji"]} {pred.replace("_"," ").upper()}</h2><p>Confidence: {result["confidence"]:.2f}%</p><p>Risk: {result["risk_emoji"]} {result["risk_level"]}</p></div>', unsafe_allow_html=True)
            probs = result['probabilities']
            fig, ax = plt.subplots(figsize=(6,2.5))
            classes = list(probs.keys())
            values = [probs[c]*100 for c in classes]
            ax.barh([c.replace('_',' ').title() for c in classes], values, color=[COLOR_MAP[c]['hex'] for c in classes], edgecolor='black')
            ax.set_xlim([0,100])
            for i, v in enumerate(values):
                ax.text(v+1, i, f'{v:.1f}%', va='center')
            st.pyplot(fig)
        with c2:
            st.markdown("### 🔍 LIME Explanation")
            ww = lime_result['word_weights']
            words = [w[0] for w in ww][::-1]
            weights = [w[1] for w in ww][::-1]
            fig, ax = plt.subplots(figsize=(6,3))
            ax.barh(range(len(words)), weights, color=['#2ecc71' if w>0 else '#e74c3c' for w in weights], edgecolor='black')
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words)
            ax.axvline(x=0, color='black')
            st.pyplot(fig)
        if st.button("📥 Download PDF Report"):
            with st.spinner("Generating PDF..."):
                pdf_path = generate_report(email_text, result['prediction'], result['confidence'], f"{result['risk_level']} Risk", result['probabilities'], ww, COLOR_MAP)
                with open(pdf_path, 'rb') as f:
                    st.download_button("💾 Download PDF", f.read(), f"report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf")
                if os.path.exists(pdf_path):
                    os.remove(pdf_path)

else:
    st.markdown("## ℹ️ About\n### Final Year Project\n🟢 Legitimate | 🔵 Traditional Phishing | 🔴 AI-Generated Phishing")

st.markdown(" preconceived notion---")
st.caption("© 2024 PhishDetect AI | Final Year Project")
