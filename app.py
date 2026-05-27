"""
PhishDetect AI - Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure page
st.set_page_config(
    page_title="PhishDetect AI",
    page_icon="🛡️",
    layout="wide"
)

# Try to import utility functions
try:
    from phishdetect_util import (
        check_url,
        generate_pdf_report,
        extract_features,
        load_model,
        load_vectorizer
    )
    st.sidebar.success("✅ Detection engine loaded")
except ImportError as e:
    st.error(f"Failed to load detection engine: {e}")
    st.info("Make sure phishdetect_util.py is in the same directory")
    st.stop()

# Title
st.title("🛡️ PhishDetect AI")
st.markdown("### Real-time Phishing URL Detection System")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This AI-powered tool analyzes URLs for phishing indicators using:
    - Machine Learning models
    - Feature extraction (45+ URL characteristics)
    - Real-time risk scoring
    """)
    
    # Model status
    st.header("Model Status")
    model = load_model()
    vectorizer = load_vectorizer()
    if model and vectorizer:
        st.success("✅ ML Model Active")
    else:
        st.warning("⚠️ Using Rule-based Detection")

# Main input
col1, col2 = st.columns([3, 1])
with col1:
    url_input = st.text_input(
        "Enter URL to analyze:",
        placeholder="https://example.com/login",
        help="Paste any URL you want to check for phishing"
    )

with col2:
    analyze_button = st.button("🔍 Analyze URL", type="primary", use_container_width=True)

# Results area
if analyze_button and url_input:
    with st.spinner("Analyzing URL..."):
        # Analyze the URL
        result = check_url(url_input, model, vectorizer)
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            risk_score = result.get('risk_score', 0)
            if result.get('is_phishing', False):
                st.metric("Risk Score", f"{risk_score:.1%}", delta="HIGH RISK", delta_color="inverse")
            else:
                st.metric("Risk Score", f"{risk_score:.1%}", delta="Low Risk", delta_color="normal")
        
        with col2:
            status = "⚠️ PHISHING" if result.get('is_phishing') else "✅ SAFE"
            st.metric("Status", status)
        
        with col3:
            method = result.get('detection_method', 'unknown').upper()
            st.metric("Detection Method", method)
        
        # Detailed analysis
        st.subheader("Detailed Analysis")
        
        # Risk factors
        if result.get('risk_factors'):
            st.write("**Risk Factors Identified:**")
            for factor in result['risk_factors']:
                st.warning(f"• {factor}")
        
        # Message
        st.info(result.get('message', 'Analysis complete'))
        
        # Generate report
        report_path = generate_pdf_report(result, f"/tmp/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        # Download button
        with open(report_path, 'rb') as f:
            st.download_button(
                label="📥 Download Report",
                data=f,
                file_name=f"phishdetect_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        
        # Feature breakdown
        with st.expander("View URL Features Analyzed"):
            features = result.get('features', {})
            if features:
                col1, col2 = st.columns(2)
                for i, (key, value) in enumerate(features.items()):
                    if i % 2 == 0:
                        col1.metric(key.replace('_', ' ').title(), value)
                    else:
                        col2.metric(key.replace('_', ' ').title(), value)

elif analyze_button and not url_input:
    st.warning("Please enter a URL to analyze")

# Footer
st.markdown("---")
st.caption("PhishDetect AI - Protecting users from phishing attacks")
