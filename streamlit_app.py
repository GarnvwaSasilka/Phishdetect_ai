import streamlit as st
from phishdetect_util import check_url

st.set_page_config(page_title="PhishDetect Dashboard", layout="wide")

# Your CSS (avoid /* ... */ comments, use clean styles)
st.markdown("""
<style>
.main-header {
    background-color: #f0f2f6;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Image handling – use a relative path that exists in the zip
st.image("assets/logo.png", width=200)

# Rest of your app...