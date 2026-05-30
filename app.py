
import streamlit as st
from phishdetect_util import check_url

st.set_page_config(page_title="PhishDetect Dashboard", layout="wide")

st.markdown("""
<style>
.main-header {
    background-color: #f0f2f6;
    padding: 1rem;
}
</style>
""", unsafe_allow_html=True)

st.title("PhishDetect Dashboard")

# Rest of your app...
