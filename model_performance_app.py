
import streamlit as st
import json
from pathlib import Path

# ---- Load the pre-computed results ----
st.set_page_config(page_title="Model Performance", page_icon="📊")

@st.cache_data
def load_results():
    # Adjust the path if you placed the files inside a folder
    with open("evaluation/meajor_accuracy_results.json") as f:
        results = json.load(f)
    return results

results = load_results()

st.title("📊 Model Performance")
st.markdown("Evaluation on the **MeAJOR Corpus** (peer-reviewed, LREC-COLING 2024)")

# ---- Metrics cards ----
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{results['accuracy']:.2%}")
with col2:
    st.metric("Precision", f"{results['precision']:.2%}")
with col3:
    st.metric("Recall", f"{results['recall']:.2%}")
with col4:
    st.metric("F1-Score", f"{results['f1_score']:.2%}")

st.markdown("---")
st.subheader("📈 ROC-AUC Score")
st.markdown(f"**ROC-AUC:** {results['roc_auc']:.4f}")

# ---- Dataset info ----
st.subheader("📚 Dataset Information")
st.markdown(f"""
- **Dataset:** {results['dataset']}
- **Citation:** {results['citation']}
- **Total samples used:** {results['total_samples']:,}
- **Class distribution:** {results['legitimate_ratio']*100:.0f}% Legitimate / {results['phishing_ratio']*100:.0f}% Phishing
""")

# ---- Display the evaluation plot ----
st.subheader("📊 Evaluation Plots")
# Load the pre-saved image
img_path = Path("evaluation/meajor_accuracy_results.png")
if img_path.exists():
    st.image(str(img_path), caption="Confusion matrix, ROC curve, metrics, feature importance")
else:
    st.warning("Image not found. Please upload 'meajor_accuracy_results.png' to the evaluation folder.")

# ---- Top features (optional, if you saved them in the JSON) ----
# If you added feature importance to the JSON, display them here.
# Otherwise you can skip this part.
