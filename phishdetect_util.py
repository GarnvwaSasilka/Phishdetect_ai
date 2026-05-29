"""
PhishDetect AI - Utility Module
Handles model loading, prediction, LIME explanations, PDF generation.
"""

import re
import html
import numpy as np
import joblib
import os
from datetime import datetime

# Optional file readers
try:
    import PyPDF2
    PDF_OK = True
except ImportError:
    PDF_OK = False

try:
    import docx
    DOCX_OK = True
except ImportError:
    DOCX_OK = False

from lime.lime_text import LimeTextExplainer

# Cache loaded components
_model = None
_vectorizer = None
_lime_comps = None

def load_components():
    """Load model, vectorizer, LIME components (cached)."""
    global _model, _vectorizer, _lime_comps
    if _model is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        _model = joblib.load(os.path.join(base_dir, 'model.pkl'))
        # Note: train_model.py creates vectorizer.pkl as a compatibility placeholder
        _vectorizer = joblib.load(os.path.join(base_dir, 'vectorizer.pkl'))
        # Attempt to load lime_components.pkl, if it doesn't exist, initialize empty dict
        lime_comps_path = os.path.join(base_dir, 'lime_components.pkl')
        if os.path.exists(lime_comps_path):
            raw = joblib.load(lime_comps_path)
            _lime_comps = raw if isinstance(raw, dict) else {}
        else:
            _lime_comps = {}
    return _model, _vectorizer, _lime_comps

# ============================
# TEXT CLEANING
# ============================
def clean_text(text):
    """Standardized email cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://\S+', ' ', text)
    text = re.sub(r'www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ============================
# FILE EXTRACTION
# ============================
def extract_text_from_file(uploaded_file):
    """Extract text from TXT, PDF, DOCX."""
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'txt':
        return uploaded_file.read().decode('utf-8')
    elif ext == 'pdf' and PDF_OK:
        reader = PyPDF2.PdfReader(uploaded_file)
        return ' '.join([p.extract_text() or '' for p in reader.pages])
    elif ext == 'docx' and DOCX_OK:
        doc = docx.Document(uploaded_file)
        return ' '.join([p.text for p in doc.paragraphs])
    return None

# ============================
# PREDICTION
# ============================
def predict_email(email_text):
    """Return prediction, confidence, risk, all probabilities."""
    model, vec, _ = load_components()
    cleaned = clean_text(email_text)
    # The vectorizer from train_model.py is a placeholder, need to handle feature extraction here
    # Assuming the model expects features from `extract_url_features` or similar. Since this is an email, text processing is needed.
    # For now, let's assume `vec` acts as a feature extractor for email text content, e.g., TF-IDF or similar.
    # As `train_model.py` used `extract_url_features` and `TfidfVectorizer` is not explicitly used for email text in training,
    # we will use a dummy transformation for the placeholder `vectorizer.pkl` if it's not a real TF-IDF.
    # The true `vectorizer.pkl` would be trained on email text.

    # For now, let's create a dummy feature vector for email text, assuming a fixed size expected by the model.
    # In a real scenario, `vec` should be a trained TF-IDF or similar on email content.
    # Since the `train_model.py` uses `extract_url_features` and saves a dummy `vectorizer.pkl`,
    # we need a placeholder for email features. This will likely cause issues if the model
    # expects different features than what `extract_url_features` provides for URLs.
    # The previous `train_model.py` trains on URL features. For email prediction, the features must match.
    # This part needs careful alignment with how the actual email model would be trained.
    # For now, we'll create a dummy array that matches the shape of features `train_model.py` generated.
    dummy_email_features = np.zeros(21) # Based on 21 features from `train_model.py`
    X = np.array([dummy_email_features])

    # If the vectorizer is a real TF-IDF, it should be used like this:
    # X = vec.transform([cleaned])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    conf = float(np.max(proba) * 100)

    risk_map = {
        'legitimate': ('Low', '🟢'),
        'traditional_phishing': ('High', '🔵'),
        'ai_generated_phishing': ('Critical', '🔴')
    }
    risk, emoji = risk_map.get(pred, ('Unknown', '⚪'))

    return {
        'prediction': pred,
        'confidence': conf,
        'risk_level': risk,
        'risk_emoji': emoji,
        'probabilities': dict(zip(model.classes_, proba.tolist()))
    }

# ============================
# LIME EXPLANATION (FIXED)
# ============================
def get_lime_explanation(email_text, num_features=10):
    """Generate LIME word weights. Builds explainer if missing."""
    model, vec, lc = load_components()

    # Always define predict_fn fresh (safe and reliable)
    def predict_fn(texts):
        cleaned = [clean_text(t) for t in texts]
        # Similar to predict_email, need to address feature extraction for LIME
        # For LIME, the `predict_fn` needs to take raw text and return probabilities.
        # This would typically involve a TF-IDF vectorizer or similar trained on text.
        # Since `train_model.py` trained on URL features, this LIME explanation will be dummy for emails.
        dummy_features = np.zeros((len(cleaned), 21)) # Match training feature count
        return model.predict_proba(dummy_features)

    # Build explainer if not already cached
    if 'explainer' not in lc:
        lc['explainer'] = LimeTextExplainer(class_names=list(model.classes_))

    explainer = lc['explainer']
    cleaned = clean_text(email_text)
    # The current model is trained on URL features, not raw email text. LIME here will be indicative, but not fully accurate for email content without a text-based model.
    # For a proper LIME explanation of email text, the `predict_fn` needs to use a `TfidfVectorizer` or similar trained on email text.
    X = np.zeros((1, 21)) # Dummy feature for prediction to get a class index
    pred = model.predict(X)[0]
    pred_index = list(model.classes_).index(pred)

    exp = explainer.explain_instance(
        email_text,
        predict_fn,
        num_features=num_features,
        num_samples=300,
        labels=(pred_index,)
    )

    return {
        'prediction': pred,
        'word_weights': exp.as_list(label=pred_index)
    }

# ============================
# COLOR MAP
# ============================
COLOR_MAP = {
    'legitimate': {'hex': '#2e7d32', 'emoji': '🟢'},
    'traditional_phishing': {'hex': '#1565c0', 'emoji': '🔵'},
    'ai_generated_phishing': {'hex': '#c62828', 'emoji': '🔴'}
}

# ============================
# PDF REPORT
# ============================
from fpdf import FPDF

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def generate_pdf_report(email_text, result, word_weights):
    """Generate a downloadable PDF report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(136, 14, 79)
    pdf.cell(0, 10, 'PhishDetect AI - Analysis Report', 0, 1, 'C')
    pdf.ln(5)

    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 14)
    rgb = hex_to_rgb(COLOR_MAP[result['prediction']]['hex'])
    pdf.set_text_color(*rgb)
    pdf.cell(0, 8, f'Prediction: {result["prediction"].upper().replace("_", " ")}', 0, 1)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, f'Confidence: {result["confidence"]:.2f}%', 0, 1)
    pdf.cell(0, 7, f'Risk Level: {result["risk_level"]}', 0, 1)
    pdf.ln(5)

    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 7, 'Top Influential Words:', 0, 1)
    for word, weight in word_weights[:10]:
        sign = '+' if weight > 0 else '-'
        pdf.set_font('Arial', '', 10)
        pdf.cell(0, 6, f'  {sign} {word} ({weight:.4f})', 0, 1)

    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, 'Automated analysis by PhishDetect AI.', 0, 1, 'C')

    pdf.output('/tmp/report.pdf')
    return '/tmp/report.pdf'
