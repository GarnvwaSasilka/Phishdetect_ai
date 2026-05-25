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
        _vectorizer = joblib.load(os.path.join(base_dir, 'vectorizer.pkl'))
        raw = joblib.load(os.path.join(base_dir, 'lime_components.pkl'))
        _lime_comps = raw if isinstance(raw, dict) else {}
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
    X = vec.transform([cleaned])
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
        vectors = vec.transform(cleaned)
        return model.predict_proba(vectors)

    # Build explainer if not already cached
    if 'explainer' not in lc:
        lc['explainer'] = LimeTextExplainer(class_names=list(model.classes_))

    explainer = lc['explainer']
    cleaned = clean_text(email_text)
    X = vec.transform([cleaned])
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
