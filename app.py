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
    dummy_email_features = np.zeros(21)
    X = np.array([dummy_email_features])

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
# LIME EXPLANATION
# ============================
def get_lime_explanation(email_text, num_features=10):
    """Generate LIME word weights. Builds explainer if missing."""
    model, vec, lc = load_components()

    def predict_fn(texts):
        cleaned = [clean_text(t) for t in texts]
        dummy_features = np.zeros((len(cleaned), 21))
        return model.predict_proba(dummy_features)

    if 'explainer' not in lc:
        lc['explainer'] = LimeTextExplainer(class_names=list(model.classes_))

    explainer = lc['explainer']
    cleaned = clean_text(email_text)
    X = np.zeros((1, 21))
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
# URL CHECKING
# ============================
def check_url(url):
    """Check if a URL is likely a phishing link based on heuristics."""
    if not isinstance(url, str) or not url.strip():
        return {
            "url": url,
            "prediction": "unknown",
            "confidence": 0.0,
            "risk_level": "Unknown",
            "risk_emoji": "⚪"
        }

    suspicious_keywords = [
        "login", "verify", "secure", "account", "update", "bank",
        "confirm", "password", "signin", "webscr", "free",
        "lucky", "winner", "click", "urgent", "suspend", "unusual",
        "validate", "authenticate", "billing", "checkout"
    ]
    suspicious_tlds = [".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf", ".gq"]

    score = 0
    url_lower = url.lower()

    # Keyword matches
    score += sum(1 for kw in suspicious_keywords if kw in url_lower)

    # Suspicious TLD
    score += sum(2 for tld in suspicious_tlds if url_lower.endswith(tld))

    # IP address used instead of domain
    score += 2 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0

    # Too many subdomains
    score += 1 if url.count('.') > 4 else 0

    # Very long URL
    score += 1 if len(url) > 100 else 0

    # Contains @ symbol (used to trick browsers)
    score += 2 if '@' in url else 0

    # Multiple hyphens in domain
    domain_part = url.split('/')[2] if len(url.split('/')) > 2 else url
    score += 1 if domain_part.count('-') > 2 else 0

    if score >= 4:
        prediction = "ai_generated_phishing"
        risk_level = "Critical"
        risk_emoji = "🔴"
        confidence = float(min(90 + score, 99))
    elif score >= 2:
        prediction = "traditional_phishing"
        risk_level = "High"
        risk_emoji = "🔵"
        confidence = float(min(60 + score * 5, 89))
    else:
        prediction = "legitimate"
        risk_level = "Low"
        risk_emoji = "🟢"
        confidence = float(max(90 - score * 10, 60))

    return {
        "url": url,
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": risk_level,
        "risk_emoji": risk_emoji
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
