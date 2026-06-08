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
        _model      = joblib.load(os.path.join(base_dir, 'model.pkl'))
        _vectorizer = joblib.load(os.path.join(base_dir, 'vectorizer.pkl'))
        lime_path   = os.path.join(base_dir, 'lime_components.pkl')
        if os.path.exists(lime_path):
            raw = joblib.load(lime_path)
            _lime_comps = raw if isinstance(raw, dict) else {}
        else:
            _lime_comps = {}
    return _model, _vectorizer, _lime_comps


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text):
    """Standardised email cleaning — must match training preprocessing."""
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


# ── File extraction ────────────────────────────────────────────────────────────
def extract_text_from_file(uploaded_file):
    """Extract plain text from TXT, PDF, or DOCX uploads."""
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'txt':
        return uploaded_file.read().decode('utf-8', errors='replace')
    elif ext == 'pdf' and PDF_OK:
        reader = PyPDF2.PdfReader(uploaded_file)
        return ' '.join([p.extract_text() or '' for p in reader.pages])
    elif ext == 'docx' and DOCX_OK:
        doc = docx.Document(uploaded_file)
        return ' '.join([p.text for p in doc.paragraphs])
    return None


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_email(email_text):
    """
    Vectorise the email text and run the trained model.
    Returns prediction label, confidence %, risk level, and per-class probabilities.
    """
    model, vec, _ = load_components()

    # BUG WAS HERE: code used np.zeros(21) instead of the actual email features.
    # FIX: clean the text and transform it through the same vectorizer used at training.
    cleaned = clean_text(email_text)
    X = vec.transform([cleaned])          # sparse matrix, shape (1, vocab_size)

    pred  = model.predict(X)[0]           # e.g. 'legitimate' or numpy int
    proba = model.predict_proba(X)[0]     # array of shape (n_classes,)
    conf  = float(np.max(proba) * 100)

    # Normalise prediction to a clean string label
    label = _normalise_label(pred)

    risk_map = {
        'legitimate':            ('Low',      '🟢'),
        'traditional_phishing':  ('High',     '🔵'),
        'ai_generated_phishing': ('Critical', '🔴'),
    }
    risk, emoji = risk_map.get(label, ('Unknown', '⚪'))

    return {
        'prediction':    label,
        'confidence':    conf,
        'risk_level':    risk,
        'risk_emoji':    emoji,
        'probabilities': {
            _normalise_label(cls): float(p)
            for cls, p in zip(model.classes_, proba)
        },
    }


def _normalise_label(pred):
    """
    Convert any model output — numpy.int64, int, or string — to a
    lowercase underscore string label so the rest of the app is consistent.
    """
    if isinstance(pred, str):
        return pred.lower().strip().replace(' ', '_')
    try:
        mapping = {
            0: 'legitimate',
            1: 'traditional_phishing',
            2: 'ai_generated_phishing',
        }
        return mapping.get(int(pred), 'unknown')
    except (TypeError, ValueError):
        return 'unknown'


# ── LIME explanation ───────────────────────────────────────────────────────────
def get_lime_explanation(email_text, num_features=10):
    """
    Return the top positive and negative words for the predicted class.
    """
    model, vec, lc = load_components()

    # BUG WAS HERE: predict_fn used np.zeros and ignored the text entirely.
    # FIX: vectorize each perturbed text sample through the real vectorizer.
    def predict_fn(texts):
        cleaned = [clean_text(t) for t in texts]
        X = vec.transform(cleaned)         # shape (n_samples, vocab_size)
        return model.predict_proba(X)

    if 'explainer' not in lc:
        lc['explainer'] = LimeTextExplainer(
            class_names=[_normalise_label(c) for c in model.classes_]
        )

    explainer = lc['explainer']

    # Determine the predicted class index so LIME explains the right class
    cleaned_text = clean_text(email_text)
    X_single     = vec.transform([cleaned_text])
    pred_raw     = model.predict(X_single)[0]
    pred_label   = _normalise_label(pred_raw)
    class_names  = [_normalise_label(c) for c in model.classes_]

    try:
        pred_index = class_names.index(pred_label)
    except ValueError:
        pred_index = 0

    exp = explainer.explain_instance(
        email_text,           # raw text — LIME does its own perturbation
        predict_fn,
        num_features=num_features,
        num_samples=300,
        labels=(pred_index,)
    )

    word_weights = exp.as_list(label=pred_index)

    # Split into positive (phishing signals) and negative (clean signals)
    positive = [w for w, weight in word_weights if weight > 0]
    negative = [w for w, weight in word_weights if weight < 0]

    return {
        'prediction':   pred_label,
        'positive':     positive,
        'negative':     negative,
        'word_weights': word_weights,   # kept for PDF report
    }


# ── URL checking ───────────────────────────────────────────────────────────────
def check_url(url):
    """Heuristic URL risk scorer."""
    if not isinstance(url, str) or not url.strip():
        return {"url": url, "prediction": "unknown", "confidence": 0.0,
                "risk_level": "Unknown", "risk_emoji": "⚪"}

    suspicious_keywords = [
        "login", "verify", "secure", "account", "update", "bank",
        "confirm", "password", "signin", "webscr", "free", "lucky",
        "winner", "click", "urgent", "suspend", "unusual", "validate",
        "authenticate", "billing", "checkout",
    ]
    suspicious_tlds = [".xyz", ".top", ".click", ".tk", ".ml", ".ga", ".cf", ".gq"]

    score     = 0
    url_lower = url.lower()

    score += sum(1 for kw  in suspicious_keywords if kw  in url_lower)
    score += sum(2 for tld in suspicious_tlds      if url_lower.endswith(tld))
    score += 2 if re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url) else 0
    score += 1 if url.count('.') > 4 else 0
    score += 1 if len(url) > 100 else 0
    score += 2 if '@' in url else 0
    domain = url.split('/')[2] if len(url.split('/')) > 2 else url
    score += 1 if domain.count('-') > 2 else 0

    if score >= 4:
        return {"url": url, "prediction": "ai_generated_phishing",
                "confidence": float(min(90 + score, 99)),
                "risk_level": "Critical", "risk_emoji": "🔴"}
    elif score >= 2:
        return {"url": url, "prediction": "traditional_phishing",
                "confidence": float(min(60 + score * 5, 89)),
                "risk_level": "High", "risk_emoji": "🔵"}
    else:
        return {"url": url, "prediction": "legitimate",
                "confidence": float(max(90 - score * 10, 60)),
                "risk_level": "Low", "risk_emoji": "🟢"}


# ── Colour map ─────────────────────────────────────────────────────────────────
COLOR_MAP = {
    'legitimate':            {'hex': '#2e7d32', 'emoji': '🟢'},
    'traditional_phishing':  {'hex': '#1565c0', 'emoji': '🔵'},
    'ai_generated_phishing': {'hex': '#c62828', 'emoji': '🔴'},
}


# ── PDF report ─────────────────────────────────────────────────────────────────
from fpdf import FPDF

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def generate_pdf_report(email_text, result, lime_result):
    """Generate a downloadable PDF threat report."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Title
    pdf.set_font('Arial', 'B', 20)
    pdf.set_text_color(136, 14, 79)
    pdf.cell(0, 10, 'PhishDetect AI - Analysis Report', 0, 1, 'C')
    pdf.ln(5)

    # Metadata
    pdf.set_font('Arial', '', 11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 7, f'Date: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)

    # Prediction
    pred_label = result.get('prediction', 'unknown')
    color_hex  = COLOR_MAP.get(pred_label, {}).get('hex', '#000000')
    pdf.set_font('Arial', 'B', 14)
    pdf.set_text_color(*hex_to_rgb(color_hex))
    pdf.cell(0, 8, f'Prediction: {pred_label.upper().replace("_", " ")}', 0, 1)

    pdf.set_text_color(0, 0, 0)
    pdf.set_font('Arial', '', 11)
    pdf.cell(0, 7, f'Confidence: {result["confidence"]:.2f}%', 0, 1)
    pdf.cell(0, 7, f'Risk Level: {result["risk_level"]}', 0, 1)
    pdf.ln(5)

    # LIME words
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 7, 'Top Influential Words:', 0, 1)

    word_weights = []
    if lime_result:
        word_weights = lime_result.get('word_weights', [])

    if word_weights:
        for word, weight in word_weights[:10]:
            sign = '+' if weight > 0 else '-'
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f'  {sign} {word} ({weight:.4f})', 0, 1)
    else:
        pdf.set_font('Arial', 'I', 10)
        pdf.cell(0, 6, '  No LIME data available.', 0, 1)

    # Footer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 5, 'Automated analysis by PhishDetect AI.', 0, 1, 'C')

    out_path = '/tmp/report.pdf'
    pdf.output(out_path)

    # Return bytes so Streamlit download_button works directly
    with open(out_path, 'rb') as f:
        return f.read()
