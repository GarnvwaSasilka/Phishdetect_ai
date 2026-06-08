"""
phishdetect_util.py
Core utility module for PhishDetect AI.
Handles: text extraction, preprocessing, prediction, LIME, PDF generation, URL checking.
"""

import re
import html
import io
import os
import urllib.parse
import urllib.request
import socket
from datetime import datetime

import joblib
import numpy as np

# ── Optional / lazy imports ────────────────────────────────────────────────────
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))
    _STEMMER    = PorterStemmer()
    _NLTK_OK    = True
except Exception:
    _STOP_WORDS = set()
    _STEMMER    = None
    _NLTK_OK    = False

try:
    from lime.lime_text import LimeTextExplainer
    _LIME_OK = True
except Exception:
    _LIME_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer,
        Table, TableStyle, HRFlowable,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    _REPORTLAB_OK = True
except Exception:
    _REPORTLAB_OK = False

try:
    import PyPDF2
    _PYPDF2_OK = True
except Exception:
    _PYPDF2_OK = False

try:
    from docx import Document as DocxDocument
    _DOCX_OK = True
except Exception:
    _DOCX_OK = False

# ── Color map ──────────────────────────────────────────────────────────────────
COLOR_MAP = {
    "legitimate":            "#34D399",
    "traditional phishing":  "#60A5FA",
    "ai generated phishing": "#FCD34D",
    "unknown":               "#94A3B8",
}

# ── Suspicious domain patterns ─────────────────────────────────────────────────
SUSPICIOUS_PATTERNS = [
    r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",
    r"bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|rb\.gy",
    r"\.tk$|\.ml$|\.ga$|\.cf$|\.gq$",
    r"paypa1|arnazon|g00gle|micros0ft|app1e",
    r"login|verify|secure|update|confirm|account|banking",
    r"[a-z0-9]{20,}",
]

SUSPICIOUS_TLD = {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click", ".work"}

# ── Label map ──────────────────────────────────────────────────────────────────
_LABEL_MAP = {0: "legitimate", 1: "traditional phishing", 2: "ai generated phishing"}

# ── Model state ───────────────────────────────────────────────────────────────
_MODEL      = None
_VECTORIZER = None
_ARTIFACTS_LOADED = False
_LIME_EXPLAINER   = None


# ── Artifact loading ───────────────────────────────────────────────────────────
def _load_artifacts():
    global _MODEL, _VECTORIZER, _ARTIFACTS_LOADED
    if _ARTIFACTS_LOADED:
        return
    _ARTIFACTS_LOADED = True
    base_dirs = [os.path.dirname(os.path.abspath(__file__)), os.getcwd()]
    def _find(filename):
        for d in base_dirs:
            p = os.path.join(d, filename)
            if os.path.isfile(p):
                return p
        return None
    model_path      = _find("model.pkl")
    vectorizer_path = _find("vectorizer.pkl")
    if model_path is None:
        print("[PhishDetect] WARNING: model.pkl not found in search path.")
    else:
        try:
            _MODEL = joblib.load(model_path)
            print(f"[PhishDetect] Loaded model from {model_path}  "
                  f"(expects {_MODEL.n_features_in_} features)")
        except Exception as e:
            print(f"[PhishDetect] ERROR loading model.pkl: {e}")
    if vectorizer_path is None:
        print("[PhishDetect] WARNING: vectorizer.pkl not found in search path.")
    else:
        try:
            _VECTORIZER = joblib.load(vectorizer_path)
            vocab_size = len(_VECTORIZER.vocabulary_)
            print(f"[PhishDetect] Loaded vectorizer from {vectorizer_path}  "
                  f"(vocab size {vocab_size})")
        except Exception as e:
            print(f"[PhishDetect] ERROR loading vectorizer.pkl: {e}")
    if _MODEL is not None and _VECTORIZER is not None:
        vocab_size = len(_VECTORIZER.vocabulary_)
        model_features = getattr(_MODEL, "n_features_in_", None)
        if model_features is not None and vocab_size != model_features:
            print(
                f"[PhishDetect] CRITICAL MISMATCH: vectorizer vocab={vocab_size} "
                f"but model expects {model_features} features. "
                "Ensure model.pkl and vectorizer.pkl are from the same training run."
            )
            _MODEL      = None
            _VECTORIZER = None


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S
