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
    _STEMMER = PorterStemmer()
    _NLTK_OK = True
except Exception:
    _STOP_WORDS = set()
    _STEMMER = None
    _NLTK_OK = False

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
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
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
    r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",  # raw IP
    r"bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|rb\.gy",       # url shorteners
    r"\.tk$|\.ml$|\.ga$|\.cf$|\.gq$",                     # free TLD abuse
    r"paypa1|arnazon|g00gle|micros0ft|app1e",              # typosquatting
    r"login|verify|secure|update|confirm|account|banking",# suspicious keywords in domain
    r"[a-z0-9]{20,}",                                      # very long random subdomains
]

SUSPICIOUS_TLD = {".tk", ".ml", ".ga", ".cf", ".gq", ".xyz", ".top", ".click", ".work"}

# ── Load model artifacts ───────────────────────────────────────────────────────
_MODEL = None
_VECTORIZER = None
_LIME_EXPLAINER = None
_LABEL_MAP = {0: "legitimate", 1: "traditional phishing", 2: "ai generated phishing"}

def _load_artifacts():
    global _MODEL, _VECTORIZER
    if _MODEL is None:
        try:
            _MODEL = joblib.load("model.pkl")
        except Exception as e:
            print(f"[PhishDetect] Warning: could not load model.pkl — {e}")
    if _VECTORIZER is None:
        try:
            _VECTORIZER = joblib.load("vectorizer.pkl")
        except Exception as e:
            print(f"[PhishDetect] Warning: could not load vectorizer.pkl — {e}")


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Full preprocessing pipeline: lowercase, HTML decode, URL/email removal, alpha-only."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Text extraction from uploaded files ───────────────────────────────────────
def extract_text_from_file(uploaded_file) -> str:
    """Extract plain text from .txt, .pdf, or .docx uploaded file objects."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".txt"):
            return uploaded_file.read().decode("utf-8", errors="replace")

        if name.endswith(".pdf"):
            if not _PYPDF2_OK:
                return ""
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            return "\n".join(
                (page.extract_text() or "") for page in reader.pages
            )

        if name.endswith(".docx"):
            if not _DOCX_OK:
                return ""
            doc = DocxDocument(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs)

    except Exception as e:
        print(f"[PhishDetect] extract_text_from_file error: {e}")
    return ""


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_email(email_text: str) -> dict:
    """
    Returns dict with keys:
      prediction (str), confidence (float 0-100), risk_level (str),
      probabilities (dict label->float)
    """
    _load_artifacts()

    cleaned = clean_text(email_text)

    # ── Fallback when model files are missing ──────────────────────────────────
    if _MODEL is None or _VECTORIZER is None:
        return {
            "prediction":    "unknown",
            "confidence":    0.0,
            "risk_level":    "Unknown",
            "probabilities": {"legitimate": 0.0, "traditional phishing": 0.0, "ai generated phishing": 0.0},
        }

    X = _VECTORIZER.transform([cleaned])

    raw_pred   = _MODEL.predict(X)[0]
    pred_label = _LABEL_MAP.get(int(raw_pred), str(raw_pred).lower())

    proba = _MODEL.predict_proba(X)[0]
    classes = _MODEL.classes_
    prob_dict = {
        _LABEL_MAP.get(int(c), str(c).lower()): float(p)
        for c, p in zip(classes, proba)
    }

    confidence = float(np.max(proba)) * 100

    # Risk level
    if "ai" in pred_label or "generated" in pred_label:
        risk = "Critical"
    elif "phishing" in pred_label:
        risk = "High"
    else:
        risk = "Low" if confidence >= 70 else "Medium"

    return {
        "prediction":    pred_label,
        "confidence":    confidence,
        "risk_level":    risk,
        "probabilities": prob_dict,
    }


# ── LIME explanation ───────────────────────────────────────────────────────────
def get_lime_explanation(email_text: str, num_features: int = 10) -> dict | None:
    """Return dict with 'positive' and 'negative' word lists, or None on failure."""
    if not _LIME_OK:
        return None

    _load_artifacts()
    if _MODEL is None or _VECTORIZER is None:
        return None

    global _LIME_EXPLAINER
    if _LIME_EXPLAINER is None:
        _LIME_EXPLAINER = LimeTextExplainer(
            class_names=[_LABEL_MAP[i] for i in sorted(_LABEL_MAP)],
            random_state=42,
        )

    cleaned = clean_text(email_text)

    def predict_fn(texts):
        X = _VECTORIZER.transform([clean_text(t) for t in texts])
        return _MODEL.predict_proba(X)

    try:
        exp = _LIME_EXPLAINER.explain_instance(
            cleaned,
            predict_fn,
            num_features=num_features,
            num_samples=300,
        )
        pred_idx = int(_MODEL.predict(_VECTORIZER.transform([cleaned]))[0])
        feature_weights = exp.as_list(label=pred_idx)

        positive = [w for w, s in feature_weights if s > 0]
        negative = [w for w, s in feature_weights if s < 0]
        return {"positive": positive, "negative": negative}

    except Exception as e:
        print(f"[PhishDetect] LIME error: {e}")
        return None


# ── URL checking ───────────────────────────────────────────────────────────────
def extract_urls(text: str) -> list[str]:
    """Extract all URLs from raw email text."""
    pattern = r"https?://[^\s<>\"'(){}\[\]]+"
    return re.findall(pattern, text)


def check_url(url: str) -> dict:
    """
    Lightweight URL safety check (no external API required).
    Returns dict with: url, risk (Low/Medium/High/Critical), flags (list), safe (bool).
    """
    flags = []
    risk  = "Low"

    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower()
        path   = parsed.path.lower()
        full   = url.lower()

        # Raw IP address
        if re.search(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", domain):
            flags.append("Uses raw IP address instead of domain")
            risk = "Critical"

        # Suspicious TLD
        for tld in SUSPICIOUS_TLD:
            if domain.endswith(tld):
                flags.append(f"Suspicious top-level domain ({tld})")
                risk = _escalate(risk, "High")

        # URL shortener
        if re.search(r"bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|rb\.gy|is\.gd|short\.link", domain):
            flags.append("URL shortener detected (destination hidden)")
            risk = _escalate(risk, "High")

        # Typosquatting common brands
        typos = {
            "paypa1": "PayPal", "arnazon": "Amazon", "g00gle": "Google",
            "micros0ft": "Microsoft", "app1e": "Apple", "faceb00k": "Facebook",
        }
        for typo, brand in typos.items():
            if typo in domain:
                flags.append(f"Possible typosquatting of {brand}")
                risk = _escalate(risk, "Critical")

        # Excessive subdomains
        parts = domain.split(".")
        if len(parts) > 4:
            flags.append(f"Excessive subdomains ({len(parts)-2} levels)")
            risk = _escalate(risk, "Medium")

        # Suspicious keywords in domain
        suspicious_kw = ["login", "verify", "secure", "update", "confirm",
                         "account", "banking", "password", "credential", "signin"]
        for kw in suspicious_kw:
            if kw in domain:
                flags.append(f"Suspicious keyword in domain: '{kw}'")
                risk = _escalate(risk, "High")
                break

        # Suspicious path keywords
        path_kw = ["phishing", "hack", "steal", "malware", "exec", "shell", "exploit"]
        for kw in path_kw:
            if kw in path:
                flags.append(f"Suspicious path keyword: '{kw}'")
                risk = _escalate(risk, "High")
                break

        # Very long URL
        if len(url) > 200:
            flags.append("Unusually long URL (>200 characters)")
            risk = _escalate(risk, "Medium")

        # HTTP (not HTTPS)
        if parsed.scheme == "http":
            flags.append("Uses unencrypted HTTP (not HTTPS)")
            risk = _escalate(risk, "Medium")

        # @ symbol in URL (credential stuffing trick)
        if "@" in full:
            flags.append("'@' symbol found in URL (credential-hiding trick)")
            risk = _escalate(risk, "High")

        # Double extension / executable
        if re.search(r"\.(exe|bat|cmd|ps1|vbs|js|jar|zip|rar)\b", path):
            flags.append("URL points to an executable or archive file")
            risk = _escalate(risk, "Critical")

    except Exception as e:
        flags.append(f"Could not parse URL: {e}")
        risk = "Medium"

    safe = risk == "Low"
    return {"url": url, "risk": risk, "flags": flags, "safe": safe}


def _escalate(current: str, new: str) -> str:
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    return new if order.get(new, 0) > order.get(current, 0) else current


def check_urls_in_email(email_text: str) -> list[dict]:
    """Extract and check all URLs found in the email text."""
    urls = extract_urls(email_text)
    if not urls:
        return []
    return [check_url(u) for u in urls]


# ── PDF report generation ──────────────────────────────────────────────────────
def generate_pdf_report(email_text: str, result: dict, lime_result: dict | None) -> bytes | None:
    """
    Generate a PDF threat report using ReportLab.
    Returns raw bytes on success, None on failure.
    Robust against missing/malformed result keys.
    """
    if not _REPORTLAB_OK:
        return None

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2*cm, leftMargin=2*cm,
            topMargin=2*cm,   bottomMargin=2*cm,
        )

        # ── Styles ─────────────────────────────────────────────────────────────
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "Title", parent=styles["Heading1"],
            fontSize=18, textColor=colors.HexColor("#1D4ED8"),
            spaceAfter=6, alignment=TA_LEFT,
        )
        subtitle_style = ParagraphStyle(
            "Subtitle", parent=styles["Normal"],
            fontSize=10, textColor=colors.HexColor("#64748B"),
            spaceAfter=12,
        )
        section_style = ParagraphStyle(
            "Section", parent=styles["Heading2"],
            fontSize=12, textColor=colors.HexColor("#1E293B"),
            spaceBefore=14, spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body", parent=styles["Normal"],
            fontSize=9, textColor=colors.HexColor("#1E293B"),
            spaceAfter=4, leading=14,
        )
        mono_style = ParagraphStyle(
            "Mono", parent=styles["Code"],
            fontSize=8, textColor=colors.HexColor("#334155"),
            backColor=colors.HexColor("#F1F5F9"),
            spaceAfter=4, leading=12,
        )

        story = []

        # ── Header ─────────────────────────────────────────────────────────────
        story.append(Paragraph("🛡️ PhishDetect AI — Threat Analysis Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#3B82F6")))
        story.append(Spacer(1, 0.3*cm))

        # ── Safely extract result fields ────────────────────────────────────────
        pred_label  = str(result.get("prediction", "unknown")).title()
        confidence  = result.get("confidence", 0)
        if isinstance(confidence, (int, float)):
            conf_pct = int(confidence) if confidence > 1 else int(confidence * 100)
        else:
            conf_pct = 0
        risk_level  = str(result.get("risk_level", "Unknown"))

        # ── Verdict section ────────────────────────────────────────────────────
        story.append(Paragraph("Verdict Summary", section_style))

        verdict_color = colors.HexColor("#10B981")   # green = legit
        if "phishing" in pred_label.lower():
            if "ai" in pred_label.lower() or "generated" in pred_label.lower():
                verdict_color = colors.HexColor("#F59E0B")   # amber = AI phish
            else:
                verdict_color = colors.HexColor("#EF4444")   # red = traditional phish

        verdict_data = [
            ["Classification", pred_label],
            ["Confidence",     f"{conf_pct}%"],
            ["Risk Level",     risk_level],
            ["Timestamp",      datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]
        verdict_table = Table(verdict_data, colWidths=[4*cm, 13*cm])
        verdict_table.setStyle(TableStyle([
            ("BACKGROUND",  (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR",   (0, 0), (0, -1), colors.HexColor("#64748B")),
            ("TEXTCOLOR",   (1, 0), (1, 0),  verdict_color),
            ("FONTNAME",    (1, 0), (1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 0.3*cm))

        # ── Class probabilities ────────────────────────────────────────────────
        probs = result.get("probabilities", {})
        if probs:
            story.append(Paragraph("Class Probabilities", section_style))
            prob_data = [["Class", "Probability"]]
            for label, p in probs.items():
                pct = f"{float(p)*100:.1f}%" if float(p) <= 1 else f"{float(p):.1f}%"
                prob_data.append([str(label).title(), pct])
            prob_table = Table(prob_data, colWidths=[10*cm, 7*cm])
            prob_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0, 0),(-1, -1), 5),
            ]))
            story.append(prob_table)
            story.append(Spacer(1, 0.3*cm))

        # ── LIME explanation ───────────────────────────────────────────────────
        if lime_result:
            pos = lime_result.get("positive", [])
            neg = lime_result.get("negative", [])
            if pos or neg:
                story.append(Paragraph("Key Influential Words (LIME)", section_style))
                if pos:
                    story.append(Paragraph(
                        "<b>Phishing indicators (+):</b> " + ", ".join(str(w) for w in pos[:10]),
                        body_style,
                    ))
                if neg:
                    story.append(Paragraph(
                        "<b>Legitimacy indicators (−):</b> " + ", ".join(str(w) for w in neg[:10]),
                        body_style,
                    ))
                story.append(Spacer(1, 0.3*cm))

        # ── URL analysis ───────────────────────────────────────────────────────
        url_results = check_urls_in_email(email_text)
        if url_results:
            story.append(Paragraph("URL Analysis", section_style))
            url_data = [["URL", "Risk", "Flags"]]
            for ur in url_results[:10]:   # cap at 10 URLs in report
                short_url = ur["url"][:60] + ("…" if len(ur["url"]) > 60 else "")
                flags_str  = "; ".join(ur.get("flags", [])) or "None"
                url_data.append([short_url, ur["risk"], flags_str])
            url_table = Table(url_data, colWidths=[7*cm, 2.5*cm, 7.5*cm])
            url_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1D4ED8")),
                ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
                ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING",(0, 0),(-1, -1), 4),
                ("WORDWRAP",   (0, 0), (-1, -1), True),
            ]))
            story.append(url_table)
            story.append(Spacer(1, 0.3*cm))

        # ── Email preview ──────────────────────────────────────────────────────
        story.append(Paragraph("Email Content Preview (first 800 chars)", section_style))
        preview = email_text[:800].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        story.append(Paragraph(preview, mono_style))
        story.append(Spacer(1, 0.3*cm))

        # ── Footer ─────────────────────────────────────────────────────────────
        story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E2E8F0")))
        story.append(Paragraph(
            "PhishDetect AI — For educational and research purposes only. "
            "Built with Streamlit, scikit-learn, LIME, and ReportLab.",
            ParagraphStyle("Footer", parent=styles["Normal"],
                           fontSize=7, textColor=colors.HexColor("#94A3B8"),
                           alignment=TA_CENTER, spaceBefore=6),
        ))

        doc.build(story)
        return buffer.getvalue()

    except Exception as e:
        print(f"[PhishDetect] generate_pdf_report error: {e}")
        import traceback; traceback.print_exc()
        return None
