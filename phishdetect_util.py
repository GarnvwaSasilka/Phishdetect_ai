"""
phishdetect_util.py
Core utility module for PhishDetect AI.
Handles: text extraction, preprocessing, prediction, LIME, PDF generation, URL checking.
"""

import re
import html
import io
import os
import json
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
    "legitimate":            "#0DAF80",  # teal-green — unambiguous safe
    "traditional phishing":  "#D97706",  # amber — alarm, not "calm blue"
    "ai generated phishing": "#8B5CF6",  # violet — novel threat class
    "unknown":               "#3A5270",
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

# ── Brand list — loaded from brands.json ──────────────────────────────────────
# What this is:  a brand-impersonation detector, not a whitelist.
# For each brand keyword (lowercase), it lists the ONLY domains that are legitimate
# senders for that brand. If an email mentions "paypal" but comes from a different
# domain, that is a mismatch signal. If it comes from paypal.com, it is a trust signal.
# Edit brands.json to add or remove entries — do not hardcode here.
def _load_brand_domains() -> dict[str, list[str]]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "brands.json")
    if not os.path.isfile(path):
        print("[PhishDetect] WARNING: brands.json not found — brand signals disabled.")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        # brands.json is grouped by category; flatten into a single brand→domains dict
        flat: dict[str, list[str]] = {}
        for section in raw.values():
            if isinstance(section, dict):
                for brand, domains in section.items():
                    flat[brand.lower()] = [d.lower() for d in domains]
        return flat
    except Exception as e:
        print(f"[PhishDetect] ERROR loading brands.json: {e}")
        return {}

_KNOWN_BRAND_DOMAINS: dict[str, list[str]] = _load_brand_domains()

# Flat set of ALL trusted domains — used to decide if a sender is itself a known brand
# (in which case we suppress mismatch signals for other brands it happens to link to)
_ALL_TRUSTED_DOMAINS: set[str] = {
    d for domains in _KNOWN_BRAND_DOMAINS.values() for d in domains
}

# Well-known safe domains — URL checker returns Low risk immediately for these
_SAFE_URL_DOMAINS: set[str] = {
    "github.com", "gitlab.com", "stackoverflow.com", "wikipedia.org",
    "google.com", "gmail.com", "youtube.com", "slack.com", "notion.so",
    "microsoft.com", "office.com", "outlook.com", "live.com",
    "apple.com", "icloud.com", "amazon.com", "aws.amazon.com",
    "linkedin.com", "twitter.com", "x.com", "facebook.com", "instagram.com",
    "zoom.us", "atlassian.com", "jira.atlassian.com", "trello.com",
    "dropbox.com", "figma.com", "vercel.com", "netlify.com",
    "stripe.com", "paypal.com", "paystack.com", "flutterwave.com",
    "gtbank.com", "zenithbank.com", "accessbankplc.com", "ubagroup.com",
    "kuda.com", "moniepoint.com", "piggyvest.com",
}

# ── Model state ───────────────────────────────────────────────────────────────
_MODEL      = None
_VECTORIZER = None
_SVM_MODEL  = None
_ARTIFACTS_LOADED = False
_LIME_EXPLAINER   = None


# ── Artifact loading ───────────────────────────────────────────────────────────
def _load_artifacts():
    global _MODEL, _VECTORIZER, _SVM_MODEL, _ARTIFACTS_LOADED
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
    svm_path        = _find("svm_final.pkl")
    if model_path is None:
        print("[PhishDetect] WARNING: model.pkl not found in search path.")
    else:
        try:
            _MODEL = joblib.load(model_path)
            print(f"[PhishDetect] Loaded LR model from {model_path}  "
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
    if svm_path is None:
        print("[PhishDetect] INFO: svm_final.pkl not found — using LR only.")
    else:
        try:
            _SVM_MODEL = joblib.load(svm_path)
            svm_feat = getattr(_SVM_MODEL, "n_features_in_", None)
            model_feat = getattr(_MODEL, "n_features_in_", None)
            if svm_feat and model_feat and svm_feat != model_feat:
                print(f"[PhishDetect] WARNING: SVM feature count ({svm_feat}) != "
                      f"LR feature count ({model_feat}). Disabling SVM.")
                _SVM_MODEL = None
            else:
                print(f"[PhishDetect] Loaded SVM from {svm_path} — ensemble active.")
        except Exception as e:
            print(f"[PhishDetect] ERROR loading svm_final.pkl: {e}")


# ── Text cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " url_link ", text)
    text = re.sub(r"\S+@\S+\.\S+", " email_addr ", text)
    text = re.sub(r"\b\d[\d,.]*\b", " number ", text)
    text = re.sub(r"[^a-z\s_]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Text extraction from uploaded files ───────────────────────────────────────
def extract_text_from_file(uploaded_file) -> str:
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


# ── Multi-signal helpers ───────────────────────────────────────────────────────
def _sender_domain(email_text: str) -> str | None:
    """Extract the domain portion of the From: header."""
    m = re.search(r"^From:[^\n]*@([\w.\-]+)", email_text, re.IGNORECASE | re.MULTILINE)
    return m.group(1).lower() if m else None


def _compute_multi_signals(email_text: str) -> list[dict]:
    """
    Return a list of dicts: {weight, label, direction}
    weight > 0  → phishing evidence
    weight < 0  → legitimate evidence
    Signals cover sender-brand alignment, auth failures, URL risk, and template artefacts.
    """
    signals: list[dict] = []
    body_lower = email_text.lower()

    # ── Signal 1: Unfilled template variables ─────────────────────────────────
    # Bulk-mail blasts that forgot to render variables are a clear phishing marker.
    # Weight raised to 0.40 because clean_text() strips both URLs and special chars,
    # meaning the classifier never sees template artefacts — only this signal catches them.
    if re.search(
        r"\{\{[A-Za-z_]+\}\}|\{[A-Za-z_]+\}|\[(?:F?NAME|LAST_?NAME|EMAIL|RECIPIENT|FIRST)\]|%[A-Z_]+%",
        email_text,
    ):
        signals.append({
            "weight": 0.40,
            "label":  "Unfilled template variable found (unrendered bulk mail)",
            "direction": "phishing",
        })
    # Template variable embedded directly inside a URL is even stronger evidence
    # (the URL itself was never rendered — this is almost never in a legitimate email).
    if re.search(r"https?://[^\s]*\{\{[A-Za-z_]+\}\}", email_text):
        signals.append({
            "weight": 0.25,
            "label":  "URL contains an unfilled template slot (e.g. ?email={{FEmail}})",
            "direction": "phishing",
        })

    # ── Signal 2: Sender-brand alignment ──────────────────────────────────────
    # Only scan the *display text* for brand mentions — strip URLs first so that
    # a Slack notification linking to github.com doesn't get flagged as
    # "claims to be GitHub". Also: if the sender is itself a known trusted brand,
    # suppress all mismatch signals (a verified Slack email can link to anything).
    from_domain = _sender_domain(email_text)
    if from_domain:
        display_text = re.sub(r"https?://\S+", " ", body_lower)  # remove URLs
        for brand, trusted in _KNOWN_BRAND_DOMAINS.items():
            if brand in display_text:
                is_trusted = any(
                    from_domain == td or from_domain.endswith("." + td)
                    for td in trusted
                )
                if is_trusted:
                    # Known sender matches known brand → strong legitimacy signal
                    signals.append({
                        "weight": -0.70,
                        "label":  f"Sender domain '{from_domain}' matches trusted brand '{brand}'",
                        "direction": "legitimate",
                    })
                    break
                # If sender is NOT recognised: we simply don't know — emit no signal.
                # "Not on our list" is not evidence of phishing; it just means we lack data.
                # Firing a phishing signal here would penalise every small/regional business
                # that will never appear in brands.json.

    # ── Signal 3: URL risk ────────────────────────────────────────────────────
    # Weights are deliberately conservative — URL heuristics are noisy and will
    # catch legitimate business domains with hyphens, subdomains, or action-words
    # in their paths. Only truly egregious URLs (IP addresses, malicious TLDs,
    # executable downloads) should materially shift the verdict.
    url_results = check_urls_in_email(email_text)
    if url_results:
        risk_weight = {"Low": 0, "Medium": 0.05, "High": 0.10, "Critical": 0.20}
        max_risk    = max(url_results, key=lambda ur: risk_weight.get(ur["risk"], 0))["risk"]
        w = risk_weight.get(max_risk, 0)
        if w > 0:
            signals.append({
                "weight": w,
                "label":  f"Highest URL risk in email: {max_risk}",
                "direction": "phishing",
            })

    # ── Signal 4: Email authentication failures ───────────────────────────────
    auth = re.search(r"^Authentication-Results:\s*(.+)", email_text,
                     re.IGNORECASE | re.MULTILINE)
    auth_str = auth.group(1).lower() if auth else ""
    if "spf=fail"   in auth_str:
        signals.append({"weight": 0.20, "label": "SPF check failed",  "direction": "phishing"})
    if "dkim=fail"  in auth_str:
        signals.append({"weight": 0.20, "label": "DKIM signature failed", "direction": "phishing"})
    if "dmarc=fail" in auth_str:
        signals.append({"weight": 0.20, "label": "DMARC policy failed", "direction": "phishing"})

    # ── Signal 5: Reply-To / From domain mismatch ─────────────────────────────
    from_hdr     = re.search(r"^From:\s*(.+)",     email_text, re.I | re.M)
    reply_to_hdr = re.search(r"^Reply-To:\s*(.+)", email_text, re.I | re.M)
    def _dom(addr: str) -> str:
        m = re.search(r"@([\w.\-]+)", addr)
        return m.group(1).lower() if m else ""
    if from_hdr and reply_to_hdr:
        fd = _dom(from_hdr.group(1))
        rd = _dom(reply_to_hdr.group(1))
        if fd and rd and fd != rd:
            signals.append({
                "weight": 0.30,
                "label":  f"From/Reply-To domain mismatch ({fd} vs {rd})",
                "direction": "phishing",
            })

    return signals


def _apply_signals(ensemble_proba: np.ndarray, signals: list[dict]) -> np.ndarray:
    """
    Interpolate ensemble_proba toward an oracle distribution using the net
    signal weight. Positive net → blend toward phishing oracle [0,1,0].
    Negative net → blend toward legitimate oracle [1,0,0].
    Caps at 80% blend so the text model always contributes at least 20%.
    """
    if not signals:
        return ensemble_proba

    net = sum(s["weight"] for s in signals)
    if net == 0:
        return ensemble_proba

    if net > 0:
        oracle = np.array([0.0, 1.0, 0.0])
        blend  = min(net, 0.75)
    else:
        oracle = np.array([1.0, 0.0, 0.0])
        blend  = min(abs(net), 0.80)

    adjusted = (1.0 - blend) * ensemble_proba + blend * oracle
    adjusted = np.clip(adjusted, 0, 1)
    total    = adjusted.sum()
    return adjusted / total if total > 0 else adjusted


# ── Prediction ─────────────────────────────────────────────────────────────────
def predict_email(email_text: str) -> dict:
    _load_artifacts()
    _empty = {
        "prediction":    "unknown",
        "confidence":    0.0,
        "risk_level":    "Unknown",
        "probabilities": {
            "legitimate":            0.0,
            "traditional phishing":  0.0,
            "ai generated phishing": 0.0,
        },
        "signals":      [],
        "used_ensemble": False,
    }
    if _MODEL is None or _VECTORIZER is None:
        return _empty
    cleaned = clean_text(email_text)
    try:
        X = _VECTORIZER.transform([cleaned])

        # LR+SVM soft-vote ensemble (50/50) when SVM is available
        lr_proba = _MODEL.predict_proba(X)[0]
        if _SVM_MODEL is not None:
            svm_proba      = _SVM_MODEL.predict_proba(X)[0]
            ensemble_proba = (lr_proba * 0.5) + (svm_proba * 0.5)
            used_ensemble  = True
        else:
            ensemble_proba = lr_proba
            used_ensemble  = False

        # Multi-signal fusion: sender-brand, URL risk, auth failures, templates
        signals        = _compute_multi_signals(email_text)
        adjusted_proba = _apply_signals(ensemble_proba, signals)

        pred_idx   = int(np.argmax(adjusted_proba))
        pred_label = _LABEL_MAP.get(pred_idx, "unknown")
        classes    = _MODEL.classes_
        prob_dict  = {
            _LABEL_MAP.get(int(c), str(c).lower()): float(p)
            for c, p in zip(classes, adjusted_proba)
        }
        confidence = float(np.max(adjusted_proba)) * 100

        # Risk level must be supported by evidence, not just ML confidence alone.
        # A borderline phishing call (50–79%) with no corroborating signals is
        # "Medium" — still a warning, but not "High" which implies strong confidence.
        net_phish_signals = sum(s["weight"] for s in signals if s["weight"] > 0)
        if "ai" in pred_label or "generated" in pred_label:
            risk = "Critical"
        elif "phishing" in pred_label:
            if confidence >= 80 or net_phish_signals >= 0.25:
                risk = "High"
            else:
                risk = "Medium"   # ML suspects phishing but no hard evidence to back it
        else:
            risk = "Low" if confidence >= 70 else "Medium"

        return {
            "prediction":    pred_label,
            "confidence":    confidence,
            "risk_level":    risk,
            "probabilities": prob_dict,
            "signals":       signals,
            "used_ensemble": used_ensemble,
        }
    except Exception as e:
        print(f"[PhishDetect] predict_email error: {e}")
        return _empty


# ── LIME explanation ───────────────────────────────────────────────────────────
def get_lime_explanation(email_text: str, num_features: int = 10) -> dict | None:
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
        pred_idx        = int(_MODEL.predict(_VECTORIZER.transform([cleaned]))[0])
        feature_weights = exp.as_list(label=pred_idx)
        positive = [w for w, s in feature_weights if s > 0]
        negative = [w for w, s in feature_weights if s < 0]
        return {"positive": positive, "negative": negative}
    except Exception as e:
        print(f"[PhishDetect] LIME error: {e}")
        return None


# ── URL helpers ────────────────────────────────────────────────────────────────
def _escalate(current: str, new: str) -> str:
    order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    return new if order.get(new, 0) > order.get(current, 0) else current


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s<>\"'(){}\[\]]+", text)


def check_url(url: str) -> dict:
    flags = []
    risk  = "Low"
    try:
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc.lower().lstrip("www.")
        path   = parsed.path.lower()
        full   = url.lower()

        # Fast-exit for well-known legitimate domains — no heuristics needed
        if any(domain == sd or domain.endswith("." + sd) for sd in _SAFE_URL_DOMAINS):
            return {"url": url, "risk": "Low", "flags": [], "safe": True}
        if re.search(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$", domain):
            flags.append("Uses raw IP address instead of domain")
            risk = "Critical"
        for tld in SUSPICIOUS_TLD:
            if domain.endswith(tld):
                flags.append(f"Suspicious top-level domain ({tld})")
                risk = _escalate(risk, "High")
        if re.search(r"bit\.ly|tinyurl|t\.co|goo\.gl|ow\.ly|rb\.gy|is\.gd|short\.link", domain):
            flags.append("URL shortener detected (destination hidden)")
            risk = _escalate(risk, "High")
        typos = {
            "paypa1": "PayPal", "arnazon": "Amazon", "g00gle": "Google",
            "micros0ft": "Microsoft", "app1e": "Apple", "faceb00k": "Facebook",
        }
        for typo, brand in typos.items():
            if typo in domain:
                flags.append(f"Possible typosquatting of {brand}")
                risk = _escalate(risk, "Critical")
        parts = domain.split(".")
        if len(parts) > 4:
            flags.append(f"Excessive subdomains ({len(parts) - 2} levels)")
            risk = _escalate(risk, "Medium")
        suspicious_kw = [
            "login", "verify", "secure", "update", "confirm",
            "account", "banking", "password", "credential", "signin",
        ]
        for kw in suspicious_kw:
            if kw in domain:
                flags.append(f"Suspicious keyword in domain: '{kw}'")
                risk = _escalate(risk, "High")
                break
        path_kw = ["phishing", "hack", "steal", "malware", "exec", "shell", "exploit"]
        for kw in path_kw:
            if kw in path:
                flags.append(f"Suspicious path keyword: '{kw}'")
                risk = _escalate(risk, "High")
                break
        if len(url) > 200:
            flags.append("Unusually long URL (>200 characters)")
            risk = _escalate(risk, "Medium")
        if parsed.scheme == "http":
            flags.append("Uses unencrypted HTTP (not HTTPS)")
            risk = _escalate(risk, "Medium")
        if "@" in full:
            flags.append("'@' symbol in URL (credential-hiding trick)")
            risk = _escalate(risk, "High")
        if re.search(r"\.(exe|bat|cmd|ps1|vbs|js|jar|zip|rar)\b", path):
            flags.append("URL points to an executable or archive file")
            risk = _escalate(risk, "Critical")
    except Exception as e:
        flags.append(f"Could not parse URL: {e}")
        risk = "Medium"
    return {"url": url, "risk": risk, "flags": flags, "safe": risk == "Low"}


def check_urls_in_email(email_text: str) -> list[dict]:
    urls = extract_urls(email_text)
    return [check_url(u) for u in urls] if urls else []


# ── PDF report generation ──────────────────────────────────────────────────────
def generate_pdf_report(
    email_text: str,
    result: dict,
    lime_result: dict | None,
) -> bytes | None:
    if not _REPORTLAB_OK:
        return None
    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=2 * cm, leftMargin=2 * cm,
            topMargin=2 * cm,   bottomMargin=2 * cm,
        )
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
        story.append(Paragraph("PhishDetect AI — Threat Analysis Report", title_style))
        story.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            subtitle_style,
        ))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=colors.HexColor("#3B82F6")))
        story.append(Spacer(1, 0.3 * cm))
        pred_label = str(result.get("prediction", "unknown")).title()
        confidence = result.get("confidence", 0)
        conf_pct   = int(confidence) if float(confidence) > 1 else int(float(confidence) * 100)
        risk_level = str(result.get("risk_level", "Unknown"))
        story.append(Paragraph("Verdict Summary", section_style))
        if "ai" in pred_label.lower() or "generated" in pred_label.lower():
            verdict_color = colors.HexColor("#F59E0B")
        elif "phishing" in pred_label.lower():
            verdict_color = colors.HexColor("#EF4444")
        else:
            verdict_color = colors.HexColor("#10B981")
        verdict_data = [
            ["Classification", pred_label],
            ["Confidence",     f"{conf_pct}%"],
            ["Risk Level",     risk_level],
            ["Timestamp",      datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ]
        verdict_table = Table(verdict_data, colWidths=[4 * cm, 13 * cm])
        verdict_table.setStyle(TableStyle([
            ("BACKGROUND",   (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
            ("TEXTCOLOR",    (0, 0), (0, -1), colors.HexColor("#64748B")),
            ("TEXTCOLOR",    (1, 0), (1,  0), verdict_color),
            ("FONTNAME",     (1, 0), (1,  0), "Helvetica-Bold"),
            ("FONTSIZE",     (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1),
             [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID",         (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
            ("LEFTPADDING",  (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING",   (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 6),
        ]))
        story.append(verdict_table)
        story.append(Spacer(1, 0.3 * cm))
        probs = result.get("probabilities", {})
        if probs:
            story.append(Paragraph("Class Probabilities", section_style))
            prob_data = [["Class", "Probability"]]
            for label, p in probs.items():
                pct = f"{float(p) * 100:.1f}%" if float(p) <= 1 else f"{float(p):.1f}%"
                prob_data.append([str(label).title(), pct])
            prob_table = Table(prob_data, colWidths=[10 * cm, 7 * cm])
            prob_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1,  0), colors.HexColor("#1D4ED8")),
                ("TEXTCOLOR",  (0, 0), (-1,  0), colors.white),
                ("FONTNAME",   (0, 0), (-1,  0), "Helvetica-Bold"),
                ("FONTSIZE",   (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID",       (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING",  (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]))
            story.append(prob_table)
            story.append(Spacer(1, 0.3 * cm))
        if lime_result:
            pos = lime_result.get("positive", [])
            neg = lime_result.get("negative", [])
            if pos or neg:
                story.append(Paragraph("Key Influential Words (LIME)", section_style))
                if pos:
                    story.append(Paragraph(
                        "<b>Phishing indicators (+):</b> " +
                        ", ".join(str(w) for w in pos[:10]),
                        body_style,
                    ))
                if neg:
                    story.append(Paragraph(
                        "<b>Legitimacy indicators (-):</b> " +
                        ", ".join(str(w) for w in neg[:10]),
                        body_style,
                    ))
                story.append(Spacer(1, 0.3 * cm))
        url_results = check_urls_in_email(email_text)
        if url_results:
            story.append(Paragraph("URL Analysis", section_style))
            url_data = [["URL", "Risk", "Flags"]]
            for ur in url_results[:10]:
                short_url = ur["url"][:60] + ("…" if len(ur["url"]) > 60 else "")
                flags_str = "; ".join(ur.get("flags", [])) or "None"
                url_data.append([short_url, ur["risk"], flags_str])
            url_table = Table(url_data, colWidths=[7 * cm, 2.5 * cm, 7.5 * cm])
            url_table.setStyle(TableStyle([
                ("BACKGROUND",  (0, 0), (-1,  0), colors.HexColor("#1D4ED8")),
                ("TEXTCOLOR",   (0, 0), (-1,  0), colors.white),
                ("FONTNAME",    (0, 0), (-1,  0), "Helvetica-Bold"),
                ("FONTSIZE",    (0, 0), (-1, -1), 7),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F8FAFC")]),
                ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",  (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("WORDWRAP",    (0, 0), (-1, -1), True),
            ]))
            story.append(url_table)
            story.append(Spacer(1, 0.3 * cm))
        story.append(Paragraph("Email Content Preview (first 800 chars)", section_style))
        preview = (
            email_text[:800]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        story.append(Paragraph(preview, mono_style))
        story.append(Spacer(1, 0.3 * cm))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#E2E8F0")))
        story.append(Paragraph(
            "PhishDetect AI — For security research and awareness purposes only.",
            subtitle_style,
        ))
        doc.build(story)
        return buffer.getvalue()
    except Exception as e:
        print(f"[PhishDetect] generate_pdf_report error: {e}")
        return None
