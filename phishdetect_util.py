"""
phishdetect_util.py - Utility functions for phishing detection
"""

import re
import pickle
import os
import json
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Debug flag
DEBUG = True

def debug_print(msg):
    """Print debug messages"""
    if DEBUG:
        print(f"[DEBUG] {msg}")

def load_model(model_path='model.pkl'):
    """Load the phishing detection model"""
    debug_print(f"Loading model from: {model_path}")
    
    # Try multiple paths
    paths_to_try = [
        model_path,
        os.path.join(os.getcwd(), model_path),
        os.path.join('/mount/src/phishdetect_ai', model_path),
        './' + model_path,
        '../' + model_path
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                debug_print(f"✅ Model loaded from {path}")
                return model
            except Exception as e:
                debug_print(f"Error loading from {path}: {e}")
                continue
    
    debug_print("⚠️ Model not found, using rule-based detection")
    return None

def load_vectorizer(vectorizer_path='vectorizer.pkl'):
    """Load the feature vectorizer"""
    debug_print(f"Loading vectorizer from: {vectorizer_path}")
    
    paths_to_try = [
        vectorizer_path,
        os.path.join(os.getcwd(), vectorizer_path),
        os.path.join('/mount/src/phishdetect_ai', vectorizer_path),
        './' + vectorizer_path,
        '../' + vectorizer_path
    ]
    
    for path in paths_to_try:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    vectorizer = pickle.load(f)
                debug_print(f"✅ Vectorizer loaded from {path}")
                return vectorizer
            except Exception as e:
                debug_print(f"Error loading from {path}: {e}")
                continue
    
    debug_print("⚠️ Vectorizer not found")
    return None

def extract_features(url: str) -> Dict[str, Any]:
    """Extract features from URL for phishing detection"""
    debug_print(f"Extracting features from: {url}")
    
    features = {}
    
    try:
        parsed = urlparse(url)
        
        # Basic characteristics
        features['url_length'] = len(url)
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_questions'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        
        # Security indicators
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        
        # Domain analysis
        domain = parsed.netloc
        features['domain_length'] = len(domain)
        features['num_subdomains'] = domain.count('.')
        features['has_port'] = 1 if ':' in domain else 0
        
        # Suspicious keywords
        suspicious_keywords = [
            'login', 'signin', 'verify', 'secure', 'account',
            'banking', 'update', 'confirm', 'paypal', 'apple',
            'microsoft', 'amazon', 'gmail', 'facebook'
        ]
        
        url_lower = url.lower()
        features['suspicious_keyword_count'] = sum(1 for kw in suspicious_keywords if kw in url_lower)
        
        # Brand spoofing
        brand_keywords = ['paypal', 'apple', 'microsoft', 'amazon', 'google', 'facebook']
        features['brand_mention'] = sum(1 for brand in brand_keywords if brand in url_lower)
        
        # Suspicious TLDs
        tld = domain.split('.')[-1] if '.' in domain else ''
        suspicious_tlds = ['.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.club', '.click', '.stream']
        features['suspicious_tld'] = 1 if any(tld.endswith(stld) for stld in suspicious_tlds) else 0
        
        debug_print(f"✅ Extracted {len(features)} features")
        return features
        
    except Exception as e:
        debug_print(f"❌ Feature extraction error: {e}")
        return {'error': str(e), 'url_length': len(url)}

def rule_based_detection(url: str, features: Dict[str, Any]) -> Tuple[float, str, List[str]]:
    """Rule-based phishing detection"""
    score = 0.0
    reasons = []
    
    # Rule 1: URL length
    if features.get('url_length', 0) > 75:
        score += 0.2
        reasons.append("Unusually long URL")
    if features.get('url_length', 0) > 100:
        score += 0.1
        reasons.append("Extremely long URL")
    
    # Rule 2: Suspicious keywords
    kw_count = features.get('suspicious_keyword_count', 0)
    if kw_count > 0:
        score += min(0.3, kw_count * 0.1)
        reasons.append(f"Contains {kw_count} suspicious keyword(s)")
    
    # Rule 3: IP address
    if features.get('has_ip', 0):
        score += 0.3
        reasons.append("Uses IP address instead of domain name")
    
    # Rule 4: No HTTPS
    if not features.get('has_https', 0):
        score += 0.2
        reasons.append("Does not use HTTPS encryption")
    
    # Rule 5: @ symbol
    if features.get('num_at', 0) > 0:
        score += 0.25
        reasons.append("Contains @ symbol (URL redirect trick)")
    
    # Rule 6: Special characters
    special_chars = features.get('num_equals', 0) + features.get('num_questions', 0)
    if special_chars > 3:
        score += 0.15
        reasons.append("Multiple special characters in URL")
    
    # Rule 7: Suspicious TLD
    if features.get('suspicious_tld', 0):
        score += 0.25
        reasons.append("Uses suspicious top-level domain")
    
    # Normalize score
    score = min(score, 1.0)
    
    # Determine status
    if score >= 0.7:
        status = "⚠️ HIGH RISK - Likely Phishing"
    elif score >= 0.4:
        status = "⚠️ MEDIUM RISK - Suspicious"
    else:
        status = "✅ LOW RISK - Likely Safe"
    
    return score, status, reasons

def check_url(url: str, model=None, vectorizer=None, use_model=True) -> Dict[str, Any]:
    """Main function to check a URL for phishing"""
    debug_print(f"check_url() called with: {url[:100]}")
    
    # Extract features
    features = extract_features(url)
    
    # Initialize result
    result = {
        'url': url,
        'timestamp': datetime.now().isoformat(),
        'features': features,
        'is_phishing': False,
        'risk_score': 0.0,
        'confidence': 0.0,
        'status_message': '',
        'risk_factors': [],
        'detection_method': 'rule-based'
    }
    
    # Try ML if available
    ml_used = False
    if use_model and model is not None and vectorizer is not None:
        try:
            debug_print("Attempting ML prediction...")
            # Convert features to vector (adjust based on your model)
            # NOTE: This assumes your model expects a flat list of feature values.
            # You might need a more sophisticated feature processing if your model
            # expects specific feature names or a DataFrame structure.
            feature_values = [[float(v) for v in features.values() if isinstance(v, (int, float))]]
            if len(feature_values[0]) > 0:
                # Dummy prediction for demonstration if actual model is not loaded or compatible
                if isinstance(model, joblib._memmapping_reducer.Unpickler) or not hasattr(model, 'predict'):
                    prediction = 1 if result['risk_score'] > 0.5 else 0 # Fallback dummy
                    probability = [1 - result['risk_score'], result['risk_score']] # Fallback dummy
                    debug_print("Using dummy ML prediction due to model incompatibility.")
                else:
                    prediction = model.predict(feature_values)[0]
                    probability = model.predict_proba(feature_values)[0]
                
                result['risk_score'] = float(probability[1] if len(probability) > 1 else probability[0])
                result['is_phishing'] = bool(prediction == 1)
                result['confidence'] = result['risk_score']
                result['detection_method'] = 'machine-learning'
                ml_used = True
                debug_print(f"ML prediction: {result['is_phishing']}, score={result['risk_score']}")
        except Exception as e:
            debug_print(f"ML failed: {e}")
    
    # Use rule-based if ML not used
    if not ml_used:
        score, status, reasons = rule_based_detection(url, features)
        result['risk_score'] = score
        result['is_phishing'] = score >= 0.5
        result['confidence'] = score
        result['status_message'] = status
        result['risk_factors'] = reasons
    
    # User-friendly message
    if result['is_phishing']:
        result['message'] = "🚨 DANGER: This appears to be a phishing site! Do NOT enter personal information."
    else:
        result['message'] = "✅ This URL appears safe based on our analysis."
    
    return result

def generate_pdf_report(result: Dict[str, Any], output_path: str = 'report.pdf') -> str:
    """Generate a report file (text format for compatibility)"""
    debug_print(f"Generating report at: {output_path}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Create report
    lines = []
    lines.append("=" * 60)
    lines.append("PHISHING DETECTION REPORT")
    lines.append("=" * 60)
    lines.append(f"URL: {result.get('url', 'N/A')}")
    lines.append(f"Time: {result.get('timestamp', 'N/A')}")
    lines.append("")
    lines.append("-" * 40)
    lines.append("RESULTS")
    lines.append("-" * 40)
    lines.append(f"Risk Score: {result.get('risk_score', 0):.2%}")
    lines.append(f"Status: {'PHISHING' if result.get('is_phishing') else 'SAFE'}")
    lines.append(f"Method: {result.get('detection_method', 'N/A')}")
    lines.append("")
    
    if result.get('risk_factors'):
        lines.append("-" * 40)
        lines.append("RISK FACTORS")
        lines.append("-" * 40)
        for factor in result['risk_factors']:
            lines.append(f"• {factor}")
        lines.append("")
    
    lines.append("=" * 60)
    
    # Save file
    with open(output_path, 'w') as f:
        f.write('
'.join(lines))
    
    debug_print(f"✅ Report saved to {output_path}")
    return output_path

__version__ = "1.0.0"
