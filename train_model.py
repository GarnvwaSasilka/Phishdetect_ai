
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle
import re
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("PHISHING DETECTION MODEL TRAINING")
print("="*60)

# ============================================
# PART 1: CREATE SYNTHETIC TRAINING DATA
# (In production, replace with real dataset)
# ============================================

def extract_url_features(url):
    """Extract numerical features from URL"""
    features = {}

    try:
        parsed = urlparse(url)

        # Length features
        features['url_length'] = len(url)
        features['domain_length'] = len(parsed.netloc)
        features['path_length'] = len(parsed.path)

        # Count features
        features['num_dots'] = url.count('.')
        features['num_hyphens'] = url.count('-')
        features['num_underscores'] = url.count('_')
        features['num_slashes'] = url.count('/')
        features['num_questions'] = url.count('?')
        features['num_equals'] = url.count('=')
        features['num_at'] = url.count('@')
        features['num_and'] = url.count('&')
        features['num_digits'] = sum(c.isdigit() for c in url)

        # Binary features
        features['has_https'] = 1 if parsed.scheme == 'https' else 0
        features['has_ip'] = 1 if re.match(r'\d+\.\d+\.\d+\.\d+', parsed.netloc) else 0
        features['has_login'] = 1 if 'login' in url.lower() else 0
        features['has_verify'] = 1 if 'verify' in url.lower() else 0
        features['has_secure'] = 1 if 'secure' in url.lower() else 0
        features['has_account'] = 1 if 'account' in url.lower() else 0

        # Suspicious keywords count
        suspicious = ['login', 'signin', 'verify', 'secure', 'account',
                     'banking', 'update', 'confirm', 'paypal', 'apple', 'microsoft', 'amazon']
        features['suspicious_keyword_count'] = sum(1 for kw in suspicious if kw in url.lower())

        # TLD features
        tld = parsed.netloc.split('.')[-1] if '.' in parsed.netloc else ''
        suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'xyz', 'top', 'club', 'click', 'stream', 'gq']
        features['suspicious_tld'] = 1 if tld in suspicious_tlds else 0

        # Entropy (randomness indicators)
        from collections import Counter
        if url:
            prob = [freq/len(url) for freq in Counter(url).values()]
            features['entropy'] = -sum(p * np.log2(p) for p in prob if p > 0)
        else:
            features['entropy'] = 0

    except:
        features = {f: 0 for f in range(20)}  # Fallback

    return features

# Generate phishing URLs
def generate_phishing_urls(n=1000):
    """Generate synthetic phishing URLs for training"""
    urls = []

    templates = [
        "http://paypal.com.", "https://appleid.", "http://secure-login.",
        "https://account-verify.", "http://microsoft.", "https://amazon.",
        "http://bankofamerica.", "https://chase.", "http://wellsfargo."
    ]

    suspicious_domains = [
        "login.xyz", "verify.tk", "secure.ml", "account.ga", "confirm.cf",
        "security.top", "verification.club", "authenticate.click", "user.stream"
    ]

    for _ in range(n):
        template = np.random.choice(templates)
        domain = np.random.choice(suspicious_domains)
        url = template.format(domain)

        # Add random parameters
        if np.random.random() > 0.5:
            url += f"?id={np.random.randint(1000,9999)}&session={np.random.randint(10000,99999)}"

        urls.append(url)

    return urls

# Generate legitimate URLs
def generate_legitimate_urls(n=1000):
    """Generate synthetic legitimate URLs for training"""
    urls = []

    legit_domains = [
        "google.com", "github.com", "stackoverflow.com", "reddit.com",
        "wikipedia.org", "amazon.com", "netflix.com", "spotify.com",
        "nytimes.com", "bbc.com", "cnn.com", "medium.com"
    ]

    paths = [
        "", "/search", "/login", "/signin", "/products", "/about",
        "/contact", "/help", "/docs", "/api", "/blog", "/careers"
    ]

    for _ in range(n):
        domain = np.random.choice(legit_domains)
        path = np.random.choice(paths)
        url = f"https://{domain}{path}"

        # Add query params sometimes
        if np.random.random() > 0.7:
            url += f"?q={np.random.choice(['test', 'query', 'search'])}"

        urls.append(url)

    return urls

print("\n📊 Generating training data...")
phishing_urls = generate_phishing_urls(1500)
legitimate_urls = generate_legitimate_urls(1500)

# Create feature vectors
print("🔍 Extracting features...")
X = []
y = []

for url in phishing_urls:
    features = extract_url_features(url)
    X.append(list(features.values()))
    y.append(1)  # 1 = phishing

for url in legitimate_urls:
    features = extract_url_features(url)
    X.append(list(features.values()))
    y.append(0)  # 0 = legitimate

X = np.array(X)
y = np.array(y)

print(f"✅ Dataset created: {len(X)} samples, {X.shape[1]} features")
print(f"   Phishing: {sum(y)} samples")
print(f"   Legitimate: {len(y)-sum(y)} samples")

# ============================================
# PART 2: TRAIN THE MODEL
# ============================================

print(f"\n{'='*60}")
print("TRAINING MODEL")
print("="*60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest (best for phishing detection)
print("\n🌲 Training Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_scaled, y_train)

# Train Gradient Boosting as ensemble
print("🚀 Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train_scaled, y_train)

# Create ensemble (average predictions)
class EnsembleClassifier:
    def __init__(self, models, weights=None):
        self.models = models
        self.weights = weights if weights else [1/len(models)] * len(models)

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        weighted_preds = np.average(predictions, axis=0, weights=self.weights)
        return (weighted_preds >= 0.5).astype(int)

    def predict_proba(self, X):
        probas = np.array([model.predict_proba(X) for model in self.models])
        return np.average(probas, axis=0, weights=self.weights)

ensemble = EnsembleClassifier([rf_model, gb_model], weights=[0.6, 0.4])

# ============================================
# PART 3: EVALUATE MODEL
# ============================================

print(f"\n{'='*60}")
print("MODEL EVALUATION")
print("="*60)

# Random Forest evaluation
rf_pred = rf_model.predict(X_test_scaled)
print("\n📈 Random Forest Performance:")
print(f"   Accuracy: {accuracy_score(y_test, rf_pred):.3f}")
print(classification_report(y_test, rf_pred, target_names=['Legitimate', 'Phishing']))

# Ensemble evaluation
ensemble_pred = ensemble.predict(X_test_scaled)
print("\n🌟 Ensemble Performance:")
print(f"   Accuracy: {accuracy_score(y_test, ensemble_pred):.3f}")
print(classification_report(y_test, ensemble_pred, target_names=['Legitimate', 'Phishing']))

# Feature importance
feature_names = list(extract_url_features("http://example.com").keys())
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔑 Top 10 Most Important Features:")
print(importance_df.head(10).to_string(index=False))

# ============================================
# PART 4: SAVE MODELS
# ============================================

print(f"\n{'='*60}")
print("SAVING MODELS")
print("="*60)

# Save Random Forest model
with open('model.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
print("✅ Saved: model.pkl")

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved: scaler.pkl")

# Save feature names
with open('feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Saved: feature_names.pkl")

# For backward compatibility (create dummy vectorizer if needed)
# Some code expects a vectorizer.pkl - create a placeholder
dummy_vectorizer = {"type": "feature_extractor", "features": feature_names}
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(dummy_vectorizer, f)
print("✅ Saved: vectorizer.pkl (compatibility placeholder)")

# Save ensemble model (optional)
with open('ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)
print("✅ Saved: ensemble_model.pkl")

print(f"\n{'='*60}")
print("✅ TRAINING COMPLETE!")
print("="*60)

# Test with sample URLs
print("\n🧪 Testing with sample URLs:")
test_urls = [
    "https://www.google.com",
    "http://paypal.com.login.secure.xyz",
    "https://github.com",
    "http://appleid-verify.tk",
    "https://stackoverflow.com"
]

for url in test_urls:
    features = extract_url_features(url)
    feature_vector = scaler.transform([list(features.values())])
    prediction = rf_model.predict(feature_vector)[0]
    probability = rf_model.predict_proba(feature_vector)[0]

    status = "⚠️ PHISHING" if prediction == 1 else "✅ LEGITIMATE"
    print(f"   {status}: {url[:50]}... (confidence: {probability[prediction]:.2f})")
