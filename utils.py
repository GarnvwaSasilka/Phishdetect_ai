import re, numpy as np, joblib, os
from lime.lime_text import LimeTextExplainer

_m, _v, _lc = None, None, None # model, vectorizer, lime_components

def load_components():
    global _m, _v, _lc
    if _m is None:
        b = os.path.dirname(__file__)
        _m = joblib.load(os.path.join(b, 'model.pkl'))
        _v = joblib.load(os.path.join(b, 'vectorizer.pkl'))
        _lc = joblib.load(os.path.join(b, 'lime_components.pkl'))
    return _m, _v, _lc

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_email(email_text):
    model, vec, lime_data = load_components()
    c = clean_text(email_text)
    f = vec.transform([c])
    pred_numeric = model.predict(f)[0]
    proba = model.predict_proba(f)[0]
    conf = np.max(proba) * 100
    
    pred_string = lime_data['label_mapping'][pred_numeric]

    rm = {'legitimate': ('Low', '🟢'), 'traditional_phishing': ('High', '🔵'), 'ai_generated_phishing': ('Medium', '🔴')}
    risk, emoji = rm.get(pred_string, ('Unknown', '⚪'))
    return {'prediction': pred_string, 'confidence': conf, 'risk_level': risk, 'risk_emoji': emoji, 'probabilities': dict(zip(lime_data['class_names'], proba))}

def get_lime_explanation(email_text, num_features=10):
    model, vec, lime_data = load_components()
    explainer = LimeTextExplainer(class_names=lime_data['class_names'])

    def predict_fn_for_lime(texts):
        cleaned = [clean_text(t) for t in texts]
        return model.predict_proba(vec.transform(cleaned))

    pred_numeric = model.predict(vec.transform([clean_text(email_text)]))[0]
    pred_string = lime_data['label_mapping'][pred_numeric]

    exp = explainer.explain_instance(
        email_text,
        predict_fn_for_lime,
        num_features=num_features,
        num_samples=200,
        labels=(lime_data['class_names'].index(pred_string),)
    )
    return {'prediction': pred_string, 'word_weights': exp.as_list(label=lime_data['class_names'].index(pred_string))}

COLOR_MAP = {
    'legitimate': {'hex': '#009600', 'emoji': '🟢'},
    'traditional_phishing': {'hex': '#0064C8', 'emoji': '🔵'},
    'ai_generated_phishing': {'hex': '#C80000', 'emoji': '🔴'}
}
