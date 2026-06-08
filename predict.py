
import pickle
import numpy as np

# Load models
with open("lr_final.pkl", "rb") as f:
    lr = pickle.load(f)
with open("svm_final.pkl", "rb") as f:
    svm = pickle.load(f)
with open("tfidf_final2.pkl", "rb") as f:
    tfidf = pickle.load(f)

label_map = {0: "Legitimate", 1: "Traditional Phishing", 2: "AI-Generated Phishing"}

def predict_email(text):
    vec = tfidf.transform([text])
    lr_p = lr.predict_proba(vec)
    svm_p = svm.predict_proba(vec)
    proba = (lr_p * 0.5) + (svm_p * 0.5)
    pred = np.argmax(proba, axis=1)[0]
    confidence = max(proba[0]) * 100
    return label_map[pred], round(confidence, 2)

# Test
if __name__ == "__main__":
    email = input("Enter email text: ")
    label, confidence = predict_email(email)
    print(f"Prediction: {label} ({confidence}% confidence)")
