import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "model.pkl")

# Simple rule-based medical knowledge
RULES = {
    "headache": "Migraine",
    "cold": "Common Cold",
    "cough": "Upper Respiratory Infection",
    "fever": "Viral Infection",
    "sore throat": "Throat Infection",
    "runny nose": "Common Cold",
    "sneezing": "Allergic Rhinitis",
    "body pain": "Viral Flu",
    "vomiting": "Gastroenteritis",
    "diarrhea": "Stomach Infection",

    # ❤️ HEART-RELATED RULES
    "chest pain": "Possible Heart Attack",
    "chest tightness": "Angina",
    "shortness of breath": "Cardiac Distress",
    "palpitations": "Arrhythmia",
    "heart pain": "Angina",
    "left arm pain": "Possible Heart Attack",
    "pressure in chest": "Possible Heart Attack",
}



def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def rule_based_prediction(text):
    text = text.lower()
    for symptom, diagnosis in RULES.items():
        if symptom in text:
            return diagnosis
    return None

def predict_condition(user_input):
    # Check rule-based conditions first
    rb = rule_based_prediction(user_input)
    if rb:
        return rb

    # If rules did not match → use ML model
    vectorizer, model = load_model()
    X_vec = vectorizer.transform([user_input.lower()])
    prediction = model.predict(X_vec)[0]
    return prediction
