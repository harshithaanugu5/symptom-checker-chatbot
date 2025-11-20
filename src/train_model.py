import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from preprocess import load_and_clean

def train_model():
    df = load_and_clean()

    X = df["Symptoms"]
    y = df["Diagnosis_Suggestion"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    model = MultinomialNB()
    model.fit(X_vec, y)

    with open("model.pkl", "wb") as f:
        pickle.dump((vectorizer, model), f)

    print("Model saved!")

if __name__ == "__main__":
    train_model()
