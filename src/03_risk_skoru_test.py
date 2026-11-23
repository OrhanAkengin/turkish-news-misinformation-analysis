from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_PATH = DATA_DIR / "hazir_veri.csv"

RISK_MODEL_PATH = MODELS_DIR / "risk_model_logistic.pkl"
RISK_VECTORIZER_PATH = MODELS_DIR / "risk_tfidf_vectorizer.pkl"


def risk_level(score):
    """
    0-100 arasındaki risk skorunu sözel risk seviyesine çevirir.
    """

    if score < 40:
        return "Düşük Risk"
    elif score < 70:
        return "Orta Risk"
    else:
        return "Yüksek Risk"


def train_risk_model():
    """
    Logistic Regression modelini eğitir.
    Bu modeli risk skoru üretmek için kullanıyoruz.
    """

    df = pd.read_csv(DATA_PATH)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_tfidf, y_train)

    joblib.dump(model, RISK_MODEL_PATH)
    joblib.dump(vectorizer, RISK_VECTORIZER_PATH)

    print("Risk model saved:")
    print(RISK_MODEL_PATH)

    print("\nRisk vectorizer saved:")
    print(RISK_VECTORIZER_PATH)

    return model, vectorizer


def predict_news_risk(text, model, vectorizer):
    """
    Girilen haber metni için dezenformasyon risk skoru üretir.
    """

    text_tfidf = vectorizer.transform([text])

    prediction = model.predict(text_tfidf)[0]

    # predict_proba çıktısı:
    # [0. sınıf olasılığı, 1. sınıf olasılığı]
    # 0 = gerçek haber
    # 1 = sahte haber
    probabilities = model.predict_proba(text_tfidf)[0]

    fake_probability = probabilities[1]

    risk_score = round(fake_probability * 100, 2)

    if prediction == 1:
        prediction_label = "Sahte / Yanlış Haber"
    else:
        prediction_label = "Gerçek / Doğru Haber"

    level = risk_level(risk_score)

    return prediction_label, risk_score, level


def main():
    print("Risk score test started.\n")

    model, vectorizer = train_risk_model()

    print("\nModel is ready.")
    print("You can now test a news text.")
    print("Exit için boş bırakıp Enter'a basabilirsin.\n")

    while True:
        user_text = input("Haber metnini gir: ")

        if user_text.strip() == "":
            print("\nProgram kapatıldı.")
            break

        prediction_label, risk_score, level = predict_news_risk(
            user_text,
            model,
            vectorizer
        )

        print("\n--- Sonuç ---")
        print("Model Tahmini:", prediction_label)
        print("Dezenformasyon Risk Skoru:", f"%{risk_score}")
        print("Risk Seviyesi:", level)
        print("-------------\n")


if __name__ == "__main__":
    main()