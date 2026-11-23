from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"

DATA_PATH = DATA_DIR / "hazir_veri.csv"

BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
RESULTS_PATH = OUTPUTS_DIR / "model_sonuclari.csv"


def load_dataset():
    """
    Hazırlanmış ana veri setini okur.
    """

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} bulunamadı. Önce 01_veri_hazirlama.py dosyasını çalıştırmalısın."
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset size:", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\nClass distribution:")
    print(df["label"].value_counts())

    return df


def prepare_train_test_data(df):
    """
    Veriyi eğitim ve test olarak böler.
    """

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain size:", X_train.shape[0])
    print("Test size:", X_test.shape[0])

    return X_train, X_test, y_train, y_test


def vectorize_text(X_train, X_test):
    """
    Metinleri TF-IDF yöntemiyle sayısal hale getirir.
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_features=10000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("\nTF-IDF train shape:", X_train_tfidf.shape)
    print("TF-IDF test shape:", X_test_tfidf.shape)

    return vectorizer, X_train_tfidf, X_test_tfidf


def get_models():
    """
    Denenecek makine öğrenmesi modellerini döndürür.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "Naive Bayes": MultinomialNB(),

        "Linear SVM": LinearSVC(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def evaluate_model(model_name, model, X_train_tfidf, X_test_tfidf, y_train, y_test):
    """
    Verilen modeli eğitir ve test sonuçlarını hesaplar.
    """

    print("\n" + "=" * 60)
    print(f"Training model: {model_name}")
    print("=" * 60)

    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Gercek", "Sahte"]))

    result = {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return result, model


def main():
    print("Model training started.\n")

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    df = load_dataset()

    X_train, X_test, y_train, y_test = prepare_train_test_data(df)

    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    models = get_models()

    results = []
    trained_models = {}

    for model_name, model in models.items():
        result, trained_model = evaluate_model(
            model_name,
            model,
            X_train_tfidf,
            X_test_tfidf,
            y_train,
            y_test
        )

        results.append(result)
        trained_models[model_name] = trained_model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="f1_score", ascending=False)

    print("\n" + "=" * 60)
    print("Model comparison results:")
    print("=" * 60)
    print(results_df)

    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    best_model_name = results_df.iloc[0]["model"]
    best_model = trained_models[best_model_name]

    print("\nBest model:", best_model_name)

    joblib.dump(best_model, BEST_MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    print("\nBest model saved:")
    print(BEST_MODEL_PATH)

    print("\nTF-IDF vectorizer saved:")
    print(VECTORIZER_PATH)

    print("\nModel training completed.")


if __name__ == "__main__":
    main()