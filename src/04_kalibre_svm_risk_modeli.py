from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

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

MODEL_PATH = MODELS_DIR / "calibrated_svm_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "calibrated_svm_vectorizer.pkl"
RESULTS_PATH = OUTPUTS_DIR / "calibrated_svm_sonuclari.csv"


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


def prepare_data(df):
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
    Haber metinlerini TF-IDF yöntemiyle sayısal hale getirir.
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


def train_calibrated_svm(X_train_tfidf, y_train):
    """
    Linear SVM modelini kalibre ederek olasılık çıktısı verebilir hale getirir.

    LinearSVC normalde predict_proba() üretmez.
    CalibratedClassifierCV, SVM karar skorlarını kalibre ederek
    0-1 aralığında olasılık benzeri skor üretir.
    """

    base_svm = LinearSVC(
        random_state=42
    )

    calibrated_model = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=5
    )

    print("\nTraining Calibrated Linear SVM model...")
    calibrated_model.fit(X_train_tfidf, y_train)

    return calibrated_model


def evaluate_model(model, X_test_tfidf, y_test):
    """
    Kalibre edilmiş SVM modelinin performansını ölçer.
    """

    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "=" * 60)
    print("Calibrated Linear SVM Results")
    print("=" * 60)

    print("\nAccuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1-score:", round(f1, 4))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Gercek", "Sahte"]))

    results = {
        "model": "Calibrated Linear SVM",
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    return results


def predict_sample_texts(model, vectorizer):
    """
    Modeli birkaç örnek metin üzerinde test eder.
    """

    sample_texts = [
        "Sağlık Bakanlığı, yeni aşılama takvimiyle ilgili resmi açıklama yaptı.",
        "Son dakika! Gizli uzmanlar bu ilacın tüm hastalıkları bir gecede yok ettiğini açıkladı.",
        "Ekonomi yönetimi, enflasyonla mücadele kapsamında yeni tedbirlerin değerlendirildiğini bildirdi."
    ]

    print("\n" + "=" * 60)
    print("Sample Risk Score Tests")
    print("=" * 60)

    for text in sample_texts:
        text_tfidf = vectorizer.transform([text])

        prediction = model.predict(text_tfidf)[0]

        probabilities = model.predict_proba(text_tfidf)[0]

        # probabilities[0] = gerçek haber olasılığı
        # probabilities[1] = sahte haber olasılığı
        fake_probability = probabilities[1]

        risk_score = round(fake_probability * 100, 2)
        level = risk_level(risk_score)

        if prediction == 1:
            prediction_label = "Sahte / Yanlış Haber"
        else:
            prediction_label = "Gerçek / Doğru Haber"

        print("\nMetin:")
        print(text)

        print("Model Tahmini:", prediction_label)
        print("Dezenformasyon Risk Skoru:", f"%{risk_score}")
        print("Risk Seviyesi:", level)


def save_outputs(model, vectorizer, results):
    """
    Modeli, vectorizer'ı ve sonuçları kaydeder.
    """

    MODELS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    results_df = pd.DataFrame([results])
    results_df.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    print("\nModel saved:")
    print(MODEL_PATH)

    print("\nVectorizer saved:")
    print(VECTORIZER_PATH)

    print("\nResults saved:")
    print(RESULTS_PATH)


def main():
    print("Calibrated SVM risk model training started.\n")

    df = load_dataset()

    X_train, X_test, y_train, y_test = prepare_data(df)

    vectorizer, X_train_tfidf, X_test_tfidf = vectorize_text(X_train, X_test)

    model = train_calibrated_svm(X_train_tfidf, y_train)

    results = evaluate_model(model, X_test_tfidf, y_test)

    predict_sample_texts(model, vectorizer)

    save_outputs(model, vectorizer, results)

    print("\nCalibrated SVM risk model training completed.")


if __name__ == "__main__":
    main()